// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util.DecoupledIO
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.Ignore

// Gate 3a: replays the EXACT whole-layer kick that examples/qwen3/xnpu
// generates -- descriptor-driven DMA (x batch, RoPE tables, KV history,
// streamed weight tiles), the 15-stage single-kick program, KV publish and
// residual store -- against the real Npu wired to an AXI DRAM model (DmaTb,
// the same harness LayerSpec uses for its DMA layer test).
//
// WHAT IS UNDER TEST
//   The emitter composition itself: descriptor wiring, stage order, q_dim
//   shapes (wq out=q_dim, o_proj in=q_dim), QK-Norm insertion, half-split
//   RoPE call sites, KV publish offsets, DF_LOAD_BATCH/DF_STORE_BATCH paths.
//   Per-emitter numerics were already gated at real Qwen3-0.6B shapes by
//   Qwen3QknormRtlSpec / Qwen3RopeRtlSpec; this spec runs the TINY isomorphic
//   model (same structure, head_dim != dim/n_heads, small dims) so the whole
//   kick simulates in seconds instead of streaming 15.5 MB of real weights.
//
// REFERENCE
//   examples/qwen3/xnpu/fx_layer.c runs the real run3 CPU forward (int8-act
//   shadow, the same baseline board XNPU_CHECK uses) and dumps per-position
//   references (q/k/v/attn/swiglu/layer-end x) -- NOT just position 0 (plan
//   §6 fixture-blindness discipline: the pos-0-only class once let a real
//   QK-norm batch bug through).
//
// Regen fixtures: `make -C examples/qwen3 fx-rtl`
class Qwen3LayerRtlSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  // Dual subclass (Qwen3LayerRtlDualSpec) overrides this to 2: fixtures are
  // then the XNPU_ENGINES=2 generation (dual-layer.* -- the emitter alternates
  // output tiles across both engines via per-ot MCFG2) and the Npu carries
  // both engines. Fixture refs are the same CPU shadow either way.
  protected def engines: Int = 1
  private val p = Parameters(int8Engines = engines)
  private val fxPrefix = if (engines == 1) "" else "dual-"
  private val rowBytes = p.DtcmRowBytes
  private val ITCM = BigInt(0) << 19
  private val DTCM = BigInt(1) << 19
  private val CTRL = BigInt(2) << 19
  private val OFF_CTRL = 0x00; private val OFF_STATUS = 0x04

  private def fxFile(name: String): java.io.File = {
    val rel = s"examples/qwen3/fixtures/rtl/$fxPrefix$name"
    var d = new java.io.File(System.getProperty("user.dir"))
    while (d != null && !(new java.io.File(d, rel)).exists) d = d.getParentFile
    require(d != null, s"fixture $name not found: run `make -C examples/qwen3 fx-rtl`")
    new java.io.File(d, rel)
  }

  private def readBytes(name: String): Array[Byte] =
    java.nio.file.Files.readAllBytes(fxFile(name).toPath)
  private def readU32s(name: String): Array[BigInt] = {
    val b = readBytes(name)
    val bb = java.nio.ByteBuffer.wrap(b).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    Array.fill(b.length / 4)(BigInt(bb.getInt() & 0xffffffffL))
  }
  private def readF32s(name: String): Array[Float] = {
    val b = readBytes(name)
    val bb = java.nio.ByteBuffer.wrap(b).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    Array.fill(b.length / 4)(bb.getFloat())
  }
  private def meta: Map[String, String] =
    scala.io.Source.fromFile(fxFile("layer.txt")).getLines()
      .filterNot(_.startsWith("#")).flatMap { l =>
        l.split("\\s+") match { case Array(k, v) => Some(k -> v); case _ => None }
      }.toMap
  private def mi(k: String): Int = meta(k).toInt

  /** bf16 element `idx` from a run of little-endian 32-bit DTCM words
   * (address `base` is the element-0 byte offset within the region). */
  private def bf16FromWords(words: Array[BigInt], base: Int, idx: Int): Float = {
    val byte = base + idx * 2
    val w = words(byte / 4).toLong & 0xffffffffL
    java.lang.Float.intBitsToFloat((((w >> (16 * ((byte / 2) & 1))) & 0xFFFFL).toInt) << 16)
  }

  behavior of "whole-layer kick (Qwen3 emitter composition, RTL replay)"

  it should "run one full layer kick on the real Npu and match the CPU shadow per position" in {
    val metaI = mi _
    val B = metaI("B"); val start = metaI("start_pos")
    val dim = metaI("dim"); val hidden = metaI("hidden")
    val qDim = metaI("q_dim"); val kvDim = metaI("kv_dim")
    val window = metaI("window"); val ddrBytes = metaI("ddr_bytes")

    val progRaw = readU32s("layer.prog")
    val nWords = progRaw(0).toInt
    val prog = progRaw.drop(1)
    assert(prog.length == nWords && prog.last == BigInt(0))

    val ddr = readBytes("layer.ddr")
    assert(ddr.length == ddrBytes)
    val dtcm = readU32s("layer.mem_in")
    assert(dtcm.length * 4 == window)

    val refQ = readF32s("layer.ref.q"); val refK = readF32s("layer.ref.k")
    val refV = readF32s("layer.ref.v"); val refAttn = readF32s("layer.ref.attn")
    val refFfn = readF32s("layer.ref.swiglu"); val refX = readF32s("layer.ref.x")

    test(new DmaTb(p, dramRows = (ddrBytes + rowBytes - 1) / rowBytes))
      .withAnnotations(backendAnnos) { dut =>
        val axi = dut.io.axi
        val mask32 = (BigInt(1) << 32) - 1
        def step(): Unit = dut.clock.step(1)
        axi.aw.valid.poke(false.B); axi.w.valid.poke(false.B); axi.b.ready.poke(false.B)
        axi.ar.valid.poke(false.B); axi.r.ready.poke(false.B); dut.io.preWen.poke(false.B)

        def addrSend(ch: DecoupledIO[AxiAddr], addr: BigInt, beats: Int): Unit = {
          ch.bits.id.poke(0.U); ch.bits.addr.poke(addr.U)
          ch.bits.len.poke((beats - 1).U); ch.bits.size.poke(2.U); ch.bits.burst.poke(1.U)
          ch.valid.poke(true.B)
          while (!ch.ready.peek().litToBoolean) step()
          step(); ch.valid.poke(false.B)
        }
        def writeBurst(addr: BigInt, data: Seq[BigInt]): Unit = {
          // AXI len is 8 bits: cap bursts at 256 beats, advancing by 4 B/word.
          val chunks = data.grouped(256).toSeq
          for ((c, ci) <- chunks.zipWithIndex) {
            val base = addr + BigInt(ci * 256 * 4)
            addrSend(axi.aw, base, c.length)
            for ((d, i) <- c.zipWithIndex) {
              axi.w.bits.data.poke((d & mask32).U); axi.w.bits.strb.poke("b1111".U)
              axi.w.bits.last.poke((i == c.length - 1).B); axi.w.valid.poke(true.B)
              while (!axi.w.ready.peek().litToBoolean) step()
              step()
            }
            axi.w.valid.poke(false.B); axi.b.ready.poke(true.B)
            while (!axi.b.valid.peek().litToBoolean) step()
            step(); axi.b.ready.poke(false.B)
          }
        }
        def readWord(addr: BigInt): BigInt = {
          addrSend(axi.ar, addr, 1); axi.r.ready.poke(true.B)
          while (!axi.r.valid.peek().litToBoolean) step()
          val v = axi.r.bits.data.peek().litValue
          step(); axi.r.ready.poke(false.B); v
        }
        def dramPreload(base: Int, bytes: Array[Byte], from: Int, until: Int): Unit = {
          for (r <- from / rowBytes until until / rowBytes) {
            var v = BigInt(0)
            for (b <- 0 until rowBytes)
              v |= BigInt(bytes(base + r * rowBytes + b) & 0xff) << (8 * b)
            dut.io.preAddr.poke((base + r * rowBytes).U); dut.io.preData.poke(v.U)
            dut.io.preWen.poke(true.B); step()
          }
          dut.io.preWen.poke(false.B)
        }
        def dramRow(addr: Int): BigInt = {
          dut.io.dbgAddr.poke(addr.U); step(); dut.io.dbgData.peek().litValue
        }
        def bf16AtRow(addr: Int, idx: Int): Float = {
          val row = dramRow(addr + (idx / 8) * rowBytes)
          val h = ((row >> (16 * (idx % 8))) & 0xFFFFL).toInt
          java.lang.Float.intBitsToFloat(h << 16)
        }

        // ---- PS setup: DRAM image, program, descriptor ----
        dramPreload(0, ddr, 0, ddrBytes)
        writeBurst(ITCM, prog)
        val descOff = metaI("desc_off")
        // The whole DTCM window is zero except the 64-word descriptor.
        writeBurst(DTCM + descOff, (0 until 64).map(i => dtcm(descOff / 4 + i)))

        // ---- kick + poll ----
        writeBurst(CTRL + OFF_CTRL, Seq(BigInt(1)))
        // Wedge forensics: a wedged kick used to die with nothing but a poll
        // count.  Dump every PERF counter in the failure message so the stuck
        // state is attributable from the log alone.  XNPU_POLLCAP lowers the
        // cap for VCD capture runs (a wedged design stops making progress, so
        // a short cap still captures it); the default keeps the old bound.
        val pollCap = sys.props.get("xnpu.pollcap").map(_.toInt).getOrElse(400000)
        var guard = 0
        while ((readWord(CTRL + OFF_STATUS) & 1) == 0 && guard < pollCap) {
          guard += 1
        }
        assert(guard < pollCap, "layer kick did not finish (" + guard + " polls); PERF: " +
          Seq(0x20 -> "dma_status", 0x24 -> "matrix", 0x28 -> "vector", 0x2c -> "dma",
              0x30 -> "dtcm_m", 0x34 -> "dtcm_a", 0x38 -> "depwait", 0x3c -> "dmawait",
              0x44 -> "cycles", 0x48 -> "dma_dtcm", 0x4c -> "dep_m", 0x50 -> "dep_v",
              0x54 -> "mem_wait", 0x5c -> "both_idle", 0x60 -> "depv_solo",
              0x64 -> "eng0", 0x68 -> "eng1", 0x6c -> "eng_both")
            .map { case (o, n) => n + "=" + readWord(CTRL + o) }.mkString(" "))
        info(s"kick done after $guard status polls, program $nWords words")

        // ---- barrier attribution (V1 baseline: PERF 0x44/0x54/0x58/0x5C/0x60) ----
        // cycles vs matrix/vector busy derive from the same kick; memwait is the
        // full-drain wait of scalar LW/SW (memwait - memclear = what a row-token
        // dispatch would recover), bothidle is kick dead time, depv_solo is the
        // true-bubble share of depwaitV.  Kept as info lines so every CI run
        // carries the A/B baseline for the V1 change.
        {
          val cycles  = readWord(CTRL + 0x44)
          val mwait   = readWord(CTRL + 0x54)
          val mclr    = readWord(CTRL + 0x58)
          val biddle  = readWord(CTRL + 0x5C)
          val dvsolo  = readWord(CTRL + 0x60)
          info(f"perf: cycles=$cycles memwait=$mwait memclear=$mclr " +
               f"bothidle=$biddle depv_solo=$dvsolo")
          // PERF probe: per-engine busy + port0 util.  0x70/0x74/0x78 split
          // queue occupancy from true unit execution (vectorBusy above is the
          // OR of both -- the 2026-10-29 "vector 97.3% busy" misread).
          val eng0b = readWord(CTRL + 0x64)
          val eng1b = readWord(CTRL + 0x68)
          val eng2b = readWord(CTRL + 0x6C)
          val p0 = readWord(CTRL + 0x30)
          val vUnit = readWord(CTRL + 0x70)
          val vQueue = readWord(CTRL + 0x74)
          val mEng = readWord(CTRL + 0x78)
          val soloNo = readWord(CTRL + 0x7C)
          val soloBlk = readWord(CTRL + 0x80)
          info(f"engbusy: e0=$eng0b e1=$eng1b both=$eng2b port0=$p0")
          info(f"split: vecUnit=$vUnit vecQueue=$vQueue matEngAny=$mEng " +
               f"soloNoWork=$soloNo soloBlocked=$soloBlk")
          // Full scalar-stall accounting: what the 0x38/0x4C/0x50/0x3C/0x2C
          // counters say about where the ~570 non-issue cycles per GEMM body go.
          val depw = readWord(CTRL + 0x38)
          val depm = readWord(CTRL + 0x4C)
          val depv = readWord(CTRL + 0x50)
          val daw  = readWord(CTRL + 0x3C)
          val dmab = readWord(CTRL + 0x2C)
          val dmad = readWord(CTRL + 0x48)
          val scv = readWord(CTRL + 0x84)
          val scm = readWord(CTRL + 0x88)
          val sg = readWord(CTRL + 0x8C)
          val sb = readWord(CTRL + 0x90)
          val sh = readWord(CTRL + 0x94)
          val ifetch = readWord(CTRL + 0x98)
          val diw = readWord(CTRL + 0x9C)
          val bube = readWord(CTRL + 0xA0)
          val bubh = readWord(CTRL + 0xA4)
          val dstr = readWord(CTRL + 0xA8)
          info(f"stalls: depwait=$depw dep_m=$depm dep_v=$depv dmawait=$daw " +
               f"dmaBusy=$dmab dmaDtcm=$dmad scStallVec=$scv scStallMat=$scm " +
               f"dmaIssue=$diw bubEmpty=$bube bubHalting=$bubh dmaStarve=$dstr")
          info(f"scalar: gprStall=$sg blkWait=$sb haltDrain=$sh itcmFetch=$ifetch")
          // Scalar cycle-account closure: every kick cycle is either an EX
          // retire (itcmFetch), a tapped stall, or a true bubble (bubEmpty).
          // Known INFLATION, not a leak: (a) perf 0x54 memwait counts the
          // LW/SW dispatch cycle (blkInEx && !blkIssued) as well as the wait;
          // (b) a blocking LW/SW can be gpr-stalled in the same cycle it
          // counts memwait, double-adding one cycle.  Measured overshoot:
          // 1.7% (tiny), 0.25% (real shape).  The 5% bound keeps future
          // unaccounted-stall classes (a stall with no tap) from hiding as
          // "unexplained cycles" -- the 2026-10-30 dmaIssue black hole was
          // 56% of the kick, far past any plausible inflation.
          val accounted = ifetch + sg + sb + sh + mwait + daw + scv + scm +
                          diw + bube
          val resid = (accounted - cycles).abs
          assert(resid * 20 < cycles,
            s"scalar cycle account leaked: accounted=$accounted cycles=$cycles " +
            s"residual=$resid (>5%)")
        }

        // ---- read back DTCM regions (q, atto, h2, x) via the host bus ----
        def dtcmWords(off: Int, bytes: Int): Array[BigInt] =
          (0 until bytes / 4).map(i => readWord(DTCM + off + i * 4)).toArray

        // ---- compare against the CPU shadow (fp32 refs vs bf16 RTL) ----
        def checkF32(name: String, got: Int => Float, ref: Array[Float], n: Int,
                     tol: Double = 5e-2): Unit = {
          var worst = 0.0; var scale = 0.0
          for (i <- 0 until n) {
            val e = ref(i); val g = got(i)
            if (math.abs(e) > scale) scale = math.abs(e)
            val d = math.abs(g - e)
            if (d > worst) worst = d
          }
          assert(scale > 1e-3, s"$name reference all-zero; fixture broken")
          val rel = worst / scale
          info(f"$name: $n elems, max rel err = $rel%.2e")
          if (rel >= tol) {
            // DBG: first 32 got/ref pairs, 16-element (one output-tile) groups
            for (i <- 0 until math.min(n, 32)) {
              val g = got(i); val e = ref(i)
              println(f"DBG-Q $name[$i%3d] (ot${i / 16}) got=$g%.4e ref=$e%.4e")
            }
          }
          assert(rel < tol, f"$name max rel err $rel%.3e exceeds $tol")
        }

        val qWords = dtcmWords(metaI("q_off"), B * qDim * 2)
        checkF32("q(post-rope)", i => bf16FromWords(qWords, 0, i), refQ, B * qDim)
        val attoWords = dtcmWords(metaI("atto_off"), B * qDim * 2)
        checkF32("atto(attn@V)", i => bf16FromWords(attoWords, 0, i), refAttn, B * qDim)
        val h2Words = dtcmWords(metaI("h2_off"), B * hidden * 2)
        checkF32("h2(swiglu)", i => bf16FromWords(h2Words, 0, i), refFfn, B * hidden)
        val xWords = dtcmWords(metaI("x_off"), B * dim * 2)
        checkF32("x(layer end)", i => bf16FromWords(xWords, 0, i), refX, B * dim)

        // ---- DDR: published K/V rows and the stored residual batch ----
        val kvb = metaI("kv_row_bytes")
        val kcBase = metaI("ddr_kc") + start * kvb
        val vcBase = metaI("ddr_vc") + start * kvb
        checkF32("K(published)", i => bf16AtRow(kcBase, i), refK, B * kvDim)
        checkF32("V(published)", i => bf16AtRow(vcBase, i), refV, B * kvDim)
        val xb = metaI("ddr_xbuf")
        checkF32("x(stored)", i => bf16AtRow(xb, i), refX, B * dim)

        // Negative control: the KV history rows the program DMA'd in must be
        // untouched by the publish (start_pos offset correctness).  Row 0 of
        // the mirror is the warm-up K (bf16 of the CPU cache) -- if publish
        // had gone to row 0, this would now hold token B-1's K instead.
        if (start > 0) {
          val kc0 = metaI("ddr_kc")
          var d0 = 0.0f
          for (i <- 0 until kvDim) {
            val d = math.abs(bf16AtRow(kc0, i) - bf16AtRow(kcBase, i))
            if (d > d0) d0 = d
          }
          assert(d0 > 1e-3, "history row 0 equals a batch row -- publish offset wrong")
        }
      }
  }
}

// Dual-engine replay: same whole-layer kick, fixtures generated with
// XNPU_ENGINES=2 (the emitter alternates output tiles across both engines
// via per-ot MCFG2), Npu built with both engines. Numerics must be
// bit-identical to the single-engine generation -- the engines own disjoint
// output-tile ranges and the per-tile k order is unchanged.
// RESOLVED 2026-10-26: the original kick wedge was two emitter map bugs, not
// an RTL race. (1) The descriptor rode on memset-0 underneath the ci tiles
// (layer_map_qwen3 never reserved it) -- every ot=0 SM stomped the desc
// mid-kick, so later `lw x1, desc+...` openers returned fp32 result words and
// the poisoned weight-stream pointer drove unaligned dmalds whose 0-beat
// bursts wedged the DMA FIFO. (2) The cb engine-pair select masked x13 with
// andi(0x800), which sign-extends to ~0x7FF, so cb rows landed in tmph/h2
// (caught by the VectorUnit shifted-overlap assert). Fixes: desc-first
// reservation in layer_map_qwen3, positional bit11 extraction, a signed
// 12-bit-immediate guard in enc.h, and a 0-beat-burst assert in the DMA
// engine. Single-engine fixtures predate the desc-first map and stay green;
// regenerate both sets with `make fx-rtl` / `make fx-rtl-dual`.
class Qwen3LayerRtlDualSpec extends Qwen3LayerRtlSpec {
  override protected def engines: Int = 2
}
