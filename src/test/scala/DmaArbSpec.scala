// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util.DecoupledIO
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

// Aux-port contention contracts at the REAL integration level (DmaTb: actual
// DmaEngine streaming over AXI, actual Dtcm two-port arbitration, actual
// VectorUnit). The port-1 policy is useDma = dmaReq && !auxCoreReq
// (Dtcm.scala): the DMA only takes the aux port on cycles the core's vector /
// scalar-memory clients leave free, so concurrent traffic STARVES it for the
// chain's duration. Two past arbitration experiments produced wrong NUMERICS
// (not just wedges), so every case here asserts bitwise data on BOTH sides of
// the contention, and the dmaStarve counter (PERF 0xA8) is reported -- the
// descriptor-FIFO work must keep all of this green.
//
//   1. DMALD streaming into W while a long vector chain (VSMUL V->D) plus
//      scalar LW/SW traffic holds the aux port: W rows land exactly, D is
//      bit-exact 2x V, S staging is exact.
//   2. DMAST draining R to DRAM under the same contention: DRAM gets R's
//      pre-kick rows bit-exactly.
//   3. A vector reader enqueued BEHIND an in-flight DMALD (no DMAWAIT): the
//      write lease holds it until the engine drains, so it never observes
//      partially-written rows (the early-retire failure mode).
class DmaArbSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val p = Parameters(int8Engines = 1)
  val rowBytes = p.DtcmRowBytes
  private val ITCM = BigInt(0) << 19
  private val DTCM = BigInt(1) << 19
  private val CTRL = BigInt(2) << 19

  // DTCM layout (byte offsets; every region is 16-byte row aligned).
  private val vBase = 0x0000   // vector srcA, 64 rows
  private val dBase = 0x1000   // vector dst,  64 rows
  private val wBase = 0x2000   // DMALD target, 64 rows
  private val rBase = 0x2800   // DMAST source, 64 rows
  private val sBase = 0x3000   // scalar LW/SW,  16 rows
  private val d2Base = 0x5000  // case-3 vector dst
  // DRAM layout.
  private val wSrcDram = 0x0000
  private val outDram = 0x1000

  def li(rd: Int, imm: Int): Seq[BigInt] =
    if (imm >= -2048 && imm < 2048) Seq(Encode.addi(rd, 0, imm))
    else {
      val hi = (imm + 0x800) >> 12
      Seq(Encode.lui(rd, hi & 0xfffff), Encode.addi(rd, rd, imm - (hi << 12)))
    }

  // All-normal bf16 lanes in [1.0, 2.0): bits 0x3F80..0x3F87, so VSMUL by 2.0
  // is exactly +0x80 per lane (no rounding anywhere in the chain).
  // Two bf16 elements per 32-bit word (eight per 128-bit row): element k at
  // word k/2, half k%2.
  def vPattern: Seq[BigInt] = {
    def elem(k: Int): BigInt = BigInt(0x3F80 + (k & 7))
    (0 until 64 * rowBytes / 4).map { i => elem(2 * i) | (elem(2 * i + 1) << 16) }
  }
  def vScaled: Seq[BigInt] =
    vPattern.map(w => ((w & 0xffff) + 0x80) | ((((w >> 16) & 0xffff) + 0x80) << 16))
  // Arbitrary distinct nonzero words for raw regions (no bf16 meaning).
  def tagPattern(tag: Int): Seq[BigInt] =
    (0 until 64 * rowBytes / 4).map(i => BigInt(tag * 0x10000 + i))
  def sInit: Seq[BigInt] = (0 until 16 * rowBytes / 4).map(i => BigInt(0x66000000 + i))
  // The DMALD source uses the same normal-lane pattern: the case-3 reader
  // scales it by 2.0, whose bit-exact expectation (+0x80/lane) holds only for
  // normals -- a raw tag pattern has subnormal low lanes where DAZ/FTZ-style
  // hardware semantics could legitimately differ from IEEE Scala math.
  def wSrc: Seq[BigInt] = vPattern
  def wSrcScaled: Seq[BigInt] = vScaled

  behavior of "aux-port DMA contention (DmaEngine vs vector/scalar clients)"

  // Shared vector chain: 16 VSMULs V->D (sustained aux pressure).
  def chain: Seq[BigInt] =
    li(12, vBase) ++ Seq(Encode.vseta(12)) ++
    li(14, dBase) ++ Seq(Encode.vsetd(14)) ++
    li(15, 512) ++ Seq(Encode.vsetlen(15)) ++
    li(16, 0x40000000) ++ Seq(Encode.vsets(16)) ++
    Seq.fill(16)(Encode.vsmul())

  it should "stream a DMALD bit-exactly under a long vector + scalar chain holding the aux port" in {
    val program =
      li(10, wSrcDram) ++ li(11, wBase) ++ Seq(Encode.dmald(10, 11, 64)) ++
      chain ++
      li(22, sBase) ++ (0 until 8).flatMap { k => Seq(   // scalar mem pressure
        Encode.lw(20, 22, k * 4), Encode.addi(20, 20, 1),
        Encode.sw(20, 22, 0x100 + k * 4)) } ++
      Seq(Encode.dmawait(), Encode.halt)
    runKick(program) { (check, starve) =>
      check("W(dmald)", wBase, wSrc, false)
      check("D(vsmul)", dBase, vScaled, false)
      check("S(staged)", sBase + 0x100, (0 until 8).map(k => BigInt(0x66000000 + k + 1)), false)
      info(f"[dmaStarve] load-under-vector: $starve")
    }
  }

  it should "drain a DMAST bit-exactly under the same contention" in {
    val program =
      li(10, outDram) ++ li(11, rBase) ++ Seq(Encode.dmast(10, 11, 64)) ++
      chain ++ Seq(Encode.dmawait(), Encode.halt)
    runKick(program) { (check, starve) =>
      check("D(vsmul)", dBase, vScaled, false)
      check("DRAM-out(dmast)", outDram, tagPattern(0x88), true)
      info(f"[dmaStarve] store-under-vector: $starve")
    }
  }

  it should "hold a vector reader behind an in-flight DMALD until every row lands (no partial reads)" in {
    // No DMAWAIT before the vector op: the DMALD's write lease is the only
    // thing ordering the stream. An early-retiring lease lets VSMUL read rows
    // the engine has not written yet.
    val program =
      li(10, wSrcDram) ++ li(11, wBase) ++ Seq(Encode.dmald(10, 11, 64)) ++
      li(12, wBase) ++ Seq(Encode.vseta(12)) ++
      li(14, d2Base) ++ Seq(Encode.vsetd(14)) ++
      li(15, 512) ++ Seq(Encode.vsetlen(15)) ++
      li(16, 0x40000000) ++ Seq(Encode.vsets(16)) ++
      Seq(Encode.vsmul(), Encode.dmawait(), Encode.halt)
    runKick(program) { (check, starve) =>
      check("D2(reader-after-dmald)", d2Base, wSrcScaled, false)
      info(f"[dmaStarve] reader-behind-load: $starve")
    }
  }

  // ---- DmaTb harness (the Qwen3LayerRtlSpec driving pattern, compacted) ----
  // Programs ITCM, applies the fixed DTCM/DRAM pre-state, kicks, polls done,
  // then hands the case a `check(name, base, words)` helper (word-granular,
  // DTCM or DRAM) and the dmaStarve counter.
  def runKick(program: Seq[BigInt])(
      post: ((String, BigInt, Seq[BigInt], Boolean) => Unit, BigInt) => Unit): Unit = {
    test(new DmaTb(p, dramRows = 512)).withAnnotations(backendAnnos) { dut =>
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
        val chunks = data.grouped(256).toSeq
        for ((c, ci) <- chunks.zipWithIndex) {
          addrSend(axi.aw, addr + BigInt(ci * 256 * 4), c.length)
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
      def dramPreload(base: Int, words: Seq[BigInt]): Unit = {
        words.grouped(4).zipWithIndex.foreach { case (g, r) =>
          var v = BigInt(0)
          for (i <- 0 until 4) v |= (g(i) & mask32) << (32 * i)
          dut.io.preAddr.poke((base + r * rowBytes).U); dut.io.preData.poke(v.U)
          dut.io.preWen.poke(true.B); step()
        }
        dut.io.preWen.poke(false.B)
      }
      def dramRowWord(addr: BigInt): BigInt = {
        val row = addr - (addr % rowBytes)
        val lane = ((addr % rowBytes) / 4).toInt
        dut.io.dbgAddr.poke(row.U); step()
        (dut.io.dbgData.peek().litValue >> (32 * lane)) & mask32
      }

      // Fixed pre-state shared by every case.
      writeBurst(DTCM + vBase, vPattern)
      writeBurst(DTCM + rBase, tagPattern(0x88))
      writeBurst(DTCM + sBase, sInit)
      dramPreload(wSrcDram, wSrc)

      writeBurst(ITCM, program)
      writeBurst(CTRL + 0x00, Seq(BigInt(1)))   // kick

      var guard = 0
      while ((readWord(CTRL + 0x04) & 1) == 0 && guard < 200000) guard += 1
      assert(guard < 200000, "kick did not finish under aux contention (starvation wedge?)")
      val starve = readWord(CTRL + 0xA8)

      def check(name: String, base: BigInt, expected: Seq[BigInt], dram: Boolean = false): Unit =
        expected.zipWithIndex.foreach { case (exp, i) =>
          val addr = base + i * 4
          val got = if (dram) dramRowWord(addr) else readWord(DTCM + addr)
          val (gs, es, as) = (got.toString(16), exp.toString(16), addr.toString(16))
          assert(got == exp, s"$name word $i @0x$as: got $gs expected $es")
        }
      post(check, starve)
    }
  }
}
