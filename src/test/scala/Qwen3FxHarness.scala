// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable

// Shared harness for the Qwen3 emitter RTL-replay specs (plan qwen3_0.6b_plan.md
// §5 Phase 3, gate 3a).  The C side (examples/qwen3/xnpu/fx_*.c, `make fx-rtl`)
// dumps per-emitter fixtures; these helpers load them, replay the program on the
// real Core with a 16-byte-row DTCM model, and compare against the reference
// image with a relative tolerance.
//
// WHY TOLERANCE, NOT EXACT MATCH
//   The vector unit stores bf16 while accumulating in fp32, and VRSQRT is a
//   seed+Newton approximation -- element results carry ~1e-2 relative error by
//   construction.  What these specs must catch (wrong stride/head count/gain
//   advance/pairing geometry) moves results by orders of magnitude more.  Every
//   spec pairs the reference compare with a NEGATIVE CONTROL so it cannot pass
//   by luck (plan §6 fixture-blindness discipline), and each fixture has been
//   mutation-tested once (bug injected -> spec must fail).
//
// The harness serves DMA-free programs only: it forces dmaIdle and cannot
// serve DMALD/DMAST.  fe_qknorm / fe_rope_half qualify.
trait Qwen3FxHarness extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  private val p = Parameters()
  val rowBytes = p.DtcmRowBytes

  case class Fx(prog: Array[BigInt], memIn: Array[BigInt], memRef: Array[BigInt],
                meta: Map[String, String]) {
    def metaI(k: String): Int = meta(k).toInt
  }

  /** Locate examples/qwen3/fixtures/rtl by walking up from the test CWD. */
  def loadFx(name: String): Fx = {
    val rel = s"examples/qwen3/fixtures/rtl/$name.prog"
    var d = new java.io.File(System.getProperty("user.dir"))
    while (d != null && !(new java.io.File(d, rel)).exists) d = d.getParentFile
    require(d != null, s"fixture $name not found: run `make -C examples/qwen3 fx-rtl`")
    val dir = new java.io.File(d, "examples/qwen3/fixtures/rtl")
    def readU32(f: java.io.File): Array[BigInt] = {
      val b = java.nio.file.Files.readAllBytes(f.toPath)
      val bb = java.nio.ByteBuffer.wrap(b).order(java.nio.ByteOrder.LITTLE_ENDIAN)
      Array.fill(b.length / 4)(BigInt(bb.getInt() & 0xffffffffL))
    }
    val meta = scala.io.Source.fromFile(new java.io.File(dir, s"$name.txt")).getLines()
      .filterNot(_.startsWith("#")).flatMap { l =>
        l.split("\\s+") match { case Array(k, v) => Some(k -> v); case _ => None }
      }.toMap
    val progRaw = readU32(new java.io.File(dir, s"$name.prog"))
    val nWords = progRaw(0).toInt
    val prog = progRaw.drop(1).take(nWords)
    assert(prog.length == nWords, s"program truncated: ${prog.length} of $nWords")
    assert(prog.last == BigInt(0), "program must end in HALT")
    Fx(prog, readU32(new java.io.File(dir, s"$name.mem_in")),
       readU32(new java.io.File(dir, s"$name.mem_out")), meta)
  }

  /** Replay a DMA-free program on the Core; returns the final DTCM model.
   * pOverride: run with non-default Parameters (e.g. bf16 tier builds). */
  def runCore(prog: Array[BigInt], memInit: collection.Map[BigInt, BigInt],
              maxCycles: Int = 200000,
              pOverride: Option[Parameters] = None): (mutable.Map[BigInt, BigInt], Int) = {
    val mem = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
    mem ++= memInit
    var result: (mutable.Map[BigInt, BigInt], Int) = null
    test(new Core(pOverride.getOrElse(p))).withAnnotations(backendAnnos) { dut =>
      def instWord(byteAddr: BigInt): BigInt = {
        val idx = (byteAddr >> 2).toInt
        if (idx >= 0 && idx < prog.length) prog(idx) else BigInt(0)
      }
      var pendingRead: Option[BigInt] = None
      var instAddrPrev: BigInt = 0
      dut.io.dtcm.ready.poke(true.B)
      dut.io.dmaIdle.poke(true.B)   // no core-issued DMA in these programs
      dut.clock.setTimeout(0)
      def tick(): Unit = {
        dut.io.itcmInst.poke(instWord(instAddrPrev).U)
        instAddrPrev = dut.io.itcmAddr.peek().litValue
        dut.io.dtcm.rdata.poke(pendingRead.map(x => mem(x)).getOrElse(BigInt(0)).U)
        val v = dut.io.dtcm.valid.peek().litToBoolean
        val w = dut.io.dtcm.write.peek().litToBoolean
        val ad = dut.io.dtcm.addr.peek().litValue
        if (v && w) {
          val wd = dut.io.dtcm.wdata.peek().litValue
          val strb = dut.io.dtcm.wstrb.peek().litValue
          var cur = mem(ad)
          for (byte <- 0 until rowBytes) if (((strb >> byte) & 1) == 1) {
            val mask = BigInt(0xff) << (8 * byte)
            cur = (cur & ~mask) | (((wd >> (8 * byte)) & 0xff) << (8 * byte))
          }
          mem(ad) = cur
          pendingRead = None
        } else if (v) pendingRead = Some(ad) else pendingRead = None
        dut.clock.step(1)
      }
      dut.io.start.poke(true.B); tick(); dut.io.start.poke(false.B)
      var guard = 0
      while (!dut.io.done.peek().litToBoolean) {
        tick(); guard += 1
        assert(guard < maxCycles, s"core did not halt after $guard cycles")
      }
      result = (mem, guard)
    }
    result
  }

  /** mem image (flat u32 LE) -> nonzero 16-byte rows keyed by byte address. */
  def seedRows(mem: Array[BigInt], windowBytes: Int): Map[BigInt, BigInt] =
    (for (r <- 0 until windowBytes / 16) yield {
      var v = BigInt(0); var nz = false
      for (w <- 0 until 4) {
        val x = mem(r * 4 + w)
        if (x != BigInt(0)) nz = true
        v |= x << (32 * w)
      }
      if (nz) Some(BigInt(r * 16) -> v) else None
    }).flatten.toMap

  /** bf16 element `idx` at byte offset `off` in the 16-byte-row model. */
  def bf16At(mem: collection.Map[BigInt, BigInt], off: Int, idx: Int): Float = {
    val byte = off + idx * 2
    val h = ((mem.getOrElse(BigInt(byte & ~15), BigInt(0)) >> (16 * ((byte % 16) / 2))) & 0xFFFFL).toInt
    java.lang.Float.intBitsToFloat(h << 16)
  }

  /** bf16 element `idx` at byte offset `off` in a flat u32 reference image. */
  def refBf16(ref: Array[BigInt], off: Int, idx: Int): Float = {
    val byte = off + idx * 2
    val w = ref(byte / 4).toLong & 0xffffffffL
    java.lang.Float.intBitsToFloat((((w >> (16 * ((byte / 2) & 1))) & 0xFFFFL).toInt) << 16)
  }

  /** Compare a contiguous bf16 region against the reference, rel-to-max. */
  def checkRegion(name: String, fx: Fx, mem: collection.Map[BigInt, BigInt],
                  off: Int, count: Int, tol: Double = 5e-2): Double = {
    var worst = 0.0; var scale = 0.0
    for (i <- 0 until count) {
      val exp = refBf16(fx.memRef, off, i)
      val got = bf16At(mem, off, i)
      if (math.abs(exp) > scale) scale = math.abs(exp)
      val d = math.abs(got - exp)
      if (d > worst) worst = d
    }
    assert(scale > 1e-3, s"$name reference is all-zero; fixture broken, not RTL")
    val rel = worst / scale
    info(f"$name: $count elems, max rel err = $rel%.2e")
    assert(rel < tol, f"$name max relative error $rel%.3e exceeds $tol")
    rel
  }
}
