// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable

// End-to-end: a full program runs through the Core (Decode -> Control ->
// Alu(arbiter, rd merge) -> VectorUnit -> DTCM), exercising vector config ops,
// an elementwise op writing DTCM, a reduction writing a GPR, and a dependent
// scalar instruction that must observe the (blocking) reduction's result.
class VectorProgramSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val p = Parameters()
  val L = p.Bf16LanesPerRow
  val rowBytes = p.DtcmRowBytes

  def floatToBf16(f: Float): BigInt = {
    val bits = java.lang.Float.floatToIntBits(f)
    val lsb = (bits >>> 16) & 1
    BigInt((bits + 0x7FFF + lsb) >>> 16) & 0xFFFF
  }
  def bf16ToFloat(b: BigInt): Float = java.lang.Float.intBitsToFloat((b.toInt & 0xFFFF) << 16)
  def bitsToF(b: BigInt): Float = java.lang.Float.intBitsToFloat(b.toInt)
  def packRow(vals: Seq[Float]): BigInt =
    vals.zipWithIndex.foldLeft(BigInt(0)) { case (a, (v, k)) => a | (floatToBf16(v) << (16 * k)) }

  it should "run a vector program: VADD to DTCM and VDOT to a GPR" in {
    val baseA = 0x000; val baseB = 0x100; val baseD = 0x200
    val program = Seq(
      Encode.addi(1, 0, baseA),
      Encode.addi(2, 0, baseB),
      Encode.addi(3, 0, baseD),
      Encode.addi(4, 0, 8),       // vlen = 8
      Encode.vseta(1),
      Encode.vsetb(2),
      Encode.vsetd(3),
      Encode.vsetlen(4),
      Encode.vadd(),              // dst = a + b
      Encode.vdot(5),             // x5 = fp32 sum(a*b)
      Encode.addi(6, 5, 0),       // x6 = x5 (must see the reduction result)
      Encode.halt,
    )
    def instWord(byteAddr: BigInt): BigInt = {
      val idx = (byteAddr >> 2).toInt
      if (idx >= 0 && idx < program.length) program(idx) else BigInt(0)
    }

    val a = (0 until 8).map(i => (i - 4) * 0.5f)
    val b = (0 until 8).map(i => (i % 3 + 1) * 1.0f)

    test(new Core(p)).withAnnotations(backendAnnos) { dut =>
      val mem = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
      mem(BigInt(baseA)) = packRow(a)
      mem(BigInt(baseB)) = packRow(b)

      var pendingRead: Option[BigInt] = None
      var instAddrPrev: BigInt = 0
      dut.io.dtcm.ready.poke(true.B)
      dut.io.dmaIdle.poke(true.B)   // no core-issued DMA here; engine always idle
      dut.clock.setTimeout(0)   // our own `guard` bounds the run; disable the watchdog

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
        tick(); guard += 1; assert(guard < 20000, "core did not halt")
      }

      // VADD result in DTCM.
      val outRow = mem(BigInt(baseD))
      for (i <- 0 until 8) {
        val got = bf16ToFloat((outRow >> (16 * i)) & 0xFFFF)
        val exp = a(i) + b(i)
        assert(math.abs(got - exp) <= 0.05f * (1 + math.abs(exp)), s"add lane $i: $got vs $exp")
      }

      // VDOT result in x5, propagated to x6.
      val dot = a.zip(b).map { case (x, y) => x * y }.sum
      val x5 = bitsToF(dut.io.dbg(5).peek().litValue)
      assert(math.abs(x5 - dot) <= 0.05f * (1 + math.abs(dot)), s"vdot x5: $x5 vs $dot")
      assert(dut.io.dbg(6).peek().litValue == dut.io.dbg(5).peek().litValue, "x6 must equal x5")
    }
  }
}
