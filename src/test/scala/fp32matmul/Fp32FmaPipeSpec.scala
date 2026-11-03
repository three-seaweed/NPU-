package fp32matmul

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import xnpu.XnpuBackend
import scala.util.Random

// Pipeline hazard test for Fp32FmaPipe: a back-to-back stream with NO idle
// cycles between MACs, exactly like the systolic array feeds a PE.  The acc
// feedback loop is only the final stage, so MAC k's final stage must read the
// acc written by MAC k-1 one cycle earlier.  Any loop hazard shows up as a
// wrong acc, checked against the model chained in the same order.
//
// Mid-stream sampling: zero beats preserve acc (fma(0,0,acc) = acc), so we
// insert zero gaps at selected beat indices, drain the pipe, and compare the
// acc against the golden-so-far.  Also covers Inf/NaN streams (sticky NaN),
// exact cancellation pairs, and clear-at-boundary like the array uses it.
class Fp32FmaPipeSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  private def step0(dut: Fp32FmaPipe): Unit = {
    dut.io.a.poke(0.U); dut.io.b.poke(0.U)
    dut.clock.step(1)
  }

  private def goldenStep(fused: Boolean, a: Int, b: Int, acc: Int): Int =
    if (fused) Fp32Model.fma(a, b, acc)
    else Fp32Model.add(Fp32Model.mul(a, b), acc)

  private def runStream(fused: Boolean, beats: Array[(Int, Int)], gaps: Set[Int],
                        tag: String, narrow: Boolean = false): Unit =
    test(new Fp32FmaPipe(fused, narrow)).withAnnotations(backendAnnos) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.a.poke(0.U); dut.io.b.poke(0.U)
      dut.clock.step(Fp32FmaPipe.LATENCY + 1)
      var accM = 0
      for (i <- beats.indices) {
        if (gaps.contains(i)) { // drain + mid-stream sample
          for (_ <- 0 until Fp32FmaPipe.LATENCY + 1) step0(dut)
          val got = dut.io.acc.peek().litValue.toInt
          assert(got == accM, f"$tag gap@$i: acc 0x$got%08x expected 0x$accM%08x")
        }
        dut.io.a.poke((beats(i)._1 & 0xffffffffL).U)
        dut.io.b.poke((beats(i)._2 & 0xffffffffL).U)
        dut.clock.step(1)
        accM = goldenStep(fused, beats(i)._1, beats(i)._2, accM)
      }
      for (_ <- 0 until Fp32FmaPipe.LATENCY + 1) step0(dut)
      val got = dut.io.acc.peek().litValue.toInt
      assert(got == accM, f"$tag final: acc 0x$got%08x expected 0x$accM%08x")
    }

  private def genBeats(r: Random, n: Int, special: Boolean): Array[(Int, Int)] =
    Array.tabulate(n)(_ =>
      if (special && r.nextInt(4) == 0)
        (Rand32.specials(r.nextInt(Rand32.specials.size)),
         Rand32.specials(r.nextInt(Rand32.specials.size)))
      else if (r.nextInt(4) == 0) (Rand32.normal(r, 1, 254), Rand32.normal(r, 1, 254))
      else (Rand32.normal(r, 110, 140), Rand32.normal(r, 110, 140)))

  behavior of "Fp32FmaPipe back-to-back streams (1 MAC/cycle)"

  it should "accumulate 20000 random beats, fused" in {
    val r = new Random(31)
    runStream(fused = true, genBeats(r, 20000, special = false),
      gaps = (97 until 20000 by 97).toSet, tag = "fused random")
  }
  it should "accumulate 20000 random beats, double rounding" in {
    val r = new Random(32)
    runStream(fused = false, genBeats(r, 20000, special = false),
      gaps = (131 until 20000 by 131).toSet, tag = "double random")
  }
  it should "propagate specials through the stream (sticky NaN)" in {
    val r = new Random(33)
    runStream(fused = true, genBeats(r, 4000, special = true),
      gaps = (53 until 4000 by 53).toSet, tag = "fused specials")
  }
  it should "survive exact-cancellation pairs" in {
    val r = new Random(34)
    val beats = Array.tabulate(4000) { i =>
      val a = Rand32.normal(r, 110, 140)
      val b = Rand32.normal(r, 110, 140)
      if (i % 2 == 0) (a, b) else (a, b ^ 0x80000000) // same magnitude, flipped sign
    }
    runStream(fused = true, beats, Set.empty, tag = "fused cancel")
  }

  // narrow (GRS+sticky) back end through the same back-to-back stressor
  it should "accumulate 10000 random beats, narrow fused" in {
    val r = new Random(36)
    runStream(fused = true, genBeats(r, 10000, special = false),
      gaps = (97 until 10000 by 97).toSet, tag = "narrow fused random", narrow = true)
  }
  it should "propagate specials through the stream, narrow double" in {
    val r = new Random(37)
    runStream(fused = false, genBeats(r, 4000, special = true),
      gaps = (53 until 4000 by 53).toSet, tag = "narrow double specials", narrow = true)
  }

  it should "clear at a stream boundary and restart from +0" in {
    test(new Fp32FmaPipe(fused = true)).withAnnotations(backendAnnos) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.a.poke(0.U); dut.io.b.poke(0.U)
      dut.clock.step(Fp32FmaPipe.LATENCY + 1)
      def feed(beats: Array[(Int, Int)], acc0: Int): Int = {
        var accM = acc0
        for ((a, b) <- beats) {
          dut.io.a.poke((a & 0xffffffffL).U)
          dut.io.b.poke((b & 0xffffffffL).U)
          dut.clock.step(1)
          accM = Fp32Model.fma(a, b, accM)
        }
        for (_ <- 0 until Fp32FmaPipe.LATENCY + 1) step0(dut)
        accM
      }
      val r = new Random(35)
      val acc1 = feed(genBeats(r, 500, special = false), 0)
      assert(dut.io.acc.peek().litValue.toInt == acc1, "first stream wrong")
      // array-style clear: asserted while the pipe holds only zero bubbles
      dut.io.clear.poke(true.B)
      dut.clock.step(1)
      dut.io.clear.poke(false.B)
      dut.clock.step(Fp32FmaPipe.LATENCY)
      assert(dut.io.acc.peek().litValue.toInt == 0, "clear did not reset acc")
      val acc2 = feed(genBeats(r, 500, special = false), 0)
      assert(dut.io.acc.peek().litValue.toInt == acc2, "second stream wrong")
    }
  }
}
