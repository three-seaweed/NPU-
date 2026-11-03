package fp32matmul

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import xnpu.XnpuBackend
import scala.util.Random

// Whole-flow check of the standalone fp32 matmul: C = A(n x K) * B(K x n),
// all n*n = 256 outputs compared BITWISE against the Scala golden (same
// strict k-order accumulation as the array), for both rounding modes.
// Completeness: inReady handshake during feed, done arrives after exactly
// 2n drain cycles, every output element checked, and a second run on the
// same DUT verifies the accumulators were cleared.
class Fp32MatmulSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val n = 16

  def genMat(r: Random, k: Int, special: Boolean): (Array[Array[Int]], Array[Array[Int]]) = {
    def word() =
      if (special && r.nextInt(8) == 0) Rand32.specials(r.nextInt(Rand32.specials.size))
      else Rand32.normal(r, 110, 140)
    (Array.tabulate(n, k)((_, _) => word()), Array.tabulate(k, n)((_, _) => word()))
  }

  def golden(aM: Array[Array[Int]], bM: Array[Array[Int]], fused: Boolean): Array[Array[Int]] = {
    val k = bM.length
    Array.tabulate(n, n)((i, j) => {
      var acc = 0 // +0
      for (t <- 0 until k)
        acc = if (fused) Fp32Model.fma(aM(i)(t), bM(t)(j), acc)
              else Fp32Model.add(Fp32Model.mul(aM(i)(t), bM(t)(j)), acc)
      acc
    })
  }

  def run(dut: Fp32Matmul, aM: Array[Array[Int]], bM: Array[Array[Int]]): Unit = {
    val k = bM.length
    dut.io.inA.foreach(_.poke(0.U))
    dut.io.inB.foreach(_.poke(0.U))
    dut.io.inValid.poke(false.B)
    dut.io.kLen.poke(k.U)
    dut.io.start.poke(true.B)
    dut.clock.step(1)
    dut.io.start.poke(false.B)
    for (t <- 0 until k) {
      for (i <- 0 until n) dut.io.inA(i).poke((aM(i)(t) & 0xffffffffL).U)
      for (j <- 0 until n) dut.io.inB(j).poke((bM(t)(j) & 0xffffffffL).U)
      dut.io.inValid.poke(true.B)
      assert(dut.io.inReady.peek().litToBoolean, s"inReady low at beat $t")
      dut.clock.step(1)
    }
    dut.io.inValid.poke(false.B)
    var t = 0
    while (!dut.io.done.peek().litToBoolean && t < 4 * n + 64) { dut.clock.step(1); t += 1 }
    assert(dut.io.done.peek().litToBoolean, "done never asserted")
    val expDone = 2 * n - 1 + (Fp32FmaPipe.LATENCY - 1)
    assert(t == expDone, s"drain/done took $t cycles, expected $expDone")
    dut.clock.step(1) // leave sDone, back to sIdle for the next run
  }

  def check(dut: Fp32Matmul, exp: Array[Array[Int]], tag: String): Unit = {
    var bad = 0
    for (i <- 0 until n; j <- 0 until n) {
      val got = dut.io.outC(i * n + j).peek().litValue.toInt
      if (got != exp(i)(j)) {
        bad += 1
        if (bad <= 8) println(f"$tag C($i%d,$j%d) = 0x$got%08x  expected 0x${exp(i)(j)}%08x")
      }
    }
    assert(bad == 0, s"$tag: $bad of ${n * n} outputs wrong")
  }

  private def once(fused: Boolean, k: Int, seed: Int, special: Boolean, tag: String,
                   narrow: Boolean = false): Unit =
    test(new Fp32Matmul(n, fused, narrow)).withAnnotations(backendAnnos) { dut =>
      val r = new Random(seed)
      val (aM, bM) = genMat(r, k, special)
      run(dut, aM, bM)
      check(dut, golden(aM, bM, fused), tag)
    }

  behavior of "Fp32Matmul fused (single rounding, FMA)"
  it should "compute K=1 exactly" in once(fused = true, k = 1, seed = 1, special = false, tag = "fused K=1")
  it should "compute K=16 exactly" in once(fused = true, k = 16, seed = 2, special = false, tag = "fused K=16")
  it should "compute K=64 exactly" in once(fused = true, k = 64, seed = 3, special = false, tag = "fused K=64")
  it should "propagate Inf/NaN/zero operands" in
    once(fused = true, k = 5, seed = 4, special = true, tag = "fused K=5 special")

  behavior of "Fp32Matmul mul-then-add (double rounding)"
  it should "compute K=1 exactly" in once(fused = false, k = 1, seed = 5, special = false, tag = "double K=1")
  it should "compute K=16 exactly" in once(fused = false, k = 16, seed = 6, special = false, tag = "double K=16")
  it should "compute K=64 exactly" in once(fused = false, k = 64, seed = 7, special = false, tag = "double K=64")
  it should "propagate Inf/NaN/zero operands" in
    once(fused = false, k = 5, seed = 8, special = true, tag = "double K=5 special")

  behavior of "Fp32Matmul narrow (GRS+sticky back end)"
  it should "compute K=16 exactly, fused" in
    once(fused = true, k = 16, seed = 11, special = false, tag = "narrow fused K=16", narrow = true)
  it should "compute K=64 exactly, fused" in
    once(fused = true, k = 64, seed = 12, special = false, tag = "narrow fused K=64", narrow = true)
  it should "propagate Inf/NaN/zero operands, fused" in
    once(fused = true, k = 5, seed = 13, special = true, tag = "narrow fused K=5 special", narrow = true)
  it should "compute K=16 exactly, double rounding" in
    once(fused = false, k = 16, seed = 14, special = false, tag = "narrow double K=16", narrow = true)

  it should "clear accumulators between runs on the same DUT" in {
    test(new Fp32Matmul(n, fused = true)).withAnnotations(backendAnnos) { dut =>
      val r = new Random(9)
      val (a1, b1) = genMat(r, 64, false)
      run(dut, a1, b1)
      check(dut, golden(a1, b1, true), "rerun 1st")
      val (a2, b2) = genMat(r, 8, false)
      run(dut, a2, b2)
      check(dut, golden(a2, b2, true), "rerun 2nd")
    }
  }
}
