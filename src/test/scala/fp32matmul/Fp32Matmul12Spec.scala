package fp32matmul

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import xnpu.XnpuBackend
import scala.util.Random

// 12x12 deployment-shape regression (P3 decision: 144 PEs, 59% LUT on
// ZCU102).  The design is parameterized in n, so this only re-runs the whole
// flow at the chosen shape: skew wavefront, done timing 2n-1+(L-1) = 25
// cycles, and all 144 outputs bitwise against the k-order golden.
class Fp32Matmul12Spec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val n = 12

  def genMat(r: Random, k: Int, special: Boolean): (Array[Array[Int]], Array[Array[Int]]) = {
    def word() =
      if (special && r.nextInt(8) == 0) Rand32.specials(r.nextInt(Rand32.specials.size))
      else Rand32.normal(r, 110, 140)
    (Array.tabulate(n, k)((_, _) => word()), Array.tabulate(k, n)((_, _) => word()))
  }

  def golden(aM: Array[Array[Int]], bM: Array[Array[Int]], fused: Boolean): Array[Array[Int]] = {
    val k = bM.length
    Array.tabulate(n, n)((i, j) => {
      var acc = 0
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
    dut.clock.step(1)
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

  private def once(fused: Boolean, k: Int, seed: Int, special: Boolean, tag: String): Unit =
    test(new Fp32Matmul(n, fused, narrow = true)).withAnnotations(backendAnnos) { dut =>
      val r = new Random(seed)
      val (aM, bM) = genMat(r, k, special)
      run(dut, aM, bM)
      check(dut, golden(aM, bM, fused), tag)
    }

  behavior of "Fp32Matmul 12x12 narrow (deployment shape)"
  it should "compute K=16 exactly, fused" in once(fused = true, k = 16, seed = 21, special = false, tag = "12x12 fused K=16")
  it should "compute K=64 exactly, fused" in once(fused = true, k = 64, seed = 22, special = false, tag = "12x12 fused K=64")
  it should "propagate Inf/NaN/zero operands" in once(fused = true, k = 5, seed = 23, special = true, tag = "12x12 K=5 special")
  it should "compute K=8 exactly, double rounding" in once(fused = false, k = 8, seed = 24, special = false, tag = "12x12 double K=8")
}
