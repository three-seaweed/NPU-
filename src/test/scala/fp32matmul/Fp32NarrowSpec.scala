package fp32matmul

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import xnpu.XnpuBackend
import scala.util.Random

// Narrow (GRS+sticky) back end vs the exact model, bitwise.  Pools target the
// four regimes of the coverage argument: adjacent-MSB deep cancellation,
// exponent-gap fold boundary (dP/dC crossing 24..27 and 50), products
// straddling MIN_NORMAL (e0 near 0), and cancellation landing in the
// subnormal grid (needs bits near weights -149..-151).
class Fp32NarrowSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  private def chk(what: String, a: Int, b: Int, c: Int, got: BigInt, exp: Int): Unit =
    assert(got == (exp & 0xffffffffL),
      f"$what(0x$a%08x, 0x$b%08x, 0x$c%08x) = 0x${got.toString(16)}, expected 0x${exp & 0xffffffffL}%08x")

  // c = -(a*b) shifted by k ulps on the word magnitude (normal numbers only)
  private def negProdUlp(p: Int, k: Int): Int = {
    val mag = (p & 0x7fffffff) + k // ulp steps carry into the exponent field
    (p & 0x80000000) ^ 0x80000000 | mag // flip the product's sign
  }

  behavior of "Fp32Narrow (GRS+sticky) vs exact model"

  it should "pass the directed specials cross" in {
    test(new Fp32NarrowHarness).withAnnotations(backendAnnos) { dut =>
      for (a <- Rand32.specials; b <- Rand32.specials; c <- Rand32.specials) {
        dut.io.a.poke((a & 0xffffffffL).U); dut.io.b.poke((b & 0xffffffffL).U)
        dut.io.c.poke((c & 0xffffffffL).U)
        chk("fma", a, b, c, dut.io.fma.peek().litValue, Fp32Model.fma(a, b, c))
        chk("mul", a, b, c, dut.io.mul.peek().litValue, Fp32Model.mul(a, b))
        chk("add", a, b, c, dut.io.add.peek().litValue, Fp32Model.add(a, b))
      }
      println("Fp32NarrowSpec: 512 specials x 3 ops bitwise equal")
    }
  }

  // The 2026-09-11 silicon erratum corner (docs/Fp32_NarrowFma_Erratum.md):
  // far-path landing exactly on a rounding tie (g=1, r=0, container lows 0)
  // with a nonzero shrSticky residue.  vec0/1/3 are the subtraction direction
  // the silicon rule rounded up by 1 ulp (board hit: qkv y[477]-class); vec2
  // is the addition-direction sentinel the silicon rule already got right -
  // any fix must keep it right.
  it should "match fma on the erratum corner vectors" in {
    test(new Fp32NarrowHarness).withAnnotations(backendAnnos) { dut =>
      val vecs = Seq(
        (0x6229c774, 0xcfb00000, 0x1aa10d2c),
        (0xd6a45190, 0xb59c0000, 0x82be37dd),
        (0x9c2c0000, 0xf3a51b30, 0x310011e9),
        (0x7b023000, 0xc1277400, 0x07198e28))
      for ((a, b, c) <- vecs) {
        dut.io.a.poke((a & 0xffffffffL).U); dut.io.b.poke((b & 0xffffffffL).U)
        dut.io.c.poke((c & 0xffffffffL).U)
        chk("fma", a, b, c, dut.io.fma.peek().litValue, Fp32Model.fma(a, b, c))
      }
      println("Fp32NarrowSpec: 4 erratum corner vectors bitwise equal")
    }
  }

  it should "match fma on 120k random vectors across all four regimes" in {
    test(new Fp32NarrowHarness).withAnnotations(backendAnnos) { dut =>
      val r = new Random(41)
      def t(a: Int, b: Int, c: Int): Unit = {
        dut.io.a.poke((a & 0xffffffffL).U); dut.io.b.poke((b & 0xffffffffL).U)
        dut.io.c.poke((c & 0xffffffffL).U)
        chk("fma", a, b, c, dut.io.fma.peek().litValue, Fp32Model.fma(a, b, c))
      }
      for (_ <- 0 until 30000) t(Rand32.normal(r, 1, 254), Rand32.normal(r, 1, 254), Rand32.normal(r, 1, 254))
      for (_ <- 0 until 30000) t(Rand32.normal(r, 110, 140), Rand32.normal(r, 110, 140), Rand32.normal(r, 110, 140))
      // cancellation: c = -(a*b) +- a few ulps (normal zone)
      for (_ <- 0 until 30000) {
        val a = Rand32.normal(r, 110, 140); val b = Rand32.normal(r, 110, 140)
        val p = Fp32Model.mul(a, b)
        t(a, b, negProdUlp(p, r.nextInt(9) - 4))
      }
      // products straddling MIN_NORMAL, plus cancellation into the subnormal grid
      for (_ <- 0 until 30000) {
        val a = Rand32.normal(r, 58, 66); val b = Rand32.normal(r, 58, 66)
        val p = Fp32Model.mul(a, b)
        val c = if (r.nextInt(2) == 0) Rand32.normal(r, 50, 74) else negProdUlp(p, r.nextInt(17) - 8)
        t(a, b, c)
      }
      // exponent-gap sweep: dP/dC crossing the fold boundaries 24..27 and 50
      for (gap <- 0 to 60; _ <- 0 until 100) {
        val a = Rand32.normal(r, 120, 130); val b = Rand32.normal(r, 120, 130)
        val c = Rand32.normal(r, 120 - gap, 130 - gap)
        t(a, b, c)
      }
      println("Fp32NarrowSpec: 126100 fma vectors bitwise equal")
    }
  }

  it should "match mul and add on 80k random vectors" in {
    test(new Fp32NarrowHarness).withAnnotations(backendAnnos) { dut =>
      val r = new Random(42)
      for (_ <- 0 until 40000) {
        val a = Rand32.normal(r, 1, 254); val b = Rand32.normal(r, 1, 254)
        dut.io.a.poke((a & 0xffffffffL).U); dut.io.b.poke((b & 0xffffffffL).U)
        chk("mul", a, b, 0, dut.io.mul.peek().litValue, Fp32Model.mul(a, b))
      }
      for (_ <- 0 until 20000) {
        val a = Rand32.normal(r, 1, 254); val b = Rand32.normal(r, 1, 254)
        dut.io.a.poke((a & 0xffffffffL).U); dut.io.b.poke((b & 0xffffffffL).U)
        chk("add", a, b, 0, dut.io.add.peek().litValue, Fp32Model.add(a, b))
      }
      // add cancellation pairs, x + (-x +- ulps)
      for (_ <- 0 until 20000) {
        val a = Rand32.normal(r, 1, 254)
        val b = negProdUlp(a, r.nextInt(9) - 4)
        dut.io.a.poke((a & 0xffffffffL).U); dut.io.b.poke((b & 0xffffffffL).U)
        chk("add", a, b, 0, dut.io.add.peek().litValue, Fp32Model.add(a, b))
      }
      println("Fp32NarrowSpec: 80000 mul/add vectors bitwise equal")
    }
  }
}
