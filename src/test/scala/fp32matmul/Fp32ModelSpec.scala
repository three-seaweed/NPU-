package fp32matmul

import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

object Rand32 {
  // Random normal fp32 bit pattern with exponent in [lo, hi].  Never generates
  // zero / subnormal / Inf / NaN inputs (those are covered by directed cases);
  // subnormal RESULTS still occur through cancellation.
  def normal(r: Random, lo: Int, hi: Int): Int =
    (if (r.nextBoolean()) 1 << 31 else 0) | ((lo + r.nextInt(hi - lo + 1)) << 23) | r.nextInt(1 << 23)

  val specials = Seq(
    0x00000000, // +0
    0x80000000, // -0
    0x7f800000, // +Inf
    0xff800000, // -Inf
    0x7fc00000, // NaN
    0x3f800000, // 1.0f
    0xbf800000, // -1.0f
    0x00800000, // MIN_NORMAL
  )
}

class Fp32ModelSpec extends AnyFlatSpec {
  private def f(x: Int) = java.lang.Float.intBitsToFloat(x)
  private def canonF(x: Float) = Fp32Model.canon(java.lang.Float.floatToIntBits(x))

  private var mism = 0
  private def chk(what: String, args: Seq[Int], got: Int, exp: Int): Unit = {
    if (got != exp) {
      mism += 1
      if (mism <= 12)
        println(f"$what(${args.map(a => f"0x$a%08x").mkString(", ")}) = 0x$got%08x  expected 0x$exp%08x")
    }
  }

  "Fp32Model.fma" should "be bit-identical to canonicalized Math.fma" in {
    val r = new Random(42)
    // 1. directed specials: full cross product
    for (a <- Rand32.specials; b <- Rand32.specials; c <- Rand32.specials)
      chk("fma", Seq(a, b, c),
        got = Fp32Model.fma(a, b, c),
        exp = canonF(Math.fma(f(a), f(b), f(c))))
    // 2. random pools: full exponent range + narrow band + cancellation-prone c
    for (i <- 0 until 400000) {
      val a = Rand32.normal(r, 1, 254)
      val b = Rand32.normal(r, 1, 254)
      val c =
        if (i % 2 == 0) Rand32.normal(r, if (r.nextBoolean()) 1 else 110, if (r.nextBoolean()) 254 else 140)
        else { // c ~= -(a*b) +- a few ulps -> deep cancellation
          var bits = java.lang.Float.floatToIntBits(-(f(a) * f(b))) + r.nextInt(9) - 4
          if (((bits >> 23) & 0xff) == 0 || ((bits >> 23) & 0xff) == 0xff) bits = Rand32.normal(r, 110, 140)
          bits
        }
      chk("fma", Seq(a, b, c),
        got = Fp32Model.fma(a, b, c),
        exp = canonF(Math.fma(f(a), f(b), f(c))))
    }
    assert(mism == 0, s"$mism fma mismatches")
  }

  "Fp32Model.mul" should "be bit-identical to canonicalized float multiply" in {
    val r = new Random(7)
    for (a <- Rand32.specials; b <- Rand32.specials)
      chk("mul", Seq(a, b), got = Fp32Model.mul(a, b), exp = canonF(f(a) * f(b)))
    for (_ <- 0 until 200000) {
      val a = Rand32.normal(r, 1, 254); val b = Rand32.normal(r, 1, 254)
      chk("mul", Seq(a, b), got = Fp32Model.mul(a, b), exp = canonF(f(a) * f(b)))
    }
    assert(mism == 0, s"$mism mul mismatches")
  }

  "Fp32Model.add" should "be bit-identical to canonicalized float add" in {
    val r = new Random(9)
    for (a <- Rand32.specials; b <- Rand32.specials)
      chk("add", Seq(a, b), got = Fp32Model.add(a, b), exp = canonF(f(a) + f(b)))
    for (_ <- 0 until 200000) {
      val a = Rand32.normal(r, 1, 254); val b = Rand32.normal(r, 1, 254)
      chk("add", Seq(a, b), got = Fp32Model.add(a, b), exp = canonF(f(a) + f(b)))
    }
    assert(mism == 0, s"$mism add mismatches")
  }
}
