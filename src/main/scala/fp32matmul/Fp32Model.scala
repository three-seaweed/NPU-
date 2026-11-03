// Bit-accurate pure-Scala model of the FP32 units used by the matmul array.
//
// Semantics (shared by this model and the Chisel hardware in Fp32Ops.scala):
//   * IEEE-754 fp32 inputs, RNE rounding, no exceptions.
//   * FTZ / canonical zero: any result whose exponent field is 0 (+-0 or
//     subnormal) is canonicalized to +0 (0x00000000).  A rounded subnormal
//     result that lands exactly on MIN_NORMAL is kept.
//   * Subnormal INPUTS are treated as +-0 (flush-to-zero input mode).
//   * NaN payloads collapse to the single quiet NaN 0x7fc00000.
//   * fma()  : one single rounding of the exact a*b+c  (fused, "single rounding")
//   * mul/add: each op rounds separately ("double rounding" path)
// The exact sum is kept on a wide BigInt / 512-bit container, so rounding
// decisions are exact (no GRS sticky-tie approximation): model.fma is
// bit-identical to canonicalize(java.lang.Math.fma) for normal inputs.

package fp32matmul

object Fp32Model {
  val QNAN = 0x7fc00000

  private def dec(x: Int): (Boolean, Int, BigInt) =
    (x < 0, (x >> 23) & 0xff, BigInt(x & 0x7fffff))
  private def isNan(x: Int) = ((x >> 23) & 0xff) == 0xff && (x & 0x7fffff) != 0
  private def isInf(x: Int) = ((x >> 23) & 0xff) == 0xff && (x & 0x7fffff) == 0
  private def isZeroCls(x: Int) = ((x >> 23) & 0xff) == 0 // zeros and subnormals
  private def inf(s: Boolean) = if (s) 0xff800000 else 0x7f800000

  // Canonicalize a float's bits to our output convention.
  def canon(x: Int): Int =
    if (isNan(x)) QNAN
    else if (((x >> 23) & 0xff) == 0) 0
    else x

  // Round value = mag0 * 2^xScale (exact, mag0 > 0) to fp32 under our semantics.
  private def roundGen(s: Boolean, mag0: BigInt, xScale: Int): Int = {
    require(mag0 > 0)
    val top = mag0.bitLength - 1
    val e0 = xScale + top + 127
    if (e0 >= 255) return inf(s)
    if (e0 <= 0) {
      // Result lands below the normal range: round on the 2^-149 grid; only a
      // result that reaches exactly MIN_NORMAL survives the flush.
      val sh = top - 22 - e0 // ulps = mag0 >> sh (sh may be negative)
      val ulps = if (sh >= 0) mag0 >> sh else mag0 << -sh
      val rem = if (sh >= 1) mag0 & ((BigInt(1) << sh) - 1) else BigInt(0)
      val half = BigInt(1) << (sh - 1)
      val up = rem > half || (rem == half && ulps.testBit(0))
      val k = ulps + (if (up) 1 else 0)
      if (k == (1 << 23)) 0x00800000 else 0x00000000
    } else {
      val kept = if (top >= 23) mag0 >> (top - 23) else mag0 << (23 - top)
      val g = top >= 24 && mag0.testBit(top - 24)
      val r = top >= 25 && mag0.testBit(top - 25)
      val st = top >= 25 && (mag0 & ((BigInt(1) << (top - 25)) - 1)) != 0
      val up = g && (r || st || kept.testBit(0))
      var k = kept + (if (up) 1 else 0)
      var e = e0
      if (k == (BigInt(1) << 24)) { k = 1 << 23; e += 1 }
      if (e >= 255) return inf(s)
      (if (s) 1 << 31 else 0) | (e << 23) | (k - (1 << 23)).toInt
    }
  }

  // Exact-align add of two significands placed in a 512-bit container with
  // bit 510 <-> 2^ebig.  sig/sigC: BigInt with the MSB pre-placed at bit 510.
  private def coreAdd(ps: Boolean, sigP: BigInt, eP: Int,
                      cs: Boolean, sigC: BigInt, eC: Int): Int = {
    val ebig = math.max(eP, eC)
    val sp = (sigP << (510 - 47)) >> (ebig - eP)
    val sc = (sigC << (510 - 23)) >> (ebig - eC)
    if (ps == cs) roundGen(ps, sp + sc, ebig - 510)
    else if (sp > sc) roundGen(ps, sp - sc, ebig - 510)
    else if (sp < sc) roundGen(cs, sc - sp, ebig - 510)
    else 0 // exact cancellation -> +0
  }

  def fma(a: Int, b: Int, c: Int): Int = {
    if (isNan(a) || isNan(b) || isNan(c)) return QNAN
    val (az, bz, cz) = (isZeroCls(a), isZeroCls(b), isZeroCls(c))
    val (ai, bi, ci) = (isInf(a), isInf(b), isInf(c))
    if ((ai && bz) || (bi && az)) return QNAN
    if (ai || bi) {
      val ps = dec(a)._1 ^ dec(b)._1
      if (ci) return if (dec(c)._1 == ps) inf(ps) else QNAN
      return inf(ps)
    }
    if (ci) return inf(dec(c)._1)
    val cbit = canon(c)
    if (az || bz) return if (cz) 0 else cbit
    val (as, ae, am0) = dec(a); val (bs, be, bm0) = dec(b); val (cs, ce, cm0) = dec(cbit)
    val am = (BigInt(1) << 23) | am0; val bm = (BigInt(1) << 23) | bm0
    var p = am * bm // in [2^46, 2^48)
    var px = ae + be - 300 // value = p * 2^px
    if (!p.testBit(47)) { p <<= 1; px -= 1 }
    val ps = as ^ bs
    if (cz) return roundGen(ps, p, px)
    val cm = (BigInt(1) << 23) | cm0
    coreAdd(ps, p, px + 47, cs, cm, ce - 127)
  }

  def mul(a: Int, b: Int): Int = {
    if (isNan(a) || isNan(b)) return QNAN
    val (az, bz) = (isZeroCls(a), isZeroCls(b))
    if ((isInf(a) && bz) || (isInf(b) && az)) return QNAN
    if (isInf(a) || isInf(b)) return inf(dec(a)._1 ^ dec(b)._1)
    if (az || bz) return 0
    val (as, ae, am0) = dec(a); val (bs, be, bm0) = dec(b)
    val am = (BigInt(1) << 23) | am0; val bm = (BigInt(1) << 23) | bm0
    var p = am * bm; var px = ae + be - 300
    if (!p.testBit(47)) { p <<= 1; px -= 1 }
    roundGen(as ^ bs, p, px)
  }

  def add(a0: Int, b0: Int): Int = {
    if (isNan(a0) || isNan(b0)) return QNAN
    val (ai, bi) = (isInf(a0), isInf(b0))
    if (ai && bi) return if (dec(a0)._1 == dec(b0)._1) inf(dec(a0)._1) else QNAN
    if (ai) return inf(dec(a0)._1)
    if (bi) return inf(dec(b0)._1)
    val (az, bz) = (isZeroCls(a0), isZeroCls(b0))
    if (az && bz) return 0
    if (az) return canon(b0)
    if (bz) return canon(a0)
    val (as, ae, am0) = dec(a0); val (bs, be, bm0) = dec(b0)
    val am = (BigInt(1) << 23) | am0; val bm = (BigInt(1) << 23) | bm0
    // coreAdd places its FIRST operand with MSB at bit 47 (product slot) and
    // its SECOND with MSB at bit 23 (addend slot): pre-shift only am.
    coreAdd(as, am << 24, ae - 127, bs, bm, be - 127)
  }
}
