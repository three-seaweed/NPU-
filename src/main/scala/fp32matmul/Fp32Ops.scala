// Chisel twin of Fp32Model: IEEE fp32 mul / add / fused-FMA with RNE, FTZ and
// canonical +0, NaN collapsed to 0x7fc00000.  Combinational reference; the
// pipelined PE datapath (Fp32FmaPipe) reuses front/mulBack/fmaBack so the two
// are structurally equivalent.
//
// The exact sum a*b+c is formed on a 512-bit container (bit 510 <-> 2^ebig),
// so every rounding decision is exact - bit-identical to Fp32Model, which is
// bit-identical to canonicalized java.lang.Math.fma.  A synthesis-oriented
// version would narrow this to 24+3 GRS bits with a sticky OR.

package fp32matmul

import chisel3._
import chisel3.util._

// Class bits of (a,b), pipelined alongside the product by Fp32FmaPipe.
class Fp32Fl extends Bundle {
  val ps = Bool() // sign of the product
  val aN = Bool(); val bN = Bool()
  val aI = Bool(); val bI = Bool()
  val aZ = Bool(); val bZ = Bool() // zero class: +-0 or subnormal (input FTZ)
}

object Fp32Ops {
  val QNAN = "h7fc00000".U(32.W)

  private[fp32matmul] def sgn(x: UInt) = x(31)
  private[fp32matmul] def expf(x: UInt) = x(30, 23)
  private[fp32matmul] def mant(x: UInt) = x(22, 0)
  private[fp32matmul] def isNan(x: UInt) = expf(x) === "hff".U && mant(x) =/= 0.U
  private[fp32matmul] def isInf(x: UInt) = expf(x) === "hff".U && mant(x) === 0.U
  private[fp32matmul] def isZeroCls(x: UInt) = expf(x) === 0.U
  private[fp32matmul] def inf(s: Bool) = Cat(s, "hff".U(8.W), 0.U(23.W))

  private[fp32matmul] def pad512(v: UInt): UInt =
    if (v.getWidth >= 512) v else Cat(0.U((512 - v.getWidth).W), v)

  // 24x24 -> 48 exact mantissa multiply as three explicit partial products
  // (Karatsuba 12+12 split) so synthesis maps exactly 3 DSP48E2:
  //   z2 = aH*bH (12x12), z0 = aL*bL (12x12), z1 = (aL+aH)*(bL+bH) (13x13,
  //   the pre-adders land in the DSP pre-add path)
  // Recombination uses truncating + on purpose: max value (2^24-1)^2 < 2^48,
  // the 48-bit container cannot overflow.
  private[fp32matmul] def mul24(a: UInt, b: UInt): UInt = {
    val aH = a(23, 12); val aL = a(11, 0)
    val bH = b(23, 12); val bL = b(11, 0)
    val z2 = aH * bH // 24
    val z0 = aL * bL // 24
    val z1 = (aL +& aH) * (bL +& bH) // 13x13 -> 26
    val mid = z1 - (z0 +& z2) // = aL*bH + aH*bL >= 0, no borrow, 26 bits
    (z2 << 24) + (mid << 12) + z0 // 48
  }

  // hardware twin of Fp32Model.roundGen: value = mag * 2^xScale, mag != 0.
  private[fp32matmul] def roundGen(s: Bool, mag: UInt, xScale: SInt): UInt = {
    // PriorityEncoder(Reverse(mag)) = 511 - msbIndex = the left-shift that
    // pins the MSB at bit 511.
    val l = PriorityEncoder(Reverse(mag))
    val magN = (mag << l)(511, 0) // MSB pinned at bit 511
    val xs = xScale -& l.zext // weight of bit 0 after normalize
    val e0 = xs +& 638.S // == xScale + msbIndex + 127
    val subRes = { // e0 <= 0: round on the 2^-149 grid; only MIN_NORMAL survives
      val sh = (-xs -& 149.S).asUInt // >= 489 in this branch
      val ulps = magN >> sh // < 2^23
      val up = (magN >> (sh - 1.U))(0) && ((magN >> (sh - 2.U)) =/= 0.U || (magN >> sh)(0))
      val k = (ulps +& Mux(up, 1.U, 0.U))(23, 0)
      Mux(k === (1 << 23).U, "h00800000".U, 0.U)
    }
    val normRes = {
      val kept = magN(511, 488)
      val up = magN(487) && (magN(486) || magN(485, 0).orR || kept(0))
      val k = kept +& Mux(up, 1.U, 0.U) // 25 bits
      val carry = k(24)
      val e1 = e0 +& Mux(carry, 1.S, 0.S)
      Mux(e1 >= 255.S, inf(s), Cat(s, e1(7, 0), Mux(carry, 0.U(23.W), k(22, 0))))
    }
    Mux(e0 >= 255.S, inf(s), Mux(e0 <= 0.S, subRes, normRes))
  }

  // hardware twin of Fp32Model.coreAdd: exact-align add in the 512-bit
  // container.  sigP carries its MSB at bit 47 (product slot), sigC at bit 23
  // (addend slot); eP/eC are the exponents of those MSBs.
  private[fp32matmul] def coreAdd(ps: Bool, sigP: UInt, eP: SInt, cs: Bool, sigC: UInt, eC: SInt): UInt = {
    val ebig = Mux(eP >= eC, eP, eC)
    val sp = pad512(((sigP << 463) >> (ebig -& eP).asUInt)(510, 0))
    val sc = pad512(((sigC << 487) >> (ebig -& eC).asUInt)(510, 0))
    val sameS = ps === cs
    val mag = Mux(sameS, (sp +& sc)(511, 0),
      Mux(sp >= sc, (sp - sc)(511, 0), (sc - sp)(511, 0)))
    val signOut = Mux(sameS, ps, Mux(sp >= sc, ps, cs))
    Mux(!sameS && sp === sc, 0.U,
      roundGen(signOut, pad512(mag), ebig -& 510.S))
  }

  // Decode a/b into significands plus class flags (shared by fma and mul,
  // and registered as pipeline stage 0 by Fp32FmaPipe).
  private[fp32matmul] def front(a: UInt, b: UInt): (UInt, UInt, Fp32Fl) = {
    val fl = Wire(new Fp32Fl)
    fl.ps := sgn(a) ^ sgn(b)
    fl.aN := isNan(a); fl.bN := isNan(b)
    fl.aI := isInf(a); fl.bI := isInf(b)
    fl.aZ := isZeroCls(a); fl.bZ := isZeroCls(b)
    val am = Mux(fl.aZ, 0.U, Cat(1.U(1.W), mant(a)))
    val bm = Mux(fl.bZ, 0.U, Cat(1.U(1.W), mant(b)))
    (am, bm, fl)
  }

  // Tail of mul: round the exact product once (no addend).  Takes the
  // pipelined product (pn MSB at bit 47, value = pn * 2^px) and flags.
  private[fp32matmul] def mulBack(fl: Fp32Fl, pn: UInt, px: SInt): UInt = {
    val finite = Mux(pn === 0.U, 0.U, roundGen(fl.ps, pad512(pn), px))
    MuxCase(finite, Seq(
      (fl.aN || fl.bN || (fl.aI && fl.bZ) || (fl.bI && fl.aZ)) -> QNAN,
      (fl.aI || fl.bI) -> inf(fl.ps),
    ))
  }

  // Tail of fma: exact-align the product against the LIVE addend c (the acc
  // register), one RNE, specials MuxCase.  c is decoded combinationally here,
  // which is what keeps the acc feedback loop a single cycle.
  private[fp32matmul] def fmaBack(fl: Fp32Fl, pn: UInt, px: SInt, c: UInt): UInt = {
    val cN = isNan(c); val cI = isInf(c); val cZ = isZeroCls(c)
    val cword = Mux(cZ, 0.U, c) // canonicalize the addend (+-0/subnormal -> +0)
    val finite = Mux(pn === 0.U, cword,
      Mux(cZ, roundGen(fl.ps, pad512(pn), px),
        coreAdd(fl.ps, pad512(pn), px +& 47.S, sgn(cword),
          Cat(1.U(1.W), mant(cword)), expf(cword).zext -& 127.S)))
    val prodNan = (fl.aI && fl.bZ) || (fl.bI && fl.aZ)
    MuxCase(finite, Seq(
      (fl.aN || fl.bN || cN || prodNan) -> QNAN,
      (fl.aI || fl.bI) -> Mux(cI, Mux(sgn(c) === fl.ps, inf(fl.ps), QNAN), inf(fl.ps)),
      cI -> inf(sgn(c)),
    ))
  }

  def mul(a: UInt, b: UInt): UInt = {
    val (am, bm, fl) = front(a, b)
    val prod = mul24(am, bm)
    val pn = Mux(prod(47), prod, prod << 1)
    val px = expf(a).zext +& expf(b).zext -& 300.S -& Mux(prod(47), 0.S, 1.S)
    mulBack(fl, pn, px)
  }

  def add(a0: UInt, b0: UInt): UInt = {
    val aN = isNan(a0); val bN = isNan(b0)
    val aI = isInf(a0); val bI = isInf(b0)
    val aZ = isZeroCls(a0); val bZ = isZeroCls(b0)
    val am = Mux(aZ, 0.U, Cat(1.U(1.W), mant(a0)))
    val bm = Mux(bZ, 0.U, Cat(1.U(1.W), mant(b0)))
    val eA = expf(a0).zext -& 127.S
    val eB = expf(b0).zext -& 127.S
    val core = coreAdd(sgn(a0), am << 24, eA, sgn(b0), bm, eB)
    def canonWord(x: UInt) = Mux(isZeroCls(x), 0.U, x)
    val finite = Mux(aZ && bZ, 0.U,
      Mux(aZ, canonWord(b0), Mux(bZ, canonWord(a0), core)))
    MuxCase(finite, Seq(
      (aN || bN) -> QNAN,
      (aI && bI) -> Mux(sgn(a0) === sgn(b0), inf(sgn(a0)), QNAN),
      aI -> inf(sgn(a0)),
      bI -> inf(sgn(b0)),
    ))
  }

  def fma(a: UInt, b: UInt, c: UInt): UInt = {
    val (am, bm, fl) = front(a, b)
    val prod = mul24(am, bm) // 48 bits, in [2^46, 2^48) when nonzero
    val pn = Mux(prod(47), prod, prod << 1) // MSB pinned at bit 47
    val px = expf(a).zext +& expf(b).zext -& 300.S -& Mux(prod(47), 0.S, 1.S)
    fmaBack(fl, pn, px, c)
  }
}

// Combinational test harness: exposes all three ops as module outputs.
class Fp32OpsHarness extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Input(UInt(32.W))
    val c = Input(UInt(32.W))
    val fma = Output(UInt(32.W))
    val mul = Output(UInt(32.W))
    val add = Output(UInt(32.W))
  })
  io.fma := Fp32Ops.fma(io.a, io.b, io.c)
  io.mul := Fp32Ops.mul(io.a, io.b)
  io.add := Fp32Ops.add(io.a, io.b)
}
