// Carry-save accumulator back end (2026-11-02 area diet, item 3).
//
// Replaces the per-MAC normalize+round tail (Fp32Narrow.fmaBackN, 894 LUT/PE)
// with a sticky-windowed two's-complement carry-save container:
//
//   container = (S, C, E, sticky, exc)
//   S, C : W-bit two's-complement halves, value = (S + C)
//   E    : anchor; window bit i carries weight 2^(E-W+1+i)
//   sticky: nonzero weight < 2^(E-W+1) was dropped (signless)
//   exc   : latched operand exception (Inf+sign, or NaN)
//
// Per MAC (S2, one cycle, no carry propagation, no magnitude compare):
//   * pn == 0 is an arithmetic no-op (bubble beats stay free); exceptions
//     still latch -- 0*Inf = NaN has pn == 0.
//   * eP = px+47+HEAD is the anchor a product wants.  If it sits above E the
//     window re-anchors: S and C each arithmetic-shift right by d = min(eP-E,
//     W) independently (dropped bits OR into sticky; d >= W wipes), and the
//     product's bits [47 : 47-W+1+HEAD] enter at the window top.  Otherwise
//     the sign-extended product is shifted right into the window by
//     sh = E-eP+47-(W-1); bits dropping below bit 0 OR into sticky.
//   * the aligned product joins (S, C) through a 3:2 carry-save adder.
//     Mixed signs are free -- the point of two's complement.
//   * HEAD=6 leaves room for 2^HEAD max-magnitude products on a chain, so
//     the sum of any 64 MACs cannot wrap the window.
//
// Exactness: a bf16 x bf16 product has at most 16 significant bits, so a
// product whose MSB lands inside the window enters EXACTLY (dropped low bits
// are zero); sticky only accrues from products below the window bottom and
// from re-anchoring.  Meant for the bf16 tier (bf16-widened operands only).
//
// Rounding happens ONCE, when the store path reads the accumulator
// (Fp32Csa.toF32): propagate S+C, magnitude + normalize + single RNE with
// sticky (normal and subnormal tails).  Numerics differ from the per-MAC
// narrow back end (one rounding instead of K); the tier's golden model
// changes with it -- Fp32CsaModel below mirrors the RTL bit for bit,
// including the independent S/C shift on re-anchor.

package fp32matmul

import chisel3._
import chisel3.util._

class Fp32CsaAcc extends Bundle {
  val s = UInt(Fp32Csa.W.W)
  val c = UInt(Fp32Csa.W.W)
  val e = SInt(11.W)
  val st = Bool()
  val exc = UInt(2.W)
  val exS = Bool()
}

object Fp32Csa {
  val W = 32    // container width
  val HEAD = 6  // re-anchor headroom bits (window holds 2^HEAD max products)

  val EXC_NONE = 0
  val EXC_INF = 1
  val EXC_NAN = 2

  // Anchor domain: px in [-266, 242] -> eP = px+47+HEAD in [-213, 295].
  // EMPTY sits below every possible anchor so the first product re-anchors.
  val emptyE = -512

  def cleared(): Fp32CsaAcc = {
    val w = WireDefault(0.U.asTypeOf(new Fp32CsaAcc))
    w.e := Fp32Csa.emptyE.S
    w
  }

  // One MAC.  fl/pn/px follow the S2 conventions of Fp32Narrow.fmaBackN:
  // pn's MSB pinned at bit 47, product value = pn * 2^px.
  def mac(cont: Fp32CsaAcc, fl: Fp32Fl, pn: UInt, px: SInt): Fp32CsaAcc = {
    val out = Wire(new Fp32CsaAcc)
    val eP = px +& 47.S +& HEAD.S
    val empty = cont.e === emptyE.S

    // exception latch: NaN over Inf; a latched code never clears
    val prodNan = (fl.aI && fl.bZ) || (fl.bI && fl.aZ)
    val excNan = fl.aN || fl.bN || prodNan
    val excInf = fl.aI || fl.bI

    val dMove = eP -& cont.e
    val move = empty || dMove > 0.S

    // no-move placement: product bit j lands at window bit j + eP - E - 47 +
    // (W-1), i.e. shift the sign-extended product right by sh = E-eP+47-(W-1).
    // Masks are built in bounded containers (Chisel's dynamic << widens by
    // 2^shift-bits, so a wide shift amount would explode the literal).
    val shNm = cont.e -& eP +& (47 - W + 1).S
    val shCl = Mux(shNm > 63.S, 63.S, Mux(shNm < 0.S, 0.S, shNm)).asUInt(5, 0)
    val maskNm = ((1.U(112.W) << shCl) - 1.U)(47, 0)
    // sign-extended product: 49-bit two's complement (a bare 48-bit pattern
    // cannot hold -pn for pn >= 2^47, and UInt shifts would not extend sign)
    val pExt = Cat(0.U(1.W), pn).asSInt
    val pSgn = Mux(fl.ps, (0.S(49.W) - pExt), pExt).asUInt // 49-bit pattern
    val pNm = (pSgn.asSInt >> shCl).asUInt(W - 1, 0) // arithmetic shift
    val droppedNm = shCl =/= 0.U && (pSgn & maskNm) =/= 0.U
    val stNm = Mux(shNm > 47.S, pn =/= 0.U, droppedNm)

    // move placement: shift S/C down by dm (wipe at d >= W), product's
    // [47 : 47-W+1+HEAD] at the window top (for W=32,HEAD=6: bits [47:22])
    val wipe = !empty && dMove >= W.S
    val dm = Mux(dMove > W.S, W.S, dMove).asUInt(5, 0)
    val maskDm = ((1.U(64.W) << dm) - 1.U)(W - 1, 0)
    val sSh = (cont.s.asSInt >> dm).asUInt
    val cSh = (cont.c.asSInt >> dm).asUInt
    val dropS = !empty && dm =/= 0.U && (cont.s & maskDm) =/= 0.U
    val dropC = !empty && dm =/= 0.U && (cont.c & maskDm) =/= 0.U
    // move-branch addend: +/- the product's top (W-HEAD) magnitude bits as a
    // W-bit two's complement (the sign is fl.ps -- the slice's own MSB is the
    // product's pinned leading 1, not a sign)
    val mag26 = Cat(0.U(HEAD.W), pn(47, 47 - W + 1 + HEAD)).asSInt
    val pMvX = Mux(fl.ps, (0.S(32.W) - mag26), mag26).asUInt
    val stMv = dropS || dropC || (pn(47 - W + HEAD, 0) =/= 0.U)

    val x = Mux(move, pMvX, pNm)
    val sBase = Mux(move, Mux(wipe, 0.U, sSh), cont.s)
    val cBase = Mux(move, Mux(wipe, 0.U, cSh), cont.c)
    val stAdd = Mux(move, stMv, stNm)

    // 3:2 carry-save: (s', c') keeps (x + S + C) mod 2^W exactly
    val sNew = x ^ sBase ^ cBase
    val cNew = (((x & sBase) | (sBase & cBase) | (cBase & x)) << 1)(W - 1, 0)

    val doMac = pn =/= 0.U
    val wasExc = cont.exc =/= EXC_NONE.U
    out.s := Mux(doMac, sNew, cont.s)
    out.c := Mux(doMac, cNew, cont.c)
    out.e := Mux(move && doMac, eP(10, 0).asSInt, cont.e)
    out.st := Mux(doMac, cont.st | stAdd, cont.st)
    out.exc := Mux(excNan, EXC_NAN.U, Mux(excInf, EXC_INF.U,
      Mux(wasExc, cont.exc, EXC_NONE.U)))
    out.exS := Mux(excInf && !excNan, fl.ps, cont.exS)
    out
  }

  // Drain: propagate once, normalize, single RNE with sticky.  Runs in the
  // store path (shared per lane), never in the per-MAC loop.
  def toF32(cont: Fp32CsaAcc): UInt = {
    val special = MuxCase(0.U(32.W), Seq(
      (cont.exc === EXC_NAN.U) -> Fp32Ops.QNAN,
      (cont.exc === EXC_INF.U) -> Fp32Ops.inf(cont.exS)))
    val tc = cont.s.asSInt +& cont.c.asSInt // sign-extended 33-bit sum
    val neg = tc < 0.S
    val mag = Mux(neg, (-tc).asUInt, tc.asUInt) // 33-bit magnitude
    val l = PriorityEncoder(Reverse(mag)) // leading zeros in 33-bit mag
    val magN = (mag << l)(W, 0) // MSB pinned at bit W
    // unbiased exponent of magN's MSB: e = p + (E - (W-1)), p = W - l
    val eb0 = cont.e +& (W.U - l).asSInt +& (127 - (W - 1)).S // biased, pre-carry
    // GRS below the 24-bit kept field (implicit 1 at bit W)
    val kept = magN(W, W - 23) // 24 bits
    val g = magN(W - 24)
    val r = magN(W - 25)
    val stLow = magN(W - 26, 0).orR || cont.st
    val up = g && (r || stLow || kept(0))
    val k = kept +& Mux(up, 1.U, 0.U) // 25 bits; bit 23 = implicit 1
    val carry = k(24)
    val eb = eb0 +& Mux(carry, 1.S, 0.S)
    // 32-bit exactly: neg | 8-bit exp | 23-bit mantissa (k's bit 23 is implicit)
    val normRes = Mux(eb >= 255.S, Fp32Ops.inf(neg),
      Cat(neg, eb(7, 0), k(22, 0)))
    // subnormal: round on the 2^-149 grid, window shifted down by s = 1-eb0
    val d2 = (1.S - eb0).asUInt // >= 1 in this branch
    val d2c = Mux(d2 > 33.U, 33.U, d2)
    val ext = magN >> d2c // bits [22:0] kept, [23]=g, [24]=r, above = st
    val keptN = ext(22, 0)
    val gs = ext(23); val rs = ext(24)
    val sts = ext(W, 25).orR || cont.st
    val upS = gs && (rs || sts || keptN(0))
    val kS = (keptN +& Mux(upS, 1.U, 0.U))(23, 0)
    // rounding can carry into the exponent LSB -> MIN_NORMAL (0x00800000)
    val subOut = Mux(kS === (BigInt(1) << 23).U,
      Cat(neg, 0.U(7.W), 1.U(1.W), 0.U(23.W)),
      Cat(neg, 0.U(8.W), kS(22, 0)))
    val rounded = Mux(mag =/= 0.U, Mux(eb0 >= 1.S, normRes,
      Mux(carry && eb0 === 0.S, normRes, subOut)), 0.U)
    Mux(cont.exc =/= EXC_NONE.U, special, rounded)
  }
}

// Bit-exact software mirror of the RTL container, for spec goldens.  S and C
// are tracked as separate W-bit two's-complement patterns exactly like the
// hardware (the re-anchor shifts them independently).
object Fp32CsaModel {
  import Fp32Csa._
  val M = BigInt(1) << W
  def mask(x: BigInt): BigInt = ((x % M) + M) % M // bit pattern in [0, 2^W)
  def sgnOf(pat: BigInt): BigInt = // two's-complement value of a pattern
    if (pat >= (M >> 1)) pat - M else pat
  def toPat(v: BigInt): BigInt = mask(v)

  case class Acc(s: BigInt, c: BigInt, e: Int, st: Boolean,
                 exc: Int, exS: Boolean, empty: Boolean)

  val cleared = Acc(0, 0, emptyE, false, EXC_NONE, false, empty = true)
  private val H = HEAD

  // fl fields: ps, aN, bN, aI, bI, aZ, bZ as Booleans; pn 0..2^48-1; px Int.
  def mac(a: Acc, ps: Boolean, aN: Boolean, bN: Boolean, aI: Boolean,
          bI: Boolean, aZ: Boolean, bZ: Boolean, pn: BigInt, px: Int): Acc = {
    val eP: Int = px + 47 + H
    val prodNan = (aI && bZ) || (bI && aZ)
    val excNan = aN || bN || prodNan
    val excInf = aI || bI
    val newExc =
      if (excNan) EXC_NAN
      else if (excInf) EXC_INF
      else a.exc
    val newExS = if (excInf && !excNan) ps else a.exS
    if (pn == 0) return a.copy(exc = newExc, exS = newExS)

    val empty = a.empty
    val move = empty || eP > a.e
    val pSgn: BigInt = if (ps) -pn else pn // 48-bit two's-complement value
    // right-shift a W-bit pattern arithmetically by d (< W); report dropped
    def asr(pat: BigInt, d: Int): (BigInt, Boolean) = {
      val dropped = pat & ((BigInt(1) << d) - 1)
      (toPat(sgnOf(pat) >> d), dropped != 0)
    }
    val (sOld, cOld, stAdd, eNew) =
      if (move) {
        val d0 = eP - a.e
        val wipe = !empty && d0 >= W
        val d = if (d0 > W) W else d0
        val (s1, ds) = if (empty) (BigInt(0), false)
                       else if (wipe) (BigInt(0), a.s != 0)
                       else asr(a.s, d)
        val (c1, dc) = if (empty) (BigInt(0), false)
                       else if (wipe) (BigInt(0), a.c != 0)
                       else asr(a.c, d)
        val stMv = ds || dc || ((pSgn & ((BigInt(1) << (48 - W + H)) - 1)) != 0)
        (s1, c1, stMv, eP)
      } else {
        val sh = a.e - eP + 47 - (W - 1)
        if (sh > 47) (a.s, a.c, pn != 0, a.e)
        else {
          val shc = math.min(sh, 63)
          val dropped = pSgn & ((BigInt(1) << shc) - 1)
          (a.s, a.c, shc != 0 && dropped != 0, a.e)
        }
      }
    val x: BigInt = if (move) mask(pSgn >> (48 - W + H)) // product bits [47:22]
                    else mask(pSgn >> math.min(a.e - eP + 47 - (W - 1), 63))
    // 3:2 CSA on patterns: s' = x ^ s ^ c (bitwise), c' = maj(x,s,c) << 1
    val xorv = mask(x ^ sOld ^ cOld)
    val maj = mask((x & sOld) | (sOld & cOld) | (cOld & x))
    val cNewV = mask(maj << 1)
    Acc(xorv, cNewV, eNew, a.st || stAdd, newExc, newExS, empty = false)
  }

  // Drain mirror of Fp32Csa.toF32 (returns fp32 bits as BigInt).
  def toF32(a: Acc): BigInt = {
    if (a.exc == EXC_NAN) return BigInt("7fc00000", 16)
    if (a.exc == EXC_INF) return if (a.exS) BigInt("ff800000", 16)
                                 else BigInt("7f800000", 16)
    val v = sgnOf(a.s) + sgnOf(a.c)
    if (v == 0) return 0
    val neg = v < 0
    val mag = v.abs
    val p = mag.bitLength - 1 // MSB position, 0..32
    val l = W - p
    val magN = mag << l // MSB at bit W
    val eb0 = a.e + p + (127 - (W - 1))
    val kept = magN >> (W - 23) // 24 bits incl. implicit
    val g = (magN >> (W - 24)) & 1
    val r = (magN >> (W - 25)) & 1
    val stLow = ((magN & ((BigInt(1) << (W - 25)) - 1)) != 0) || a.st
    val up = g == 1 && (r == 1 || stLow || (kept & 1) == 1)
    val k = kept + (if (up) 1 else 0)
    val carry = if (k >= (BigInt(1) << 24)) 1 else 0
    val eb = eb0 + carry
    val normRes =
      if (eb >= 255) if (neg) BigInt("ff800000", 16) else BigInt("7f800000", 16)
      else (if (neg) (BigInt(1) << 31) else BigInt(0)) | (BigInt(eb) << 23) |
           (k & ((BigInt(1) << 23) - 1))
    if (eb0 >= 1 || (carry == 1 && eb0 == 0)) return normRes
    // subnormal: 23-bit window shifted down by 1 - eb0
    val d2 = 1 - eb0
    val ext = if (d2 > 33) BigInt(0) else magN >> d2
    val keptN = ext & ((BigInt(1) << 23) - 1)
    val gs = (ext >> 23) & 1; val rs = (ext >> 24) & 1
    val sts = ((ext >> 25) != 0) || a.st
    val upS = gs == 1 && (rs == 1 || sts || (keptN & 1) == 1)
    val kS = keptN + (if (upS) 1 else 0) // may reach 2^23 -> MIN_NORMAL
    val subRes = if (kS == (BigInt(1) << 23)) BigInt("00800000", 16) else kS
    (if (neg) (BigInt(1) << 31) else BigInt(0)) | subRes
  }
}
