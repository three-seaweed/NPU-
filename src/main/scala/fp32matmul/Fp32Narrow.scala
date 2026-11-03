// Narrow (GRS + sticky) back end, P3 area-diet version.  Bit-exact against
// Fp32Model; the coverage argument is in 窄容器设计报告.md.  Structure is the
// P2 container with ONE barrel shifter instead of two: the bigger operand is
// wired into the 52-bit container exactly, only the smaller is shifted (with
// sticky).  The LZC + normalize barrel + round tail are shared by all cases.
//
// Zero-class addends route through sum() as (cm=0, eC=otherSide-1); the LZC
// then reproduces round48 (product alone) bit-exactly, so fmaBackN carries
// no parallel rounder.

package fp32matmul

import chisel3._
import chisel3.util._

object Fp32Narrow {

  // 51-bit right shift with dropped bits folded into sticky (far path).
  // dSm >= 28 is the only regime that drops anything, and the dropped value
  // is then strictly below one unit of container bit 0, which is exactly
  // what the sticky-borrow rule needs to stay exact.
  private def shrSticky(v: UInt, d: UInt): (UInt, Bool) = {
    val ds = Mux(d > 51.U, 51.U, d)
    (v >> ds, (v & ((1.U(52.W) << ds) - 1.U)) =/= 0.U)
  }

  // shared rounding tails (kept field is normalized, MSB = implicit 1).
  // stLow = exact low bits of the normalized sum: they sit strictly above the
  // tie in BOTH directions, so they always vote up.  stDrop = residue the far
  // path pushed below the container (shrSticky): it carries the SMALLER
  // operand's sign, so at an exact-tie tail (r=0, stLow=0) it moves the exact
  // value off the tie - up when the addend signs agree, DOWN when they differ,
  // and in the subtraction case round-to-even must be suppressed too
  // (docs/Fp32_NarrowFma_Erratum.md).
  private def tailNormal(s: Bool, kept: UInt, g: Bool, r: Bool, stLow: Bool, stDrop: Bool, sub: Bool, e0: SInt): UInt = {
    val up = g && (r || stLow || (!sub && stDrop) || (kept(0) && !(sub && stDrop)))
    val k = kept +& Mux(up, 1.U, 0.U) // 25 bits
    val carry = k(24)
    val e1 = e0 +& Mux(carry, 1.S, 0.S)
    Mux(e1 >= 255.S, Fp32Ops.inf(s),
      Cat(s, e1(7, 0), Mux(carry, 0.U(23.W), k(22, 0))))
  }

  private def tailSub(s: Bool, keptN: UInt, g: Bool, r: Bool, stLow: Bool, stDrop: Bool, sub: Bool): UInt = {
    val up = g && (r || stLow || (!sub && stDrop) || (keptN(0) && !(sub && stDrop)))
    val k = (keptN +& Mux(up, 1.U, 0.U))(23, 0)
    Mux(k === (1 << 23).U, "h00800000".U, 0.U)
  }

  def sum(ps: Bool, p48: UInt, eP: SInt, cs: Bool, c24: UInt, eC: SInt): UInt = {
    val ebig = Mux(eP >= eC, eP, eC)
    val dP = (ebig -& eP).asUInt
    val dC = (ebig -& eC).asUInt
    val sameS = ps === cs

    // Unified 52-bit container, bit i <-> weight ebig-50+i.  The BIGGER
    // operand is wired in exactly (no shifter): p48 << 3 has its MSB at bit
    // 50 = weight eP; c24 << 27 likewise.  Only the SMALLER goes through one
    // 51-bit barrel + sticky (dSm >= 28 is the only dropping regime and the
    // dropped residue is < 1 unit of bit 0).  The normalize barrel then
    // re-anchors the sum at bit 51, which parks any sub-bit-0 truncation
    // residue in the sticky zone of the round decision - the property that
    // makes the plain GRS rule exact (same argument as the P2 container).
    val pIsBig = eP >= eC
    val bigV = Mux(pIsBig, p48 << 3, (c24 << 27).pad(52))
    val smW = Mux(pIsBig, c24 << 27, p48 << 3)
    val dSm = Mux(pIsBig, dC, dP)
    val (sv, st) = shrSticky(smW, dSm)
    val pv = bigV.pad(52); val cv = sv.pad(52)
    val mag = Mux(sameS, (pv +& cv)(51, 0),
      Mux(pv >= cv, (pv - cv)(51, 0), (cv - pv)(51, 0)))
    // pv carries whichever operand has the bigger EXPONENT, so the sign of
    // the winning side follows the bus, not the ps/cs parameter positions
    val sPv = Mux(pIsBig, ps, cs)
    val sCv = Mux(pIsBig, cs, ps)
    val signOut = Mux(sameS, ps, Mux(pv >= cv, sPv, sCv))
    val l = PriorityEncoder(Reverse(mag)) // pins MSB at bit 51
    val magN = (mag << l)(51, 0)
    val e0 = ebig -& 50.S +& (51.S -& l.zext) +& 127.S
    val subRes = { // e0 <= 0: round on the 2^-149 grid, only MIN_NORMAL survives
      val keptN = magN(51, 29) // 23 bits, LSB weight = msbW - 22 = -149
      tailSub(signOut, keptN, magN(28), magN(27), magN(26, 0).orR, st, !sameS)
    }
    val normRes = {
      val kept = magN(51, 28) // 24 bits incl. implicit 1
      tailNormal(signOut, kept, magN(27), magN(26), magN(25, 0).orR, st, !sameS, e0)
    }
    Mux(mag === 0.U, 0.U,
      Mux(e0 >= 255.S, Fp32Ops.inf(signOut),
        Mux(e0 <= 0.S, subRes, normRes)))
  }

  // Product-alone rounding (mul tail only; the fused PE elides this path).
  def round48(s: Bool, pn: UInt, eM: SInt): UInt = {
    val mag = Cat(pn, 0.U(3.W)) // MSB pinned at bit 50
    val e0 = eM +& 127.S
    val subRes = {
      val keptN = mag(50, 28); val g = mag(27); val rr = mag(26)
      tailSub(s, keptN, g, rr, mag(25, 0).orR, false.B, false.B)
    }
    val normRes = {
      val kept = mag(50, 27); val g = mag(26); val rr = mag(25)
      tailNormal(s, kept, g, rr, mag(24, 0).orR, false.B, false.B, e0)
    }
    Mux(e0 >= 255.S, Fp32Ops.inf(s), Mux(e0 <= 0.S, subRes, normRes))
  }

  def mulBackN(fl: Fp32Fl, pn: UInt, px: SInt): UInt = {
    val finite = Mux(pn === 0.U, 0.U, round48(fl.ps, pn, px +& 47.S))
    MuxCase(finite, Seq(
      (fl.aN || fl.bN || (fl.aI && fl.bZ) || (fl.bI && fl.aZ)) -> Fp32Ops.QNAN,
      (fl.aI || fl.bI) -> Fp32Ops.inf(fl.ps),
    ))
  }

  def fmaBackN(fl: Fp32Fl, pn: UInt, px: SInt, c: UInt): UInt = {
    val cN = Fp32Ops.isNan(c); val cI = Fp32Ops.isInf(c); val cZ = Fp32Ops.isZeroCls(c)
    val cword = Mux(cZ, 0.U, c)
    // zero-class addend routes through sum() one exponent below the product:
    // forced close path, cm = 0, which reproduces round48(product) bitwise
    val eP = px +& 47.S
    val csX = Mux(cZ, fl.ps, Fp32Ops.sgn(cword))
    val cmX = Mux(cZ, 0.U, Cat(1.U(1.W), Fp32Ops.mant(cword)))
    val eCX = Mux(cZ, eP -& 1.S, Fp32Ops.expf(cword).zext -& 127.S)
    val finite = Mux(pn === 0.U, cword, sum(fl.ps, pn, eP, csX, cmX, eCX))
    val prodNan = (fl.aI && fl.bZ) || (fl.bI && fl.aZ)
    MuxCase(finite, Seq(
      (fl.aN || fl.bN || cN || prodNan) -> Fp32Ops.QNAN,
      (fl.aI || fl.bI) -> Mux(cI, Mux(Fp32Ops.sgn(c) === fl.ps, Fp32Ops.inf(fl.ps), Fp32Ops.QNAN), Fp32Ops.inf(fl.ps)),
      cI -> Fp32Ops.inf(Fp32Ops.sgn(c)),
    ))
  }

  // add tail for the double-rounding S2; zero-class operands also route
  // through sum() as (mag 0, exponent one below the other side).
  def addN(a0: UInt, b0: UInt): UInt = {
    val aN = Fp32Ops.isNan(a0); val bN = Fp32Ops.isNan(b0)
    val aI = Fp32Ops.isInf(a0); val bI = Fp32Ops.isInf(b0)
    val aZ = Fp32Ops.isZeroCls(a0); val bZ = Fp32Ops.isZeroCls(b0)
    val am = Mux(aZ, 0.U, Cat(1.U(1.W), Fp32Ops.mant(a0)))
    val bm = Mux(bZ, 0.U, Cat(1.U(1.W), Fp32Ops.mant(b0)))
    val eA0 = Fp32Ops.expf(a0).zext -& 127.S
    val eB0 = Fp32Ops.expf(b0).zext -& 127.S
    val eA = Mux(aZ, eB0 -& 1.S, eA0)
    val eB = Mux(bZ, eA0 -& 1.S, eB0)
    val core = sum(Fp32Ops.sgn(a0), am << 24, eA, Fp32Ops.sgn(b0), bm, eB)
    val finite = Mux(aZ && bZ, 0.U, core) // both zero: mag 0 -> +0 anyway
    MuxCase(finite, Seq(
      (aN || bN) -> Fp32Ops.QNAN,
      (aI && bI) -> Mux(Fp32Ops.sgn(a0) === Fp32Ops.sgn(b0), Fp32Ops.inf(Fp32Ops.sgn(a0)), Fp32Ops.QNAN),
      aI -> Fp32Ops.inf(Fp32Ops.sgn(a0)),
      bI -> Fp32Ops.inf(Fp32Ops.sgn(b0)),
    ))
  }
}

// Combinational test harness for the narrow back end.
class Fp32NarrowHarness extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Input(UInt(32.W))
    val c = Input(UInt(32.W))
    val fma = Output(UInt(32.W))
    val mul = Output(UInt(32.W))
    val add = Output(UInt(32.W))
  })
  private def prodOf(a: UInt, b: UInt): (Fp32Fl, UInt, SInt) = {
    val (am, bm, fl) = Fp32Ops.front(a, b)
    val prod = Fp32Ops.mul24(am, bm)
    val pn = Mux(prod(47), prod, prod << 1)
    val px = Fp32Ops.expf(a).zext +& Fp32Ops.expf(b).zext -& 300.S -& Mux(prod(47), 0.S, 1.S)
    (fl, pn, px)
  }
  locally {
    val (fl, pn, px) = prodOf(io.a, io.b)
    io.fma := Fp32Narrow.fmaBackN(fl, pn, px, io.c)
    io.mul := Fp32Narrow.mulBackN(fl, pn, px)
  }
  io.add := Fp32Narrow.addN(io.a, io.b)
}
