// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


// Hand-rolled bf16 / fp32 arithmetic for the vector core. No Berkeley hardfloat
// is available, so every primitive is built from integer add/shift/compare/mul.
//
// Formats:
//   bf16 : sign[15] exp[14:7] (8b, bias 127) mant[6:0]  -- same exponent field
//          as fp32, so widening is a zero-pad and narrowing is a rounded truncate.
//   fp32 : sign[31] exp[30:23] (8b, bias 127) mant[22:0]
//
// Conventions (kept deliberately simple):
//   * Flush-to-zero (FTZ): any operand with exp==0 is treated as a signed zero;
//     results that underflow (exp <= 0) are flushed to signed zero. No subnormals.
//   * exp==0xFF is Inf (mant==0) or NaN (mant!=0). NaN propagates as a canonical
//     qNaN; overflow saturates to Inf.
//   * Rounding is round-to-nearest-even (RNE), applied at fp32->bf16 narrowing and
//     inside fp32 add. bf16*bf16 products are exact in fp32 (8x8-bit significands).
//
// All functions are pure combinational (UInt in / UInt out); sequencing and any
// pipelining are the caller's responsibility.
object Bf16 {
    // Index of the most-significant set bit (floor(log2)); 0 if x==0.
    private def msbIndex(x: UInt): UInt = {
        val w = x.getWidth
        (w - 1).U - PriorityEncoder(Reverse(x))
    }

    // ---- bf16 <-> fp32 -------------------------------------------------------
    // Widen bf16 to fp32: pad the mantissa with 16 zero bits. FTZ on exp==0.
    def toF32(b: UInt): UInt = {
        val sign = b(15)
        val exp  = b(14, 7)
        val mant = b(6, 0)
        val zero = exp === 0.U
        Mux(zero, Cat(sign, 0.U(31.W)), Cat(sign, exp, mant, 0.U(16.W)))
    }

    // Narrow fp32 to bf16 with round-to-nearest-even. Saturates to Inf on
    // overflow, flushes denormals/underflow to signed zero, canonicalises NaN.
    def f32ToBf16(f: UInt): UInt = {
        val sign = f(31)
        val exp  = f(30, 23)
        val mant = f(22, 0)
        val isInf = exp === 0xFF.U && mant === 0.U
        val isNaN = exp === 0xFF.U && mant =/= 0.U
        val isZ   = exp === 0.U

        val keep    = f(22, 16)                                  // top 7 mantissa bits
        val roundUp = f(15) && (f(14, 0) =/= 0.U || f(16))       // RNE
        val combined = Cat(exp, keep)                            // 15b: [14:7]=exp,[6:0]=mant
        val rounded  = combined +& roundUp                       // 16b (carry out of exp)
        val rExp = rounded(14, 7)
        val rMant = rounded(6, 0)
        val overflow = rounded(15) || rExp === 0xFF.U

        val qnan   = Cat(sign, 0xFF.U(8.W), "b1000000".U(7.W))
        val inf    = Cat(sign, 0xFF.U(8.W), 0.U(7.W))
        val szero  = Cat(sign, 0.U(15.W))
        val normal = Cat(sign, rExp, rMant)
        MuxCase(normal, Seq(
            isNaN    -> qnan,
            isInf    -> inf,
            isZ      -> szero,
            overflow -> inf,
        ))
    }

    // ---- int32 -> fp32 (RNE) -------------------------------------------------
    // Signed two's-complement int32 to fp32: take magnitude, leading-zero
    // normalise so the top set bit lands at the fp32 implicit-1 slot (bit 23),
    // then RNE-round the low bits. |value| up to 2^31 needs up to 31 significant
    // bits, so wide values round. The int32 accumulator feeds this on dequant.
    def i32ToF32(x: UInt): UInt = {
        val sign = x(31)
        // Absolute value. |-2^31| = 2^31 fits in 32 unsigned bits.
        val mag  = Mux(sign, (~x) + 1.U, x)(31, 0)
        val isZ  = mag === 0.U
        val lead = msbIndex(mag)                              // 0..31, MSB position

        // lead <= 23: left-justify into the 24-bit significand, exact (no round).
        val leftAmt  = Mux(lead <= 23.U, 23.U - lead, 0.U)
        val sigL     = (mag << leftAmt)(23, 0)               // implicit 1 at bit 23
        val mantL    = sigL(22, 0)

        // lead > 23: shift right by r, keeping guard/round/sticky for RNE.
        val r        = Mux(lead > 23.U, lead - 23.U, 0.U)    // 1..8
        val sigR     = (mag >> r)(23, 0)                     // includes implicit 1
        val guard    = (mag >> (r - 1.U))(0)                 // bit r-1
        val stickyM  = (1.U(33.W) << (r - 1.U)) - 1.U
        val sticky   = (mag & stickyM(31, 0)).orR
        val roundUp  = guard && (sticky || sigR(0))
        val rounded  = sigR +& roundUp                        // 25 bits, may carry
        val carry    = rounded(24)
        val mantR    = Mux(carry, 0.U(23.W), rounded(22, 0))  // carry -> 2.0, mant=0

        val useRight = lead > 23.U
        val mant     = Mux(useRight, mantR, mantL)
        val eBiased  = 127.U(9.W) + lead + Mux(useRight && carry, 1.U, 0.U)
        val normal   = Cat(sign, eBiased(7, 0), mant)
        Mux(isZ, Cat(sign, 0.U(31.W)), normal)
    }

    // ---- bf16 * bf16 -> fp32 (exact) -----------------------------------------
    // 8-bit significands (implicit 1 + 7 mantissa) multiply to 16 bits, which the
    // fp32 mantissa holds exactly, so no rounding is needed.
    def mulExact(a: UInt, b: UInt): UInt = {
        val sA = a(15); val eA = a(14, 7); val mA = a(6, 0)
        val sB = b(15); val eB = b(14, 7); val mB = b(6, 0)
        val sign = sA ^ sB

        val aNaN = eA === 0xFF.U && mA =/= 0.U
        val bNaN = eB === 0xFF.U && mB =/= 0.U
        val aInf = eA === 0xFF.U && mA === 0.U
        val bInf = eB === 0xFF.U && mB === 0.U
        val aZ   = eA === 0.U
        val bZ   = eB === 0.U

        val sigA = Cat(1.U(1.W), mA)                 // 8 bits
        val sigB = Cat(1.U(1.W), mB)                 // 8 bits
        val prod = sigA * sigB                       // 16 bits, in [2^14, 2^16)
        val msb  = prod(15)                          // set when product >= 2^15 ([2,4))
        val adjust = Mux(msb, 1.S, 0.S)
        // fraction bits left-justified into the 23-bit fp32 mantissa.
        val mant23 = Mux(msb, Cat(prod(14, 0), 0.U(8.W)),   // 15 frac bits << 8
                              Cat(prod(13, 0), 0.U(9.W)))   // 14 frac bits << 9
        val e = (eA +& eB).zext - 127.S + adjust     // SInt

        val qnan = Cat(sign, 0xFF.U(8.W), (1 << 22).U(23.W))
        val inf  = Cat(sign, 0xFF.U(8.W), 0.U(23.W))
        val szero = Cat(sign, 0.U(31.W))
        val normal = Cat(sign, e(7, 0), mant23)

        MuxCase(normal, Seq(
            (aNaN || bNaN)         -> qnan,
            (aInf && bZ)           -> qnan,        // inf * 0
            (bInf && aZ)           -> qnan,
            (aInf || bInf)         -> inf,
            (aZ || bZ)             -> szero,
            (e >= 0xFF.S)          -> inf,
            (e <= 0.S)             -> szero,        // underflow -> FTZ
        ))
    }

    // ---- fp32 + fp32 -> fp32 (RNE) -------------------------------------------
    // The hardest primitive: exponent align with guard/round/sticky, magnitude
    // add or subtract, leading-zero normalise, RNE. FTZ on underflow.
    def addF32(x: UInt, y: UInt): UInt = {
        def parts(v: UInt) = {
            val s = v(31); val e = v(30, 23); val m = v(22, 0)
            val isZ = e === 0.U
            val sig = Mux(isZ, 0.U(24.W), Cat(1.U(1.W), m))   // 24b significand, FTZ
            (s, e, sig, isZ)
        }
        val (sx, ex, sigx, zx) = parts(x)
        val (sy, ey, sigy, zy) = parts(y)

        val xNaN = ex === 0xFF.U && x(22, 0) =/= 0.U
        val yNaN = ey === 0xFF.U && y(22, 0) =/= 0.U
        val xInf = ex === 0xFF.U && x(22, 0) === 0.U
        val yInf = ey === 0xFF.U && y(22, 0) === 0.U

        // Order operands by magnitude (exp, then significand).
        val xBigger = (ex > ey) || (ex === ey && sigx >= sigy)
        val sBig = Mux(xBigger, sx, sy);   val sSml = Mux(xBigger, sy, sx)
        val eBig = Mux(xBigger, ex, ey)
        val eSml = Mux(xBigger, ey, ex)
        val sigBig = Mux(xBigger, sigx, sigy)
        val sigSml = Mux(xBigger, sigy, sigx)

        val diff = eBig - eSml
        val shiftAmt = Mux(diff > 31.U, 31.U, diff)

        // Align the smaller significand right with a sticky bit. 3 guard bits.
        val sigSmlG = Cat(sigSml, 0.U(3.W))                   // 27 bits
        val shifted = sigSmlG >> shiftAmt
        val lostMask = (1.U(33.W) << shiftAmt) - 1.U
        val sticky = (sigSmlG & lostMask(26, 0)).orR
        val sigSmlAligned = shifted | sticky                  // 27 bits

        val sigBigG = Cat(sigBig, 0.U(3.W))                   // 27 bits
        val sameSign = sBig === sSml
        // 28-bit magnitude result (carry on add).
        val sumMag = Mux(sameSign, sigBigG +& sigSmlAligned,
                                   sigBigG -& sigSmlAligned)(27, 0)

        val resSign = sBig
        val isZeroResult = sumMag === 0.U

        // Normalise so the leading 1 lands at bit 26 (sigBigG's implicit-1 slot).
        val lead = msbIndex(sumMag)                           // 0..27
        val rightShift = lead > 26.U                          // carry case (lead 27)
        val rNorm = sumMag >> 1
        val rSticky = sumMag(0)
        val lNorm = (sumMag << (26.U - lead))(27, 0)
        val norm = Mux(rightShift, Cat(0.U(1.W), rNorm(26, 0)), lNorm)
        val expDelta = lead.zext - 26.S            // eRes = eBig + (lead - 26)

        // norm: [26]=leading 1, [25:3]=mantissa, [2]=guard,[1]=round,[0]=sticky.
        val mant23 = norm(25, 3)
        val guard  = norm(2)
        val round  = norm(1)
        val stky   = norm(0) | (rightShift && rSticky)
        val roundUp = guard && (round || stky || mant23(0))
        val mantR = mant23 +& roundUp                         // 24 bits (carry into implicit)
        val mantCarry = mantR(23)
        val finalMant = Mux(mantCarry, 0.U(23.W), mantR(22, 0))
        val eRes = eBig.zext + expDelta + Mux(mantCarry, 1.S, 0.S)

        val overflow = eRes >= 0xFF.S
        val underflow = eRes <= 0.S

        val qnan = Cat(0.U(1.W), 0xFF.U(8.W), (1 << 22).U(23.W))
        val infP = Cat(resSign, 0xFF.U(8.W), 0.U(23.W))
        val szero = Cat(resSign, 0.U(31.W))
        val normal = Cat(resSign, eRes(7, 0), finalMant)

        MuxCase(normal, Seq(
            (xNaN || yNaN)              -> qnan,
            (xInf && yInf && sx =/= sy) -> qnan,    // inf - inf
            xInf                        -> Cat(sx, 0xFF.U(8.W), 0.U(23.W)),
            yInf                        -> Cat(sy, 0xFF.U(8.W), 0.U(23.W)),
            // Under round-to-nearest-even, exact cancellation of finite values
            // is canonical +0 regardless of which operand supplied sBig.
            isZeroResult                -> 0.U(32.W),
            overflow                    -> infP,
            underflow                   -> szero,
        ))
    }

    // ---- fp32 * fp32 -> fp32 (RNE) -------------------------------------------
    // Needed by the activation unit's Newton iterations. 24x24 significand
    // product -> 48 bits, normalised and RNE-rounded to 23 mantissa bits.
    def mulF32(a: UInt, b: UInt): UInt = {
        val sA = a(31); val eA = a(30, 23); val mA = a(22, 0)
        val sB = b(31); val eB = b(30, 23); val mB = b(22, 0)
        val sign = sA ^ sB
        val aNaN = eA === 0xFF.U && mA =/= 0.U
        val bNaN = eB === 0xFF.U && mB =/= 0.U
        val aInf = eA === 0xFF.U && mA === 0.U
        val bInf = eB === 0xFF.U && mB === 0.U
        val aZ   = eA === 0.U
        val bZ   = eB === 0.U

        val sigA = Cat(1.U(1.W), mA)                 // 24 bits
        val sigB = Cat(1.U(1.W), mB)                 // 24 bits
        val prod = sigA * sigB                       // 48 bits, in [2^46, 2^48)
        val msb  = prod(47)
        val adjust = Mux(msb, 1.S, 0.S)
        val mant    = Mux(msb, prod(46, 24), prod(45, 23))      // 23 bits
        val guard   = Mux(msb, prod(23), prod(22))
        val lower   = Mux(msb, prod(22, 0), Cat(0.U(1.W), prod(21, 0)))
        val roundUp = guard && (lower.orR || mant(0))
        val mantR   = mant +& roundUp                           // 24 bits
        val carry   = mantR(23)
        val finalMant = Mux(carry, 0.U(23.W), mantR(22, 0))
        val e = (eA +& eB).zext - 127.S + adjust + Mux(carry, 1.S, 0.S)

        val qnan = Cat(sign, 0xFF.U(8.W), (1 << 22).U(23.W))
        val inf  = Cat(sign, 0xFF.U(8.W), 0.U(23.W))
        val szero = Cat(sign, 0.U(31.W))
        val normal = Cat(sign, e(7, 0), finalMant)
        MuxCase(normal, Seq(
            (aNaN || bNaN)       -> qnan,
            (aInf && bZ)         -> qnan,
            (bInf && aZ)         -> qnan,
            (aInf || bInf)       -> inf,
            (aZ || bZ)           -> szero,
            (e >= 0xFF.S)        -> inf,
            (e <= 0.S)           -> szero,
        ))
    }

    // Flip the sign bit of an fp32 value (used to form a - b via add).
    def negF32(a: UInt): UInt = Cat(~a(31), a(30, 0))

    // ---- Fused multiply-add: bf16*bf16 + fp32 acc -> fp32 --------------------
    def fmaF32(a: UInt, b: UInt, acc: UInt): UInt = addF32(mulExact(a, b), acc)

    // Magnitude comparison helpers on fp32 (sign-magnitude), used for VMAX.
    def gtF32(a: UInt, b: UInt): Bool = {
        val sa = a(31); val sb = b(31)
        val ma = a(30, 0); val mb = b(30, 0)
        Mux(sa =/= sb, sb,                              // positive > negative
            Mux(sa, ma < mb, ma > mb))                 // both neg: smaller mag is larger
    }
}
