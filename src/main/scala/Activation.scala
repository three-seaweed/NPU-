// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


// Transcendental / activation primitives for the vector core: exp, rsqrt,
// reciprocal on fp32. All hand-rolled (no hardfloat):
//   * exp  : range-reduce to 2^(x*log2e), integer part -> fp32 exponent, fractional
//            part from a 64-entry ROM of 2^f with linear interpolation. The ROM is
//            computed in Scala at elaboration time, so no runtime float is needed.
//   * rsqrt: 0x5f3759df bit-trick seed + one Newton step.
//   * recip: 0x7EEEEEEE bit-trick seed + two Newton steps.
// sigmoid = 1/(1+exp(-x)) is composed from recip+exp in vector-assembly, so it
// needs no dedicated hardware here.
//
// Accuracy target is bf16-level (~2^-7 relative); these approximations clear it
// comfortably. Pure combinational; the caller registers the result.
// Carried between the two halves of `exp`. Splitting there puts one multiplier
// in each half: stage 1 ends with x*log2e, stage 2 starts with the 2^f table.
class ExpMid extends Bundle {
    // x * log2e in Q16.16. 25-bit magnitude x 18-bit constant = 43 bits, minus the
    // 16-bit rescale, so exactly 27 bits.
    val tQ       = SInt(27.W)
    val sign     = Bool()
    val isNaN    = Bool()
    val isInf    = Bool()
    val overflow = Bool()       // |x| >= 128, saturates without consulting the table
}

object Activation {
    private val ExpN = 64

    // ---- exp(x), split into two register-separated halves ---------------------
    // Range reduction. Contains a variable right shift (up to 134) and the
    // 25x18 x*log2e multiply -- one long operation, and nothing after it.
    def expStage1(x: UInt): ExpMid = {
        val sign = x(31)
        val ex   = x(30, 23)
        val mant = x(22, 0)
        val sig24 = Cat(1.U(1.W), mant)                       // 24b significand

        // fp32 -> Q16.16 magnitude: value*2^16 = sig24 >> (134 - ex), for ex <= 133.
        // |x| >= 128 (ex >= 134) over/underflows, so the shift is short-circuited.
        val rsh = Mux(ex >= 134.U, 0.U, 134.U - ex)
        val fixedMag = (sig24 >> rsh)(23, 0)                  // |x| in Q16.16, < 2^23
        val fixedSigned = Mux(sign, -(fixedMag.zext), fixedMag.zext)   // SInt

        // t = x * log2e in Q16.16. log2e ~= 94548 / 2^16.
        val tFull = fixedSigned * 94548.S                     // Q32.32 (SInt)

        val m = Wire(new ExpMid)
        m.tQ       := tFull >> 16                             // Q16.16
        m.sign     := sign
        m.isNaN    := ex === 0xFF.U && mant =/= 0.U
        m.isInf    := ex === 0xFF.U && mant === 0.U
        m.overflow := ex >= 134.U
        m
    }

    // 2^f from a 64-entry LUT with linear interpolation, then reassembly. Contains
    // the table lookup and the 25x10 interpolation multiply.
    def expStage2(m: ExpMid): UInt = {
        val k = (m.tQ >> 16).asSInt                           // integer part (signed)
        val f = m.tQ.asUInt(15, 0)                            // fraction [0,1)

        val table = VecInit((0 to ExpN).map { i =>
            BigInt(Math.round((Math.pow(2.0, i.toDouble / ExpN) - 1.0) * (1 << 23))).U(24.W)
        })
        val idx  = Cat(0.U(1.W), f(15, 10))                  // 7b 0..63 (room for idx+1)
        val frac = f(9, 0)                                    // 10-bit interpolation weight
        val s0 = (BigInt(1 << 23).U(25.W)) + table(idx)
        val s1 = (BigInt(1 << 23).U(25.W)) + table(idx + 1.U)
        val slope = s1 - s0                                   // >= 0
        val sigInterp = (s0 + ((slope * frac) >> 10))(24, 0)  // 2^f * 2^23, in [2^23, 2^24]

        // Normalise a boundary overflow to [1,2).
        val sigOvf = sigInterp(24)
        val sigN  = Mux(sigOvf, (sigInterp >> 1)(23, 0), sigInterp(23, 0))
        val kAdj  = k + Mux(sigOvf, 1.S, 0.S)

        val eRes = 127.S + kAdj
        val normal = Cat(0.U(1.W), eRes(7, 0), sigN(22, 0))

        val inf  = Cat(0.U(1.W), 0xFF.U(8.W), 0.U(23.W))
        MuxCase(normal, Seq(
            m.isNaN                       -> Cat(0.U(1.W), 0xFF.U(8.W), (1 << 22).U(23.W)),
            (m.isInf && !m.sign)          -> inf,
            (m.isInf && m.sign)           -> 0.U(32.W),
            (m.overflow && !m.sign)       -> inf,
            (m.overflow && m.sign)        -> 0.U(32.W),
            (eRes >= 0xFF.S)              -> inf,
            (eRes <= 0.S)                 -> 0.U(32.W),
        ))
    }

    // Combinational exp, both halves in one cycle. Kept for ActivationSpec and for
    // any caller that can afford the path; VectorUnit uses the split form.
    def exp(x: UInt): UInt = expStage2(expStage1(x))

    // ---- rsqrt(x) : 1/sqrt(x), one Newton step -------------------------------
    def rsqrt(x: UInt): UInt = {
        val sign = x(31)
        val ex   = x(30, 23)
        val isZero = ex === 0.U
        val xabs = Cat(0.U(1.W), x(30, 0))
        // 0.5*x by decrementing the exponent.
        val xhalf = Cat(0.U(1.W), ex - 1.U, x(22, 0))
        val seed = "h5f3759df".U(32.W) - (xabs >> 1)
        val y0   = seed
        val y0sq = Bf16.mulF32(y0, y0)
        val t    = Bf16.mulF32(xhalf, y0sq)                   // 0.5*x*y0^2
        val onePointFive = "h3fc00000".U(32.W)
        val t2   = Bf16.addF32(onePointFive, Bf16.negF32(t))  // 1.5 - t
        val y    = Bf16.mulF32(y0, t2)

        val inf  = Cat(0.U(1.W), 0xFF.U(8.W), 0.U(23.W))
        val qnan = Cat(0.U(1.W), 0xFF.U(8.W), (1 << 22).U(23.W))
        MuxCase(y, Seq(
            isZero -> inf,        // rsqrt(0) = +Inf
            sign   -> qnan,       // rsqrt(negative) = NaN
        ))
    }

    // ---- recip(x) : 1/x, two Newton steps ------------------------------------
    def recip(x: UInt): UInt = {
        val sign = x(31)
        val ex   = x(30, 23)
        val mant = x(22, 0)
        val isZero = ex === 0.U
        val isInf  = ex === 0xFF.U && mant === 0.U
        val xabs = Cat(0.U(1.W), x(30, 0))
        val seed = "h7eeeeeee".U(32.W) - xabs
        val y0   = seed
        val two  = "h40000000".U(32.W)
        def step(y: UInt): UInt = {
            val p = Bf16.mulF32(xabs, y)
            val c = Bf16.addF32(two, Bf16.negF32(p))          // 2 - x*y
            Bf16.mulF32(y, c)
        }
        val y2 = step(step(y0))
        val res = Cat(sign, y2(30, 0))                        // restore sign

        val infS = Cat(sign, 0xFF.U(8.W), 0.U(23.W))
        MuxCase(res, Seq(
            isZero -> infS,       // 1/0 = signed Inf
            isInf  -> Cat(sign, 0.U(31.W)),  // 1/Inf = signed 0
        ))
    }
}
