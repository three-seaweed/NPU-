// Pipelined PE datapath, throughput 1 MAC/cycle, LATENCY = 3 cycles from
// operands on io.a/io.b to the updated acc being readable.
//
//   S0  decode a/b (class flags, 24-bit significands) + Karatsuba partial
//       products z2/z0/z1 -> reg r1           (3 DSP48E2, pre-adder on z1)
//   S1  recombine the 48-bit product, pin its MSB at bit 47, exponent sum
//       -> reg r2                             (fabric adds only)
//   S2  exact accumulate against the LIVE acc register and write it back:
//       fused ? fmaBack(r2, acc) : add(r2.prodF32, acc)
//
// The acc feedback loop wraps ONLY S2.  MAC k occupies S2 at cycle
// (k + skew) + 2 and reads the acc written by MAC k-1 at cycle
// (k-1 + skew) + 2 -> exactly one cycle earlier, so back-to-back beats at
// 1/cycle never read a stale accumulator: strict k-order accumulation and
// full throughput are preserved by construction.
//
// Bubble beats (operands = 0, fed by the array when inReady gates the feed)
// compute fma(0,0,acc) = acc / add(0,acc) = acc, so idle cycles preserve the
// accumulator exactly like the old combinational PE did.

package fp32matmul

import chisel3._
import chisel3.util._

object Fp32FmaPipe {
  val LATENCY = 3 // cycles: operand on input -> updated acc readable
}

class Pipe1 extends Bundle {
  val fl = new Fp32Fl
  val z2 = UInt(24.W)
  val z0 = UInt(24.W)
  val z1 = UInt(26.W)
  val eab = SInt(10.W) // expf(a) + expf(b), in [2, 508]
}

class Pipe2 extends Bundle {
  val fl = new Fp32Fl
  val pn = UInt(48.W) // product MSB pinned at bit 47, value = pn * 2^px
  val px = SInt(12.W)
  val prodF32 = UInt(32.W) // double-rounding variant only (0 otherwise)
}

class Fp32FmaPipe(val fused: Boolean, val narrow: Boolean = false) extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Input(UInt(32.W))
    val clear = Input(Bool())
    val acc = Output(UInt(32.W))
  })
  val acc = RegInit(0.U(32.W)) // +0

  // ---- S0: decode + Karatsuba partial products
  val fl0 = Wire(new Fp32Fl)
  fl0.ps := Fp32Ops.sgn(io.a) ^ Fp32Ops.sgn(io.b)
  fl0.aN := Fp32Ops.isNan(io.a); fl0.bN := Fp32Ops.isNan(io.b)
  fl0.aI := Fp32Ops.isInf(io.a); fl0.bI := Fp32Ops.isInf(io.b)
  fl0.aZ := Fp32Ops.isZeroCls(io.a); fl0.bZ := Fp32Ops.isZeroCls(io.b)
  val am = Mux(fl0.aZ, 0.U, Cat(1.U(1.W), Fp32Ops.mant(io.a)))
  val bm = Mux(fl0.bZ, 0.U, Cat(1.U(1.W), Fp32Ops.mant(io.b)))
  val aH = am(23, 12); val aL = am(11, 0)
  val bH = bm(23, 12); val bL = bm(11, 0)
  val r1 = RegInit(0.U.asTypeOf(new Pipe1))
  r1.fl := fl0
  r1.z2 := aH * bH // 12x12 -> 1 DSP
  r1.z0 := aL * bL // 12x12 -> 1 DSP
  r1.z1 := (aL +& aH) * (bL +& bH) // 13x13 -> 1 DSP (pre-add)
  r1.eab := Fp32Ops.expf(io.a).zext +& Fp32Ops.expf(io.b).zext

  // ---- S1: recombine, normalize, exponent path
  // truncating + is safe here: max value (2^24-1)^2 < 2^48
  val prod = (r1.z2 << 24) + ((r1.z1 - (r1.z0 +& r1.z2)) << 12) + r1.z0
  val pn1 = Mux(prod(47), prod, prod << 1)
  val px1 = r1.eab -& 300.S -& Mux(prod(47), 0.S, 1.S)
  val r2 = RegInit(0.U.asTypeOf(new Pipe2))
  r2.fl := r1.fl
  r2.pn := pn1
  r2.px := px1
  r2.prodF32 := (if (fused) 0.U
                 else if (narrow) Fp32Narrow.mulBackN(r1.fl, pn1, px1)
                 else Fp32Ops.mulBack(r1.fl, pn1, px1))

  // ---- S2: the 1-cycle acc loop (exact 512-bit, or narrow GRS+sticky)
  val res = if (narrow) {
    if (fused) Fp32Narrow.fmaBackN(r2.fl, r2.pn, r2.px, acc)
    else Fp32Narrow.addN(r2.prodF32, acc)
  } else {
    if (fused) Fp32Ops.fmaBack(r2.fl, r2.pn, r2.px, acc)
    else Fp32Ops.add(r2.prodF32, acc)
  }
  acc := Mux(io.clear, 0.U, res)
  io.acc := acc
}
