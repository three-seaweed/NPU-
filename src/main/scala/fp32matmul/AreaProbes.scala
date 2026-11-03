// Synthesis-only area probes: each module isolates ONE combinational stage /
// one FSM path of the bf16 16x16 tier so Vivado reports its LUT cost alone.
// Inputs are driven from free-running registers so no logic constant-folds.
// Not testbenches - elaboration + report_utilization only.

package fp32matmul

import chisel3._
import chisel3.util._
import xnpu.{Parameters, Bf16, Activation}

// ---- S0: decode + Karatsuba partial products -------------------------------
class PipeS0 extends Module {
  val io = IO(new Bundle {
    val fl  = Output(new Fp32Fl)
    val z2  = Output(UInt(24.W))
    val z0  = Output(UInt(24.W))
    val z1  = Output(UInt(26.W))
    val eab = Output(SInt(10.W))
  })
  val aR = Reg(UInt(32.W)); aR := aR + 1.U
  val bR = Reg(UInt(32.W)); bR := bR + 3.U
  val fl = Wire(new Fp32Fl)
  fl.ps := Fp32Ops.sgn(aR) ^ Fp32Ops.sgn(bR)
  fl.aN := Fp32Ops.isNan(aR); fl.bN := Fp32Ops.isNan(bR)
  fl.aI := Fp32Ops.isInf(aR); fl.bI := Fp32Ops.isInf(bR)
  fl.aZ := Fp32Ops.isZeroCls(aR); fl.bZ := Fp32Ops.isZeroCls(bR)
  val am = Mux(fl.aZ, 0.U, Cat(1.U(1.W), Fp32Ops.mant(aR)))
  val bm = Mux(fl.bZ, 0.U, Cat(1.U(1.W), Fp32Ops.mant(bR)))
  val aH = am(23, 12); val aL = am(11, 0)
  val bH = bm(23, 12); val bL = bm(11, 0)
  io.fl := fl
  io.z2 := aH * bH
  io.z0 := aL * bL
  io.z1 := (aL +& aH) * (bL +& bH)
  io.eab := Fp32Ops.expf(aR).zext +& Fp32Ops.expf(bR).zext
}

// ---- S1: 48-bit recombine + normalize + exponent path ----------------------
class PipeS1 extends Module {
  val io = IO(new Bundle {
    val fl2 = Output(new Fp32Fl)
    val pn  = Output(UInt(48.W))
    val px  = Output(SInt(12.W))
  })
  val flR = Reg(UInt(8.W)); flR := flR + 1.U
  val z2R = Reg(UInt(24.W)); z2R := z2R + 1.U
  val z0R = Reg(UInt(24.W)); z0R := z0R + 2.U
  val z1R = Reg(UInt(26.W)); z1R := z1R + 3.U
  val eabR = Reg(SInt(10.W)); eabR := eabR + 1.S
  val fl = Wire(new Fp32Fl)
  // unpack the 8-bit driver back into the flag bundle (keeps the fields live)
  fl.ps := flR(0); fl.aN := flR(1); fl.bN := flR(2)
  fl.aI := flR(3); fl.bI := flR(4); fl.aZ := flR(5); fl.bZ := flR(6)
  val prod = (z2R << 24) + ((z1R - (z0R +& z2R)) << 12) + z0R
  val pn1 = Mux(prod(47), prod, prod << 1)
  val px1 = eabR -& 300.S -& Mux(prod(47), 0.S, 1.S)
  io.fl2 := fl; io.pn := pn1; io.px := px1
}

// ---- S2 (narrow, fused): the live acc feedback tail ------------------------
class PipeS2 extends Module {
  val io = IO(new Bundle { val r = Output(UInt(32.W)) })
  val flR = Reg(UInt(8.W)); flR := flR + 1.U
  val pnR = Reg(UInt(48.W)); pnR := pnR + 1.U
  val pxR = Reg(SInt(12.W)); pxR := pxR + 1.S
  val cR  = Reg(UInt(32.W)); cR := cR + 1.U
  val fl = Wire(new Fp32Fl)
  fl.ps := flR(0); fl.aN := flR(1); fl.bN := flR(2)
  fl.aI := flR(3); fl.bI := flR(4); fl.aZ := flR(5); fl.bZ := flR(6)
  io.r := Fp32Narrow.fmaBackN(fl, pnR, pxR, cR)
}

// ---- S2 (exact 512-bit container, fused): what narrow already saved ---------
class PipeS2Exact extends Module {
  val io = IO(new Bundle { val r = Output(UInt(32.W)) })
  val flR = Reg(UInt(8.W)); flR := flR + 1.U
  val pnR = Reg(UInt(48.W)); pnR := pnR + 1.U
  val pxR = Reg(SInt(12.W)); pxR := pxR + 1.S
  val cR  = Reg(UInt(32.W)); cR := cR + 1.U
  val fl = Wire(new Fp32Fl)
  fl.ps := flR(0); fl.aN := flR(1); fl.bN := flR(2)
  fl.aI := flR(3); fl.bI := flR(4); fl.aZ := flR(5); fl.bZ := flR(6)
  io.r := Fp32Ops.fmaBack(fl, pnR, pxR, cR)
}

// ---- full real pipe (sanity: should match the hierarchical 826) ------------
class PipeFull extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(32.W)); val b = Input(UInt(32.W))
    val clear = Input(Bool()); val acc = Output(UInt(32.W))
  })
  val pipe = Module(new Fp32FmaPipe(fused = true, narrow = true))
  pipe.io.a := io.a; pipe.io.b := io.b; pipe.io.clear := io.clear
  io.acc := pipe.io.acc
}

// ---- tier FSM store path (the 8x 32-bit acc muxes + f32ToBf16) --------------
class Bf16Store(p: Parameters) extends Module {
  val n = p.Bf16MatrixDim
  val bpr = n * p.Bf16ElemBits / p.DtcmRowBits      // beats per row (1 at n=8)
  val storeBeats = n * bpr
  val lanesPerBeat = p.DtcmRowBits / p.Bf16ElemBits
  val io = IO(new Bundle {
    val acc    = Input(Vec(n * n, UInt(32.W)))
    val smBase = Input(UInt(p.AddrBits.W))
    val stCnt  = Input(UInt(5.W))
    val addr   = Output(UInt(p.AddrBits.W))
    val wdata  = Output(UInt(p.DtcmRowBits.W))
    val wstrb  = Output(UInt(p.DtcmRowBytes.W))
  })
  val row = if (bpr == 2) io.stCnt(log2Up(storeBeats) - 1, 1)
            else io.stCnt(log2Up(storeBeats) - 1, 0)
  val lane = Wire(Vec(lanesPerBeat, UInt(p.Bf16ElemBits.W)))
  for (e <- 0 until lanesPerBeat) {
    val j = if (bpr == 2) e.U + Mux(io.stCnt(0) === 1.U, lanesPerBeat.U, 0.U) else e.U
    lane(e) := Bf16.f32ToBf16(io.acc((row * n.U + j)(log2Up(n * n) - 1, 0)))
  }
  io.addr  := io.smBase + io.stCnt * p.DtcmRowBytes.U
  io.wdata := lane.asUInt
  io.wstrb := ~0.U(p.DtcmRowBytes.W)
}

// ---- tier FSM feed path (32 dynamic 16:1 operand muxes + widen) ------------
class Bf16Feed(p: Parameters) extends Module {
  val n = p.Bf16MatrixDim
  val io = IO(new Bundle {
    val aRows   = Input(Vec(n, Vec(n, UInt(p.Bf16ElemBits.W))))
    val bRows   = Input(Vec(n, Vec(n, UInt(p.Bf16ElemBits.W))))
    val beatCnt = Input(UInt(5.W))
    val aOut = Output(Vec(n, UInt(32.W)))
    val bOut = Output(Vec(n, UInt(32.W)))
  })
  for (i <- 0 until n) {
    io.aOut(i) := Cat(io.aRows(i)(io.beatCnt(log2Up(n) - 1, 0)), 0.U(16.W))
    io.bOut(i) := Cat(io.bRows(io.beatCnt(log2Up(n) - 1, 0))(i), 0.U(16.W))
  }
}

// ---- tier FSM load path (row stitch at 2 beats/row + dynamic row write) -----
class Bf16Load(p: Parameters) extends Module {
  val n          = p.Bf16MatrixDim
  val bpr        = n * p.Bf16ElemBits / p.DtcmRowBits  // beats per row (1 at n=8)
  val tileBeats  = n * bpr
  val io = IO(new Bundle {
    val rdata = Input(UInt(p.DtcmRowBits.W))
    val fBeat = Input(UInt(6.W))
    val ldFire = Input(Bool())
    val aOut = Output(Vec(n, Vec(n, UInt(p.Bf16ElemBits.W))))
    val bOut = Output(Vec(n, Vec(n, UInt(p.Bf16ElemBits.W))))
  })
  val asmLo: Option[UInt] =
    if (bpr == 2) Some(RegInit(0.U(p.DtcmRowBits.W))) else None
  val aRows = Reg(Vec(n, Vec(n, UInt(p.Bf16ElemBits.W)))); aRows := aRows
  val bRows = Reg(Vec(n, Vec(n, UInt(p.Bf16ElemBits.W)))); bRows := bRows
  when (io.ldFire) {
    val rowBeat = Mux(io.fBeat < tileBeats.U, io.fBeat, io.fBeat - tileBeats.U)
    val isA     = io.fBeat < tileBeats.U
    def splitRow(full: UInt, rowIdx: UInt): Unit = {
      val elems = Wire(Vec(n, UInt(p.Bf16ElemBits.W)))
      for (k <- 0 until n) elems(k) := full(p.Bf16ElemBits * (k + 1) - 1, p.Bf16ElemBits * k)
      when (isA) { aRows(rowIdx) := elems }.otherwise { bRows(rowIdx) := elems }
    }
    if (bpr == 2) {
      val rowIdx = rowBeat(log2Up(2 * n) - 1, 1)
      when (io.fBeat(0) === 0.U) {
        asmLo.get := io.rdata
      }.otherwise {
        splitRow(Cat(io.rdata, asmLo.get), rowIdx)
      }
    } else {
      splitRow(io.rdata, rowBeat(log2Up(n) - 1, 0))
    }
  }
  io.aOut := aRows; io.bOut := bRows
}

// ---- whole tier standalone (inputs driven from toggles; settles the FSM
//      own-logic attribution question against the full-design hier report) ----
class Bf16TierProbe(p: Parameters) extends Module {
  val io = IO(new Bundle { val done = Output(Bool()) })
  val tier = Module(new xnpu.Bf16MatmulUnit(p))
  val opR = Reg(UInt(8.W)); opR := opR + 1.U
  val fireR = Reg(Bool()); fireR := ~fireR
  val dataR = Reg(UInt(32.W)); dataR := dataR + 1.U
  val rsV = Reg(Bool()); rsV := ~rsV
  val ctxV = Reg(Bool()); ctxV := ~ctxV
  val ctxAR = Reg(UInt(p.AddrBits.W)); ctxAR := ctxAR + 1.U
  val ctxBR = Reg(UInt(p.AddrBits.W)); ctxBR := ctxBR + 1.U
  val rdataR = Reg(UInt(p.DtcmRowBits.W)); rdataR := rdataR + 1.U
  val readyR = Reg(Bool()); readyR := ~readyR
  tier.io.req.valid := true.B
  tier.io.req.bits.op := opR(5, 0).asTypeOf(xnpu.AluOp())
  tier.io.req.bits.addr := opR(4, 0)
  tier.io.req.bits.imm := dataR
  tier.io.rs1.valid := rsV
  tier.io.rs1.data := dataR
  tier.io.context.valid := ctxV
  tier.io.context.aBase := ctxAR
  tier.io.context.bBase := ctxBR
  tier.io.dtcm.rdata := rdataR
  tier.io.dtcm.ready := readyR
  val dV = Wire(Bool()); dV := tier.io.dtcm.valid
  val dW = Wire(Bool()); dW := tier.io.dtcm.write
  val dA = Wire(UInt(p.AddrBits.W)); dA := tier.io.dtcm.addr
  val dD = Wire(UInt(p.DtcmRowBits.W)); dD := tier.io.dtcm.wdata
  val dS = Wire(UInt(p.DtcmRowBytes.W)); dS := tier.io.dtcm.wstrb
  val dR = Wire(Bool()); dR := tier.io.req.ready
  io.done := tier.io.done
}

// ---- vector-unit per-lane fp32 unit costs (2026-11-01 resource audit) -------
// VectorUnit instantiates 6 addF32 / 9 mulF32 / 4 Activation sites x 8 lanes
// as parallel cones that never fire on the same cycle (FSM-serialized chains).
// These probes price one instance per lane so the sharing saving is measurable.

class VecAdd8 extends Module {
  val io = IO(new Bundle { val out = Output(Vec(8, UInt(32.W))) })
  val aR = Reg(Vec(8, UInt(32.W))); val bR = Reg(Vec(8, UInt(32.W)))
  for (k <- 0 until 8) { aR(k) := aR(k) + (k + 1).U; bR(k) := bR(k) + (k + 3).U }
  for (k <- 0 until 8) io.out(k) := Bf16.addF32(aR(k), bR(k))
}

class VecMul8 extends Module {
  val io = IO(new Bundle { val out = Output(Vec(8, UInt(32.W))) })
  val aR = Reg(Vec(8, UInt(32.W))); val bR = Reg(Vec(8, UInt(32.W)))
  for (k <- 0 until 8) { aR(k) := aR(k) + (k + 1).U; bR(k) := bR(k) + (k + 3).U }
  for (k <- 0 until 8) io.out(k) := Bf16.mulF32(aR(k), bR(k))
}

class VecExp8 extends Module {
  val io = IO(new Bundle {
    val s1 = Output(Vec(8, UInt(64.W)))
    val s2 = Output(Vec(8, UInt(32.W)))
  })
  val aR = Reg(Vec(8, UInt(32.W)))
  for (k <- 0 until 8) aR(k) := aR(k) + (k + 1).U
  for (k <- 0 until 8) {
    io.s1(k) := Activation.expStage1(aR(k)).asUInt
    io.s2(k) := Activation.expStage2(Activation.expStage1(aR(k) + 1.U))
  }
}

// ---- tier shell cost with REAL module outputs (the old Bf16TierProbe died
//      of DCE because its outputs dangled on internal Wires) ------------------
class Bf16Shell(p: Parameters) extends Module {
  val io = IO(new Bundle {
    val dv = Output(Bool()); val dw = Output(Bool())
    val da = Output(UInt(p.AddrBits.W))
    val dd = Output(UInt(p.DtcmRowBits.W))
    val ds = Output(UInt(p.DtcmRowBytes.W))
    val rdy = Output(Bool()); val done = Output(Bool())
  })
  val tier = Module(new xnpu.Bf16MatmulUnit(p))
  val opR = Reg(UInt(8.W)); opR := opR + 1.U
  val dataR = Reg(UInt(32.W)); dataR := dataR + 1.U
  val rsV = Reg(Bool()); rsV := ~rsV
  val ctxV = Reg(Bool()); ctxV := ~ctxV
  val ctxAR = Reg(UInt(p.AddrBits.W)); ctxAR := ctxAR + 1.U
  val ctxBR = Reg(UInt(p.AddrBits.W)); ctxBR := ctxBR + 3.U
  val rdataR = Reg(UInt(p.DtcmRowBits.W)); rdataR := rdataR + 1.U
  val readyR = Reg(Bool()); readyR := ~readyR
  tier.io.req.valid := true.B
  tier.io.req.bits.op := opR(5, 0).asTypeOf(xnpu.AluOp())
  tier.io.req.bits.addr := opR(4, 0)
  tier.io.req.bits.imm := dataR
  tier.io.rs1.valid := rsV
  tier.io.rs1.data := dataR
  tier.io.context.valid := ctxV
  tier.io.context.aBase := ctxAR
  tier.io.context.bBase := ctxBR
  tier.io.dtcm.rdata := rdataR
  tier.io.dtcm.ready := readyR
  io.dv := tier.io.dtcm.valid
  io.dw := tier.io.dtcm.write
  io.da := tier.io.dtcm.addr
  io.dd := tier.io.dtcm.wdata
  io.ds := tier.io.dtcm.wstrb
  io.rdy := tier.io.req.ready
  io.done := tier.io.done
}
