// Output-stationary fp32 systolic-array matmul, standalone from the XiangJiang
// NPU (no DTCM, no instruction stream - the testbench feeds operands directly).
//
//   Fp32Pe          one accumulator: acc <- a*b+acc, one MAC per cycle.  The
//                    FMA itself is the 3-stage Fp32FmaPipe (Karatsuba x3 DSP
//                    front, exact 512-bit accumulate back, 1-cycle acc loop).
//   Fp32SystolicArray n x n PEs with the diagonal skew built in:
//                    aIn(i)=A(i,k), bIn(j)=B(k,j) on beat k; PE(i,j) MACs
//                    A(i,k)*B(k,j) at cycle k+i+j, so every PE accumulates in
//                    strict k order (bit-exact against the software golden).
//   Fp32Matmul      FSM top: start -> run (kLen beats, inReady/inValid) ->
//                    drain -> capture C -> one-cycle done pulse.
//
// fused=true  : one rounding per MAC (single rounding, FMA)
// fused=false : mul then add, one rounding each (double rounding)

package fp32matmul

import chisel3._
import chisel3.util._

class Fp32Pe(val fused: Boolean, val narrow: Boolean = false) extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Input(UInt(32.W))
    val clear = Input(Bool())
    val acc = Output(UInt(32.W))
    val aOut = Output(UInt(32.W))
    val bOut = Output(UInt(32.W))
  })
  val pipe = Module(new Fp32FmaPipe(fused, narrow))
  pipe.io.a := io.a
  pipe.io.b := io.b
  pipe.io.clear := io.clear
  io.acc := pipe.io.acc
  // operand pass-through stays 1 cycle so the wavefront/skew timing of the
  // array is unchanged by pipelining the FMA
  val aR = RegInit(0.U(32.W))
  val bR = RegInit(0.U(32.W))
  aR := io.a
  bR := io.b
  io.aOut := aR
  io.bOut := bR
}

class Fp32SystolicArray(val n: Int, val fused: Boolean, val narrow: Boolean = false) extends Module {
  val io = IO(new Bundle {
    val aIn = Input(Vec(n, UInt(32.W))) // aIn(i) = A(i, k) on beat k
    val bIn = Input(Vec(n, UInt(32.W))) // bIn(j) = B(k, j) on beat k
    val clear = Input(Bool())
    val acc = Output(Vec(n * n, UInt(32.W)))
  })
  val pes = Seq.tabulate(n, n)((_, _) => Module(new Fp32Pe(fused, narrow)))
  for (i <- 0 until n; j <- 0 until n) {
    val pe = pes(i)(j)
    // row operands flow right, column operands flow down, both through the
    // PE operand registers; the input-side skew (row i by i, col j by j) puts
    // the (i,k)/(k,j) pair of every k on PE(i,j) in the same cycle.
    pe.io.a := (if (j == 0) ShiftRegister(io.aIn(i), i, 0.U, true.B)
                else pes(i)(j - 1).io.aOut)
    pe.io.b := (if (i == 0) ShiftRegister(io.bIn(j), j, 0.U, true.B)
                else pes(i - 1)(j).io.bOut)
    pe.io.clear := io.clear
    io.acc(i * n + j) := pe.io.acc
  }
}

class Fp32Matmul(val n: Int = 16, val fused: Boolean = true,
                 val narrow: Boolean = false) extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())
    val kLen = Input(UInt(16.W))
    val inValid = Input(Bool())
    val inReady = Output(Bool())
    val inA = Input(Vec(n, UInt(32.W)))
    val inB = Input(Vec(n, UInt(32.W)))
    val done = Output(Bool())
    val busy = Output(Bool())
    val outC = Output(Vec(n * n, UInt(32.W)))
  })

  val sIdle :: sRun :: sDrain :: sDone :: Nil = Enum(4)
  val state = RegInit(sIdle)
  val beat = RegInit(0.U(16.W))
  val drain = RegInit(0.U(8.W))
  val kReg = RegInit(1.U(16.W))
  val cRegs = Reg(Vec(n * n, UInt(32.W)))

  val arr = Module(new Fp32SystolicArray(n, fused, narrow))
  val inReady = state === sRun && beat =/= kReg
  val feed = inReady && io.inValid
  (arr.io.aIn, io.inA).zipped.foreach((p, v) => p := Mux(feed, v, 0.U))
  (arr.io.bIn, io.inB).zipped.foreach((p, v) => p := Mux(feed, v, 0.U))
  arr.io.clear := state === sIdle
  io.inReady := inReady

  when(feed) { beat := beat + 1.U }
  when(state === sIdle && io.start) {
    state := sRun; beat := 0.U; kReg := Mux(io.kLen === 0.U, 1.U, io.kLen)
  }.elsewhen(state === sRun && beat === kReg) {
    state := sDrain; drain := 0.U
  }.elsewhen(state === sDrain) {
    // last MAC pair reaches PE(n-1,n-1) at (last beat) + 2n-2; its acc
    // register is written LATENCY-1 cycles later and readable one cycle
    // after that.  Capture when acc is stable, then pulse done.
    when(drain === (2 * n - 3 + Fp32FmaPipe.LATENCY - 1).U) {
      cRegs := arr.io.acc
      state := sDone
    }.otherwise { drain := drain + 1.U }
  }.elsewhen(state === sDone) {
    state := sIdle
  }

  io.done := state === sDone
  io.busy := state =/= sIdle
  io.outC := cRegs
}
