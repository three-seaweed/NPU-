// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


// NxN output-stationary systolic array (N = p.MatrixDim).
//
// Operand A is streamed in at the left edge (one element per row, per cycle)
// and flows left-to-right; operand B is streamed in at the top edge (one
// element per column) and flows top-to-bottom. With the caller staggering the
// edge feeds by row/column index, PE(i,j) accumulates Sum_k A(i,k)*B(k,j).
//
// WS dormant datapath (OS-constant driven until the C4 driver lands):
// `mode/wLoad/wClr/wCnt` broadcast to every PE; partial sums enter each
// column at the top (`psIn`, injected by the driver as one zero wave per
// output row) and drain at the bottom (`psOut`), one register per row -- the
// same cadence as the B operand chain they share skew alignment with.
class SystolicArray(p: Parameters) extends Module {
    val n = p.MatrixDim

    val io = IO(new Bundle {
        val en    = Input(Bool())
        val clear = Input(Bool())                                     // zero accumulators
        val flush = Input(Bool())                                     // zero operand pipeline
        val aIn   = Input(Vec(n, SInt(p.ElemBits.W)))                 // left edge, one per row
        val bIn   = Input(Vec(n, SInt(p.ElemBits.W)))                 // top edge, one per column
        val acc   = Output(Vec(n, Vec(n, SInt(p.AccBits.W))))         // acc(row)(col)
        // WS dormant datapath
        val mode  = Input(Bool())
        val wLoad = Input(Bool())
        val wClr  = Input(Bool())
        val wCnt  = Input(UInt(log2Up(2 * n).W))
        val psIn  = Input(Vec(n, SInt(p.PsBits.W)))                   // WS ps wave injection (top)
        val psOut = Output(Vec(n, SInt(p.PsBits.W)))                  // WS ps drain (bottom)
    })

    val pe = Seq.tabulate(n, n)((i, _) => Module(new PE(p, i)))

    for (i <- 0 until n) {
        for (j <- 0 until n) {
            pe(i)(j).io.en    := io.en
            pe(i)(j).io.clear := io.clear
            pe(i)(j).io.flush := io.flush
            pe(i)(j).io.mode  := io.mode
            pe(i)(j).io.wLoad := io.wLoad
            pe(i)(j).io.wClr  := io.wClr
            pe(i)(j).io.wCnt  := io.wCnt
            // A enters from the left edge of the row, then flows right.
            pe(i)(j).io.inA := (if (j == 0) io.aIn(i) else pe(i)(j - 1).io.outA)
            // B enters from the top edge of the column, then flows down.
            pe(i)(j).io.inB := (if (i == 0) io.bIn(j) else pe(i - 1)(j).io.outB)
            // WS partial sums enter at the top of each column and flow down,
            // one register per row -- same cadence as the b chain.
            pe(i)(j).io.psIn := (if (i == 0) io.psIn(j) else pe(i - 1)(j).io.psOut)
            io.acc(i)(j) := pe(i)(j).io.acc
            if (i == n - 1) io.psOut(j) := pe(i)(j).io.psOut
        }
    }
}
