// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


// Processing element of the output-stationary systolic array (OS is the
// default and, until the C4 driver lands, the only dataflow the Matmul unit
// activates -- the WS datapath below is driven with OS-constant controls).
//
// Each enabled cycle it multiplies its incoming operands and accumulates
// the product into a stationary int32 register, while passing the operands on
// to the right (a) and downward (b) neighbours with a one-cycle delay. The
// register pipeline is what realises the diagonal "skew" of the array.
//
// `clear` zeroes only the accumulator (start a fresh C = A x B). `flush` zeroes
// the operand-passthrough registers so a new feed begins from a clean pipeline;
// without it, leftover operands from a previous matmul would inject spurious
// products during the next feed's ramp-up. The caller pulses `flush` at the
// start of every matmul; `clear` only when *not* accumulating.
//
// ---- WS (weight-stationary) datapath, gemmini-style -----------------------
// `mode`=WS holds one resident weight `wReg` while activations keep flowing
// right on the a chain, and partial sums flow DOWN a parallel ps chain at one
// register per row. Unlike the gemmini shared-multiplier lesson, WS gets its
// OWN multiplier: `pReg := psIn + inA*wReg` written as a bare MACC lets Vivado
// absorb the 20-bit partial-sum add into a second DSP's C port, removing the
// per-PE LUT adder (~5-7K LUT across 256 PE at the 83.7% congestion wall;
// the spare DSP costs ~33% of a pool we are not close to filling).
// BUILD REQUIREMENT: plain emission loses this in full-design context --
// Vivado's resource sharing merges the two common-operand multiplies back
// into a LUT-fabric multiplier (measured 2026-11-03: +24.6K LUT, DSP 568->312,
// even though a standalone PE infers 2 DSP + 32 LUT fine). The staged
// NpuTop.sv MUST stamp (* use_dsp = "yes" *) on every PE module at the
// staging copy step; with it the full tree lands at 222,305 LUT / 824 DSP
// (2 per PE + bf16 tier + vector) / FF 48,019 (acc+pReg absorbed into DSP
// registers) vs the 241,194 / 568 / 63,383 baseline: -18,889 LUT.
// The ps wavefront for output row m meets the
// a-wave at PE(i,j) on cycle m+i+j -- the same skew injection as OS, so the
// driver needs no second skew logic. `psOut = reg(psIn + inA*wReg)`: at K=16
// per tile the widest possible column sum is 16*127*127 < 2^19, so the PsBits
// chain cannot overflow (see Parameters.PsBits).
// Weights are preloaded down the EXISTING b chain (one row per cycle; W[i][j]
// fed at count i reaches row i on count 2i): `wLoad && wCnt === 2*row`
// captures into wReg. `wClr` drops the resident weight (mode switches pulse
// it together with clear+flush -- switching invalidates the weights, an ISA
// contract documented with MCFG2).
class PE(p: Parameters, row: Int = 0) extends Module {
    val io = IO(new Bundle {
        val en    = Input(Bool())                  // run / accumulate enable
        val clear = Input(Bool())                  // reset accumulator to 0
        val flush = Input(Bool())                  // reset operand passthrough regs
        val inA   = Input(SInt(p.ElemBits.W))      // operand from left
        val inB   = Input(SInt(p.ElemBits.W))      // operand from top
        val outA  = Output(SInt(p.ElemBits.W))     // registered passthrough to right
        val outB  = Output(SInt(p.ElemBits.W))     // registered passthrough below
        val acc   = Output(SInt(p.AccBits.W))      // accumulated dot-product
        // WS dormant datapath (OS-constant driven until the C4 driver lands)
        val mode  = Input(Bool())                  // 0=OS 1=WS
        val wLoad = Input(Bool())                  // weight waterfall capture window
        val wClr  = Input(Bool())                  // drop resident weight (mode switch)
        val wCnt  = Input(UInt(log2Up(2 * p.MatrixDim).W)) // waterfall cycle count
        val psIn  = Input(SInt(p.PsBits.W))        // partial sum from above (WS)
        val psOut = Output(SInt(p.PsBits.W))       // registered partial sum below (WS)
    })

    val acc = RegInit(0.S(p.AccBits.W))
    val aReg = RegInit(0.S(p.ElemBits.W))
    val bReg = RegInit(0.S(p.ElemBits.W))
    val wReg = RegInit(0.S(p.ElemBits.W))          // WS resident weight
    val pReg = RegInit(0.S(p.PsBits.W))            // WS partial-sum pass register

    // int8 * int8 -> int16 products, one per dataflow. OS keeps the P-feedback
    // MAC accumulator; the bare WS MACC shape below lets the 20-bit partial-sum
    // add absorb into a second DSP's C port. The B operands differ (inB vs
    // wReg), so synthesis resource sharing cannot re-merge the two multiplies.
    // A product computed under the other mode is gated off below and never
    // observed.
    val osProduct = io.inA * io.inB
    val wsProduct = io.inA * wReg

    when (io.clear) {
        acc := 0.S
    } .elsewhen (io.en && !io.mode) {
        acc := acc + osProduct
    }

    when (io.flush) {
        aReg := 0.S
        bReg := 0.S
    } .elsewhen (io.en) {
        aReg := io.inA
        bReg := io.inB
    }

    // WS weight capture: the waterfall rides the b chain (1 cycle per row),
    // so W[i][j] fed into the top at count i arrives at this row on count 2i.
    when (io.wClr) {
        wReg := 0.S
    } .elsewhen (io.wLoad && io.wCnt === (2 * row).U) {
        wReg := io.inB
    }

    // WS partial-sum chain: same one-cycle-per-row cadence as the b chain, so
    // the drain at the bottom edge stays aligned with the a-wave skew.
    when (io.flush || io.wClr) {
        pReg := 0.S
    } .elsewhen (io.en && io.mode) {
        pReg := io.psIn + wsProduct
    }

    io.outA := aReg
    io.outB := bReg
    io.psOut := pReg
    io.acc  := acc
}
