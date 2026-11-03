// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._
import fp32matmul.Fp32SystolicArray

// Shared interface of the two FP.* tiers (fp32 4x4 and bf16 NxN): the
// Matmul shell routes to whichever the parameters selected, type-agnostically.
class FpTierBundle(p: Parameters) extends Bundle {
    val req     = Flipped(Decoupled(new AluCmd))
    val rs1     = Flipped(new RegfileReadDataIO)      // FP.SM store base
    val context = Input(new MatrixContext(p))         // A/B base snapshot
    val dtcm    = new DtcmPortIO(p)                   // 128-bit, int8-style
    val done    = Output(Bool())                      // token retire pulse
}
trait FpTier { self: Module =>
    val io: FpTierBundle
}

// bf16-in / fp32-accumulate / bf16-out NxN matmul tier behind the FP.*
// instructions (C6; Parameters(bf16Matmul = true), mutually exclusive with the
// fp32 4x4 tier). Contract:
//   * inputs are IEEE bf16 bits; widening bf16 -> fp32 is exact (<<16)
//   * the MAC is the existing Fp32FmaPipe (fused, single rounding per MAC,
//     narrow erratum backend) so every PE accumulates in strict k order --
//     bit-exact against a serial Fp32Model.fma fold over widened operands
//   * FP.MATMULA keeps the fp32 accumulators across tiles: the ONLY rounding
//     to bf16 happens once, in FP.SM's exit (Bf16.f32ToBf16, RNE) -- never at
//     a tile boundary
//
// Beat shape follows Bf16MatrixDim: an n-bf16 row is n*16 bit, i.e. 2 DTCM
// beats at n=16 (loads stitch two beats into a row register) but exactly ONE
// beat at n=8 (row == beat, like the 4x4 fp32 unit -- no stitch register).
// Stores always pack 8 bf16 results per beat.
//
//   LMA/LMB    latch operand bases (same broadcast contract as int8); 1 cycle
//   FP.MATMUL  load A, load B (tileBeats each), clear, feed n beats,
//              drain (2n-3+Fp32PipeLatency-1) -> done
//   FP.MATMULA same without the clear; k order across tiles stays serial
//   FP.SM      convert + burst-store the NxN bf16 tile, storeBeats -> done
class Bf16MatmulUnit(p: Parameters) extends Module with FpTier {
    val n          = p.Bf16MatrixDim                        // 8
    val rowBits    = n * p.Bf16ElemBits                     // 128
    val beatsPerRow = rowBits / p.DtcmRowBits               // 1 (2 at n=16)
    val tileBeats  = n * beatsPerRow                        // 8 beats per tile
    val storeBeats = n * beatsPerRow                        // 8 (bf16 out, same shape)
    // Feed end -> last acc stable: last beat reaches PE(n-1,n-1) after 2n-2
    // skew, the FMA pipe writes acc Fp32PipeLatency-1 cycles later (= 15).
    val drainCycles = 2 * n - 3 + p.Fp32PipeLatency - 1
    require(beatsPerRow == 1 || beatsPerRow == 2,
        "a bf16 row must be one or two whole DTCM beats")
    require(n % (p.DtcmRowBits / p.Bf16ElemBits) == 0,
        "a store beat must hold a whole number of bf16 results")

    val io = IO(new FpTierBundle(p))

    val sIdle :: sLoad :: sFeed :: sStore :: Nil = Enum(4)
    val state = RegInit(sIdle)

    val aBase = RegInit(0.U(p.AddrBits.W))
    val bBase = RegInit(0.U(p.AddrBits.W))

    // Loaded operand tiles, row-major bf16 bits: aRows(i)(k) = A(i,k).
    val aRows = Reg(Vec(n, Vec(n, UInt(p.Bf16ElemBits.W))))
    val bRows = Reg(Vec(n, Vec(n, UInt(p.Bf16ElemBits.W))))

    val array = Module(new Fp32SystolicArray(n, fused = true, narrow = true))
    // narrow = true is the erratum-fixed backend; do not set false (iron rule 5).

    val clearPending = RegInit(false.B)
    val ldBeat   = RegInit(0.U(log2Up(2 * tileBeats).W))  // 0..2*tileBeats-1: A then B
    val beatCnt  = RegInit(0.U(log2Up(n + 1).W))          // feed beat 0..n
    val drainCnt = RegInit(0.U(log2Up(drainCycles + 1).W))
    val smBase   = RegInit(0.U(p.AddrBits.W))
    val stCnt    = RegInit(0.U(log2Up(storeBeats + 1).W))
    // Low-half stitch register for the first beat of each 2-beat row (n=16
    // only; at n=8 a row IS one beat and no stitch exists).
    val asmLo    = if (beatsPerRow == 2) Some(Reg(UInt(p.DtcmRowBits.W))) else None

    // ---- Defaults ------------------------------------------------------------
    io.dtcm.valid := false.B
    io.dtcm.write := false.B
    io.dtcm.addr  := 0.U
    io.dtcm.wdata := 0.U
    io.dtcm.wstrb := 0.U
    io.done := false.B
    array.io.clear := false.B
    for (i <- 0 until n) {
        array.io.aIn(i) := 0.U                             // idle feed 0 => fma(0,0,acc)=acc
        array.io.bIn(i) := 0.U
    }

    io.req.ready := state === sIdle

    // ---- Dispatch ------------------------------------------------------------
    when (state === sIdle && io.req.fire) {
        switch (io.req.bits.op) {
            is (AluOp.LMA) { aBase := io.rs1.data }
            is (AluOp.LMB) { bBase := io.rs1.data }
            is (AluOp.FP_SM) {
                smBase := io.rs1.data
                stCnt  := 0.U
                state  := sStore
            }
            is (AluOp.FP_MATMUL, AluOp.FP_MATMULA) {
                clearPending := io.req.bits.op === AluOp.FP_MATMUL
                when (io.context.valid) {
                    aBase := io.context.aBase
                    bBase := io.context.bBase
                }
                ldBeat   := 0.U
                beatCnt  := 0.U
                drainCnt := 0.U
                state    := sLoad
            }
        }
    }

    // ---- Load ----------------------------------------------------------------
    // One beat read per cycle; a row spans beatsPerRow beats (low half first
    // when two). The engine alternates A rows then B rows (tileBeats each).
    when (state === sLoad) {
        val isA = ldBeat < tileBeats.U
        val rowBeat = Mux(isA, ldBeat, ldBeat - tileBeats.U)   // 0..tileBeats-1
        io.dtcm.valid := true.B
        io.dtcm.write := false.B
        io.dtcm.addr  := Mux(isA, aBase, bBase) + rowBeat * p.DtcmRowBytes.U
        when (io.dtcm.ready) { ldBeat := ldBeat + 1.U }
    }
    // Read returns one cycle after issue; row index rides one reg.
    val ldFire = RegNext(state === sLoad && io.dtcm.valid && io.dtcm.ready, false.B)
    val fBeat  = RegNext(ldBeat, 0.U)
    when (ldFire) {
        val rowBeat = Mux(fBeat < tileBeats.U, fBeat, fBeat - tileBeats.U)
        val isA     = fBeat < tileBeats.U
        def splitRow(full: UInt, rowIdx: UInt): Unit = {
            val elems = Wire(Vec(n, UInt(p.Bf16ElemBits.W)))
            for (k <- 0 until n) {
                elems(k) := full(p.Bf16ElemBits * (k + 1) - 1, p.Bf16ElemBits * k)
            }
            when (isA) { aRows(rowIdx) := elems } .otherwise { bRows(rowIdx) := elems }
        }
        if (beatsPerRow == 2) {
            val rowIdx = rowBeat(log2Up(2 * n) - 1, 1)        // row = beat/2
            when (fBeat(0) === 0.U) {
                asmLo.get := io.dtcm.rdata                    // low 8 elements
            } .otherwise {
                splitRow(Cat(io.dtcm.rdata, asmLo.get), rowIdx) // 256-bit bf16 row
            }
        } else {
            splitRow(io.dtcm.rdata, rowBeat(log2Up(n) - 1, 0)) // beat == row
        }
    }
    // Switch to sFeed on the cycle the LAST beat's issue fires (not when the
    // counter wraps -- that would issue a stray read beyond the token's
    // declared rows; the 4x4 unit's off-by-one lesson, same shape).
    when (state === sLoad && ldBeat === (2 * tileBeats - 1).U && io.dtcm.ready) {
        state := sFeed
    }

    // ---- Feed ----------------------------------------------------------------
    // 1 clear cycle (FP.MATMUL only), then n back-to-back beats feeding
    // A column k / B row k, each element widened exactly by <<16. PE(i,j)
    // MACs A(i,k)*B(k,j) on cycle k+i+j -- strict k order, bit-exact contract.
    def widen(h: UInt): UInt = Cat(h, 0.U(16.W))             // exact bf16 -> fp32
    when (state === sFeed) {
        when (clearPending) {
            array.io.clear := true.B
            clearPending := false.B
        } .elsewhen (beatCnt =/= n.U) {
            for (i <- 0 until n) {
                array.io.aIn(i) := widen(aRows(i)(beatCnt(log2Up(n) - 1, 0)))  // A(i,k)
                array.io.bIn(i) := widen(bRows(beatCnt(log2Up(n) - 1, 0))(i))  // B(k,i)
            }
            beatCnt := beatCnt + 1.U
        } .otherwise {
            drainCnt := drainCnt + 1.U
            when (drainCnt === drainCycles.U) {
                io.done := true.B
                state := sIdle
            }
        }
    }

    // ---- Store ---------------------------------------------------------------
    // One rounding point in the whole tier: fp32 acc -> bf16 (RNE) here, at
    // the final exit. Beat layout mirrors load: 8 elements per beat, low
    // half of a row first (row == beat at n=8), row-major.
    when (state === sStore) {
        val issue = stCnt =/= storeBeats.U
        io.dtcm.valid := issue
        io.dtcm.write := true.B
        io.dtcm.addr  := smBase + stCnt * p.DtcmRowBytes.U
        val row  = if (beatsPerRow == 2) stCnt(log2Up(storeBeats) - 1, 1) // row = beat/2
                   else stCnt(log2Up(storeBeats) - 1, 0)                 // beat == row
        val lane = Wire(Vec(p.DtcmRowBits / p.Bf16ElemBits, UInt(p.Bf16ElemBits.W)))
        val lanesPerBeat = p.DtcmRowBits / p.Bf16ElemBits     // 8 bf16 per beat
        for (e <- 0 until lanesPerBeat) {
            val j = if (beatsPerRow == 2)                     // e (+8 on the hi half)
                        e.U + Mux(stCnt(0) === 1.U, lanesPerBeat.U, 0.U)
                    else e.U
            lane(e) := Bf16.f32ToBf16(
                array.io.acc((row * n.U + j)(log2Up(n * n) - 1, 0)))
        }
        io.dtcm.wdata := lane.asUInt
        io.dtcm.wstrb := ~0.U(p.DtcmRowBytes.W)
        when (issue && io.dtcm.ready) {
            stCnt := stCnt + 1.U
            when (stCnt === (storeBeats - 1).U) {
                io.done := true.B
                state := sIdle
            }
        }
    }
}
