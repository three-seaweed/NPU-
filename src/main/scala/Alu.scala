// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


object Alu {
    def apply(p:Parameters): Alu = {
        return Module(new Alu(p))
    }
}


object AluOp extends ChiselEnum {
    // RV32I scalar subset (single-cycle datapath). Register-register and
    // register-immediate forms are distinct ops; the immediate forms read the
    // second operand from req.imm, the register forms from rs2.
    val ADD  = Value // rd = rs1 + rs2
    val SUB  = Value // rd = rs1 - rs2
    val AND  = Value // rd = rs1 & rs2
    val OR   = Value // rd = rs1 | rs2
    val XOR  = Value // rd = rs1 ^ rs2
    val SLL  = Value // rd = rs1 << rs2[4:0]
    val SRL  = Value // rd = rs1 >> rs2[4:0]  (logical)
    val SRA  = Value // rd = rs1 >>> rs2[4:0] (arithmetic)
    val SLT  = Value // rd = (rs1 <s rs2) ? 1 : 0
    val SLTU = Value // rd = (rs1 <u rs2) ? 1 : 0
    val ADDI = Value // rd = rs1 + imm
    val ANDI = Value // rd = rs1 & imm
    val ORI  = Value // rd = rs1 | imm
    val XORI = Value // rd = rs1 ^ imm
    val SLLI = Value // rd = rs1 << imm[4:0]
    val SRLI = Value // rd = rs1 >> imm[4:0]  (logical)
    val SRAI = Value // rd = rs1 >>> imm[4:0] (arithmetic)
    val SLTI = Value // rd = (rs1 <s imm) ? 1 : 0
    val SLTIU= Value // rd = (rs1 <u imm) ? 1 : 0
    val LUI  = Value // rd = imm (imm already shifted left by 12 in decode)
    val AUIPC= Value // rd = pc + imm   (result computed in Control, which holds pc)

    // Control flow (resolved in Control's EX stage; the Alu treats them as NOPs).
    val BEQ  = Value
    val BNE  = Value
    val BLT  = Value
    val BGE  = Value
    val BLTU = Value
    val BGEU = Value
    val JAL  = Value // rd = pc + 4; redirect to pc + imm
    val JALR = Value // rd = pc + 4; redirect to (rs1 + imm) & ~1

    // Scalar DTCM word access (blocking-in-EX, shares the wide matrix port).
    val LW   = Value // rd = DTCM[rs1 + imm]
    val SW   = Value // DTCM[rs1 + imm] = rs2

    // NPU
    // MatMul (4x4 systolic array)
    val LMA     = Value // load matrix A from dtcm into the A skew-buffer
    val LMB     = Value // load matrix B from dtcm into the B skew-buffer
    val MATMUL  = Value // matrix multiplication C = A x B (clears accumulators)
    val MATMULA = Value // accumulating matmul C += A x B (no clear)
    val SM      = Value // store the matmul result from accumulators to dtcm
    val MQCFG   = Value // bf16-A quantization context: scale, enable and group metadata

    // Core-issued DMA (custom-0). DMALD/DMAST trigger the AXI-master DMA engine
    // and are fire-and-forget (issue, then keep executing); DMAWAIT blocks in EX
    // until the engine is idle. dram=GPR[rs1], dtcm=GPR[rs2], rows=imm.
    val DMALD   = Value // DRAM[rs1] -> DTCM[rs2], imm rows
    val DMAST   = Value // DTCM[rs2] -> DRAM[rs1], imm rows
    val DMAWAIT = Value // stall until the DMA engine is idle

    // Vector core (custom-1). Config ops latch state from rs1; compute/reduce ops
    // stream bf16 vectors from DTCM, with explicit format conversion ops for the
    // matrix data path. See docs/ISA.md and VectorUnit.scala.
    val VSETA    = Value // srcABase := rs1
    val VSETB    = Value // srcBBase := rs1
    val VSETD    = Value // dstBase  := rs1
    val VSETLEN  = Value // vlen     := rs1 (element count)
    val VSETS    = Value // scalarOp := rs1 (raw fp32 bits, broadcast operand)
    val VADD     = Value // dst = srcA + srcB        (elementwise bf16)
    val VSUB     = Value // dst = srcA - srcB
    val VMUL     = Value // dst = srcA * srcB
    val VFMA     = Value // dst = srcA * srcB + dst   (fp32 accumulate, RMW)
    val VSMUL    = Value // dst = srcA * scalar
    val VSADD    = Value // dst = srcA + scalar
    val VEXP     = Value // dst = exp(srcA)
    val VRECIP   = Value // dst = 1 / srcA
    val VRSQRT   = Value // dst = 1 / sqrt(srcA)
    val VSUM     = Value // rd = fp32 sum(srcA)
    val VMAX     = Value // rd = fp32 max(srcA)
    val VDOT     = Value // rd = fp32 sum(srcA * srcB)
    val VSOFTEXP = Value // dst = exp(srcA - scalar); rd = fp32 sum(dst)

    // Fundamental vector/data-path operations. Quantization policy (scales, zero
    // points and grouping) is assembled in software from these primitives.
    val VABS       = Value // dst_bf16 = abs(srcA_bf16)
    val VCVT_I8    = Value // dst_i8 = saturating RNE conversion from srcA_bf16
    val VCVT_BF16  = Value // dst_bf16 = RNE conversion from srcA_i32

    // Native fp32 matrix tier (custom-0 funct7 variants): self-describing
    // MATMUL/MATMULA/SM that route to the fp32 array regardless of MQCFG.
    val FP_MATMUL  = Value // fp32 C = A x B (clears the fp32 accumulators)
    val FP_MATMULA = Value // fp32 C += A x B (k-tile accumulation in-array)
    val FP_SM      = Value // store the n x n fp32 result tile to dtcm

    // Engine/dataflow configuration (custom-0, MCFG2). Config-only: no DTCM
    // range, latches the int8 engine's mode like LMA/LMB latch bases.
    val MCFG2      = Value
}

class AluCmd extends Bundle {
    val addr = UInt(5.W)        // destination register (rd)
    val op   = AluOp()
    val imm  = UInt(32.W)       // immediate (scalar ops); matrix base goes via rs1
}

// Matrix and vector commands may outlive the scalar EX stage.  Capture both
// source operands alongside the decoded command so later scalar writeback or
// forwarding cannot change the values observed by the background unit.
class DtcmRange(p: Parameters) extends Bundle {
    val valid = Bool()
    val base  = UInt(p.AddrBits.W)
    val rows  = UInt(16.W)       // number of 16-byte physical DTCM rows
}

class DtcmAccess(p: Parameters) extends Bundle {
    val read  = Vec(3, new DtcmRange(p))
    val write = new DtcmRange(p)
}

class DtcmToken(p: Parameters, slots: Int) extends Bundle {
    val valid = Bool()
    val access = new DtcmAccess(p)
    // Bits name earlier commands this token must wait for. A command may issue
    // once every referenced token has retired.
    val deps = UInt(slots.W)
}

// Quantization state is carried alongside every Matrix command instead of being
// read from mutable scalar registers when the loader eventually starts. `aBf16`
// selects the 64-byte bf16 activation row layout; `enable` applies fp32 scale
// then the same RNE/saturating int8 conversion as VCVT.I8. groupSize is metadata
// for software's MATMULA/SM group boundary and is checked by the assembler/tests.
class MatrixQuantConfig(p: Parameters) extends Bundle {
    val aBf16 = Bool()
    val enable = Bool()
    val scale = UInt(32.W)
    val groupSize = UInt(16.W)
    // Identifies the K-group represented by this command.  The Matrix datapath
    // does not consume it, but keeping it in MatrixIdex lets software associate
    // each SM int32 partial with the right activation/weight scales.
    val groupIndex = UInt(15.W)
}

class MatrixContext(p: Parameters) extends Bundle {
    val valid = Bool()
    val aBase = UInt(p.AddrBits.W)
    val bBase = UInt(p.AddrBits.W)
}

class VectorContext(p: Parameters) extends Bundle {
    val valid = Bool()
    val srcA = UInt(p.AddrBits.W)
    val srcB = UInt(p.AddrBits.W)
    val dst = UInt(p.AddrBits.W)
    val vlen = UInt(16.W)
    val scalar = UInt(32.W)
}

class QueuedAluCmd(p: Parameters) extends Bundle {
    val cmd = new AluCmd
    val rs1 = UInt(32.W)
    val rs2 = UInt(32.W)
    // Int8 engine this entry targets (snapshotted from engineSelShadow at
    // enqueue; MCFG2 entries carry their own rs2(0) target). FP-class ops
    // ignore it.
    val engine = UInt(1.W)
    val matrixQuant = new MatrixQuantConfig(p)
    val matrixContext = new MatrixContext(p)
    val vectorContext = new VectorContext(p)
    // Area diet 2026-11-02: the full DtcmAccess (196 bit) was snapshotted per
    // entry but consumed after enqueue ONLY through needsToken() = OR of its
    // four range-valid bits (issue gating + tokenBusy latching).  Store that
    // one bit instead; the ranges themselves live in the token table only.
    val needsTok = Bool()
    val tokenId = UInt(log2Ceil(p.TokenSlots).W)
}

class Alu(p: Parameters) extends Module {
    val io = IO(new Bundle {
        // Decode.
        val req = Flipped(Decoupled(new AluCmd))

        // Execute cycle.
        val rs1 = Flipped(new RegfileReadDataIO)
        val rs2 = Flipped(new RegfileReadDataIO) // Consumer IN-IO: Flipped(valid,bits,ready)
        val rd  = Decoupled(new RegfileWriteDataIO) // Producer OUT-IO: Decoupled(valid,bits,ready)

        // The physical DTCM has two independent row ports.  Matrix owns one;
        // Vector, scalar memory and DMA share the other at the top level.
        val dtcmMatrix = new DtcmPortIO(p)
        val dtcmAux    = new DtcmPortIO(p)

        // Ready admits one more command into the corresponding snapshot queue;
        // idle means both the queue and the execution unit have drained.
        val matrixReady = Output(Bool())
        val vectorReady = Output(Bool())
        val dmaReady    = Output(Bool())
        // Blocking LW/SW may dispatch: MemUnit idle AND no earlier range token
        // conflicts with the op's 1-row access (V1; see the scalar memory path).
        val memReady    = Output(Bool())
        val dependencyWait = Output(Bool())
        // Attribution split of dependencyWait for profiling: which queue's head
        // is the one blocked on range tokens. DMA-side waits are not attributed.
        val depWaitM = Output(Bool())
        val depWaitV = Output(Bool())
        val matrixIdle = Output(Bool())
        val vectorIdle = Output(Bool())
        val vectorDone = Output(Bool())   // 1-cycle pulse: a vector op completed

        // Vector reductions write GPRs after their asynchronous DTCM work
        // completes.  This bypasses the scalar EX/MEM/WB path.
        val vectorWrite = Decoupled(new RegfileWriteDataIO)
        val memIdle    = Output(Bool())   // scalar mem unit is idle
        val memDone    = Output(Bool())   // 1-cycle pulse: a scalar LW/SW completed

        // Core-issued DMA: a 1-cycle start pulse to the (top-level) AXI-master DMA
        // engine with its descriptor; dmaIdle reports the engine's descriptor FIFO
        // has room (engine idleness is no longer required -- a following
        // descriptor may stage behind the executing one).
        val dmaStart = Output(Bool())
        val dmaDir   = Output(Bool())            // 0 = load (DRAM->DTCM), 1 = store
        val dmaDram  = Output(UInt(p.AddrBits.W))
        val dmaDtcm  = Output(UInt(p.AddrBits.W))
        val dmaRows  = Output(UInt(16.W))
        val dmaIdle  = Input(Bool())
        // Lease tag carried by this descriptor; the engine echoes it on the
        // matching completion pulse (dmaDone/dmaDoneTok), and dmaDrained reports
        // that no leased core descriptor is still outstanding.
        val dmaTok     = Output(new DmaTokenTag(log2Ceil(p.TokenSlots)))
        val dmaDone    = Input(Bool())
        val dmaDoneTok = Input(new DmaTokenTag(log2Ceil(p.TokenSlots)))
        val dmaDrained = Output(Bool())

        // PERF probe: per-int8-engine busy, readable at PERF 0x64/0x68/0x6C.
        val matEngBusy = Output(Vec(2, Bool()))
        // PERF split (queue vs unit): vectorIdle above is "queue empty AND
        // unit ready" -- the old PERF vector count therefore read as a unit
        // utilization while actually measuring queue occupancy.  These two
        // expose the two halves (0x70/0x74); matEngBusy(0)||matEngBusy(1)
        // gives the matrix side's true execution rate.
        val vectorUnitBusy = Output(Bool())
        val vectorQueued   = Output(Bool())
        // PERF solo-engine attribution (0x7C/0x80, dual builds): exactly one
        // int8 engine busy -- split by whether the idle engine's next chain
        // is even enqueued (a software cadence gap) or sits issue-blocked.
        val engSoloNoWork  = Output(Bool())
        val engSoloBlocked = Output(Bool())
        // PERF scalar-EX stall split (0x84/0x88): the instruction in EX is a
        // matrix/vector command whose enqueue readiness is false.  These are
        // the only scalar stalls no other counter sees (dmawait/memwait have
        // theirs; dependencyWait is a queue-head metric, not an EX metric).
        val scStallVec = Output(Bool())
        val scStallMat = Output(Bool())
    })

    val isMatrixOp = io.req.bits.op.isOneOf(
        AluOp.LMA, AluOp.LMB, AluOp.MATMUL, AluOp.MATMULA, AluOp.SM, AluOp.MQCFG,
        AluOp.MCFG2,
        AluOp.FP_MATMUL, AluOp.FP_MATMULA, AluOp.FP_SM)
    val isVectorOp = io.req.bits.op.isOneOf(
        AluOp.VSETA, AluOp.VSETB, AluOp.VSETD, AluOp.VSETLEN, AluOp.VSETS,
        AluOp.VADD, AluOp.VSUB, AluOp.VMUL, AluOp.VFMA, AluOp.VSMUL, AluOp.VSADD,
        AluOp.VEXP, AluOp.VRECIP, AluOp.VRSQRT, AluOp.VSUM, AluOp.VMAX, AluOp.VDOT,
        AluOp.VSOFTEXP, AluOp.VABS, AluOp.VCVT_I8, AluOp.VCVT_BF16)
    val isMemOp = io.req.bits.op.isOneOf(AluOp.LW, AluOp.SW)
    val isDmaOp = io.req.bits.op.isOneOf(AluOp.DMALD, AluOp.DMAST)

    // ---- Matrix path ---------------------------------------------------------
    // A directed token table records queued and active background DTCM
    // commands.  Unlike an active-only lease, this also protects a future SM or
    // Vector write already sitting in its queue: a later consumer cannot start
    // before that producer has actually completed.  Widened from 16 so the three
    // allocators (matrix/vector/DMA) keep enough queued producers in flight;
    // QueuedAluCmd.tokenId must stay wide enough for this value.
    // (A 2026-10-30 experiment at 64 slots + iq/vq 48 measured ZERO counter
    // changes on the real-shape replay -- the "32 caps the dependency ring"
    // hypothesis was falsified, so the table stays at 32.)
    val TokenSlots = p.TokenSlots
    def clearAccess(access: DtcmAccess): Unit = {
        for (range <- access.read) {
            range.valid := false.B
            range.base := 0.U
            range.rows := 0.U
        }
        access.write.valid := false.B
        access.write.base := 0.U
        access.write.rows := 0.U
    }
    // An empty (zero-row) interval can never conflict with anything, so it must
    // not consume a token slot: a command whose unit retires it in the very cycle
    // it starts (a zero-length Vector op) would otherwise leak the slot forever.
    def rangeActive(range: DtcmRange): Bool = range.valid && range.rows =/= 0.U
    def hasAccess(access: DtcmAccess): Bool =
        rangeActive(access.write) || access.read.map(rangeActive).reduce(_ || _)
    def overlaps(a: DtcmRange, b: DtcmRange): Bool = {
        val aEnd = a.base +& (a.rows << log2Ceil(p.DtcmRowBytes))
        val bEnd = b.base +& (b.rows << log2Ceil(p.DtcmRowBytes))
        a.valid && b.valid && a.base < bEnd && b.base < aEnd
    }
    // `later` depends on `earlier` for RAW, WAR and WAW. Read/read overlap
    // intentionally has no edge and can use the two DTCM ports concurrently.
    def dependsOn(later: DtcmAccess, earlier: DtcmAccess): Bool = {
        val readAfterWrite = later.read.map(r => overlaps(r, earlier.write)).reduce(_ || _)
        val writeAfterRead = earlier.read.map(r => overlaps(later.write, r)).reduce(_ || _)
        readAfterWrite || writeAfterRead || overlaps(later.write, earlier.write)
    }
    val tokens = RegInit(VecInit(Seq.fill(TokenSlots)(0.U.asTypeOf(new DtcmToken(p, TokenSlots)))))
    val TokenIdBits = log2Ceil(TokenSlots)

    // Slots are recycled, so a dependency bit only names a producer for as long
    // as that producer still owns the slot. Erase every retiring producer from
    // all dependency sets in the same cycle it retires; otherwise a later command
    // that happens to reuse the slot re-arms the stale bit, and if that command
    // sits *behind* the waiter in the same queue neither can ever issue.
    // Per-class matrix token triples (0=e0, 1=e1, 2=fp32): each engine
    // retires its own range token by done source. All classes merge into ONE
    // retire mask / ONE clearing loop below -- a second loop would race the
    // first on the same regs (last-connect-wins, the DMA-deadlock lesson).
    val matrixTokenId = Reg(Vec(3, UInt(TokenIdBits.W)))
    val vectorTokenId = Reg(UInt(TokenIdBits.W))
    // The DMA side keeps no id register: the engine echoes the descriptor's own
    // lease tag on completion (see the DMA section), so the retiring id comes
    // from io.dmaDoneTok. A shared register would be overwritten by a second
    // in-flight descriptor and leak the first slot forever.
    val matrixRetire  = Wire(Vec(3, Bool()))
    val vectorRetire  = Wire(Bool())
    val dmaRetire     = Wire(Bool())
    val tokenRetire =
        Mux(matrixRetire(0), UIntToOH(matrixTokenId(0), TokenSlots), 0.U(TokenSlots.W)) |
        Mux(matrixRetire(1), UIntToOH(matrixTokenId(1), TokenSlots), 0.U(TokenSlots.W)) |
        Mux(matrixRetire(2), UIntToOH(matrixTokenId(2), TokenSlots), 0.U(TokenSlots.W)) |
        Mux(vectorRetire, UIntToOH(vectorTokenId, TokenSlots), 0.U(TokenSlots.W)) |
        Mux(dmaRetire,    UIntToOH(io.dmaDoneTok.id, TokenSlots), 0.U(TokenSlots.W))
    for (i <- 0 until TokenSlots) {
        tokens(i).deps := tokens(i).deps & (~tokenRetire).asUInt
        when (tokenRetire(i)) { tokens(i).valid := false.B }
    }

    val tokenValidBits = VecInit(tokens.map(_.valid)).asUInt
    val tokenFreeBits = VecInit(tokens.map(token => !token.valid)).asUInt
    val tokenAvailable = tokenFreeBits.orR
    val freeTokenId = PriorityEncoder(tokenFreeBits)
    def tokenDeps(access: DtcmAccess): UInt =
        VecInit(tokens.map(token => token.valid && dependsOn(access, token.access))).asUInt
    // Dependency set recorded with a newly allocated token. A producer retiring
    // this very cycle is already satisfied, so it is dropped here rather than
    // being carried into the new slot's snapshot.
    def freshDeps(access: DtcmAccess): UInt = tokenDeps(access) & (~tokenRetire).asUInt
    def tokenReady(id: UInt): Bool = !(tokens(id).deps & tokenValidBits).orR
    def needsToken(access: DtcmAccess): Bool = hasAccess(access)

    // The queue keeps Matrix context ordered.  These shadows describe the
    // context after every accepted entry, so a queued MATMUL observes a queued
    // LMA/LMB even before MatrixUnit has executed that configuration command.
    // C5: one set per int8 engine. LMA/LMB are broadcasts -- they update every
    // engine's shadows (and the fp32 unit latches them itself), so the FP.*
    // token footprint reads engine 0's pair, which is by construction equal
    // to what every engine holds.
    val nInt8Eng = p.int8Engines
    val matrixAShadow = RegInit(VecInit(Seq.fill(nInt8Eng)(0.U(p.AddrBits.W))))
    val matrixBShadow = RegInit(VecInit(Seq.fill(nInt8Eng)(0.U(p.AddrBits.W))))
    val matrixQuantShadow =
        RegInit(VecInit(Seq.fill(nInt8Eng)(0.U.asTypeOf(new MatrixQuantConfig(p)))))
    // Per-engine mode as of the last accepted MCFG2 for that engine: bit0 =
    // dataflow (0=OS, 1=WS), bit1 = ws_keep. Like the other shadows this
    // describes the state a QUEUED MATMUL will observe, which is exactly what
    // the token footprint below must agree with (WS-keep declares no B rows).
    val matrixModeShadow = RegInit(VecInit(Seq.fill(nInt8Eng)(0.U(2.W))))
    // Engine the most recent MCFG2 selected: LMA/LMB broadcasts aside, this
    // is the engine MQCFG configures and workers snapshot into their entry.
    val engineSelShadow = RegInit(0.U(1.W))
    val accEng = engineSelShadow   // access/enqueue-time engine view
    // Static-index helpers: at nInt8Eng == 1 every dynamic index is 0, and a
    // width-1 dynamic index into a one-entry Vec only emits W004 noise.
    def aShadow(e: UInt) = if (nInt8Eng == 1) matrixAShadow(0) else matrixAShadow(e)
    def bShadow(e: UInt) = if (nInt8Eng == 1) matrixBShadow(0) else matrixBShadow(e)
    def qShadow(e: UInt) = if (nInt8Eng == 1) matrixQuantShadow(0) else matrixQuantShadow(e)
    def mShadow(e: UInt) = if (nInt8Eng == 1) matrixModeShadow(0) else matrixModeShadow(e)
    val matrixAccess = Wire(new DtcmAccess(p))
    clearAccess(matrixAccess)
    val matrixOperandRows = (p.MatrixDim * p.BeatsPerArrayRow).U(16.W)
    val matrixBf16Rows = (p.MatrixDim * p.Bf16Bits * p.MatrixDim /
        p.DtcmRowBits).U(16.W)
    val matrixStoreRows = (p.MatrixDim * p.MatrixDim / p.WordsPerRow).U(16.W)
    // FP.*-tier tile footprints: the self-describing FP.* instructions target
    // the FP tier whatever the snapshot says. bf16 8x8 (C6): operands are
    // 8 rows x 16 B = 8 DTCM rows each; the bf16 result tile is also 8 rows.
    val fp32OperandRows = (p.Bf16MatrixDim * p.Bf16MatrixDim * p.Bf16ElemBits
        / 8 / p.DtcmRowBytes).U(16.W)
    val fp32StoreRows = fp32OperandRows
    val fp32Insn = io.req.bits.op.isOneOf(AluOp.FP_MATMUL, AluOp.FP_MATMULA, AluOp.FP_SM)
    val wsKeepActive = mShadow(accEng)(0) && mShadow(accEng)(1)
    when (io.req.bits.op.isOneOf(AluOp.MATMUL, AluOp.MATMULA,
            AluOp.FP_MATMUL, AluOp.FP_MATMULA)) {
        matrixAccess.read(0).valid := true.B
        matrixAccess.read(0).base := Mux(fp32Insn, matrixAShadow(0), aShadow(accEng))
        matrixAccess.read(0).rows := Mux(fp32Insn, fp32OperandRows,
            Mux(qShadow(accEng).aBf16, matrixBf16Rows, matrixOperandRows))
        matrixAccess.read(1).valid := true.B
        matrixAccess.read(1).base := Mux(fp32Insn, matrixBShadow(0), bShadow(accEng))
        // WS with ws_keep: the weights are already resident in the array, the
        // engine reads no B rows at all (zero-row range = no token edge, same
        // semantics as a zero-length Vector op).
        matrixAccess.read(1).rows := Mux(fp32Insn, fp32OperandRows,
            Mux(wsKeepActive, 0.U, matrixOperandRows))
    }
    when (io.req.bits.op.isOneOf(AluOp.SM, AluOp.FP_SM)) {
        matrixAccess.write.valid := true.B
        matrixAccess.write.base := io.rs1.data
        matrixAccess.write.rows := Mux(fp32Insn, fp32StoreRows, matrixStoreRows)
    }

    val matmul = Matmul(p)
    // Hand-rolled 4-entry matrix ISSUE queue, kept COMPACTED: valid entries
    // always occupy slots 0 until PopCount(valid) in program order.  Unlike a
    // stock FIFO this scans every entry and dispatches the first one whose
    // range token is satisfied: a head blocked on a DTCM dependency no longer
    // stalls independent work queued behind it.
    //
    // The compaction invariant is what makes the readiness rules below mean
    // what they say: "below" is a lower slot index == earlier program order.
    // (First cut enqueued at the lowest FREE slot instead; issues free slots
    // out of program order, slot indices decoupled from arrival order, and a
    // config could end up ABOVE later work -- board G2 P2 k4: the engine
    // consumed k-tiles 0,2,3,SM then tile 1, breaking both LMA/LMB latching
    // and the MATMULA accumulator chain.  int8 stayed bitwise-clean only
    // because its MATMUL bases come from the enqueue-time context snapshot.)
    // Safety of out-of-order issue itself: per-entry token deps are
    // snapshotted at enqueue (freshDeps) regardless of position.
    // Config commands (LMA/LMB/MQCFG -- no DTCM range, they latch unit state)
    // do not use the scan: they issue only when nothing valid sits below them,
    // i.e. strictly in program order, so their latching semantics hold.
    // Dual-engine builds need room for BOTH engines' chains: at depth 4 the
    // queue drains to the SM before the next group's MCFG2/LMA can enqueue,
    // which forces the engines into lockstep (both=0 in the PERF engbusy
    // probe) even with cross-class issue and per-engine LMA routing.
    // Real-shape GEMM chains are ~11 commands (gs128: LMA+LMB+MATMUL+7xMATMULA
    // +SM +MCFG2), so 12 could not even hold one chain plus the next group's
    // head -- the scalar could never run ahead far enough to overlap the two
    // engines (real-shape replay: e0 48% / e1 57% busy, both=14%).  Single is
    // deepened for the same reason: one chain alone overflowed depth 4.
    val iqDepth = if (p.int8Engines >= 2) 24 else 16
    val iqValid = RegInit(VecInit(Seq.fill(iqDepth)(false.B)))
    val iqEntry = Reg(Vec(iqDepth, new QueuedAluCmd(p)))
    val iqFull  = iqValid(iqDepth - 1)   // compacted: top occupied <=> full
    val iqEmpty = !iqValid(0)            // compacted: bottom free <=> empty

    val matrixNeedsToken = needsToken(matrixAccess)
    val matrixCanEnqueue = !matrixNeedsToken || tokenAvailable
    val iqEnqFire = io.req.valid && isMatrixOp && matrixCanEnqueue && !iqFull

    val iqIn = Wire(new QueuedAluCmd(p))
    iqIn.cmd := io.req.bits
    iqIn.rs1 := io.rs1.data
    iqIn.rs2 := io.rs2.data
    // MCFG2 carries its own target in rs2(0); every other entry snapshots the
    // currently selected engine.
    iqIn.engine := Mux(io.req.bits.op === AluOp.MCFG2, io.rs2.data(0, 0), engineSelShadow)
    iqIn.matrixQuant := qShadow(accEng)
    iqIn.matrixContext.valid := io.req.bits.op.isOneOf(
        AluOp.MATMUL, AluOp.MATMULA, AluOp.FP_MATMUL, AluOp.FP_MATMULA)
    iqIn.matrixContext.aBase := Mux(fp32Insn, matrixAShadow(0), aShadow(accEng))
    iqIn.matrixContext.bBase := Mux(fp32Insn, matrixBShadow(0), bShadow(accEng))
    iqIn.vectorContext.valid := false.B
    iqIn.vectorContext.srcA := 0.U
    iqIn.vectorContext.srcB := 0.U
    iqIn.vectorContext.dst := 0.U
    iqIn.vectorContext.vlen := 0.U
    iqIn.vectorContext.scalar := 0.U
    iqIn.needsTok := matrixNeedsToken
    iqIn.tokenId := freeTokenId

    // Readiness.  Three things carry ordering, each with one job:
    //   tokenReady     -- data dependencies across units (DTCM range edges);
    //   belowSameUnit  -- order WITHIN one unit.  Configs (LMA/LMB/MQCFG/
    //     MCFG2) latch unit state, and MATMULA chains accumulate in float
    //     over disjoint DTCM ranges (the token table models no edge between
    //     k-tiles), so an entry may never pass an older entry of its own
    //     unit: issue order is the only thing carrying that sequence, and
    //     float accumulation is not commutative (board G2 P2 k4 caught the
    //     first cut without this term).  Across units nothing is shared --
    //     workers carry bases/quant as enqueue-time snapshots and configs
    //     latch only their own engine's state -- so cross-unit passing is
    //     free (a strict in-order fence serialized the two engines
    //     completely, 62ad4e2).
    //   unitReady(unit) -- the target unit can latch the command now.
    // Unit ids: 0/1 = int8 engines, 2 = fp32 tier.
    // Latent trap, documented: LMA/LMB read 16 DTCM rows but declare NO
    // range -- their source buffers are stable only by emitter discipline
    // (dmawait before chain enqueue).  An emitter that reloads a live
    // buffer must add range tokens or this ordering silently breaks.
    val iqReady = Wire(Vec(iqDepth, Bool()))
    def unitOf(e: QueuedAluCmd): UInt = Mux(e.cmd.op.isOneOf(
        AluOp.FP_MATMUL, AluOp.FP_MATMULA, AluOp.FP_SM), 2.U, e.engine)
    val unitBits = VecInit(iqEntry.map(unitOf))
    val allUnitsReady = matmul.io.unitReady.asUInt.andR
    for (i <- 0 until iqDepth) {
        val worker = iqEntry(i).needsTok
        val unit = unitBits(i)
        // same-unit live entry below i (any op, config included)
        val belowSameUnit = (0 until i)
            .map(j => iqValid(j) && unitBits(j) === unit).foldLeft(false.B)(_ || _)
        iqReady(i) := iqValid(i) && !belowSameUnit && matmul.io.unitReady(unit) &&
            (!worker || tokenReady(iqEntry(i).tokenId))
    }
    val anyReady = iqReady.asUInt.orR
    val issueOh  = Mux(anyReady, UIntToOH(PriorityEncoder(iqReady), iqDepth), 0.U)

    // Solo-engine attribution (PERF 0x7C/0x80): when exactly one int8 engine
    // runs, is the other one starved because its next chain has not been
    // enqueued yet, or because a queued entry is fence/token/unit blocked?
    // Meaningless in single-engine builds, so tied false there.
    if (p.int8Engines >= 2) {
        val e0b = matmul.io.engBusy(0); val e1b = matmul.io.engBusy(1)
        val idleUnit = Mux(e0b, 1.U, 0.U)
        def unitHasEntry(u: UInt): Bool = (0 until iqDepth)
            .map(i => iqValid(i) && unitBits(i) === u).foldLeft(false.B)(_ || _)
        io.engSoloNoWork  := (e0b =/= e1b) && !unitHasEntry(idleUnit)
        io.engSoloBlocked := (e0b =/= e1b) &&  unitHasEntry(idleUnit)
    } else {
        io.engSoloNoWork  := false.B
        io.engSoloBlocked := false.B
    }

    // One combined update keeps the queue compacted even when an issue and an
    // enqueue land on the same cycle: survivors squeeze down in order, the new
    // command lands behind them at PopCount(survivors).  (Two writes can never
    // fight over one slot: survivor destinations are 0 until keptCount-1, the
    // enqueue lands at keptCount.)
    val keptOh = iqValid.asUInt & ~Mux(matmul.io.req.fire, issueOh, 0.U(iqDepth.W))
    val nextValid = Wire(Vec(iqDepth, Bool()))
    val nextEntry = Wire(Vec(iqDepth, new QueuedAluCmd(p)))
    nextValid := VecInit(Seq.fill(iqDepth)(false.B))
    nextEntry := iqEntry   // slots past the new tail keep stale data; gated by valid
    val keptCount = PopCount(keptOh)
    // Index bits covering 0..iqDepth-1.  The old hard-coded (1,0) slice was
    // exact only for the 4-entry single-engine queue: in the 12-deep dual
    // queue it wrapped the destination mod 4, so an enqueue (or a
    // compaction move) onto a >=4-deep queue OVERWROTE a survivor instead of
    // appending, silently dropping the command.  A dropped SM's range token
    // then never retired and the kick wedged with the vector head waiting on
    // it (dual replay, 2026-10-28; firtool's W005 "width 2 too narrow for
    // Vec of size 12" had flagged exactly these lines all along).  Any
    // enqueue-timing change could trip it -- the VSET* queue bypass did.
    val iqIdxHi = log2Ceil(iqDepth) - 1
    for (i <- 0 until iqDepth) {
        when (keptOh(i)) {
            // survivors land below i, so dst <= i <= depth-1
            val dst = PopCount(keptOh & ((BigInt(1) << i) - 1).U(iqDepth.W))(iqIdxHi, 0)
            nextValid(dst) := true.B
            nextEntry(dst) := iqEntry(i)
        }
    }
    when (iqEnqFire) {
        // enqFire implies !iqFull, so keptCount <= depth-1
        nextValid(keptCount(iqIdxHi, 0)) := true.B
        nextEntry(keptCount(iqIdxHi, 0)) := iqIn
    }
    when (matmul.io.req.fire || iqEnqFire) {
        iqValid := nextValid
        iqEntry := nextEntry
    }

    // The token gate has to mask `valid`, not just the pop: Matmul's req.ready
    // is simply "idle", so a command left in the queue would still be latched
    // and executed -- and then executed a second time when the queue finally
    // does pop it.  Gating valid with anyReady (and popping on fire) keeps the
    // selected entry out of the unit until it may really run.
    matmul.io.req.valid := anyReady
    matmul.io.req.bits  := Mux1H(issueOh, iqEntry.map(_.cmd))
    matmul.io.rs1.valid := true.B
    matmul.io.rs1.data  := Mux1H(issueOh, iqEntry.map(_.rs1))
    matmul.io.rs2.valid := true.B
    matmul.io.rs2.data  := Mux1H(issueOh, iqEntry.map(_.rs2))
    matmul.io.quant   := Mux1H(issueOh, iqEntry.map(_.matrixQuant))
    matmul.io.context := Mux1H(issueOh, iqEntry.map(_.matrixContext))
    matmul.io.engine  := Mux1H(issueOh, iqEntry.map(_.engine))
    matmul.io.rd.ready := true.B
    io.matEngBusy := matmul.io.engBusy   // PERF probe (dual-overlap work)
    io.matrixReady := !iqFull && matrixCanEnqueue
    io.scStallMat := io.req.valid && isMatrixOp && !io.matrixReady
    // HALT drains the whole unit: every engine class must be idle.
    io.matrixIdle  := iqEmpty && allUnitsReady

    when (iqEnqFire && matrixNeedsToken) {
        tokens(freeTokenId).valid := true.B
        tokens(freeTokenId).access := matrixAccess
        tokens(freeTokenId).deps := freshDeps(matrixAccess)
    }
    val matrixStart = matmul.io.req.fire
    val matrixTokenBusy = RegInit(VecInit(Seq.fill(3)(false.B)))
    val issueWorker = Mux1H(issueOh, iqEntry.map(_.needsTok))
    val issueTokenId = Mux1H(issueOh, iqEntry.map(_.tokenId))
    // Class of the issuing command (2 = fp32 tier, else the entry's engine).
    val startUnit = Mux1H(issueOh, unitBits)
    for (k <- 0 until 3) {
        when (matrixStart && startUnit === k.U) {
            matrixTokenBusy(k) := issueWorker
            when (issueWorker) {
                matrixTokenId(k) := issueTokenId
            }
        }
        // Only the command that actually owns a slot may retire one: a
        // command with an empty range (LMA/LMB/MQCFG/MCFG2, or a zero-length
        // op) leaves matrixTokenId pointing at an already-recycled slot. The
        // !start term keeps a same-class done+start pair from retiring the
        // newcomer's token; other classes may retire freely this cycle.
        matrixRetire(k) := matmul.io.done(k) &&
            !(matrixStart && startUnit === k.U) && matrixTokenBusy(k)
        when (matrixRetire(k)) { matrixTokenBusy(k) := false.B }
    }
    when (iqEnqFire && io.req.bits.op === AluOp.LMA) {
        // Scoped to the selected engine (same convention as MQCFG above), so
        // dual builds can load per-engine tiles: MCFG2(engine=1) then LMA
        // retargets engine 1 only. Single-engine builds are unchanged
        // (accEng is constant 0). The fp32 tier latches its own base on the
        // LMA fire and is only reachable with the default engine select.
        matrixAShadow(accEng) := io.rs1.data
    }
    when (iqEnqFire && io.req.bits.op === AluOp.LMB) {
        matrixBShadow(accEng) := io.rs1.data
    }
    when (iqEnqFire && io.req.bits.op === AluOp.MQCFG) {
        qShadow(accEng).aBf16 := io.rs2.data(0)
        qShadow(accEng).enable := io.rs2.data(0)
        qShadow(accEng).scale := io.rs1.data
        qShadow(accEng).groupSize := io.rs2.data(31, 16)
        qShadow(accEng).groupIndex := io.rs2.data(15, 1)
    }
    when (iqEnqFire && io.req.bits.op === AluOp.MCFG2) {
        assert(io.rs2.data(0) < nInt8Eng.U,
            "Alu: MCFG2 engine select out of range for this build")
        engineSelShadow := io.rs2.data(0, 0)
        mShadow(io.rs2.data(0, 0)) := io.rs2.data(2, 1)
    }

    // ---- Vector path ---------------------------------------------------------
    // Vector configuration is likewise shadowed at enqueue time.  The compute
    // entry therefore carries ranges derived from the preceding VSET* commands,
    // not from mutable state when it eventually reaches VectorUnit.
    val vectorAShadow = RegInit(0.U(p.AddrBits.W))
    val vectorBShadow = RegInit(0.U(p.AddrBits.W))
    val vectorDShadow = RegInit(0.U(p.AddrBits.W))
    val vectorLenShadow = RegInit(0.U(16.W))
    val vectorSShadow = RegInit(0.U(32.W))
    val vectorRows = (vectorLenShadow + (p.Bf16LanesPerRow - 1).U) >>
        log2Ceil(p.Bf16LanesPerRow)
    def vectorUsesB(op: AluOp.Type): Bool =
        op.isOneOf(AluOp.VADD, AluOp.VSUB, AluOp.VMUL, AluOp.VFMA, AluOp.VDOT)
    def vectorWritesDst(op: AluOp.Type): Bool =
        op.isOneOf(AluOp.VADD, AluOp.VSUB, AluOp.VMUL, AluOp.VFMA, AluOp.VSMUL,
            AluOp.VSADD, AluOp.VEXP, AluOp.VRECIP, AluOp.VRSQRT, AluOp.VSOFTEXP,
            AluOp.VABS, AluOp.VCVT_I8, AluOp.VCVT_BF16)
    val vectorAccess = Wire(new DtcmAccess(p))
    clearAccess(vectorAccess)
    val vectorOp = io.req.bits.op
    val vectorCompute = isVectorOp && !vectorOp.isOneOf(
        AluOp.VSETA, AluOp.VSETB, AluOp.VSETD, AluOp.VSETLEN, AluOp.VSETS)
    when (vectorCompute) {
        vectorAccess.read(0).valid := true.B
        vectorAccess.read(0).base := vectorAShadow
        vectorAccess.read(0).rows := Mux(vectorOp === AluOp.VCVT_BF16,
            vectorRows << 1, vectorRows)
        when (vectorUsesB(vectorOp)) {
            vectorAccess.read(1).valid := true.B
            vectorAccess.read(1).base := vectorBShadow
            vectorAccess.read(1).rows := vectorRows
        }
        when (vectorOp === AluOp.VFMA) {
            vectorAccess.read(2).valid := true.B
            vectorAccess.read(2).base := vectorDShadow
            vectorAccess.read(2).rows := vectorRows
        }
        when (vectorWritesDst(vectorOp)) {
            vectorAccess.write.valid := true.B
            vectorAccess.write.base := vectorDShadow
            vectorAccess.write.rows := Mux(vectorOp === AluOp.VCVT_I8,
                (vectorRows + 1.U) >> 1, vectorRows)
        }
    }
    val vectorContext = Wire(new VectorContext(p))
    vectorContext.valid := vectorCompute
    vectorContext.srcA := vectorAShadow
    vectorContext.srcB := vectorBShadow
    vectorContext.dst := vectorDShadow
    vectorContext.vlen := vectorLenShadow
    vectorContext.scalar := vectorSShadow

    val vector = VectorUnit(p)
    // V2: compacted vector issue queue with scan-issue -- the structural twin
    // of the matrix iq above.  The stock FIFO could only offer its head, so a
    // head frozen on a range token idled the unit behind a full queue (41% of
    // kick cycles were head-token waits; 90.8% of queue-full cycles had the
    // unit idle in the 2026-10-27 whole-layer VCD).  As with the matrix iq,
    // out-of-program-order issue is safe under the range-token table alone:
    // disjoint ranges carry no edge, overlapping ranges do, and every entry
    // runs from its own enqueue-time context snapshot, so reordering is
    // invisible to software.
    // Depth 32. The "36 > 32 dependency ring" argument motivated a 48-deep
    // experiment; that round (48-deep queues + a 64-slot token table) measured
    // ZERO counter changes on the real-shape replay, so the ring is not capped
    // by queue depth and the depth stays 32.
    val vqDepth = 32
    val vqValid = RegInit(VecInit(Seq.fill(vqDepth)(false.B)))
    val vqEntry = Reg(Vec(vqDepth, new QueuedAluCmd(p)))
    val vqFull  = vqValid(vqDepth - 1)   // compacted: top occupied <=> full

    val vectorNeedsToken = needsToken(vectorAccess)
    val vectorCanEnqueue = !vectorNeedsToken || tokenAvailable
    // V1.1 VSET* bypass.  A queued VSET* was pure slot overhead: it carries no
    // DTCM access (no token), the shadow registers it updates live on the
    // ENQUEUE side (below) and feed the next compute op's context snapshot,
    // and the unit-side architectural registers it mutates are read by no one
    // on the queue path -- every queued compute op runs from its own snapshot
    // (vectorContext.valid).  In the 2026-10-27 whole-layer VCD, VSET* were
    // 74.8% of vector enqueues, and 90.8% of queue-full stall cycles had the
    // vector unit idle (head frozen on a token, config no-ops stacked behind
    // it, scalar stuck in EX).  Filtering them out multiplies effective depth
    // ~4x and removes that stall class, with zero visible behavior change:
    // DTCM traffic is untouched (VSET* never touched it) and program order
    // still governs both shadow updates (EX order) and snapshots (each
    // compute op's own enqueue).
    val vectorEnqFire = io.req.valid && vectorCompute && vectorCanEnqueue && !vqFull
    val vqIn = Wire(new QueuedAluCmd(p))
    vqIn.cmd := io.req.bits
    vqIn.rs1 := io.rs1.data
    vqIn.rs2 := io.rs2.data
    vqIn.engine := 0.U   // vector ops carry no engine class
    vqIn.matrixQuant.aBf16 := false.B
    vqIn.matrixQuant.enable := false.B
    vqIn.matrixQuant.scale := 0.U
    vqIn.matrixQuant.groupSize := 0.U
    vqIn.matrixQuant.groupIndex := 0.U
    vqIn.matrixContext.valid := false.B
    vqIn.matrixContext.aBase := 0.U
    vqIn.matrixContext.bBase := 0.U
    vqIn.vectorContext := vectorContext
    vqIn.needsTok := vectorNeedsToken
    vqIn.tokenId := freeTokenId

    // Same as the Matrix path: readiness carries the valid bit (stale entries
    // would otherwise shadow real work forever -- their zeroed/stale access
    // reads as tokenless, i.e. always ready), the token gate (a tokenless
    // entry -- a zero-length op -- is ready by definition), and issue fires
    // only when the unit accepts.  One issue port, first ready in program
    // order.
    val vqReady = VecInit(vqEntry.zipWithIndex.map { case (e, i) =>
        vqValid(i) && (!e.needsTok || tokenReady(e.tokenId)) })
    val anyVReady = vqReady.asUInt.orR
    val vIssueOh = Mux(anyVReady, UIntToOH(PriorityEncoder(vqReady), vqDepth), 0.U(vqDepth.W))
    vector.io.req.valid := anyVReady
    vector.io.req.bits  := Mux1H(vIssueOh, vqEntry.map(_.cmd))
    vector.io.rs1.valid := true.B
    vector.io.rs1.data  := Mux1H(vIssueOh, vqEntry.map(_.rs1))
    vector.io.rs2.valid := true.B
    vector.io.rs2.data  := Mux1H(vIssueOh, vqEntry.map(_.rs2))
    vector.io.context := Mux1H(vIssueOh, vqEntry.map(_.vectorContext))
    vector.io.rd.ready := io.vectorWrite.ready

    // Combined issue+enqueue compaction (same invariant as the matrix iq:
    // survivors squeeze down in order, the newcomer lands at keptCount, and
    // the two writers never fight over one slot).
    val vectorStart = vector.io.req.fire
    val vKeptOh = vqValid.asUInt & ~Mux(vectorStart, vIssueOh, 0.U(vqDepth.W))
    val vNextValid = Wire(Vec(vqDepth, Bool()))
    val vNextEntry = Wire(Vec(vqDepth, new QueuedAluCmd(p)))
    vNextValid := VecInit(Seq.fill(vqDepth)(false.B))
    vNextEntry := vqEntry
    val vKeptCount = PopCount(vKeptOh)
    // Full index width (log2Ceil(vqDepth)-1,0).  The dual-iq mod-4 wrap lesson:
    // a hard-coded low-bit slice silently dropped entries on deeper queues.
    val vqIdxHi = log2Ceil(vqDepth) - 1
    for (i <- 0 until vqDepth) {
        when (vKeptOh(i)) {
            // survivors land below i, so dst <= i <= depth-1
            val dst = PopCount(vKeptOh & ((BigInt(1) << i) - 1).U(vqDepth.W))(vqIdxHi, 0)
            vNextValid(dst) := true.B
            vNextEntry(dst) := vqEntry(i)
        }
    }
    when (vectorEnqFire) {
        vNextValid(vKeptCount(vqIdxHi, 0)) := true.B
        vNextEntry(vKeptCount(vqIdxHi, 0)) := vqIn
    }
    when (vectorStart || vectorEnqFire) {
        vqValid := vNextValid
        vqEntry := vNextEntry
    }

    // A bypassed VSET* is accepted the cycle it is presented: it needs no
    // queue slot and no token, so there is nothing to wait for.  (vectorReady
    // is only sampled by Control while EX holds a vector op.)
    io.vectorReady := !vectorCompute || (!vqFull && vectorCanEnqueue)
    io.scStallVec := io.req.valid && vectorCompute && !io.vectorReady
    io.vectorIdle := !vqValid.asUInt.orR && vector.io.req.ready
    io.vectorUnitBusy := vector.io.busy
    io.vectorQueued := vqValid.asUInt.orR
    io.vectorDone := vector.io.done
    io.vectorWrite.valid := vector.io.rd.valid
    io.vectorWrite.bits.addr := vector.io.rd.bits.addr
    io.vectorWrite.bits.data := vector.io.rd.bits.data

    when (vectorEnqFire && vectorNeedsToken) {
        tokens(freeTokenId).valid := true.B
        tokens(freeTokenId).access := vectorAccess
        tokens(freeTokenId).deps := freshDeps(vectorAccess)
    }
    val vIssueWorker = Mux1H(vIssueOh, vqEntry.map(_.needsTok))
    val vIssueTokenId = Mux1H(vIssueOh, vqEntry.map(_.tokenId))
    val vectorTokenBusy = RegInit(false.B)
    when (vectorStart) {
        vectorTokenBusy := vIssueWorker
        when (vIssueWorker) {
            vectorTokenId := vIssueTokenId
        }
    }
    // Back-to-back dispatch (VectorUnit req.ready on the sRun/allDone cycle)
    // makes done and the next start coincide: retire the OUTGOING token then,
    // from the still-current tokenId register (it updates only on the clock
    // edge), and re-arm tracking for the newcomer in the same cycle.  The
    // clear is !vectorStart so a retiring token cannot clobber the newcomer's
    // just-latched tracking state.
    vectorRetire := vector.io.done && vectorTokenBusy
    when (vectorRetire && !vectorStart) { vectorTokenBusy := false.B }
    // VSET* shadows latch on the bypass cycle (they no longer enter the
    // queue, so enq.fire only fires for compute ops now -- but a compute
    // enqueue needs no shadow case).  If some other stall source holds EX for
    // several cycles the latch repeats; EX is occupied by this op meanwhile,
    // so no compute op can snapshot in between, and the last write before EX
    // advances carries the latest forwarded rs1 -- the same value the queue
    // path latched at its fire cycle.
    val vsetBypass = io.req.valid && isVectorOp && !vectorCompute
    when (vsetBypass) {
        switch (io.req.bits.op) {
            is (AluOp.VSETA)   { vectorAShadow := io.rs1.data }
            is (AluOp.VSETB)   { vectorBShadow := io.rs1.data }
            is (AluOp.VSETD)   { vectorDShadow := io.rs1.data }
            is (AluOp.VSETLEN) { vectorLenShadow := io.rs1.data(15, 0) }
            is (AluOp.VSETS)   { vectorSShadow := io.rs1.data }
        }
    }

    // ---- Scalar memory path (LW/SW) ------------------------------------------
    // V1: dispatch on range tokens instead of a full drain. A blocking LW/SW
    // needs NO token slot of its own -- the core is held at EX until memDone,
    // so nothing can enqueue behind the access, and it is fully complete before
    // any later command snapshots its dependencies ("slot-less == complete" is
    // guaranteed by the core block, not timing luck). It only must not START
    // until every EARLIER command conflicting with its 1-row access
    // [rs1 & ~(rowBytes-1), +16) has retired: LW waits on earlier writers
    // (RAW), SW on earlier readers and writers (WAR+WAW). Unlike the old
    // matrixIdle&&vectorIdle gate this also orders LW/SW against DMA -- the
    // pre-V1 design never did, relying on software discipline alone.
    val mem = MemUnit(p)
    val memAccess = Wire(new DtcmAccess(p))
    clearAccess(memAccess)
    when (io.req.bits.op === AluOp.LW) {
        memAccess.read(0).valid := true.B
        memAccess.read(0).base := io.rs1.data & ~((p.DtcmRowBytes - 1).U(p.AddrBits.W))
        memAccess.read(0).rows := 1.U
    } .elsewhen (io.req.bits.op === AluOp.SW) {
        memAccess.write.valid := true.B
        memAccess.write.base := io.rs1.data & ~((p.DtcmRowBytes - 1).U(p.AddrBits.W))
        memAccess.write.rows := 1.U
    }
    // Deliberately the unmasked dependency set, exactly like dmaDepsResolved:
    // masking in tokenRetire would close a combinational loop through
    // memReady -> Control blkDispatch -> req.valid. A token retiring this
    // cycle keeps the check false one extra cycle -- conservative, no leak.
    val memDepsResolved = !tokenDeps(memAccess).orR
    mem.io.req.valid := io.req.valid && isMemOp && memDepsResolved
    mem.io.req.bits  := io.req.bits
    mem.io.rs1 <> io.rs1
    mem.io.rs2 <> io.rs2
    mem.io.rd.ready := io.rd.ready
    io.memIdle := mem.io.req.ready
    io.memDone := mem.io.done
    io.memReady := mem.io.req.ready && memDepsResolved

    // ---- Core-issued DMA -----------------------------------------------------
    // Fire-and-forget: when a DMALD/DMAST fires (Control gates req.valid on the
    // engine being idle), emit a 1-cycle start pulse with the descriptor. rs1/rs2
    // arrive raw (Control does not add imm for DMA ops); imm carries the row count.
    // The engine itself touches DTCM via its own port at the Npu top level.
    val dmaAccess = Wire(new DtcmAccess(p))
    clearAccess(dmaAccess)
    when (io.req.bits.op === AluOp.DMALD) {
        dmaAccess.write.valid := true.B
        dmaAccess.write.base := io.rs2.data
        dmaAccess.write.rows := io.req.bits.imm(15, 0)
    }
    when (io.req.bits.op === AluOp.DMAST) {
        dmaAccess.read(0).valid := true.B
        dmaAccess.read(0).base := io.rs2.data
        dmaAccess.read(0).rows := io.req.bits.imm(15, 0)
    }
    val dmaNeedsToken = needsToken(dmaAccess)
    // Deliberately the unmasked dependency set. dmaRetire now comes from the
    // engine's completion echo (an input), so folding tokenRetire in here would
    // not form a loop; it is still left out to keep the issue path independent
    // of the completion path. The cost is that a DMA whose last producer retires
    // this cycle issues one cycle later.
    val dmaDepsResolved = !tokenDeps(dmaAccess).orR
    io.dmaReady := io.dmaIdle && (!dmaNeedsToken || (tokenAvailable && dmaDepsResolved))
    io.dmaStart := io.req.fire && isDmaOp
    io.dmaDir   := io.req.bits.op === AluOp.DMAST
    io.dmaDram  := io.rs1.data
    io.dmaDtcm  := io.rs2.data
    io.dmaRows  := io.req.bits.imm(15, 0)
    // The lease tag rides with the descriptor through the engine's FIFO and
    // comes back on the matching completion pulse. Retirement is a done-echo,
    // NOT engine idleness: with two descriptors staged, an idleness-triggered
    // retire would read a single shared id register that the second descriptor
    // had already overwritten, so the first slot never retires (the
    // DMA-deadlock class in docs/DMA死锁验证/).
    io.dmaTok.en := io.dmaStart && dmaNeedsToken
    io.dmaTok.id := freeTokenId
    dmaRetire := io.dmaDone && io.dmaDoneTok.en
    when (io.dmaStart && dmaNeedsToken) {
        tokens(freeTokenId).valid := true.B
        tokens(freeTokenId).access := dmaAccess
        tokens(freeTokenId).deps := freshDeps(dmaAccess)
    }
    // Drain = no core descriptor with a lease still outstanding, counted at
    // enqueue and at done-echo. The depth-2 FIFO plus the executing entry bound
    // it at 3, hence 2 bits. DMAWAIT/HALT gate on this instead of engine
    // idleness; zero-row descriptors carry no lease and complete in order at
    // their pop, and host descriptors never carry one.
    val dmaInFlight = RegInit(0.U(2.W))
    dmaInFlight := dmaInFlight +
        (io.dmaStart && dmaNeedsToken).asUInt - dmaRetire.asUInt
    io.dmaDrained := dmaInFlight === 0.U
    assert(!(dmaRetire && dmaInFlight === 0.U),
        "DMA lease retired with no core descriptor in flight")

    // Exposed for layer-level profiling. This reports a real token dependency
    // (rather than ordinary queue-full backpressure) and is sampled by Npu's
    // per-kick performance counters.
    // Dependency wait, post-scan semantics: the queue holds work but nothing in
    // it may issue.  While the unit is merely busy the ready entries are not
    // dependency-blocked and must not count.
    io.depWaitM := !iqEmpty && !anyReady
    io.depWaitV := vqValid.asUInt.orR && !anyVReady
    io.dependencyWait := io.depWaitM || io.depWaitV ||
        (isDmaOp && io.dmaIdle && !io.dmaReady)

    // ---- DTCM ports ----------------------------------------------------------
    // Matrix has a dedicated physical port. Vector and scalar memory share the
    // auxiliary port. Since V1 a scalar LW/SW dispatches on range tokens, so
    // MemUnit may hold a request while the vector unit still runs.  The arb
    // grants per cycle, vector first: the vector unit owns the port only on
    // cycles where it actually has a fetch/store row outstanding, and its
    // internal pipeline bubbles (compute-chain cycles with no port request --
    // 38.7% of its busy time in the real-shape VCD) now leak to the scalar
    // LW/SW stream instead of stalling it.  No whole-command lock is needed
    // for ordering: row tokens already serialize the two sides' DTCM ranges,
    // and MemUnit tolerates multi-cycle backpressure the same way the DMA
    // path does.
    io.dtcmMatrix.valid := matmul.io.dtcm.valid
    io.dtcmMatrix.write := matmul.io.dtcm.write
    io.dtcmMatrix.addr  := matmul.io.dtcm.addr
    io.dtcmMatrix.wdata := matmul.io.dtcm.wdata
    io.dtcmMatrix.wstrb := matmul.io.dtcm.wstrb
    matmul.io.dtcm.ready := io.dtcmMatrix.ready
    matmul.io.dtcm.rdata := io.dtcmMatrix.rdata

    val vecOwnsAux = vector.io.dtcm.valid
    def auxArb(v: UInt, me: UInt): UInt = Mux(vecOwnsAux, v, me)
    io.dtcmAux.valid := auxArb(vector.io.dtcm.valid, mem.io.dtcm.valid)
    io.dtcmAux.write := auxArb(vector.io.dtcm.write, mem.io.dtcm.write)
    io.dtcmAux.addr  := auxArb(vector.io.dtcm.addr,  mem.io.dtcm.addr)
    io.dtcmAux.wdata := auxArb(vector.io.dtcm.wdata, mem.io.dtcm.wdata)
    io.dtcmAux.wstrb := auxArb(vector.io.dtcm.wstrb, mem.io.dtcm.wstrb)
    vector.io.dtcm.ready := Mux(vecOwnsAux, io.dtcmAux.ready, false.B)
    vector.io.dtcm.rdata := io.dtcmAux.rdata
    mem.io.dtcm.ready := Mux(vecOwnsAux, false.B, io.dtcmAux.ready)
    mem.io.dtcm.rdata := io.dtcmAux.rdata
    // Both sides may REQUEST the aux port in the same cycle; the arb grants
    // it exclusively, vector first. The invariant worth asserting is grant
    // exclusivity.
    assert(!(mem.io.dtcm.ready && vecOwnsAux),
        "scalar memory was granted the aux port while the vector unit owns it")

    // ---- Scalar path (single-cycle, combinational) ---------------------------
    // The register forms take their second operand from rs2, the immediate forms
    // from req.imm. AUIPC/JAL/JALR produce their PC-relative result in Control
    // (which holds pc); here they fall through to 0 and Control overrides it.
    val a       = io.rs1.data
    val b       = io.rs2.data
    val imm     = io.req.bits.imm
    val shamtR  = b(4, 0)
    val shamtI  = imm(4, 0)
    val scalarResult = WireDefault(0.U(32.W))
    switch (io.req.bits.op) {
        is (AluOp.ADD)   { scalarResult := a + b }
        is (AluOp.SUB)   { scalarResult := a - b }
        is (AluOp.AND)   { scalarResult := a & b }
        is (AluOp.OR)    { scalarResult := a | b }
        is (AluOp.XOR)   { scalarResult := a ^ b }
        is (AluOp.SLL)   { scalarResult := (a << shamtR)(31, 0) }
        is (AluOp.SRL)   { scalarResult := a >> shamtR }
        is (AluOp.SRA)   { scalarResult := (a.asSInt >> shamtR).asUInt }
        is (AluOp.SLT)   { scalarResult := (a.asSInt < b.asSInt).asUInt }
        is (AluOp.SLTU)  { scalarResult := (a < b).asUInt }
        is (AluOp.ADDI)  { scalarResult := a + imm }
        is (AluOp.ANDI)  { scalarResult := a & imm }
        is (AluOp.ORI)   { scalarResult := a | imm }
        is (AluOp.XORI)  { scalarResult := a ^ imm }
        is (AluOp.SLLI)  { scalarResult := (a << shamtI)(31, 0) }
        is (AluOp.SRLI)  { scalarResult := a >> shamtI }
        is (AluOp.SRAI)  { scalarResult := (a.asSInt >> shamtI).asUInt }
        is (AluOp.SLTI)  { scalarResult := (a.asSInt < imm.asSInt).asUInt }
        is (AluOp.SLTIU) { scalarResult := (a < imm).asUInt }
        is (AluOp.LUI)   { scalarResult := imm }
    }
    val scalarValid = io.req.valid && !isMatrixOp && !isVectorOp && !isMemOp && !isDmaOp

    // ---- Merge writeback / handshake -----------------------------------------
    io.req.ready := Mux(isMatrixOp, io.matrixReady,
                    Mux(isVectorOp, io.vectorReady,
                    Mux(isMemOp,    io.memReady,
                    Mux(isDmaOp,    io.dmaReady, true.B))))

    io.rd.valid := Mux(isMemOp, mem.io.rd.valid, scalarValid)
    io.rd.bits.addr := Mux(isMemOp, mem.io.rd.bits.addr, io.req.bits.addr)
    io.rd.bits.data := Mux(isMemOp, mem.io.rd.bits.data, scalarResult)
}
