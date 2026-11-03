// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


// In-order 5-stage pipelined control unit (IF / ID / EX / MEM / WB) with operand
// forwarding and asynchronous Matrix / Vector command queues.  Matrix and Vector
// may execute concurrently through the two DTCM ports. Scalar instructions can
// execute while either background engine is active.
//
//   IF  : present PC to ITCM (synchronous read; inst valid the next cycle).
//   ID  : decode + register-file read (the regfile bypasses a same-cycle write).
//   EX  : ALU compute (scalar, with forwarding) / dispatch to the matrix unit.
//   MEM : pass-through. This ISA has no scalar load/store; matrix DTCM traffic is
//         handled inside the background matrix unit, not in a pipeline stage.
//   WB  : write the register file.
//
// Hazards handled:
//   * RAW between scalar ALU ops -> forwarding (EX/MEM and MEM/WB into EX) plus
//     the regfile's same-cycle write bypass.
//   * RAW/WAW around asynchronous Vector reductions -> a GPR scoreboard.  A
//     dependent scalar or vector-config instruction holds in EX until the direct
//     Vector writeback can be forwarded or has reached the register file.
//   * Matrix/Vector/DMA DTCM hazards -> Alu range leases only issue commands
//     whose declared read/write intervals are compatible with active producers.
//     Scalar LW/SW join the same discipline (V1): they dispatch once their
//     single DTCM row is free of earlier producers/consumers, instead of
//     draining both units.
//   * HALT -> stop fetching, drain Matrix, Vector, scalar memory and DMA, then
//     let the scalar pipeline retire.
//
// The matrix unit's idle status is taken from io.matrixIdle (a dedicated signal
// from the Alu) rather than req.ready, because req.ready is opcode-gated and
// would read "ready" whenever EX holds a scalar/HALT instruction.
class Control(p: Parameters) extends Module {
    val io = IO(new Bundle {
        // Instruction memory (synchronous read: inst valid one cycle after addr).
        val itcmAddr = Output(UInt(p.AddrBits.W))
        val itcmInst = Input(UInt(32.W))

        // Register file.
        val rs1Addr = Output(UInt(5.W))
        val rs2Addr = Output(UInt(5.W))
        val rs1Data = Input(UInt(32.W))
        val rs2Data = Input(UInt(32.W))
        val wrValid = Output(Bool())
        val wrAddr  = Output(UInt(5.W))
        val wrData  = Output(UInt(32.W))

        // Execution unit.
        val req    = Decoupled(new AluCmd)
        val aluRs1 = Output(UInt(32.W))   // drives Alu.io.rs1.data
        val aluRs2 = Output(UInt(32.W))   // drives Alu.io.rs2.data
        val rd     = Flipped(Decoupled(new RegfileWriteDataIO))

        val matrixReady = Input(Bool())   // Matrix command queue can accept an op
        // PERF scalar-EX stall taps (0x8C/0x90/0x94): gprStall gates req.valid
        // low so no queue-side counter sees it; blkWait is the LW/SW completion
        // phase (after dispatch, before memDone); haltDrain is the post-HALT
        // unit-drain wait.  Together with memwait/dmawait/scStall* they close
        // the scalar's cycle account.
        val stallGpr   = Output(Bool())
        val stallBlk   = Output(Bool())
        val stallHalt  = Output(Bool())
        // PERF 0x98: cycles EX actually retires an instruction (~= the kick's
        // executed-instruction count; branches/etc. included).
        val scAdvance  = Output(Bool())
        // PERF cycle-account closure (0x9C/0xA0/0xA4): dmaIssueStall was the one
        // exStall component with no tap -- a DMALD/DMAST parked in EX behind the
        // DMA engine masqueraded as an unexplained bubble in every earlier
        // accounting.  bubEmpty is the true iv=0 bubble residue; bubHalting its
        // post-HALT (fetch-stopped) subset.
        val dmaIssueWait = Output(Bool())
        val bubEmpty     = Output(Bool())
        val bubHalting   = Output(Bool())
        val vectorReady = Input(Bool())   // Vector command queue can accept an op
        val matrixIdle = Input(Bool())    // Matrix queue and unit both drained
        val vectorIdle = Input(Bool())    // Vector queue and unit both drained
        val vectorDone = Input(Bool())    // retained as an execution status pulse
        val vectorWrite = Flipped(Decoupled(new RegfileWriteDataIO))
        val memIdle    = Input(Bool())    // scalar mem unit is idle
        val memDone    = Input(Bool())    // pulse: the dispatched scalar LW/SW finished
        val dmaDrained = Input(Bool())    // no leased core DMA descriptor outstanding
        val dmaReady   = Input(Bool())    // DMA idle and its declared DTCM range is safe
        // V1: the blocking LW/SW may dispatch -- MemUnit idle and no earlier
        // range token conflicts with its 1-row access (replaces the old
        // matrixIdle && vectorIdle full-drain precondition).
        val memReady   = Input(Bool())
        val dmaWaitStalled = Output(Bool())
        // Blocking scalar LW/SW sitting in EX waiting to dispatch (pre-V1: the
        // full-drain wait; PERF_MEM_WAIT at the top level).
        val memWaitStalled = Output(Bool())

        // Run control.
        val start = Input(Bool())    // pulse: launch from PC = 0
        val busy  = Output(Bool())   // actively executing a program
        val done  = Output(Bool())   // halted on a HALT instruction
    })

    // ---- Pipeline-register payloads ------------------------------------------
    class IdEx extends Bundle {
        val valid    = Bool()
        val op       = AluOp()
        val pc       = UInt(p.AddrBits.W)   // address of this instruction (for PC-relative ops)
        val rd       = UInt(5.W)
        val rs1      = UInt(5.W)
        val rs2      = UInt(5.W)
        val imm      = UInt(32.W)
        val isMatrix = Bool()
        val isVector = Bool()
        val isMem    = Bool()
        val isDma    = Bool()
        val isDmaWait = Bool()
        val isBranch = Bool()
        val branchPredTaken = Bool()
        val isJump   = Bool()
        val isJalr   = Bool()
        val isAuipc  = Bool()
        val writesRd = Bool()
        val halt     = Bool()
        val rs1Data  = UInt(32.W)
        val rs2Data  = UInt(32.W)
    }
    // Shared shape for the EX/MEM and MEM/WB registers (a scalar result on its
    // way to writeback, plus the HALT marker used to retire the program).
    class ExWb extends Bundle {
        val valid    = Bool()
        val writesRd = Bool()
        val rd       = UInt(5.W)
        val result   = UInt(32.W)
        val halt     = Bool()
    }

    // ---- Architectural / pipeline state --------------------------------------
    val sIdle :: sRun :: sHalt :: Nil = Enum(3)
    val state = RegInit(sIdle)

    val pc      = RegInit(0.U(p.AddrBits.W))  // address of the next new instruction
    val idPc    = RegInit(0.U(p.AddrBits.W))  // address of the instruction now in ID
    val idValid = RegInit(false.B)            // ID holds a real (non-bubble) inst
    val halting = RegInit(false.B)            // a HALT is in the pipe; stop fetching

    val idex  = RegInit(0.U.asTypeOf(new IdEx))
    val exmem = RegInit(0.U.asTypeOf(new ExWb))
    val memwb = RegInit(0.U.asTypeOf(new ExWb))

    // A set bit means one accepted Vector reduction will eventually overwrite
    // this GPR.  WAW is included because a later scalar producer must not commit
    // before the asynchronous producer that precedes it in program order.
    val gprPending = RegInit(VecInit(Seq.fill(32)(false.B)))

    // ---- ID stage ------------------------------------------------------------
    val dec = Decode(io.itcmInst)
    io.rs1Addr := dec.rs1
    io.rs2Addr := dec.rs2

    // Stop fetching once a HALT is seen so nothing runs past it.
    val stopFetch = halting || (idValid && dec.halt)

    // ---- EX stage: stall, forwarding, dispatch -------------------------------
    // Matrix and Vector leave EX when their snapshot queue accepts them.  Scalar
    // memory remains blocking because it returns a GPR result through the scalar
    // writeback pipe -- but since V1 it waits only for its DTCM row to be free
    // (io.memReady, Alu range tokens), not for both units to drain.
    val blkInEx     = idex.valid && idex.isMem
    val blkIssued   = RegInit(false.B)   // current EX blocking op handed to its unit
    val blkDoneReg  = RegInit(false.B)   // its completion observed
    val blkDispatch = blkInEx && io.memReady && !blkIssued

    val matrixStall = idex.valid && idex.isMatrix && !io.matrixReady
    val vectorStall = idex.valid && idex.isVector && !io.vectorReady
    val blkStall    = blkInEx && !blkDoneReg
    // DMALD/DMAST only wait for FIFO room plus their declared DTCM interval.
    // Range conflicts are tracked in Alu, which permits deterministic ping-pong
    // prefetch to overlap independent Matrix and Vector work.
    val dmaIssueStall = idex.valid && idex.isDma && !io.dmaReady
    val dmaWaitStall  = idex.valid && idex.isDmaWait && !io.dmaDrained
    io.dmaWaitStalled := dmaWaitStall
    // Barrier attribution: cycles the blocking op waits for its dispatch window
    // (blkIssued clears only when the op leaves EX, so this is exactly the
    // wait phase, never the unit-access phase).
    io.memWaitStalled := blkInEx && !blkIssued
    val haltStall   = idex.valid && idex.halt &&
                        !(io.matrixIdle && io.vectorIdle && io.memIdle && io.dmaDrained)

    val scalarWr = memwb.valid && memwb.writesRd
    io.vectorWrite.ready := !scalarWr
    val vectorWriteFire = io.vectorWrite.valid && io.vectorWrite.ready

    def pendingSource(src: UInt): Bool = {
        val completingNow = vectorWriteFire && io.vectorWrite.bits.addr === src
        src =/= 0.U && gprPending(src) && !completingNow
    }

    // rs1 is used by every regular scalar operation except the explicit
    // immediate/PC-only forms, by the Matrix base-setting/store commands, DMA,
    // and Vector configuration.  Vector compute/reduce commands use latched
    // vector state and therefore have no architectural GPR source.
    val rs1Unused = idex.op.isOneOf(
        AluOp.LUI, AluOp.AUIPC, AluOp.JAL, AluOp.MATMUL, AluOp.MATMULA,
        AluOp.FP_MATMUL, AluOp.FP_MATMULA, AluOp.MCFG2,
        AluOp.DMAWAIT, AluOp.VADD, AluOp.VSUB, AluOp.VMUL, AluOp.VFMA,
        AluOp.VSMUL, AluOp.VSADD, AluOp.VEXP, AluOp.VRECIP, AluOp.VRSQRT,
        AluOp.VSUM, AluOp.VMAX, AluOp.VDOT, AluOp.VSOFTEXP, AluOp.VABS,
        AluOp.VCVT_I8, AluOp.VCVT_BF16)
    val usesRs1 = idex.valid && !idex.halt && !rs1Unused
    val usesRs2 = idex.op.isOneOf(
        AluOp.ADD, AluOp.SUB, AluOp.AND, AluOp.OR, AluOp.XOR, AluOp.SLL,
        AluOp.SRL, AluOp.SRA, AluOp.SLT, AluOp.SLTU, AluOp.BEQ, AluOp.BNE,
        AluOp.BLT, AluOp.BGE, AluOp.BLTU, AluOp.BGEU, AluOp.SW,
        AluOp.DMALD, AluOp.DMAST, AluOp.MQCFG, AluOp.MCFG2)
    val gprStall = idex.valid && (
        (usesRs1 && pendingSource(idex.rs1)) ||
        (usesRs2 && pendingSource(idex.rs2)) ||
        (idex.writesRd && idex.rd =/= 0.U && pendingSource(idex.rd)))

    val exStall = matrixStall || vectorStall || blkStall || dmaIssueStall ||
                  dmaWaitStall || haltStall || gprStall

    // PERF stall taps -- placed after every stall val above exists.
    io.stallGpr  := gprStall
    io.stallBlk  := blkStall && blkIssued          // LW/SW completion phase only
    io.stallHalt := haltStall
    io.scAdvance := idex.valid && !exStall
    io.dmaIssueWait := dmaIssueStall
    io.bubEmpty   := !idex.valid && !exStall
    io.bubHalting := !idex.valid && !exStall && halting

    // Forward a source operand from the most recent in-flight producer (EX/MEM
    // first, then MEM/WB), else use the value read from the regfile in ID.
    def fwd(src: UInt, srcData: UInt): UInt = {
        val fromVector = vectorWriteFire && io.vectorWrite.bits.addr =/= 0.U &&
                         io.vectorWrite.bits.addr === src
        val fromMem = exmem.valid && exmem.writesRd && exmem.rd =/= 0.U && exmem.rd === src
        val fromWb  = memwb.valid && memwb.writesRd && memwb.rd =/= 0.U && memwb.rd === src
        Mux(fromVector, io.vectorWrite.bits.data,
            Mux(fromMem, exmem.result, Mux(fromWb, memwb.result, srcData)))
    }

    val fwdRs1 = fwd(idex.rs1, idex.rs1Data)
    val fwdRs2 = fwd(idex.rs2, idex.rs2Data)

    val matrixDispatch = idex.valid && idex.isMatrix
    val vectorDispatch = idex.valid && idex.isVector
    io.req.valid     := !gprStall && (matrixDispatch || vectorDispatch ||
                        (idex.valid && idex.isDma && io.dmaReady) || blkDispatch)
    io.req.bits.op   := idex.op
    io.req.bits.addr := idex.rd
    io.req.bits.imm  := idex.imm
    // Matrix ops and scalar LW/SW take their DTCM base from GPR[rs1] + signed
    // offset (imm); scalar ALU and vector-config ops take rs1 as-is.
    io.aluRs1 := Mux((idex.isMatrix && !idex.op.isOneOf(AluOp.MQCFG, AluOp.MCFG2)) ||
        idex.isMem, fwdRs1 + idex.imm, fwdRs1)
    io.aluRs2 := fwdRs2

    // ---- Branch / jump resolution (EX) ---------------------------------------
    // Conditions use forwarded operands; targets are PC-relative (JALR uses rs1).
    val brEq  = fwdRs1 === fwdRs2
    val brLt  = fwdRs1.asSInt < fwdRs2.asSInt
    val brLtu = fwdRs1 < fwdRs2
    val brCond = MuxLookup(idex.op.asUInt, false.B)(Seq(
        AluOp.BEQ.asUInt  -> brEq,
        AluOp.BNE.asUInt  -> !brEq,
        AluOp.BLT.asUInt  -> brLt,
        AluOp.BGE.asUInt  -> !brLt,
        AluOp.BLTU.asUInt -> brLtu,
        AluOp.BGEU.asUInt -> !brLtu,
    ))
    val branchTaken = idex.valid && idex.isBranch && brCond
    val doJump      = idex.valid && idex.isJump
    val takenTarget = Mux(idex.isJalr, (fwdRs1 + idex.imm) & ~1.U(32.W),
                          idex.pc + idex.imm)
    val branchTarget = Mux(branchTaken, takenTarget, idex.pc + 4.U)
    val branchMispredict = idex.valid && idex.isBranch &&
        (branchTaken =/= idex.branchPredTaken)
    val brTarget    = Mux(idex.isBranch, branchTarget, takenTarget)
    val redirect    = (branchMispredict || doJump) && !exStall

    // Static BTFNT: loop-closing conditional branches are predicted taken,
    // forward branches are predicted not taken. The predictor is intentionally
    // stateless; it adds no table RAM and keeps deterministic layer kernels
    // (tile/group/ping-pong loops) on the useful side of the trade-off.
    val idBranchPredTaken = idValid && dec.isBranch && dec.imm.asSInt < 0.S
    val idBranchPredTarget = idPc + dec.imm

    // PC-relative writeback results: JAL/JALR write the link (pc+4), AUIPC writes
    // pc+imm; everything else uses the Alu's scalar result.
    val exResult = Mux(idex.isJump,  idex.pc + 4.U,
                   Mux(idex.isAuipc, idex.pc + idex.imm, io.rd.bits.data))

    // Track the scalar-memory blocking op.  Matrix and Vector commands are now
    // queued snapshots, so their execution continues after EX advances.
    val blkFiring = io.req.fire && idex.isMem
    val blkDone   = io.memDone
    when (blkFiring) { blkIssued := true.B }
    when ((blkIssued || blkFiring) && blkDone) { blkDoneReg := true.B }
    when (blkInEx && !exStall) { blkIssued := false.B; blkDoneReg := false.B }

    // The Alu drives rd.bits.data combinationally with the scalar result
    // (consumed via exResult above, which also handles PC-relative writebacks).
    io.rd.ready := true.B

    // ---- WB stage ------------------------------------------------------------
    io.wrValid := scalarWr || io.vectorWrite.valid
    io.wrAddr  := Mux(scalarWr, memwb.rd, io.vectorWrite.bits.addr)
    io.wrData  := Mux(scalarWr, memwb.result, io.vectorWrite.bits.data)

    io.busy := state === sRun
    io.done := state === sHalt

    // ---- Fetch address -------------------------------------------------------
    // While stalled, re-present the ID instruction's address so it holds in ID
    // (ITCM is stateless); otherwise present the running fetch pointer.
    io.itcmAddr := Mux(exStall, idPc, pc)

    // ---- Sequencing ----------------------------------------------------------
    def launch(): Unit = {
        pc          := 0.U
        idPc        := 0.U
        idValid     := false.B
        halting     := false.B
        idex.valid  := false.B
        exmem.valid := false.B
        memwb.valid := false.B
        blkIssued   := false.B
        blkDoneReg  := false.B
        for (i <- 0 until 32) { gprPending(i) := false.B }
        state       := sRun
    }

    // Latch HALT only for an in-flight (non-squashed) instruction: a wrong-path
    // sentinel behind a taken branch/jump must not stop the front-end.
    when (idValid && dec.halt && !redirect) { halting := true.B }

    when (state === sRun) {
        val vectorReductionFiring = io.req.fire && idex.isVector && idex.writesRd && idex.rd =/= 0.U
        for (i <- 1 until 32) {
            when (vectorWriteFire && io.vectorWrite.bits.addr === i.U) {
                gprPending(i) := false.B
            }
            when (vectorReductionFiring && idex.rd === i.U) {
                gprPending(i) := true.B
            }
        }

        // A bypass is only present for one cycle, but the consumer can remain in
        // EX for longer for an unrelated reason.  In particular, a SW following a
        // Vector reduction sees the reduction writeback while it is still waiting
        // for vectorIdle; by the time MemUnit can accept the store, the direct
        // Vector bypass is gone.  Preserve the fully-forwarded operands in the
        // held EX payload so every later dispatch observes the same values.
        when (exStall && idex.valid) {
            idex.rs1Data := fwdRs1
            idex.rs2Data := fwdRs2
        }

        // Front-end + IF/ID advance (frozen while EX stalls). A taken branch/jump
        // in EX redirects the fetch pointer and squashes the instruction being
        // fetched this cycle (it is wrong-path).
        when (!exStall) {
            when (redirect) {
                pc      := brTarget
                idValid := false.B        // squash the wrong-path fetch in flight
            } .elsewhen (idBranchPredTaken) {
                // The sequential successor has already been presented to the
                // synchronous ITCM this cycle. Mark its response invalid, then
                // request the predicted target for the following cycle.
                pc      := idBranchPredTarget
                idValid := false.B
            } .otherwise {
                when (!stopFetch) { pc := pc + 4.U }
                idValid := !stopFetch     // post-HALT fetches become bubbles
            }
            idPc := io.itcmAddr           // = pc here; addr of the inst entering ID
        }

        // ID -> EX (only when EX can accept, i.e. not stalled). On a redirect the
        // instruction now in ID is the branch's wrong-path successor -> squash it.
        when (!exStall) {
            idex.valid    := idValid && !redirect
            idex.op       := dec.op
            idex.pc       := idPc
            idex.rd       := dec.rd
            idex.rs1      := dec.rs1
            idex.rs2      := dec.rs2
            idex.imm      := dec.imm
            idex.isMatrix := dec.isMatrix
            idex.isVector := dec.isVector
            idex.isMem    := dec.isMem
            idex.isDma    := dec.isDma
            idex.isDmaWait := dec.isDmaWait
            idex.isBranch := dec.isBranch
            idex.branchPredTaken := idBranchPredTaken
            idex.isJump   := dec.isJump
            idex.isJalr   := dec.isJalr
            idex.isAuipc  := dec.isAuipc
            idex.writesRd := dec.writesRd
            idex.halt     := dec.halt
            idex.rs1Data  := io.rs1Data
            idex.rs2Data  := io.rs2Data
        }

        // EX -> MEM. Matrix and Vector commands retire here without a scalar
        // writeback; Vector reductions use the direct asynchronous port above.
        when (!exStall) {
            exmem.valid    := idex.valid
            exmem.writesRd := idex.valid && idex.writesRd && !idex.isMatrix && !idex.isVector
            exmem.rd       := idex.rd
            exmem.result   := exResult
            exmem.halt     := idex.valid && idex.halt
        } .otherwise {
            exmem.valid    := false.B
            exmem.writesRd := false.B
            exmem.halt     := false.B
        }

        // MEM -> WB (never stalls).
        memwb := exmem

        // Retire HALT once it reaches WB: every older instruction has already
        // written back and the matrix unit drained in EX, so results are final.
        when (memwb.valid && memwb.halt) {
            state := sHalt
        }
    } .otherwise {
        // sIdle / sHalt: pipeline held empty; (re)launch on a start pulse.
        when (io.start) { launch() }
    }
}
