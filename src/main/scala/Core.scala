// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


object Core {
    def apply(p: Parameters, useDualDtcm: Boolean = false): Core =
        Module(new Core(p, useDualDtcm))
}

// Minimal in-order core: Control fetches instructions from an external ITCM
// (io.itcm) and drives the register file and the Alu/Matmul execution unit.
// The legacy combined DTCM port keeps direct-Core unit tests simple.  Npu uses
// the two physical ports, with Matrix on dtcmMatrix and Vector/Mem on dtcmAux.
// The selection is compile-time so an unused interface has no hardware effect.
class Core(p: Parameters, useDualDtcm: Boolean = false) extends Module {
    val io = IO(new Bundle {
        // Instruction memory port (synchronous read: inst valid one cycle later).
        val itcmAddr = Output(UInt(p.AddrBits.W))
        val itcmInst = Input(UInt(32.W))

        // Compatibility single-port DTCM view used by standalone Core tests.
        val dtcm = new DtcmPortIO(p)

        // True dual-port DTCM views used by Npu.
        val dtcmMatrix = new DtcmPortIO(p)
        val dtcmAux    = new DtcmPortIO(p)

        // Core-issued DMA: descriptor + start pulse out to the top-level AXI-master
        // DMA engine; dmaIdle reports its descriptor FIFO has room (a following
        // descriptor may stage behind the executing one). The lease tag rides
        // with the descriptor and returns on dmaDone/dmaDoneTok; dmaDrained
        // reports that no leased descriptor is outstanding (DMAWAIT/HALT gate).
        val dmaStart = Output(Bool())
        val dmaDir   = Output(Bool())
        val dmaDram  = Output(UInt(p.AddrBits.W))
        val dmaDtcm  = Output(UInt(p.AddrBits.W))
        val dmaRows  = Output(UInt(16.W))
        val dmaIdle  = Input(Bool())
        val dmaTok   = Output(new DmaTokenTag(log2Ceil(p.TokenSlots)))
        val dmaDone  = Input(Bool())
        val dmaDoneTok = Input(new DmaTokenTag(log2Ceil(p.TokenSlots)))
        val dmaDrained = Output(Bool())

        // Run control / status.
        val start = Input(Bool())
        val busy  = Output(Bool())
        val done  = Output(Bool())
        val dbg   = Output(Vec(32, UInt(32.W)))
        // Per-kick profiling taps consumed by the Npu control register block.
        val matrixBusy = Output(Bool())
        val vectorBusy = Output(Bool())
        val dependencyWait = Output(Bool())
        val depWaitM = Output(Bool())
        val depWaitV = Output(Bool())
        val dmaWaitStalled = Output(Bool())
        // Blocking scalar LW/SW in EX waiting to dispatch (barrier attribution).
        val memWaitStalled = Output(Bool())
        // PERF probe: per-int8-engine busy, readable at PERF 0x64/0x68/0x6C.
        val matEngBusy = Output(Vec(2, Bool()))
        // PERF split: true vector unit execution vs queue occupancy
        // (readable at PERF 0x70/0x74) -- vectorBusy above is the OR of both.
        val vectorUnitBusy = Output(Bool())
        val vectorQueued = Output(Bool())
        // PERF solo-engine attribution, dual builds (0x7C/0x80).
        val engSoloNoWork = Output(Bool())
        val engSoloBlocked = Output(Bool())
        // PERF scalar-EX stall split (0x84/0x88).
        val scStallVec = Output(Bool())
        val scStallMat = Output(Bool())
        // PERF scalar-EX stall taps (0x8C/0x90/0x94).
        val stallGpr = Output(Bool())
        val stallBlk = Output(Bool())
        val stallHalt = Output(Bool())
        val scAdvance = Output(Bool())
        // PERF cycle-account closure (0x9C/0xA0/0xA4).
        val dmaIssueWait = Output(Bool())
        val bubEmpty = Output(Bool())
        val bubHalting = Output(Bool())
    })

    val ctrl = Module(new Control(p))
    val rf   = Module(new Regfile(p))
    val alu  = Alu(p)

    // ITCM (external).
    io.itcmAddr := ctrl.io.itcmAddr
    ctrl.io.itcmInst := io.itcmInst

    // Register file.
    rf.io.rs1Addr := ctrl.io.rs1Addr
    rf.io.rs2Addr := ctrl.io.rs2Addr
    ctrl.io.rs1Data := rf.io.rs1.data
    ctrl.io.rs2Data := rf.io.rs2.data
    rf.io.wrValid := ctrl.io.wrValid
    rf.io.wrAddr  := ctrl.io.wrAddr
    rf.io.wrData  := ctrl.io.wrData

    // Execution unit.
    alu.io.req <> ctrl.io.req
    alu.io.rs1.valid := true.B
    alu.io.rs1.data  := ctrl.io.aluRs1
    alu.io.rs2.valid := true.B
    alu.io.rs2.data  := ctrl.io.aluRs2
    ctrl.io.rd <> alu.io.rd
    ctrl.io.vectorWrite <> alu.io.vectorWrite
    ctrl.io.matrixReady := alu.io.matrixReady
    ctrl.io.vectorReady := alu.io.vectorReady
    ctrl.io.matrixIdle := alu.io.matrixIdle
    ctrl.io.vectorIdle := alu.io.vectorIdle
    ctrl.io.vectorDone := alu.io.vectorDone
    ctrl.io.memIdle    := alu.io.memIdle
    ctrl.io.memDone    := alu.io.memDone
    ctrl.io.dmaDrained := alu.io.dmaDrained
    ctrl.io.dmaReady   := alu.io.dmaReady
    ctrl.io.memReady   := alu.io.memReady

    // Core-issued DMA: Alu emits the descriptor + start pulse; the engine lives at
    // the Npu top. Alu consumes dmaIdle (FIFO room) for req.ready; Control
    // consumes dmaDrained (Alu's lease count) for DMAWAIT/HALT.
    alu.io.dmaIdle := io.dmaIdle
    io.dmaStart    := alu.io.dmaStart
    io.dmaDir      := alu.io.dmaDir
    io.dmaDram     := alu.io.dmaDram
    io.dmaDtcm     := alu.io.dmaDtcm
    io.dmaRows     := alu.io.dmaRows
    io.dmaTok      := alu.io.dmaTok
    alu.io.dmaDone    := io.dmaDone
    alu.io.dmaDoneTok := io.dmaDoneTok
    io.dmaDrained     := alu.io.dmaDrained

    // Memory + run control / status.  A direct Core test may continue to use a
    // single memory model; the Npu top selects the independent physical ports.
    def disable(port: DtcmPortIO): Unit = {
        port.valid := false.B
        port.write := false.B
        port.addr  := 0.U
        port.wdata := 0.U
        port.wstrb := 0.U
    }
    disable(io.dtcm)
    disable(io.dtcmMatrix)
    disable(io.dtcmAux)
    alu.io.dtcmMatrix.ready := false.B
    alu.io.dtcmMatrix.rdata := 0.U
    alu.io.dtcmAux.ready := false.B
    alu.io.dtcmAux.rdata := 0.U

    if (useDualDtcm) {
        io.dtcmMatrix.valid := alu.io.dtcmMatrix.valid
        io.dtcmMatrix.write := alu.io.dtcmMatrix.write
        io.dtcmMatrix.addr  := alu.io.dtcmMatrix.addr
        io.dtcmMatrix.wdata := alu.io.dtcmMatrix.wdata
        io.dtcmMatrix.wstrb := alu.io.dtcmMatrix.wstrb
        alu.io.dtcmMatrix.ready := io.dtcmMatrix.ready
        alu.io.dtcmMatrix.rdata := io.dtcmMatrix.rdata

        io.dtcmAux.valid := alu.io.dtcmAux.valid
        io.dtcmAux.write := alu.io.dtcmAux.write
        io.dtcmAux.addr  := alu.io.dtcmAux.addr
        io.dtcmAux.wdata := alu.io.dtcmAux.wdata
        io.dtcmAux.wstrb := alu.io.dtcmAux.wstrb
        alu.io.dtcmAux.ready := io.dtcmAux.ready
        alu.io.dtcmAux.rdata := io.dtcmAux.rdata
    } else {
        // Matrix wins a legacy shared-port cycle. The loser is back-pressured,
        // preserving the ready/rdata contract used by the two real ports.
        val useMatrix = alu.io.dtcmMatrix.valid
        io.dtcm.valid := Mux(useMatrix, alu.io.dtcmMatrix.valid, alu.io.dtcmAux.valid)
        io.dtcm.write := Mux(useMatrix, alu.io.dtcmMatrix.write, alu.io.dtcmAux.write)
        io.dtcm.addr  := Mux(useMatrix, alu.io.dtcmMatrix.addr,  alu.io.dtcmAux.addr)
        io.dtcm.wdata := Mux(useMatrix, alu.io.dtcmMatrix.wdata, alu.io.dtcmAux.wdata)
        io.dtcm.wstrb := Mux(useMatrix, alu.io.dtcmMatrix.wstrb, alu.io.dtcmAux.wstrb)
        alu.io.dtcmMatrix.ready := io.dtcm.ready && useMatrix
        alu.io.dtcmMatrix.rdata := io.dtcm.rdata
        alu.io.dtcmAux.ready := io.dtcm.ready && !useMatrix
        alu.io.dtcmAux.rdata := io.dtcm.rdata
    }
    ctrl.io.start := io.start
    io.busy := ctrl.io.busy
    io.done := ctrl.io.done
    io.dbg  := rf.io.dbg
    io.matrixBusy := !alu.io.matrixIdle
    io.vectorBusy := !alu.io.vectorIdle
    io.dependencyWait := alu.io.dependencyWait
    io.depWaitM := alu.io.depWaitM
    io.depWaitV := alu.io.depWaitV
    io.dmaWaitStalled := ctrl.io.dmaWaitStalled
    io.memWaitStalled := ctrl.io.memWaitStalled
    io.matEngBusy := alu.io.matEngBusy   // PERF probe (dual-overlap work)
    io.vectorUnitBusy := alu.io.vectorUnitBusy  // PERF split (queue vs unit)
    io.vectorQueued := alu.io.vectorQueued      // PERF split (queue vs unit)
    io.engSoloNoWork := alu.io.engSoloNoWork    // PERF solo-engine attribution
    io.engSoloBlocked := alu.io.engSoloBlocked  // PERF solo-engine attribution
    io.scStallVec := alu.io.scStallVec          // PERF scalar stall split
    io.scStallMat := alu.io.scStallMat          // PERF scalar stall split
    io.stallGpr := ctrl.io.stallGpr             // PERF scalar stall taps
    io.stallBlk := ctrl.io.stallBlk
    io.stallHalt := ctrl.io.stallHalt
    io.scAdvance := ctrl.io.scAdvance
    io.dmaIssueWait := ctrl.io.dmaIssueWait     // PERF cycle-account closure
    io.bubEmpty := ctrl.io.bubEmpty
    io.bubHalting := ctrl.io.bubHalting
}
