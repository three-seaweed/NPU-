// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


object Npu {
    def apply(p: Parameters): Npu = Module(new Npu(p))
}

// PL top level: a single AXI4 slave port lets the Zynq PS (via its AXI DMA)
// load instructions into ITCM, load data into DTCM, launch the core, poll for
// completion, and read results back from DTCM.
//
// AXI byte-address map (addr[20:19] selects the region; DTCM needs a >=512KB
// window so the region select sits above DTCM's 19-bit intra-region address):
//   00  ITCM   0x00_0000  (instruction RAM, host writes)
//   01  DTCM   0x08_0000  (data RAM, host read/write; 512KB)
//   10  CTRL   0x10_0000  (control/status registers)
//        +0x00 CTRL          (W): bit0 start, bit1 clear-done
//        +0x04 STATUS        (R): bit0 done, bit1 busy
//        +0x10 DMA_DRAM_ADDR (W): DRAM byte base for the next DMA descriptor
//        +0x14 DMA_DTCM_ADDR (W): DTCM byte base for the next DMA descriptor
//        +0x18 DMA_LEN       (W): transfer length in bytes (DtcmRowBytes-aligned)
//        +0x1C DMA_CTRL      (W): bit0 go, bit1 dir (0=DRAM->DTCM, 1=DTCM->DRAM)
//        +0x20 DMA_STATUS    (R): bit0 dma_done, bit1 dma_busy
//        +0x24 PERF_MATRIX   (R): Matrix queue/unit busy cycles in this kick
//        +0x28 PERF_VECTOR   (R): Vector queue/unit busy cycles in this kick
//        +0x2C PERF_DMA      (R): DMA active cycles in this kick
//        +0x30 PERF_DTCM_M   (R): Matrix DTCM port transfers
//        +0x34 PERF_DTCM_A   (R): Aux DTCM port transfers (Vector/Mem/DMA)
//        +0x38 PERF_DEPWAIT  (R): range-token wait cycles
//        +0x3C PERF_DMAWAIT  (R): DMAWAIT instruction stall cycles
//        +0x40 PERF_KICKS    (R): accepted PS-to-PL start commands since reset
//        +0x44 PERF_CYCLES   (R): total busy cycles in this kick
//        +0x48 PERF_DMA_DTCM (R): DMA DTCM-side 16-byte transfers in this kick
//        +0x4C PERF_DEP_M    (R): cycles the Matrix queue head is token-blocked
//        +0x50 PERF_DEP_V    (R): cycles the Vector queue head is token-blocked
//        +0x54 PERF_MEM_WAIT (R): cycles a blocking scalar LW/SW waits to dispatch
//        +0x58 PERF_MEM_CLR  (R): ... of those, cycles matrix+vector are already idle
//        +0x5C PERF_BOTH_IDLE(R): kick cycles with matrix AND vector both idle
//        +0x60 PERF_DEP_V_SOLO(R): depwaitV cycles with matrix and DMA also idle
//
// A second AXI4 *master* port (M_AXI, see io.maxi) lets the on-chip DMA engine
// burst those weights/activations straight from PS DRAM into DTCM and results
// back out, instead of the PS pushing every word over the 32-bit slave.
//
// While the core is running (busy=1) it owns ITCM/DTCM; while a DMA runs
// (dma_busy=1) it owns DTCM. Host accesses to an owned region are back-pressured.
// The CTRL region is always accessible so the PS can poll STATUS / DMA_STATUS.
class Npu(p: Parameters) extends Module {
    val AxiAddrBits = 32
    val AxiIdBits   = 4

    val RegItcm = 0
    val RegDtcm = 1
    val RegCtrl = 2

    // CTRL-region word selects. Keep bit 6 as well: +0x40 and above must not
    // alias CTRL at +0x00 (the original addr[5:2] slice only covered +0x3c).
    val SelCtrl     = 0   // 0x00
    val SelStatus   = 1   // 0x04
    val SelDmaDram  = 4   // 0x10
    val SelDmaDtcm  = 5   // 0x14
    val SelDmaLen   = 6   // 0x18
    val SelDmaCtrl  = 7   // 0x1C
    val SelDmaStat  = 8   // 0x20
    val SelPerfMatrix = 9 // 0x24
    val SelPerfVector = 10 // 0x28
    val SelPerfDma = 11 // 0x2C
    val SelPerfDtcmMatrix = 12 // 0x30
    val SelPerfDtcmAux = 13 // 0x34
    val SelPerfDepWait = 14 // 0x38
    val SelPerfDmaWait = 15 // 0x3C
    val SelPerfKicks = 16 // 0x40
    val SelPerfCycles = 17 // 0x44
    val SelPerfDmaDtcm = 18 // 0x48
    val SelPerfDepM = 19 // 0x4C
    val SelPerfDepV = 20 // 0x50
    val SelPerfMemWait = 21 // 0x54
    val SelPerfMemClr = 22 // 0x58
    val SelPerfBothIdle = 23 // 0x5C
    val SelPerfDepVSolo = 24 // 0x60
    val SelPerfEng0 = 25     // 0x64 PERF probe (dual-overlap work)
    val SelPerfEng1 = 26     // 0x68 PERF probe (dual-overlap work)
    val SelPerfEngBoth = 27  // 0x6C PERF probe (dual-overlap work)
    val SelPerfVecUnit = 28  // 0x70 PERF split: vector unit true execution
    val SelPerfVecQueue = 29 // 0x74 PERF split: vector queue occupancy
    val SelPerfMatEng = 30   // 0x78 PERF split: either int8 engine executing
    val SelPerfSoloNoWork = 31  // 0x7C PERF: one engine busy, other not enqueued
    val SelPerfSoloBlocked = 32 // 0x80 PERF: one engine busy, other issue-blocked
    val SelPerfScStallVec = 33  // 0x84 PERF: scalar EX stalled on vector enqueue
    val SelPerfScStallMat = 34  // 0x88 PERF: scalar EX stalled on matrix enqueue
    val SelPerfStallGpr = 35    // 0x8C PERF: scalar EX gpr-hazard stall
    val SelPerfStallBlk = 36    // 0x90 PERF: scalar LW/SW completion wait
    val SelPerfStallHalt = 37   // 0x94 PERF: HALT unit-drain wait
    val SelPerfItcmFetch = 38  // 0x98 PERF: ITCM instruction fetches (= retired)
    val SelPerfDmaIssue = 39   // 0x9C PERF: DMALD/DMAST parked in EX on dmaReady
    val SelPerfBubEmpty = 40   // 0xA0 PERF: EX empty, no stall (true bubbles)
    val SelPerfBubHalt = 41    // 0xA4 PERF: bubEmpty with fetch stopped (post-HALT)
    val SelPerfDmaStarve = 42  // 0xA8 PERF: DMA wanted the aux DTCM port, core held it

    val io = IO(new Bundle {
        val axi  = new Axi4SlaveIO(AxiAddrBits, 32, AxiIdBits)
        val maxi = new Axi4MasterIO(AxiAddrBits, p.DtcmRowBits, AxiIdBits)
    })

    val itcmWords = p.ItcmSizeInWords.toInt
    val dtcmRows  = (p.DtcmSizeInBytes / p.DtcmRowBytes).toInt

    val core = Core(p, useDualDtcm = true)
    val itcm = Module(new Sram(p, itcmWords))
    val dtcm = Module(new DtcmRam(p, dtcmRows))
    val axi  = Module(new AxiMemSlave(AxiAddrBits, AxiIdBits))
    val dma  = DmaEngine(p, AxiAddrBits, AxiIdBits, log2Ceil(p.TokenSlots))

    io.axi  <> axi.io.axi
    io.maxi <> dma.io.maxi
    val host = axi.io.host

    // ---- Control / status registers ------------------------------------------
    val busy    = RegInit(false.B)
    val doneReg = RegInit(false.B)
    val perfMatrixBusy = RegInit(0.U(32.W))
    val perfVectorBusy = RegInit(0.U(32.W))
    val perfDmaBusy = RegInit(0.U(32.W))
    val perfDtcmMatrix = RegInit(0.U(32.W))
    val perfDtcmAux = RegInit(0.U(32.W))
    val perfDependencyWait = RegInit(0.U(32.W))
    val perfDmaWait = RegInit(0.U(32.W))
    val perfKicks = RegInit(0.U(32.W))
    val perfCycles = RegInit(0.U(32.W))
    val perfDmaDtcm = RegInit(0.U(32.W))
    val perfDepM = RegInit(0.U(32.W))
    val perfDepV = RegInit(0.U(32.W))
    val perfMemWait = RegInit(0.U(32.W))
    val perfMemClr = RegInit(0.U(32.W))
    val perfBothIdle = RegInit(0.U(32.W))
    val perfDepVSolo = RegInit(0.U(32.W))
    val perfEng0 = RegInit(0.U(32.W))    // PERF probe (dual-overlap work)
    val perfEng1 = RegInit(0.U(32.W))    // PERF probe (dual-overlap work)
    val perfEngBoth = RegInit(0.U(32.W)) // PERF probe (dual-overlap work)
    val perfVecUnit = RegInit(0.U(32.W))   // PERF split (queue vs unit)
    val perfVecQueue = RegInit(0.U(32.W))  // PERF split (queue vs unit)
    val perfMatEng = RegInit(0.U(32.W))    // PERF split (queue vs unit)
    val perfSoloNoWork = RegInit(0.U(32.W))  // solo-engine attribution
    val perfSoloBlocked = RegInit(0.U(32.W)) // solo-engine attribution
    val perfScStallVec = RegInit(0.U(32.W))  // scalar stall split
    val perfScStallMat = RegInit(0.U(32.W))  // scalar stall split
    val perfStallGpr = RegInit(0.U(32.W))    // scalar stall taps
    val perfStallBlk = RegInit(0.U(32.W))
    val perfStallHalt = RegInit(0.U(32.W))
    val perfItcmFetch = RegInit(0.U(32.W))   // scalar instruction count
    val perfDmaIssue = RegInit(0.U(32.W))    // DMALD/DMAST EX-wait on the engine
    val perfBubEmpty = RegInit(0.U(32.W))    // true scalar bubbles (iv=0, no stall)
    val perfBubHalt = RegInit(0.U(32.W))     // ... with the front end stopped
    val perfDmaStarve = RegInit(0.U(32.W))   // aux-port cycles the DMA was denied

    val region  = host.req.bits.addr(20, 19)
    val isItcm  = region === RegItcm.U
    val isDtcm  = region === RegDtcm.U
    val isCtrl  = region === RegCtrl.U
    // CTRL-region word select. The width is DERIVED from the Sel table: the old
    // hardcoded addr[6:2] (5 bits) silently aliased every PERF tap at 0x84-0x98
    // onto STATUS/unmapped words (0x84 read back the done bit; "scStallVec=1"
    // in the 9f7479b sim round was STATUS, not the stall counter). Growing the
    // register map now grows the decode automatically.
    private val selVals = Seq(SelCtrl, SelStatus, SelDmaDram, SelDmaDtcm,
        SelDmaLen, SelDmaCtrl, SelDmaStat, SelPerfMatrix, SelPerfVector,
        SelPerfDma, SelPerfDtcmMatrix, SelPerfDtcmAux, SelPerfDepWait,
        SelPerfDmaWait, SelPerfKicks, SelPerfCycles, SelPerfDmaDtcm,
        SelPerfDepM, SelPerfDepV, SelPerfMemWait, SelPerfMemClr,
        SelPerfBothIdle, SelPerfDepVSolo, SelPerfEng0, SelPerfEng1,
        SelPerfEngBoth, SelPerfVecUnit, SelPerfVecQueue, SelPerfMatEng,
        SelPerfSoloNoWork, SelPerfSoloBlocked, SelPerfScStallVec,
        SelPerfScStallMat, SelPerfStallGpr, SelPerfStallBlk, SelPerfStallHalt,
        SelPerfItcmFetch, SelPerfDmaIssue, SelPerfBubEmpty, SelPerfBubHalt,
        SelPerfDmaStarve)
    require(selVals.distinct.size == selVals.size, "duplicate CTRL Sel value")
    val ctrlSel = host.req.bits.addr(log2Ceil(selVals.max + 1) + 1, 2)

    // ---- DMA descriptor registers + run control ------------------------------
    val dmaActive  = RegInit(false.B)
    val dmaDoneReg = RegInit(false.B)
    val dmaDram    = RegInit(0.U(AxiAddrBits.W))
    val dmaDtcm    = RegInit(0.U(AxiAddrBits.W))
    val dmaLen     = RegInit(0.U(AxiAddrBits.W))

    val ctrlWrite = host.req.fire && isCtrl && host.req.bits.write

    when (ctrlWrite && ctrlSel === SelDmaDram.U) { dmaDram := host.req.bits.wdata }
    when (ctrlWrite && ctrlSel === SelDmaDtcm.U) { dmaDtcm := host.req.bits.wdata }
    when (ctrlWrite && ctrlSel === SelDmaLen.U)  { dmaLen  := host.req.bits.wdata }

    val dmaGo  = ctrlWrite && ctrlSel === SelDmaCtrl.U && host.req.bits.wdata(0)
    val dmaDir = host.req.bits.wdata(1)

    // Two trigger sources share the one DMA engine:
    //   * PS path: MMIO descriptor + go, only while the core is idle (legacy).
    //   * Core path: a DMALD/DMAST instruction enqueues a descriptor *while the
    //     core runs* -- this is what lets the layer program stream its own weights
    //     and overlap the load with compute (double buffering). The core only
    //     issues when the engine FIFO has room; a host go still requires a fully
    //     idle engine (hence an empty FIFO), so the two sources never collide.
    val rowShift = log2Up(p.DtcmRowBytes)     // 4: byte length -> row (beat) count
    val coreDma  = core.io.dmaStart
    val hostDma  = dmaGo && !dmaActive && !busy
    assert(!(hostDma && !dma.io.enq.ready),
        "host DMA descriptor issued without FIFO room")
    dma.io.enq.valid         := coreDma || hostDma
    dma.io.enq.bits.dir      := Mux(coreDma, core.io.dmaDir,  dmaDir)
    dma.io.enq.bits.dramAddr := Mux(coreDma, core.io.dmaDram, dmaDram)
    dma.io.enq.bits.dtcmAddr := Mux(coreDma, core.io.dmaDtcm, dmaDtcm)
    dma.io.enq.bits.numRows  := Mux(coreDma, core.io.dmaRows, dmaLen(rowShift + 15, rowShift))
    // Only a core descriptor carries a token lease; the host path owns no slot.
    dma.io.enq.bits.tok.en   := coreDma && core.io.dmaTok.en
    dma.io.enq.bits.tok.id   := core.io.dmaTok.id

    // FIFO room, not engine idleness: DMALD/DMAST may stage behind the executing
    // transfer. DMAWAIT/HALT drain on Alu's lease count (core.io.dmaDrained).
    core.io.dmaIdle := dma.io.enq.ready
    // Completion echo: the descriptor's own lease tag returns to Alu, which
    // retires the slot and decrements the in-flight counter.
    core.io.dmaDone       := dma.io.done
    core.io.dmaDoneTok.en := dma.io.doneTok.en
    core.io.dmaDoneTok.id := dma.io.doneTok.id

    // ITCM/DTCM are owned by the core (busy) or the DMA (dmaActive) while active.
    host.req.ready := Mux(isCtrl, true.B,
                      Mux(isDtcm, !busy && !dmaActive, !busy))

    val startPulse = ctrlWrite && ctrlSel === SelCtrl.U && host.req.bits.wdata(0)
    val clearPulse = ctrlWrite && ctrlSel === SelCtrl.U && host.req.bits.wdata(1)

    core.io.start := startPulse && !busy
    when (startPulse && !busy) {
        busy    := true.B
        doneReg := false.B
        perfKicks := perfKicks + 1.U
        perfMatrixBusy := 0.U
        perfVectorBusy := 0.U
        perfDmaBusy := 0.U
        perfDtcmMatrix := 0.U
        perfDtcmAux := 0.U
        perfDependencyWait := 0.U
        perfDmaWait := 0.U
        perfCycles := 0.U
        perfDmaDtcm := 0.U
        perfDepM := 0.U
        perfDepV := 0.U
        perfMemWait := 0.U
        perfMemClr := 0.U
        perfBothIdle := 0.U
        perfDepVSolo := 0.U
        perfEng0 := 0.U
        perfEng1 := 0.U
        perfEngBoth := 0.U
        perfVecUnit := 0.U
        perfVecQueue := 0.U
        perfMatEng := 0.U
        perfSoloNoWork := 0.U
        perfSoloBlocked := 0.U
        perfScStallVec := 0.U
        perfScStallMat := 0.U
        perfStallGpr := 0.U
        perfStallBlk := 0.U
        perfStallHalt := 0.U
        perfItcmFetch := 0.U
        perfDmaIssue := 0.U
        perfBubEmpty := 0.U
        perfBubHalt := 0.U
        perfDmaStarve := 0.U
    }
    when (busy && core.io.done) {
        busy    := false.B
        doneReg := true.B
    }
    when (clearPulse) {
        doneReg := false.B
    }

    // Per-kick counters saturate only by natural 32-bit wrap. They intentionally
    // sample the physical port handshakes, not request valid, so a same-row
    // collision is visible as lost utilization rather than falsely counted work.
    when (busy) {
        perfCycles := perfCycles + 1.U
        when (core.io.matrixBusy) { perfMatrixBusy := perfMatrixBusy + 1.U }
        when (core.io.matEngBusy(0)) { perfEng0 := perfEng0 + 1.U }                       // PERF probe (dual-overlap work)
        when (core.io.matEngBusy(1)) { perfEng1 := perfEng1 + 1.U }                       // PERF probe (dual-overlap work)
        when (core.io.matEngBusy(0) && core.io.matEngBusy(1)) { perfEngBoth := perfEngBoth + 1.U } // PERF probe (dual-overlap work)
        when (core.io.vectorUnitBusy) { perfVecUnit := perfVecUnit + 1.U }  // PERF split (queue vs unit)
        when (core.io.vectorQueued) { perfVecQueue := perfVecQueue + 1.U }  // PERF split (queue vs unit)
        when (core.io.matEngBusy(0) || core.io.matEngBusy(1)) { perfMatEng := perfMatEng + 1.U } // PERF split (queue vs unit)
        when (core.io.engSoloNoWork) { perfSoloNoWork := perfSoloNoWork + 1.U }
        when (core.io.engSoloBlocked) { perfSoloBlocked := perfSoloBlocked + 1.U }
        when (core.io.scStallVec) { perfScStallVec := perfScStallVec + 1.U }
        when (core.io.scStallMat) { perfScStallMat := perfScStallMat + 1.U }
        when (core.io.stallGpr) { perfStallGpr := perfStallGpr + 1.U }
        when (core.io.stallBlk) { perfStallBlk := perfStallBlk + 1.U }
        when (core.io.stallHalt) { perfStallHalt := perfStallHalt + 1.U }
        when (core.io.scAdvance) { perfItcmFetch := perfItcmFetch + 1.U }
        when (core.io.dmaIssueWait) { perfDmaIssue := perfDmaIssue + 1.U }
        when (core.io.bubEmpty) { perfBubEmpty := perfBubEmpty + 1.U }
        when (core.io.bubHalting) { perfBubHalt := perfBubHalt + 1.U }
        // Aux-port starvation: the engine holds a DTCM beat but the port stayed
        // with the core this cycle (Dtcm: useDma = dmaReq && !auxCoreReq).
        when (dma.io.dtcm.valid && !dma.io.dtcm.ready) {
            perfDmaStarve := perfDmaStarve + 1.U
        }
        when (core.io.vectorBusy) { perfVectorBusy := perfVectorBusy + 1.U }
        when (dmaActive) { perfDmaBusy := perfDmaBusy + 1.U }
        when (core.io.dtcmMatrix.valid && core.io.dtcmMatrix.ready) {
            perfDtcmMatrix := perfDtcmMatrix + 1.U
        }
        when ((core.io.dtcmAux.valid && core.io.dtcmAux.ready) ||
            (dma.io.dtcm.valid && dma.io.dtcm.ready)) {
            perfDtcmAux := perfDtcmAux + 1.U
        }
        when (dma.io.dtcm.valid && dma.io.dtcm.ready) {
            perfDmaDtcm := perfDmaDtcm + 1.U
        }
        when (core.io.dependencyWait) { perfDependencyWait := perfDependencyWait + 1.U }
        when (core.io.depWaitM) { perfDepM := perfDepM + 1.U }
        when (core.io.depWaitV) { perfDepV := perfDepV + 1.U }
        when (core.io.dmaWaitStalled) { perfDmaWait := perfDmaWait + 1.U }
        // Barrier attribution (V1 plan): MEM_WAIT - MEM_CLR is the drain cost a
        // row-token dispatch would remove; BOTH_IDLE is kick dead time;
        // DEP_V_SOLO is the share of depwaitV that is a true vector-side bubble
        // (every other unit idle) rather than overlap with matrix/DMA work.
        when (core.io.memWaitStalled) { perfMemWait := perfMemWait + 1.U }
        when (core.io.memWaitStalled && !core.io.matrixBusy && !core.io.vectorBusy) {
            perfMemClr := perfMemClr + 1.U
        }
        when (!core.io.matrixBusy && !core.io.vectorBusy) {
            perfBothIdle := perfBothIdle + 1.U
        }
        when (core.io.depWaitV && !core.io.matrixBusy && !dmaActive) {
            perfDepVSolo := perfDepVSolo + 1.U
        }
    }

    // Latch dmaActive when a descriptor is accepted (core or host); flag done on
    // the busy falling edge. dma.io.busy covers the descriptor FIFO too, so the
    // window closes only after the last accepted transfer has finished.
    when (dma.io.enq.fire) {
        dmaActive  := true.B
        dmaDoneReg := false.B
    }
    when (dmaActive && !dma.io.busy) {
        dmaActive  := false.B
        dmaDoneReg := true.B
    }

    val status    = Cat(0.U(30.W), busy, doneReg)
    val dmaStatus = Cat(0.U(30.W), dmaActive, dmaDoneReg)

    // ---- ITCM port mux (core read while busy, host otherwise) ----------------
    val coreItcmIdx = (core.io.itcmAddr >> 2)(itcm.addrBits - 1, 0)
    val hostItcmIdx = (host.req.bits.addr >> 2)(itcm.addrBits - 1, 0)
    itcm.io.en    := Mux(busy, true.B, host.req.fire && isItcm)
    itcm.io.write := Mux(busy, false.B, host.req.bits.write)
    itcm.io.addr  := Mux(busy, coreItcmIdx, hostItcmIdx)
    itcm.io.wdata := host.req.bits.wdata
    itcm.io.wstrb := host.req.bits.wstrb
    core.io.itcmInst := itcm.io.rdata

    // ---- Dual-port DTCM ------------------------------------------------------
    // Matrix owns port 0; Vector/Mem and DMA arbitrate on port 1.  The legacy
    // Core port is tied inactive because this top selects the physical ports.
    dtcm.io.busy      := busy
    dtcm.io.dmaActive := dmaActive
    dtcm.io.matrix <> core.io.dtcmMatrix
    dtcm.io.aux <> core.io.dtcmAux
    dtcm.io.dma <> dma.io.dtcm
    core.io.dtcm.ready := false.B
    core.io.dtcm.rdata := 0.U
    dtcm.io.hostEn    := host.req.fire && isDtcm
    dtcm.io.hostWrite := host.req.bits.write
    dtcm.io.hostAddr  := host.req.bits.addr
    dtcm.io.hostWdata := host.req.bits.wdata
    dtcm.io.hostWstrb := host.req.bits.wstrb

    // ---- Host read return (1-cycle latency; route by the region read) --------
    val rdRegion = Reg(UInt(2.W))
    val ctrlReadReg = Reg(UInt(32.W))
    when (host.req.fire && !host.req.bits.write) {
        rdRegion := region
        when (isCtrl) {
            ctrlReadReg := MuxLookup(ctrlSel, 0.U)(Seq(
                SelStatus.U  -> status,
                SelDmaStat.U -> dmaStatus,
                SelPerfMatrix.U -> perfMatrixBusy,
                SelPerfVector.U -> perfVectorBusy,
                SelPerfDma.U -> perfDmaBusy,
                SelPerfDtcmMatrix.U -> perfDtcmMatrix,
                SelPerfDtcmAux.U -> perfDtcmAux,
                SelPerfDepWait.U -> perfDependencyWait,
                SelPerfDmaWait.U -> perfDmaWait,
                SelPerfKicks.U -> perfKicks,
                SelPerfCycles.U -> perfCycles,
                SelPerfDmaDtcm.U -> perfDmaDtcm,
                SelPerfDepM.U -> perfDepM,
                SelPerfDepV.U -> perfDepV,
                SelPerfMemWait.U -> perfMemWait,
                SelPerfMemClr.U -> perfMemClr,
                SelPerfBothIdle.U -> perfBothIdle,
                SelPerfDepVSolo.U -> perfDepVSolo,
                SelPerfEng0.U -> perfEng0,
                SelPerfEng1.U -> perfEng1,
                SelPerfEngBoth.U -> perfEngBoth,
                SelPerfVecUnit.U -> perfVecUnit,
                SelPerfVecQueue.U -> perfVecQueue,
                SelPerfMatEng.U -> perfMatEng,
                SelPerfSoloNoWork.U -> perfSoloNoWork,
                SelPerfSoloBlocked.U -> perfSoloBlocked,
                SelPerfScStallVec.U -> perfScStallVec,
                SelPerfScStallMat.U -> perfScStallMat,
                SelPerfStallGpr.U -> perfStallGpr,
                SelPerfStallBlk.U -> perfStallBlk,
                SelPerfStallHalt.U -> perfStallHalt,
                SelPerfItcmFetch.U -> perfItcmFetch,
                SelPerfDmaIssue.U -> perfDmaIssue,
                SelPerfBubEmpty.U -> perfBubEmpty,
                SelPerfBubHalt.U -> perfBubHalt,
                SelPerfDmaStarve.U -> perfDmaStarve,
            ))
        }
    }
    host.rdata := MuxLookup(rdRegion, 0.U)(Seq(
        RegItcm.U -> itcm.io.rdata,
        RegDtcm.U -> dtcm.io.hostRdata,
        RegCtrl.U -> ctrlReadReg,
    ))
}
