// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


object DmaEngine {
    def apply(p: Parameters, addrBits: Int, idBits: Int, tokBits: Int): DmaEngine =
        Module(new DmaEngine(p, addrBits, idBits, tokBits))
}

// DTCM range-token lease tag carried by one DMA descriptor. `en` false means
// the descriptor owns no token slot (host MMIO descriptors, zero-row
// transfers). The tag travels WITH the descriptor through the engine and comes
// back on the matching `done` pulse, so retirement is per descriptor instead of
// inferred from engine idleness.
class DmaTokenTag(tokBits: Int) extends Bundle {
    val en = Bool()
    val id = UInt(tokBits.W)
}

// One descriptor as staged in the engine's FIFO.
class DmaDesc(addrBits: Int, tokBits: Int) extends Bundle {
    val dir      = Bool()               // 0 = load (DRAM->DTCM), 1 = store
    val dramAddr = UInt(addrBits.W)
    val dtcmAddr = UInt(addrBits.W)
    val numRows  = UInt(16.W)           // DtcmRowBytes-wide rows to move
    val tok      = new DmaTokenTag(tokBits)
}

// AXI4 master DMA engine: bursts data between PS DRAM (via M_AXI, connected to a
// PS HP slave port) and the NPU's wide DTCM, so the PS no longer has to push
// every weight/result word over the 32-bit S_AXI slave one single-beat AXI
// transaction at a time.
//
// One descriptor moves `numRows` consecutive DtcmRowBytes-wide rows (one row =
// one 128-bit AXI beat = one wide DTCM word):
//   dir = 0 : DRAM[dramAddr ..] -> DTCM[dtcmAddr ..]   (load weights/activations)
//   dir = 1 : DTCM[dtcmAddr ..] -> DRAM[dramAddr ..]   (store results)
//
// The transfer is split into INCR bursts, each capped to 256 beats (AXI len is
// 8-bit) AND to the next 4KB boundary (AXI forbids a burst crossing 4KB). The
// engine keeps a single outstanding burst at a time.
//
// Descriptors enter a depth-2 FIFO (`enq`) and execute strictly in order. The
// scalar core's issue gate is `enq.ready` (FIFO has room), NOT engine idleness:
// a following DMALD can stage while the previous transfer is still bursting, so
// the scalar->DMA hand-off stops being a depth-1 serial chain. Descriptors are
// popped into the FSM in sIdle, one per idle visit (a zero-row entry completes
// at its pop and lets the next entry pop the following cycle).
//
// Each entry carries its own token lease (`tok`) and the completion pulse echoes
// it (`doneTok`). A single shared lease register at the consumer would be
// overwritten by the second in-flight descriptor, leaving the first slot held
// forever -- the DMA-deadlock class documented in docs/DMA死锁验证/. `doneTok` is
// valid only in the `done` cycle; for a zero-row entry it describes the entry
// popped that same cycle, otherwise the entry latched at its pop.
class DmaEngine(p: Parameters, addrBits: Int, idBits: Int, tokBits: Int) extends Module {
    val dataBits = p.DtcmRowBits        // 128: one matrix row per beat
    val rowBytes = p.DtcmRowBytes       // 16
    val rowShift = log2Up(rowBytes)     // 4
    val sizeEnc  = log2Up(rowBytes)     // AXI ARSIZE/AWSIZE = 4 (16 bytes/beat)
    val maxBeats = 256                  // AXI INCR limit (len = beats - 1, 8-bit)
    val descDepth = 2                   // staged descriptors behind the executing one

    val io = IO(new Bundle {
        val maxi = new Axi4MasterIO(addrBits, dataBits, idBits)
        val dtcm = new DtcmPortIO(p)

        // Descriptor enqueue: `ready` is FIFO room, not engine idleness.
        val enq = Flipped(Decoupled(new DmaDesc(addrBits, tokBits)))

        val busy = Output(Bool())       // FIFO non-empty or a transfer in progress
        val done = Output(Bool())       // 1-cycle: one descriptor completed
        val doneTok = Output(new DmaTokenTag(tokBits))  // valid with `done`
    })

    val sIdle :: sLoad :: sStoreAddr :: sStoreRead :: sStoreData :: sStoreResp :: Nil = Enum(6)
    val state = RegInit(sIdle)

    // ---- Descriptor FIFO -----------------------------------------------------
    // Explicit handshake wiring (both ends are inputs inside this module).
    val descQ = Module(new Queue(new DmaDesc(addrBits, tokBits), descDepth))
    descQ.io.enq.valid := io.enq.valid
    descQ.io.enq.bits  := io.enq.bits
    io.enq.ready       := descQ.io.enq.ready
    descQ.io.deq.ready := false.B        // set in sIdle: pop + start

    // Lease of the descriptor currently executing (latched at its pop).
    val curTok = Reg(new DmaTokenTag(tokBits))
    val pop     = descQ.io.deq.fire
    val popZero = pop && descQ.io.deq.bits.numRows === 0.U

    // Store path state (serialized: one burst at a time).
    val curDram = Reg(UInt(addrBits.W))   // running DRAM byte address
    val curDtcm = Reg(UInt(addrBits.W))   // running DTCM byte address
    val remRows = Reg(UInt(16.W))         // rows still to move (whole transfer)
    val beats   = Reg(UInt(9.W))          // beats in the current burst (1..256)
    val beat    = Reg(UInt(9.W))          // beats issued so far in current burst

    // Load path state (multiple outstanding bursts). AR-issue and R-receive advance
    // independently: `*I` are the issue-side running values, `*R` the receive-side.
    // burstLenQ holds the beat-count of each in-flight burst in FIFO order; with a
    // single AXI id, R data returns in the order ARs were issued, so the queue head
    // always describes the burst whose beats are arriving now.
    val curDramI    = Reg(UInt(addrBits.W))   // issue-side DRAM address
    val curDtcmR    = Reg(UInt(addrBits.W))   // receive-side DTCM address
    val remIssue    = Reg(UInt(16.W))         // rows whose AR is not yet issued
    val remRecv     = Reg(UInt(16.W))         // rows whose R beat is not yet received
    val beatInBurst = Reg(UInt(9.W))          // beats received in the current burst
    val burstLenQ   = Module(new Queue(UInt(9.W), p.DmaMaxOutstanding))

    // Current burst length from a running address + remaining count: capped to 256
    // beats and to the next 4KB boundary (AXI forbids a burst crossing 4KB). The
    // address is DtcmRowBytes-aligned, so beatsTo4k is in [1, 256].
    def burstBeatsFrom(addr: UInt, rem: UInt): UInt = {
        val off4k     = addr(11, 0)
        val beatsTo4k = (4096.U - off4k) >> rowShift
        val cap       = Mux(rem > maxBeats.U, maxBeats.U, rem)
        Mux(beatsTo4k < cap, beatsTo4k, cap)(8, 0)
    }
    val burstBeats  = burstBeatsFrom(curDram, remRows)    // store-side
    val burstBeatsI = burstBeatsFrom(curDramI, remIssue)  // load-side (issue)

    // ---- Defaults ------------------------------------------------------------
    def driveAddr(ch: DecoupledIO[AxiAddr]): Unit = {
        ch.valid     := false.B
        ch.bits.id    := 0.U
        ch.bits.addr  := curDram
        ch.bits.len   := burstBeats - 1.U
        ch.bits.size  := sizeEnc.U
        ch.bits.burst := 1.U                 // INCR
    }
    driveAddr(io.maxi.aw)
    driveAddr(io.maxi.ar)

    io.maxi.w.valid     := false.B
    io.maxi.w.bits.data := 0.U
    io.maxi.w.bits.strb := ~0.U((dataBits / 8).W)
    io.maxi.w.bits.last := false.B
    io.maxi.b.ready     := false.B
    io.maxi.r.ready     := false.B

    burstLenQ.io.enq.valid := false.B
    burstLenQ.io.enq.bits  := burstBeatsI
    burstLenQ.io.deq.ready := false.B

    io.dtcm.valid := false.B
    io.dtcm.write := false.B
    io.dtcm.addr  := curDtcm
    io.dtcm.wdata := 0.U
    io.dtcm.wstrb := 0.U

    // Queued descriptors count as busy: the host DTCM-ownership window and the
    // DMA_STATUS busy bit must stay asserted until the FIFO is empty too.
    io.busy := state =/= sIdle || descQ.io.count =/= 0.U
    val doneWire = WireDefault(false.B)
    io.done := doneWire
    io.doneTok.en := doneWire && Mux(popZero, descQ.io.deq.bits.tok.en, curTok.en)
    io.doneTok.id := Mux(popZero, descQ.io.deq.bits.tok.id, curTok.id)

    // Store path: DTCM read has 1-cycle latency, so latch the row and hold it on
    // the W channel across back-pressure. `justRead` is true the first cycle in
    // sStoreData (when io.dtcm.rdata is valid for the read issued in sStoreRead).
    // DTCM may back-pressure (the core wins the shared port), so a store read is
    // "served" only when dtcm.ready was high in sStoreRead; latch its row the next
    // cycle. justRead aligns the captured rdata with the W beat.
    val wbuf     = Reg(UInt(dataBits.W))
    val justRead = RegNext(state === sStoreRead && io.dtcm.ready, false.B)
    when (justRead) { wbuf := io.dtcm.rdata }

    val lastBeat = beat === (beats - 1.U)

    switch (state) {
        is (sIdle) {
            descQ.io.deq.ready := true.B
            when (descQ.io.deq.valid) {
                curTok := descQ.io.deq.bits.tok
                when (descQ.io.deq.bits.numRows === 0.U) {
                    doneWire := true.B
                } .elsewhen (descQ.io.deq.bits.dir) {
                    curDram := descQ.io.deq.bits.dramAddr
                    curDtcm := descQ.io.deq.bits.dtcmAddr
                    remRows := descQ.io.deq.bits.numRows
                    state   := sStoreAddr
                } .otherwise {
                    curDramI    := descQ.io.deq.bits.dramAddr
                    curDtcmR    := descQ.io.deq.bits.dtcmAddr
                    remIssue    := descQ.io.deq.bits.numRows
                    remRecv     := descQ.io.deq.bits.numRows
                    beatInBurst := 0.U
                    state       := sLoad
                }
            }
        }

        // ---- Load: DRAM -> DTCM, multiple outstanding bursts -----------------
        is (sLoad) {
            // AR issue: launch a new burst whenever one is left to issue and the
            // in-flight FIFO has room (this bounds outstanding to DmaMaxOutstanding).
            io.maxi.ar.valid     := (remIssue =/= 0.U) && burstLenQ.io.enq.ready
            io.maxi.ar.bits.addr := curDramI
            io.maxi.ar.bits.len  := burstBeatsI - 1.U
            // burstBeatsFrom returns 0 for an unaligned address in a 4K page
            // tail; the AR below would then carry len=0xFFFF and the length
            // FIFO would wedge on a never-finished burst. Row-aligned
            // descriptors can never do this, so treat it as a sim-fatal
            // software bug rather than a silent 400k-poll hang.
            assert(!(io.maxi.ar.valid && burstBeatsI === 0.U),
                "DMA load: 0-beat burst -- unaligned curDramI at a 4K page tail")
            burstLenQ.io.enq.valid := io.maxi.ar.fire     // enqueue the burst's length
            when (io.maxi.ar.fire) {
                curDramI := curDramI + (burstBeatsI << rowShift)
                remIssue := remIssue - burstBeatsI
            }

            // R receive: beats return in FIFO order (single id). Accept a beat only
            // when the DTCM port grants the write this cycle (core keeps priority),
            // and pop the FIFO on each burst's last beat.
            val headLen      = burstLenQ.io.deq.bits
            val lastOfBurst  = beatInBurst === (headLen - 1.U)
            io.dtcm.valid   := burstLenQ.io.deq.valid && io.maxi.r.valid
            io.dtcm.write   := true.B
            io.dtcm.addr    := curDtcmR
            io.dtcm.wdata   := io.maxi.r.bits.data
            io.dtcm.wstrb   := ~0.U(rowBytes.W)
            io.maxi.r.ready := burstLenQ.io.deq.valid && io.dtcm.ready
            burstLenQ.io.deq.ready := io.maxi.r.fire && lastOfBurst
            when (io.maxi.r.fire) {
                curDtcmR    := curDtcmR + rowBytes.U
                beatInBurst := Mux(lastOfBurst, 0.U, beatInBurst + 1.U)
                remRecv     := remRecv - 1.U
                when (remRecv === 1.U) { state := sIdle; doneWire := true.B }
            }
        }

        // ---- Store: DTCM -> DRAM ---------------------------------------------
        is (sStoreAddr) {
            io.maxi.aw.valid := true.B
            assert(burstBeats =/= 0.U,
                "DMA store: 0-beat burst -- unaligned curDram at a 4K page tail")
            when (io.maxi.aw.fire) {
                beats := burstBeats
                beat  := 0.U
                state := sStoreRead
            }
        }
        is (sStoreRead) {
            // Issue a DTCM read; rdata is valid next cycle (in sStoreData). Hold
            // here until the shared port serves us (the core has priority).
            io.dtcm.valid := true.B
            io.dtcm.write := false.B
            io.dtcm.addr  := curDtcm
            when (io.dtcm.ready) { state := sStoreData }
        }
        is (sStoreData) {
            io.maxi.w.valid     := true.B
            io.maxi.w.bits.data := Mux(justRead, io.dtcm.rdata, wbuf)
            io.maxi.w.bits.strb := ~0.U((dataBits / 8).W)
            io.maxi.w.bits.last := lastBeat
            when (io.maxi.w.fire) {
                curDtcm := curDtcm + rowBytes.U
                curDram := curDram + rowBytes.U
                remRows := remRows - 1.U
                beat    := beat + 1.U
                when (lastBeat) { state := sStoreResp }
                .otherwise      { state := sStoreRead }
            }
        }
        is (sStoreResp) {
            io.maxi.b.ready := true.B
            when (io.maxi.b.valid) {
                when (remRows === 0.U) { state := sIdle; doneWire := true.B }
                .otherwise             { state := sStoreAddr }
            }
        }
    }
}
