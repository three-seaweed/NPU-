// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


object MemUnit {
    def apply(p: Parameters): MemUnit = Module(new MemUnit(p))
}

// Scalar DTCM word load/store unit (LW/SW). It mirrors VectorUnit's interface --
// a Decoupled req (ready only when idle), a wide DtcmPortIO, a Decoupled rd for a
// load result, and a 1-cycle `done` pulse -- so Control can treat it as just
// another blocking-in-EX background op sharing the single wide DTCM port.
//
// The DTCM is row-wide (DtcmRowBits). A scalar 32-bit access targets word
// `addr[3:2]` within the row at `addr & ~(rowBytes-1)`: a load reads the row and
// selects that word; a store writes only that word's bytes via wstrb. The
// address arrives pre-summed (rs1 + imm) on io.rs1.data, mirroring how Matmul
// receives its base; store data is io.rs2.data.
//
// Since V1 Control dispatches this unit once its DTCM row is free of earlier
// range tokens (not on a full drain), so it may hold a request while the
// vector unit still owns the aux port; the arb in Alu backpressures it until
// the port frees (its request is persistent).
class MemUnit(p: Parameters) extends Module {
    val rowBytes  = p.DtcmRowBytes                 // 16
    val rowWords  = p.WordsPerRow                  // 4 words per wide row
    val rowShift  = log2Up(rowBytes)               // 4
    val laneBits  = log2Up(rowWords)               // 2

    val io = IO(new Bundle {
        val req  = Flipped(Decoupled(new AluCmd))
        val rs1  = Flipped(new RegfileReadDataIO)  // effective address (rs1 + imm)
        val rs2  = Flipped(new RegfileReadDataIO)  // store data (SW)
        val rd   = Decoupled(new RegfileWriteDataIO)
        val dtcm = new DtcmPortIO(p)
        val done = Output(Bool())                  // 1-cycle pulse when an op completes
    })

    val sIdle :: sRead :: sCap :: sWrite :: Nil = Enum(4)
    val state = RegInit(sIdle)

    val addr    = RegInit(0.U(p.AddrBits.W))       // effective byte address
    val wdata   = RegInit(0.U(32.W))               // store data
    val rdReg   = RegInit(0.U(5.W))
    val result  = RegInit(0.U(32.W))

    val rowAddr = addr & ~((rowBytes - 1).U(p.AddrBits.W))
    val lane    = (addr >> 2)(laneBits - 1, 0)     // which 32-bit word in the row

    // ---- Defaults ------------------------------------------------------------
    io.dtcm.valid := false.B
    io.dtcm.write := false.B
    io.dtcm.addr  := rowAddr
    io.dtcm.wdata := 0.U
    io.dtcm.wstrb := 0.U
    io.rd.valid     := false.B
    io.rd.bits.addr := rdReg
    io.rd.bits.data := result
    io.req.ready := state === sIdle
    val doneWire = WireDefault(false.B)
    io.done := doneWire

    // ---- Idle / dispatch -----------------------------------------------------
    when (state === sIdle) {
        when (io.req.fire) {
            addr  := io.rs1.data
            wdata := io.rs2.data
            rdReg := io.req.bits.addr
            state := Mux(io.req.bits.op === AluOp.SW, sWrite, sRead)
        }
    }

    // ---- Load: issue read, then capture + select the word --------------------
    when (state === sRead) {
        io.dtcm.valid := true.B
        io.dtcm.write := false.B
        when (io.dtcm.ready) { state := sCap }
    }
    when (state === sCap) {
        val words = VecInit(Seq.tabulate(rowWords)(w => io.dtcm.rdata(32 * w + 31, 32 * w)))
        result   := words(lane)
        io.rd.valid := true.B
        io.rd.bits.data := words(lane)
        doneWire := true.B
        state := sIdle
    }

    // ---- Store: write one word's bytes via wstrb -----------------------------
    when (state === sWrite) {
        io.dtcm.valid := true.B
        io.dtcm.write := true.B
        // Replicate the 32-bit word into every lane; wstrb enables only the target.
        io.dtcm.wdata := Fill(rowWords, wdata)
        io.dtcm.wstrb := Cat((0 until rowWords).reverse.map(w =>
            Mux(w.U === lane, "b1111".U(4.W), "b0000".U(4.W))))
        when (io.dtcm.ready) {
            doneWire := true.B
            state := sIdle
        }
    }
}
