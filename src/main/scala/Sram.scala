// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


// Single-port synchronous RAM (maps to BRAM). Read data appears one cycle after
// a read access (SyncReadMem semantics). Byte write-enables via wstrb support
// partial-word writes.
class Sram(p: Parameters, words: Int) extends Module {
    val addrBits = log2Up(words)

    val io = IO(new Bundle {
        val en    = Input(Bool())
        val write = Input(Bool())
        val addr  = Input(UInt(addrBits.W))     // word address
        val wdata = Input(UInt(32.W))
        val wstrb = Input(UInt(4.W))
        val rdata = Output(UInt(32.W))          // valid one cycle after a read
    })

    val mem = SyncReadMem(words, Vec(4, UInt(8.W)))

    val wvec = VecInit(Seq.tabulate(4)(i => io.wdata(8 * i + 7, 8 * i)))

    // Synchronous read; the enable gates the internal address register so the
    // output holds between reads.
    val rdata = mem.read(io.addr, io.en && !io.write)

    when (io.en && io.write) {
        mem.write(io.addr, wvec, io.wstrb.asBools)
    }

    io.rdata := rdata.asUInt
}
