// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


// Regfile read port, defined from the regfile's point of view (it drives the
// data out). Consumers Flip it. Mirrors CoralNPU's RegfileReadDataIO.
class RegfileReadDataIO extends Bundle {
    val valid = Output(Bool())
    val data  = Output(UInt(32.W))
}

// Regfile write port, defined from the regfile's point of view (it receives the
// address/data). Producers Flip it.
class RegfileWriteDataIO extends Bundle {
    val addr = Input(UInt(5.W))
    val data = Input(UInt(32.W))
}

// Wide DTCM access port used by the matrix unit for burst load/store. One full
// matrix row (DtcmRowBits = WordsPerRow words = 128 bits) per cycle, handshaked.
// Read data returns the cycle after a read request fires (1-cycle synchronous
// read), mirroring a simple wide single-port SRAM. The AXI host keeps a separate
// 32-bit view of the same memory (see Npu); this port is matrix-only.
class DtcmPortIO(p: Parameters) extends Bundle {
    val valid = Output(Bool())                    // request asserted
    val ready = Input(Bool())                     // memory accepts the request
    val write = Output(Bool())                    // 1 = store, 0 = load
    val addr  = Output(UInt(p.AddrBits.W))        // byte address, DtcmRowBytes-aligned
    val wdata = Output(UInt(p.DtcmRowBits.W))     // store data (one row)
    val wstrb = Output(UInt(p.DtcmRowBytes.W))    // per-byte write enables
    val rdata = Input(UInt(p.DtcmRowBits.W))      // load data, valid cycle after a read fires
}
