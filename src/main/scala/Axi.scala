// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


// Minimal AXI4 channel bundles, just enough for a memory-mapped slave that the
// Zynq PS (via its AXI DMA) bursts into. Data width is fixed at 32 bits to
// match the NPU word; only INCR bursts and a single outstanding transaction
// are supported.
class AxiAddr(addrBits: Int, idBits: Int) extends Bundle {
    val id    = UInt(idBits.W)
    val addr  = UInt(addrBits.W)
    val len   = UInt(8.W)   // beats - 1
    val size  = UInt(3.W)   // log2(bytes per beat); expected 2 (=4 bytes)
    val burst = UInt(2.W)   // 01 = INCR
}

class AxiWriteData(dataBits: Int) extends Bundle {
    val data = UInt(dataBits.W)
    val strb = UInt((dataBits / 8).W)
    val last = Bool()
}

class AxiWriteResp(idBits: Int) extends Bundle {
    val id   = UInt(idBits.W)
    val resp = UInt(2.W)    // 00 = OKAY
}

class AxiReadData(dataBits: Int, idBits: Int) extends Bundle {
    val id   = UInt(idBits.W)
    val data = UInt(dataBits.W)
    val resp = UInt(2.W)    // 00 = OKAY
    val last = Bool()
}

// Slave-side view (the NPU is the slave): address/data/write-data come in,
// responses/read-data go out.
class Axi4SlaveIO(val addrBits: Int, val dataBits: Int, val idBits: Int) extends Bundle {
    val aw = Flipped(Decoupled(new AxiAddr(addrBits, idBits)))
    val w  = Flipped(Decoupled(new AxiWriteData(dataBits)))
    val b  = Decoupled(new AxiWriteResp(idBits))
    val ar = Flipped(Decoupled(new AxiAddr(addrBits, idBits)))
    val r  = Decoupled(new AxiReadData(dataBits, idBits))
}

// Master-side view (the NPU is the master): the DMA engine drives address and
// write-data out and consumes responses/read-data. Used for the M_AXI port that
// lets the NPU burst weights/activations straight from PS DRAM (a PS HP slave),
// instead of having the PS push every word over the 32-bit S_AXI slave. Data
// width is DtcmRowBits (128) so one beat carries one full matrix row.
class Axi4MasterIO(val addrBits: Int, val dataBits: Int, val idBits: Int) extends Bundle {
    val aw = Decoupled(new AxiAddr(addrBits, idBits))
    val w  = Decoupled(new AxiWriteData(dataBits))
    val b  = Flipped(Decoupled(new AxiWriteResp(idBits)))
    val ar = Decoupled(new AxiAddr(addrBits, idBits))
    val r  = Flipped(Decoupled(new AxiReadData(dataBits, idBits)))
}
