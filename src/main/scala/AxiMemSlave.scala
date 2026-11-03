// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


// Host-side memory request emitted by the AXI slave. The top level decodes the
// region from `addr` and routes it to ITCM / DTCM / control registers.
class HostMemReq(addrBits: Int) extends Bundle {
    val write = Bool()
    val addr  = UInt(addrBits.W)   // full byte address
    val wdata = UInt(32.W)
    val wstrb = UInt(4.W)
}

class HostMemIO(addrBits: Int) extends Bundle {
    val req   = Decoupled(new HostMemReq(addrBits))
    val rdata = Input(UInt(32.W))  // valid one cycle after a read req fires
}

// AXI4 memory-mapped slave: turns INCR read/write bursts into a stream of
// single-word host memory accesses. One outstanding transaction at a time,
// write-prioritised. 32-bit data only.
class AxiMemSlave(addrBits: Int, idBits: Int) extends Module {
    val io = IO(new Bundle {
        val axi  = new Axi4SlaveIO(addrBits, 32, idBits)
        val host = new HostMemIO(addrBits)
    })

    val sIdle :: sWrite :: sResp :: sReadReq :: sReadData :: Nil = Enum(5)
    val state = RegInit(sIdle)

    val curAddr = Reg(UInt(addrBits.W))
    val len     = Reg(UInt(8.W))
    val id      = Reg(UInt(idBits.W))
    val beat    = Reg(UInt(8.W))

    // Defaults.
    io.axi.aw.ready := false.B
    io.axi.ar.ready := false.B
    io.axi.w.ready  := false.B
    io.axi.b.valid  := false.B
    io.axi.b.bits.id   := id
    io.axi.b.bits.resp := 0.U
    io.axi.r.valid := false.B
    io.axi.r.bits.id   := id
    io.axi.r.bits.resp := 0.U
    io.axi.r.bits.last := false.B
    io.axi.r.bits.data := 0.U

    io.host.req.valid     := false.B
    io.host.req.bits.write := false.B
    io.host.req.bits.addr  := curAddr
    io.host.req.bits.wdata := io.axi.w.bits.data
    io.host.req.bits.wstrb := io.axi.w.bits.strb

    // Read-data capture (latency 1; hold across R-channel back-pressure).
    val rdataReg = Reg(UInt(32.W))
    val justRead = RegNext(state === sReadReq && io.host.req.fire, false.B)
    when (justRead) { rdataReg := io.host.rdata }

    switch (state) {
        is (sIdle) {
            io.axi.aw.ready := true.B
            io.axi.ar.ready := !io.axi.aw.valid   // write priority
            when (io.axi.aw.fire) {
                curAddr := io.axi.aw.bits.addr
                len     := io.axi.aw.bits.len
                id      := io.axi.aw.bits.id
                state   := sWrite
            } .elsewhen (io.axi.ar.fire) {
                curAddr := io.axi.ar.bits.addr
                len     := io.axi.ar.bits.len
                id      := io.axi.ar.bits.id
                beat    := 0.U
                state   := sReadReq
            }
        }

        is (sWrite) {
            // Each write beat issues one host write.
            io.host.req.valid      := io.axi.w.valid
            io.host.req.bits.write := true.B
            io.axi.w.ready         := io.host.req.ready
            when (io.axi.w.fire) {
                curAddr := curAddr + 4.U
                when (io.axi.w.bits.last) { state := sResp }
            }
        }

        is (sResp) {
            io.axi.b.valid := true.B
            when (io.axi.b.fire) { state := sIdle }
        }

        is (sReadReq) {
            io.host.req.valid      := true.B
            io.host.req.bits.write := false.B
            when (io.host.req.fire) { state := sReadData }
        }

        is (sReadData) {
            io.axi.r.valid     := true.B
            io.axi.r.bits.data := Mux(justRead, io.host.rdata, rdataReg)
            io.axi.r.bits.last := beat === len
            when (io.axi.r.fire) {
                beat    := beat + 1.U
                curAddr := curAddr + 4.U
                when (beat === len) { state := sIdle } .otherwise { state := sReadReq }
            }
        }
    }
}
