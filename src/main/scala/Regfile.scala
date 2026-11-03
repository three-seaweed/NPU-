// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


// 32-entry, 32-bit register file with two combinational read ports and one
// write port. x0 is hardwired to zero (writes to it are dropped, reads return
// 0). The whole array is exposed on `dbg` for test introspection.
class Regfile(p: Parameters) extends Module {
    val io = IO(new Bundle {
        // Read: caller drives the address, regfile drives the data.
        val rs1Addr = Input(UInt(5.W))
        val rs2Addr = Input(UInt(5.W))
        val rs1 = new RegfileReadDataIO
        val rs2 = new RegfileReadDataIO

        // Write port.
        val wrValid = Input(Bool())
        val wrAddr  = Input(UInt(5.W))
        val wrData  = Input(UInt(32.W))

        val dbg = Output(Vec(32, UInt(32.W)))
    })

    val regs = RegInit(VecInit(Seq.fill(32)(0.U(32.W))))

    when (io.wrValid && io.wrAddr =/= 0.U) {
        regs(io.wrAddr) := io.wrData
    }

    // Write-to-read bypass: a read sees the value being written this same cycle.
    // In a pipelined consumer this lets an instruction writing back (WB) forward
    // its result to a younger instruction reading in the same cycle (its decode),
    // so no extra forwarding path is needed for that distance. x0 still reads 0.
    def read(addr: UInt): UInt =
        Mux(addr === 0.U, 0.U,
            Mux(io.wrValid && io.wrAddr === addr, io.wrData, regs(addr)))

    io.rs1.valid := true.B
    io.rs1.data  := read(io.rs1Addr)
    io.rs2.valid := true.B
    io.rs2.data  := read(io.rs2Addr)

    for (i <- 0 until 32) {
        io.dbg(i) := Mux(i.U === 0.U, 0.U, regs(i))
    }
}
