// Copyright 2026 BUPT

package xnpu

import chisel3._

// Synthesis top level for Zynq integration.
//
// The Zynq `proc_sys_reset` block drives `peripheral_aresetn`, which is
// active-LOW, while Chisel's default reset is active-HIGH. This wrapper exposes
// the AXI-standard active-low reset `aresetn` on the boundary and inverts it
// for the inner Npu, so the PL reset can wire directly to the PS reset with the
// correct polarity (no inversion needed in the Verilog/block design).
//
// RawModule is used so clock and reset are explicit ports we fully control,
// rather than the implicit clock/reset a normal Module would emit.
class NpuTop(p: Parameters) extends RawModule {
    val clock   = IO(Input(Clock()))
    val aresetn = IO(Input(Bool()))                  // active-low (peripheral_aresetn)
    val io      = IO(new Bundle {
        val axi  = new Axi4SlaveIO(32, 32, 4)
        val maxi = new Axi4MasterIO(32, p.DtcmRowBits, 4)   // DMA master -> PS HP slave
    })

    // Invert active-low -> active-high. Kept as a Bool so the inner design keeps
    // its original *synchronous* reset semantics (the same behaviour the tests
    // exercise). Use `(!aresetn).asAsyncReset` instead if asynchronous assertion
    // is desired.
    withClockAndReset(clock, !aresetn) {
        val npu = Npu(p)
        io.axi  <> npu.io.axi
        io.maxi <> npu.io.maxi
    }
}
