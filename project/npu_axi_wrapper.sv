// Copyright 2026 BUPT
//
// Vivado integration wrapper for the Chisel-generated NpuTop.
//
// Chisel/firtool emits bare ports (clock, aresetn, io_axi_*) with NO Xilinx
// interface metadata, so IP Integrator cannot tell which clock drives the AXI
// bus or at what frequency -> "S_AXI is not associated to any clock pin" and
// the empty-FREQ_HZ SmartConnect failure.
//
// This wrapper fixes that by:
//   1. renaming the flat Chisel ports to the standard AXI4 names (S_AXI_* /
//      M_AXI_*),
//   2. tagging every AXI signal with explicit aximm X_INTERFACE_INFO metadata
//      so IP Integrator does not have to guess the bus interfaces from names, and
//   3. tagging the clock/reset with X_INTERFACE attributes so the clock is
//      associated to both S_AXI and M_AXI and the reset is marked ACTIVE_LOW.
//
// The current NPU program model is "one kick per layer": PS writes the layer
// program/descriptor/input activations through S_AXI, writes CTRL.start once,
// then the core issues DMALD/DMAST/DMAWAIT instructions to drive the on-chip DMA
// engine over M_AXI. The legacy PS-issued DMA registers (+0x10..+0x20) are still
// exposed through S_AXI for preload/readback flows, but a layer program does not
// need the PS to push per-tile DMA descriptors.
//
// S_AXI APERTURE: NpuTop decodes the region from addr[20:19] (Npu.scala), so the
// slave window is 2 MB, not the size of any one TCM:
//   +0x00_0000 ITCM (32KB)   +0x08_0000 DTCM (512KB)   +0x10_0000 CTRL
// fpga/build_bd.tcl must therefore keep `set RANGE 2M`. A smaller aperture still
// elaborates and still lets the PS load ITCM, but CTRL falls outside the segment,
// so the kick never lands and the PS spins forever on STATUS.done.
//
// CTRL words (offsets from +0x10_0000; Npu.scala decodes addr[6:2], so the region
// is 128 B wide and every word below is inside the 2 MB aperture):
//   +0x00 CTRL (W)      +0x04 STATUS (R)
//   +0x10..+0x20        legacy PS DMA descriptor + status
//   +0x24..+0x48        per-kick performance counters:
//                       MATRIX, VECTOR, DMA busy; DTCM port-M / port-A transfers;
//                       range-token wait; DMAWAIT stall; KICKS; CYCLES; DMA_DTCM
// sw/xnpu_optimized_layer_test.c reads PERF_CYCLES (+0x44) after every kick and
// fails when it is zero, which is how it distinguishes a current bitstream from an
// older one whose CTRL decode stopped at +0x3c and returned 0 for the tail.
//
// Everything the layer pipeline added is internal to NpuTop and changes no
// boundary signal: the MatrixIdex / VectorIdex command queues, the DTCM range
// token scoreboard, BTFNT branch prediction, the MQCFG bf16 activation quantizer
// in the Matmul loader, and the split of DTCM into a true dual-port memory
// (Matrix on port 0; Vector / scalar load-store / DMA arbitrated on port 1).
// Vector and matrix instructions still arrive over this same S_AXI program path
// and their operands/results still live in DTCM, so this wrapper only has to keep
// mapping the generated NpuTop ports one-for-one.
//
// SOURCES: this design needs THREE files, not two --
//   fpga/npu_axi_wrapper.sv     (this file, the IP / block-design top)
//   generated/NpuTop.sv
//   generated/xnpu_dtcm_bank.v  (hand-written true-dual-port RAM, see below)
// Add all three to the IP. Omitting the third leaves xnpu_dtcm_bank as a black
// box and the DTCM reads back zero.
//
// Why the DTCM bank is a separate hand-written file: Vivado will not infer block
// RAM from what firtool emits for a two-write-port Chisel memory. An OOC run on
// xczu9eg listed only the ITCM in the block-RAM report and turned the whole
// 512 KiB DTCM into fabric (~1.05M 8-bit muxes, 50 GB synthesis RSS), which
// cannot place. xnpu_dtcm_bank.v is the UG901 true-dual-port template instead.
//
// Note for the utilization report: DTCM is 32 banks of 1024 x 128 bits. A RAMB36
// in true-dual-port mode is 1024 x 36, so each bank is exactly four RAMB36 with
// byte enables on both ports and no depth expansion -- expect 128 RAMB36 for the
// DTCM plus 8 for the 32 KB ITCM, and check that both appear in the "Block RAM:
// Final Mapping Report" before chasing timing.
//
// After regenerating NpuTop, confirm the instantiation below still matches:
//   awk '/^module NpuTop\(/,/^\);/' generated/NpuTop.sv
// and check every port name, direction and width against the port list here.
// The current interface is 60 ports: clock, aresetn and 58 io_* signals.
//
// FREQ_HZ is pinned to 99990005, the exact frequency the Zynq PS pl_clk0 can
// synthesize (its integer PLL divisors cannot hit a clean 100000000). This
// matches the driving clock so the SmartConnect clock-frequency DRC passes.
// NOTE: this ties the wrapper to a PS configured for ~100 MHz pl_clk0; if you
// retune pl_clk0, update this literal to match (or drop it and let FREQ_HZ
// propagate from the connected clock instead).
//
// Use THIS module (npu_axi_wrapper) as the IP / block-design top, not NpuTop.
// Add fpga/npu_axi_wrapper.sv, generated/NpuTop.sv and
// generated/xnpu_dtcm_bank.v as design sources.

// M_AXI: a 128-bit AXI4 *master* the shared on-chip DMA engine uses to burst
// weights/results directly to/from PS DRAM via a PS HP/HPC slave port. It is
// driven either by the legacy S_AXI DMA descriptor registers while the core is
// idle, or by core-issued DMALD/DMAST instructions while a layer program runs.
// The clock is associated to both S_AXI and M_AXI; M_AXI's master role is
// inferred from the AW/AR/W ports being outputs. Optional AXI signals
// (PROT/CACHE/LOCK/QOS/REGION) are omitted and tied off by the SmartConnect.
module npu_axi_wrapper (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 S_AXI_ACLK CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI:M_AXI, ASSOCIATED_RESET S_AXI_ARESETN, FREQ_HZ 99990005" *)
    input  wire        S_AXI_ACLK,

    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 S_AXI_ARESETN RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        S_AXI_ARESETN,

    // ---- AXI4 slave (32-bit, PS HPM -> NPU register/TCM/control window) ------
    // PS uses this plane before a kick to load ITCM, input activations, and layer
    // descriptors, and during/after a kick to poll CTRL/STATUS. ITCM/DTCM slave
    // accesses are back-pressured while the core owns them; CTRL remains live.
    // Write address channel
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWID" *)
    input  wire [3:0]  S_AXI_AWID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI, PROTOCOL AXI4, DATA_WIDTH 32, ADDR_WIDTH 32, ID_WIDTH 4, FREQ_HZ 99990005, HAS_BURST 1, HAS_LOCK 0, HAS_PROT 0, HAS_CACHE 0, HAS_QOS 0, HAS_REGION 0, SUPPORTS_NARROW_BURST 0, MAX_BURST_LENGTH 256" *)
    input  wire [31:0] S_AXI_AWADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWLEN" *)
    input  wire [7:0]  S_AXI_AWLEN,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWSIZE" *)
    input  wire [2:0]  S_AXI_AWSIZE,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWBURST" *)
    input  wire [1:0]  S_AXI_AWBURST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  wire        S_AXI_AWVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output wire        S_AXI_AWREADY,
    // Write data channel
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  wire [31:0] S_AXI_WDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  wire [3:0]  S_AXI_WSTRB,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WLAST" *)
    input  wire        S_AXI_WLAST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  wire        S_AXI_WVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output wire        S_AXI_WREADY,
    // Write response channel
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BID" *)
    output wire [3:0]  S_AXI_BID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output wire [1:0]  S_AXI_BRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output wire        S_AXI_BVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input  wire        S_AXI_BREADY,
    // Read address channel
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARID" *)
    input  wire [3:0]  S_AXI_ARID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input  wire [31:0] S_AXI_ARADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARLEN" *)
    input  wire [7:0]  S_AXI_ARLEN,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARSIZE" *)
    input  wire [2:0]  S_AXI_ARSIZE,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARBURST" *)
    input  wire [1:0]  S_AXI_ARBURST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input  wire        S_AXI_ARVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output wire        S_AXI_ARREADY,
    // Read data channel
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RID" *)
    output wire [3:0]  S_AXI_RID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output wire [31:0] S_AXI_RDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output wire [1:0]  S_AXI_RRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RLAST" *)
    output wire        S_AXI_RLAST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output wire        S_AXI_RVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input  wire        S_AXI_RREADY,

    // ---- AXI4 master (128-bit, NPU DMA -> PS HP/HPC slave port) --------------
    // One beat is one DTCM row (16 bytes). The DMA engine emits INCR bursts,
    // capped at 256 beats and split at 4KB boundaries in Dma.scala.
    // Write address channel
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWID" *)
    output wire [3:0]   M_AXI_AWID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXI, PROTOCOL AXI4, DATA_WIDTH 128, ADDR_WIDTH 32, ID_WIDTH 4, FREQ_HZ 99990005, HAS_BURST 1, HAS_LOCK 0, HAS_PROT 0, HAS_CACHE 0, HAS_QOS 0, HAS_REGION 0, SUPPORTS_NARROW_BURST 0, MAX_BURST_LENGTH 256" *)
    output wire [31:0]  M_AXI_AWADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWLEN" *)
    output wire [7:0]   M_AXI_AWLEN,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWSIZE" *)
    output wire [2:0]   M_AXI_AWSIZE,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWBURST" *)
    output wire [1:0]   M_AXI_AWBURST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWVALID" *)
    output wire         M_AXI_AWVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWREADY" *)
    input  wire         M_AXI_AWREADY,
    // Write data channel
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WDATA" *)
    output wire [127:0] M_AXI_WDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WSTRB" *)
    output wire [15:0]  M_AXI_WSTRB,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WLAST" *)
    output wire         M_AXI_WLAST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WVALID" *)
    output wire         M_AXI_WVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WREADY" *)
    input  wire         M_AXI_WREADY,
    // Write response channel
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BID" *)
    input  wire [3:0]   M_AXI_BID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BRESP" *)
    input  wire [1:0]   M_AXI_BRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BVALID" *)
    input  wire         M_AXI_BVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BREADY" *)
    output wire         M_AXI_BREADY,
    // Read address channel
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARID" *)
    output wire [3:0]   M_AXI_ARID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARADDR" *)
    output wire [31:0]  M_AXI_ARADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARLEN" *)
    output wire [7:0]   M_AXI_ARLEN,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARSIZE" *)
    output wire [2:0]   M_AXI_ARSIZE,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARBURST" *)
    output wire [1:0]   M_AXI_ARBURST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARVALID" *)
    output wire         M_AXI_ARVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARREADY" *)
    input  wire         M_AXI_ARREADY,
    // Read data channel
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RID" *)
    input  wire [3:0]   M_AXI_RID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RDATA" *)
    input  wire [127:0] M_AXI_RDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RRESP" *)
    input  wire [1:0]   M_AXI_RRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RLAST" *)
    input  wire         M_AXI_RLAST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RVALID" *)
    input  wire         M_AXI_RVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RREADY" *)
    output wire         M_AXI_RREADY
);

    NpuTop u_npu (
        .clock                (S_AXI_ACLK),
        .aresetn              (S_AXI_ARESETN),

        .io_axi_aw_ready      (S_AXI_AWREADY),
        .io_axi_aw_valid      (S_AXI_AWVALID),
        .io_axi_aw_bits_id    (S_AXI_AWID),
        .io_axi_aw_bits_addr  (S_AXI_AWADDR),
        .io_axi_aw_bits_len   (S_AXI_AWLEN),
        .io_axi_aw_bits_size  (S_AXI_AWSIZE),
        .io_axi_aw_bits_burst (S_AXI_AWBURST),

        .io_axi_w_ready       (S_AXI_WREADY),
        .io_axi_w_valid       (S_AXI_WVALID),
        .io_axi_w_bits_data   (S_AXI_WDATA),
        .io_axi_w_bits_strb   (S_AXI_WSTRB),
        .io_axi_w_bits_last   (S_AXI_WLAST),

        .io_axi_b_ready       (S_AXI_BREADY),
        .io_axi_b_valid       (S_AXI_BVALID),
        .io_axi_b_bits_id     (S_AXI_BID),
        .io_axi_b_bits_resp   (S_AXI_BRESP),

        .io_axi_ar_ready      (S_AXI_ARREADY),
        .io_axi_ar_valid      (S_AXI_ARVALID),
        .io_axi_ar_bits_id    (S_AXI_ARID),
        .io_axi_ar_bits_addr  (S_AXI_ARADDR),
        .io_axi_ar_bits_len   (S_AXI_ARLEN),
        .io_axi_ar_bits_size  (S_AXI_ARSIZE),
        .io_axi_ar_bits_burst (S_AXI_ARBURST),

        .io_axi_r_ready       (S_AXI_RREADY),
        .io_axi_r_valid       (S_AXI_RVALID),
        .io_axi_r_bits_id     (S_AXI_RID),
        .io_axi_r_bits_data   (S_AXI_RDATA),
        .io_axi_r_bits_resp   (S_AXI_RRESP),
        .io_axi_r_bits_last   (S_AXI_RLAST),

        // ---- DMA master ----
        .io_maxi_aw_ready     (M_AXI_AWREADY),
        .io_maxi_aw_valid     (M_AXI_AWVALID),
        .io_maxi_aw_bits_id   (M_AXI_AWID),
        .io_maxi_aw_bits_addr (M_AXI_AWADDR),
        .io_maxi_aw_bits_len  (M_AXI_AWLEN),
        .io_maxi_aw_bits_size (M_AXI_AWSIZE),
        .io_maxi_aw_bits_burst(M_AXI_AWBURST),

        .io_maxi_w_ready      (M_AXI_WREADY),
        .io_maxi_w_valid      (M_AXI_WVALID),
        .io_maxi_w_bits_data  (M_AXI_WDATA),
        .io_maxi_w_bits_strb  (M_AXI_WSTRB),
        .io_maxi_w_bits_last  (M_AXI_WLAST),

        .io_maxi_b_ready      (M_AXI_BREADY),
        .io_maxi_b_valid      (M_AXI_BVALID),
        .io_maxi_b_bits_id    (M_AXI_BID),
        .io_maxi_b_bits_resp  (M_AXI_BRESP),

        .io_maxi_ar_ready     (M_AXI_ARREADY),
        .io_maxi_ar_valid     (M_AXI_ARVALID),
        .io_maxi_ar_bits_id   (M_AXI_ARID),
        .io_maxi_ar_bits_addr (M_AXI_ARADDR),
        .io_maxi_ar_bits_len  (M_AXI_ARLEN),
        .io_maxi_ar_bits_size (M_AXI_ARSIZE),
        .io_maxi_ar_bits_burst(M_AXI_ARBURST),

        .io_maxi_r_ready      (M_AXI_RREADY),
        .io_maxi_r_valid      (M_AXI_RVALID),
        .io_maxi_r_bits_id    (M_AXI_RID),
        .io_maxi_r_bits_data  (M_AXI_RDATA),
        .io_maxi_r_bits_resp  (M_AXI_RRESP),
        .io_maxi_r_bits_last  (M_AXI_RLAST)
    );

endmodule
