// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


case class Parameters(
    int8Engines: Int = 1,        // int8 脉动阵列实例数：1 = legacy 单引擎（默认），
                                 // 2 = 双实例（C5 起的 "dual8"/"full" 变体）
    bf16Matmul: Boolean = false  // true = FP.* 层由 bf16 8×8 核接管（C6）；
                                 // false = 无 FP 层（纯 int8 形状）
) {
    require(int8Engines == 1 || int8Engines == 2, "int8Engines must be 1 or 2")
    // 默认 Parameters() = legacy 形状（单 int8 引擎，无 FP 层）：全部既有
    // int8 spec 用默认值构造，全绿即"int8 逐位不变"硬门的证明——新增功能
    // 一律以休眠数据通路/独立变体落地，不动默认行为。FP.* 一律走 bf16Matmul
    // = true（fp32 4×4 tier 已于 2026-11-01 删除）。
    val AddrBits = 32
    val AddWidth = AddrBits.W

    val DataBits = 32
    val DataWidth = DataBits.W

    val InstructionBits = 32
    val InstructionWidth = InstructionBits.W

    val ByteBits = 8
    val ByteWidth = ByteBits.W
    val WordSize = Math.ceil(DataBits / ByteBits).toInt

    val PhysicalRegisters = 32
    val PhysicalRegisterAddrBits = log2Up(PhysicalRegisters)
    val PhysicalRegisterAddrWidth = PhysicalRegisterAddrBits.W

    // Sizes are compile-time constants (plain BigInt), not hardware literals,
    // so Parameters() can be constructed outside a Module elaboration.
    // The mixed vector/matrix data path keeps bf16 vector data, int8 matrix inputs,
    // int32 matrix results, and software-managed workspaces resident. DTCM is 512KB;
    // the Npu region decode uses addr[20:19] for a >=512KB window (addr[18:0]).
    val ItcmSizeInBytes = BigInt(1) << 15 // 32KB
    val ItcmSizeInWords = ItcmSizeInBytes / 4

    val DtcmSizeInBytes = BigInt(1) << 19 // 512KB
    val DtcmSizeInWords = DtcmSizeInBytes / 4

    val MemorySizeInBytes = BigInt(1) << 32 // 4GB
    val MemorySizeInWords = MemorySizeInBytes / 4

    // NxN systolic-array matrix-multiply unit.
    val MatrixDim = 16                      // NxN matrix / array dimension
    val ElemBits = 8                        // int8 matrix elements
    val ElemsPerWord = DataBits / ElemBits  // 4 int8 elements packed per 32-bit word
    val AccBits = 32                        // int32 accumulator
    // WS partial-sum chain width: one K=16 tile keeps the widest possible
    // column sum under 2^19 (16 * 127 * 127), so a 20-bit down-chain cannot
    // overflow; longer K is the caller's business (WS-MATMULA repeats into a
    // full-width int32 output buffer -- gemmini §5.3 contract, internalised).
    val PsBits = 20

    // Fp32FmaPipe stage count -- the bf16 tier's PE pipeline latency
    // (fp32matmul package: the fp32-arithmetic FMA engine bf16 PEs reuse).
    val Fp32PipeLatency = 3 //Fp32FmaPipe.LATENCY

    // bf16 NxN tier (C6, Bf16MatmulUnit): FP.* served by a bf16-in /
    // fp32-accumulate / bf16-out array when bf16Matmul = true. Shares
    // Fp32PipeLatency with the fp32 tier (same Fp32FmaPipe PEs). Dim 8 from
    // 2026-11-01 (LUT diet: 256-PE 16x16 could not fit ZCU102); one 8-bf16
    // row is exactly one 128-bit DTCM beat, so loads/stores are 1 beat/row.
    val Bf16MatrixDim = 8
    val Bf16ElemBits  = 16

    // Physical wide-memory beat: the DTCM row and the AXI-master beat are a fixed
    // 128-bit word, INDEPENDENT of MatrixDim. One beat = WordsPerRow 32-bit words =
    // DtcmRowBytes bytes = 16 packed int8 lanes (or 8 bf16 lanes). Keeping this fixed
    // (rather than derived from MatrixDim) lets the systolic array scale while the
    // DMA/AXI master and DTCM SRAM stay 128-bit wide (Zynq HP-port friendly).
    val DtcmRowBits  = 128
    val DtcmRowBytes = DtcmRowBits / ByteBits      // 16
    val WordsPerRow  = DtcmRowBits / DataBits       // 4 (32-bit words per beat)

    // Logical systolic-array row: MatrixDim int8 operands. When the array is wider
    // than one physical beat, one array row spans BeatsPerArrayRow consecutive beats;
    // the matrix unit assembles (load) / spills (store) a row over that many DTCM
    // accesses. BeatsPerArrayRow == 1 at MatrixDim=16 (row == beat, the original case).
    val ArrayRowBits     = MatrixDim * ElemBits        // 128 (n=16), 256 (n=32)
    val ArrayRowBytes    = ArrayRowBits / ByteBits
    val ArrayRowWords    = ArrayRowBytes / WordSize    // 32-bit words per matrix row (4, 8)
    val BeatsPerArrayRow = ArrayRowBits / DtcmRowBits  // 1 (n=16), 2 (n=32)

    // Cycles to fully stream a NxN output-stationary array: 3N-2 feeds, +1 to settle.
    val SystolicRunCycles = 3 * MatrixDim - 2

    // DMA: max AXI read bursts kept in flight at once (hides DDR read latency across
    // burst boundaries / 4KB splits on the memory-bound weight-load path).
    val DmaMaxOutstanding = 4

    // DTCM range-token scoreboard depth (Alu). The DMA descriptor FIFO carries a
    // token slot id per entry and the id width crosses the Core/Npu boundary, so
    // the capacity lives here instead of inside Alu (a 2026-10-30 experiment at
    // 64 slots measured zero counter changes: the "32 caps the dependency ring"
    // hypothesis was falsified).
    val TokenSlots = 32

    // Vector core: bf16 (sign1/exp8/mant7) elements packed in the wide DTCM row.
    // One 128-bit row holds Bf16LanesPerRow bf16 lanes; lane k in bits [16k+15:16k].
    // Reductions accumulate in fp32 internally for precision.
    val Bf16Bits        = 16
    val Bf16MantBits    = 7
    val Bf16ExpBits     = 8
    val Bf16LanesPerRow = DtcmRowBits / Bf16Bits   // 8 bf16 per 128-bit row
    val F32Bits         = 32
    val F32MantBits     = 23
    val F32ExpBits      = 8
    val F32ExpBias      = 127
}
