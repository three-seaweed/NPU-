// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


// Instruction encoding constants shared by the decoder and the (test-side)
// assembler in object Encode.
object Opcodes {
    val OP_IMM  = "b0010011".U(7.W) // ADDI/ANDI/.../SLLI/SRLI/SRAI (funct3 selects)
    val OP      = "b0110011".U(7.W) // ADD/SUB/AND/.../SLT (funct3+funct7 select)
    val LUI     = "b0110111".U(7.W) // LUI
    val AUIPC   = "b0010111".U(7.W) // AUIPC
    val BRANCH  = "b1100011".U(7.W) // BEQ/BNE/BLT/BGE/BLTU/BGEU
    val JAL     = "b1101111".U(7.W) // JAL
    val JALR    = "b1100111".U(7.W) // JALR (funct3=0)
    val LOAD    = "b0000011".U(7.W) // LW (funct3=010)
    val STORE   = "b0100011".U(7.W) // SW (funct3=010)
    val FENCE   = "b0001111".U(7.W) // FENCE (decoded as a NOP)
    val CUSTOM0 = "b0001011".U(7.W) // matrix ops (LMA/LMB/MATMUL/SM); base = rs1+imm
    val CUSTOM1 = "b0101011".U(7.W) // vector ops (VSET*/V{ADD,MUL,...}/VSUM/...)

    // funct3 selecting the matrix operation under CUSTOM0.
    val F3_LMA        = 0
    val F3_LMB        = 1
    val F3_MATMUL     = 2
    val F3_SM         = 3
    val F3_MATMUL_ACC = 4
    // F3_MATMUL with a nonzero funct7 reserves Matrix configuration encodings.
    // MQCFG: rs1 is raw fp32 activation scale; rs2 bit0 enables bf16-A loader
    // quantization, rs2[15:1] carries the group index and rs2[31:16] carries
    // the K-group size metadata.
    val MF7_MQCFG     = 1
    val MF7_FP_MATMUL  = 2  // f3=MATMUL slot: FP.MATMUL, fp32 C = A x B
    val MF7_FP_MATMULA = 1  // f3=MATMUL_ACC slot: FP.MATMULA, fp32 C += A x B
    val MF7_FP_SM      = 1  // f3=SM slot: FP.SM, 4x4 fp32 store
    // MCFG2: engine/dataflow configuration riding the MATMUL slot. rs2 bit0
    // selects the int8 engine (single-engine builds accept only 0), bit1 the
    // dataflow (0=OS, 1=WS) and bit2 ws_keep (weights stay resident across
    // MATMULA so B is neither re-read from DTCM nor re-captured). Accepting a
    // MCFG2 always invalidates resident weights (clear+flush+wClr pulse) --
    // switching dataflows with stale weights in the array is a silent-corruption
    // hazard gemmini §6.3.1 closes with a flush; so do we.
    val MF7_MCFG2      = 3  // f3=MATMUL slot: MCFG2, queued matrix config
    // Core-issued DMA (custom-0): stream rows between DRAM and DTCM. dram=GPR[rs1],
    // dtcm=GPR[rs2], rows=S-type imm. DMALD/DMAST are fire-and-forget; DMAWAIT blocks.
    val F3_DMALD      = 5
    val F3_DMAST      = 6
    val F3_DMAWAIT    = 7

    // CUSTOM1 vector ops: funct3 selects a class, funct7 the specific op.
    val VF3_CONFIG  = 0   // VSETA/VSETB/VSETD/VSETLEN/VSETS (value in rs1)
    val VF3_COMPUTE = 1   // elementwise / scalar / activation (dst <- DTCM)
    val VF3_REDUCE  = 2   // VSUM/VMAX/VDOT (scalar result -> rd)
    val VF3_COMPOUND = 3  // VSOFTEXP (dst + scalar result -> rd)

    val VF7_SETA = 0; val VF7_SETB = 1; val VF7_SETD = 2; val VF7_SETLEN = 3; val VF7_SETS = 4
    val VF7_ADD = 0; val VF7_SUB = 1; val VF7_MUL = 2; val VF7_FMA = 3
    val VF7_SMUL = 4; val VF7_SADD = 5; val VF7_EXP = 6; val VF7_RECIP = 7; val VF7_RSQRT = 8
    // Fundamental format/data-path operations. Quantization policy is software.
    val VF7_ABS = 9; val VF7_CVT_I8 = 10; val VF7_CVT_BF16 = 11
    val VF7_SUM = 0; val VF7_MAX = 1; val VF7_DOT = 2
    val VF7_SOFTEXP = 0
}

// Result of decoding a 32-bit instruction word.
class DecodedInst extends Bundle {
    val op       = AluOp()
    val rd       = UInt(5.W)
    val rs1      = UInt(5.W)
    val rs2      = UInt(5.W)
    val imm      = UInt(32.W)   // scalar immediate, or matrix/vector base offset
    val isMatrix = Bool()
    val isVector = Bool()
    val isMem    = Bool()       // scalar DTCM load/store (LW/SW; blocking-in-EX)
    val isDma    = Bool()       // core-issued DMA load/store (DMALD/DMAST; fire-and-forget)
    val isDmaWait = Bool()      // DMAWAIT: block in EX until the DMA engine is idle
    val isBranch = Bool()       // conditional branch (resolved in EX)
    val isJump   = Bool()       // JAL / JALR (unconditional, writes link = pc+4)
    val isJalr   = Bool()       // JALR specifically (target from rs1+imm)
    val isAuipc  = Bool()       // AUIPC (result = pc + imm, computed in Control)
    val writesRd = Bool()
    val halt     = Bool()       // inst == 0x00000000 sentinel
    val legal    = Bool()
}

object Decode {
    def apply(inst: UInt): DecodedInst = {
        val d = Wire(new DecodedInst)

        val opcode = inst(6, 0)
        val funct3 = inst(14, 12)
        val funct7 = inst(31, 25)

        // Immediates (all sign-extended to 32 bits where applicable).
        val immI = Cat(Fill(20, inst(31)), inst(31, 20))   // I-type
        val immU = Cat(inst(31, 12), 0.U(12.W))            // U-type (LUI/AUIPC)
        val immS = Cat(Fill(20, inst(31)), inst(31, 25), inst(11, 7))           // S-type
        val immB = Cat(Fill(19, inst(31)), inst(31), inst(7),
                       inst(30, 25), inst(11, 8), 0.U(1.W))                      // B-type
        val immJ = Cat(Fill(11, inst(31)), inst(31), inst(19, 12), inst(20),
                       inst(30, 21), 0.U(1.W))                                   // J-type
        val bit30 = inst(30)   // distinguishes ADD/SUB, SRL/SRA, SRLI/SRAI

        d.rd  := inst(11, 7)
        d.rs1 := inst(19, 15)
        d.rs2 := inst(24, 20)

        // Defaults.
        d.op       := AluOp.ADDI
        d.imm      := 0.U
        d.isMatrix := false.B
        d.isVector := false.B
        d.isMem    := false.B
        d.isDma    := false.B
        d.isDmaWait := false.B
        d.isBranch := false.B
        d.isJump   := false.B
        d.isJalr   := false.B
        d.isAuipc  := false.B
        d.writesRd := false.B
        d.halt     := inst === 0.U
        d.legal    := false.B

        when (inst === 0.U) {
            // HALT sentinel: legal, does nothing.
            d.legal := true.B
        } .elsewhen (opcode === Opcodes.OP_IMM) {
            d.imm := immI; d.writesRd := true.B; d.legal := true.B
            switch (funct3) {
                is ("b000".U) { d.op := AluOp.ADDI }
                is ("b010".U) { d.op := AluOp.SLTI }
                is ("b011".U) { d.op := AluOp.SLTIU }
                is ("b100".U) { d.op := AluOp.XORI }
                is ("b110".U) { d.op := AluOp.ORI }
                is ("b111".U) { d.op := AluOp.ANDI }
                is ("b001".U) { d.op := AluOp.SLLI }
                is ("b101".U) { d.op := Mux(bit30, AluOp.SRAI, AluOp.SRLI) }
            }
        } .elsewhen (opcode === Opcodes.OP) {
            d.writesRd := true.B; d.legal := true.B
            switch (funct3) {
                is ("b000".U) { d.op := Mux(bit30, AluOp.SUB, AluOp.ADD) }
                is ("b001".U) { d.op := AluOp.SLL }
                is ("b010".U) { d.op := AluOp.SLT }
                is ("b011".U) { d.op := AluOp.SLTU }
                is ("b100".U) { d.op := AluOp.XOR }
                is ("b101".U) { d.op := Mux(bit30, AluOp.SRA, AluOp.SRL) }
                is ("b110".U) { d.op := AluOp.OR }
                is ("b111".U) { d.op := AluOp.AND }
            }
        } .elsewhen (opcode === Opcodes.LUI) {
            d.op := AluOp.LUI; d.imm := immU; d.writesRd := true.B; d.legal := true.B
        } .elsewhen (opcode === Opcodes.AUIPC) {
            d.op := AluOp.AUIPC; d.imm := immU; d.isAuipc := true.B
            d.writesRd := true.B; d.legal := true.B
        } .elsewhen (opcode === Opcodes.BRANCH) {
            d.isBranch := true.B; d.imm := immB; d.legal := true.B
            switch (funct3) {
                is ("b000".U) { d.op := AluOp.BEQ }
                is ("b001".U) { d.op := AluOp.BNE }
                is ("b100".U) { d.op := AluOp.BLT }
                is ("b101".U) { d.op := AluOp.BGE }
                is ("b110".U) { d.op := AluOp.BLTU }
                is ("b111".U) { d.op := AluOp.BGEU }
            }
        } .elsewhen (opcode === Opcodes.JAL) {
            d.op := AluOp.JAL; d.imm := immJ; d.isJump := true.B
            d.writesRd := true.B; d.legal := true.B
        } .elsewhen (opcode === Opcodes.JALR && funct3 === 0.U) {
            d.op := AluOp.JALR; d.imm := immI; d.isJump := true.B; d.isJalr := true.B
            d.writesRd := true.B; d.legal := true.B
        } .elsewhen (opcode === Opcodes.LOAD && funct3 === "b010".U) {
            d.op := AluOp.LW; d.imm := immI; d.isMem := true.B
            d.writesRd := true.B; d.legal := true.B
        } .elsewhen (opcode === Opcodes.STORE && funct3 === "b010".U) {
            d.op := AluOp.SW; d.imm := immS; d.isMem := true.B; d.legal := true.B
        } .elsewhen (opcode === Opcodes.FENCE) {
            // FENCE: this in-order single-issue core has no reordering to fence;
            // decode as a legal NOP (ADDI x0, x0, 0).
            d.op := AluOp.ADDI; d.rd := 0.U; d.legal := true.B
        } .elsewhen (opcode === Opcodes.CUSTOM0) {
            // Matrix ops: DTCM base = GPR[rs1] + signed 12-bit offset (immI).
            d.isMatrix := true.B; d.imm := immI; d.legal := true.B
            switch (funct3) {
                is (Opcodes.F3_LMA.U)        { d.op := AluOp.LMA }
                is (Opcodes.F3_LMB.U)        { d.op := AluOp.LMB }
                is (Opcodes.F3_MATMUL.U) {
                    when (funct7 === Opcodes.MF7_MQCFG.U) {
                        d.op := AluOp.MQCFG
                        d.imm := 0.U
                    }.elsewhen(funct7 === Opcodes.MF7_FP_MATMUL.U){
                        d.op := AluOp.FP_MATMUL
                        d.imm := 0.U
                    }.elsewhen(funct7 === Opcodes.MF7_MCFG2.U){
                        d.op := AluOp.MCFG2
                        d.imm := 0.U
                    }.otherwise {
                        d.op := AluOp.MATMUL
                    }
                }
                is (Opcodes.F3_SM.U){
                    when(funct7 === Opcodes.MF7_FP_SM.U){
                        d.op := AluOp.FP_SM
                        d.imm := 0.U
                    }.otherwise { 
                        d.op := AluOp.SM 
                    }
                }
                
                is (Opcodes.F3_MATMUL_ACC.U) {
                    when (funct7 === Opcodes.MF7_FP_MATMULA.U){
                        d.op :=AluOp.FP_MATMULA;
                        d.imm :=0.U
                    } .otherwise{ d.op := AluOp.MATMULA 
                    }
                }
                // DMA ops are NOT matrix ops: rs1/rs2 pass raw (dram/dtcm bases),
                // S-type imm carries the row count. Override the isMatrix default.
                is (Opcodes.F3_DMALD.U) {
                    d.op := AluOp.DMALD; d.isMatrix := false.B; d.isDma := true.B; d.imm := immS
                }
                is (Opcodes.F3_DMAST.U) {
                    d.op := AluOp.DMAST; d.isMatrix := false.B; d.isDma := true.B; d.imm := immS
                }
                is (Opcodes.F3_DMAWAIT.U) {
                    d.op := AluOp.DMAWAIT; d.isMatrix := false.B; d.isDmaWait := true.B
                }
            }
        } .elsewhen (opcode === Opcodes.CUSTOM1) {
            d.isVector := true.B; d.legal := true.B
            switch (funct3) {
                is (Opcodes.VF3_CONFIG.U) {
                    switch (funct7) {
                        is (Opcodes.VF7_SETA.U)   { d.op := AluOp.VSETA }
                        is (Opcodes.VF7_SETB.U)   { d.op := AluOp.VSETB }
                        is (Opcodes.VF7_SETD.U)   { d.op := AluOp.VSETD }
                        is (Opcodes.VF7_SETLEN.U) { d.op := AluOp.VSETLEN }
                        is (Opcodes.VF7_SETS.U)   { d.op := AluOp.VSETS }
                    }
                }
                is (Opcodes.VF3_COMPUTE.U) {
                    switch (funct7) {
                        is (Opcodes.VF7_ADD.U)   { d.op := AluOp.VADD }
                        is (Opcodes.VF7_SUB.U)   { d.op := AluOp.VSUB }
                        is (Opcodes.VF7_MUL.U)   { d.op := AluOp.VMUL }
                        is (Opcodes.VF7_FMA.U)   { d.op := AluOp.VFMA }
                        is (Opcodes.VF7_SMUL.U)  { d.op := AluOp.VSMUL }
                        is (Opcodes.VF7_SADD.U)  { d.op := AluOp.VSADD }
                        is (Opcodes.VF7_EXP.U)   { d.op := AluOp.VEXP }
                        is (Opcodes.VF7_RECIP.U) { d.op := AluOp.VRECIP }
                        is (Opcodes.VF7_RSQRT.U) { d.op := AluOp.VRSQRT }
                        is (Opcodes.VF7_ABS.U)      { d.op := AluOp.VABS }
                        is (Opcodes.VF7_CVT_I8.U)   { d.op := AluOp.VCVT_I8 }
                        is (Opcodes.VF7_CVT_BF16.U) { d.op := AluOp.VCVT_BF16 }
                    }
                }
                is (Opcodes.VF3_REDUCE.U) {
                    d.writesRd := true.B
                    switch (funct7) {
                        is (Opcodes.VF7_SUM.U) { d.op := AluOp.VSUM }
                        is (Opcodes.VF7_MAX.U) { d.op := AluOp.VMAX }
                        is (Opcodes.VF7_DOT.U) { d.op := AluOp.VDOT }
                    }
                }
                is (Opcodes.VF3_COMPOUND.U) {
                    d.writesRd := true.B
                    switch (funct7) {
                        is (Opcodes.VF7_SOFTEXP.U) { d.op := AluOp.VSOFTEXP }
                    }
                }
            }
        }

        d
    }
}

// Scala-side assembler: build 32-bit instruction words for tests / ITCM
// initialisation. All values are plain BigInt.
object Encode {
    // ---- RV32I integer ops ---------------------------------------------------
    private def rtype(f7: Int, f3: Int, rd: Int, rs1: Int, rs2: Int): BigInt =
        (BigInt(f7) << 25) | (BigInt(rs2) << 20) | (BigInt(rs1) << 15) |
        (BigInt(f3) << 12) | (BigInt(rd) << 7) | 0x33
    private def itype(f3: Int, rd: Int, rs1: Int, imm: Int): BigInt =
        ((BigInt(imm) & 0xfff) << 20) | (BigInt(rs1) << 15) |
        (BigInt(f3) << 12) | (BigInt(rd) << 7) | 0x13

    def add(rd: Int, rs1: Int, rs2: Int): BigInt  = rtype(0x00, 0, rd, rs1, rs2)
    def sub(rd: Int, rs1: Int, rs2: Int): BigInt  = rtype(0x20, 0, rd, rs1, rs2)
    def sll(rd: Int, rs1: Int, rs2: Int): BigInt  = rtype(0x00, 1, rd, rs1, rs2)
    def slt(rd: Int, rs1: Int, rs2: Int): BigInt  = rtype(0x00, 2, rd, rs1, rs2)
    def sltu(rd: Int, rs1: Int, rs2: Int): BigInt = rtype(0x00, 3, rd, rs1, rs2)
    def xor(rd: Int, rs1: Int, rs2: Int): BigInt  = rtype(0x00, 4, rd, rs1, rs2)
    def srl(rd: Int, rs1: Int, rs2: Int): BigInt  = rtype(0x00, 5, rd, rs1, rs2)
    def sra(rd: Int, rs1: Int, rs2: Int): BigInt  = rtype(0x20, 5, rd, rs1, rs2)
    def or(rd: Int, rs1: Int, rs2: Int): BigInt   = rtype(0x00, 6, rd, rs1, rs2)
    def and(rd: Int, rs1: Int, rs2: Int): BigInt  = rtype(0x00, 7, rd, rs1, rs2)

    def addi(rd: Int, rs1: Int, imm: Int): BigInt  = itype(0, rd, rs1, imm)
    def slti(rd: Int, rs1: Int, imm: Int): BigInt  = itype(2, rd, rs1, imm)
    def sltiu(rd: Int, rs1: Int, imm: Int): BigInt = itype(3, rd, rs1, imm)
    def xori(rd: Int, rs1: Int, imm: Int): BigInt  = itype(4, rd, rs1, imm)
    def ori(rd: Int, rs1: Int, imm: Int): BigInt   = itype(6, rd, rs1, imm)
    def andi(rd: Int, rs1: Int, imm: Int): BigInt  = itype(7, rd, rs1, imm)
    def slli(rd: Int, rs1: Int, sh: Int): BigInt   = itype(1, rd, rs1, sh & 0x1f)
    def srli(rd: Int, rs1: Int, sh: Int): BigInt   = itype(5, rd, rs1, sh & 0x1f)
    def srai(rd: Int, rs1: Int, sh: Int): BigInt   = itype(5, rd, rs1, (sh & 0x1f) | 0x400)

    // imm20 is the raw [31:12] field; the register receives imm20 << 12.
    def lui(rd: Int, imm20: Int): BigInt =
        ((BigInt(imm20) & 0xfffff) << 12) | (BigInt(rd) << 7) | 0x37
    def auipc(rd: Int, imm20: Int): BigInt =
        ((BigInt(imm20) & 0xfffff) << 12) | (BigInt(rd) << 7) | 0x17

    // ---- Control flow --------------------------------------------------------
    // Branch/jump immediates are byte offsets relative to the instruction's PC.
    private def branch(f3: Int, rs1: Int, rs2: Int, off: Int): BigInt = {
        val i = BigInt(off)
        (((i >> 12) & 0x1) << 31) | (((i >> 5) & 0x3f) << 25) | (BigInt(rs2) << 20) |
        (BigInt(rs1) << 15) | (BigInt(f3) << 12) | (((i >> 1) & 0xf) << 8) |
        (((i >> 11) & 0x1) << 7) | 0x63
    }
    def beq(rs1: Int, rs2: Int, off: Int): BigInt  = branch(0, rs1, rs2, off)
    def bne(rs1: Int, rs2: Int, off: Int): BigInt  = branch(1, rs1, rs2, off)
    def blt(rs1: Int, rs2: Int, off: Int): BigInt  = branch(4, rs1, rs2, off)
    def bge(rs1: Int, rs2: Int, off: Int): BigInt  = branch(5, rs1, rs2, off)
    def bltu(rs1: Int, rs2: Int, off: Int): BigInt = branch(6, rs1, rs2, off)
    def bgeu(rs1: Int, rs2: Int, off: Int): BigInt = branch(7, rs1, rs2, off)
    def jal(rd: Int, off: Int): BigInt = {
        val i = BigInt(off)
        (((i >> 20) & 0x1) << 31) | (((i >> 1) & 0x3ff) << 21) | (((i >> 11) & 0x1) << 20) |
        (((i >> 12) & 0xff) << 12) | (BigInt(rd) << 7) | 0x6f
    }
    def jalr(rd: Int, rs1: Int, imm: Int): BigInt =
        ((BigInt(imm) & 0xfff) << 20) | (BigInt(rs1) << 15) | (BigInt(rd) << 7) | 0x67

    // ---- Scalar DTCM word load/store -----------------------------------------
    def lw(rd: Int, rs1: Int, imm: Int): BigInt =
        ((BigInt(imm) & 0xfff) << 20) | (BigInt(rs1) << 15) | (2 << 12) | (BigInt(rd) << 7) | 0x03
    def sw(rs2: Int, rs1: Int, imm: Int): BigInt = {
        val i = BigInt(imm)
        (((i >> 5) & 0x7f) << 25) | (BigInt(rs2) << 20) | (BigInt(rs1) << 15) |
        (2 << 12) | ((i & 0x1f) << 7) | 0x23
    }

    // ---- Matrix ops (custom-0): DTCM base = GPR[rs1] + signed offset ----------
    private def matrix(f3: Int, rd: Int, rs1: Int, off: Int): BigInt =
        ((BigInt(off) & 0xfff) << 20) | (BigInt(rs1) << 15) |
        (BigInt(f3) << 12) | (BigInt(rd) << 7) | 0x0b
    private def matrixR(f3: Int, f7: Int, rs1: Int, rs2: Int): BigInt =
        (BigInt(f7) << 25) | (BigInt(rs2) << 20) | (BigInt(rs1) << 15) |
        (BigInt(f3) << 12) | 0x0b
    // Native fp32-tier encoders: funct7 sits in imm[11:5]; no offset field.
    private def matrixF(f3: Int, f7: Int, rs1: Int): BigInt =
        (BigInt(f7) << 25) | (BigInt(rs1) << 15) | (BigInt(f3) << 12) | 0x0b
    def lma(rs1: Int, off: Int = 0): BigInt = matrix(Opcodes.F3_LMA, 0, rs1, off)
    def lmb(rs1: Int, off: Int = 0): BigInt = matrix(Opcodes.F3_LMB, 0, rs1, off)
    def matmul(): BigInt    = matrix(Opcodes.F3_MATMUL, 0, 0, 0)
    def matmulAcc(): BigInt = matrix(Opcodes.F3_MATMUL_ACC, 0, 0, 0)
    def sm(rs1: Int, off: Int = 0): BigInt  = matrix(Opcodes.F3_SM, 0, rs1, off)
    // Matrix bf16-activation quantizer configuration. scaleReg contains raw fp32
    // multiplication scale; cfgReg bit0 enables it, bits[15:1] identify the
    // group partial and bits[31:16] hold a group size (a multiple of MatrixDim,
    // zero means one MatrixDim tile).
    def mqcfg(scaleReg: Int, cfgReg: Int): BigInt =
        matrixR(Opcodes.F3_MATMUL, Opcodes.MF7_MQCFG, scaleReg, cfgReg)
    // Engine/dataflow configuration: cfgReg bit0=engine (0 only until C5),
    // bit1=dataflow (0=OS 1=WS), bit2=ws_keep. rs1 unused (0).
    def mcfg2(cfgReg: Int): BigInt =
        matrixR(Opcodes.F3_MATMUL, Opcodes.MF7_MCFG2, 0, cfgReg)

    def fpMatmul(): BigInt = matrixF(Opcodes.F3_MATMUL, Opcodes.MF7_FP_MATMUL, 0)
    def fpMatmulAcc(): BigInt  = matrixF(Opcodes.F3_MATMUL_ACC, Opcodes.MF7_FP_MATMULA, 0)
    def fpSm(rs1: Int): BigInt = matrixF(Opcodes.F3_SM, Opcodes.MF7_FP_SM, rs1)
    // ---- Core-issued DMA (custom-0): S-type imm = row count -------------------
    // dram=GPR[rs1], dtcm=GPR[rs2], rows=imm (one row = DtcmRowBytes = one AXI beat).
    private def dmaInst(f3: Int, rs1: Int, rs2: Int, rows: Int): BigInt = {
        val i = BigInt(rows)
        (((i >> 5) & 0x7f) << 25) | (BigInt(rs2) << 20) | (BigInt(rs1) << 15) |
        (BigInt(f3) << 12) | ((i & 0x1f) << 7) | 0x0b
    }
    def dmald(rs1: Int, rs2: Int, rows: Int): BigInt = dmaInst(Opcodes.F3_DMALD, rs1, rs2, rows)
    def dmast(rs1: Int, rs2: Int, rows: Int): BigInt = dmaInst(Opcodes.F3_DMAST, rs1, rs2, rows)
    def dmawait(): BigInt = dmaInst(Opcodes.F3_DMAWAIT, 0, 0, 0)

    // CUSTOM1 vector ops. (funct7<<25)|(rs2<<20)|(rs1<<15)|(funct3<<12)|(rd<<7)|0x2b
    private def vec(f3: Int, f7: Int, rd: Int, rs1: Int, rs2: Int): BigInt =
        (BigInt(f7) << 25) | (BigInt(rs2) << 20) | (BigInt(rs1) << 15) |
        (BigInt(f3) << 12) | (BigInt(rd) << 7) | 0x2b
    // Config: value (base address / length / fp32 scalar bits) comes from GPR rs1.
    def vseta(rs1: Int): BigInt   = vec(Opcodes.VF3_CONFIG, Opcodes.VF7_SETA, 0, rs1, 0)
    def vsetb(rs1: Int): BigInt   = vec(Opcodes.VF3_CONFIG, Opcodes.VF7_SETB, 0, rs1, 0)
    def vsetd(rs1: Int): BigInt   = vec(Opcodes.VF3_CONFIG, Opcodes.VF7_SETD, 0, rs1, 0)
    def vsetlen(rs1: Int): BigInt = vec(Opcodes.VF3_CONFIG, Opcodes.VF7_SETLEN, 0, rs1, 0)
    def vsets(rs1: Int): BigInt   = vec(Opcodes.VF3_CONFIG, Opcodes.VF7_SETS, 0, rs1, 0)
    // Compute (operands/dst from latched bases).
    def vadd(): BigInt   = vec(Opcodes.VF3_COMPUTE, Opcodes.VF7_ADD, 0, 0, 0)
    def vsub(): BigInt   = vec(Opcodes.VF3_COMPUTE, Opcodes.VF7_SUB, 0, 0, 0)
    def vmul(): BigInt   = vec(Opcodes.VF3_COMPUTE, Opcodes.VF7_MUL, 0, 0, 0)
    def vfma(): BigInt   = vec(Opcodes.VF3_COMPUTE, Opcodes.VF7_FMA, 0, 0, 0)
    def vsmul(): BigInt  = vec(Opcodes.VF3_COMPUTE, Opcodes.VF7_SMUL, 0, 0, 0)
    def vsadd(): BigInt  = vec(Opcodes.VF3_COMPUTE, Opcodes.VF7_SADD, 0, 0, 0)
    def vexp(): BigInt   = vec(Opcodes.VF3_COMPUTE, Opcodes.VF7_EXP, 0, 0, 0)
    def vrecip(): BigInt = vec(Opcodes.VF3_COMPUTE, Opcodes.VF7_RECIP, 0, 0, 0)
    def vrsqrt(): BigInt = vec(Opcodes.VF3_COMPUTE, Opcodes.VF7_RSQRT, 0, 0, 0)
    // Fundamental compute/data-path operations. Their operands and destination are
    // taken from the latched bases; quantization policy is emitted as a sequence.
    def vabs(): BigInt      = vec(Opcodes.VF3_COMPUTE, Opcodes.VF7_ABS, 0, 0, 0)
    def vcvtI8(): BigInt    = vec(Opcodes.VF3_COMPUTE, Opcodes.VF7_CVT_I8, 0, 0, 0)
    def vcvtBf16(): BigInt  = vec(Opcodes.VF3_COMPUTE, Opcodes.VF7_CVT_BF16, 0, 0, 0)
    // Reductions write the fp32 result into GPR rd.
    def vsum(rd: Int): BigInt = vec(Opcodes.VF3_REDUCE, Opcodes.VF7_SUM, rd, 0, 0)
    def vmax(rd: Int): BigInt = vec(Opcodes.VF3_REDUCE, Opcodes.VF7_MAX, rd, 0, 0)
    def vdot(rd: Int): BigInt = vec(Opcodes.VF3_REDUCE, Opcodes.VF7_DOT, rd, 0, 0)
    def vsoftexp(rd: Int): BigInt = vec(Opcodes.VF3_COMPOUND, Opcodes.VF7_SOFTEXP, rd, 0, 0)

    def halt: BigInt = BigInt(0)
}
