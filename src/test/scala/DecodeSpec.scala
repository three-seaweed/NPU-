// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

// White-box decode checks for the FP.* instruction encodings and the matrix
// offset immediate. The machine-code literals are hand-assembled on purpose:
// they pin the bit fields independently of the Encode helpers, so a slip on
// either side is caught here. (12 仓 DecodeSpec 原样移植，handbook §3.4。)
class DecodeBench extends Module {
    val io = IO(new Bundle {
        val inst = Input(UInt(32.W))
        val out  = Output(new DecodedInst)
    })
    io.out := Decode(io.inst)
}

class DecodeSpec extends AnyFlatSpec with ChiselScalatestTester {
    val mask32 = (BigInt(1) << 32) - 1

    behavior of "Decode custom-0 encodings"

    it should "decode the hand-assembled FP.* machine codes" in {
        test(new DecodeBench).withAnnotations(Seq()) { dut =>
            def opOf(word: BigInt): BigInt = {
                dut.io.inst.poke(word.U)
                dut.io.out.op.peek().litValue
            }
            def immOf(word: BigInt): BigInt = {
                dut.io.inst.poke(word.U)
                dut.io.out.imm.peek().litValue
            }

            // FP.MATMUL: f3=2, f7=2, rs1=x8 -> 0x0404200B. imm must be forced
            // to zero: the funct7 bits ARE imm[11:5], and Control adds
            // rs1+imm for matrix ops, so a stale imm would corrupt the base.
            val fpMat = BigInt("0404200B", 16)
            assert(opOf(fpMat) == AluOp.FP_MATMUL.litValue, "FP.MATMUL literal")
            assert(immOf(fpMat) == 0, "FP.MATMUL must decode with imm=0")

            // FP.MATMULA: f3=4, f7=1 -> 0x0204400B.
            val fpAcc = BigInt("0204400B", 16)
            assert(opOf(fpAcc) == AluOp.FP_MATMULA.litValue, "FP.MATMULA literal")
            assert(immOf(fpAcc) == 0, "FP.MATMULA must decode with imm=0")

            // FP.SM: f3=3, f7=1, rs1=x8 -> 0x0204300B (base register consumed).
            val fpSm = BigInt("0204300B", 16)
            dut.io.inst.poke(fpSm.U)
            assert(dut.io.out.op.peek().litValue == AluOp.FP_SM.litValue, "FP.SM literal")
            assert(dut.io.out.rs1.peek().litValue == 8, "FP.SM rs1")
            assert(dut.io.out.imm.peek().litValue == 0, "FP.SM must decode with imm=0")
            assert(dut.io.out.isMatrix.peek().litToBoolean, "FP.SM is a matrix op")
            assert(dut.io.out.legal.peek().litToBoolean, "FP.SM legal")
        }
    }

    it should "keep the Encode helpers bit-identical to the literals" in {
        // Pure Scala: the assembler side of the contract.
        assert(Encode.fpMatmul() == BigInt("0400200B", 16), "Encode.fpMatmul")
        assert(Encode.fpMatmulAcc() == BigInt("0200400B", 16), "Encode.fpMatmulAcc")
        assert(Encode.fpSm(8) == BigInt("0204300B", 16), "Encode.fpSm")
        assert(Encode.fpSm(12) == BigInt("0206300B", 16), "Encode.fpSm rs1=x12")
        // MCFG2 rides the MATMUL slot with f7=3; cfg is a REGISTER index.
        assert(Encode.mcfg2(11) == BigInt("06B0200B", 16), "Encode.mcfg2")
    }

    it should "disambiguate funct3=2 between MATMUL/MQCFG/FP.MATMUL/MCFG2 and stay tolerant past f7=3" in {
        test(new DecodeBench).withAnnotations(Seq()) { dut =>
            def op(word: BigInt): BigInt = {
                dut.io.inst.poke(word.U)
                dut.io.out.op.peek().litValue
            }
            def imm(word: BigInt): BigInt = {
                dut.io.inst.poke(word.U)
                dut.io.out.imm.peek().litValue
            }
            def lit(f7: Int, rs1: Int, f3: Int): BigInt =
                (BigInt(f7) << 25) | (BigInt(rs1) << 15) | (BigInt(f3) << 12) | 0x0b
            assert(op(lit(0, 0, 2)) == AluOp.MATMUL.litValue)
            assert(op(lit(1, 0, 2)) == AluOp.MQCFG.litValue)
            assert(op(lit(2, 0, 2)) == AluOp.FP_MATMUL.litValue)
            assert(op(lit(3, 0, 2)) == AluOp.MCFG2.litValue)
            assert(imm(lit(3, 0, 2)) == 0, "MCFG2 must decode with imm=0")
            // Documented tolerance: unknown funct7 keeps the legacy op.
            assert(op(lit(4, 0, 2)) == AluOp.MATMUL.litValue)
            assert(op(lit(3, 0, 3)) == AluOp.SM.litValue)
            assert(op(lit(3, 0, 4)) == AluOp.MATMULA.litValue)
            assert(op(lit(1, 0, 4)) == AluOp.FP_MATMULA.litValue)
            assert(op(lit(1, 0, 3)) == AluOp.FP_SM.litValue)
        }
    }

    it should "sign-extend positive and negative matrix offsets" in {
        test(new DecodeBench).withAnnotations(Seq()) { dut =>
            def imm(word: BigInt): BigInt = {
                dut.io.inst.poke(word.U)
                dut.io.out.imm.peek().litValue
            }
            // LMA/LMB/SM keep the full signed 12-bit offset range.
            assert(imm(Encode.lma(10, 64)) == 64)
            assert(imm(Encode.lmb(11, -16)) == (BigInt(-16) & mask32))
            assert(imm(Encode.sm(12, 2047)) == 2047)
            assert(imm(Encode.lma(10, -2048)) == (BigInt(-2048) & mask32))
            // A legacy SM whose offset lands in the funct7 window (imm 32..63)
            // is bit-identical to FP.SM and decodes as it, with imm=0: that
            // encoding range is reserved for the fp32 store now. Locked here
            // so the convention is explicit.
            assert(Encode.sm(12, 32) == BigInt("0206300B", 16))
            assert(Encode.sm(12, 32) == Encode.fpSm(12))
            dut.io.inst.poke(Encode.sm(12, 32).U)
            assert(dut.io.out.op.peek().litValue == AluOp.FP_SM.litValue)
            assert(dut.io.out.imm.peek().litValue == 0)
        }
    }
}
