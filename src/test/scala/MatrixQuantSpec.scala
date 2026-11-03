// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable

class MatrixQuantSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  private val p = Parameters()
  private val n = p.MatrixDim
  private val rowBytes = p.DtcmRowBytes

  private def bf16(x: Float): BigInt = {
    val bits = java.lang.Float.floatToIntBits(x)
    val lsb = (bits >>> 16) & 1
    BigInt((bits + 0x7fff + lsb) >>> 16) & 0xffff
  }
  private def packBf16(xs: Seq[Float]): BigInt =
    xs.zipWithIndex.foldLeft(BigInt(0)) { case (word, (x, lane)) =>
      word | (bf16(x) << (16 * lane))
    }
  private def packI8(xs: Seq[Int]): BigInt =
    xs.zipWithIndex.foldLeft(BigInt(0)) { case (word, (x, lane)) =>
      word | (BigInt(x & 0xff) << (8 * lane))
    }

  behavior of "Matmul bf16 activation quantizer"

  it should "quantize bf16 rows in the loader without a DTCM int8 scratch tile" in {
    test(new Matmul(p)).withAnnotations(backendAnnos) { dut =>
      val baseA = 0x0000
      val baseB = 0x1000
      val baseC = 0x2000
      val source = Seq(0.0f, 0.5f, 1.25f, 63.75f, 64.0f, -64.0f, -64.5f, Float.NaN)
      val expected = Seq(0, 1, 2, 127, 127, -128, -128, 0)
      val a = Array.tabulate(n, n)((r, c) => source((r + c) % source.length))
      val b = Array.tabulate(n, n)((r, c) => if (r == c) 1 else 0)
      val mem = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))

      // A is 32B/row bf16 (two physical rows); B remains 16B/row int8.
      for (r <- 0 until n) {
        a(r).grouped(p.Bf16LanesPerRow).zipWithIndex.foreach { case (lanes, beat) =>
          mem(BigInt(baseA + r * n * 2 + beat * rowBytes)) = packBf16(lanes)
        }
        b(r).grouped(rowBytes).zipWithIndex.foreach { case (lanes, beat) =>
          mem(BigInt(baseB + r * n + beat * rowBytes)) = packI8(lanes)
        }
      }

      dut.io.dtcm.ready.poke(true.B)
      dut.io.rd.ready.poke(true.B)
      dut.io.rs1.valid.poke(true.B); dut.io.rs2.valid.poke(true.B)
      dut.io.req.valid.poke(false.B)
      dut.io.quant.aBf16.poke(true.B)
      dut.io.quant.enable.poke(true.B)
      dut.io.quant.scale.poke("h40000000".U) // 2.0
      dut.io.quant.groupSize.poke(n.U)
      dut.io.quant.groupIndex.poke(0.U)
      dut.io.context.valid.poke(false.B)
      dut.io.context.aBase.poke(0.U)
      dut.io.context.bBase.poke(0.U)

      var pendingRead: Option[BigInt] = None
      def tick(): Unit = {
        dut.io.dtcm.rdata.poke(pendingRead.map(mem).getOrElse(BigInt(0)).U)
        val valid = dut.io.dtcm.valid.peek().litToBoolean
        val write = dut.io.dtcm.write.peek().litToBoolean
        val addr = dut.io.dtcm.addr.peek().litValue
        if (valid && write) {
          val data = dut.io.dtcm.wdata.peek().litValue
          val strb = dut.io.dtcm.wstrb.peek().litValue
          var word = mem(addr)
          for (byte <- 0 until rowBytes if ((strb >> byte) & 1) == 1) {
            val mask = BigInt(0xff) << (8 * byte)
            word = (word & ~mask) | (((data >> (8 * byte)) & 0xff) << (8 * byte))
          }
          mem(addr) = word
          pendingRead = None
        } else if (valid) {
          pendingRead = Some(addr)
        } else {
          pendingRead = None
        }
        dut.clock.step()
      }
      def issue(op: AluOp.Type, rs1: Int): Unit = {
        dut.io.rs1.data.poke(rs1.U)
        dut.io.req.bits.op.poke(op)
        dut.io.req.bits.addr.poke(0.U)
        dut.io.req.bits.imm.poke(0.U)
        dut.io.req.valid.poke(true.B)
        assert(dut.io.req.ready.peek().litToBoolean, s"$op did not start idle")
        tick()
        dut.io.req.valid.poke(false.B)
        var cycles = 0
        while (!dut.io.req.ready.peek().litToBoolean) {
          tick()
          cycles += 1
          assert(cycles < 5000, s"$op did not complete")
        }
      }

      issue(AluOp.LMA, baseA)
      issue(AluOp.LMB, baseB)
      issue(AluOp.MATMUL, 0)
      issue(AluOp.SM, baseC)

      // B is identity, so each int32 output must equal the quantized A lane.
      for (r <- 0 until n; c <- 0 until n) {
        val group = c / p.WordsPerRow
        val lane = c % p.WordsPerRow
        val word = mem(BigInt(baseC + (r * (n / p.WordsPerRow) + group) * rowBytes))
        val gotU = ((word >> (32 * lane)) & BigInt("ffffffff", 16)).toLong
        val got = if ((gotU & 0x80000000L) != 0) (gotU - 0x100000000L).toInt else gotU.toInt
        assert(got == expected((r + c) % expected.length), s"C($r,$c) = $got")
      }
    }
  }
}
