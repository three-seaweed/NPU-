// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable

class MatmulSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val p = Parameters()
  val n = p.MatrixDim

  // Reference NxN integer matrix multiply.
  def matmul(a: Array[Array[Int]], b: Array[Array[Int]]): Array[Array[Int]] =
    Array.tabulate(n, n)((i, j) => (0 until n).map(k => a(i)(k) * b(k)(j)).sum)

  // Deterministic NxN int8 operands (small signed values, varied sign pattern).
  val A = Array.tabulate(n, n)((i, j) => ((i + 2 * j + 1) % 9) - 4)
  val B = Array.tabulate(n, n)((i, j) => ((3 * i + j + 2) % 7) - 3)
  val C = matmul(A, B)

  // Pack one matrix row into WordsPerRow 32-bit words, ElemsPerWord int8 per word
  // (element k of a word in bits [8k+7:8k]).
  val epw = p.ElemsPerWord
  val arrayRowWords = p.ArrayRowWords
  def rowWords(row: Array[Int]): Seq[BigInt] =
    (0 until arrayRowWords).map { g =>
      (0 until epw).foldLeft(BigInt(0)) { (acc, e) =>
        acc | (BigInt(row(g * epw + e) & 0xff) << (8 * e))
      }
    }

  behavior of "SystolicArray"

  it should "compute C = A x B with skewed streaming" in {
    test(new SystolicArray(p)).withAnnotations(backendAnnos) { dut =>
      // Clear accumulators.
      dut.io.clear.poke(true.B)
      dut.io.en.poke(false.B)
      for (i <- 0 until n) { dut.io.aIn(i).poke(0.S); dut.io.bIn(i).poke(0.S) }
      dut.clock.step(1)
      dut.io.clear.poke(false.B)

      // Stream feed cycles s = 0 .. 3N-2, staggered by row/column index.
      dut.io.en.poke(true.B)
      for (s <- 0 until p.SystolicRunCycles) {
        for (i <- 0 until n) {
          val wa = s - i
          dut.io.aIn(i).poke((if (wa >= 0 && wa < n) A(i)(wa) else 0).S)
          val wb = s - i
          dut.io.bIn(i).poke((if (wb >= 0 && wb < n) B(wb)(i) else 0).S)
        }
        dut.clock.step(1)
      }
      dut.io.en.poke(false.B)

      for (i <- 0 until n; j <- 0 until n) {
        dut.io.acc(i)(j).expect(C(i)(j).S, s"acc($i)($j)")
      }
    }
  }

  behavior of "Matmul"

  it should "load, multiply and store via the DTCM port" in {
    test(new Matmul(p)).withAnnotations(backendAnnos) { dut =>
      val baseA = 0
      val baseB = n * arrayRowWords * 4          // A occupies n*arrayRowWords words
      val baseC = 2 * n * arrayRowWords * 4       // results land past A and B
      val mem = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
      for (r <- 0 until n) {
        for ((w, g) <- rowWords(A(r)).zipWithIndex)
          mem(BigInt(baseA + (r * arrayRowWords + g) * 4)) = w
        for ((w, g) <- rowWords(B(r)).zipWithIndex)
          mem(BigInt(baseB + (r * arrayRowWords + g) * 4)) = w
      }

      var pendingRead: Option[BigInt] = None
      dut.io.dtcm.ready.poke(true.B)   // memory always accepts
      dut.io.rd.ready.poke(true.B)
      dut.io.rs1.valid.poke(true.B)
      dut.io.rs2.valid.poke(true.B)
      dut.io.req.valid.poke(false.B)
      dut.io.quant.aBf16.poke(false.B)
      dut.io.quant.enable.poke(false.B)
      dut.io.quant.scale.poke(0.U)
      dut.io.quant.groupSize.poke(0.U)
      dut.io.quant.groupIndex.poke(0.U)
      dut.io.context.valid.poke(false.B)
      dut.io.context.aBase.poke(0.U)
      dut.io.context.bBase.poke(0.U)

      // Wide (128-bit / one-row) DTCM model: each access moves WordsPerRow 32-bit
      // words at a 16-byte-aligned byte address. mem stays keyed by word byte addr.
      val mask32 = (BigInt(1) << 32) - 1
      def packWide(a: BigInt): BigInt =
        (0 until p.WordsPerRow).foldLeft(BigInt(0))((acc, i) => acc | (mem(a + 4 * i) << (32 * i)))
      def writeWide(a: BigInt, data: BigInt): Unit =
        for (i <- 0 until p.WordsPerRow) mem(a + 4 * i) = (data >> (32 * i)) & mask32

      // One cycle of the always-ready memory model plus clock step.
      def tick(): Unit = {
        dut.io.dtcm.rdata.poke(pendingRead.map(packWide).getOrElse(BigInt(0)).U)
        val v = dut.io.dtcm.valid.peek().litToBoolean
        val w = dut.io.dtcm.write.peek().litToBoolean
        val a = dut.io.dtcm.addr.peek().litValue
        if (v) {
          if (w) writeWide(a, dut.io.dtcm.wdata.peek().litValue)
          pendingRead = if (!w) Some(a) else None
        } else {
          pendingRead = None
        }
        dut.clock.step(1)
      }

      def issue(op: AluOp.Type, base: Int): Unit = {
        dut.io.rs1.data.poke(base.U)
        dut.io.req.bits.op.poke(op)
        dut.io.req.bits.addr.poke(0.U)
        dut.io.req.valid.poke(true.B)
        // Idle => ready is high; fire this cycle.
        assert(dut.io.req.ready.peek().litToBoolean, "unit not idle before issue")
        tick()
        dut.io.req.valid.poke(false.B)
        // Run until the unit returns to idle.
        var guard = 0
        while (!dut.io.req.ready.peek().litToBoolean) {
          tick()
          guard += 1
          assert(guard < 1000, s"op $op did not complete")
        }
      }

      issue(AluOp.LMA, baseA)
      issue(AluOp.LMB, baseB)
      issue(AluOp.MATMUL, 0)
      issue(AluOp.SM, baseC)

      // Check the 16 int32 results written back to DTCM (row-major).
      for (i <- 0 until n; j <- 0 until n) {
        val got = mem(BigInt(baseC + (i * n + j) * 4))
        val exp = BigInt(C(i)(j)) & ((BigInt(1) << 32) - 1)
        assert(got == exp, s"C($i)($j): got $got expected $exp")
      }
    }
  }

  it should "accumulate across tiles with MATMUL then MATMUL.ACC" in {
    test(new Matmul(p)).withAnnotations(backendAnnos) { dut =>
      // Two K-tiles; expect C = A0 x B0 + A1 x B1 in the accumulators.
      val A0 = Array.tabulate(n, n)((i, j) => ((i + 2 * j + 1) % 9) - 4)
      val B0 = Array.tabulate(n, n)((i, j) => ((3 * i + j + 2) % 7) - 3)
      val A1 = Array.tabulate(n, n)((i, j) => ((2 * i + j + 5) % 11) - 5)
      val B1 = Array.tabulate(n, n)((i, j) => ((i + 3 * j + 4) % 5) - 2)
      val Cacc = Array.tabulate(n, n)((i, j) =>
        (0 until n).map(k => A0(i)(k) * B0(k)(j) + A1(i)(k) * B1(k)(j)).sum)

      val stride = n * arrayRowWords * 4          // bytes per matrix
      val baseA0 = 0
      val baseB0 = stride
      val baseA1 = 2 * stride
      val baseB1 = 3 * stride
      val baseC  = 4 * stride

      val mem = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
      def load(base: Int, m: Array[Array[Int]]): Unit =
        for (r <- 0 until n; (w, g) <- rowWords(m(r)).zipWithIndex)
          mem(BigInt(base + (r * arrayRowWords + g) * 4)) = w
      load(baseA0, A0); load(baseB0, B0); load(baseA1, A1); load(baseB1, B1)

      var pendingRead: Option[BigInt] = None
      dut.io.dtcm.ready.poke(true.B)
      dut.io.rd.ready.poke(true.B)
      dut.io.rs1.valid.poke(true.B)
      dut.io.rs2.valid.poke(true.B)
      dut.io.req.valid.poke(false.B)
      dut.io.quant.aBf16.poke(false.B)
      dut.io.quant.enable.poke(false.B)
      dut.io.quant.scale.poke(0.U)
      dut.io.quant.groupSize.poke(0.U)
      dut.io.quant.groupIndex.poke(0.U)
      dut.io.context.valid.poke(false.B)
      dut.io.context.aBase.poke(0.U)
      dut.io.context.bBase.poke(0.U)

      val mask32 = (BigInt(1) << 32) - 1
      def packWide(a: BigInt): BigInt =
        (0 until p.WordsPerRow).foldLeft(BigInt(0))((acc, i) => acc | (mem(a + 4 * i) << (32 * i)))
      def writeWide(a: BigInt, data: BigInt): Unit =
        for (i <- 0 until p.WordsPerRow) mem(a + 4 * i) = (data >> (32 * i)) & mask32

      def tick(): Unit = {
        dut.io.dtcm.rdata.poke(pendingRead.map(packWide).getOrElse(BigInt(0)).U)
        val v = dut.io.dtcm.valid.peek().litToBoolean
        val w = dut.io.dtcm.write.peek().litToBoolean
        val a = dut.io.dtcm.addr.peek().litValue
        if (v) {
          if (w) writeWide(a, dut.io.dtcm.wdata.peek().litValue)
          pendingRead = if (!w) Some(a) else None
        } else pendingRead = None
        dut.clock.step(1)
      }
      def issue(op: AluOp.Type, base: Int): Unit = {
        dut.io.rs1.data.poke(base.U)
        dut.io.req.bits.op.poke(op)
        dut.io.req.bits.addr.poke(0.U)
        dut.io.req.valid.poke(true.B)
        assert(dut.io.req.ready.peek().litToBoolean, "unit not idle before issue")
        tick()
        dut.io.req.valid.poke(false.B)
        var guard = 0
        while (!dut.io.req.ready.peek().litToBoolean) {
          tick(); guard += 1; assert(guard < 1000, s"op $op did not complete")
        }
      }

      issue(AluOp.LMA, baseA0)
      issue(AluOp.LMB, baseB0)
      issue(AluOp.MATMUL, 0)        // C = A0 x B0
      issue(AluOp.LMA, baseA1)
      issue(AluOp.LMB, baseB1)
      issue(AluOp.MATMULA, 0)       // C += A1 x B1
      issue(AluOp.SM, baseC)

      for (i <- 0 until n; j <- 0 until n) {
        val got = mem(BigInt(baseC + (i * n + j) * 4))
        val exp = BigInt(Cacc(i)(j)) & mask32
        assert(got == exp, s"Cacc($i)($j): got $got expected $exp")
      }
    }
  }
}
