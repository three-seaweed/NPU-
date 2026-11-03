// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable

class CoreSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val p = Parameters()
  val n = p.MatrixDim

  // Deterministic NxN int8 operands.
  val A = Array.tabulate(n, n)((i, j) => ((i + 2 * j + 1) % 9) - 4)
  val B = Array.tabulate(n, n)((i, j) => ((3 * i + j + 2) % 7) - 3)
  val C = Array.tabulate(n, n)((i, j) => (0 until n).map(k => A(i)(k) * B(k)(j)).sum)

  // DTCM byte offsets for A, B, and the result, each matrix spanning
  // n*WordsPerRow words.
  val arrayRowWords = p.ArrayRowWords
  val baseA = 0
  val baseB = n * arrayRowWords * 4
  val baseC = 2 * n * arrayRowWords * 4

  // Pack one matrix row into its WordsPerRow words (ElemsPerWord int8 each).
  def rowWords(row: Array[Int]): Seq[BigInt] =
    (0 until arrayRowWords).map { g =>
      (0 until p.ElemsPerWord).foldLeft(BigInt(0)) { (acc, e) =>
        acc | (BigInt(row(g * p.ElemsPerWord + e) & 0xff) << (8 * e))
      }
    }

  // Write matrix `m` into the test memory model at byte offset `base`.
  def loadMatrix(mem: mutable.Map[BigInt, BigInt], base: Int, m: Array[Array[Int]]): Unit =
    for (r <- 0 until n; (w, g) <- rowWords(m(r)).zipWithIndex)
      mem(BigInt(base + (r * arrayRowWords + g) * 4)) = w

  // Wide (128-bit / one-row) DTCM model helpers: each access moves WordsPerRow
  // 32-bit words at a 16-byte-aligned address; mem stays keyed by word byte addr.
  val mask32 = (BigInt(1) << 32) - 1
  def packWide(mem: mutable.Map[BigInt, BigInt], a: BigInt): BigInt =
    (0 until p.WordsPerRow).foldLeft(BigInt(0))((acc, i) => acc | (mem(a + 4 * i) << (32 * i)))
  def writeWide(mem: mutable.Map[BigInt, BigInt], a: BigInt, data: BigInt): Unit =
    for (i <- 0 until p.WordsPerRow) mem(a + 4 * i) = (data >> (32 * i)) & mask32

  behavior of "Core"

  it should "fetch a program from ITCM and run scalar + matrix instructions" in {
    // Program: two scalar ops, then the matrix sequence, then HALT.
    val program = Seq(
      Encode.addi(1, 0, 7),   // x1 = 7
      Encode.add(2, 1, 1),    // x2 = x1 + x1 = 14
      Encode.addi(10, 0, baseA),   // x10 = A base
      Encode.addi(11, 0, baseB),   // x11 = B base
      Encode.addi(12, 0, baseC),   // x12 = C base
      Encode.lma(10),
      Encode.lmb(11),
      Encode.matmul(),
      Encode.sm(12),
      Encode.halt,
    )

    // External instruction-memory model: inst valid one cycle after addr.
    def instWord(byteAddr: BigInt): BigInt = {
      val idx = (byteAddr >> 2).toInt
      if (idx >= 0 && idx < program.length) program(idx) else BigInt(0)
    }

    test(new Core(p)).withAnnotations(backendAnnos) { dut =>
      val mem = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
      loadMatrix(mem, baseA, A)
      loadMatrix(mem, baseB, B)

      var pendingRead: Option[BigInt] = None
      var instAddrPrev: BigInt = 0
      dut.io.dtcm.ready.poke(true.B)
      dut.io.dmaIdle.poke(true.B)   // no core-issued DMA here; engine always idle

      def tick(): Unit = {
        // ITCM: drive the instruction fetched for the address presented last cycle.
        dut.io.itcmInst.poke(instWord(instAddrPrev).U)
        instAddrPrev = dut.io.itcmAddr.peek().litValue

        // DTCM memory model.
        dut.io.dtcm.rdata.poke(pendingRead.map(a => packWide(mem, a)).getOrElse(BigInt(0)).U)
        val v = dut.io.dtcm.valid.peek().litToBoolean
        val w = dut.io.dtcm.write.peek().litToBoolean
        val a = dut.io.dtcm.addr.peek().litValue
        if (v) {
          if (w) writeWide(mem, a, dut.io.dtcm.wdata.peek().litValue)
          pendingRead = if (!w) Some(a) else None
        } else {
          pendingRead = None
        }
        dut.clock.step(1)
      }

      // Launch the run (single start pulse), then run until the core halts.
      dut.io.start.poke(true.B)
      tick()
      dut.io.start.poke(false.B)

      var guard = 0
      while (!dut.io.done.peek().litToBoolean) {
        tick()
        guard += 1
        assert(guard < 2000, "core did not halt")
      }

      // Scalar results.
      dut.io.dbg(1).expect(7.U)
      dut.io.dbg(2).expect(14.U)

      // Matrix results written back to DTCM (row-major int32).
      for (i <- 0 until n; j <- 0 until n) {
        val got = mem(BigInt(baseC + (i * n + j) * 4))
        val exp = BigInt(C(i)(j)) & ((BigInt(1) << 32) - 1)
        assert(got == exp, s"C($i)($j): got $got expected $exp")
      }
    }
  }

  // Exercises scalar/matrix decoupling: the three scalar ops sit *behind* MATMUL
  // and must execute concurrently while it computes, yet SM (which follows them)
  // must still wait for MATMUL to finish before storing. Correct scalar AND
  // matrix results prove the pipeline overlapped the work without reordering the
  // matrix unit's effects.
  it should "run scalar ops concurrently with an in-flight matrix op" in {
    val program = Seq(
      Encode.addi(10, 0, baseA),   // x10 = A base
      Encode.addi(11, 0, baseB),   // x11 = B base
      Encode.addi(12, 0, baseC),   // x12 = C base
      Encode.lma(10),
      Encode.lmb(11),
      Encode.matmul(),       // multi-cycle background compute starts here
      Encode.addi(3, 0, 5),  // x3 = 5   ) these retire while MATMUL is computing
      Encode.addi(4, 0, 9),  // x4 = 9   )
      Encode.add(2, 3, 4),   // x2 = x3 + x4 = 14 (RAW on x3 and x4 -> forwarding)
      Encode.sm(12),         // must stall in EX until MATMUL drains
      Encode.halt,
    )

    def instWord(byteAddr: BigInt): BigInt = {
      val idx = (byteAddr >> 2).toInt
      if (idx >= 0 && idx < program.length) program(idx) else BigInt(0)
    }

    test(new Core(p)).withAnnotations(backendAnnos) { dut =>
      val mem = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
      loadMatrix(mem, baseA, A)
      loadMatrix(mem, baseB, B)

      var pendingRead: Option[BigInt] = None
      var instAddrPrev: BigInt = 0
      dut.io.dtcm.ready.poke(true.B)
      dut.io.dmaIdle.poke(true.B)   // no core-issued DMA here; engine always idle

      def tick(): Unit = {
        dut.io.itcmInst.poke(instWord(instAddrPrev).U)
        instAddrPrev = dut.io.itcmAddr.peek().litValue

        dut.io.dtcm.rdata.poke(pendingRead.map(a => packWide(mem, a)).getOrElse(BigInt(0)).U)
        val v = dut.io.dtcm.valid.peek().litToBoolean
        val w = dut.io.dtcm.write.peek().litToBoolean
        val a = dut.io.dtcm.addr.peek().litValue
        if (v) {
          if (w) writeWide(mem, a, dut.io.dtcm.wdata.peek().litValue)
          pendingRead = if (!w) Some(a) else None
        } else {
          pendingRead = None
        }
        dut.clock.step(1)
      }

      dut.io.start.poke(true.B)
      tick()
      dut.io.start.poke(false.B)

      var guard = 0
      while (!dut.io.done.peek().litToBoolean) {
        tick()
        guard += 1
        assert(guard < 2000, "core did not halt")
      }

      // Scalar results (computed concurrently with MATMUL).
      dut.io.dbg(3).expect(5.U)
      dut.io.dbg(4).expect(9.U)
      dut.io.dbg(2).expect(14.U)

      // Matrix results: SM still stored the correct product after the drain.
      for (i <- 0 until n; j <- 0 until n) {
        val got = mem(BigInt(baseC + (i * n + j) * 4))
        val exp = BigInt(C(i)(j)) & ((BigInt(1) << 32) - 1)
        assert(got == exp, s"C($i)($j): got $got expected $exp")
      }
    }
  }
}
