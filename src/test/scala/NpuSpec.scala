// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable.ArrayBuffer

class NpuSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val p = Parameters()
  val n = p.MatrixDim
  val mask32 = (BigInt(1) << 32) - 1

  // Deterministic NxN int8 operands.
  val A = Array.tabulate(n, n)((i, j) => ((i + 2 * j + 1) % 9) - 4)
  val B = Array.tabulate(n, n)((i, j) => ((3 * i + j + 2) % 7) - 3)
  val C = Array.tabulate(n, n)((i, j) => (0 until n).map(k => A(i)(k) * B(k)(j)).sum)

  // DTCM byte offsets; each matrix spans n*WordsPerRow words.
  val arrayRowWords = p.ArrayRowWords
  val baseA = 0
  val baseB = n * arrayRowWords * 4
  val baseC = 2 * n * arrayRowWords * 4

  // Pack a matrix into its DTCM word stream (row-major, ElemsPerWord int8/word).
  def matrixWords(m: Array[Array[Int]]): Seq[BigInt] =
    (0 until n).flatMap { r =>
      (0 until arrayRowWords).map { g =>
        (0 until p.ElemsPerWord).foldLeft(BigInt(0)) { (acc, e) =>
          acc | (BigInt(m(r)(g * p.ElemsPerWord + e) & 0xff) << (8 * e))
        }
      }
    }

  // Region bases (addr[20:19]).
  val ITCM = BigInt(0) << 19
  val DTCM = BigInt(1) << 19
  val CTRL = BigInt(2) << 19

  behavior of "Npu"

  it should "let an AXI master DMA in a program + data, run, and read back results" in {
    test(new Npu(p)).withAnnotations(backendAnnos) { dut =>
      val axi = dut.io.axi
      def step(): Unit = dut.clock.step(1)

      // Idle all master-driven channels.
      axi.aw.valid.poke(false.B); axi.w.valid.poke(false.B); axi.b.ready.poke(false.B)
      axi.ar.valid.poke(false.B); axi.r.ready.poke(false.B)

      def addrSend(ch: DecoupledIO[AxiAddr], addr: BigInt, beats: Int): Unit = {
        ch.bits.id.poke(0.U); ch.bits.addr.poke(addr.U)
        ch.bits.len.poke((beats - 1).U); ch.bits.size.poke(2.U); ch.bits.burst.poke(1.U)
        ch.valid.poke(true.B)
        while (!ch.ready.peek().litToBoolean) step()
        step()
        ch.valid.poke(false.B)
      }

      def writeBurst(addr: BigInt, data: Seq[BigInt]): Unit = {
        addrSend(axi.aw, addr, data.length)
        for ((d, i) <- data.zipWithIndex) {
          axi.w.bits.data.poke((d & mask32).U)
          axi.w.bits.strb.poke("b1111".U)
          axi.w.bits.last.poke((i == data.length - 1).B)
          axi.w.valid.poke(true.B)
          while (!axi.w.ready.peek().litToBoolean) step()
          step()
        }
        axi.w.valid.poke(false.B)
        axi.b.ready.poke(true.B)
        while (!axi.b.valid.peek().litToBoolean) step()
        step()
        axi.b.ready.poke(false.B)
      }

      def readBurst(addr: BigInt, beats: Int): Seq[BigInt] = {
        addrSend(axi.ar, addr, beats)
        axi.r.ready.poke(true.B)
        val buf = ArrayBuffer[BigInt]()
        while (buf.length < beats) {
          if (axi.r.valid.peek().litToBoolean) buf += axi.r.bits.data.peek().litValue
          step()
        }
        axi.r.ready.poke(false.B)
        buf.toSeq
      }

      // Program: scalar warm-up + matrix sequence + HALT.
      val program = Seq(
        Encode.addi(1, 0, 7),
        Encode.add(2, 1, 1),
        Encode.addi(10, 0, baseA),
        Encode.addi(11, 0, baseB),
        Encode.addi(12, 0, baseC),
        Encode.lma(10),
        Encode.lmb(11),
        Encode.matmul(),
        Encode.sm(12),
        Encode.halt,
      )

      // 1. DMA the program into ITCM and the matrices into DTCM.
      writeBurst(ITCM, program)
      writeBurst(DTCM + baseA, matrixWords(A))
      writeBurst(DTCM + baseB, matrixWords(B))

      // 2. Start, then poll STATUS.done.
      writeBurst(CTRL, Seq(BigInt(1)))   // CTRL.start = 1
      var guard = 0
      while ((readBurst(CTRL + 4, 1).head & 1) == 0) {
        guard += 1
        assert(guard < 2000, "NPU did not finish")
      }

      // 3. DMA the n*n int32 results back out of DTCM.
      val got = readBurst(DTCM + baseC, n * n)
      for (i <- 0 until n; j <- 0 until n) {
        val exp = BigInt(C(i)(j)) & mask32
        assert(got(i * n + j) == exp, s"C($i)($j): got ${got(i * n + j)} expected $exp")
      }
    }
  }
}
