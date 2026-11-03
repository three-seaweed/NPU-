// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable.ArrayBuffer

// Npu 级 AXI 重放（2026-11-03 板测 bug 复现第二层）：与板上 bf16selftest
// 完全同构的访问路径——host window（AXI→port1）写 ITCM/DTCM、CTRL kick/
// poll、AXI 读回 C 区，中间隔着真·双口 DTCM blackbox 与 port0/port1 仲裁。
//
// 层级排除法：Core 级单口模型已绿（Qwen3Bf16CoreSpec：逻辑/队列/token 全对）。
// 若板上"C 行 1 低 64 位垃圾"源于 RTL 的双口/仲裁/host 路径，此层必红；
// 若此层也绿，则嫌疑移出 RTL（bank verilog 之外：AXI 互联、PS 侧 posted
// write、物理实现），另行排查。
class Qwen3Bf16NpuSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend
    with Qwen3FxHarness {
  behavior of "Npu-level FP.* bf16 8x8 (AXI host-window replay of board bf16selftest)"

  private val mask32 = (BigInt(1) << 32) - 1
  private val ITCM = BigInt(0) << 19
  private val DTCM = BigInt(1) << 19
  private val CTRL = BigInt(2) << 19
  private val MAX_BEATS = 256   // AXI len is 8-bit; chunk longer bursts

  private def replay(name: String): Unit = {
    val fx = loadFx(name)
    val pb = Parameters(bf16Matmul = true)
    test(new Npu(pb)).withAnnotations(backendAnnos) { dut =>
      val axi = dut.io.axi
      def step(): Unit = dut.clock.step(1)
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

      def writeChunked(addr: BigInt, data: Seq[BigInt]): Unit =
        data.grouped(MAX_BEATS).zipWithIndex.foreach { case (chunk, i) =>
          writeBurst(addr + BigInt(i * MAX_BEATS * 4), chunk)
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

      // 1. ITCM program + DTCM operand window via the host window (port1 path).
      writeBurst(ITCM, fx.prog)
      writeChunked(DTCM, fx.memIn)

      // 2. Kick and poll STATUS.done, exactly like hal.c.
      writeBurst(CTRL, Seq(BigInt(1)))
      var guard = 0
      while ((readBurst(CTRL + 4, 1).head & 1) == 0) {
        guard += 1
        assert(guard < 200000, "NPU did not finish")
      }

      // 3. Read the C tile back through the same host window the board checks.
      val c = fx.metaI("c")
      val got = readBurst(DTCM + c, 128 / 4)
      val bad = ArrayBuffer[String]()
      for (w <- 0 until 128 / 4) {
        val exp = fx.memRef(c / 4 + w)
        if (got(w) != exp)
          bad += f"C word $w (row ${w / 4}): got 0x${got(w).toString(16)} want 0x${exp.toString(16)}"
      }
      assert(bad.isEmpty,
        s"${bad.size} word mismatches (board symptom = row 1 lo half): ${bad.take(8).mkString(", ")}")
    }
  }

  it should "replay board k8 chain through AXI bit-exactly" in { replay("bf16k8") }

  it should "replay board k16 chain through AXI bit-exactly" in { replay("bf16k16") }
}
