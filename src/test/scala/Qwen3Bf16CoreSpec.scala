// Copyright 2026 BUPT

package xnpu

import org.scalatest.flatspec.AnyFlatSpec

// 2026-11-03 板测复现专用：bf16 8x8 tier 的 Core 级 FP.* 重放。夹具
// fx_bf16self.c = 板上 bf16selftest 程序逐字 dump（同 LCG 激励、同
// emit_k8/emit_k16 指令序列、同 DTCM 基址 A1=0x000 A2=0x200 B=0x400
// C=0x600）。
//
// 为何此前抓不到：层程序不发 FP.*（见 conduct_docs 与 bf16 tier 记录），
// 唯一触达 bf16 tier 的 MultiEngineSpec 是 Matmul 单元级直驱（tb.issue），
// Core/Alu 队列层从未仿真过 FP.*。板上症状：FP.SM 写 C 时行 1 低 64 位
// 为垃圾（随后又被写对，跨进程读回 golden）；高半与其余行全部 bit-exact。
//
// 对拍契约：bit-exact（黄金 = bf16 宽化 + 严格 k 序 fmaf + 出口单次 RNE，
// 与仿真金标同构）——C 窗口 32 个 u32 逐字比较。已知板上错位（行1低半）
// 若为逻辑 bug，在此必红。
class Qwen3Bf16CoreSpec extends AnyFlatSpec with Qwen3FxHarness {
  behavior of "Core-level FP.* bf16 8x8 tier (board bf16selftest replay)"

  private val pb = Parameters(bf16Matmul = true)

  /** u32 word at byte offset in the 16-byte-row DTCM model. */
  private def wordAt(mem: collection.Map[BigInt, BigInt], byteOff: Int): BigInt =
    (mem.getOrElse(BigInt(byteOff & ~15), BigInt(0)) >> (8 * (byteOff % 16))) & 0xffffffffL

  private def replay(name: String): Unit = {
    val fx = loadFx(name)
    val (mem, cyc) = runCore(fx.prog, seedRows(fx.memIn, fx.metaI("window")),
                             pOverride = Some(pb))
    info(s"$name: halted after $cyc cycles")
    val c = fx.metaI("c")
    val bad = scala.collection.mutable.ArrayBuffer[String]()
    for (w <- 0 until 128 / 4) {   // bf16 8x8 tile = 32 u32 words
      val exp = fx.memRef(c / 4 + w)
      val got = wordAt(mem, c + w * 4)
      if (got != exp)
        bad += f"C word $w (row ${w / 4}): got 0x${got.toString(16)} want 0x${exp.toString(16)}"
    }
    assert(bad.isEmpty,
      s"${bad.size} word mismatches (board symptom = row 1 lo half): ${bad.take(8).mkString(", ")}")
  }

  it should "replay board k8 chain (LMA LMB FP.MATMUL FP.SM) bit-exactly" in {
    replay("bf16k8")
  }

  it should "replay board k16 chain (FP.MATMULA accumulate) bit-exactly" in {
    replay("bf16k16")
  }
}
