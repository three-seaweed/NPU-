// Copyright 2026 BUPT

package xnpu

import org.scalatest.flatspec.AnyFlatSpec

// Replays the EXACT fe_qknorm instruction stream that examples/qwen3/xnpu
// generates for Qwen3-0.6B shapes (q_dim=2048, 16+8 heads, head_dim=128)
// through the real Core against a bf16-boundary fp32 reference.
//
// WHY THIS SPEC EXISTS
// --------------------
// The layer program the board executes is built at run time by C; every other
// spec in this suite writes its program by hand in Scala, so none of them can
// see the code the board will actually run.  This fixture closes that gap for
// the first Qwen3 emitters (plan qwen3_0.6b_plan.md §5 Phase 3, gate 3a).
//
// What is under test is fe_qknorm's emitted sequence: the per-head stride, the
// head count, the shared (head_dim,) gain, and in-place writeback.  The Q/K
// gain leak check is a negative control: the fixture gains differ precisely so
// it cannot pass by luck (plan §6 fixture-blindness discipline).  Mutation-
// tested 2026-10-17: injecting "gain pointer advances per head" (the class
// that once let a real QK-norm batch bug through) drives the relative error
// from 4.55e-03 to 1.16 and fails.
//
// Regen fixtures: `make -C examples/qwen3 fx-rtl`
class Qwen3QknormRtlSpec extends AnyFlatSpec with Qwen3FxHarness {
  behavior of "fe_qknorm (Qwen3 emitter, RTL replay)"

  it should "normalise every head of Q and K on the real Core" in {
    val fx = loadFx("qknorm")
    val headDim = fx.metaI("head_dim")
    val nHeads = fx.metaI("n_heads")
    val nKv = fx.metaI("n_kv")
    val B = fx.metaI("B")
    val qOff = fx.metaI("q_off")
    val kOff = fx.metaI("k_off")
    val windowBytes = fx.metaI("window")

    val (mem, cycles) = runCore(fx.prog, seedRows(fx.memIn, windowBytes))
    info(s"ran ${fx.prog.length} instructions in $cycles cycles")

    checkRegion("Q", fx, mem, qOff, B * nHeads * headDim)
    checkRegion("K", fx, mem, kOff, B * nKv * headDim)

    // K's gain must not have been applied with Q's gain (they differ in the
    // fixture precisely so this cannot pass by luck).
    var qLeak = 0.0
    for (i <- 0 until headDim) {
      val d = math.abs(bf16At(mem, kOff, i) - refBf16(fx.memRef, qOff, i))
      if (d > qLeak) qLeak = d
    }
    assert(qLeak > 1e-3, "Q and K outputs are identical -- the two gains were not distinguished")
  }
}
