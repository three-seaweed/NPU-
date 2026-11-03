// Copyright 2026 BUPT

package xnpu

import org.scalatest.flatspec.AnyFlatSpec

// Replays the EXACT fe_rope_half instruction stream that examples/qwen3/xnpu
// generates (Qwen3 half-split RoPE: theta=1e6, j pairs with j+64 -- NOT llama2
// interleave) through the real Core against a bf16-boundary reference.
//
// What is under test is the emitter's swap geometry (scalar word loop
// exchanging word w with w+head_dim/4 inside each head), the per-head stride,
// and the v*cos + swap(v)*sin_signed vector tail shared with runq's fe_rope.
// The fixture starts at position 3 so consecutive tokens carry different
// angles; the negative controls below verify position and head dependence so a
// "one table for every token/head" bug cannot pass by luck.
//
// Regen fixtures: `make -C examples/qwen3 fx-rtl`
class Qwen3RopeRtlSpec extends AnyFlatSpec with Qwen3FxHarness {
  behavior of "fe_rope_half (Qwen3 emitter, RTL replay)"

  it should "rotate Q and K with half-split pairing on the real Core" in {
    val fx = loadFx("rope")
    val B = fx.metaI("B")
    val headDim = fx.metaI("head_dim")
    val nHeads = fx.metaI("n_heads")
    val nKv = fx.metaI("n_kv")
    val qOff = fx.metaI("q_off")
    val kOff = fx.metaI("k_off")
    val windowBytes = fx.metaI("window")

    val (mem, cycles) = runCore(fx.prog, seedRows(fx.memIn, windowBytes))
    info(s"ran ${fx.prog.length} instructions in $cycles cycles")

    checkRegion("Q", fx, mem, qOff, B * nHeads * headDim)
    checkRegion("K", fx, mem, kOff, B * nKv * headDim)

    val qRowBytes = nHeads * headDim * 2
    val kRowBytes = nKv * headDim * 2

    // Negative control 1: position dependence.  Token b runs at position
    // start_pos+b, so identical outputs in the same (head, j) slot across
    // tokens would mean every token consumed one shared table.
    var posDep = 0.0
    for (h <- 0 until nHeads; j <- 0 until headDim) {
      val d = math.abs(bf16At(mem, qOff + h * headDim * 2, j) -
                       bf16At(mem, qOff + qRowBytes + h * headDim * 2, j))
      if (d > posDep) posDep = d
    }
    assert(posDep > 1e-3, "token 0 and token 1 outputs identical -- position dependence lost")

    // Negative control 2: head independence.  All heads share the SAME angle
    // table; only their inputs differ, so identical outputs across heads
    // would mean every head consumed one input row.
    var headDep = 0.0
    for (j <- 0 until headDim) {
      val d = math.abs(bf16At(mem, qOff, j) - bf16At(mem, qOff + headDim * 2, j))
      if (d > headDep) headDep = d
    }
    assert(headDep > 1e-3, "head 0 and head 1 outputs identical -- head stride lost")

    // The swap scratch must not corrupt the reference tables (they sit in the
    // same window; s1/s2 live between them): tables after the run must equal
    // their input image.
    for (off <- Seq(fx.metaI("cosq_off"), fx.metaI("sinq_off"),
                    fx.metaI("cosk_off"), fx.metaI("sink_off"))) {
      val n = B * nHeads * headDim
      var drift = 0.0
      for (i <- 0 until n) {
        val d = math.abs(bf16At(mem, off, i) - bf16At(seedRows(fx.memIn, windowBytes), off, i))
        if (d > drift) drift = d
      }
      assert(drift == 0.0, s"table @0x${off.toHexString} was clobbered (drift $drift)")
    }
  }
}
