// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import fp32matmul.Fp32Model
import scala.collection.mutable

// WS (weight-stationary) dataflow tests, C4: the dormant PE datapath driven
// end-to-end through the Matmul shell. Goldens are exact int32 integer
// matmuls -- WS is bit-deterministic by construction (integer accumulation,
// fixed systolic schedule verified by hand on n=2).
//
//   1. basic WS: MCFG2(WS) -> MATMUL -> SM           == A x B
//   2. WS-keep:  waterfall once, then a second MATMULA reusing the resident
//                weights                          == A1 x B + A2 x B, with B
//                read from DTCM exactly once (16 rows, first pass only)
//   3. OS roundtrip after WS: switch back with MCFG2(OS) -> MATMUL -> SM
//                reproduces the (already suite-locked) OS semantics
class MultiEngineSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val p = Parameters()
  val n = p.MatrixDim

  def matmul(a: Array[Array[Int]], b: Array[Array[Int]]): Array[Array[Int]] =
    Array.tabulate(n, n)((i, j) => (0 until n).map(k => a(i)(k) * b(k)(j)).sum)

  // Deterministic small-magnitude int8 operands, varied sign patterns.
  val A1 = Array.tabulate(n, n)((i, j) => ((i + 2 * j + 1) % 9) - 4)
  val A2 = Array.tabulate(n, n)((i, j) => ((5 * i + 3 * j + 4) % 11) - 5)
  val B  = Array.tabulate(n, n)((i, j) => ((3 * i + j + 2) % 7) - 3)

  val epw = p.ElemsPerWord
  val arrayRowWords = p.ArrayRowWords
  def rowWords(row: Array[Int]): Seq[BigInt] =
    (0 until arrayRowWords).map { g =>
      (0 until epw).foldLeft(BigInt(0)) { (acc, e) =>
        acc | (BigInt(row(g * epw + e) & 0xff) << (8 * e))
      }
    }

  // MCFG2 rs2 payloads: bit0 = engine (0 in single-engine builds), bit1 =
  // dataflow (0=OS 1=WS), bit2 = ws_keep.
  val CFG_OS = 0
  val CFG_WS = 2
  val CFG_WS_KEEP = 6

  behavior of "Matmul WS dataflow"

  it should "compute C = A x W with the weight-stationary dataflow" in {
    test(new Matmul(p)).withAnnotations(backendAnnos) { dut =>
      val baseA = 0
      val baseB = n * arrayRowWords * 4
      val baseC = 2 * n * arrayRowWords * 4
      val C = matmul(A1, B)

      val tb = new MatmulTb(dut, baseA, baseB, baseC)
      tb.loadTiles(A1, B)
      tb.issue(AluOp.LMA, baseA, 0)
      tb.issue(AluOp.LMB, baseB, 0)
      tb.issue(AluOp.MCFG2, 0, CFG_WS)
      tb.issue(AluOp.MATMUL, 0, 0)
      tb.issue(AluOp.SM, baseC, 0)

      for (i <- 0 until n; j <- 0 until n) {
        val got = tb.mem(BigInt(baseC + (i * n + j) * 4))
        val exp = BigInt(C(i)(j)) & ((BigInt(1) << 32) - 1)
        assert(got == exp, s"WS C($i)($j): got $got expected $exp")
      }
    }
  }

  it should "keep resident weights across MATMULA and read B exactly once" in {
    test(new Matmul(p)).withAnnotations(backendAnnos) { dut =>
      val baseA = 0
      val baseB = n * arrayRowWords * 4
      val baseC = 2 * n * arrayRowWords * 4
      val C = Array.tabulate(n, n)((i, j) => matmul(A1, B)(i)(j) + matmul(A2, B)(i)(j))

      val tb = new MatmulTb(dut, baseA, baseB, baseC)
      tb.loadTiles(A1, B)
      tb.issue(AluOp.LMA, baseA, 0)
      tb.issue(AluOp.LMB, baseB, 0)
      tb.issue(AluOp.MCFG2, 0, CFG_WS)
      tb.issue(AluOp.MATMULA, 0, 0)          // waterfall load, C += A1 x B
      tb.bReads = 0                            // count from here on
      tb.loadTiles(A2, B)                      // swap A in DTCM (B untouched)
      tb.issue(AluOp.MCFG2, 0, CFG_WS_KEEP)    // keep toggle preserves weights
      tb.issue(AluOp.MATMULA, 0, 0)            // A2 only, C += A2 x B
      tb.issue(AluOp.SM, baseC, 0)

      assert(tb.bReads == 0,
        s"WS-keep MATMULA still read ${tb.bReads} B rows from DTCM")
      for (i <- 0 until n; j <- 0 until n) {
        val got = tb.mem(BigInt(baseC + (i * n + j) * 4))
        val exp = BigInt(C(i)(j)) & ((BigInt(1) << 32) - 1)
        assert(got == exp, s"WS-keep C($i)($j): got $got expected $exp")
      }
    }
  }

  it should "reproduce OS semantics after switching back from WS" in {
    test(new Matmul(p)).withAnnotations(backendAnnos) { dut =>
      val baseA = 0
      val baseB = n * arrayRowWords * 4
      val baseC = 2 * n * arrayRowWords * 4
      val C = matmul(A2, B)

      val tb = new MatmulTb(dut, baseA, baseB, baseC)
      tb.loadTiles(A2, B)
      tb.issue(AluOp.LMA, baseA, 0)
      tb.issue(AluOp.LMB, baseB, 0)
      // Detour through WS first: the switch back must clear/wipe everything.
      tb.issue(AluOp.MCFG2, 0, CFG_WS)
      tb.issue(AluOp.MATMUL, 0, 0)
      tb.issue(AluOp.MCFG2, 0, CFG_OS)
      tb.issue(AluOp.MATMUL, 0, 0)
      tb.issue(AluOp.SM, baseC, 0)

      for (i <- 0 until n; j <- 0 until n) {
        val got = tb.mem(BigInt(baseC + (i * n + j) * 4))
        val exp = BigInt(C(i)(j)) & ((BigInt(1) << 32) - 1)
        assert(got == exp, s"OS-after-WS C($i)($j): got $got expected $exp")
      }
    }
  }

  it should "clear the WS output buffer on a dataflow switch (board wsbench 2026-10-25)" in {
    test(new Matmul(p)).withAnnotations(backendAnnos) { dut =>
      val baseA = 0
      val baseB = n * arrayRowWords * 4
      val baseC = 2 * n * arrayRowWords * 4
      // Two independent WS chains with an OS roundtrip between them. The
      // dfSwitch invalidate must zero the WS outBuf: sStore drains but never
      // clears it, and the OS path never touches it -- without the clear the
      // second chain silently sums the first chain's C (found on the board,
      // where outBuf survives across HALT instead of resetting per test()).
      val C = matmul(A2, B)

      val tb = new MatmulTb(dut, baseA, baseB, baseC)
      tb.loadTiles(A1, B)
      tb.issue(AluOp.LMA, baseA, 0)
      tb.issue(AluOp.LMB, baseB, 0)
      tb.issue(AluOp.MCFG2, 0, CFG_WS)
      tb.issue(AluOp.MATMULA, 0, 0)
      tb.issue(AluOp.SM, baseC, 0)           // chain 1: C1 = A1 x B (drained)
      tb.loadTiles(A2, B)
      tb.issue(AluOp.MCFG2, 0, CFG_OS)       // OS roundtrip: both switches
      tb.issue(AluOp.MCFG2, 0, CFG_WS)       // fire the dfSwitch invalidate
      tb.issue(AluOp.MATMULA, 0, 0)
      tb.issue(AluOp.SM, baseC, 0)           // chain 2: exact C2, no C1

      for (i <- 0 until n; j <- 0 until n) {
        val got = tb.mem(BigInt(baseC + (i * n + j) * 4))
        val exp = BigInt(C(i)(j)) & ((BigInt(1) << 32) - 1)
        assert(got == exp, s"WS chain 2 C($i)($j): got $got expected $exp")
      }
    }
  }

  // ---- Shared drive harness (MatmulSpec's TB, parameterised) ----------------
  // Wraps the always-ready wide DTCM model, one-cycle read latency, and the
  // issue-until-idle helper; counts reads into the B region for the keep test.
  class MatmulTb(dut: Matmul, baseA: Int, baseB: Int, baseC: Int) {
    val mem = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
    var bReads = 0
    var pendingRead: Option[BigInt] = None
    val donePulses = Array(0, 0, 0)

    dut.io.dtcm.ready.poke(true.B)
    dut.io.rd.ready.poke(true.B)
    dut.io.rs1.valid.poke(true.B)
    dut.io.rs2.valid.poke(true.B)
    dut.io.req.valid.poke(false.B)
    dut.io.engine.poke(0.U)
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

    def loadTiles(a: Array[Array[Int]], b: Array[Array[Int]]): Unit =
      for (r <- 0 until n) {
        for ((w, g) <- rowWords(a(r)).zipWithIndex)
          mem(BigInt(baseA + (r * arrayRowWords + g) * 4)) = w
        for ((w, g) <- rowWords(b(r)).zipWithIndex)
          mem(BigInt(baseB + (r * arrayRowWords + g) * 4)) = w
      }

    def tick(): Unit = {
      dut.io.dtcm.rdata.poke(pendingRead.map(packWide).getOrElse(BigInt(0)).U)
      for (k <- 0 until 3) {
        if (dut.io.done(k).peek().litToBoolean) donePulses(k) += 1
      }
      val v = dut.io.dtcm.valid.peek().litToBoolean
      val w = dut.io.dtcm.write.peek().litToBoolean
      val a = dut.io.dtcm.addr.peek().litValue
      if (v) {
        if (w) writeWide(a, dut.io.dtcm.wdata.peek().litValue)
        else {
          if (a >= baseB && a < baseB + n * arrayRowWords * 4) bReads += 1
          pendingRead = Some(a)
        }
        if (w) pendingRead = None
      } else {
        pendingRead = None
      }
      dut.clock.step(1)
    }

    def issue(op: AluOp.Type, base: Int, rs2v: Int): Unit = {
      dut.io.rs1.data.poke(base.U)
      dut.io.rs2.data.poke(rs2v.U)
      dut.io.req.bits.op.poke(op)
      dut.io.req.bits.addr.poke(0.U)
      dut.io.req.valid.poke(true.B)
      assert(dut.io.req.ready.peek().litToBoolean, "unit not idle before issue")
      tick()
      dut.io.req.valid.poke(false.B)
      var guard = 0
      while (!dut.io.req.ready.peek().litToBoolean) {
        tick()
        guard += 1
        assert(guard < 2000, s"op $op did not complete")
      }
    }

    // Fire-and-forget for dual-engine concurrency tests: waits for the
    // command's targets to be ready (visible via the aggregate req.ready with
    // the engine field poked), fires exactly one cycle, and returns WITHOUT
    // waiting for completion.
    def send(op: AluOp.Type, base: Int, rs2v: Int, eng: Int): Unit = {
      dut.io.rs1.data.poke(base.U)
      dut.io.rs2.data.poke(rs2v.U)
      dut.io.engine.poke(eng.U)
      dut.io.req.bits.op.poke(op)
      dut.io.req.bits.addr.poke(0.U)
      dut.io.req.valid.poke(true.B)
      var guard = 0
      while (!dut.io.req.ready.peek().litToBoolean) {
        tick()
        guard += 1
        assert(guard < 2000, s"send $op eng=$eng never became ready")
      }
      tick()   // the fire cycle
      dut.io.req.valid.poke(false.B)
      dut.io.engine.poke(0.U)
    }

    def waitCls(eng: Int): Unit = {
      var guard = 0
      while (!dut.io.unitReady(eng).peek().litToBoolean) {
        tick()
        guard += 1
        assert(guard < 3000, s"engine $eng never drained")
      }
    }
  }

  // ---- Dual-engine tests (C5) ------------------------------------------------
  // These only run against Parameters(int8Engines = 2): routing, per-engine
  // latches, concurrency through the shared-port arbiter, and done tagging.
  behavior of "Matmul dual engines"

  it should "route commands per engine and keep results independent" in {
    val p2 = Parameters(int8Engines = 2)
    test(new Matmul(p2)).withAnnotations(backendAnnos) { dut =>
      val baseA = 0
      val baseB = n * arrayRowWords * 4
      val baseC = 2 * n * arrayRowWords * 4
      val C0 = matmul(A1, B)
      val C1 = matmul(A2, B)

      val tb = new MatmulTb(dut, baseA, baseB, baseC)
      tb.loadTiles(A1, B)
      tb.issue(AluOp.LMA, baseA, 0)
      tb.issue(AluOp.LMB, baseB, 0)
      tb.issue(AluOp.MCFG2, 0, CFG_WS)                 // engine 0 (engine field 0)
      tb.issue(AluOp.MATMUL, 0, 0)
      tb.issue(AluOp.SM, baseC, 0)
      // Re-load different tiles.  Dual builds target LMA/LMB at the selected
      // engine only, so each engine gets its own base load: engine 0 re-latches
      // (idle, will not run again), and engine 1 -- never addressed before --
      // latches the new tiles before its MATMUL.
      tb.loadTiles(A2, B)
      tb.issue(AluOp.LMA, baseA, 0)
      tb.issue(AluOp.LMB, baseB, 0)
      tb.send(AluOp.MCFG2, 0, 1 + CFG_WS, 1)           // MCFG2 targeting engine 1
      tb.send(AluOp.LMA, baseA, 0, 1)                  // per-engine bases for e1
      tb.send(AluOp.LMB, baseB, 0, 1)
      tb.send(AluOp.MATMUL, 0, 0, 1)
      tb.waitCls(1)
      tb.send(AluOp.SM, baseC + 4096, 0, 1)
      tb.waitCls(1)   // send() is fire-and-forget: drain e1's store first

      for (i <- 0 until n; j <- 0 until n) {
        val g0 = tb.mem(BigInt(baseC + (i * n + j) * 4))
        val e0 = BigInt(C0(i)(j)) & ((BigInt(1) << 32) - 1)
        assert(g0 == e0, s"dual e0 C($i)($j): got $g0 expected $e0")
        val g1 = tb.mem(BigInt(baseC + 4096 + (i * n + j) * 4))
        val e1 = BigInt(C1(i)(j)) & ((BigInt(1) << 32) - 1)
        assert(g1 == e1, s"dual e1 C($i)($j): got $g1 expected $e1")
      }
    }
  }

  it should "run both engines concurrently with one done pulse each" in {
    val p2 = Parameters(int8Engines = 2)
    test(new Matmul(p2)).withAnnotations(backendAnnos) { dut =>
      val baseA = 0
      val baseB = n * arrayRowWords * 4
      val baseC = 2 * n * arrayRowWords * 4
      val C = matmul(A1, B)

      val tb = new MatmulTb(dut, baseA, baseB, baseC)
      tb.loadTiles(A1, B)
      tb.issue(AluOp.LMA, baseA, 0)
      tb.issue(AluOp.LMB, baseB, 0)
      tb.issue(AluOp.MCFG2, 0, CFG_WS)                 // configure engine 0
      tb.send(AluOp.MCFG2, 0, 1 + CFG_WS, 1)           // configure engine 1
      tb.send(AluOp.LMA, baseA, 0, 1)                  // dual builds latch bases
      tb.send(AluOp.LMB, baseB, 0, 1)                  // per engine
      tb.donePulses(0) = 0
      tb.donePulses(1) = 0
      tb.send(AluOp.MATMUL, 0, 0, 0)                   // engine 0 starts WS
      tb.send(AluOp.MATMUL, 0, 0, 1)                   // engine 1 must fire
                                                       // while engine 0 runs
      tb.waitCls(0)
      tb.waitCls(1)
      tb.send(AluOp.SM, baseC, 0, 0)
      tb.send(AluOp.SM, baseC + 4096, 0, 1)
      tb.waitCls(0)
      tb.waitCls(1)

      assert(tb.donePulses(0) == 2,
        s"engine 0 produced ${tb.donePulses(0)} done pulses (want MATMUL+SM = 2)")
      assert(tb.donePulses(1) == 2,
        s"engine 1 produced ${tb.donePulses(1)} done pulses (want MATMUL+SM = 2)")
      for (i <- 0 until n; j <- 0 until n) {
        val exp = BigInt(C(i)(j)) & ((BigInt(1) << 32) - 1)
        val g0 = tb.mem(BigInt(baseC + (i * n + j) * 4))
        assert(g0 == exp, s"concurrent e0 C($i)($j)")
        val g1 = tb.mem(BigInt(baseC + 4096 + (i * n + j) * 4))
        assert(g1 == exp, s"concurrent e1 C($i)($j)")
      }
    }
  }

  // ---- bf16 8x8 tier (C6) ----------------------------------------------------
  // Bit-exact contract: widen bf16->fp32 exactly, accumulate with
  // Fp32Model.fma in strict k order (the array's FMA PEs), round ONCE to bf16
  // (RNE) at the FP.SM exit. FP.MATMULA chains keep the fp32 accumulator
  // across tiles -- no rounding at tile boundaries.
  behavior of "Matmul bf16 8x8 tier"

  def widen16(h: Int): Long = (h.toLong & 0xffffL) << 16
  def round32ToBf16(f: Int): Long = {
    val u = f.toLong & 0xffffffffL
    if (((u >>> 23) & 0xffL) == 0xffL) (u >>> 16) & 0xffffL   // Inf/NaN pass through
    else ((u + 0x7fffL + ((u >>> 16) & 1L)) >>> 16) & 0xffffL
  }
  // fma on bit patterns: reuse the fp32 PE library's software golden.
  def fmaBits(a: Long, b: Long, c: Long): Long =
    Fp32Model.fma(a.toInt, b.toInt, c.toInt).toLong & 0xffffffffL

  var lcg = 0x243F6A8885A308D3L
  def rnd16(): Int = {
    lcg = lcg * 6364136223846793005L + 1442695040888963407L
    (lcg >>> 40).toInt & 0xffff
  }
  // Well-scaled bf16 stimulus (exp 110..125 keeps n-term fp32 sums finite);
  // sprinkle zeros to exercise fma(0,x,acc)=acc between feeds.
  def bfTile(zeroEvery: Int, n: Int): Array[Array[Int]] =
    Array.tabulate(n, n)((i, j) =>
      if ((i * 7 + j * 13 + zeroEvery) % zeroEvery == 0) 0x0000
      else ((rnd16() & 1) << 15) | ((110 + (rnd16() & 0xf)) << 7) | (rnd16() & 0x7f))

  // TB mem is keyed by 32-bit word address: a 128-bit beat at byte address b
  // spans keys b+0..b+12. Pack/unpack must split accordingly. An n-bf16 row is
  // n/8 beats (one at n=8, two at n=16).
  def bfPackTile(tb: MultiEngineSpec#MatmulTb, base: Int, t: Array[Array[Int]],
                 n: Int): Unit = {
    val bpr = n / 8
    for (r <- 0 until n; h <- 0 until bpr) {
      // beat (r*bpr+h) holds elements h*8..h*8+7, element k in bits [16k..]
      var word = BigInt(0)
      for (e <- 0 until 8) word |= BigInt(t(r)(h * 8 + e) & 0xffff) << (16 * e)
      val b = base + (r * bpr + h) * 16
      for (w <- 0 until 4)
        tb.mem(BigInt(b + 4 * w)) = (word >> (32 * w)) & 0xffffffffL
    }
  }
  def bfUnpack(tb: MultiEngineSpec#MatmulTb, base: Int, r: Int, j: Int,
               n: Int): Long = {
    val bpr = n / 8
    val b = base + (r * bpr + j / 8) * 16
    var beat = BigInt(0)
    for (w <- 0 until 4) beat |= tb.mem(BigInt(b + 4 * w)) << (32 * w)
    (beat >> (16 * (j % 8))).toLong & 0xffffL
  }

  it should "compute C = A x B bit-exactly (fp32 strict-K fma, one final RNE)" in {
    val pb = Parameters(bf16Matmul = true)
    val n = pb.Bf16MatrixDim
    test(new Matmul(pb)).withAnnotations(backendAnnos) { dut =>
      val baseA = 0
      val baseB = 512
      val baseC = 1024
      val A = bfTile(11, n)
      val B = bfTile(13, n)
      val golden = Array.tabulate(n, n) { (i, j) =>
        val acc = (0 until n).foldLeft(0L) { (c, k) =>
          fmaBits(widen16(A(i)(k)), widen16(B(k)(j)), c)
        }
        round32ToBf16(acc.toInt)
      }

      val tb = new MatmulTb(dut, baseA, baseB, baseC)
      bfPackTile(tb, baseA, A, n)
      bfPackTile(tb, baseB, B, n)
      tb.issue(AluOp.LMA, baseA, 0)
      tb.issue(AluOp.LMB, baseB, 0)
      tb.issue(AluOp.FP_MATMUL, 0, 0)
      tb.issue(AluOp.FP_SM, baseC, 0)

      val bad = scala.collection.mutable.ArrayBuffer[String]()
      for (i <- 0 until n; j <- 0 until n) {
        val got = bfUnpack(tb, baseC, i, j, n)
        if (got != golden(i)(j)) bad += s"C($i,$j)=0x${got.toHexString} want 0x${golden(i)(j).toHexString}"
      }
      assert(bad.isEmpty, s"${bad.size} mismatches: ${bad.take(10).mkString(", ")}")
    }
  }

  it should "keep fp32 precision across FP.MATMULA (no tile-boundary rounding)" in {
    val pb = Parameters(bf16Matmul = true)
    val n = pb.Bf16MatrixDim
    test(new Matmul(pb)).withAnnotations(backendAnnos) { dut =>
      val baseA = 0
      val baseB = 512
      val baseC = 1024
      val A1 = bfTile(11, n)
      val A2 = bfTile(17, n)
      val B  = bfTile(13, n)
      // Continuous fp32 fold over k = 0..2n-1; the single bf16 rounding
      // happens at the FP.SM exit. (Rounding after tile 1 would visibly
      // differ on ties near the rounding boundary.)
      val golden = Array.tabulate(n, n) { (i, j) =>
        val acc = (0 until n).foldLeft(0L) { (c, k) =>
          fmaBits(widen16(A1(i)(k)), widen16(B(k)(j)), c)
        }
        val acc2 = (0 until n).foldLeft(acc) { (c, k) =>
          fmaBits(widen16(A2(i)(k)), widen16(B(k)(j)), c)
        }
        round32ToBf16(acc2.toInt)
      }

      val tb = new MatmulTb(dut, baseA, baseB, baseC)
      bfPackTile(tb, baseA, A1, n)
      bfPackTile(tb, baseB, B, n)
      tb.issue(AluOp.LMA, baseA, 0)
      tb.issue(AluOp.LMB, baseB, 0)
      tb.issue(AluOp.FP_MATMUL, 0, 0)
      bfPackTile(tb, baseA, A2, n)             // next A tile at the same base
      tb.issue(AluOp.FP_MATMULA, 0, 0)
      tb.issue(AluOp.FP_SM, baseC, 0)

      val bad2 = scala.collection.mutable.ArrayBuffer[String]()
      for (i <- 0 until n; j <- 0 until n) {
        val got = bfUnpack(tb, baseC, i, j, n)
        if (got != golden(i)(j)) bad2 += s"C($i,$j)=0x${got.toHexString} want 0x${golden(i)(j).toHexString}"
      }
      assert(bad2.isEmpty, s"${bad2.size} mismatches: ${bad2.take(10).mkString(", ")}")
    }
  }
}
