// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable

// Full-layer inference on the NPU. A linear layer Y = ReLU(X * W) is built from
// the pieces exercised here, each as its own case driven by a real program the
// scalar core runs end to end:
//
//   * the scalar load/store path (LW/SW) round-tripping through DTCM;
//   * the ReLU activation epilogue -- a runtime branch loop over DTCM words using
//     LW / BGE / SW (the same code a layer runs after its GEMM);
//   * a tiled GEMM whose K dimension (32) exceeds the 16x16 systolic array, so the
//     depth-32 dot products are accumulated *inside* the array across two K-tiles
//     with MATMUL then MATMUL.ACC, then read back once with SM -- the layer's
//     matrix projection, beyond the array's size;
//   * a head-to-head cycle measurement of the systolic array vs an equivalent
//     scalar software MAC loop, quantifying the acceleration.
//
// Tests drive a Core directly with an in-memory DTCM model (mutable.Map) -- the
// same fast harness CoreSpec uses, with wstrb honoured so a scalar SW writes only
// its lane. This avoids the AXI host path entirely, which keeps the 16x16-array
// simulation tractable on the default Treadle backend.
class LayerSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val p = Parameters()
  val n = p.MatrixDim                 // 16: systolic array / tile dimension
  val mask32 = (BigInt(1) << 32) - 1
  val wordsPerRow = p.WordsPerRow     // 32-bit words per 128-bit beat (physical, =4)
  val arrayRowWords = p.ArrayRowWords // 32-bit words per matrix row (4 at n=16, 8 at n=32)
  val beatsPerRow = p.BeatsPerArrayRow // physical beats per matrix row (1, 2)
  val tileBytes   = n * p.ArrayRowBytes // one n x n int8 tile footprint (256 B / 1024 B)
  val tileRows    = n * beatsPerRow    // beats to DMA one tile

  // ---- Scala-side assembler helper -----------------------------------------
  // Load an arbitrary (here always small, < 8KB) constant into a register.
  def li(rd: Int, imm: Int): Seq[BigInt] =
    if (imm >= -2048 && imm < 2048) Seq(Encode.addi(rd, 0, imm))
    else {
      val hi = (imm + 0x800) >> 12
      Seq(Encode.lui(rd, hi & 0xfffff), Encode.addi(rd, rd, imm - (hi << 12)))
    }

  // ---- Matrix packing (row-major, ElemsPerWord int8 per word) --------------
  def rowWords(row: Array[Int]): Seq[BigInt] =
    (0 until arrayRowWords).map { g =>
      (0 until p.ElemsPerWord).foldLeft(BigInt(0)) { (acc, e) =>
        acc | (BigInt(row(g * p.ElemsPerWord + e) & 0xff) << (8 * e))
      }
    }
  def subTile(m: Array[Array[Int]], r0: Int, c0: Int): Array[Array[Int]] =
    Array.tabulate(n, n)((r, c) => m(r0 + r)(c0 + c))
  // Write an n x n int8 tile into the memory model at byte base.
  def loadTile(mem: mutable.Map[BigInt, BigInt], base: Int, tile: Array[Array[Int]]): Unit =
    for (r <- 0 until n; (w, g) <- rowWords(tile(r)).zipWithIndex)
      mem(BigInt(base + (r * arrayRowWords + g) * 4)) = w

  // ---- Core harness with a wstrb-aware wide DTCM model ----------------------
  // Runs `program` from PC=0 to HALT against an in-memory DTCM seeded with
  // `memInit` (word byte-addr -> 32-bit word). Returns the final memory and the
  // cycle count. The model serves one wide row (wordsPerRow words) per access and
  // honours per-byte write strobes, so a scalar SW touches only its lane while a
  // matrix SM (wstrb all ones) writes the whole row.
  def runCore(program: Seq[BigInt], memInit: Map[BigInt, BigInt] = Map(),
              maxCycles: Int = 300000): (mutable.Map[BigInt, BigInt], Long) = {
    def instWord(byteAddr: BigInt): BigInt = {
      val idx = (byteAddr >> 2).toInt
      if (idx >= 0 && idx < program.length) program(idx) else BigInt(0)
    }
    val mem = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
    memInit.foreach { case (k, v) => mem(k) = v }
    var cycles = 0L
    test(new Core(p)).withAnnotations(backendAnnos) { dut =>
      var pendingRead: Option[BigInt] = None
      var instAddrPrev: BigInt = 0
      dut.io.dtcm.ready.poke(true.B)
      dut.io.dmaIdle.poke(true.B)   // runCore programs issue no core DMA; engine idle
      dut.clock.setTimeout(0)

      def packWide(a: BigInt): BigInt =
        (0 until wordsPerRow).foldLeft(BigInt(0))((acc, i) => acc | (mem(a + 4 * i) << (32 * i)))
      def writeWide(a: BigInt, data: BigInt, strb: BigInt): Unit =
        for (i <- 0 until wordsPerRow) {
          var word = mem(a + 4 * i)
          for (b <- 0 until 4) if (((strb >> (i * 4 + b)) & 1) == 1) {
            val nb = (data >> (32 * i + 8 * b)) & 0xff
            word = (word & ~(BigInt(0xff) << (8 * b))) | (nb << (8 * b))
          }
          mem(a + 4 * i) = word & mask32
        }

      def tick(): Unit = {
        dut.io.itcmInst.poke(instWord(instAddrPrev).U)
        instAddrPrev = dut.io.itcmAddr.peek().litValue
        dut.io.dtcm.rdata.poke(pendingRead.map(packWide).getOrElse(BigInt(0)).U)
        val v = dut.io.dtcm.valid.peek().litToBoolean
        val w = dut.io.dtcm.write.peek().litToBoolean
        val a = dut.io.dtcm.addr.peek().litValue
        if (v) {
          if (w) writeWide(a, dut.io.dtcm.wdata.peek().litValue, dut.io.dtcm.wstrb.peek().litValue)
          pendingRead = if (!w) Some(a) else None
        } else pendingRead = None
        dut.clock.step(1)
      }

      dut.io.start.poke(true.B); tick(); dut.io.start.poke(false.B)
      while (!dut.io.done.peek().litToBoolean) {
        tick(); cycles += 1
        assert(cycles < maxCycles, "core did not halt")
      }
    }
    (mem, cycles)
  }

  behavior of "NPU full-layer inference"

  // ---------------------------------------------------------------------------
  it should "round-trip the scalar LW/SW path through DTCM" in {
    // SW a value, LW it back, prove the loaded value feeds the ALU, store again.
    val program =
      li(1, 0x123) ++              // x1 = 0x123
      li(2, 0)     ++              // x2 = DTCM byte base 0
      Seq(
        Encode.sw(1, 2, 0),        // DTCM[0] = 0x123
        Encode.lw(3, 2, 0),        // x3 = DTCM[0]
        Encode.addi(4, 3, 1),      // x4 = x3 + 1  (load result must forward)
        Encode.sw(4, 2, 4),        // DTCM[1] = 0x124  (must not disturb DTCM[0])
        Encode.halt,
      )
    val (mem, _) = runCore(program)
    assert(mem(0) == BigInt(0x123), s"DTCM[0] = ${mem(0)}")
    assert(mem(4) == BigInt(0x124), s"DTCM[1] = ${mem(4)} (LW->ALU->SW round trip)")
  }

  // ---------------------------------------------------------------------------
  // V1 end-to-end: a scalar LW of a row a still-running VSMUL is writing must
  // wait on the row token (no full-drain instruction exists), then read the
  // stored value.  The scalar is 1.0f so the product is bit-exact (x * 1.0 in
  // the fp32 internal path), making the expected word simply the source word.
  it should "LW a row behind a live VSMUL via row tokens (V1)" in {
    val src = 0x200; val dst = 0x100; val out = 0x300
    val srcWord0 = BigInt("3f803f80", 16)      // two bf16 1.0 lanes
    val memInit = Map(BigInt(src) -> srcWord0)
    val program =
      li(1, src) ++ li(2, dst) ++ li(3, 0x3f800000) ++   // scalar s = 1.0f
      li(4, out) ++ li(7, 8) ++ Seq(          // x7 = lane count (VSET* read GPRs)
        Encode.vseta(1),                        // A  = src row
        Encode.vsetd(2),                        // D  = dst row
        Encode.vsets(3),                        // s  = 1.0f
        Encode.vsetlen(7),                      // one row of bf16 lanes
        Encode.vsmul(),                         // dst = src * 1.0 (async)
        Encode.lw(5, 2, 0),                     // LW dst word 0: RAW on the
                                               // VSMUL write token -- must wait
        Encode.sw(5, 4, 0),                     // park it where the TB reads
        Encode.halt,
      )
    val (mem, _) = runCore(program, memInit)
    assert(mem(out) == srcWord0,
      f"V1 LW-behind-VSMUL: got 0x${mem(out)}%08x want 0x${srcWord0.toString(16)}")
  }

  // ---------------------------------------------------------------------------
  // The activation epilogue a layer runs after its GEMM: sweep the int32 outputs
  // and clamp negatives to zero, using LW / BGE / SW + a backward branch.
  it should "apply a ReLU epilogue over DTCM via LW/BGE/SW" in {
    val vals = Seq(5, -3, 0, -7, 9, -1, -100, 42).map(BigInt(_))
    val expect = vals.map(v => if (v < 0) BigInt(0) else v)
    val memInit = vals.zipWithIndex.map { case (v, i) => (BigInt(i * 4), v & mask32) }.toMap
    val end = vals.length * 4
    val program = li(20, 0) ++ li(21, end) ++ Seq(
      Encode.lw(22, 20, 0),          // +0: x22 = C[i]
      Encode.bge(22, 0, 8),          // +1: if x22 >= 0 -> store (skip the zeroing)
      Encode.addi(22, 0, 0),         // +2: x22 = 0
      Encode.sw(22, 20, 0),          // +3: store
      Encode.addi(20, 20, 4),        // +4: ptr += 4
      Encode.blt(20, 21, -20),       // +5: if ptr < end -> loop (back 5 words)
      Encode.halt,
    )
    val (mem, _) = runCore(program, memInit)
    for (i <- vals.indices)
      assert(mem(i * 4) == expect(i), s"ReLU[$i]: got ${mem(i * 4)} expected ${expect(i)}")
  }

  // ---------------------------------------------------------------------------
  // The layer's matrix projection: M=N=16, K=32. K exceeds the 16x16 array, so the
  // depth-32 dot products are split into two K-tiles accumulated inside the array
  // (MATMUL then MATMUL.ACC), then read back once with SM. Bases are set up front
  // (like CoreSpec) so no matrix op sits adjacent to the instruction that computes
  // its base. Verifies the full output tile against an int32 reference.
  it should "run a tiled GEMM with in-array K-accumulation (K=32 > array)" in {
    val M = n; val N = n; val K = 2 * n           // 16 x 16, depth 32
    val X = Array.tabulate(M, K)((i, j) => ((i + 2 * j + 1) % 7) - 3)   // -3..3
    val W = Array.tabulate(K, N)((i, j) => ((3 * i + j + 2) % 5) - 2)   // -2..2
    val Cref = Array.tabulate(M, N)((i, j) => (0 until K).map(k => X(i)(k) * W(k)(j)).sum)

    val xT0 = 0; val xT1 = tileBytes                 // X K-tiles (tileBytes = n*n)
    val wBase = M * K                                // X footprint = M*K bytes (int8)
    val wT0 = wBase; val wT1 = wBase + tileBytes     // W K-tiles
    val cBase = wBase + K * N                         // result after X and W

    val mem = mutable.Map[BigInt, BigInt]()
    loadTile(mem, xT0, subTile(X, 0, 0)); loadTile(mem, xT1, subTile(X, 0, n))
    loadTile(mem, wT0, subTile(W, 0, 0)); loadTile(mem, wT1, subTile(W, n, 0))

    val program =
      li(10, xT0) ++ li(11, xT1) ++ li(12, wT0) ++ li(13, wT1) ++ li(14, cBase) ++ Seq(
        Encode.lma(10), Encode.lmb(12), Encode.matmul(),     // C  = X0 * W0
        Encode.lma(11), Encode.lmb(13), Encode.matmulAcc(),  // C += X1 * W1
        Encode.sm(14), Encode.halt,
      )
    val (out, _) = runCore(program, mem.toMap)
    for (i <- 0 until M; j <- 0 until N) {
      val exp = BigInt(Cref(i)(j)) & mask32
      val got = out(BigInt(cBase + (i * N + j) * 4))
      assert(got == exp, s"C($i)($j): got $got expected $exp")
    }
  }

  // ---------------------------------------------------------------------------
  // Quantify acceleration: cycles for one 16x16x16 = 4096-MAC tile on the systolic
  // array vs an equivalent int8 shift-add MAC loop in scalar software. The stub
  // DTCM reads 0; every matmul-feed and multiply-step count here is data-
  // independent, so the cycle counts are faithful without real operands. The
  // scalar loop is kept small because the array is re-evaluated every cycle.
  it should "accelerate matmul over a scalar software MAC loop" in {
    val npuProg = Seq(
      Encode.addi(10, 0, 0), Encode.addi(11, 0, 256), Encode.addi(12, 0, 512),
      Encode.lma(10), Encode.lmb(11), Encode.matmul(), Encode.sm(12), Encode.halt,
    )
    val npuMacs = n * n * n
    val (_, npuCycles) = runCore(npuProg)

    val OUTER = 1; val Kk = 1
    val scalarMacs = OUTER * Kk
    // x1 acc, x2 outer, x3 OUTER, x4 k, x5 aptr, x6 bptr, x11 K;
    // multiply a=x7 b=x8 -> x9 ; x10 bit, x13 step count (fixed 8-step shift-add).
    val scalarProg = Seq(
      Encode.addi(1, 0, 0),          // 0: acc = 0
      Encode.addi(2, 0, 0),          // 1: outer = 0
      Encode.addi(3, 0, OUTER),      // 2
      Encode.addi(11, 0, Kk),        // 3
      Encode.addi(4, 0, 0),          // 4: outer_loop: k = 0
      Encode.addi(5, 0, 0),          // 5: aptr = 0
      Encode.addi(6, 0, 64),         // 6: bptr
      Encode.lw(7, 5, 0),            // 7: inner_loop: a = DTCM[aptr]
      Encode.lw(8, 6, 0),            // 8: b = DTCM[bptr]
      Encode.addi(9, 0, 0),          // 9: prod = 0
      Encode.addi(13, 0, 8),         // 10: step = 8
      Encode.andi(10, 8, 1),         // 11: mul_loop: bit = b & 1
      Encode.beq(10, 0, 8),          // 12: if bit==0 -> skip add (+2 words)
      Encode.add(9, 9, 7),           // 13: prod += a
      Encode.slli(7, 7, 1),          // 14: a <<= 1
      Encode.srli(8, 8, 1),          // 15: b >>= 1
      Encode.addi(13, 13, -1),       // 16: step--
      Encode.bne(13, 0, -24),        // 17: if step!=0 -> mul_loop (-6 words)
      Encode.add(1, 1, 9),           // 18: acc += prod
      Encode.addi(5, 5, 4),          // 19: aptr += 4
      Encode.addi(6, 6, 4),          // 20: bptr += 4
      Encode.addi(4, 4, 1),          // 21: k++
      Encode.blt(4, 11, -60),        // 22: if k<K -> inner_loop (-15 words)
      Encode.addi(2, 2, 1),          // 23: outer++
      Encode.blt(2, 3, -80),         // 24: if outer<OUTER -> outer_loop (-20 words)
      Encode.halt,                   // 25
    )
    val (_, scalarCycles) = runCore(scalarProg)

    val npuMpc    = npuMacs.toDouble / npuCycles
    val scalarMpc = scalarMacs.toDouble / scalarCycles
    val speedup   = npuMpc / scalarMpc
    info(f"NPU   : $npuMacs%5d MACs in $npuCycles%6d cycles = $npuMpc%7.2f MACs/cycle")
    info(f"scalar: $scalarMacs%5d MACs in $scalarCycles%6d cycles = $scalarMpc%7.4f MACs/cycle")
    info(f"systolic-array acceleration over scalar software: ${speedup}%.1fx")
    assert(speedup > 20.0, s"expected a large speedup, got ${speedup}x")
  }

  // ===========================================================================
  // Core-issued DMA: one PS "kick" runs a whole tiled GEMM layer that streams its
  // own weights from DRAM (DMALD), overlaps the next tile's load with the current
  // MATMUL (double buffer), and DMASTs the result back to DRAM. Driven against the
  // full Npu + an AXI DRAM model (DmaTb / AxiRam, shared with DmaSpec) so the
  // AXI-master path and the relaxed DTCM arbiter are exercised end to end.
  // ===========================================================================
  behavior of "core-issued DMA layer"

  // Region bases (addr[20:19]) and CTRL-region offsets, as in DmaSpec/NpuSpec.
  private val ITCM = BigInt(0) << 19
  private val DTCM = BigInt(1) << 19
  private val CTRL = BigInt(2) << 19
  private val OFF_CTRL = 0x00; private val OFF_STATUS = 0x04

  // One matrix -> its DRAM/DTCM beat stream: n rows x beatsPerRow 128-bit beats
  // (elemsPerBeat int8 per beat, element e at byte e).
  private def dram128(m: Array[Array[Int]]): Seq[BigInt] = {
    val elemsPerBeat = p.DtcmRowBits / p.ElemBits
    for { r <- 0 until n; b <- 0 until beatsPerRow } yield
      (0 until elemsPerBeat).foldLeft(BigInt(0))((acc, e) =>
        acc | (BigInt(m(r)(b * elemsPerBeat + e) & 0xff) << (8 * e)))
  }
  // One matrix -> its DTCM 32-bit word stream (row-major, ElemsPerWord int8/word),
  // for host writes through the 32-bit AXI slave.
  private def dtcmWords(m: Array[Array[Int]]): Seq[BigInt] =
    (0 until n).flatMap { r =>
      (0 until arrayRowWords).map { g =>
        (0 until p.ElemsPerWord).foldLeft(BigInt(0)) { (acc, e) =>
          acc | (BigInt(m(r)(g * p.ElemsPerWord + e) & 0xff) << (8 * e))
        }
      }
    }

  it should "stream weights from DRAM and run a double-buffered tiled GEMM in one kick" in {
    val M = n; val N = n; val K = 2 * n
    val X = Array.tabulate(M, K)((i, j) => ((i + 2 * j + 1) % 7) - 3)
    val W = Array.tabulate(K, N)((i, j) => ((3 * i + j + 2) % 5) - 2)
    val Cref = Array.tabulate(M, N)((i, j) => (0 until K).map(k => X(i)(k) * W(k)(j)).sum)

    // DTCM byte layout (tileBytes = one n x n int8 tile).
    val xT0 = 0; val xT1 = tileBytes; val wBufA = 2 * tileBytes
    val wBufB = 3 * tileBytes; val cBase = 4 * tileBytes
    // DRAM byte layout.
    val wDram0 = 0; val wDram1 = tileBytes; val cDram = 2 * tileBytes

    // One kick: load bufA, prefetch bufB while MATMUL(bufA) runs, then accumulate.
    val program =
      li(10, xT0) ++ li(11, xT1) ++ li(12, cBase) ++
      li(13, wDram0) ++ li(14, wDram1) ++ li(15, wBufA) ++
      li(16, wBufB) ++ li(17, cDram) ++ Seq(
        Encode.dmald(13, 15, tileRows),                    // bufA <- W tile0 (tileRows beats)
        Encode.dmawait(),
        Encode.dmald(14, 16, tileRows),                    // prefetch bufB <- W tile1 (overlaps next MATMUL)
        Encode.lma(10), Encode.lmb(15), Encode.matmul(),   // C  = X0 * W0  (concurrent with bufB load)
        Encode.dmawait(),                                  // bufB ready
        Encode.lma(11), Encode.lmb(16), Encode.matmulAcc(),// C += X1 * W1
        Encode.sm(12),                                     // C -> DTCM[cBase]
        Encode.dmast(17, 12, n * n * 4 / p.DtcmRowBytes),  // C(DTCM) -> DRAM[cDram] (result beats)
        Encode.halt,
      )

    test(new DmaTb(p, dramRows = 512)).withAnnotations(backendAnnos) { dut =>
      val axi = dut.io.axi
      val mask32 = (BigInt(1) << 32) - 1
      def step(): Unit = dut.clock.step(1)
      axi.aw.valid.poke(false.B); axi.w.valid.poke(false.B); axi.b.ready.poke(false.B)
      axi.ar.valid.poke(false.B); axi.r.ready.poke(false.B); dut.io.preWen.poke(false.B)

      def addrSend(ch: DecoupledIO[AxiAddr], addr: BigInt, beats: Int): Unit = {
        ch.bits.id.poke(0.U); ch.bits.addr.poke(addr.U)
        ch.bits.len.poke((beats - 1).U); ch.bits.size.poke(2.U); ch.bits.burst.poke(1.U)
        ch.valid.poke(true.B)
        while (!ch.ready.peek().litToBoolean) step()
        step(); ch.valid.poke(false.B)
      }
      def writeBurst(addr: BigInt, data: Seq[BigInt]): Unit = {
        addrSend(axi.aw, addr, data.length)
        for ((d, i) <- data.zipWithIndex) {
          axi.w.bits.data.poke((d & mask32).U); axi.w.bits.strb.poke("b1111".U)
          axi.w.bits.last.poke((i == data.length - 1).B); axi.w.valid.poke(true.B)
          while (!axi.w.ready.peek().litToBoolean) step()
          step()
        }
        axi.w.valid.poke(false.B); axi.b.ready.poke(true.B)
        while (!axi.b.valid.peek().litToBoolean) step()
        step(); axi.b.ready.poke(false.B)
      }
      def readWord(addr: BigInt): BigInt = {
        addrSend(axi.ar, addr, 1); axi.r.ready.poke(true.B)
        while (!axi.r.valid.peek().litToBoolean) step()
        val v = axi.r.bits.data.peek().litValue
        step(); axi.r.ready.poke(false.B); v
      }
      def dramPreload(base: Int, words: Seq[BigInt]): Unit = {
        for ((w, r) <- words.zipWithIndex) {
          dut.io.preAddr.poke((base + r * p.DtcmRowBytes).U); dut.io.preData.poke(w.U)
          dut.io.preWen.poke(true.B); step()
        }
        dut.io.preWen.poke(false.B)
      }
      def dramRead(addr: Int): BigInt = {
        dut.io.dbgAddr.poke(addr.U); step(); dut.io.dbgData.peek().litValue
      }

      // PS setup: weights into DRAM, program into ITCM, input activations into DTCM.
      dramPreload(wDram0, dram128(subTile(W, 0, 0)))
      dramPreload(wDram1, dram128(subTile(W, n, 0)))
      writeBurst(ITCM, program)
      writeBurst(DTCM + xT0, dtcmWords(subTile(X, 0, 0)))
      writeBurst(DTCM + xT1, dtcmWords(subTile(X, 0, n)))

      // Single kick + one poll for done.
      writeBurst(CTRL + OFF_CTRL, Seq(BigInt(1)))
      var guard = 0
      while ((readWord(CTRL + OFF_STATUS) & 1) == 0) {
        guard += 1; assert(guard < 4000, "NPU layer did not finish")
      }

      // Result was DMAd to DRAM by the program itself (4 int32 lanes per wide row).
      for (i <- 0 until M; j <- 0 until N) {
        val f = i * N + j
        val row = dramRead(cDram + (f / 4) * p.DtcmRowBytes)
        val got = (row >> (32 * (f % 4))) & mask32
        val exp = BigInt(Cref(i)(j)) & mask32
        assert(got == exp, s"C($i)($j): got $got expected $exp")
      }
    }
  }

  it should "round-trip data through DRAM with core-issued DMALD/DMAST" in {
    // No compute: load a tile from DRAM into DTCM, then DMAST it back to a new DRAM
    // address. Validates the DMA instruction path + DMAWAIT barrier in isolation.
    val tile = Array.tabulate(n, n)((i, j) => ((i * 3 + j + 1) % 11) - 5)
    val srcDram = 0; val dstDram = tileBytes; val dtcmBuf = 0

    val program =
      li(10, srcDram) ++ li(11, dtcmBuf) ++ li(12, dstDram) ++ Seq(
        Encode.dmald(10, 11, tileRows),    // DTCM[dtcmBuf] <- DRAM[srcDram]
        Encode.dmawait(),
        Encode.dmast(12, 11, tileRows),    // DRAM[dstDram] <- DTCM[dtcmBuf]
        Encode.halt,
      )

    test(new DmaTb(p, dramRows = 512)).withAnnotations(backendAnnos) { dut =>
      val axi = dut.io.axi
      val mask32 = (BigInt(1) << 32) - 1
      def step(): Unit = dut.clock.step(1)
      axi.aw.valid.poke(false.B); axi.w.valid.poke(false.B); axi.b.ready.poke(false.B)
      axi.ar.valid.poke(false.B); axi.r.ready.poke(false.B); dut.io.preWen.poke(false.B)

      def addrSend(ch: DecoupledIO[AxiAddr], addr: BigInt, beats: Int): Unit = {
        ch.bits.id.poke(0.U); ch.bits.addr.poke(addr.U)
        ch.bits.len.poke((beats - 1).U); ch.bits.size.poke(2.U); ch.bits.burst.poke(1.U)
        ch.valid.poke(true.B)
        while (!ch.ready.peek().litToBoolean) step()
        step(); ch.valid.poke(false.B)
      }
      def writeBurst(addr: BigInt, data: Seq[BigInt]): Unit = {
        addrSend(axi.aw, addr, data.length)
        for ((d, i) <- data.zipWithIndex) {
          axi.w.bits.data.poke((d & mask32).U); axi.w.bits.strb.poke("b1111".U)
          axi.w.bits.last.poke((i == data.length - 1).B); axi.w.valid.poke(true.B)
          while (!axi.w.ready.peek().litToBoolean) step()
          step()
        }
        axi.w.valid.poke(false.B); axi.b.ready.poke(true.B)
        while (!axi.b.valid.peek().litToBoolean) step()
        step(); axi.b.ready.poke(false.B)
      }
      def readWord(addr: BigInt): BigInt = {
        addrSend(axi.ar, addr, 1); axi.r.ready.poke(true.B)
        while (!axi.r.valid.peek().litToBoolean) step()
        val v = axi.r.bits.data.peek().litValue
        step(); axi.r.ready.poke(false.B); v
      }
      def dramPreload(base: Int, words: Seq[BigInt]): Unit = {
        for ((w, r) <- words.zipWithIndex) {
          dut.io.preAddr.poke((base + r * p.DtcmRowBytes).U); dut.io.preData.poke(w.U)
          dut.io.preWen.poke(true.B); step()
        }
        dut.io.preWen.poke(false.B)
      }
      def dramRead(addr: Int): BigInt = {
        dut.io.dbgAddr.poke(addr.U); step(); dut.io.dbgData.peek().litValue
      }

      dramPreload(srcDram, dram128(tile))
      writeBurst(ITCM, program)
      writeBurst(CTRL + OFF_CTRL, Seq(BigInt(1)))
      var guard = 0
      while ((readWord(CTRL + OFF_STATUS) & 1) == 0) {
        guard += 1; assert(guard < 2000, "DMA echo did not finish")
      }
      for (r <- 0 until tileRows)
        assert(dramRead(dstDram + r * p.DtcmRowBytes) == dram128(tile)(r),
          s"beat $r mismatch after DMALD/DMAST round-trip")
    }
  }
}
