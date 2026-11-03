// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable

// Fast Core-level test for the core-issued DMA instructions (DMALD/DMAST/DMAWAIT).
// The full-Npu AXI path is exercised by LayerSpec; here we drive a bare Core with
// an in-memory DTCM (like LayerSpec.runCore) plus a *mock DMA engine*: the harness
// pokes dmaIdle, watches the dmaStart pulse + descriptor, and performs the row
// move between a DRAM model and the DTCM model. This keeps the systolic-array
// simulation tractable on Treadle while validating the issue/stall/barrier logic
// and the program-visible data movement end to end.
class DmaCoreSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val p = Parameters()
  val n = p.MatrixDim
  val mask32 = (BigInt(1) << 32) - 1
  val wordsPerRow = p.WordsPerRow          // 32-bit words per 128-bit beat (physical, =4)
  val arrayRowWords = p.ArrayRowWords      // 32-bit words per matrix row (4 at n=16, 8 at n=32)
  val beatsPerRow = p.BeatsPerArrayRow     // physical beats per matrix row (1, 2)
  val rowBytes = p.DtcmRowBytes
  val tileBytes = n * p.ArrayRowBytes      // one n x n int8 tile footprint
  val tileRows  = n * beatsPerRow          // beats to DMA one tile

  def li(rd: Int, imm: Int): Seq[BigInt] =
    if (imm >= -2048 && imm < 2048) Seq(Encode.addi(rd, 0, imm))
    else {
      val hi = (imm + 0x800) >> 12
      Seq(Encode.lui(rd, hi & 0xfffff), Encode.addi(rd, rd, imm - (hi << 12)))
    }

  def rowWords(row: Array[Int]): Seq[BigInt] =
    (0 until arrayRowWords).map { g =>
      (0 until p.ElemsPerWord).foldLeft(BigInt(0)) { (acc, e) =>
        acc | (BigInt(row(g * p.ElemsPerWord + e) & 0xff) << (8 * e))
      }
    }
  def subTile(m: Array[Array[Int]], r0: Int, c0: Int): Array[Array[Int]] =
    Array.tabulate(n, n)((r, c) => m(r0 + r)(c0 + c))
  def loadTile(mem: mutable.Map[BigInt, BigInt], base: Int, tile: Array[Array[Int]]): Unit =
    for (r <- 0 until n; (w, g) <- rowWords(tile(r)).zipWithIndex)
      mem(BigInt(base + (r * arrayRowWords + g) * 4)) = w

  // Run `program` against an in-memory DTCM (seeded with `dtcmInit`) and a DRAM
  // model (seeded with `dramInit`), with a mock DMA engine. Returns (dtcm, dram).
  def runCoreDma(program: Seq[BigInt],
                 dtcmInit: Map[BigInt, BigInt] = Map(),
                 dramInit: Map[BigInt, BigInt] = Map(),
                 maxCycles: Int = 50000): (mutable.Map[BigInt, BigInt], mutable.Map[BigInt, BigInt]) = {
    def instWord(byteAddr: BigInt): BigInt = {
      val idx = (byteAddr >> 2).toInt
      if (idx >= 0 && idx < program.length) program(idx) else BigInt(0)
    }
    val dtcm = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
    val dram = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
    dtcmInit.foreach { case (k, v) => dtcm(k) = v }
    dramInit.foreach { case (k, v) => dram(k) = v }

    test(new Core(p)).withAnnotations(backendAnnos) { dut =>
      var pendingRead: Option[BigInt] = None
      var instAddrPrev: BigInt = 0
      var dmaBusy = 0          // cycles left on the in-flight mock transfer
      var curTokEn = false     // lease of the executing descriptor
      var curTokId = 0
      dut.io.dtcm.ready.poke(true.B)
      dut.clock.setTimeout(0)

      def packWide(a: BigInt): BigInt =
        (0 until wordsPerRow).foldLeft(BigInt(0))((acc, i) => acc | (dtcm(a + 4 * i) << (32 * i)))
      def writeWide(a: BigInt, data: BigInt, strb: BigInt): Unit =
        for (i <- 0 until wordsPerRow) {
          var word = dtcm(a + 4 * i)
          for (b <- 0 until 4) if (((strb >> (i * 4 + b)) & 1) == 1) {
            val nb = (data >> (32 * i + 8 * b)) & 0xff
            word = (word & ~(BigInt(0xff) << (8 * b))) | (nb << (8 * b))
          }
          dtcm(a + 4 * i) = word & mask32
        }
      // Move `rows` 16-byte rows between the two models (4 words per row).
      def dmaMove(dir: Boolean, dramA: BigInt, dtcmA: BigInt, rows: BigInt): Unit =
        for (r <- 0 until rows.toInt; w <- 0 until wordsPerRow) {
          val off = r * rowBytes + w * 4
          if (dir) dram(dramA + off) = dtcm(dtcmA + off)   // store DTCM->DRAM
          else     dtcm(dtcmA + off) = dram(dramA + off)   // load  DRAM->DTCM
        }

      def tick(): Unit = {
        // Poke ALL combinational inputs (esp. dmaIdle, which feeds exStall ->
        // itcmAddr) BEFORE sampling itcmAddr for the next fetch. On the real board
        // dmaIdle is a registered signal, so this models its in-cycle stability.
        dut.io.itcmInst.poke(instWord(instAddrPrev).U)
        dut.io.dtcm.rdata.poke(pendingRead.map(packWide).getOrElse(BigInt(0)).U)
        dut.io.dmaIdle.poke((dmaBusy == 0).B)
        instAddrPrev = dut.io.itcmAddr.peek().litValue

        // Mock DMA: a start pulse latches the descriptor and moves the data; the
        // engine then reports busy for ~rows cycles so DMAWAIT actually waits.
        // The descriptor's token lease is latched with it and echoed on the
        // completion pulse -- retirement is per descriptor (Alu's contract), so
        // a mock that never pulses dmaDone deadlocks on the first DMAWAIT.
        var done = false; var doneEn = false; var doneId = 0
        if (dut.io.dmaStart.peek().litToBoolean && dmaBusy == 0) {
          val dir  = dut.io.dmaDir.peek().litToBoolean
          val dramA = dut.io.dmaDram.peek().litValue
          val dtcmA = dut.io.dmaDtcm.peek().litValue
          val rows  = dut.io.dmaRows.peek().litValue
          dmaMove(dir, dramA, dtcmA, rows)
          curTokEn = dut.io.dmaTok.en.peek().litToBoolean
          curTokId = dut.io.dmaTok.id.peek().litValue.toInt
          dmaBusy = rows.toInt + 3
        }
        if (dmaBusy > 0) {
          dmaBusy -= 1
          if (dmaBusy == 0) { done = true; doneEn = curTokEn; doneId = curTokId }
        }
        dut.io.dmaDone.poke(done.B)
        dut.io.dmaDoneTok.en.poke(doneEn.B)
        dut.io.dmaDoneTok.id.poke(doneId.U)

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
      var cycles = 0L
      while (!dut.io.done.peek().litToBoolean) {
        tick(); cycles += 1
        assert(cycles < maxCycles, "core did not halt")
      }
    }
    (dtcm, dram)
  }

  behavior of "core-issued DMA"

  it should "round-trip a tile DRAM -> DTCM -> DRAM via DMALD/DMAWAIT/DMAST" in {
    val tile = Array.tabulate(n, n)((i, j) => ((i * 3 + j + 1) % 11) - 5)
    val srcDram = 0; val dstDram = tileBytes; val dtcmBuf = 0
    val dram = mutable.Map[BigInt, BigInt]()
    loadTile(dram, srcDram, tile)   // pack the tile into the DRAM model

    val program =
      li(10, srcDram) ++ li(11, dtcmBuf) ++ li(12, dstDram) ++ Seq(
        Encode.dmald(10, 11, tileRows),
        Encode.dmawait(),
        Encode.dmast(12, 11, tileRows),
        Encode.halt,
      )
    val (_, outDram) = runCoreDma(program, dramInit = dram.toMap)
    for (r <- 0 until n; (w, g) <- rowWords(tile(r)).zipWithIndex)
      assert(outDram(BigInt(dstDram + (r * arrayRowWords + g) * 4)) == w,
        s"row $r word $g mismatch after round-trip")
  }

  it should "run a double-buffered tiled GEMM streaming weights from DRAM" in {
    val M = n; val N = n; val K = 2 * n
    val X = Array.tabulate(M, K)((i, j) => ((i + 2 * j + 1) % 7) - 3)
    val W = Array.tabulate(K, N)((i, j) => ((3 * i + j + 2) % 5) - 2)
    val Cref = Array.tabulate(M, N)((i, j) => (0 until K).map(k => X(i)(k) * W(k)(j)).sum)

    val xT0 = 0; val xT1 = tileBytes; val wBufA = 2 * tileBytes
    val wBufB = 3 * tileBytes; val cBase = 4 * tileBytes
    val wDram0 = 0; val wDram1 = tileBytes; val cDram = 2 * tileBytes

    val dtcm = mutable.Map[BigInt, BigInt]()
    loadTile(dtcm, xT0, subTile(X, 0, 0)); loadTile(dtcm, xT1, subTile(X, 0, n))
    val dram = mutable.Map[BigInt, BigInt]()
    loadTile(dram, wDram0, subTile(W, 0, 0)); loadTile(dram, wDram1, subTile(W, n, 0))

    val program =
      li(10, xT0) ++ li(11, xT1) ++ li(12, cBase) ++
      li(13, wDram0) ++ li(14, wDram1) ++ li(15, wBufA) ++
      li(16, wBufB) ++ li(17, cDram) ++ Seq(
        Encode.dmald(13, 15, tileRows),
        Encode.dmawait(),
        Encode.dmald(14, 16, tileRows),
        Encode.lma(10), Encode.lmb(15), Encode.matmul(),
        Encode.dmawait(),
        Encode.lma(11), Encode.lmb(16), Encode.matmulAcc(),
        Encode.sm(12),
        Encode.dmast(17, 12, n * n * 4 / rowBytes),
        Encode.halt,
      )
    val (_, outDram) = runCoreDma(program, dtcmInit = dtcm.toMap, dramInit = dram.toMap)
    for (i <- 0 until M; j <- 0 until N) {
      val exp = BigInt(Cref(i)(j)) & mask32
      val got = outDram(BigInt(cDram + (i * N + j) * 4))
      assert(got == exp, s"C($i)($j): got $got expected $exp")
    }
  }
}
