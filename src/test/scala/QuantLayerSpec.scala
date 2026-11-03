// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable

// End-to-end quantized linear layer built only from the vector primitives. The
// one-kick Core program computes an activation scale with VABS+VMAX, applies the
// scale with VSMUL, converts with VCVT.I8, runs int8 GEMV, converts int32 output
// with VCVT.BF16, and applies activation/per-channel scales with VSMUL+VMUL. This
// deliberately keeps quantization policy in the program rather than in the RTL.
class QuantLayerSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val p = Parameters()
  val n = p.MatrixDim                       // default 16: K, N and array dimension
  val mask32 = (BigInt(1) << 32) - 1
  val wordsPerRow = p.WordsPerRow           // 4 (32-bit words per 128-bit beat)
  val arrayRowWords = p.ArrayRowWords       // 4 (32-bit words per int8 array row)
  val beatsPerRow = p.BeatsPerArrayRow      // 1
  val rowBytes = p.DtcmRowBytes             // 16
  val tileRows = n * beatsPerRow            // beats to DMA one int8 tile (16)

  // ---- bf16 <-> float ------------------------------------------------------
  def f2bf16(f: Float): BigInt = {
    if (f.isNaN) return BigInt(0x7FC0)
    val b = java.lang.Float.floatToIntBits(f)
    val lsb = (b >>> 16) & 1
    BigInt((b + 0x7FFF + lsb) >>> 16) & 0xFFFF
  }
  def bf162f(b: BigInt): Float = java.lang.Float.intBitsToFloat((b.toInt & 0xFFFF) << 16)
  def f32bits(f: Float): BigInt = BigInt(java.lang.Float.floatToIntBits(f).toLong & 0xFFFFFFFFL)

  def li(rd: Int, imm: Int): Seq[BigInt] =
    if (imm >= -2048 && imm < 2048) Seq(Encode.addi(rd, 0, imm))
    else {
      val hi = (imm + 0x800) >> 12
      Seq(Encode.lui(rd, hi & 0xfffff), Encode.addi(rd, rd, imm - (hi << 12)))
    }
  def liBits(rd: Int, bits: BigInt): Seq[BigInt] = {
    val hi = ((bits + 0x800) >> 12) & 0xfffff
    val lo = (bits - (hi << 12)).toInt & 0xfff
    Seq(Encode.lui(rd, hi.toInt), Encode.addi(rd, rd, if (lo >= 0x800) lo - 0x1000 else lo))
  }

  // Signed round-half-to-even fp32->int8 with native signed saturation, mirroring
  // VectorUnit.f32ToI8RneSat.
  def cvtI8(x: Float): Int = math.max(-128, math.min(127, math.rint(x).toInt))

  // Minimal Core harness: in-memory DTCM (32-bit-word keyed), no DMA. Runs the
  // program to HALT and returns the final DTCM. Shared by the tests below.
  def runCore(prog: Seq[BigInt], dtcm: mutable.Map[BigInt, BigInt], maxCycles: Int): Unit = {
    def instWord(byteAddr: BigInt): BigInt = {
      val idx = (byteAddr >> 2).toInt
      if (idx >= 0 && idx < prog.length) prog(idx) else BigInt(0)
    }
    test(new Core(p)).withAnnotations(backendAnnos) { dut =>
      var pendingRead: Option[BigInt] = None
      var instAddrPrev: BigInt = 0
      dut.io.dtcm.ready.poke(true.B)
      dut.io.dmaIdle.poke(true.B)
      dut.clock.setTimeout(0)
      def packWide(a: BigInt): BigInt =
        (0 until wordsPerRow).foldLeft(BigInt(0))((acc, i) => acc | (dtcm(a + 4 * i) << (32 * i)))
      def tick(): Unit = {
        dut.io.itcmInst.poke(instWord(instAddrPrev).U)
        dut.io.dtcm.rdata.poke(pendingRead.map(packWide).getOrElse(BigInt(0)).U)
        instAddrPrev = dut.io.itcmAddr.peek().litValue
        val v = dut.io.dtcm.valid.peek().litToBoolean
        val w = dut.io.dtcm.write.peek().litToBoolean
        val a = dut.io.dtcm.addr.peek().litValue
        if (v && w) {
          val wd = dut.io.dtcm.wdata.peek().litValue
          val strb = dut.io.dtcm.wstrb.peek().litValue
          for (i <- 0 until wordsPerRow) {
            var word = dtcm(a + 4 * i)
            for (b <- 0 until 4) if (((strb >> (i * 4 + b)) & 1) == 1) {
              val nb = (wd >> (32 * i + 8 * b)) & 0xff
              word = (word & ~(BigInt(0xff) << (8 * b))) | (nb << (8 * b))
            }
            dtcm(a + 4 * i) = word & mask32
          }
          pendingRead = None
        } else if (v) pendingRead = Some(a) else pendingRead = None
        dut.clock.step(1)
      }
      dut.io.start.poke(true.B); tick(); dut.io.start.poke(false.B)
      var cyc = 0
      while (!dut.io.done.peek().litToBoolean) {
        tick(); cyc += 1; assert(cyc < maxCycles, "core did not halt")
      }
    }
  }

  behavior of "vector format primitives in the Core"

  it should "compose scaling and conversions without a fixed quantization instruction" in {
    val XN = 0x0000; val XS = 0x0200; val AQ = 0x0400
    val CI = 0x1000; val CB = 0x1800; val CS = 0x1c00; val SWb = 0x0100; val YO = 0x2000
    val x  = Array.tabulate(n)(i => ((i % 9) - 4) * 0.35f + 0.1f)
    val cInt = Array.tabulate(n)(i => (i - 15) * 41)      // preset int32 "accumulators"
    val sw = Array.tabulate(n)(j => 0.02f + 0.005f * j)
    val invF = 20.0f; val sxF = 0.03f

    val dtcm = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
    // Seed x and the per-column scales as bf16, and the matrix result as int32.
    for (i <- 0 until n) {
      val wa = XN + (i / 8) * rowBytes + ((i % 8) / 2) * 4
      dtcm(BigInt(wa)) = dtcm(BigInt(wa)) | (f2bf16(x(i)) << (16 * (i % 2)))
      val ws = SWb + (i / 8) * rowBytes + ((i % 8) / 2) * 4
      dtcm(BigInt(ws)) = dtcm(BigInt(ws)) | (f2bf16(sw(i)) << (16 * (i % 2)))
    }
    // seed int32 accumulators (4 per 128-bit row)
    for (i <- 0 until n) dtcm(BigInt(CI + i * 4)) = BigInt(cInt(i).toLong & 0xFFFFFFFFL)

    // The program selects its scale policy by composing VSMUL/VMUL around explicit
    // representation conversions. VCVT instructions consume no scale or srcB.
    val prog =
      liBits(20, f32bits(invF)) ++ liBits(21, f32bits(sxF)) ++
      li(7, XN) ++ li(8, XS) ++ li(9, AQ) ++ li(10, CI) ++ li(11, CB) ++ li(12, CS) ++
      li(13, SWb) ++ li(14, YO) ++ li(23, n) ++ Seq(
        Encode.vsets(20), Encode.vseta(7), Encode.vsetd(8), Encode.vsetlen(23), Encode.vsmul(),
        Encode.vseta(8), Encode.vsetd(9), Encode.vsetlen(23), Encode.vcvtI8(),
        Encode.vseta(10), Encode.vsetd(11), Encode.vsetlen(23), Encode.vcvtBf16(),
        Encode.vsets(21), Encode.vseta(11), Encode.vsetd(12), Encode.vsetlen(23), Encode.vsmul(),
        Encode.vseta(12), Encode.vsetb(13), Encode.vsetd(14), Encode.vsetlen(23), Encode.vmul(),
        Encode.halt,
      )
    runCore(prog, dtcm, 8000)

    // VSMUL + VCVT.I8: int8 at AQ.
    def rdI8(base: Int, i: Int): Int = {
      val b = ((dtcm(BigInt(base + (i / 16) * rowBytes + (i % 16) / 4 * 4)) >>
                (8 * (i % 4))) & 0xFF).toInt
      if (b >= 128) b - 256 else b
    }
    for (i <- 0 until n) {
      val scaled = bf162f(f2bf16(bf162f(f2bf16(x(i))) * invF))
      assert(rdI8(AQ, i) == cvtI8(scaled), s"vcvt.i8 lane $i: got ${rdI8(AQ, i)}")
    }
    // VCVT.BF16 + VSMUL + VMUL: bf16 at YO.
    for (j <- 0 until n) {
      val converted = bf162f(f2bf16(cInt(j).toFloat))
      val scaled = bf162f(f2bf16(converted * sxF))
      val exp = f2bf16(scaled * bf162f(f2bf16(sw(j))))
      val got = (dtcm(BigInt(YO + (j / 8) * rowBytes + (j % 8) / 2 * 4)) >> (16 * (j % 2))) & 0xFFFF
      assert((exp - got).abs <= 1, s"converted lane $j: got 0x${got.toString(16)} exp 0x${exp.toString(16)}")
    }
  }

  behavior of "on-PL quantized linear layer"

  it should "match a quantized GEMV reference end to end" in {
    // ---- operands ----------------------------------------------------------
    val rng = new scala.util.Random(2)
    val x  = Array.tabulate(n)(i => ((i % 9) - 4) * 0.35f + 0.1f)          // activation (bf16)
    val W  = Array.tabulate(n, n)((k, j) => ((3 * k + 2 * j + 1) % 7) - 3) // int8 weights
    val sw = Array.tabulate(n)(j => 0.02f + 0.005f * j)                    // per-column scales

    // ---- DTCM / DRAM byte layout -------------------------------------------
    val XN = 0x0000; val SWb = 0x0100
    val SC = 0x0200; val SC2 = 0x0210; val SC3 = 0x0220; val SC4 = 0x0230
    val XA = 0x0300; val AQ = 0x0400; val XQS = 0x0500; val WB = 0x0800
    val CI = 0x1000; val CB = 0x1800; val CS = 0x1c00; val YO = 0x2000
    val wDram = 0x0000

    // DTCM word-addressed model (32-bit words), matching the mock harness below.
    val dtcm = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
    val dram = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
    def putBf16(base: Int, vals: Seq[Float]): Unit =
      vals.zipWithIndex.foreach { case (v, i) =>
        val wa = base + (i / 8) * rowBytes + ((i % 8) / 2) * 4
        val sh = 16 * (i % 2)
        dtcm(BigInt(wa)) = (dtcm(BigInt(wa)) & ~(BigInt(0xFFFF) << sh)) | (f2bf16(v) << sh)
      }
    def getBf16(m: mutable.Map[BigInt, BigInt], base: Int, i: Int): BigInt = {
      val wa = base + (i / 8) * rowBytes + ((i % 8) / 2) * 4
      (m(BigInt(wa)) >> (16 * (i % 2))) & 0xFFFF
    }
    putBf16(XN, x)
    putBf16(SWb, sw)
    // W int8 -> DRAM, packed as the matmul array rows (arrayRowWords words / row).
    for (k <- 0 until n; g <- 0 until arrayRowWords)
      dram(BigInt(wDram + (k * arrayRowWords + g) * 4)) =
        (0 until p.ElemsPerWord).foldLeft(BigInt(0)) { (acc, e) =>
          acc | (BigInt(W(k)(g * p.ElemsPerWord + e) & 0xff) << (8 * e))
        }

    // ---- kernel program ----------------------------------------------------
    // GPRs: 5=wDram 6=WB 7=XN 8=XA 9=XQS 10=AQ 11=SWb 12=CI 13=CB 14=CS 15=YO
    //       16=SC 17=SC2 18=SC3 19=SC4 20=127.0 21=1/127 22=amax 23=n 24=1
    //       25=tmp 26=inv 27=s_x
    val prog =
      li(5, wDram) ++ li(6, WB) ++ li(7, XN) ++ li(8, XA) ++ li(9, XQS) ++ li(10, AQ) ++
      li(11, SWb) ++ li(12, CI) ++ li(13, CB) ++ li(14, CS) ++ li(15, YO) ++
      li(16, SC) ++ li(17, SC2) ++ li(18, SC3) ++ li(19, SC4) ++
      li(23, n) ++ li(24, 1) ++
      liBits(20, f32bits(127.0f)) ++ liBits(21, f32bits(1.0f / 127.0f)) ++ Seq(
        Encode.dmald(5, 6, tileRows), Encode.dmawait(),          // stream weights DRAM->DTCM
        // Software-defined activation scale: abs -> max -> reciprocal.
        Encode.vseta(7), Encode.vsetd(8), Encode.vsetlen(23), Encode.vabs(),
        Encode.vseta(8), Encode.vsetlen(23), Encode.vmax(22),   // x22 = max(|x|)
        Encode.srli(25, 22, 16),                                // bf16(amax) in low 16
        Encode.sw(25, 16, 0),                                   // DTCM[SC] = amax(bf16)
        Encode.vseta(16), Encode.vsetlen(24), Encode.vsetd(17), Encode.vrecip(),  // SC2 = 1/amax
        Encode.vseta(17), Encode.vsets(20), Encode.vsetd(18), Encode.vsmul(),     // SC3 = 127/amax = inv
        Encode.vseta(16), Encode.vsets(21), Encode.vsetd(19), Encode.vsmul(),     // SC4 = amax/127 = s_x
        Encode.lw(25, 18, 0), Encode.slli(26, 25, 16),           // x26 = inv (bf16 widened to f32)
        Encode.lw(25, 19, 0), Encode.slli(27, 25, 16),           // x27 = s_x
        // Quantization policy in software: scale first, then convert to int8.
        Encode.vsets(26), Encode.vseta(7), Encode.vsetd(9), Encode.vsetlen(23), Encode.vsmul(),
        Encode.vseta(9), Encode.vsetd(10), Encode.vsetlen(23), Encode.vcvtI8(),
        // int8 GEMV
        Encode.lma(10), Encode.lmb(6), Encode.matmul(), Encode.sm(12),
        // Dequantization policy in software: convert, then apply scale factors.
        Encode.vseta(12), Encode.vsetd(13), Encode.vsetlen(23), Encode.vcvtBf16(),
        Encode.vsets(27), Encode.vseta(13), Encode.vsetd(14), Encode.vsetlen(23), Encode.vsmul(),
        Encode.vseta(14), Encode.vsetb(11), Encode.vsetd(15), Encode.vsetlen(23), Encode.vmul(),
        Encode.halt,
      )

    // ---- run on a bare Core with an in-memory DTCM and a mock DMA engine ----
    def instWord(byteAddr: BigInt): BigInt = {
      val idx = (byteAddr >> 2).toInt
      if (idx >= 0 && idx < prog.length) prog(idx) else BigInt(0)
    }
    test(new Core(p)).withAnnotations(backendAnnos) { dut =>
      var pendingRead: Option[BigInt] = None
      var instAddrPrev: BigInt = 0
      var dmaBusy = 0
      var curTokEn = false
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
      def dmaMove(dir: Boolean, dramA: BigInt, dtcmA: BigInt, rows: BigInt): Unit =
        for (r <- 0 until rows.toInt; w <- 0 until wordsPerRow) {
          val off = r * rowBytes + w * 4
          if (dir) dram(dramA + off) = dtcm(dtcmA + off)
          else     dtcm(dtcmA + off) = dram(dramA + off)
        }

      def tick(): Unit = {
        dut.io.itcmInst.poke(instWord(instAddrPrev).U)
        dut.io.dtcm.rdata.poke(pendingRead.map(packWide).getOrElse(BigInt(0)).U)
        dut.io.dmaIdle.poke((dmaBusy == 0).B)
        instAddrPrev = dut.io.itcmAddr.peek().litValue
        var done = false; var doneEn = false; var doneId = 0
        if (dut.io.dmaStart.peek().litToBoolean && dmaBusy == 0) {
          val dir  = dut.io.dmaDir.peek().litToBoolean
          val dramA = dut.io.dmaDram.peek().litValue
          val dtcmA = dut.io.dmaDtcm.peek().litValue
          val rows  = dut.io.dmaRows.peek().litValue
          dmaMove(dir, dramA, dtcmA, rows)
          // The lease travels with the descriptor and returns on completion.
          curTokEn = dut.io.dmaTok.en.peek().litToBoolean
          curTokId = dut.io.dmaTok.id.peek().litValue.toInt
          dmaBusy = rows.toInt + 3
        }
        if (dmaBusy > 0) {
          dmaBusy -= 1
          if (dmaBusy == 0) { done = true; doneEn = curTokEn; doneId = curTokId }
        }
        // Completion pulse: retirement is per descriptor now (DMAWAIT drains on
        // the lease count, not on engine idleness).
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
        assert(cycles < 20000, "core did not halt")
      }
    }

    // ---- golden: use the hardware's own inv / s_x, mirror the datapath -------
    val invF = bf162f(dtcm(BigInt(SC3)) & 0xFFFF)     // hardware-computed inv (bf16)
    val sxF  = bf162f(dtcm(BigInt(SC4)) & 0xFFFF)     // hardware-computed s_x (bf16)
    assert(invF > 0f && sxF > 0f, s"scales not computed: inv=$invF sx=$sxF")

    val xq = (0 until n).map { k =>
      val scaled = bf162f(f2bf16(bf162f(f2bf16(x(k))) * invF))
      cvtI8(scaled)
    }
    val cInt = (0 until n).map(j => (0 until n).map(k => xq(k) * W(k)(j)).sum)
    for (j <- 0 until n) {
      val converted = bf162f(f2bf16(cInt(j).toFloat))
      val t1 = bf162f(f2bf16(converted * sxF))
      val exp = f2bf16(t1 * bf162f(f2bf16(sw(j))))
      val got = getBf16(dtcm, YO, j)
      // allow a 1-ulp bf16 difference from mul-rounding order.
      val diff = (exp - got).abs
      assert(diff <= 1, s"y[$j]: got 0x${got.toString(16)} (${bf162f(got)}) exp 0x${exp.toString(16)} (${bf162f(exp)}), cInt=${cInt(j)}")
    }
  }
}
