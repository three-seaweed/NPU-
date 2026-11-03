// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable

// Core-level contracts of the core-issued DMA token path under the depth-2
// descriptor FIFO (dmaIdle = FIFO room, not engine idleness):
//   1. two back-to-back DMALDs both issue without waiting for the engine: the
//      second stages behind the first, and readers of BOTH regions are held by
//      the per-descriptor write leases until each completion echo retires them;
//   2. a reader of a DMA-written region (vector compute) is held behind the
//      DMA write lease and never sees stale rows;
//   3. WAR staging: DMAST reads out the OLD rows, a following DMALD to the same
//      region installs the NEW rows, readers see the NEW rows, and nothing
//      deadlocks (bounded cycles);
//   4. a zero-row DMALD completes at its pop, in order, echoing no lease;
//   5. three descriptors in flight fill the 2-bit lease counter to 3; HALT
//      drains all three (an early "drained" -- wrap or miscount -- loses data).
//
// Harness: bare Core + in-memory DTCM + mock DMA engine that models the FIFO
// (2 staged entries + one executing), executes strictly in order, moves the
// rows at the pop, and echoes each descriptor's own lease tag on completion.
// Real-engine AXI timing is covered by DmaEngineSpec/DmaArbSpec at engine and
// Npu level.
class DmaTokenTimingSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val p = Parameters()
  val mask32 = (BigInt(1) << 32) - 1
  val wordsPerRow = p.WordsPerRow
  val rowBytes = p.DtcmRowBytes

  def li(rd: Int, imm: Int): Seq[BigInt] =
    if (imm >= -2048 && imm < 2048) Seq(Encode.addi(rd, 0, imm))
    else {
      val hi = (imm + 0x800) >> 12
      Seq(Encode.lui(rd, hi & 0xfffff), Encode.addi(rd, rd, imm - (hi << 12)))
    }

  // (cycle, lease-en, lease-id) of one enqueue or one completion.
  case class TagEv(cyc: Int, en: Boolean, id: Int)
  case class RunOut(dtcm: mutable.Map[BigInt, BigInt],
                    dram: mutable.Map[BigInt, BigInt],
                    enqLog: mutable.ArrayBuffer[TagEv],
                    doneLog: mutable.ArrayBuffer[TagEv])

  def runCore(program: Seq[BigInt],
              dtcmInit: Map[BigInt, BigInt] = Map(),
              dramInit: Map[BigInt, BigInt] = Map(),
              maxCycles: Int = 20000): RunOut = {
    def instWord(byteAddr: BigInt): BigInt = {
      val idx = (byteAddr >> 2).toInt
      if (idx >= 0 && idx < program.length) program(idx) else BigInt(0)
    }
    val dtcm = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
    val dram = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
    dtcmInit.foreach { case (k, v) => dtcm(k) = v }
    dramInit.foreach { case (k, v) => dram(k) = v }
    val enqLog  = mutable.ArrayBuffer[TagEv]()
    val doneLog = mutable.ArrayBuffer[TagEv]()

    test(new Core(p)).withAnnotations(backendAnnos) { dut =>
      var pendingRead: Option[BigInt] = None
      var instAddrPrev: BigInt = 0

      case class MockDesc(dir: Boolean, dramA: BigInt, dtcmA: BigInt, rows: Int,
                          tokEn: Boolean, tokId: Int)
      val dq = mutable.Queue[MockDesc]()     // staged behind the executing one
      var exec: Option[MockDesc] = None      // descriptor being "transferred"
      var engineBusy = 0                     // cycles left on the executing transfer
      var cyc = 0

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
        // dmaIdle = FIFO room (depth 2), exactly like the RTL engine. The model
        // queue counts staged entries only; the executing one has been popped.
        val room = dq.size < 2
        dut.io.dmaIdle.poke(room.B)
        instAddrPrev = dut.io.itcmAddr.peek().litValue

        // Enqueue side: one descriptor per fire; its lease tag travels with it.
        if (dut.io.dmaStart.peek().litToBoolean && room) {
          val d = MockDesc(dut.io.dmaDir.peek().litToBoolean,
            dut.io.dmaDram.peek().litValue, dut.io.dmaDtcm.peek().litValue,
            dut.io.dmaRows.peek().litValue.toInt,
            dut.io.dmaTok.en.peek().litToBoolean,
            dut.io.dmaTok.id.peek().litValue.toInt)
          dq.enqueue(d)
          enqLog += TagEv(cyc, d.tokEn, d.tokId)
        }

        // Engine side: pop the head when idle. A zero-row entry completes at its
        // pop (no transfer, no busy window); a real one runs rows+3 cycles and
        // echoes the lease latched with it.
        var done = false; var doneEn = false; var doneId = 0
        if (engineBusy == 0 && exec.isEmpty && dq.nonEmpty) {
          val d = dq.dequeue()
          if (d.rows == 0) {
            done = true; doneEn = d.tokEn; doneId = d.tokId
          } else {
            dmaMove(d.dir, d.dramA, d.dtcmA, d.rows)
            exec = Some(d)
            engineBusy = d.rows + 3
          }
        }
        if (engineBusy > 0) {
          engineBusy -= 1
          if (engineBusy == 0) {
            val d = exec.get
            done = true; doneEn = d.tokEn; doneId = d.tokId
            exec = None
          }
        }
        if (done) doneLog += TagEv(cyc, doneEn, doneId)
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
        cyc += 1
        dut.clock.step(1)
      }

      dut.io.start.poke(true.B); tick(); dut.io.start.poke(false.B)
      var cycles = 0L
      while (!dut.io.done.peek().litToBoolean) {
        tick(); cycles += 1
        // The deadlock signature of a leaked DMA lease is a scalar reader (or
        // HALT drain) that never proceeds; the bound turns it into a failure
        // with a cycle count instead of a hang.
        assert(cycles < maxCycles, s"core did not halt after $cycles cycles (token leak?)")
      }
    }
    RunOut(dtcm, dram, enqLog, doneLog)
  }

  def rowPattern(base: BigInt, tag: Int): Map[BigInt, BigInt] =
    (0 until 16).map(r => (base + r * rowBytes, BigInt(tag * 0x10000 + r))).toMap

  behavior of "core-issued DMA token timing under the descriptor FIFO"

  it should "stage a back-to-back DMALD behind the first and keep both regions' leases retiring" in {
    val aDram = 0x10000; val bDram = 0x20000
    val aDtcm = 0x400; val bDtcm = 0x800
    val rows = 8
    val dram = rowPattern(aDram, 0xA1) ++ rowPattern(bDram, 0xB2)
    // No DMAWAIT between the two loads: with FIFO room the second stages while
    // the first transfers (it no longer waits for engine idleness).
    val program =
      li(10, aDram) ++ li(11, aDtcm) ++ li(12, bDram) ++ li(13, bDtcm) ++
      li(23, 0x1000) ++ Seq(
        Encode.dmald(10, 11, rows),
        Encode.dmald(12, 13, rows),
        // Scalar readers of BOTH regions straight after (no DMAWAIT): held by
        // the write leases, must observe the DMA'd rows, must not hang.
        Encode.lw(20, 11, 0), Encode.lw(21, 13, rowBytes * 4),
        Encode.sw(20, 23, 0), Encode.sw(21, 23, 4),
        Encode.halt,
      )
    val out = runCore(program, dramInit = dram)
    assert(out.enqLog.size == 2, s"expected 2 DMA issues, saw ${out.enqLog.size}")
    assert(out.doneLog.size == 2, s"expected 2 completions, saw ${out.doneLog.size}")
    // Staging evidence: the second descriptor was accepted before the first
    // completed. (Under the old depth-1 gate this ordering was impossible.)
    assert(out.enqLog(1).cyc < out.doneLog(0).cyc,
      s"second DMALD did not stage behind the first: enq=${out.enqLog.map(_.cyc)} " +
      s"done=${out.doneLog.map(_.cyc)}")
    assert(out.dtcm(BigInt(0x1000)) == BigInt(0x00A10000))
    assert(out.dtcm(BigInt(0x1004)) == BigInt(0x00B20004))
  }

  // 16 DRAM rows whose every bf16 lane is a distinct normal in [1.0, 2.0):
  // lane bits 0x3F80..0x3F87, so scaling by 2.0 is exactly +0x80 per lane
  // (exponent bump, no rounding). Subnormal/zero patterns would break the
  // bit-exact expectation.
  def normalLanes(base: BigInt): Map[BigInt, BigInt] = {
    // A 128-bit row holds eight bf16 elements = two per 32-bit word: element k
    // lives at word k/2, half k%2. (An earlier 4-lanes-per-word packing was
    // wrong -- the hardware dutifully computed it and the assert fired.)
    def elem(k: Int): BigInt = BigInt(0x3F80 + (k & 7))
    (0 until 16).flatMap { r =>
      (0 until 4).map { w =>
        (base + r * rowBytes + w * 4,
          elem(r * 8 + 2 * w) | (elem(r * 8 + 2 * w + 1) << 16))
      }
    }.toMap
  }

  it should "hold a vector reader behind a live DMALD write lease (cross-unit WAR)" in {
    val wDram = 0x30000; val wDtcm = 0x400  // 16 rows = 128 bf16 lanes
    val dDtcm = 0x800
    val dram = normalLanes(wDram)
    // srcA = the DMA target; VSMUL by 2.0 (bit-exact) issues only once the DMA
    // lease retires. VSMUL reads rows = vlen/8 with vlen = 128 elems.
    val prog = li(10, wDram) ++ li(11, wDtcm) ++ li(14, dDtcm) ++ Seq(
        Encode.dmald(10, 11, 16),
        Encode.vseta(11), Encode.vsetd(14)) ++ li(15, 128) ++ Seq(
        Encode.vsetlen(15)) ++ li(16, 0x40000000) ++ Seq(  // fp32 2.0
        Encode.vsets(16),
        Encode.vsmul(),
        Encode.dmawait(),
        Encode.halt,
      )
    val out = runCore(prog, dramInit = dram)
    for (r <- 0 until 16; w <- 0 until 4) {
      val src = dram(wDram + r * rowBytes + w * 4)
      val exp = ((src & 0xffff) + 0x80) | ((((src >> 16) & 0xffff) + 0x80) << 16)
      val got = out.dtcm(BigInt(dDtcm + r * rowBytes + w * 4))
      val (gs, es) = (got.toString(16), exp.toString(16))
      assert(got == exp, s"dst row $r word $w: got $gs expected $es")
    }
  }

  it should "stage WAR data across DMAST then DMALD to the same region without a barrier" in {
    val rDtcm = 0x400; val outDram = 0x40000
    val newDram = 0x50000
    val rows = 8
    val dtcmOld = rowPattern(rDtcm, 0xD4)
    val dramNew = rowPattern(newDram, 0xE5)
    val program =
      li(10, outDram) ++ li(11, rDtcm) ++ li(12, newDram) ++ li(23, 0x1000) ++ Seq(
        Encode.dmast(10, 11, rows),   // store OLD rows out
        Encode.dmald(12, 11, rows),   // overwrite with NEW rows (WAR, no barrier)
        Encode.lw(20, 11, 0),         // reader must see the NEW row 0
        Encode.sw(20, 23, 0),
        Encode.halt,
      )
    val out = runCore(program, dtcmInit = dtcmOld, dramInit = dramNew)
    assert(out.dram(BigInt(outDram)) == BigInt(0x00D40000), "DMAST stored post-overwrite rows")
    assert(out.dtcm(BigInt(0x1000)) == BigInt(0x00E50000), "reader saw stale rows")
    // In-order execution: the store's completion (with its read lease) precedes
    // the load's, and both DONE pulses carry the tags allocated at their issues.
    assert(out.doneLog.size == 2 && out.enqLog.size == 2,
      s"expected 2 enq/2 done, saw ${out.enqLog.size}/${out.doneLog.size}")
    assert(out.enqLog(0).en && out.enqLog(1).en, "both DMAST/DMALD must declare leases")
    assert(out.doneLog(0).id == out.enqLog(0).id,
      s"store completion echoed ${out.doneLog(0)} not its own tag ${out.enqLog(0)}")
    assert(out.doneLog(1).id == out.enqLog(1).id,
      s"load completion echoed ${out.doneLog(1)} not its own tag ${out.enqLog(1)}")
    assert(out.doneLog(0).cyc < out.doneLog(1).cyc, "completions out of order")
  }

  it should "complete a zero-row DMALD at its pop without disturbing leases" in {
    val aDram = 0x60000; val aDtcm = 0x400
    val dram = rowPattern(aDram, 0xF6)
    val program =
      li(10, aDram) ++ li(11, aDtcm) ++ li(23, 0x1000) ++ Seq(
        Encode.dmald(10, 11, 8),
        Encode.dmald(10, 11, 0),      // zero rows: completes at pop, no lease
        Encode.lw(20, 11, 0),
        Encode.sw(20, 23, 0),
        Encode.halt,
      )
    val out = runCore(program, dramInit = dram)
    assert(out.dtcm(BigInt(0x1000)) == BigInt(0x00F60000))
    assert(out.enqLog.size == 2 && out.doneLog.size == 2,
      s"expected 2 enq/2 done, saw ${out.enqLog.size}/${out.doneLog.size}")
    assert(out.enqLog(1).en == false, "zero-row DMALD must not declare a lease")
    assert(out.doneLog(1).en == false, "zero-row completion must echo no lease")
    assert(out.doneLog(0).cyc < out.doneLog(1).cyc, "zero-row entry did not complete in order")
  }

  it should "fill the 2-bit lease counter to three in-flight descriptors and drain at HALT" in {
    val rows = 4
    // Three disjoint regions; no DMAWAIT between loads. The depth-2 FIFO plus
    // the executing entry let all three stage, so the in-flight count reaches 3
    // before the first completes. A wrap (1-bit counter) or a miscount would
    // release HALT early and the readers below would see stale rows.
    val r0 = (0x70000, 0x400); val r1 = (0x71000, 0x500); val r2 = (0x72000, 0x600)
    val dram = rowPattern(BigInt(r0._1), 0x11) ++
               rowPattern(BigInt(r1._1), 0x22) ++
               rowPattern(BigInt(r2._1), 0x33)
    val program =
      li(10, r0._1) ++ li(11, r0._2) ++ li(12, r1._1) ++ li(13, r1._2) ++
      li(14, r2._1) ++ li(15, r2._2) ++ li(23, 0x1000) ++ Seq(
        Encode.dmald(10, 11, rows),
        Encode.dmald(12, 13, rows),
        Encode.dmald(14, 15, rows),
        Encode.lw(20, 11, 0), Encode.lw(21, 13, 0), Encode.lw(22, 15, 0),
        Encode.sw(20, 23, 0), Encode.sw(21, 23, 4), Encode.sw(22, 23, 8),
        Encode.halt,
      )
    val out = runCore(program, dramInit = dram)
    assert(out.enqLog.size == 3, s"expected 3 DMA issues, saw ${out.enqLog.size}")
    assert(out.doneLog.size == 3, s"expected 3 completions, saw ${out.doneLog.size}")
    // All three staged before the first completion: three leases outstanding.
    assert(out.enqLog(2).cyc < out.doneLog(0).cyc,
      s"third DMALD did not stage before the first completion: " +
      s"enq=${out.enqLog.map(_.cyc)} done=${out.doneLog.map(_.cyc)}")
    // Readers see each region's row 0 (HALT truly drained all three).
    assert(out.dtcm(BigInt(0x1000)) == BigInt(0x00110000))
    assert(out.dtcm(BigInt(0x1004)) == BigInt(0x00220000))
    assert(out.dtcm(BigInt(0x1008)) == BigInt(0x00330000))
  }
}
