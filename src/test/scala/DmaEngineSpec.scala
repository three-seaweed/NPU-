// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable

// Focused unit test for the multi-outstanding DmaEngine load path (Phase 2). Drives
// the engine's M_AXI port with a lightweight behavioural read slave that accepts
// several ARs before returning any R data, so the "keep DmaMaxOutstanding bursts in
// flight" behaviour is actually exercised -- and it does so WITHOUT instantiating the
// systolic array, so it runs fast on Treadle (unlike the full-Npu DmaSpec).
class DmaEngineSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val p = Parameters()
  val rowBytes = p.DtcmRowBytes            // 16
  val mask128  = (BigInt(1) << p.DtcmRowBits) - 1

  it should "load DRAM->DTCM correctly while keeping multiple bursts in flight" in {
    test(new DmaEngine(p, addrBits = 32, idBits = 4,
                       tokBits = log2Ceil(p.TokenSlots))).withAnnotations(backendAnnos) { dut =>
      // 600 rows from DRAM base 0 => bursts split at 4KB: 256 + 256 + 88 = 3 ARs,
      // which (with DmaMaxOutstanding=4) should all be outstanding at once.
      val numRows  = 600
      val dramBase = 0
      val dtcmBase = 0x2000

      // DRAM model: byte-addr (16-aligned) -> 128-bit beat.
      val dram = mutable.Map[BigInt, BigInt]()
      for (i <- 0 until numRows)
        dram(BigInt(dramBase + i * rowBytes)) = (BigInt(i) * 0x1111 + 7) & mask128
      // DTCM sink captured from the engine's writes.
      val dtcm = mutable.Map[BigInt, BigInt]()

      // Behavioural AXI read slave: accept every AR immediately into a queue, then
      // serve R beats from the head burst with a deliberate 2-cycle gap so ARs pile
      // up ahead of the data (exercising multiple outstanding).
      val pending = mutable.Queue[(BigInt, Int)]()   // (start addr, beats)
      var servAddr = BigInt(0); var servLeft = 0; var gap = 0
      var maxOutstanding = 0

      dut.io.maxi.ar.ready.poke(true.B)              // always accept AR
      dut.io.maxi.r.valid.poke(false.B)
      dut.io.maxi.aw.ready.poke(false.B)
      dut.io.maxi.w.ready.poke(false.B)
      dut.io.maxi.b.valid.poke(false.B)
      dut.io.dtcm.rdata.poke(0.U)
      var bp = 0                                     // DTCM back-pressure counter

      // Descriptor: enqueue handshake (ready = FIFO room, not engine idleness).
      dut.io.enq.valid.poke(true.B)
      dut.io.enq.bits.dir.poke(false.B)
      dut.io.enq.bits.dramAddr.poke(dramBase.U)
      dut.io.enq.bits.dtcmAddr.poke(dtcmBase.U)
      dut.io.enq.bits.numRows.poke(numRows.U)
      dut.io.enq.bits.tok.en.poke(false.B)
      dut.io.enq.bits.tok.id.poke(0.U)
      assert(dut.io.enq.ready.peek().litToBoolean, "empty FIFO must accept a descriptor")
      dut.clock.step(1)
      dut.io.enq.valid.poke(false.B)

      var guard = 0
      var running = true
      while (running) {
        // DTCM back-pressures every 3rd cycle (models the core winning the shared port).
        val dtcmRdy = (bp % 3) != 0
        dut.io.dtcm.ready.poke(dtcmRdy.B)
        bp += 1
        // --- capture an accepted AR this cycle (ar.ready held true) ---
        if (dut.io.maxi.ar.valid.peek().litToBoolean) {
          val a = dut.io.maxi.ar.bits.addr.peek().litValue
          val len = dut.io.maxi.ar.bits.len.peek().litValue.toInt + 1
          pending.enqueue((a, len))
        }
        // --- serve R: drive the head burst's next beat (after a small gap) ---
        val outstanding = pending.size + (if (servLeft > 0) 1 else 0)
        if (outstanding > maxOutstanding) maxOutstanding = outstanding
        if (servLeft == 0 && pending.nonEmpty && gap == 0) {
          val (a, n) = pending.dequeue(); servAddr = a; servLeft = n
        }
        val driveR = servLeft > 0 && gap == 0
        dut.io.maxi.r.valid.poke(driveR.B)
        if (driveR) {
          dut.io.maxi.r.bits.data.poke(dram(servAddr).U)
          dut.io.maxi.r.bits.last.poke((servLeft == 1).B)
        }
        // --- capture a DTCM write the engine issues this cycle ---
        val dtcmFire = dut.io.dtcm.valid.peek().litToBoolean && dut.io.dtcm.write.peek().litToBoolean && dtcmRdy
        val dtcmAddr = dut.io.dtcm.addr.peek().litValue
        val dtcmData = dut.io.dtcm.wdata.peek().litValue
        val rFire = driveR && dut.io.maxi.r.ready.peek().litToBoolean
        // `done` pulses the same cycle the final beat lands, so sample it before the
        // step (the last DTCM write is captured this same cycle via dtcmFire).
        val doneNow = dut.io.done.peek().litToBoolean

        dut.clock.step(1)

        if (dtcmFire) dtcm(dtcmAddr) = dtcmData
        if (rFire) { servAddr += rowBytes; servLeft -= 1; gap = 2 }
        else if (gap > 0) gap -= 1

        if (doneNow) running = false
        guard += 1
        assert(guard < 20000, "DMA load did not finish")
      }

      // Every row must have landed in DTCM with the right data.
      for (i <- 0 until numRows) {
        val got = dtcm.getOrElse(BigInt(dtcmBase + i * rowBytes), BigInt(-1))
        assert(got == dram(BigInt(dramBase + i * rowBytes)),
          s"row $i: got $got expected ${dram(BigInt(dramBase + i * rowBytes))}")
      }
      assert(maxOutstanding >= 2,
        s"expected multiple outstanding bursts, saw max $maxOutstanding")
    }
  }

  // Contract of the depth-2 descriptor FIFO that the core-issued DMA path relies
  // on: a following descriptor is ACCEPTED while the previous transfer is still
  // bursting (enq.ready = FIFO room, not engine idleness), descriptors complete
  // strictly in order, and each completion echoes its OWN lease tag -- including
  // a zero-row entry, which completes at its pop without touching DTCM.
  it should "stage descriptors behind the executing one and echo each lease tag in order" in {
    test(new DmaEngine(p, addrBits = 32, idBits = 4,
                       tokBits = log2Ceil(p.TokenSlots))).withAnnotations(backendAnnos) { dut =>
      val mask128 = (BigInt(1) << p.DtcmRowBits) - 1
      def rowVal(base: Int, i: Int): BigInt = (BigInt(base + i) * 0x010101 + 3) & mask128
      val aDram = 0x00000; val aDtcm = 0x0000; val aRows = 8
      val cDram = 0x20000; val cDtcm = 0x0800; val cRows = 8
      val dram = mutable.Map[BigInt, BigInt]()
      for (i <- 0 until aRows) dram(BigInt(aDram + i * rowBytes)) = rowVal(0x100, i)
      for (i <- 0 until cRows) dram(BigInt(cDram + i * rowBytes)) = rowVal(0x900, i)
      val dtcm = mutable.Map[BigInt, BigInt]()

      // AXI read slave: accept every AR immediately, serve R beats back-to-back.
      val pending = mutable.Queue[(BigInt, Int)]()
      var servAddr = BigInt(0); var servLeft = 0
      dut.io.maxi.ar.ready.poke(true.B)
      dut.io.maxi.aw.ready.poke(false.B)
      dut.io.maxi.w.ready.poke(false.B)
      dut.io.maxi.b.valid.poke(false.B)
      dut.io.dtcm.ready.poke(true.B)
      dut.io.dtcm.rdata.poke(0.U)

      case class Desc(dramA: Int, dtcmA: Int, rows: Int, tokEn: Boolean, tokId: Int)
      // A (8-row load, lease 5) -> B (zero-row, no lease) -> C (8-row load,
      // lease 7). All three must stage: A pops into execution immediately, so
      // the 2-deep FIFO has room for B and C while A still bursts.
      val toEnq = mutable.Queue(
        Desc(aDram, aDtcm, aRows, tokEn = true, tokId = 5),
        Desc(0, 0, 0, tokEn = false, tokId = 0),
        Desc(cDram, cDtcm, cRows, tokEn = true, tokId = 7))
      val enqCycle = mutable.ArrayBuffer[Int]()
      val doneInfo = mutable.ArrayBuffer[(Int, Boolean, Int)]()  // (cycle, en, id)

      var cyc = 0
      var guard = 0
      while (doneInfo.size < 3) {
        val has = toEnq.nonEmpty
        dut.io.enq.valid.poke(has.B)
        if (has) {
          val d = toEnq.head
          dut.io.enq.bits.dir.poke(false.B)
          dut.io.enq.bits.dramAddr.poke(d.dramA.U)
          dut.io.enq.bits.dtcmAddr.poke(d.dtcmA.U)
          dut.io.enq.bits.numRows.poke(d.rows.U)
          dut.io.enq.bits.tok.en.poke(d.tokEn.B)
          dut.io.enq.bits.tok.id.poke(d.tokId.U)
          if (dut.io.enq.ready.peek().litToBoolean) { toEnq.dequeue(); enqCycle += cyc }
        }
        if (dut.io.maxi.ar.valid.peek().litToBoolean) {
          pending.enqueue((dut.io.maxi.ar.bits.addr.peek().litValue,
                           dut.io.maxi.ar.bits.len.peek().litValue.toInt + 1))
        }
        if (servLeft == 0 && pending.nonEmpty) {
          val (a, n) = pending.dequeue(); servAddr = a; servLeft = n
        }
        val driveR = servLeft > 0
        dut.io.maxi.r.valid.poke(driveR.B)
        if (driveR) {
          dut.io.maxi.r.bits.data.poke(dram(servAddr).U)
          dut.io.maxi.r.bits.last.poke((servLeft == 1).B)
        }
        val doneNow  = dut.io.done.peek().litToBoolean
        val rFire    = driveR && dut.io.maxi.r.ready.peek().litToBoolean
        val dtcmFire = dut.io.dtcm.valid.peek().litToBoolean && dut.io.dtcm.write.peek().litToBoolean
        val dtcmAddr = dut.io.dtcm.addr.peek().litValue
        val dtcmData = dut.io.dtcm.wdata.peek().litValue
        if (doneNow)
          doneInfo += ((cyc, dut.io.doneTok.en.peek().litToBoolean,
                             dut.io.doneTok.id.peek().litValue.toInt))

        dut.clock.step(1)
        cyc += 1
        if (dtcmFire) dtcm(dtcmAddr) = dtcmData
        if (rFire) { servAddr += rowBytes; servLeft -= 1 }
        guard += 1
        assert(guard < 5000, "descriptor FIFO did not drain")
      }

      assert(enqCycle.size == 3, s"expected 3 accepted descriptors, saw ${enqCycle.size}")
      // Staging: B and C were accepted while A was still in flight.
      assert(enqCycle(1) < doneInfo(0)._1, "second descriptor did not stage behind the first")
      assert(enqCycle(2) < doneInfo(0)._1, "third descriptor did not stage behind the first")
      // Strict completion order, each completion carrying its own tag.
      assert(doneInfo(0)._1 < doneInfo(1)._1 && doneInfo(1)._1 < doneInfo(2)._1,
        s"completion order broken: $doneInfo")
      assert(doneInfo(0)._2 && doneInfo(0)._3 == 5, s"A echoed ${doneInfo(0)}")
      assert(!doneInfo(1)._2, s"zero-row entry echoed a lease: ${doneInfo(1)}")
      assert(doneInfo(2)._2 && doneInfo(2)._3 == 7, s"C echoed ${doneInfo(2)}")
      // Data landed despite the later descriptors staging behind it.
      for (i <- 0 until aRows) {
        val got = dtcm.getOrElse(BigInt(aDtcm + i * rowBytes), BigInt(-1))
        assert(got == dram(BigInt(aDram + i * rowBytes)), s"A row $i: got $got")
      }
      for (i <- 0 until cRows) {
        val got = dtcm.getOrElse(BigInt(cDtcm + i * rowBytes), BigInt(-1))
        assert(got == dram(BigInt(cDram + i * rowBytes)), s"C row $i: got $got")
      }
    }
  }
}
