// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable

// V1 regression: scalar LW/SW dispatch on DTCM row tokens (no full drain).
// Alu-level, DtcmTokenSpec's harness idiom plus LayerSpec's aux-port DTCM
// model (pendingRead one cycle later) so an LW's value is checkable.
//
// The contract under test (Alu scalar-memory path):
//   * LW waits until every EARLIER token writing its row retired (RAW);
//   * SW waits until every EARLIER token reading OR writing its row retired
//     (WAR+WAW);
//   * non-overlapping rows never wait even while the conflicting command's
//     unit is mid-execution (the old gate waited for BOTH units to drain);
//   * the check is a superset of the old semantics w.r.t. DMA: a live DMA
//     read token now holds a same-row SW (pre-V1 it never did).
class MemRowClearSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  private val p = Parameters()
  private val rowBytes = p.DtcmRowBytes
  private val wordsPerRow = p.WordsPerRow

  behavior of "Alu scalar-mem row-token dispatch (V1)"

  it should "hold an LW behind a live Vector writer of its row (RAW), then return the stored word" in {
    test(new Alu(p)).withAnnotations(backendAnnos) { dut =>
      val mem = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
      dut.io.req.valid.poke(false.B)
      dut.io.req.bits.op.poke(AluOp.ADDI)
      dut.io.req.bits.addr.poke(0.U)
      dut.io.req.bits.imm.poke(0.U)
      dut.io.rs1.valid.poke(true.B); dut.io.rs2.valid.poke(true.B)
      dut.io.rs1.data.poke(0.U); dut.io.rs2.data.poke(0.U)
      dut.io.rd.ready.poke(true.B); dut.io.vectorWrite.ready.poke(true.B)
      dut.io.dtcmMatrix.ready.poke(true.B); dut.io.dtcmMatrix.rdata.poke(0.U)
      dut.io.dmaIdle.poke(true.B)

      var pendingRead: Option[BigInt] = None
      def packWide(a: BigInt): BigInt =
        (0 until wordsPerRow).foldLeft(BigInt(0))((acc, i) => acc | (mem(a + 4 * i) << (32 * i)))
      def writeWide(a: BigInt, data: BigInt, strb: BigInt): Unit =
        for (i <- 0 until wordsPerRow) {
          var word = mem(a + 4 * i)
          for (b <- 0 until 4) if (((strb >> (i * 4 + b)) & 1) == 1) {
            val nb = (data >> (32 * i + 8 * b)) & 0xff
            word = (word & ~(BigInt(0xff) << (8 * b))) | (nb << (8 * b))
          }
          mem(a + 4 * i) = word
        }
      var lastRd: Option[BigInt] = None
      var vectorStoreRow: Option[BigInt] = None
      var storeDone = false
      var lwReadBeforeStore = false
      var lwResult: Option[BigInt] = None
      def tick(): Unit = {
        dut.io.dtcmAux.ready.poke(true.B)
        dut.io.dtcmAux.rdata.poke(pendingRead.map(packWide).getOrElse(BigInt(0)).U)
        val v = dut.io.dtcmAux.valid.peek().litToBoolean
        val w = dut.io.dtcmAux.write.peek().litToBoolean
        val a = dut.io.dtcmAux.addr.peek().litValue
        if (v) {
          if (w) writeWide(a, dut.io.dtcmAux.wdata.peek().litValue, dut.io.dtcmAux.wstrb.peek().litValue)
          pendingRead = if (!w) Some(a) else None
        } else pendingRead = None
        // Sample rd AFTER the rdata poke (sCap's bits.data is combinational).
        if (dut.io.rd.valid.peek().litToBoolean) {
          lastRd = Some(dut.io.rd.bits.data.peek().litValue)
        }
        // Track the VADD's store beat and any premature read of its row: all
        // post-poke, post-model-update (writeWide above) so captures reflect
        // the cycle's committed state.
        if (v && w && a == 0x000) {
          vectorStoreRow = Some(packWide(0x000))
          storeDone = true
        }
        if (v && !w && a == 0x000 && !storeDone) lwReadBeforeStore = true
        dut.clock.step(1)
      }

      def issue(op: AluOp.Type, rs1: Int = 0, rs2: Int = 0): Unit = {
        dut.io.rs1.data.poke(rs1.U)
        dut.io.rs2.data.poke(rs2.U)
        dut.io.req.bits.op.poke(op)
        dut.io.req.valid.poke(true.B)
        var wait = 0
        while (!dut.io.req.ready.peek().litToBoolean) {
          tick(); wait += 1; assert(wait < 200, s"$op could not enqueue")
        }
        tick()
        dut.io.req.valid.poke(false.B)
      }

      // VADD dst row 0x000 <- A row 0x100 + B row 0x200 (1 row of bf16 lanes).
      // Preload nonzero source words so the stored row (and thus the LW value)
      // is meaningful; the expectation is derived from whatever the VADD
      // actually stores, captured by the model below.
      for (i <- 0 until wordsPerRow) {
        mem(0x100 + 4 * i) = BigInt(0x3f803f80L) + BigInt(i)
        mem(0x200 + 4 * i) = BigInt(0x40004000L) + BigInt(2 * i)
      }
      issue(AluOp.VSETA, 0x100)
      issue(AluOp.VSETB, 0x200)
      issue(AluOp.VSETD, 0x000)
      issue(AluOp.VSETLEN, p.Bf16LanesPerRow)
      issue(AluOp.VADD)

      // Present LW word 2 of row 0x000 while the VADD token is live. All
      // ordering observation lives inside tick() (store-beat capture and the
      // premature-read check); this loop only waits for the dispatch gate.
      var memReadySeen = false
      dut.io.rs1.data.poke(0x008.U)
      dut.io.req.bits.op.poke(AluOp.LW)
      dut.io.req.valid.poke(true.B)
      var wait = 0
      while (!dut.io.req.ready.peek().litToBoolean) {
        if (!dut.io.memReady.peek().litToBoolean) memReadySeen = true
        tick(); wait += 1
        assert(wait < 300, "LW could not dispatch behind the VADD token")
      }
      // req fired: the access runs in MemUnit (sRead/sCap); capture the result.
      var cap = 0
      while (lwResult.isEmpty && cap < 40) {
        tick(); cap += 1
        lwResult = lastRd
      }
      dut.io.req.valid.poke(false.B)

      assert(memReadySeen, "memReady was never held low while the VADD write token lived")
      assert(storeDone && vectorStoreRow.isDefined, "the VADD never stored its dst row")
      assert(!lwReadBeforeStore, "the LW read row 0x000 before the VADD store retired")
      val expect = (vectorStoreRow.get >> 64) & 0xffffffffL   // word 2 of the row
      assert(lwResult.contains(expect),
        s"LW returned ${lwResult.map(_.toString(16))} instead of stored word 0x${expect.toString(16)}")
    }
  }

  it should "hold an SW behind a live Vector reader of its row (WAR)" in {
    test(new Alu(p)).withAnnotations(backendAnnos) { dut =>
      val mem = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
      dut.io.req.valid.poke(false.B)
      dut.io.req.bits.op.poke(AluOp.ADDI)
      dut.io.req.bits.addr.poke(0.U)
      dut.io.req.bits.imm.poke(0.U)
      dut.io.rs1.valid.poke(true.B); dut.io.rs2.valid.poke(true.B)
      dut.io.rs1.data.poke(0.U); dut.io.rs2.data.poke(0.U)
      dut.io.rd.ready.poke(true.B); dut.io.vectorWrite.ready.poke(true.B)
      dut.io.dtcmMatrix.ready.poke(true.B); dut.io.dtcmMatrix.rdata.poke(0.U)
      dut.io.dmaIdle.poke(true.B)

      var pendingRead: Option[BigInt] = None
      def packWide(a: BigInt): BigInt =
        (0 until wordsPerRow).foldLeft(BigInt(0))((acc, i) => acc | (mem(a + 4 * i) << (32 * i)))
      def writeWide(a: BigInt, data: BigInt, strb: BigInt): Unit =
        for (i <- 0 until wordsPerRow) {
          var word = mem(a + 4 * i)
          for (b <- 0 until 4) if (((strb >> (i * 4 + b)) & 1) == 1) {
            val nb = (data >> (32 * i + 8 * b)) & 0xff
            word = (word & ~(BigInt(0xff) << (8 * b))) | (nb << (8 * b))
          }
          mem(a + 4 * i) = word
        }
      def tick(): Unit = {
        dut.io.dtcmAux.ready.poke(true.B)
        dut.io.dtcmAux.rdata.poke(pendingRead.map(packWide).getOrElse(BigInt(0)).U)
        val v = dut.io.dtcmAux.valid.peek().litToBoolean
        val w = dut.io.dtcmAux.write.peek().litToBoolean
        val a = dut.io.dtcmAux.addr.peek().litValue
        if (v) {
          if (w) writeWide(a, dut.io.dtcmAux.wdata.peek().litValue, dut.io.dtcmAux.wstrb.peek().litValue)
          pendingRead = if (!w) Some(a) else None
        } else pendingRead = None
        dut.clock.step(1)
      }

      def issue(op: AluOp.Type, rs1: Int = 0, rs2: Int = 0): Unit = {
        dut.io.rs1.data.poke(rs1.U)
        dut.io.rs2.data.poke(rs2.U)
        dut.io.req.bits.op.poke(op)
        dut.io.req.valid.poke(true.B)
        var wait = 0
        while (!dut.io.req.ready.peek().litToBoolean) {
          tick(); wait += 1; assert(wait < 200, s"$op could not enqueue")
        }
        tick()
        dut.io.req.valid.poke(false.B)
      }

      // VABS reads srcA row 0x300, writes a DISJOINT dst row 0x380.
      issue(AluOp.VSETA, 0x300)
      issue(AluOp.VSETD, 0x380)
      issue(AluOp.VSETLEN, p.Bf16LanesPerRow)
      issue(AluOp.VABS)

      // Present SW of a word in row 0x300 (WAR on the VABS read token).
      var memReadySeen = false
      var lastFetchIdx = -1L
      var swWriteIdx = -1L
      var now = 0L
      dut.io.rs1.data.poke(0x304.U)
      dut.io.rs2.data.poke(BigInt("deadbeef", 16).U)
      dut.io.req.bits.op.poke(AluOp.SW)
      dut.io.req.valid.poke(true.B)
      var wait = 0
      while (!(dut.io.req.ready.peek().litToBoolean)) {
        if (!dut.io.memReady.peek().litToBoolean) memReadySeen = true
        val v = dut.io.dtcmAux.valid.peek().litToBoolean
        val w = dut.io.dtcmAux.write.peek().litToBoolean
        val a = dut.io.dtcmAux.addr.peek().litValue
        if (v && !w && a == 0x300) lastFetchIdx = now
        if (v && w && a == 0x300 && swWriteIdx < 0) swWriteIdx = now
        tick(); now += 1; wait += 1
        assert(wait < 300, "SW could not dispatch behind the VABS read token")
      }
      // Let the store beat land and keep sampling through it.
      for (_ <- 0 until 12) {
        val v = dut.io.dtcmAux.valid.peek().litToBoolean
        val w = dut.io.dtcmAux.write.peek().litToBoolean
        val a = dut.io.dtcmAux.addr.peek().litValue
        if (v && w && a == 0x300 && swWriteIdx < 0) swWriteIdx = now
        tick(); now += 1
      }
      dut.io.req.valid.poke(false.B)

      assert(memReadySeen, "memReady was never held low while the VABS read token lived")
      assert(swWriteIdx > 0, "the SW never landed its write beat")
      assert(swWriteIdx > lastFetchIdx,
        s"WAR violated: SW beat at $swWriteIdx preceded the last 0x300 fetch at $lastFetchIdx")
      assert(mem(0x304) == (BigInt(0xdeadbeefL) & 0xffffffffL),
        s"SW did not land: mem(0x304)=${mem(0x304).toString(16)}")
    }
  }

  it should "not wait on rows the live Vector command does not touch (negative control)" in {
    test(new Alu(p)).withAnnotations(backendAnnos) { dut =>
      val mem = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
      mem(0x008) = BigInt("cafef00d", 16)
      dut.io.req.valid.poke(false.B)
      dut.io.req.bits.op.poke(AluOp.ADDI)
      dut.io.req.bits.addr.poke(0.U)
      dut.io.req.bits.imm.poke(0.U)
      dut.io.rs1.valid.poke(true.B); dut.io.rs2.valid.poke(true.B)
      dut.io.rs1.data.poke(0.U); dut.io.rs2.data.poke(0.U)
      dut.io.rd.ready.poke(true.B); dut.io.vectorWrite.ready.poke(true.B)
      dut.io.dtcmMatrix.ready.poke(true.B); dut.io.dtcmMatrix.rdata.poke(0.U)
      dut.io.dmaIdle.poke(true.B)

      var pendingRead: Option[BigInt] = None
      def packWide(a: BigInt): BigInt =
        (0 until wordsPerRow).foldLeft(BigInt(0))((acc, i) => acc | (mem(a + 4 * i) << (32 * i)))
      var lastRd: Option[BigInt] = None
      def tick(): Unit = {
        dut.io.dtcmAux.ready.poke(true.B)
        dut.io.dtcmAux.rdata.poke(pendingRead.map(packWide).getOrElse(BigInt(0)).U)
        val v = dut.io.dtcmAux.valid.peek().litToBoolean
        val w = dut.io.dtcmAux.write.peek().litToBoolean
        val a = dut.io.dtcmAux.addr.peek().litValue
        pendingRead = if (v && !w) Some(a) else None
        // Sample rd AFTER the rdata poke: bits.data in sCap is combinational
        // off io.dtcm.rdata, so a pre-poke peek reads the stale beat.
        if (dut.io.rd.valid.peek().litToBoolean) {
          lastRd = Some(dut.io.rd.bits.data.peek().litValue)
        }
        dut.clock.step(1)
      }

      def issue(op: AluOp.Type, rs1: Int = 0, rs2: Int = 0): Unit = {
        dut.io.rs1.data.poke(rs1.U)
        dut.io.rs2.data.poke(rs2.U)
        dut.io.req.bits.op.poke(op)
        dut.io.req.valid.poke(true.B)
        var wait = 0
        while (!dut.io.req.ready.peek().litToBoolean) {
          tick(); wait += 1; assert(wait < 200, s"$op could not enqueue")
        }
        tick()
        dut.io.req.valid.poke(false.B)
      }

      // VADD writes row 0x400 -- disjoint from the LW's row 0x000.
      issue(AluOp.VSETA, 0x500)
      issue(AluOp.VSETB, 0x600)
      issue(AluOp.VSETD, 0x400)
      issue(AluOp.VSETLEN, p.Bf16LanesPerRow)
      issue(AluOp.VADD)

      // The LW must be dispatchable IMMEDIATELY (the VADD is executing, its
      // write token live, but on a disjoint row). memReady is the dispatch
      // gate, checked on the very cycle of presentation; once dispatched the
      // unit legitimately goes busy (memReady drops) and the port beat may
      // queue behind the running VADD -- that coarse-arb serialization is a
      // known, separate property (V1b), not a false wait.
      dut.io.rs1.data.poke(0x008.U)
      dut.io.req.bits.op.poke(AluOp.LW)
      dut.io.req.valid.poke(true.B)
      assert(dut.io.memReady.peek().litToBoolean,
        s"dispatch gate blocked although no token touches the LW's row " +
        s"(memIdle=${dut.io.memIdle.peek().litToBoolean} -- full-drain residue?)")
      var wait = 0
      while (!dut.io.req.ready.peek().litToBoolean) {
        tick(); wait += 1; assert(wait < 200, "LW could not enqueue on a clear row")
      }
      tick()
      dut.io.req.valid.poke(false.B)
      var lwResult: Option[BigInt] = None
      var cap = 0
      while (lwResult.isEmpty && cap < 40) {
        tick(); cap += 1
        lwResult = lastRd
      }
      assert(lwResult.contains(BigInt("cafef00d", 16)),
        s"LW returned ${lwResult.map(_.toString(16))} instead of cafef00d")
    }
  }

  it should "hold an SW behind a live DMA read token of its row (superset of the old gate)" in {
    test(new Alu(p)).withAnnotations(backendAnnos) { dut =>
      dut.io.req.valid.poke(false.B)
      dut.io.req.bits.op.poke(AluOp.ADDI)
      dut.io.req.bits.addr.poke(0.U)
      dut.io.req.bits.imm.poke(0.U)
      dut.io.rs1.valid.poke(true.B); dut.io.rs2.valid.poke(true.B)
      dut.io.rs1.data.poke(0.U); dut.io.rs2.data.poke(0.U)
      dut.io.rd.ready.poke(true.B); dut.io.vectorWrite.ready.poke(true.B)
      dut.io.dtcmMatrix.ready.poke(true.B); dut.io.dtcmMatrix.rdata.poke(0.U)
      dut.io.dtcmAux.ready.poke(true.B); dut.io.dtcmAux.rdata.poke(0.U)
      dut.io.dmaIdle.poke(true.B)

      def tick(): Unit = dut.clock.step(1)

      // DMAST reads DTCM rows [0x700, 0x700+16): its token stays live until the
      // engine echoes the descriptor's lease on the completion pulse. Alu fires
      // the start pulse at enqueue; capture the slot it allocated and simply
      // never pulse dmaDone so the read lease persists.
      dut.io.rs1.data.poke(0.U)              // dram addr (unused at this level)
      dut.io.rs2.data.poke(0x700.U)          // dtcm base
      dut.io.req.bits.imm.poke(1.U)          // 1 row
      dut.io.req.bits.op.poke(AluOp.DMAST)
      dut.io.req.valid.poke(true.B)
      var wait = 0
      while (!dut.io.req.ready.peek().litToBoolean) {
        tick(); wait += 1; assert(wait < 100, "DMAST could not enqueue")
      }
      assert(dut.io.dmaTok.en.peek().litToBoolean, "DMAST start did not declare a token lease")
      val tokEn = dut.io.dmaTok.en.peek().litToBoolean
      val tokId = dut.io.dmaTok.id.peek().litValue.toInt
      tick()
      dut.io.req.valid.poke(false.B)
      for (_ <- 0 until 3) tick()

      // SW into row 0x700: must be held while the DMA read token lives.
      dut.io.rs1.data.poke(0x700.U)
      dut.io.rs2.data.poke(BigInt("12345678", 16).U)
      dut.io.req.bits.op.poke(AluOp.SW)
      dut.io.req.valid.poke(true.B)
      var held = false
      wait = 0
      while (wait < 40 && !dut.io.req.ready.peek().litToBoolean) {
        if (!dut.io.memReady.peek().litToBoolean) held = true
        tick(); wait += 1
      }
      assert(held && wait == 40,
        s"SW dispatched (or vanished) while the DMA read lease lived (held=$held wait=$wait)")
      // The completion echo retires the lease; the held SW then dispatches.
      dut.io.dmaDone.poke(true.B)
      dut.io.dmaDoneTok.en.poke(tokEn.B)
      dut.io.dmaDoneTok.id.poke(tokId.U)
      tick()
      dut.io.dmaDone.poke(false.B)
      wait = 0
      while (!dut.io.req.ready.peek().litToBoolean) {
        tick(); wait += 1; assert(wait < 20, "SW did not dispatch after the DMA lease retired")
      }
      tick()
      dut.io.req.valid.poke(false.B)
    }
  }
}
