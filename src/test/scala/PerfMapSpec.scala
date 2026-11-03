// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util.DecoupledIO
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

// CTRL register-map golden test. The 2026-10-30 readback bug class: the PERF
// read mux was wired for a wider Sel table than the address decode covered
// (addr[6:2], 5 bits), so every tap at 0x84-0x98 silently aliased onto
// STATUS/unmapped words -- a full sim round of stall counters was read back
// from the wrong registers and nobody noticed, because the aliased values
// looked plausible. This spec pins the ADDRESSING, not the semantics.
// The DMA descriptor words (0x10-0x1C) are write-only by design (absent from
// the read mux), so discrimination comes from the naturally-nonzero anchors
// of a trivial kick instead of sentinel writes:
//   1. every unmapped word offset must read 0 (a new register forces a map
//      update here; a widened decode leaks nonzero into the old holes);
//   2. the nonzero anchors (status, kicks, itcmFetch, cycles, bubEmpty)
//      appear exactly at their own offsets: value 1 ONLY at 0x00/0x40, the
//      itcmFetch/cycles/bubEmpty values nowhere else.  Under the old 5-bit
//      bug this trips three ways: 0x84 echoes status (1 != 0), 0x98 echoes
//      dmaLen (0 or stale, != >= 6), 0xA0 echoes dmaStatus (0 != > 0);
//   3. structurally-forced zeros: no unit ran, so every unit/stall counter
//      is exactly 0.
//
// Runs on the DmaTb AXI harness (same as Qwen3LayerRtlSpec) with a tiny DRAM.
class PerfMapSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val p = Parameters()
  val rowBytes = p.DtcmRowBytes
  private val ITCM = BigInt(0) << 19
  private val DTCM = BigInt(1) << 19
  private val CTRL = BigInt(2) << 19

  // The full CtrlSel table: offset -> register. Kept EXPLICIT so a new Sel
  // constant that is missing here (or an offset collision) fails loudly.
  private val mapped: Seq[(Int, String)] = Seq(
    0x00 -> "ctrl", 0x04 -> "status",
    0x10 -> "dmaDram", 0x14 -> "dmaDtcm", 0x18 -> "dmaLen", 0x1C -> "dmaCtrl",
    0x20 -> "dmaStatus",
    0x24 -> "matrix", 0x28 -> "vector", 0x2C -> "dmaBusy", 0x30 -> "dtcmM",
    0x34 -> "dtcmA", 0x38 -> "depwait", 0x3C -> "dmawait", 0x40 -> "kicks",
    0x44 -> "cycles", 0x48 -> "dmaDtcm", 0x4C -> "depM", 0x50 -> "depV",
    0x54 -> "memwait", 0x58 -> "memclear", 0x5C -> "bothidle", 0x60 -> "depvSolo",
    0x64 -> "eng0", 0x68 -> "eng1", 0x6C -> "engBoth",
    0x70 -> "vecUnit", 0x74 -> "vecQueue", 0x78 -> "matEng",
    0x7C -> "soloNoWork", 0x80 -> "soloBlocked",
    0x84 -> "scStallVec", 0x88 -> "scStallMat",
    0x8C -> "stallGpr", 0x90 -> "stallBlk", 0x94 -> "stallHalt",
    0x98 -> "itcmFetch", 0x9C -> "dmaIssue", 0xA0 -> "bubEmpty",
    0xA4 -> "bubHalting", 0xA8 -> "dmaStarve")

  behavior of "Npu CTRL register map"

  it should "read every CTRL word from its own address (no aliasing, no holes leaking)" in {
    // 5 register-only instructions + HALT: no unit ops, no DMA, no branches,
    // and distinct rd with rs1=x0 so not even a GPR-hazard stall exists --
    // everything except the pipeline itself is structurally zero.
    val prog = (1 to 5).map(k => Encode.addi(k, 0, k)) :+ Encode.halt

    test(new DmaTb(p, dramRows = 16)).withAnnotations(backendAnnos) { dut =>
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

      writeBurst(ITCM, prog)
      writeBurst(CTRL + 0x00, Seq(BigInt(1)))   // kick

      var guard = 0
      while ((readWord(CTRL + 0x04) & 1) == 0 && guard < 10000) guard += 1
      assert(guard < 10000, "trivial kick did not finish")

      val reads = (0 until 64).map(w => (w * 4, readWord(CTRL + w * 4)))
      def at(off: Int): BigInt = reads(off / 4)._2
      // (1) unmapped offsets read 0
      val holes = (0 until 64).map(_ * 4).filterNot(o => mapped.exists(_._1 == o))
      holes.foreach { off =>
        assert(at(off) == BigInt(0),
          s"unmapped CTRL offset 0x${Integer.toHexString(off)} reads nonzero ${at(off)}")
      }
      // (2) the nonzero anchors.  Value-uniqueness across the window is NOT
      // asserted: 0x00 is the write-only kick/clear word (reads 0), and
      // bothidle legitimately equals cycles for a no-unit program.  Aliasing
      // detection comes from the anchors themselves -- under the old 5-bit
      // decode: 0x84 reads STATUS (1 != forced 0), 0x98 reads dmaLen
      // (0 != >= 6), 0xA0 reads dmaStatus (0 != > 0).
      Seq(0x44 -> "cycles", 0x98 -> "itcmFetch", 0xA0 -> "bubEmpty").foreach { case (off, what) =>
        val v = at(off)
        assert(v > BigInt(0), s"$what (0x${Integer.toHexString(off)}) = $v, must be nonzero")
      }
      // (3) structurally-forced values: done, one kick, units/stalls all zero,
      // every retired instruction counted.
      def expect(off: Int, v: BigInt, what: String): Unit =
        assert(at(off) == v, s"$what (0x${Integer.toHexString(off)}) = ${at(off)}, expected $v")
      expect(0x04, BigInt(1), "status.done")
      expect(0x40, BigInt(1), "kicks")
      expect(0x24, BigInt(0), "matrixBusy")
      expect(0x28, BigInt(0), "vectorBusy")
      expect(0x2C, BigInt(0), "dmaBusy")
      expect(0x38, BigInt(0), "depwait")
      expect(0x4C, BigInt(0), "depM")
      expect(0x50, BigInt(0), "depV")
      expect(0x84, BigInt(0), "scStallVec")
      expect(0x88, BigInt(0), "scStallMat")
      expect(0x8C, BigInt(0), "stallGpr")
      expect(0x94, BigInt(0), "stallHalt")
      expect(0x9C, BigInt(0), "dmaIssue")
      expect(0xA8, BigInt(0), "dmaStarve")
      // bubHalting is the post-HALT drain (front end stopped, HALT still
      // walking to WB while state=sRun): every kick has a few of these -- the
      // real-shape replay also reads 3.  Bound it, don't force 0.
      assert(at(0xA4) > BigInt(0) && at(0xA4) < BigInt(16),
        s"bubHalting (0xa4) = ${at(0xA4)}, expected a small post-HALT drain")
      assert(reads(0x98 / 4)._2 >= BigInt(6),
        s"itcmFetch = ${reads(0x98 / 4)._2}, expected >= 6 retired instructions")
      assert(reads(0x44 / 4)._2 > BigInt(0), "cycles = 0")
      assert(reads(0xA0 / 4)._2 > BigInt(0), "bubEmpty = 0: pipeline-fill bubbles must exist")
    }
  }
}
