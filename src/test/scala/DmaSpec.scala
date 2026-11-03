// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable.ArrayBuffer

// Test-only AXI4 memory model standing in for PS DRAM behind the NPU's M_AXI
// master. DtcmRowBits-wide (one beat = one matrix row), INCR bursts, single
// outstanding. Extra debug ports let the test preload operands and read back
// results without going through the bus.
class AxiRam(addrBits: Int, dataBits: Int, idBits: Int, depthRows: Int) extends Module {
    val io = IO(new Bundle {
        val axi = new Axi4SlaveIO(addrBits, dataBits, idBits)
        val dbgAddr = Input(UInt(addrBits.W))
        val dbgData = Output(UInt(dataBits.W))
        val preAddr = Input(UInt(addrBits.W))
        val preData = Input(UInt(dataBits.W))
        val preWen  = Input(Bool())
    })

    val rowBytes  = dataBits / 8
    val rowShift  = log2Up(rowBytes)
    val rowIdxW   = log2Up(depthRows)
    val mem = Mem(depthRows, UInt(dataBits.W))

    def rowOf(a: UInt): UInt = (a >> rowShift)(rowIdxW - 1, 0)

    val sIdle :: sWrite :: sResp :: sRead :: Nil = Enum(4)
    val state = RegInit(sIdle)
    val curAddr = Reg(UInt(addrBits.W))
    val len     = Reg(UInt(8.W))
    val id      = Reg(UInt(idBits.W))
    val beat    = Reg(UInt(8.W))

    // Defaults.
    io.axi.aw.ready := false.B
    io.axi.ar.ready := false.B
    io.axi.w.ready  := false.B
    io.axi.b.valid  := false.B
    io.axi.b.bits.id   := id
    io.axi.b.bits.resp := 0.U
    io.axi.r.valid := false.B
    io.axi.r.bits.id   := id
    io.axi.r.bits.resp := 0.U
    io.axi.r.bits.last := false.B
    io.axi.r.bits.data := mem.read(rowOf(curAddr))

    // Debug / preload (used only while idle in the test).
    when (io.preWen) { mem.write(rowOf(io.preAddr), io.preData) }
    io.dbgData := mem.read(rowOf(io.dbgAddr))

    switch (state) {
        is (sIdle) {
            io.axi.aw.ready := true.B
            io.axi.ar.ready := !io.axi.aw.valid
            when (io.axi.aw.fire) {
                curAddr := io.axi.aw.bits.addr; len := io.axi.aw.bits.len; id := io.axi.aw.bits.id
                state := sWrite
            } .elsewhen (io.axi.ar.fire) {
                curAddr := io.axi.ar.bits.addr; len := io.axi.ar.bits.len; id := io.axi.ar.bits.id
                beat := 0.U; state := sRead
            }
        }
        is (sWrite) {
            io.axi.w.ready := true.B
            when (io.axi.w.fire) {
                mem.write(rowOf(curAddr), io.axi.w.bits.data)
                curAddr := curAddr + rowBytes.U
                when (io.axi.w.bits.last) { state := sResp }
            }
        }
        is (sResp) {
            io.axi.b.valid := true.B
            when (io.axi.b.fire) { state := sIdle }
        }
        is (sRead) {
            io.axi.r.valid     := true.B
            io.axi.r.bits.last := beat === len
            when (io.axi.r.fire) {
                curAddr := curAddr + rowBytes.U
                beat    := beat + 1.U
                when (beat === len) { state := sIdle }
            }
        }
    }
}

// Harness: the NPU with its M_AXI master wired to the DRAM model, the slave bus
// and DRAM debug ports brought out for the test ("PS") to drive.
class DmaTb(p: Parameters, dramRows: Int) extends Module {
    val io = IO(new Bundle {
        val axi     = new Axi4SlaveIO(32, 32, 4)
        val dbgAddr = Input(UInt(32.W))
        val dbgData = Output(UInt(p.DtcmRowBits.W))
        val preAddr = Input(UInt(32.W))
        val preData = Input(UInt(p.DtcmRowBits.W))
        val preWen  = Input(Bool())
    })
    val dut = Module(new Npu(p))
    val ram = Module(new AxiRam(32, p.DtcmRowBits, 4, dramRows))

    io.axi     <> dut.io.axi
    ram.io.axi <> dut.io.maxi
    ram.io.dbgAddr := io.dbgAddr
    ram.io.preAddr := io.preAddr
    ram.io.preData := io.preData
    ram.io.preWen  := io.preWen
    io.dbgData     := ram.io.dbgData
}

class DmaSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val p = Parameters()
  val n = p.MatrixDim
  val mask32  = (BigInt(1) << 32) - 1
  val rowBytes = p.DtcmRowBytes

  val A = Array.tabulate(n, n)((i, j) => ((i + 2 * j + 1) % 9) - 4)
  val B = Array.tabulate(n, n)((i, j) => ((3 * i + j + 2) % 7) - 3)
  val C = Array.tabulate(n, n)((i, j) => (0 until n).map(k => A(i)(k) * B(k)(j)).sum)

  val beatsPerRow  = p.BeatsPerArrayRow
  val elemsPerBeat = p.DtcmRowBits / p.ElemBits   // int8 per 128-bit beat (=16)
  val matBytes     = n * p.ArrayRowBytes           // one int8 matrix footprint (256 B / 1024 B)
  val baseA = 0
  val baseB = matBytes
  val baseC = 2 * matBytes

  // DRAM byte addresses: operands (matBytes each) then the n*n int32 result.
  val dramA = 0
  val dramB = matBytes
  val dramC = 2 * matBytes

  val ITCM = BigInt(0) << 19
  val DTCM = BigInt(1) << 19
  val CTRL = BigInt(2) << 19

  // CTRL-region offsets.
  val OFF_CTRL    = 0x00
  val OFF_STATUS  = 0x04
  val OFF_DMA_DR  = 0x10
  val OFF_DMA_DT  = 0x14
  val OFF_DMA_LEN = 0x18
  val OFF_DMA_GO  = 0x1c
  val OFF_DMA_ST  = 0x20

  // Pack one int8 matrix into its DTCM/DRAM beats: n rows x beatsPerRow beats, each
  // a 128-bit word holding elemsPerBeat int8 (element e of the beat at byte e).
  def matBeats(m: Array[Array[Int]]): Seq[BigInt] =
    for { r <- 0 until n; b <- 0 until beatsPerRow } yield
      (0 until elemsPerBeat).foldLeft(BigInt(0)) { (acc, e) =>
        acc | (BigInt(m(r)(b * elemsPerBeat + e) & 0xff) << (8 * e))
      }

  behavior of "DmaEngine"

  it should "burst operands DRAM->DTCM, run, and burst results DTCM->DRAM" in {
    test(new DmaTb(p, dramRows = 512)).withAnnotations(backendAnnos) { dut =>
      val axi = dut.io.axi
      def step(): Unit = dut.clock.step(1)

      axi.aw.valid.poke(false.B); axi.w.valid.poke(false.B); axi.b.ready.poke(false.B)
      axi.ar.valid.poke(false.B); axi.r.ready.poke(false.B)
      dut.io.preWen.poke(false.B)

      def addrSend(ch: DecoupledIO[AxiAddr], addr: BigInt, beats: Int): Unit = {
        ch.bits.id.poke(0.U); ch.bits.addr.poke(addr.U)
        ch.bits.len.poke((beats - 1).U); ch.bits.size.poke(2.U); ch.bits.burst.poke(1.U)
        ch.valid.poke(true.B)
        while (!ch.ready.peek().litToBoolean) step()
        step()
        ch.valid.poke(false.B)
      }
      def writeBurst(addr: BigInt, data: Seq[BigInt]): Unit = {
        addrSend(axi.aw, addr, data.length)
        for ((d, i) <- data.zipWithIndex) {
          axi.w.bits.data.poke((d & mask32).U)
          axi.w.bits.strb.poke("b1111".U)
          axi.w.bits.last.poke((i == data.length - 1).B)
          axi.w.valid.poke(true.B)
          while (!axi.w.ready.peek().litToBoolean) step()
          step()
        }
        axi.w.valid.poke(false.B)
        axi.b.ready.poke(true.B)
        while (!axi.b.valid.peek().litToBoolean) step()
        step()
        axi.b.ready.poke(false.B)
      }
      def readBurst(addr: BigInt, beats: Int): Seq[BigInt] = {
        addrSend(axi.ar, addr, beats)
        axi.r.ready.poke(true.B)
        val buf = ArrayBuffer[BigInt]()
        while (buf.length < beats) {
          if (axi.r.valid.peek().litToBoolean) buf += axi.r.bits.data.peek().litValue
          step()
        }
        axi.r.ready.poke(false.B)
        buf.toSeq
      }
      def reg32(off: Int, v: BigInt): Unit = writeBurst(CTRL + off, Seq(v))

      // Preload DRAM with A and B (one DtcmRowBits-wide row each).
      def dramPreload(base: Int, words: Seq[BigInt]): Unit = {
        for ((w, r) <- words.zipWithIndex) {
          dut.io.preAddr.poke((base + r * rowBytes).U)
          dut.io.preData.poke(w.U)
          dut.io.preWen.poke(true.B)
          step()
        }
        dut.io.preWen.poke(false.B)
      }
      def dramRead(addr: Int): BigInt = {
        dut.io.dbgAddr.poke(addr.U)
        step()
        dut.io.dbgData.peek().litValue
      }

      // Trigger a DMA descriptor and spin on DMA_STATUS.done.
      def dma(dram: Int, dtcm: Int, lenBytes: Int, store: Boolean): Unit = {
        reg32(OFF_DMA_DR, BigInt(dram))
        reg32(OFF_DMA_DT, BigInt(dtcm))
        reg32(OFF_DMA_LEN, BigInt(lenBytes))
        reg32(OFF_DMA_GO, if (store) BigInt(3) else BigInt(1))   // bit0 go, bit1 dir
        var guard = 0
        while ((readBurst(CTRL + OFF_DMA_ST, 1).head & 1) == 0) {
          guard += 1; assert(guard < 5000, "DMA did not finish")
        }
      }

      dramPreload(dramA, matBeats(A))
      dramPreload(dramB, matBeats(B))

      // 1. DMA operands DRAM -> DTCM.
      dma(dramA, baseA, matBytes, store = false)
      dma(dramB, baseB, matBytes, store = false)

      // 2. Load the program and run. Base addresses may exceed addi's 12-bit signed
      // immediate (e.g. baseC = 2048 at n=32), so materialise them with lui+addi.
      def li(rd: Int, imm: Int): Seq[BigInt] =
        if (imm >= -2048 && imm < 2048) Seq(Encode.addi(rd, 0, imm))
        else {
          val hi = (imm + 0x800) >> 12
          Seq(Encode.lui(rd, hi & 0xfffff), Encode.addi(rd, rd, imm - (hi << 12)))
        }
      val program =
        li(10, baseA) ++ li(11, baseB) ++ li(12, baseC) ++ Seq(
          Encode.lma(10),
          Encode.lmb(11),
          Encode.matmul(),
          Encode.sm(12),
          Encode.halt,
        )
      writeBurst(ITCM, program)
      writeBurst(CTRL + OFF_CTRL, Seq(BigInt(1)))   // start
      var guard = 0
      while ((readBurst(CTRL + OFF_STATUS, 1).head & 1) == 0) {
        guard += 1; assert(guard < 2000, "NPU did not finish")
      }

      // 3. DMA the n*n int32 results DTCM -> DRAM.
      dma(dramC, baseC, n * n * 4, store = true)

      // 4. Verify the result landed in DRAM (4 int32 lanes per wide row).
      for (i <- 0 until n; j <- 0 until n) {
        val f    = i * n + j
        val row  = dramRead(dramC + (f / 4) * rowBytes)
        val lane = f % 4
        val got  = (row >> (32 * lane)) & mask32
        val exp  = BigInt(C(i)(j)) & mask32
        assert(got == exp, s"C($i)($j): got $got expected $exp")
      }
    }
  }
}
