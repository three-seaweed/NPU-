// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable

// Validates the *general looping GEMV kernel* that runq.c loads into the NPU to
// offload llama2's quantized matmul `W(d,n) int8 @ x(n) int8 -> partials(int32)`.
// One CTRL.start runs the whole matmul: the core streams tile-major weights from
// "resmem" (DRAM) via DMALD, does grouped int8 GEMM (MATMUL / MATMUL.ACC), and
// DMASTs per-(rowtile,group) int32 result tiles back to DRAM. The host then reads
// column 0 of each tile and applies the float group scales (done in runq.c, not
// here). This is the Scala reference for the C kernel emitter in runq.c.
//
// Driven against the full Npu + an AXI DRAM model (DmaTb/AxiRam, shared w/ DmaSpec)
// so the real DmaEngine + relaxed DTCM arbiter run end to end. Shapes are tiny --
// full-Npu Treadle is ~1 cycle/s.
class GemvKernelSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val p = Parameters()
  val nTile = p.MatrixDim               // systolic-array tile dimension (16 or 32)
  val rowBytes = p.DtcmRowBytes         // 16 (physical 128-bit beat)
  val beatsPerRow  = p.BeatsPerArrayRow // physical beats per matrix row (1, 2)
  val elemsPerBeat = p.DtcmRowBits / p.ElemBits  // int8 per beat (=16)
  val mask32 = (BigInt(1) << 32) - 1

  // Tile footprints derived from the array dimension.
  val tileBytes    = nTile * p.ArrayRowBytes      // int8 weight/x tile (256 B / 1024 B)
  val tileRows     = nTile * beatsPerRow          // beats to DMA one int8 tile
  val resTileBytes = nTile * nTile * 4            // int32 result tile (1024 B / 4096 B)
  val resTileRows  = resTileBytes / rowBytes      // beats to DMA one result tile
  val resRowBytes  = nTile * 4                     // bytes per int32 result row

  // ---- DTCM layout (byte offsets) the kernel + host agree on -----------------
  val DESC  = 0x0000     // 7-word descriptor (host writes via slave before kick)
  val XBASE = 0x0020     // padded-x tiles (kernel DMALDs them here from x_phys)
  val WBUF  = 0x4000     // one weight tile
  val CBUF  = WBUF + tileBytes   // SM target (nTile*nTile int32)

  // ===========================================================================
  // The kernel program (general, descriptor-driven). Uses running pointers so no
  // multiply is needed (RV32I has no MUL). Mirror this exactly in runq.c.
  //   x1=wp(weight phys)  x2=op(out phys)  x3=xsrc(x phys)
  //   x4=n_rowtiles x5=n_groups x6=ktiles_per_group x7=n_ktiles
  //   x10=ti x11=g x12=kt  x13=xtp(x tile dtcm) x14=WBUF x15=CBUF x16=XBASE
  //   x17=xdld(preload dst) x18=kc(preload counter)
  // Descriptor words @DESC: [0]weight_phys [1]x_phys [2]out_phys [3]n_rowtiles
  //   [4]n_groups [5]ktiles_per_group [6]n_ktiles
  // ===========================================================================
  def kernelProgram(): Seq[BigInt] = {
    val prog = mutable.ArrayBuffer[(Map[String, Int], Int) => BigInt]()
    val labels = mutable.Map[String, Int]()
    def emit(b: BigInt): Unit = prog += ((_, _) => b)
    def label(nm: String): Unit = labels(nm) = prog.length
    // branch/jump to a label, offset = (labelIdx - selfIdx)*4 bytes
    def blt(rs1: Int, rs2: Int, L: String): Unit = { val ins = prog.length; prog += ((lb, i) => Encode.blt(rs1, rs2, (lb(L) - i) * 4)); require(ins >= 0) }
    def bne(rs1: Int, rs2: Int, L: String): Unit = prog += ((lb, i) => Encode.bne(rs1, rs2, (lb(L) - i) * 4))
    def jal(rd: Int, L: String): Unit = prog += ((lb, i) => Encode.jal(rd, (lb(L) - i) * 4))
    def li(rd: Int, imm: Int): Unit =
      if (imm >= -2048 && imm < 2048) emit(Encode.addi(rd, 0, imm))
      else { val hi = (imm + 0x800) >> 12; emit(Encode.lui(rd, hi & 0xfffff)); emit(Encode.addi(rd, rd, imm - (hi << 12))) }

    // ---- load descriptor (DESC at DTCM 0, base x0, small imm) ----
    emit(Encode.lw(1, 0, DESC + 0))    // weight_phys
    emit(Encode.lw(3, 0, DESC + 4))    // x_phys
    emit(Encode.lw(2, 0, DESC + 8))    // out_phys
    emit(Encode.lw(4, 0, DESC + 12))   // n_rowtiles
    emit(Encode.lw(5, 0, DESC + 16))   // n_groups
    emit(Encode.lw(6, 0, DESC + 20))   // ktiles_per_group
    emit(Encode.lw(7, 0, DESC + 24))   // n_ktiles
    li(14, WBUF); li(15, CBUF); li(16, XBASE)
    li(19, tileBytes); li(20, resTileBytes)   // strides (may exceed addi's 12-bit imm)

    // ---- preload padded x: n_ktiles tiles from x_phys -> XBASE ----
    emit(Encode.addi(17, 16, 0))       // xdld = XBASE
    emit(Encode.addi(18, 0, 0))        // kc = 0
    label("preload")
    emit(Encode.dmald(3, 17, tileRows)) // DTCM[xdld] <- DRAM[xsrc], one x tile
    emit(Encode.dmawait())
    emit(Encode.add(3, 3, 19))         // xsrc += tileBytes
    emit(Encode.add(17, 17, 19))       // xdld += tileBytes
    emit(Encode.addi(18, 18, 1))       // kc++
    blt(18, 7, "preload")

    // ---- main: for ti, for g, for kt ----
    emit(Encode.addi(10, 0, 0))        // ti = 0
    label("loop_ti")
    emit(Encode.addi(13, 16, 0))       // xtp = XBASE (reset each rowtile)
    emit(Encode.addi(11, 0, 0))        // g = 0
    label("loop_g")
    emit(Encode.addi(12, 0, 0))        // kt = 0
    label("loop_kt")
    emit(Encode.dmald(1, 14, tileRows)) // WBUF <- weight tile
    emit(Encode.dmawait())
    emit(Encode.lma(14))               // A = weight tile
    emit(Encode.lmb(13))               // B = padded x tile
    bne(12, 0, "do_acc")               // kt!=0 -> accumulate
    emit(Encode.matmul())              // kt==0: clear + C = A x B
    jal(0, "after_mm")
    label("do_acc")
    emit(Encode.matmulAcc())           // C += A x B
    label("after_mm")
    emit(Encode.add(1, 1, 19))         // wp += tileBytes (tile-major: next ktile)
    emit(Encode.add(13, 13, 19))       // xtp += tileBytes
    emit(Encode.addi(12, 12, 1))       // kt++
    blt(12, 6, "loop_kt")
    emit(Encode.sm(15))                // CBUF <- nTile*nTile int32
    emit(Encode.dmast(2, 15, resTileRows)) // DRAM[op] <- CBUF (result tile)
    emit(Encode.add(2, 2, 20))         // op += resTileBytes
    emit(Encode.addi(11, 11, 1))       // g++
    blt(11, 5, "loop_g")
    emit(Encode.addi(10, 10, 1))       // ti++
    blt(10, 4, "loop_ti")
    emit(Encode.halt)

    val lb = labels.toMap
    prog.zipWithIndex.map { case (f, i) => f(lb, i) }.toSeq
  }

  // ---- run a GEMV through the kernel on the DmaTb harness, return partials ----
  // Returns partial(i)(g) = int32 sum over group g of W[i][k]*x[k].
  def runGemv(W: Array[Array[Int]], x: Array[Int], d: Int, n: Int, GS: Int): Array[Array[BigInt]] = {
    val nRowTiles = (d + nTile - 1) / nTile
    val nGroups   = n / GS
    val ktpg      = GS / nTile
    val nKtiles   = n / nTile
    val prog = kernelProgram()

    // DRAM (AxiRam) byte layout: weights (tile-major) | padded-x | out partials.
    val wBytes = nRowTiles * nKtiles * tileBytes
    val xBytes = nKtiles * tileBytes
    val oBytes = nRowTiles * nGroups * resTileBytes
    val weightPhys = 0
    val xPhys = wBytes
    val outPhys = wBytes + xBytes
    val dramBytes = wBytes + xBytes + oBytes
    val dramRows = (dramBytes + rowBytes - 1) / rowBytes + 16

    // Tile-major weights: tile (ti,kt) = nTile x nTile block, stored as nTile rows,
    // each beatsPerRow 128-bit beats (elemsPerBeat int8 each). Zero-pad out of range.
    def wAt(r: Int, c: Int): Int = if (r < d && c < n) W(r)(c) else 0
    val dram = mutable.Map[Int, BigInt]()   // byte-addr (16-aligned) -> 128-bit beat
    def packBeat(bytes: Array[Int]): BigInt =
      (0 until elemsPerBeat).foldLeft(BigInt(0))((acc, k) => acc | (BigInt(bytes(k) & 0xff) << (8 * k)))
    for (ti <- 0 until nRowTiles; kt <- 0 until nKtiles; r <- 0 until nTile; b <- 0 until beatsPerRow) {
      val beat = Array.tabulate(elemsPerBeat)(c => wAt(ti * nTile + r, kt * nTile + b * elemsPerBeat + c))
      dram((ti * nKtiles + kt) * tileBytes + r * p.ArrayRowBytes + b * rowBytes) = packBeat(beat)
    }
    // Padded x: tile kt, row r has x[kt*nTile+r] at col 0 (beat 0, byte 0).
    for (kt <- 0 until nKtiles; r <- 0 until nTile; b <- 0 until beatsPerRow) {
      val xv = { val idx = kt * nTile + r; if (idx < n) x(idx) else 0 }
      val beat = Array.tabulate(elemsPerBeat)(c => if (b == 0 && c == 0) xv else 0)
      dram(xPhys + kt * tileBytes + r * p.ArrayRowBytes + b * rowBytes) = packBeat(beat)
    }

    val ITCM = BigInt(0) << 19
    val DTCM = BigInt(1) << 19
    val CTRL = BigInt(2) << 19
    val OFF_CTRL = 0x00; val OFF_STATUS = 0x04

    val partials = Array.ofDim[BigInt](nRowTiles * nTile, nGroups)
    test(new DmaTb(p, dramRows = dramRows)).withAnnotations(backendAnnos) { dut =>
      val axi = dut.io.axi
      def step(): Unit = dut.clock.step(1)
      axi.aw.valid.poke(false.B); axi.w.valid.poke(false.B); axi.b.ready.poke(false.B)
      axi.ar.valid.poke(false.B); axi.r.ready.poke(false.B); dut.io.preWen.poke(false.B)
      dut.clock.setTimeout(0)

      def addrSend(ch: DecoupledIO[AxiAddr], addr: BigInt, beats: Int): Unit = {
        ch.bits.id.poke(0.U); ch.bits.addr.poke(addr.U)
        ch.bits.len.poke((beats - 1).U); ch.bits.size.poke(2.U); ch.bits.burst.poke(1.U)
        ch.valid.poke(true.B)
        while (!ch.ready.peek().litToBoolean) step()
        step(); ch.valid.poke(false.B)
      }
      def writeBurst(addr: BigInt, data: Seq[BigInt]): Unit = {
        addrSend(axi.aw, addr, data.length)
        for ((dw, i) <- data.zipWithIndex) {
          axi.w.bits.data.poke((dw & mask32).U); axi.w.bits.strb.poke("b1111".U)
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
      def dramPreload(byteAddr: Int, row: BigInt): Unit = {
        dut.io.preAddr.poke(byteAddr.U); dut.io.preData.poke(row.U)
        dut.io.preWen.poke(true.B); step(); dut.io.preWen.poke(false.B)
      }
      def dramReadRow(byteAddr: Int): BigInt = {
        dut.io.dbgAddr.poke(byteAddr.U); step(); dut.io.dbgData.peek().litValue
      }

      // Preload DRAM (weights + padded x).
      dram.toSeq.sortBy(_._1).foreach { case (a, row) => dramPreload(a, row) }

      // Descriptor -> DTCM[DESC], program -> ITCM.
      val desc = Seq[BigInt](weightPhys, xPhys, outPhys, nRowTiles, nGroups, ktpg, nKtiles)
      writeBurst(DTCM + DESC, desc)
      writeBurst(ITCM, prog)

      // Single kick + poll.
      writeBurst(CTRL + OFF_CTRL, Seq(BigInt(1)))
      var guard = 0
      while ((readWord(CTRL + OFF_STATUS) & 1) == 0) {
        guard += 1; assert(guard < 200000, "GEMV kernel did not finish")
      }

      // Read column-0 int32 of each (ti,g) result tile from DRAM.
      // tile (ti,g) at outPhys + (ti*nGroups+g)*resTileBytes; col0 row r at +r*resRowBytes.
      for (ti <- 0 until nRowTiles; g <- 0 until nGroups; r <- 0 until nTile) {
        val base = outPhys + (ti * nGroups + g) * resTileBytes + r * resRowBytes
        val row = dramReadRow(base & ~0xf)          // 16B-aligned wide beat
        val lane = (base & 0xf) / 4                  // which int32 in the beat
        partials(ti * nTile + r)(g) = (row >> (32 * lane)) & mask32
      }
    }
    partials
  }

  // Scala reference: int32 partial per (output row, group).
  def refPartials(W: Array[Array[Int]], x: Array[Int], d: Int, n: Int, GS: Int): Array[Array[BigInt]] = {
    val nGroups = n / GS
    Array.tabulate(d, nGroups) { (i, g) =>
      var acc = 0
      for (k <- g * GS until (g + 1) * GS) acc += W(i)(k) * x(k)
      BigInt(acc) & mask32
    }
  }

  def check(d: Int, n: Int, GS: Int, seed: Int): Unit = {
    val W = Array.tabulate(d, n)((i, j) => ((i * 5 + j * 3 + seed) % 13) - 6)
    val x = Array.tabulate(n)(j => ((j * 7 + seed * 2 + 1) % 11) - 5)
    val got = runGemv(W, x, d, n, GS)
    val exp = refPartials(W, x, d, n, GS)
    val nGroups = n / GS
    for (i <- 0 until d; g <- 0 until nGroups)
      assert(got(i)(g) == exp(i)(g), s"d=$d n=$n GS=$GS: partial($i)($g) got ${got(i)(g)} exp ${exp(i)(g)}")
  }

  behavior of "GEMV kernel"

  it should "compute a single tile (d=n=GS=nTile)" in { check(nTile, nTile, nTile, 1) }
  it should "accumulate two k-tiles within a group (n=GS=2*nTile)" in { check(nTile, 2 * nTile, 2 * nTile, 2) }
  it should "loop row-tiles and groups (d=n=2*nTile, GS=nTile)" in { check(2 * nTile, 2 * nTile, nTile, 3) }
}
