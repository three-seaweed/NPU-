// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable

class VectorUnitSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val p = Parameters()
  val L = p.Bf16LanesPerRow            // 8
  val rowBytes = p.DtcmRowBytes        // 16

  // ---- bf16 <-> float helpers ---------------------------------------------
  def floatToBf16(f: Float): BigInt = {
    if (f.isNaN) return BigInt(0x7FC0)
    val bits = java.lang.Float.floatToIntBits(f)
    val lsb = (bits >>> 16) & 1
    BigInt((bits + 0x7FFF + lsb) >>> 16) & 0xFFFF
  }
  def bf16ToFloat(b: BigInt): Float = java.lang.Float.intBitsToFloat((b.toInt & 0xFFFF) << 16)
  def f32bits(f: Float): BigInt = BigInt(java.lang.Float.floatToIntBits(f).toLong & 0xFFFFFFFFL)
  def bitsToF(b: BigInt): Float = java.lang.Float.intBitsToFloat(b.toInt)

  def packRow(vals: Seq[Float]): BigInt =
    vals.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (v, k)) => acc | (floatToBf16(v) << (16 * k)) }

  // ---- DTCM model: row-addressed 128-bit words ----------------------------
  class Dtcm {
    val mem = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
    def writeVec(base: Int, vals: Seq[Float]): Unit =
      vals.grouped(L).zipWithIndex.foreach { case (g, r) =>
        mem(BigInt(base + r * rowBytes)) = packRow(g.padTo(L, 0f))
      }
    def readLane(base: Int, i: Int): Float = {
      val row = mem(BigInt(base + (i / L) * rowBytes))
      bf16ToFloat((row >> (16 * (i % L))) & 0xFFFF)
    }
    // int8 view: 16 bytes per 128-bit row (used to read VCVT.I8 output).
    def readInt8(base: Int, i: Int): Int = {
      val row = mem(BigInt(base + (i / 16) * rowBytes))
      val b = ((row >> (8 * (i % 16))) & 0xFF).toInt
      if (b >= 128) b - 256 else b
    }
    // int32 view: 4 words per 128-bit row (used to seed VCVT.BF16 input).
    def writeInt32(base: Int, vals: Seq[Int]): Unit =
      vals.grouped(4).zipWithIndex.foreach { case (g, r) =>
        mem(BigInt(base + r * rowBytes)) =
          g.padTo(4, 0).zipWithIndex.foldLeft(BigInt(0)) { case (acc, (v, k)) =>
            acc | ((BigInt(v.toLong & 0xFFFFFFFFL)) << (32 * k))
          }
      }
  }

  // Drive one instruction and run the unit to idle.
  def run(dut: VectorUnit, d: Dtcm)(op: AluOp.Type, rs1: BigInt = 0, rd: Int = 0): Unit = {
    var pendingRead: Option[BigInt] = None
    def tick(): Unit = {
      dut.io.dtcm.rdata.poke(pendingRead.map(a => d.mem(a)).getOrElse(BigInt(0)).U)
      val v = dut.io.dtcm.valid.peek().litToBoolean
      val w = dut.io.dtcm.write.peek().litToBoolean
      val a = dut.io.dtcm.addr.peek().litValue
      if (v && w) {
        val wd = dut.io.dtcm.wdata.peek().litValue
        val strb = dut.io.dtcm.wstrb.peek().litValue
        var cur = d.mem(a)
        for (b <- 0 until rowBytes) if (((strb >> b) & 1) == 1) {
          val mask = BigInt(0xff) << (8 * b)
          cur = (cur & ~mask) | (((wd >> (8 * b)) & 0xff) << (8 * b))
        }
        d.mem(a) = cur
        pendingRead = None
      } else if (v) {
        pendingRead = Some(a)
      } else pendingRead = None
      dut.clock.step(1)
    }
    dut.io.rs1.data.poke(rs1.U)
    dut.io.req.bits.op.poke(op)
    dut.io.req.bits.addr.poke(rd.U)
    dut.io.req.valid.poke(true.B)
    assert(dut.io.req.ready.peek().litToBoolean, "unit not idle before issue")
    tick()
    dut.io.req.valid.poke(false.B)
    var guard = 0
    while (!dut.io.req.ready.peek().litToBoolean) {
      tick(); guard += 1; assert(guard < 5000, s"op $op did not complete")
    }
  }

  // Same as `run`, but reports how many cycles the op occupied the unit (from the
  // issue tick until req.ready returns). Used by the throughput regression below.
  def runCycles(dut: VectorUnit, d: Dtcm)(op: AluOp.Type, rs1: BigInt = 0, rd: Int = 0): Int = {
    var pendingRead: Option[BigInt] = None
    def tick(): Unit = {
      dut.io.dtcm.rdata.poke(pendingRead.map(a => d.mem(a)).getOrElse(BigInt(0)).U)
      val v = dut.io.dtcm.valid.peek().litToBoolean
      val w = dut.io.dtcm.write.peek().litToBoolean
      val a = dut.io.dtcm.addr.peek().litValue
      if (v && w) {
        val wd = dut.io.dtcm.wdata.peek().litValue
        val strb = dut.io.dtcm.wstrb.peek().litValue
        var cur = d.mem(a)
        for (b <- 0 until rowBytes) if (((strb >> b) & 1) == 1) {
          val mask = BigInt(0xff) << (8 * b)
          cur = (cur & ~mask) | (((wd >> (8 * b)) & 0xff) << (8 * b))
        }
        d.mem(a) = cur
        pendingRead = None
      } else if (v) pendingRead = Some(a) else pendingRead = None
      dut.clock.step(1)
    }
    dut.io.rs1.data.poke(rs1.U)
    dut.io.req.bits.op.poke(op)
    dut.io.req.bits.addr.poke(rd.U)
    dut.io.req.valid.poke(true.B)
    tick()
    dut.io.req.valid.poke(false.B)
    var cyc = 1
    while (!dut.io.req.ready.peek().litToBoolean) {
      tick(); cyc += 1; assert(cyc < 5000, s"op $op did not complete")
    }
    cyc
  }

  def config(dut: VectorUnit, d: Dtcm, srcA: Int, srcB: Int, dst: Int, len: Int, scalar: Float = 0f): Unit = {
    run(dut, d)(AluOp.VSETA, BigInt(srcA))
    run(dut, d)(AluOp.VSETB, BigInt(srcB))
    run(dut, d)(AluOp.VSETD, BigInt(dst))
    run(dut, d)(AluOp.VSETLEN, BigInt(len))
    run(dut, d)(AluOp.VSETS, f32bits(scalar))
  }

  val baseA = 0x000; val baseB = 0x100; val baseD = 0x200

  def setup(dut: VectorUnit): Unit = {
    dut.io.dtcm.ready.poke(true.B)
    dut.io.rd.ready.poke(true.B)
    dut.io.rs1.valid.poke(true.B)
    dut.io.rs2.valid.poke(true.B)
    dut.io.req.valid.poke(false.B)
    dut.io.context.valid.poke(false.B)
    dut.io.context.srcA.poke(0.U)
    dut.io.context.srcB.poke(0.U)
    dut.io.context.dst.poke(0.U)
    dut.io.context.vlen.poke(0.U)
    dut.io.context.scalar.poke(0.U)
  }

  def relOk(got: Float, exp: Float, tol: Double = 0.02): Boolean =
    if (math.abs(exp) < 1e-6) math.abs(got) < 0.05 else math.abs((got - exp) / exp) < tol

  behavior of "VectorUnit elementwise"

  it should "compute VADD / VSUB / VMUL over two rows" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      val a = (0 until 16).map(i => (i - 8) * 0.5f)
      val b = (0 until 16).map(i => (i % 5) * 1.25f + 0.5f)
      d.writeVec(baseA, a); d.writeVec(baseB, b)
      for ((op, ref) <- Seq[(AluOp.Type, (Float, Float) => Float)](
            (AluOp.VADD, _ + _), (AluOp.VSUB, _ - _), (AluOp.VMUL, _ * _))) {
        config(dut, d, baseA, baseB, baseD, 16)
        run(dut, d)(op)
        for (i <- 0 until 16)
          assert(relOk(d.readLane(baseD, i), ref(a(i), b(i))), s"$op lane $i")
      }
    }
  }

  it should "compute scalar VSMUL / VSADD" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      val a = (0 until 8).map(i => (i + 1) * 0.75f)
      d.writeVec(baseA, a)
      config(dut, d, baseA, baseB, baseD, 8, scalar = 2.5f)
      run(dut, d)(AluOp.VSMUL)
      for (i <- 0 until 8) assert(relOk(d.readLane(baseD, i), a(i) * 2.5f), s"smul $i")
      config(dut, d, baseA, baseB, baseD, 8, scalar = 2.5f)
      run(dut, d)(AluOp.VSADD)
      for (i <- 0 until 8) assert(relOk(d.readLane(baseD, i), a(i) + 2.5f), s"sadd $i")
    }
  }

  it should "compute VFMA (dst += a*b)" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      val a = (0 until 8).map(i => (i + 1) * 0.5f)
      val b = (0 until 8).map(i => (i % 3 + 1) * 1.0f)
      val c = (0 until 8).map(i => i * 0.25f)
      d.writeVec(baseA, a); d.writeVec(baseB, b); d.writeVec(baseD, c)
      config(dut, d, baseA, baseB, baseD, 8)
      run(dut, d)(AluOp.VFMA)
      for (i <- 0 until 8) assert(relOk(d.readLane(baseD, i), a(i) * b(i) + c(i)), s"fma $i")
    }
  }

  it should "honour a partial tail (vlen=5)" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      val a = (0 until 8).map(i => (i + 1) * 1.0f)
      val b = (0 until 8).map(i => (i + 1) * 2.0f)
      d.writeVec(baseA, a); d.writeVec(baseB, b)
      // sentinel in dst lanes 5..7 must survive.
      d.writeVec(baseD, Seq.fill(8)(99f))
      config(dut, d, baseA, baseB, baseD, 5)
      run(dut, d)(AluOp.VADD)
      for (i <- 0 until 5) assert(relOk(d.readLane(baseD, i), a(i) + b(i)), s"tail $i")
      for (i <- 5 until 8) assert(relOk(d.readLane(baseD, i), 99f), s"tail untouched $i")
    }
  }

  behavior of "VectorUnit activations"
  it should "compute VEXP / VRECIP / VRSQRT" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      val a = (0 until 8).map(i => (i + 1) * 0.5f)     // 0.5 .. 4.0, all > 0
      d.writeVec(baseA, a)
      config(dut, d, baseA, baseB, baseD, 8)
      run(dut, d)(AluOp.VEXP)
      for (i <- 0 until 8) assert(relOk(d.readLane(baseD, i), math.exp(a(i)).toFloat, 0.04), s"exp $i")
      config(dut, d, baseA, baseB, baseD, 8)
      run(dut, d)(AluOp.VRECIP)
      for (i <- 0 until 8) assert(relOk(d.readLane(baseD, i), 1f / a(i), 0.03), s"recip $i")
      config(dut, d, baseA, baseB, baseD, 8)
      run(dut, d)(AluOp.VRSQRT)
      for (i <- 0 until 8) assert(relOk(d.readLane(baseD, i), (1.0 / math.sqrt(a(i))).toFloat, 0.03), s"rsqrt $i")
    }
  }

  behavior of "VectorUnit reductions"
  it should "compute VSUM / VMAX / VDOT into rd" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      val a = (0 until 12).map(i => (i - 3) * 0.5f)
      val b = (0 until 12).map(i => (i % 4 + 1) * 0.5f)
      d.writeVec(baseA, a); d.writeVec(baseB, b)

      config(dut, d, baseA, baseB, baseD, 12)
      run(dut, d)(AluOp.VSUM, rd = 5)
      assert(relOk(bitsToF(dut.io.rd.bits.data.peek().litValue), a.sum), "vsum")

      config(dut, d, baseA, baseB, baseD, 12)
      run(dut, d)(AluOp.VMAX, rd = 5)
      assert(relOk(bitsToF(dut.io.rd.bits.data.peek().litValue), a.max), "vmax")

      config(dut, d, baseA, baseB, baseD, 12)
      run(dut, d)(AluOp.VDOT, rd = 5)
      val dot = a.zip(b).map { case (x, y) => x * y }.sum
      assert(relOk(bitsToF(dut.io.rd.bits.data.peek().litValue), dot, 0.03), "vdot")
    }
  }

  behavior of "VectorUnit format conversion primitives"

  it should "compose VABS and VMAX to calculate a magnitude" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      val a = (0 until 12).map(i => (i - 7) * 0.5f)   // spans negative & positive
      d.writeVec(baseA, a)
      config(dut, d, baseA, baseB, baseD, 12)
      run(dut, d)(AluOp.VABS)
      config(dut, d, baseD, baseB, baseA, 12)
      run(dut, d)(AluOp.VMAX, rd = 7)
      assert(relOk(bitsToF(dut.io.rd.bits.data.peek().litValue), a.map(math.abs).max), "abs + vmax")
    }
  }

  it should "convert bf16 to signed int8 with RNE and native saturation" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      // 20 elements span multiple half-rows and a partial tail. The values cover
      // round-to-even ties and both signed saturation endpoints.
      val a = Seq(0.25f, 0.75f, 1.25f, 1.75f, -0.25f, -0.75f, -1.25f, -1.75f,
        127.5f, -127.5f, 130f, -130f) ++ (0 until 8).map(i => (i - 4) * 1.3f)
      d.writeVec(baseA, a)
      // Sentinels in the output region: only converted lanes must change.
      d.writeVec(baseD, Seq.fill(24)(0f))
      config(dut, d, baseA, baseB, baseD, a.length)
      run(dut, d)(AluOp.VCVT_I8)
      def cvt(x: Float): Int = {
        val xb = bf16ToFloat(floatToBf16(x))
        math.max(-128, math.min(127, math.rint(xb).toInt))
      }
      for (i <- a.indices)
        assert(d.readInt8(baseD, i) == cvt(a(i)), s"vcvt.i8 lane $i: got ${d.readInt8(baseD, i)} exp ${cvt(a(i))}")
    }
  }

  it should "map NaN to zero during bf16-to-int8 conversion" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      d.writeVec(baseA, Seq.fill(16)(Float.NaN))
      config(dut, d, baseA, baseB, baseD, 16)
      run(dut, d)(AluOp.VCVT_I8)
      for (i <- 0 until 16) assert(d.readInt8(baseD, i) == 0, s"NaN conversion lane $i")
    }
  }

  it should "convert int32 to bf16 without applying a quantization scale" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      val c   = (0 until 12).map(i => (i - 5) * 37)             // int32 accumulators
      d.writeInt32(baseA, c)
      config(dut, d, baseA, baseB, baseD, 12)
      run(dut, d)(AluOp.VCVT_BF16)
      for (i <- 0 until 12) {
        val exp = c(i).toFloat
        assert(relOk(d.readLane(baseD, i), exp, 0.03), s"vcvt.bf16 lane $i: got ${d.readLane(baseD, i)} exp $exp")
      }
    }
  }

  behavior of "VectorUnit compound"
  it should "compute VSOFTEXP (dst=exp(a-s), rd=sum)" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      val a = (0 until 8).map(i => (i - 4) * 0.5f)
      val mx = a.max
      d.writeVec(baseA, a)
      config(dut, d, baseA, baseB, baseD, 8, scalar = mx)
      run(dut, d)(AluOp.VSOFTEXP, rd = 6)
      val e = a.map(x => math.exp((x - mx).toDouble).toFloat)
      for (i <- 0 until 8) assert(relOk(d.readLane(baseD, i), e(i), 0.04), s"softexp lane $i")
      assert(relOk(bitsToF(dut.io.rd.bits.data.peek().litValue), e.sum, 0.04), "softexp sum")
    }
  }

  // The unit is a row pipeline (fetch / compute / store run concurrently on
  // different rows), so steady-state cost per row is set by DTCM port traffic
  // rather than by the sum of the per-row FSM steps. Asserting upper bounds here
  // is what makes "it got faster" a checked property instead of a claim, and
  // stops a later change from silently serialising the pipeline again.
  //
  // Port accesses per row: VSMUL 1 read + 1 write, VMUL 2 reads + 1 write,
  // VDOT 2 reads and no write. The bound allows a small constant for fill/drain
  // plus one issue cycle; before pipelining these cost 5, 7 and 10 cycles/row.
  behavior of "VectorUnit throughput"
  it should "stream rows at DTCM-port rate" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      val nRows = 16
      val len = nRows * L
      val a = (0 until len).map(i => 0.5f + 0.01f * i)
      val b = (0 until len).map(i => 1.5f - 0.01f * i)
      d.writeVec(baseA, a)
      d.writeVec(baseB, b)

      def perRow(op: AluOp.Type, rd: Int = 0): Double = {
        config(dut, d, baseA, baseB, baseD, len, scalar = 2.0f)
        runCycles(dut, d)(op, rd = rd).toDouble / nRows
      }

      val smul = perRow(AluOp.VSMUL)
      val mul  = perRow(AluOp.VMUL)
      val dot  = perRow(AluOp.VDOT, rd = 7)
      info(f"cycles/row: VSMUL $smul%.2f  VMUL $mul%.2f  VDOT $dot%.2f")
      assert(smul <= 3.0, f"VSMUL $smul%.2f cycles/row (2 port accesses; want <= 3)")
      assert(mul  <= 4.0, f"VMUL $mul%.2f cycles/row (3 port accesses; want <= 4)")
      assert(dot  <= 3.5, f"VDOT $dot%.2f cycles/row (2 port accesses; want <= 3.5)")
    }
  }

  // Reductions must stay bit-exact: the tree is now free-running, but rows still
  // enter it in order and `acc` is still folded once per row in row order. fp32
  // addition is not associative, so any reassociation would show up here.
  it should "fold a reduction in strict row order" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      // Magnitudes chosen so that a different summation order gives a different
      // fp32 result: a large leading term absorbs later small ones.
      val a = Seq(1.0e7f) ++ Seq.fill(31)(1.0f)
      d.writeVec(baseA, a)
      config(dut, d, baseA, baseB, baseD, a.length)
      run(dut, d)(AluOp.VSUM, rd = 5)
      // Golden: the exact per-row tree, folded into acc row by row.
      var acc = 0.0f
      for (rowVals <- a.grouped(L)) {
        val padded = rowVals.map(v => bf16ToFloat(floatToBf16(v)))
        var lvl: Seq[Float] = padded
        while (lvl.length > 1) lvl = lvl.grouped(2).map(g => if (g.length == 2) g(0) + g(1) else g(0)).toSeq
        acc += lvl.head
      }
      val got = bitsToF(dut.io.rd.bits.data.peek().litValue)
      assert(got == acc, f"VSUM order changed: got $got%.6g exp $acc%.6g")
    }
  }
}
