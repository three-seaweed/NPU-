// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable

// Replays the activation inputs that sw/xnpu_optimized_layer_test.c actually
// feeds on the board. VectorUnitSpec's activation case only ever uses 0.5..4.0,
// so every special-case path in the inline Newton chains -- 1/0, 1/sqrt(0),
// 1/sqrt(-1), exp of an exact zero -- was unexercised, and those are precisely
// the lanes the bare-metal [vector] section reported wrong.
//
// vlen is 16 (two rows) rather than the 8 used elsewhere, matching the board
// program: a multi-cycle chain holds its input buffer while the fetch segment
// runs ahead into the other buffer, and that only happens with more than one row.
class ActivationEdgeSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val p = Parameters()
  val L = p.Bf16LanesPerRow
  val rowBytes = p.DtcmRowBytes

  def floatToBf16(f: Float): BigInt = {
    if (f.isNaN) return BigInt(0x7FC0)
    val bits = java.lang.Float.floatToIntBits(f)
    val lsb = (bits >>> 16) & 1
    BigInt((bits + 0x7FFF + lsb) >>> 16) & 0xFFFF
  }
  def f32bits(f: Float): BigInt = BigInt(java.lang.Float.floatToIntBits(f).toLong & 0xFFFFFFFFL)
  def bitsToF(b: BigInt): Float = java.lang.Float.intBitsToFloat(b.toInt)

  class Dtcm {
    val mem = mutable.Map[BigInt, BigInt]().withDefaultValue(BigInt(0))
    def writeVec(base: Int, vals: Seq[Float]): Unit =
      vals.grouped(L).zipWithIndex.foreach { case (g, r) =>
        mem(BigInt(base + r * rowBytes)) =
          g.padTo(L, 0f).zipWithIndex.foldLeft(BigInt(0)) {
            case (acc, (v, k)) => acc | (floatToBf16(v) << (16 * k))
          }
      }
    def readRaw(base: Int, i: Int): Int = {
      val row = mem(BigInt(base + (i / L) * rowBytes))
      (((row >> (16 * (i % L))) & 0xFFFF).toInt) & 0xFFFF
    }
    def readLane(base: Int, i: Int): Float =
      java.lang.Float.intBitsToFloat(readRaw(base, i) << 16)
  }

  def run(dut: VectorUnit, d: Dtcm)(op: AluOp.Type, rs1: BigInt = 0, rd: Int = 0): Unit = {
    var pending: Option[BigInt] = None
    def tick(): Unit = {
      dut.io.dtcm.rdata.poke(pending.map(a => d.mem(a)).getOrElse(BigInt(0)).U)
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
        pending = None
      } else if (v) pending = Some(a) else pending = None
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

  def setup(dut: VectorUnit): Unit = {
    dut.io.dtcm.ready.poke(true.B)
    dut.io.rd.ready.poke(true.B)
    dut.io.rs1.valid.poke(true.B)
    dut.io.rs2.valid.poke(true.B)
    dut.io.req.valid.poke(false.B)
    dut.io.context.valid.poke(false.B)
    dut.io.context.srcA.poke(0.U); dut.io.context.srcB.poke(0.U)
    dut.io.context.dst.poke(0.U); dut.io.context.vlen.poke(0.U)
    dut.io.context.scalar.poke(0.U)
  }

  def config(dut: VectorUnit, d: Dtcm, a: Int, b: Int, dst: Int, len: Int, s: Float = 0f): Unit = {
    run(dut, d)(AluOp.VSETA, BigInt(a))
    run(dut, d)(AluOp.VSETB, BigInt(b))
    run(dut, d)(AluOp.VSETD, BigInt(dst))
    run(dut, d)(AluOp.VSETLEN, BigInt(len))
    run(dut, d)(AluOp.VSETS, f32bits(s))
  }

  val baseA = 0x000; val baseB = 0x100; val baseD = 0x200
  val VL = 16

  // Same as the board: D_VEC_POS = [2, 4, 0, -1] repeating.
  val pos: Seq[Float] = (0 until VL).map(i => (i & 3) match {
    case 0 => 2.0f
    case 1 => 4.0f
    case 2 => 0.0f
    case _ => -1.0f
  })

  def close(got: Float, want: Float, tol: Float): Boolean =
    !got.isNaN && math.abs(got - want) <= tol

  behavior of "VectorUnit activation special cases"

  it should "return exactly 1.0 for VEXP of an all-zero vector" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      d.writeVec(baseA, Seq.fill(VL)(0.0f))
      config(dut, d, baseA, baseB, baseD, VL, 0.0f)
      run(dut, d)(AluOp.VEXP)
      for (i <- 0 until VL) {
        val raw = d.readRaw(baseD, i)
        assert(raw == 0x3F80, f"VEXP[$i] = 0x$raw%04x, expected 0x3f80")
      }
    }
  }

  it should "handle VRECIP of 2, 4, 0 and -1" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      d.writeVec(baseA, pos)
      config(dut, d, baseA, baseB, baseD, VL)
      run(dut, d)(AluOp.VRECIP)
      for (i <- 0 until VL) {
        val raw = d.readRaw(baseD, i)
        val got = d.readLane(baseD, i)
        (i & 3) match {
          case 0 => assert(close(got, 0.5f, 0.03f), f"VRECIP[$i]=0x$raw%04x, expected ~0.5")
          case 1 => assert(close(got, 0.25f, 0.03f), f"VRECIP[$i]=0x$raw%04x, expected ~0.25")
          case 2 => assert(raw == 0x7F80, f"VRECIP[$i]=0x$raw%04x, expected +Inf (1/0)")
          case _ => assert(close(got, -1.0f, 0.03f), f"VRECIP[$i]=0x$raw%04x, expected ~-1.0")
        }
      }
    }
  }

  it should "handle VRSQRT of 2, 4, 0 and -1" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      d.writeVec(baseA, pos)
      config(dut, d, baseA, baseB, baseD, VL)
      run(dut, d)(AluOp.VRSQRT)
      for (i <- 0 until VL) {
        val raw = d.readRaw(baseD, i)
        val got = d.readLane(baseD, i)
        (i & 3) match {
          case 0 => assert(close(got, 0.70710678f, 0.04f), f"VRSQRT[$i]=0x$raw%04x, expected ~0.707")
          case 1 => assert(close(got, 0.5f, 0.04f), f"VRSQRT[$i]=0x$raw%04x, expected ~0.5")
          case 2 => assert(raw == 0x7F80, f"VRSQRT[$i]=0x$raw%04x, expected +Inf (1/sqrt 0)")
          case _ => assert((raw & 0x7F80) == 0x7F80 && (raw & 0x007F) != 0,
                           f"VRSQRT[$i]=0x$raw%04x, expected NaN (1/sqrt of negative)")
        }
      }
    }
  }

  it should "compute VSOFTEXP over an all-zero vector with a zero scalar" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      d.writeVec(baseA, Seq.fill(VL)(0.0f))
      // The board runs this in place: VSETA and VSETD both point at D_VEC_SOFT.
      config(dut, d, baseA, baseB, baseA, VL, 0.0f)
      run(dut, d)(AluOp.VSOFTEXP, rd = 7)
      val sum = bitsToF(dut.io.rd.bits.data.peek().litValue)
      for (i <- 0 until VL) {
        val raw = d.readRaw(baseA, i)
        assert(raw == 0x3F80, f"VSOFTEXP.dst[$i] = 0x$raw%04x, expected 0x3f80")
      }
      assert(sum == 16.0f, s"VSOFTEXP sum = $sum, expected 16.0")
    }
  }

  // The softmax chain the toy layer runs: VDOT of two all-ones vectors, then a
  // VMAX, then VSOFTEXP against that max, then VRECIP of the sum.
  it should "reproduce the toy layer's single-token softmax chain" in {
    test(new VectorUnit(p)).withAnnotations(backendAnnos) { dut =>
      setup(dut)
      val d = new Dtcm
      d.writeVec(baseA, Seq.fill(VL)(1.0f))
      d.writeVec(baseB, Seq.fill(VL)(1.0f))

      config(dut, d, baseA, baseB, baseD, VL)
      run(dut, d)(AluOp.VDOT, rd = 4)
      val qk = bitsToF(dut.io.rd.bits.data.peek().litValue)
      assert(qk == 16.0f, s"VDOT = $qk, expected 16.0")

      // Score row: one lane holding bf16(qk), scaled by 1/sqrt(head_dim).
      d.writeVec(baseD, Seq(qk))
      config(dut, d, baseD, baseB, baseD, 1, 1.0f / 5.656854249f)
      run(dut, d)(AluOp.VSMUL)
      val scaled = d.readLane(baseD, 0)
      assert(close(scaled, 16.0f / 5.656854249f, 0.02f), s"scaled score = $scaled")

      config(dut, d, baseD, baseB, baseD, 1)
      run(dut, d)(AluOp.VMAX, rd = 9)
      val mx = bitsToF(dut.io.rd.bits.data.peek().litValue)
      assert(mx == scaled, s"VMAX = $mx, expected the single lane $scaled")

      config(dut, d, baseD, baseB, baseD, 1, mx)
      run(dut, d)(AluOp.VSOFTEXP, rd = 10)
      val sum = bitsToF(dut.io.rd.bits.data.peek().litValue)
      assert(d.readRaw(baseD, 0) == 0x3F80,
             f"exp(x-max) = 0x${d.readRaw(baseD, 0)}%04x, expected 0x3f80")
      assert(sum == 1.0f, s"softmax sum = $sum, expected 1.0")

      // 1/sum, then normalise: the layer asserts this comes back as exactly 1.0.
      d.writeVec(baseB, Seq(sum))
      config(dut, d, baseB, baseB, baseB, 1)
      run(dut, d)(AluOp.VRECIP)
      val inv = d.readLane(baseB, 0)
      assert(inv == 1.0f, s"1/sum = $inv, expected 1.0")

      config(dut, d, baseD, baseB, baseD, 1, inv)
      run(dut, d)(AluOp.VSMUL)
      assert(d.readRaw(baseD, 0) == 0x3F80,
             f"final softmax = 0x${d.readRaw(baseD, 0)}%04x, expected 0x3f80")
    }
  }
}
