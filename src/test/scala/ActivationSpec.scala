// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

class ActHarness extends Module {
  val io = IO(new Bundle {
    val in    = Input(UInt(32.W))
    val exp   = Output(UInt(32.W))
    val rsqrt = Output(UInt(32.W))
    val recip = Output(UInt(32.W))
  })
  io.exp   := Activation.exp(io.in)
  io.rsqrt := Activation.rsqrt(io.in)
  io.recip := Activation.recip(io.in)
}

class ActivationSpec extends AnyFlatSpec with ChiselScalatestTester {
  def f32bits(f: Float): Long = java.lang.Float.floatToIntBits(f).toLong & 0xFFFFFFFFL
  def bitsToF(b: Long): Float = java.lang.Float.intBitsToFloat(b.toInt)
  val rng = new Random(7)

  def relErr(got: Float, exp: Float): Double =
    if (exp == 0f) math.abs(got).toDouble else math.abs((got - exp) / exp).toDouble

  behavior of "Activation.exp"
  it should "approximate e^x within 1% over [-10,10]" in {
    test(new ActHarness) { dut =>
      for (_ <- 0 until 400) {
        val x = (rng.nextDouble() * 20 - 10).toFloat
        dut.io.in.poke(f32bits(x).U)
        val got = bitsToF(dut.io.exp.peek().litValue.toLong & 0xFFFFFFFFL)
        val exp = math.exp(x.toDouble).toFloat
        assert(relErr(got, exp) < 0.01, s"exp($x): got $got exp $exp")
      }
    }
  }
  it should "clamp extremes" in {
    test(new ActHarness) { dut =>
      dut.io.in.poke(f32bits(0f).U)
      assert(math.abs(bitsToF(dut.io.exp.peek().litValue.toLong & 0xFFFFFFFFL) - 1f) < 0.01f)
      dut.io.in.poke(f32bits(-50f).U)
      assert(bitsToF(dut.io.exp.peek().litValue.toLong & 0xFFFFFFFFL) < 1e-10f)
      dut.io.in.poke(f32bits(100f).U)
      assert(bitsToF(dut.io.exp.peek().litValue.toLong & 0xFFFFFFFFL).isInfinite)
    }
  }

  behavior of "Activation.rsqrt"
  it should "approximate 1/sqrt(x) within 0.5%" in {
    test(new ActHarness) { dut =>
      for (_ <- 0 until 400) {
        val x = (rng.nextDouble() * 1000 + 1e-3).toFloat
        dut.io.in.poke(f32bits(x).U)
        val got = bitsToF(dut.io.rsqrt.peek().litValue.toLong & 0xFFFFFFFFL)
        val exp = (1.0 / math.sqrt(x.toDouble)).toFloat
        assert(relErr(got, exp) < 0.005, s"rsqrt($x): got $got exp $exp")
      }
      dut.io.in.poke(f32bits(0f).U)
      assert(bitsToF(dut.io.rsqrt.peek().litValue.toLong & 0xFFFFFFFFL).isInfinite)
      dut.io.in.poke(f32bits(-4f).U)
      assert(bitsToF(dut.io.rsqrt.peek().litValue.toLong & 0xFFFFFFFFL).isNaN)
    }
  }

  behavior of "Activation.recip"
  it should "approximate 1/x within 0.2%" in {
    test(new ActHarness) { dut =>
      for (_ <- 0 until 400) {
        val mag = (rng.nextDouble() * 100 + 0.01).toFloat
        val x = if (rng.nextBoolean()) mag else -mag
        dut.io.in.poke(f32bits(x).U)
        val got = bitsToF(dut.io.recip.peek().litValue.toLong & 0xFFFFFFFFL)
        val exp = (1.0 / x.toDouble).toFloat
        assert(relErr(got, exp) < 0.002, s"recip($x): got $got exp $exp")
      }
    }
  }
}
