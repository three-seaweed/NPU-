// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

// Combinational harness exposing every Bf16 primitive for direct poke/peek.
class Bf16Harness extends Module {
  val io = IO(new Bundle {
    val a16 = Input(UInt(16.W))
    val b16 = Input(UInt(16.W))
    val x32 = Input(UInt(32.W))
    val y32 = Input(UInt(32.W))
    val acc = Input(UInt(32.W))
    val i32 = Input(UInt(32.W))

    val toF32  = Output(UInt(32.W))
    val narrow = Output(UInt(16.W))
    val mul    = Output(UInt(32.W))
    val add    = Output(UInt(32.W))
    val fma    = Output(UInt(32.W))
    val gt     = Output(Bool())
    val i2f    = Output(UInt(32.W))
  })
  io.toF32  := Bf16.toF32(io.a16)
  io.narrow := Bf16.f32ToBf16(io.x32)
  io.mul    := Bf16.mulExact(io.a16, io.b16)
  io.add    := Bf16.addF32(io.x32, io.y32)
  io.fma    := Bf16.fmaF32(io.a16, io.b16, io.acc)
  io.gt     := Bf16.gtF32(io.x32, io.y32)
  io.i2f    := Bf16.i32ToF32(io.i32)
}

class Bf16Spec extends AnyFlatSpec with ChiselScalatestTester {
  // ---- Scala reference helpers --------------------------------------------
  def floatToBf16(f: Float): Int = {
    if (f.isNaN) return 0x7FC0
    val bits = java.lang.Float.floatToIntBits(f)
    val lsb = (bits >>> 16) & 1
    val rounded = bits + 0x7FFF + lsb        // RNE
    (rounded >>> 16) & 0xFFFF
  }
  def bf16ToFloat(b: Int): Float = java.lang.Float.intBitsToFloat((b & 0xFFFF) << 16)
  def f32bits(f: Float): Long = java.lang.Float.floatToIntBits(f).toLong & 0xFFFFFFFFL

  // ulp distance between two fp32 bit patterns (sign-magnitude aware enough for
  // same-sign normal values; we only use it on positive-ish results).
  def ulpDiff(a: Long, b: Long): Long = math.abs(a - b)

  val rng = new Random(1)
  def randBf16Float(): Float = {
    // moderate exponents to stay clear of FTZ/denormal/overflow corners.
    val sign = if (rng.nextBoolean()) -1f else 1f
    val mant = 1f + rng.nextInt(128) / 128f
    val exp  = rng.nextInt(20) - 10           // 2^-10 .. 2^9
    sign * mant * math.pow(2.0, exp).toFloat
  }

  behavior of "Bf16.toF32"
  it should "widen bf16 to the exact fp32 value" in {
    test(new Bf16Harness) { dut =>
      for (_ <- 0 until 400) {
        val f = randBf16Float()
        val b = floatToBf16(f)
        dut.io.a16.poke(b.U)
        val exp = f32bits(bf16ToFloat(b))
        dut.io.toF32.expect(exp.U, s"toF32(${b.toHexString})")
      }
      // signed zero
      dut.io.a16.poke(0x0000.U); dut.io.toF32.expect(0.U)
      dut.io.a16.poke(0x8000.U); dut.io.toF32.expect(0x80000000L.U)
    }
  }

  behavior of "Bf16.f32ToBf16"
  it should "narrow fp32 to bf16 with RNE" in {
    test(new Bf16Harness) { dut =>
      for (_ <- 0 until 400) {
        val f = randBf16Float() * (1f + rng.nextInt(1000) / 1000f)  // off-grid
        dut.io.x32.poke(f32bits(f).U)
        dut.io.narrow.expect(floatToBf16(f).U, s"narrow($f)")
      }
    }
  }

  behavior of "Bf16.mulExact"
  it should "multiply bf16 operands exactly into fp32" in {
    test(new Bf16Harness) { dut =>
      for (_ <- 0 until 600) {
        val fa = randBf16Float(); val fb = randBf16Float()
        val ba = floatToBf16(fa); val bb = floatToBf16(fb)
        dut.io.a16.poke(ba.U); dut.io.b16.poke(bb.U)
        val prod = bf16ToFloat(ba) * bf16ToFloat(bb)   // exact in fp32
        dut.io.mul.expect(f32bits(prod).U, s"mul($fa,$fb)")
      }
      // zero handling
      dut.io.a16.poke(0.U); dut.io.b16.poke(floatToBf16(3.5f).U)
      dut.io.mul.expect(0.U)
    }
  }

  behavior of "Bf16.addF32"
  it should "add fp32 within 1 ulp (RNE)" in {
    test(new Bf16Harness) { dut =>
      for (_ <- 0 until 1000) {
        val fa = randBf16Float(); val fb = randBf16Float()
        dut.io.x32.poke(f32bits(fa).U); dut.io.y32.poke(f32bits(fb).U)
        val got = dut.io.add.peek().litValue.toLong & 0xFFFFFFFFL
        val exp = f32bits(fa + fb)
        assert(ulpDiff(got, exp) <= 1,
          s"add($fa,$fb): got ${got.toHexString} exp ${exp.toHexString}")
      }
    }
  }
  it should "handle exact cancellation to zero" in {
    test(new Bf16Harness) { dut =>
      val f = 12.5f
      dut.io.x32.poke(f32bits(f).U); dut.io.y32.poke(f32bits(-f).U)
      dut.io.add.expect(0.U, "x + (-x) = 0")
      dut.io.x32.poke(f32bits(-f).U); dut.io.y32.poke(f32bits(f).U)
      dut.io.add.expect(0.U, "(-x) + x = 0")
    }
  }

  behavior of "Bf16.fmaF32"
  it should "compute a*b + acc within 1 ulp" in {
    test(new Bf16Harness) { dut =>
      for (_ <- 0 until 600) {
        val fa = randBf16Float(); val fb = randBf16Float(); val fc = randBf16Float()
        val ba = floatToBf16(fa); val bb = floatToBf16(fb)
        dut.io.a16.poke(ba.U); dut.io.b16.poke(bb.U); dut.io.acc.poke(f32bits(fc).U)
        val got = dut.io.fma.peek().litValue.toLong & 0xFFFFFFFFL
        val exp = f32bits(bf16ToFloat(ba) * bf16ToFloat(bb) + fc)
        assert(ulpDiff(got, exp) <= 1, s"fma: got ${got.toHexString} exp ${exp.toHexString}")
      }
    }
  }

  behavior of "Bf16.i32ToF32"
  it should "convert signed int32 to fp32 with RNE" in {
    test(new Bf16Harness) { dut =>
      def check(v: Int): Unit = {
        dut.io.i32.poke((BigInt(v) & 0xFFFFFFFFL).U)
        dut.io.i2f.expect(f32bits(v.toFloat).U, s"i32ToF32($v)")
      }
      // boundary cases
      Seq(0, 1, -1, 2, -2, 127, -127, 128, -128,
          0x7FFFFFFF, -0x7FFFFFFF, Int.MinValue,   // -2^31
          0x00FFFFFF, 0x01000000, 0x01000001,      // rounding around 24-bit boundary
          16777217, -16777217).foreach(check)
      // random large values that require rounding
      for (_ <- 0 until 2000) check(rng.nextInt())
    }
  }

  behavior of "Bf16.gtF32"
  it should "order fp32 by signed value" in {
    test(new Bf16Harness) { dut =>
      val cases = Seq((1.0f, 2.0f), (-1.0f, 2.0f), (-3.0f, -2.0f), (5.0f, 5.0f), (0.0f, -1.0f))
      for ((a, b) <- cases) {
        dut.io.x32.poke(f32bits(a).U); dut.io.y32.poke(f32bits(b).U)
        dut.io.gt.expect((a > b).B, s"gt($a,$b)")
      }
    }
  }
}
