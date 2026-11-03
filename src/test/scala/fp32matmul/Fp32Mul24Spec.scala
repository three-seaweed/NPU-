package fp32matmul

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import xnpu.XnpuBackend
import scala.util.Random

// Karatsuba 3-partial-product 24x24 multiplier must be BIT-IDENTICAL to a
// plain 24x24 multiply (it is an integer identity, no rounding involved).
// This pins the slicing/recombination before trusting it inside the FMA.
class Mul24Harness extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(24.W))
    val b = Input(UInt(24.W))
    val o = Output(UInt(48.W))
  })
  io.o := Fp32Ops.mul24(io.a, io.b)
}

class Fp32Mul24Spec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  private def big(x: BigInt) = (x & 0xffffffffffffL).U(48.W).litValue

  "Fp32Ops.mul24" should "match a plain 24x24 multiply bitwise" in {
    test(new Mul24Harness).withAnnotations(backendAnnos) { dut =>
      def chk(a: Int, b: Int): Unit = {
        dut.io.a.poke(a.U); dut.io.b.poke(b.U)
        val got = dut.io.o.peek().litValue
        val exp = BigInt(a) * BigInt(b)
        assert(got == exp, s"mul24(0x$a%06x, 0x$b%06x) = 0x$got%012x, expected 0x${exp.toString(16)}")
      }
      // directed corners: split boundaries, all-ones, MSB/LSB patterns
      val corners = Seq(0, 1, 2, 0xfff, 0x1000, 0x800, 0xfff000, 0x1000000 / 0x1000,
                        0x7fffff, 0x800000, 0xffffff, 0xabcde, 0x555555, 0xaaaaaa, 0x123456)
        .map(x => if (x >= (1 << 24)) (1 << 24) - 1 else x)
      for (a <- corners; b <- corners) chk(a, b)
      // around the 12-bit split boundary
      for (a <- Seq(0xfff, 0x1000, 0x1001); b <- Seq(0xfff, 0x1000, 0x1001, 0xffffff)) chk(a, b)
      val r = new Random(21)
      for (_ <- 0 until 200000) chk(r.nextInt(1 << 24), r.nextInt(1 << 24))
      println("Fp32Mul24Spec: 225 directed + 200000 random, all bitwise equal")
    }
  }
}
