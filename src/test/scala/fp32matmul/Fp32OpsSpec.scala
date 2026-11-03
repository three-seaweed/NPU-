package fp32matmul

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import xnpu.XnpuBackend
import scala.util.Random

// Unit-level check: the Chisel ops must be bit-identical to Fp32Model (which
// is itself bit-identical to canonicalized Math.fma, see Fp32ModelSpec).
// Combinational DUT: poke, peek, no clock stepping needed.
class Fp32OpsSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  private def f(x: Int) = java.lang.Float.intBitsToFloat(x)

  "Fp32Ops" should "match Fp32Model bitwise" in {
    test(new Fp32OpsHarness).withAnnotations(backendAnnos) { dut =>
      var tested = 0
      def chk(a: Int, b: Int, c: Int): Unit = {
        dut.io.a.poke((a & 0xffffffffL).U); dut.io.b.poke((b & 0xffffffffL).U); dut.io.c.poke((c & 0xffffffffL).U)
        val gf = dut.io.fma.peek().litValue.toInt
        val gm = dut.io.mul.peek().litValue.toInt
        val ga = dut.io.add.peek().litValue.toInt
        val ef = Fp32Model.fma(a, b, c)
        val em = Fp32Model.mul(a, b)
        val ea = Fp32Model.add(a, b)
        assert(gf == ef, f"fma(0x$a%08x,0x$b%08x,0x$c%08x): hw 0x$gf%08x model 0x$ef%08x")
        assert(gm == em, f"mul(0x$a%08x,0x$b%08x): hw 0x$gm%08x model 0x$em%08x")
        assert(ga == ea, f"add(0x$a%08x,0x$b%08x): hw 0x$ga%08x model 0x$ea%08x")
        tested += 1
      }
      // directed specials, full cross
      for (a <- Rand32.specials; b <- Rand32.specials; c <- Rand32.specials) chk(a, b, c)
      // wide-exponent randoms
      val r = new Random(11)
      for (_ <- 0 until 5000)
        chk(Rand32.normal(r, 1, 254), Rand32.normal(r, 1, 254), Rand32.normal(r, 1, 254))
      // cancellation-prone: c ~= -(a*b) +- a few ulps
      for (_ <- 0 until 3000) {
        val a = Rand32.normal(r, 110, 140); val b = Rand32.normal(r, 110, 140)
        var c = java.lang.Float.floatToIntBits(-(f(a) * f(b))) + r.nextInt(9) - 4
        if (((c >> 23) & 0xff) == 0 || ((c >> 23) & 0xff) == 0xff) c = Rand32.normal(r, 110, 140)
        chk(a, b, c)
      }
      println(s"Fp32OpsSpec: $tested vectors x 3 ops, all bitwise equal")
    }
  }
}
