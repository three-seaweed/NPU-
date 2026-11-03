// Bit-exactness spec for the carry-save accumulator back end: RTL container
// (Fp32Csa.mac / toF32) vs its software mirror (Fp32CsaModel) over random
// product chains, including window re-anchoring, deep underflow, zeros,
// Inf*0 and NaN operands, and cancellation to near zero.

package fp32matmul

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import xnpu.XnpuBackend

class Fp32CsaHarness extends Module {
  val io = IO(new Bundle {
    val doMac = Input(Bool())
    val fl = Input(new Fp32Fl)
    val pn = Input(UInt(48.W))
    val px = Input(SInt(12.W))
    val doClear = Input(Bool())
    val accIn = Input(new Fp32CsaAcc)
    val doLoad = Input(Bool())
    val acc = Output(new Fp32CsaAcc)
    val f32 = Output(UInt(32.W))
  })
  val accR = RegInit(Fp32Csa.cleared())
  when(io.doClear) { accR := Fp32Csa.cleared() }
    .elsewhen(io.doLoad) { accR := io.accIn }
    .elsewhen(io.doMac) { accR := Fp32Csa.mac(accR, io.fl, io.pn, io.px) }
  io.acc := accR
  io.f32 := Fp32Csa.toF32(accR)
}

class Fp32CsaSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  private var lcg = 0x243F6A8885A308D3L
  def rnd(): BigInt = {
    lcg = lcg * 6364136223846793005L + 1442695040888963407L
    (lcg >>> 16) & 0xffffffffffffL
  }

  behavior of "Fp32Csa carry-save accumulator"

  it should "drain the known t1 container exactly" in {
    test(new Fp32CsaHarness).withAnnotations(backendAnnos) { dut =>
      // s=0x0274d3ff c=0xfb165800 e=118 st=1: v = -40913665, p=25, eb0=239
      dut.io.doClear.poke(false.B); dut.io.doMac.poke(false.B)
      dut.io.doLoad.poke(true.B)
      dut.io.accIn.s.poke("h0274d3ff".U)
      dut.io.accIn.c.poke("hfb165800".U)
      dut.io.accIn.e.poke(118.S)
      dut.io.accIn.st.poke(true.B)
      dut.io.accIn.exc.poke(0.U)
      dut.io.accIn.exS.poke(false.B)
      dut.clock.step()
      dut.io.doLoad.poke(false.B)
      val m = Fp32CsaModel.Acc(
        BigInt("274d3ff", 16), BigInt("fb165800", 16), 118, true, 0, false, false)
      val want = Fp32CsaModel.toF32(m)
      val got = dut.io.f32.peek().litValue
      assert(got == want,
        f"fixed-container drain: toF32=0x$got%08x want 0x${want.toString(16)}")
    }
  }

  it should "match the model bit-for-bit over random product chains" in {
    test(new Fp32CsaHarness).withAnnotations(backendAnnos) { dut =>
      for (trial <- 0 until 300) {
        dut.io.doClear.poke(true.B); dut.io.doMac.poke(false.B)
        dut.clock.step()
        dut.io.doClear.poke(false.B)
        var m = Fp32CsaModel.cleared
        // chain length 1..16, exponent regime per trial (tight, wide, extreme)
        val regime = trial % 4
        val chain = 1 + (rnd() % 16).toInt
        for (k <- 0 until chain) {
          val kind = rnd() % 16
          val (ps, aN, bN, aI, bI, aZ, bZ, pn, px) =
            if (kind == 0) { // zero product (a or b zero)
              (false, false, false, false, false, true, false, BigInt(0), 0)
            } else if (kind == 1) { // NaN operand
              (false, true, false, false, false, false, false,
                BigInt(1) << 47, 20)
            } else if (kind == 2) { // Inf * 0 -> NaN
              (true, false, false, true, false, true, false, BigInt(0), 30)
            } else if (kind == 3) { // Inf operand
              (true, false, false, true, false, false, false,
                BigInt(1) << 47, -20)
            } else {
              // bf16-like product: <=16 significant bits, MSB pinned at 47
              val m16 = 0x8000L | (rnd() % 0x10000L)
              val pnB = BigInt(m16.toLong) << 32
              val pxI = regime match {
                case 0 => (rnd() % 24 - 12).toInt        // tight: +-12
                case 1 => (rnd() % 160 - 80).toInt       // wide: +-80
                case 2 => (rnd() % 400 - 200).toInt      // extreme: +-200
                case _ => 20 - (rnd() % 60).toInt        // downward drift
              }
              (rnd() % 2 == 0, false, false, false, false, false, false,
                pnB, pxI)
            }
          dut.io.doMac.poke(true.B)
          dut.io.fl.ps.poke(ps.B); dut.io.fl.aN.poke(aN.B); dut.io.fl.bN.poke(bN.B)
          dut.io.fl.aI.poke(aI.B); dut.io.fl.bI.poke(bI.B)
          dut.io.fl.aZ.poke(aZ.B); dut.io.fl.bZ.poke(bZ.B)
          dut.io.pn.poke(pn.U)
          dut.io.px.poke(px.S)
          dut.clock.step()
          m = Fp32CsaModel.mac(m, ps, aN, bN, aI, bI, aZ, bZ, pn, px)
          assert(dut.io.acc.s.peek().litValue == m.s, s"t$trial k$k S")
          assert(dut.io.acc.c.peek().litValue == m.c, s"t$trial k$k C")
          assert(dut.io.acc.e.peek().litValue == BigInt(m.e),
            s"t$trial k$k E ${dut.io.acc.e.peek().litValue} vs ${m.e}")
          assert(dut.io.acc.st.peek().litToBoolean == m.st, s"t$trial k$k sticky")
          assert(dut.io.acc.exc.peek().litValue == m.exc, s"t$trial k$k exc")
          assert(dut.io.acc.exS.peek().litToBoolean == m.exS, s"t$trial k$k exS")
        }
        dut.io.doMac.poke(false.B)
        val want = Fp32CsaModel.toF32(m)
        val got = dut.io.f32.peek().litValue
        assert(got == want,
          f"t$trial drain: got 0x$got%08x want 0x${want.toString(16)}")
      }
    }
  }
}
