// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

// Exercises the scalar control-flow front-end added to drive looped programs:
// the completed RV32I ALU, conditional branches with the EX-stage redirect +
// 2-slot pipeline flush, JAL/JALR linking, and AUIPC. Pure-scalar programs (no
// DTCM traffic), so the data port is just held ready.
class CtrlFlowSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  val p = Parameters()

  // Run a straight ITCM program (indexed by word) until the core halts, then
  // return the architectural register file for assertions.
  def run(program: Seq[BigInt], maxCycles: Int = 5000): IndexedSeq[BigInt] = {
    def instWord(byteAddr: BigInt): BigInt = {
      val idx = (byteAddr >> 2).toInt
      if (idx >= 0 && idx < program.length) program(idx) else BigInt(0)
    }
    var regs: IndexedSeq[BigInt] = IndexedSeq.fill(32)(BigInt(0))
    test(new Core(p)).withAnnotations(backendAnnos) { dut =>
      var instAddrPrev: BigInt = 0
      dut.io.dtcm.ready.poke(true.B)
      dut.io.dmaIdle.poke(true.B)   // no core-issued DMA here; engine always idle
      dut.clock.setTimeout(0)
      def tick(): Unit = {
        dut.io.itcmInst.poke(instWord(instAddrPrev).U)
        instAddrPrev = dut.io.itcmAddr.peek().litValue
        dut.clock.step(1)
      }
      dut.io.start.poke(true.B); tick(); dut.io.start.poke(false.B)
      var guard = 0
      while (!dut.io.done.peek().litToBoolean) {
        tick(); guard += 1; assert(guard < maxCycles, "core did not halt")
      }
      regs = (0 until 32).map(i => dut.io.dbg(i).peek().litValue)
    }
    regs
  }

  behavior of "Control flow"

  it should "run a backward-branch loop (sum 1..N)" in {
    val N = 10
    // x1=sum, x2=i, x3=limit; loop body at idx 3; blt back while i<limit.
    val program = Seq(
      Encode.addi(1, 0, 0),       // 0: sum = 0
      Encode.addi(2, 0, 1),       // 1: i = 1
      Encode.addi(3, 0, N + 1),   // 2: limit = N+1
      Encode.add(1, 1, 2),        // 3: sum += i        (loop:)
      Encode.addi(2, 2, 1),       // 4: i++
      Encode.blt(2, 3, -8),       // 5: if i < limit -> loop (idx 3)
      Encode.halt,                // 6
    )
    val regs = run(program)
    assert(regs(1) == BigInt(N * (N + 1) / 2), s"sum = ${regs(1)}")
    assert(regs(2) == BigInt(N + 1), s"i = ${regs(2)}")
  }

  it should "take/!take the six branch conditions correctly" in {
    // For each condition, branch over a 'poison' addi; x-reg stays 0 iff taken.
    // Layout per case: cmp setup once, then branch +8 over addi(rX,x0,1).
    val program = Seq(
      Encode.addi(1, 0, 5),       // 0: x1 = 5
      Encode.addi(2, 0, 7),       // 1: x2 = 7  (so 5<7 signed & unsigned)
      // BEQ not taken (5!=7): x10 set to 1
      Encode.beq(1, 2, 8),        // 2
      Encode.addi(10, 0, 1),      // 3  (executed)
      // BNE taken (5!=7): skip x11
      Encode.bne(1, 2, 8),        // 4
      Encode.addi(11, 0, 1),      // 5  (skipped)
      // BLT taken (5<7): skip x12
      Encode.blt(1, 2, 8),        // 6
      Encode.addi(12, 0, 1),      // 7  (skipped)
      // BGE not taken (5>=7 false): x13 set
      Encode.bge(1, 2, 8),        // 8
      Encode.addi(13, 0, 1),      // 9  (executed)
      Encode.halt,                // 10
    )
    val regs = run(program)
    assert(regs(10) == BigInt(1), "BEQ should not have been taken")
    assert(regs(11) == BigInt(0), "BNE should have been taken")
    assert(regs(12) == BigInt(0), "BLT should have been taken")
    assert(regs(13) == BigInt(1), "BGE should not have been taken")
  }

  it should "link and return through JAL/JALR" in {
    // call dbl(x10); dbl doubles x10 and returns via the link register x1.
    val program = Seq(
      Encode.addi(10, 0, 21),     // 0: a0 = 21
      Encode.jal(1, 12),          // 1: call dbl (idx 4); link x1 = 8
      Encode.addi(5, 10, 0),      // 2: x5 = a0   (return lands here)
      Encode.jal(0, 12),          // 3: goto end (idx 6)
      Encode.add(10, 10, 10),     // 4: dbl: a0 *= 2
      Encode.jalr(0, 1, 0),       // 5: return to x1
      Encode.addi(6, 0, 99),      // 6: end: x6 = 99
      Encode.halt,                // 7
    )
    val regs = run(program)
    assert(regs(10) == BigInt(42), s"a0 = ${regs(10)}")
    assert(regs(5) == BigInt(42), s"x5 = ${regs(5)}")
    assert(regs(6) == BigInt(99), s"x6 = ${regs(6)}")
    assert(regs(1) == BigInt(8), s"link = ${regs(1)}")
  }

  it should "compute PC-relative addresses with AUIPC" in {
    val program = Seq(
      Encode.auipc(7, 0),         // 0: x7 = pc(0) + 0      = 0
      Encode.auipc(8, 1),         // 1: x8 = pc(4) + 4096   = 4100
      Encode.halt,                // 2
    )
    val regs = run(program)
    assert(regs(7) == BigInt(0), s"x7 = ${regs(7)}")
    assert(regs(8) == BigInt(4100), s"x8 = ${regs(8)}")
  }

  it should "cover the new ALU ops (sub/shift/logic/compare)" in {
    val program = Seq(
      Encode.addi(1, 0, 100),     // x1 = 100
      Encode.addi(2, 0, 30),      // x2 = 30
      Encode.sub(3, 1, 2),        // x3 = 70
      Encode.slli(4, 1, 3),       // x4 = 100 << 3 = 800
      Encode.and(5, 1, 2),        // x5 = 100 & 30 = 4
      Encode.or(6, 1, 2),         // x6 = 126
      Encode.xor(7, 1, 2),        // x7 = 122
      Encode.slti(8, 2, 50),      // x8 = (30<50)=1
      Encode.sltu(9, 1, 2),       // x9 = (100<30)=0
      Encode.srli(11, 1, 2),      // x11 = 100>>2 = 25
      Encode.xori(12, 1, -1),     // x12 = ~100 = 0xFFFFFF9B
      Encode.halt,
    )
    val regs = run(program)
    assert(regs(3) == BigInt(70))
    assert(regs(4) == BigInt(800))
    assert(regs(5) == BigInt(100 & 30))
    assert(regs(6) == BigInt(100 | 30))
    assert(regs(7) == BigInt(100 ^ 30))
    assert(regs(8) == BigInt(1))
    assert(regs(9) == BigInt(0))
    assert(regs(11) == BigInt(25))
    assert(regs(12) == (BigInt(100) ^ ((BigInt(1) << 32) - 1)))
  }
}
