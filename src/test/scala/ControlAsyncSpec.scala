// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

// Focused Control-level regression for the asynchronous Vector GPR contract.
// The Vector engine is represented by a delayed direct writeback so this test
// covers scoreboard behaviour without elaborating the systolic array.
class ControlAsyncSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  behavior of "Control Vector scoreboard"

  it should "run independent scalar instructions while holding a dependent GPR consumer" in {
    val program = Seq(
      Encode.vsum(5),
      Encode.addi(7, 0, 85),
      Encode.addi(8, 7, 1),
      Encode.addi(6, 5, 2),
      Encode.halt,
    )

    test(new Control(Parameters())).withAnnotations(backendAnnos) { dut =>
      val regs = Array.fill[BigInt](32)(BigInt(0))
      var instAddrPrev = BigInt(0)
      var vectorDelay = -1
      var independentCommit = -1
      var vectorCommit = -1

      dut.io.matrixReady.poke(true.B)
      dut.io.vectorReady.poke(true.B)
      dut.io.matrixIdle.poke(true.B)
      dut.io.vectorIdle.poke(true.B)
      dut.io.vectorDone.poke(false.B)
      dut.io.memIdle.poke(true.B)
      dut.io.memDone.poke(false.B)
      dut.io.memReady.poke(true.B)   // V1: scalar-mem dispatch gate (no mem ops here)
      dut.io.dmaDrained.poke(true.B)
      dut.io.req.ready.poke(true.B)
      dut.io.rd.valid.poke(false.B)
      dut.io.rd.bits.addr.poke(0.U)
      dut.io.rd.bits.data.poke(0.U)

      def instruction(addr: BigInt): BigInt = {
        val index = (addr >> 2).toInt
        if (index >= 0 && index < program.length) program(index) else BigInt(0)
      }

      def tick(): Unit = {
        dut.io.itcmInst.poke(instruction(instAddrPrev).U)
        instAddrPrev = dut.io.itcmAddr.peek().litValue
        dut.io.rs1Data.poke(regs(dut.io.rs1Addr.peek().litValue.toInt).U)
        dut.io.rs2Data.poke(regs(dut.io.rs2Addr.peek().litValue.toInt).U)

        val vectorValid = vectorDelay === 0
        dut.io.vectorWrite.valid.poke(vectorValid.B)
        dut.io.vectorWrite.bits.addr.poke(5.U)
        dut.io.vectorWrite.bits.data.poke(40.U)

        // Control consumes Alu.rd.bits.data as a combinational scalar result
        // even when req.valid is low.  This program contains only ADDI scalar
        // instructions, so model that narrow part of the Alu contract here.
        val op = dut.io.req.bits.op.peek().litValue
        val scalarData = if (op == AluOp.ADDI.litValue) {
          (dut.io.aluRs1.peek().litValue + dut.io.req.bits.imm.peek().litValue) & 0xffffffffL
        } else BigInt(0)
        dut.io.rd.bits.data.poke(scalarData.U)

        val reqFire = dut.io.req.valid.peek().litToBoolean &&
                      dut.io.req.ready.peek().litToBoolean &&
                      dut.io.req.bits.addr.peek().litValue == 5
        val vectorFire = vectorValid && dut.io.vectorWrite.ready.peek().litToBoolean
        val writeFire = dut.io.wrValid.peek().litToBoolean
        val writeAddr = dut.io.wrAddr.peek().litValue.toInt
        val writeData = dut.io.wrData.peek().litValue

        dut.clock.step()

        if (writeFire && writeAddr != 0) regs(writeAddr) = writeData
        if (reqFire) vectorDelay = 6
        else if (vectorDelay > 0) vectorDelay -= 1
        if (vectorFire) vectorDelay = -1
      }

      dut.io.start.poke(true.B)
      tick()
      dut.io.start.poke(false.B)

      var cycle = 0
      while (!dut.io.done.peek().litToBoolean) {
        tick()
        cycle += 1
        if (independentCommit < 0 && regs(7) == 85) independentCommit = cycle
        if (vectorCommit < 0 && regs(5) == 40) vectorCommit = cycle
        assert(cycle < 100, "Control did not retire the test program")
      }

      assert(regs(8) == 86, s"independent scalar chain wrote ${regs(8)}")
      assert(regs(6) == 42, s"dependent scalar observed ${regs(6)} instead of vector result")
      assert(independentCommit >= 0 && vectorCommit >= 0 && independentCommit < vectorCommit,
        s"independent scalar committed at $independentCommit, vector at $vectorCommit")
    }
  }

  // V1 note: scalar LW/SW no longer wait for a full vector drain (that gate is
  // gone); what this program still exercises is the GPR scoreboard -- the SW's
  // rs2 (the reduction's rd) must hold in EX until the asynchronous vector
  // writeback forwards, so the store never captures the stale pre-reduction
  // value even though the store may now dispatch while the vector unit runs.
  it should "retain a Vector reduction result while a dependent store waits for the vector GPR writeback" in {
    val stale = 85
    val reduction = BigInt("41000000", 16)
    val program = Seq(
      Encode.addi(5, 0, stale),
      Encode.addi(10, 0, 64),
      Encode.vsum(5),
      Encode.sw(5, 10, 0),
      Encode.halt,
    )

    test(new Control(Parameters())).withAnnotations(backendAnnos) { dut =>
      val regs = Array.fill[BigInt](32)(BigInt(0))
      var instAddrPrev = BigInt(0)
      var vectorDelay = -1
      var memDelay = -1
      var stored = Option.empty[BigInt]

      dut.io.matrixReady.poke(true.B)
      dut.io.vectorReady.poke(true.B)
      dut.io.matrixIdle.poke(true.B)
      dut.io.vectorDone.poke(false.B)
      dut.io.memReady.poke(true.B)   // V1: scalar-mem dispatch gate
      dut.io.dmaDrained.poke(true.B)
      dut.io.req.ready.poke(true.B)
      dut.io.rd.valid.poke(false.B)
      dut.io.rd.bits.addr.poke(0.U)
      dut.io.rd.bits.data.poke(0.U)

      def instruction(addr: BigInt): BigInt = {
        val index = (addr >> 2).toInt
        if (index >= 0 && index < program.length) program(index) else BigInt(0)
      }

      def tick(): Unit = {
        dut.io.itcmInst.poke(instruction(instAddrPrev).U)
        instAddrPrev = dut.io.itcmAddr.peek().litValue
        dut.io.rs1Data.poke(regs(dut.io.rs1Addr.peek().litValue.toInt).U)
        dut.io.rs2Data.poke(regs(dut.io.rs2Addr.peek().litValue.toInt).U)

        val vectorValid = vectorDelay == 0
        dut.io.vectorIdle.poke((vectorDelay < 0).B)
        dut.io.vectorWrite.valid.poke(vectorValid.B)
        dut.io.vectorWrite.bits.addr.poke(5.U)
        dut.io.vectorWrite.bits.data.poke(reduction.U)

        val memDone = memDelay == 0
        dut.io.memIdle.poke((memDelay < 0).B)
        dut.io.memDone.poke(memDone.B)

        val op = dut.io.req.bits.op.peek().litValue
        val scalarData = if (op == AluOp.ADDI.litValue) {
          (dut.io.aluRs1.peek().litValue + dut.io.req.bits.imm.peek().litValue) & 0xffffffffL
        } else BigInt(0)
        dut.io.rd.bits.data.poke(scalarData.U)

        val reqFire = dut.io.req.valid.peek().litToBoolean &&
                      dut.io.req.ready.peek().litToBoolean
        val vectorFire = vectorValid && dut.io.vectorWrite.ready.peek().litToBoolean
        val writeFire = dut.io.wrValid.peek().litToBoolean
        val writeAddr = dut.io.wrAddr.peek().litValue.toInt
        val writeData = dut.io.wrData.peek().litValue
        val reqOp = dut.io.req.bits.op.peek().litValue
        val storeData = dut.io.aluRs2.peek().litValue

        dut.clock.step()

        if (writeFire && writeAddr != 0) regs(writeAddr) = writeData
        if (reqFire && reqOp == AluOp.VSUM.litValue) vectorDelay = 5
        else if (vectorDelay > 0) vectorDelay -= 1
        if (vectorFire) vectorDelay = -1

        if (reqFire && reqOp == AluOp.SW.litValue) {
          stored = Some(storeData)
          memDelay = 2
        } else if (memDelay > 0) {
          memDelay -= 1
        } else if (memDone) {
          memDelay = -1
        }
      }

      dut.io.start.poke(true.B)
      tick()
      dut.io.start.poke(false.B)

      var cycles = 0
      while (!dut.io.done.peek().litToBoolean) {
        tick()
        cycles += 1
        assert(cycles < 100, "Control did not retire the reduction-store program")
      }

      assert(regs(5) == reduction, f"Vector writeback was 0x${regs(5)}%08x")
      assert(stored.contains(reduction),
        f"dependent SW captured 0x${stored.getOrElse(BigInt(-1)) & 0xffffffffL}%08x instead of 0x$reduction%08x")
    }
  }
}
