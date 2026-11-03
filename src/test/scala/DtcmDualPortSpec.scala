// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class DtcmDualPortSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  private val p = Parameters()

  behavior of "DtcmRam dual-port arbitration"

  it should "serve same-bank and cross-bank rows in parallel and serialize a same-row collision" in {
    test(new DtcmRam(p, rows = 16, bankRows = 2)).withAnnotations(backendAnnos) { dut =>
      def idle(): Unit = {
        dut.io.busy.poke(true.B)
        dut.io.matrix.valid.poke(false.B); dut.io.matrix.write.poke(false.B)
        dut.io.matrix.addr.poke(0.U); dut.io.matrix.wdata.poke(0.U); dut.io.matrix.wstrb.poke(0.U)
        dut.io.aux.valid.poke(false.B); dut.io.aux.write.poke(false.B)
        dut.io.aux.addr.poke(0.U); dut.io.aux.wdata.poke(0.U); dut.io.aux.wstrb.poke(0.U)
        dut.io.dmaActive.poke(false.B); dut.io.dma.valid.poke(false.B); dut.io.dma.write.poke(false.B)
        dut.io.dma.addr.poke(0.U); dut.io.dma.wdata.poke(0.U); dut.io.dma.wstrb.poke(0.U)
        dut.io.hostEn.poke(false.B); dut.io.hostWrite.poke(false.B); dut.io.hostAddr.poke(0.U)
        dut.io.hostWdata.poke(0.U); dut.io.hostWstrb.poke(0.U)
      }

      val first = BigInt("112233445566778899aabbccddeeff00", 16)
      val second = BigInt("0123456789abcdef0123456789abcdef", 16)
      val third = BigInt("fedcba9876543210fedcba9876543210", 16)
      val full = (BigInt(1) << p.DtcmRowBytes) - 1

      // Seed row 0 through Matrix port.
      idle()
      dut.io.matrix.valid.poke(true.B); dut.io.matrix.write.poke(true.B)
      dut.io.matrix.wdata.poke(first.U); dut.io.matrix.wstrb.poke(full.U)
      dut.io.matrix.ready.expect(true.B)
      dut.clock.step()

      // Matrix read of row 0 and Aux write of row 1 use both physical ports.
      idle()
      dut.io.matrix.valid.poke(true.B); dut.io.matrix.addr.poke(0.U)
      dut.io.aux.valid.poke(true.B); dut.io.aux.write.poke(true.B); dut.io.aux.addr.poke(16.U)
      dut.io.aux.wdata.poke(second.U); dut.io.aux.wstrb.poke(full.U)
      dut.io.matrix.ready.expect(true.B)
      dut.io.aux.ready.expect(true.B)
      dut.clock.step()
      dut.io.matrix.rdata.expect(first.U)

      // Row 8 resides in another bank for the 16-row test RAM. Its data must
      // be selected from that bank without changing Matrix port behaviour.
      idle()
      dut.io.matrix.valid.poke(true.B); dut.io.matrix.addr.poke(0.U)
      dut.io.aux.valid.poke(true.B); dut.io.aux.write.poke(true.B); dut.io.aux.addr.poke(128.U)
      dut.io.aux.wdata.poke(third.U); dut.io.aux.wstrb.poke(full.U)
      dut.io.matrix.ready.expect(true.B)
      dut.io.aux.ready.expect(true.B)
      dut.clock.step()
      dut.io.matrix.rdata.expect(first.U)

      // Both ports addressing row 0 are deterministic: Matrix proceeds; Aux is
      // held until the next cycle, rather than relying on RAM read/write mode.
      idle()
      dut.io.matrix.valid.poke(true.B); dut.io.matrix.addr.poke(0.U)
      dut.io.aux.valid.poke(true.B); dut.io.aux.write.poke(true.B); dut.io.aux.addr.poke(0.U)
      dut.io.aux.wdata.poke(second.U); dut.io.aux.wstrb.poke(full.U)
      dut.io.matrix.ready.expect(true.B)
      dut.io.aux.ready.expect(false.B)
      dut.clock.step()
      dut.io.matrix.rdata.expect(first.U)

      // Once Matrix releases the row, the retained Aux request is accepted.
      dut.io.matrix.valid.poke(false.B)
      dut.io.aux.ready.expect(true.B)
      dut.clock.step()

      idle()
      dut.io.aux.valid.poke(true.B); dut.io.aux.addr.poke(0.U)
      dut.clock.step()
      dut.io.aux.rdata.expect(second.U)

      // Read back the writes to a different row in the same bank and to row 8
      // in a different bank, verifying both bank address and data selection.
      idle()
      dut.io.aux.valid.poke(true.B); dut.io.aux.addr.poke(16.U)
      dut.clock.step()
      dut.io.aux.rdata.expect(second.U)

      idle()
      dut.io.aux.valid.poke(true.B); dut.io.aux.addr.poke(128.U)
      dut.clock.step()
      dut.io.aux.rdata.expect(third.U)
    }
  }

  // A MATMUL alternates between its A and B tiles on consecutive beats, and the
  // two tiles usually sit in different banks. Read data arrives a cycle after the
  // request, so the bank-select mux has to remember which bank issued the read;
  // selecting on the live address returns the neighbouring beat's bank instead.
  it should "return the issuing bank's data when consecutive reads alternate banks" in {
    test(new DtcmRam(p, rows = 16, bankRows = 2)).withAnnotations(backendAnnos) { dut =>
      val full = (BigInt(1) << p.DtcmRowBytes) - 1
      // Row r lives in bank r/2, so rows 0 and 8 are four banks apart.
      val rowA = 0
      val rowB = 8
      val dataA = BigInt("0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a", 16)
      val dataB = BigInt("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b", 16)

      def idle(): Unit = {
        dut.io.busy.poke(true.B)
        dut.io.matrix.valid.poke(false.B); dut.io.matrix.write.poke(false.B)
        dut.io.matrix.addr.poke(0.U); dut.io.matrix.wdata.poke(0.U); dut.io.matrix.wstrb.poke(0.U)
        dut.io.aux.valid.poke(false.B); dut.io.aux.write.poke(false.B)
        dut.io.aux.addr.poke(0.U); dut.io.aux.wdata.poke(0.U); dut.io.aux.wstrb.poke(0.U)
        dut.io.dmaActive.poke(false.B); dut.io.dma.valid.poke(false.B); dut.io.dma.write.poke(false.B)
        dut.io.dma.addr.poke(0.U); dut.io.dma.wdata.poke(0.U); dut.io.dma.wstrb.poke(0.U)
        dut.io.hostEn.poke(false.B); dut.io.hostWrite.poke(false.B); dut.io.hostAddr.poke(0.U)
        dut.io.hostWdata.poke(0.U); dut.io.hostWstrb.poke(0.U)
      }
      def matWrite(row: Int, data: BigInt): Unit = {
        idle()
        dut.io.matrix.valid.poke(true.B); dut.io.matrix.write.poke(true.B)
        dut.io.matrix.addr.poke((row * p.DtcmRowBytes).U)
        dut.io.matrix.wdata.poke(data.U); dut.io.matrix.wstrb.poke(full.U)
        dut.clock.step()
      }

      matWrite(rowA, dataA)
      matWrite(rowB, dataB)

      // Issue A, then B, then A on three consecutive cycles without ever letting
      // the port go idle, and sample each result on the following cycle.
      idle()
      dut.io.matrix.valid.poke(true.B); dut.io.matrix.addr.poke((rowA * p.DtcmRowBytes).U)
      dut.clock.step()
      dut.io.matrix.addr.poke((rowB * p.DtcmRowBytes).U)
      dut.io.matrix.rdata.expect(dataA.U, "beat 1 must return the row issued one cycle earlier")
      dut.clock.step()
      dut.io.matrix.addr.poke((rowA * p.DtcmRowBytes).U)
      dut.io.matrix.rdata.expect(dataB.U, "beat 2 crossed a bank boundary")
      dut.clock.step()
      dut.io.matrix.valid.poke(false.B)
      dut.io.matrix.rdata.expect(dataA.U, "beat 3 crossed back")

      // Repeat the same continuous sequence on port B. Vector/Mem and DMA use
      // this port on hardware, so testing only matBankReg would leave the board's
      // failing Vector path uncovered even though the two muxes look symmetric.
      idle()
      dut.io.aux.valid.poke(true.B); dut.io.aux.addr.poke((rowA * p.DtcmRowBytes).U)
      dut.clock.step()
      dut.io.aux.addr.poke((rowB * p.DtcmRowBytes).U)
      dut.io.aux.rdata.expect(dataA.U, "Aux beat 1 must return the row issued one cycle earlier")
      dut.clock.step()
      dut.io.aux.addr.poke((rowA * p.DtcmRowBytes).U)
      dut.io.aux.rdata.expect(dataB.U, "Aux beat 2 crossed a bank boundary")
      dut.clock.step()
      dut.io.aux.valid.poke(false.B)
      dut.io.aux.rdata.expect(dataA.U, "Aux beat 3 crossed back")
    }
  }

}
