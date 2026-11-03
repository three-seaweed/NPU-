// Copyright 2026 BUPT

package xnpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable

class DtcmTokenSpec extends AnyFlatSpec with ChiselScalatestTester with XnpuBackend {
  // FP.* token 行数来自 bf16 8x8 tier（fp32 4x4 已删）；本 spec 的 FP 用例
  // 需要活着的 FP 层，其余 int8 用例不受影响。
  private val p = Parameters(bf16Matmul = true)

  behavior of "Alu DTCM dependency tokens"

  it should "hold a Matrix reader behind an earlier queued Vector producer" in {
    test(new Alu(p)).withAnnotations(backendAnnos) { dut =>
      dut.io.req.valid.poke(false.B)
      dut.io.req.bits.op.poke(AluOp.ADDI)
      dut.io.req.bits.addr.poke(0.U)
      dut.io.req.bits.imm.poke(0.U)
      dut.io.rs1.valid.poke(true.B); dut.io.rs2.valid.poke(true.B)
      dut.io.rs1.data.poke(0.U); dut.io.rs2.data.poke(0.U)
      dut.io.rd.ready.poke(true.B); dut.io.vectorWrite.ready.poke(true.B)
      dut.io.dtcmMatrix.ready.poke(true.B); dut.io.dtcmMatrix.rdata.poke(0.U)
      dut.io.dtcmAux.ready.poke(true.B); dut.io.dtcmAux.rdata.poke(0.U)
      dut.io.dmaIdle.poke(true.B)

      def tick(): Unit = dut.clock.step()
      def issue(op: AluOp.Type, rs1: Int = 0, rs2: Int = 0): Unit = {
        dut.io.rs1.data.poke(rs1.U)
        dut.io.rs2.data.poke(rs2.U)
        dut.io.req.bits.op.poke(op)
        dut.io.req.bits.addr.poke(0.U)
        dut.io.req.bits.imm.poke(0.U)
        dut.io.req.valid.poke(true.B)
        var wait = 0
        while (!dut.io.req.ready.peek().litToBoolean) {
          tick(); wait += 1; assert(wait < 100, s"$op could not enqueue")
        }
        tick()
        dut.io.req.valid.poke(false.B)
      }

      // Vector writes row 0. The later MATMUL reads A from row 0; token creation
      // happens at enqueue, so it must wait even if the Vector command has not
      // reached VectorUnit yet.
      issue(AluOp.VSETA, 0x000)
      issue(AluOp.VSETB, 0x100)
      issue(AluOp.VSETD, 0x000)
      issue(AluOp.VSETLEN, p.Bf16LanesPerRow)
      issue(AluOp.VADD)
      issue(AluOp.LMA, 0x000)
      issue(AluOp.LMB, 0x100)
      issue(AluOp.MATMUL)

      var vectorStoreSeen = false
      var matrixReadBeforeStore = false
      var matrixReadAfterStore = false
      // A blocked command must not be latched by the unit while it still sits in
      // the queue, because it would then run a second time once the queue does
      // pop it. Counting operand-load bursts catches that: one MATMUL is exactly
      // one contiguous run of Matrix reads.
      var matrixReadBursts = 0
      var wasReading = false
      // Attribution outputs (PERF_DEP_M/V source): the blocked queue here is
      // Matrix -- Vector must never be blamed in this scenario.
      var depWaitMObserved = false
      var depWaitVObserved = false
      // Long enough to cover the full 3N-2 feed plus a hypothetical replay.
      for (_ <- 0 until 220) {
        val vectorStore = dut.io.dtcmAux.valid.peek().litToBoolean &&
          dut.io.dtcmAux.write.peek().litToBoolean
        val matrixRead = dut.io.dtcmMatrix.valid.peek().litToBoolean &&
          !dut.io.dtcmMatrix.write.peek().litToBoolean
        if (matrixRead && !wasReading) matrixReadBursts += 1
        wasReading = matrixRead
        if (matrixRead && !vectorStoreSeen) matrixReadBeforeStore = true
        if (vectorStore) vectorStoreSeen = true
        if (matrixRead && vectorStoreSeen) matrixReadAfterStore = true
        if (dut.io.depWaitM.peek().litToBoolean) depWaitMObserved = true
        if (dut.io.depWaitV.peek().litToBoolean) depWaitVObserved = true
        tick()
      }

      assert(vectorStoreSeen, "Vector producer never wrote its declared output range")
      assert(!matrixReadBeforeStore, "Matrix consumer read before the Vector producer token retired")
      assert(matrixReadAfterStore, "Matrix reader did not start after the Vector producer completed")
      assert(matrixReadBursts == 1,
        s"the single MATMUL streamed its operands $matrixReadBursts times")
      assert(depWaitMObserved,
        "the blocked Matrix head never raised depWaitM (attribution dead?)")
      assert(!depWaitVObserved,
        "depWaitV asserted while the Vector queue was never blocked -- misattribution")
    }
  }

  // ---- fp32 侧扩展（handbook §3.4）------------------------------------------

  // fp32 的 A 区间只有 4 行。第二个 VADD 把 row8 写一遍：正确实现申报的
  // [0..3) 不含 row8，FP.MATMUL 的取数可以在 row8 写完之前开始；错报 16 行
  // 的实现会被 row8 的写 token 挡住，永远等它完成 —— 用这个差异锁死 rows mux。
  it should "let FP.MATMUL read A[0..3) while a Vector write to row 8 is still in flight" in {
    test(new Alu(p)).withAnnotations(backendAnnos) { dut =>
      dut.io.req.valid.poke(false.B)
      dut.io.req.bits.op.poke(AluOp.ADDI)
      dut.io.req.bits.addr.poke(0.U)
      dut.io.req.bits.imm.poke(0.U)
      dut.io.rs1.valid.poke(true.B); dut.io.rs2.valid.poke(true.B)
      dut.io.rs1.data.poke(0.U); dut.io.rs2.data.poke(0.U)
      dut.io.rd.ready.poke(true.B); dut.io.vectorWrite.ready.poke(true.B)
      dut.io.dtcmMatrix.ready.poke(true.B); dut.io.dtcmMatrix.rdata.poke(0.U)
      dut.io.dtcmAux.ready.poke(true.B); dut.io.dtcmAux.rdata.poke(0.U)
      dut.io.dmaIdle.poke(true.B)

      var row0StoreSeen = false
      var row8StoreSeen = false
      var matrixReadBeforeRow8 = false
      var matrixReadBeforeRow0 = false
      var matrixReadBursts = 0
      var wasReading = false
      // 采样在 tick 里：issue 的等待拍和 fire 拍也要数，否则命令前几拍会落在
      // 采样窗口之外（case c 的 38≠40 教训）。
      def tick(): Unit = {
        val addr = dut.io.dtcmAux.addr.peek().litValue
        val vectorStore = dut.io.dtcmAux.valid.peek().litToBoolean &&
          dut.io.dtcmAux.write.peek().litToBoolean
        val matrixRead = dut.io.dtcmMatrix.valid.peek().litToBoolean &&
          !dut.io.dtcmMatrix.write.peek().litToBoolean
        if (vectorStore && addr == 0x000) row0StoreSeen = true
        if (vectorStore && addr == 0x080) row8StoreSeen = true
        if (matrixRead && !wasReading) matrixReadBursts += 1
        wasReading = matrixRead
        if (matrixRead && !row0StoreSeen) matrixReadBeforeRow0 = true
        if (matrixRead && !row8StoreSeen) matrixReadBeforeRow8 = true
        dut.clock.step()
      }
      def issue(op: AluOp.Type, rs1: Int = 0, rs2: Int = 0): Unit = {
        dut.io.rs1.data.poke(rs1.U)
        dut.io.rs2.data.poke(rs2.U)
        dut.io.req.bits.op.poke(op)
        dut.io.req.bits.addr.poke(0.U)
        dut.io.req.bits.imm.poke(0.U)
        dut.io.req.valid.poke(true.B)
        var wait = 0
        while (!dut.io.req.ready.peek().litToBoolean) {
          tick(); wait += 1; assert(wait < 100, s"$op could not enqueue")
        }
        tick()
        dut.io.req.valid.poke(false.B)
      }

      // Vector 写 row0（FP.MATMUL 的 A 生产者），随后写 row8（假 16x16 行数
      // 申报的陷阱行，真 8x8 的 8 行区间之外）。
      issue(AluOp.VSETA, 0x000)
      issue(AluOp.VSETB, 0x100)
      issue(AluOp.VSETD, 0x000)
      issue(AluOp.VSETLEN, p.Bf16LanesPerRow)
      issue(AluOp.VADD)
      issue(AluOp.VSETD, 0x080)     // row8 = byte 0x080
      issue(AluOp.VADD)
      issue(AluOp.LMA, 0x000)
      issue(AluOp.LMB, 0x100)
      issue(AluOp.FP_MATMUL)

      for (_ <- 0 until 300) {
        tick()
      }

      assert(row0StoreSeen && row8StoreSeen, "Vector producers never wrote both rows")
      assert(!matrixReadBeforeRow0, "FP.MATMUL read A before its row-0 producer retired")
      // 判别核心：8 行申报下，取数不必等 row8 的写。若申报成 16x16 的 32 行，这里必挂。
      assert(matrixReadBeforeRow8,
        "FP.MATMUL never overlapped the row-8 Vector write: rows mux looks wrong (16-row claim?)")
      assert(matrixReadBursts == 1,
        s"the single FP.MATMUL streamed its operands $matrixReadBursts times")
    }
  }

  // DMAST 读 C 区（DMA token），随后 FP.SM 写同区：WAR 顺序。DMA 是 EX 直发
  // （无队列），dmaStart 单拍在 issue 当拍；描述符携带自己的 token 槽号，
  // 引擎的完成脉冲把它带回来后才退休。压住 dmaDone 把读 token 扣住，FP.SM
  // 必须零写拍；用起发时捕获的槽号回一个完成脉冲后恰好 4 拍写完。
  it should "hold FP.SM's C-region writes behind a live DMAST read token" in {
    test(new Alu(p)).withAnnotations(backendAnnos) { dut =>
      dut.io.req.valid.poke(false.B)
      dut.io.req.bits.op.poke(AluOp.ADDI)
      dut.io.req.bits.addr.poke(0.U)
      dut.io.req.bits.imm.poke(0.U)
      dut.io.rs1.valid.poke(true.B); dut.io.rs2.valid.poke(true.B)
      dut.io.rs1.data.poke(0.U); dut.io.rs2.data.poke(0.U)
      dut.io.rd.ready.poke(true.B); dut.io.vectorWrite.ready.poke(true.B)
      dut.io.dtcmMatrix.ready.poke(true.B); dut.io.dtcmMatrix.rdata.poke(0.U)
      dut.io.dtcmAux.ready.poke(true.B); dut.io.dtcmAux.rdata.poke(0.U)
      dut.io.dmaIdle.poke(true.B)

      var dmaStarted = false
      var tokEn = false
      var tokId = 0
      def tick(): Unit = {
        if (dut.io.dmaStart.peek().litToBoolean) {
          dmaStarted = true
          tokEn = dut.io.dmaTok.en.peek().litToBoolean
          tokId = dut.io.dmaTok.id.peek().litValue.toInt
        }
        dut.clock.step()
      }
      def issue(op: AluOp.Type, rs1: Int = 0, rs2: Int = 0, imm: Int = 0): Unit = {
        dut.io.rs1.data.poke(rs1.U)
        dut.io.rs2.data.poke(rs2.U)
        dut.io.req.bits.op.poke(op)
        dut.io.req.bits.addr.poke(0.U)
        dut.io.req.bits.imm.poke(imm.U)
        dut.io.req.valid.poke(true.B)
        var wait = 0
        while (!dut.io.req.ready.peek().litToBoolean) {
          tick(); wait += 1; assert(wait < 100, s"$op could not enqueue")
        }
        tick()
        dut.io.req.valid.poke(false.B)
      }

      // DMAST: 读 DTCM row0..3（rs2=0x000, rows=4）搬去 DRAM，start 当拍占
      // 读 token；没有完成脉冲它就不退休，FP.SM 排进矩阵队列等待。
      issue(AluOp.DMAST, rs1 = 0x8000, rs2 = 0x000, imm = 4)
      issue(AluOp.FP_SM, rs1 = 0x000)

      var heldWrites = 0
      for (_ <- 0 until 30) {
        val w = dut.io.dtcmMatrix.valid.peek().litToBoolean &&
          dut.io.dtcmMatrix.write.peek().litToBoolean
        if (w) heldWrites += 1
        tick()
      }
      // 引擎按描述符携带的槽号回报完成：读 token 退休，FP.SM 放行。
      assert(tokEn, "DMAST start did not declare a token lease")
      dut.io.dmaDone.poke(true.B)
      dut.io.dmaDoneTok.en.poke(tokEn.B)
      dut.io.dmaDoneTok.id.poke(tokId.U)
      tick()
      dut.io.dmaDone.poke(false.B)
      var releasedWrites = 0
      for (_ <- 0 until 100) {
        val w = dut.io.dtcmMatrix.valid.peek().litToBoolean &&
          dut.io.dtcmMatrix.write.peek().litToBoolean
        if (w) releasedWrites += 1
        tick()
      }

      assert(dmaStarted, "DMAST never issued")
      assert(heldWrites == 0,
        s"FP.SM wrote $heldWrites beats while the DMAST read token was still live")
      assert(releasedWrites == 8,
        s"FP.SM stored $releasedWrites beats after the token retired, expected 8 (bf16 8x8)")
    }
  }

  // fp32 与 int8 交错（MATMUL → FP.MATMUL → SM → FP.SM）：全部完成且取数/
  // 回写拍数精确 —— 每条命令恰执行一次，无队列重放。
  it should "interleave int8 and fp32 matrix commands without replay" in {
    test(new Alu(p)).withAnnotations(backendAnnos) { dut =>
      dut.io.req.valid.poke(false.B)
      dut.io.req.bits.op.poke(AluOp.ADDI)
      dut.io.req.bits.addr.poke(0.U)
      dut.io.req.bits.imm.poke(0.U)
      dut.io.rs1.valid.poke(true.B); dut.io.rs2.valid.poke(true.B)
      dut.io.rs1.data.poke(0.U); dut.io.rs2.data.poke(0.U)
      dut.io.rd.ready.poke(true.B); dut.io.vectorWrite.ready.poke(true.B)
      dut.io.dtcmMatrix.ready.poke(true.B); dut.io.dtcmMatrix.rdata.poke(0.U)
      dut.io.dtcmAux.ready.poke(true.B); dut.io.dtcmAux.rdata.poke(0.U)
      dut.io.dmaIdle.poke(true.B)

      var readBeats = 0; var writeBeats = 0
      var readBursts = 0; var writeBursts = 0
      var wasReading = false; var wasWriting = false
      val readAddrs = mutable.ArrayBuffer[String]()
      // 采样进 tick：issue 的等待拍和 fire 拍也要数（前两拍加载落在 issue
      // 窗口里就是 38≠40 的教训）。
      def tick(): Unit = {
        val v = dut.io.dtcmMatrix.valid.peek().litToBoolean
        val w = dut.io.dtcmMatrix.write.peek().litToBoolean
        val a = dut.io.dtcmMatrix.addr.peek().litValue
        if (v && !w) { readBeats += 1; readAddrs += f"0x$a%03x"; if (!wasReading) readBursts += 1 }
        if (v && w)  { writeBeats += 1; if (!wasWriting) writeBursts += 1 }
        wasReading = v && !w
        wasWriting = v && w
        dut.clock.step()
      }
      def issue(op: AluOp.Type, rs1: Int = 0): Unit = {
        dut.io.rs1.data.poke(rs1.U)
        dut.io.req.bits.op.poke(op)
        dut.io.req.bits.addr.poke(0.U)
        dut.io.req.bits.imm.poke(0.U)
        dut.io.req.valid.poke(true.B)
        var wait = 0
        while (!dut.io.req.ready.peek().litToBoolean) {
          tick(); wait += 1; assert(wait < 100, s"$op could not enqueue")
        }
        tick()
        dut.io.req.valid.poke(false.B)
      }

      issue(AluOp.LMA, 0x000)
      issue(AluOp.LMB, 0x100)
      issue(AluOp.MATMUL)          // 读 32 拍（16+16 行）
      issue(AluOp.FP_MATMUL)       // 读 16 拍（8+8 行）
      issue(AluOp.SM, 0x000)       // 写 64 拍（int32 16x16）
      issue(AluOp.FP_SM, 0x000)    // 写 8 拍（bf16 8x8）

      for (_ <- 0 until 400) {
        tick()
      }

      // 精确拍数：既证明每条命令都跑了，也证明没有一条跑了两遍。
      assert(readBeats == 32 + 16,
        s"matrix read beats $readBeats, expected 32 (int8) + 16 (bf16); addrs: ${readAddrs.mkString(",")}")
      assert(writeBeats == 64 + 8, s"matrix write beats $writeBeats, expected 64 (SM) + 8 (FP.SM)")
      // C5 起跨类命令可并发在飞：int8 与 fp32 的读流/写流经 port0 仲裁可
      // 以背靠背连成一串（burst 计数的 wasReading 启发式分不开两台引擎）。
      // 每条命令恰执行一次由上面的精确 beats 断言承担；这里只锁总量级。
      assert(readBursts == 1 || readBursts == 2,
        s"read bursts $readBursts, expected 1-2 (concurrent classes may merge)")
      assert(writeBursts == 1 || writeBursts == 2,
        s"write bursts $writeBursts, expected 1-2 (concurrent classes may merge)")
    }
  }

  // ---- 就绪扫描（Phase B）：被 token 挡住的队头不拖累后面的独立命令 ----------
  //
  // 向量生产者写 row0 → int8 MATMUL 读 A[0..16) 被 RAW 挡住 → 随后的
  // FP.MATMUL 读 A[0x40..)/B[0x50..) 与 row0 不相交、无依赖。就绪扫描必须把
  // FP.MATMUL 提升过被挡的队头（其基址来自入队时的 context 快照，与 LMA 的
  // 执行顺序无关）；int8 MATMUL 仍必须等 row0 写完才开始读。
  it should "promote a ready FP.MATMUL past a token-blocked int8 MATMUL head" in {
    test(new Alu(p)).withAnnotations(backendAnnos) { dut =>
      dut.io.req.valid.poke(false.B)
      dut.io.req.bits.op.poke(AluOp.ADDI)
      dut.io.req.bits.addr.poke(0.U)
      dut.io.req.bits.imm.poke(0.U)
      dut.io.rs1.valid.poke(true.B); dut.io.rs2.valid.poke(true.B)
      dut.io.rs1.data.poke(0.U); dut.io.rs2.data.poke(0.U)
      dut.io.rd.ready.poke(true.B); dut.io.vectorWrite.ready.poke(true.B)
      dut.io.dtcmMatrix.ready.poke(true.B); dut.io.dtcmMatrix.rdata.poke(0.U)
      dut.io.dtcmAux.ready.poke(true.B); dut.io.dtcmAux.rdata.poke(0.U)
      dut.io.dmaIdle.poke(true.B)

      var readBeats = 0
      var cycle = 0
      var fp32LastReadCycle = -1
      var int8FirstReadCycle = -1
      var vectorStoreSeen = false
      var int8ReadAfterStore = false
      def tick(): Unit = {
        val v = dut.io.dtcmMatrix.valid.peek().litToBoolean
        val w = dut.io.dtcmMatrix.write.peek().litToBoolean
        val a = dut.io.dtcmMatrix.addr.peek().litValue
        val vs = dut.io.dtcmAux.valid.peek().litToBoolean &&
          dut.io.dtcmAux.write.peek().litToBoolean
        if (vs) vectorStoreSeen = true
        cycle += 1
        if (v && !w) {
          readBeats += 1
          // fp32 tile lives at 0x400/0x500; int8 MATMUL at 0x000/0x100
          if (a >= 0x400) fp32LastReadCycle = cycle
          else {
            if (int8FirstReadCycle < 0) int8FirstReadCycle = cycle
            if (vectorStoreSeen) int8ReadAfterStore = true
          }
        }
        dut.clock.step()
      }
      def issue(op: AluOp.Type, rs1: Int = 0): Unit = {
        dut.io.rs1.data.poke(rs1.U)
        dut.io.req.bits.op.poke(op)
        dut.io.req.bits.addr.poke(0.U)
        dut.io.req.bits.imm.poke(0.U)
        dut.io.req.valid.poke(true.B)
        var wait = 0
        while (!dut.io.req.ready.peek().litToBoolean) {
          tick(); wait += 1; assert(wait < 100, s"$op could not enqueue")
        }
        tick()
        dut.io.req.valid.poke(false.B)
      }

      // 生产者：VADD 写 32 行（拉长执行，确保后面三条命令在其退休前入队）
      issue(AluOp.VSETA, 0x000)
      issue(AluOp.VSETB, 0x100)
      issue(AluOp.VSETD, 0x000)
      issue(AluOp.VSETLEN, 0x100)
      issue(AluOp.VADD)
      // 被挡的 int8 MATMUL（A 区间 [0x00..0x10) 盖在生产者写区间内）
      issue(AluOp.LMA, 0x000)
      issue(AluOp.LMB, 0x100)
      issue(AluOp.MATMUL)
      // 无依赖的 FP.MATMUL：就绪扫描必须先发它
      issue(AluOp.LMA, 0x400)
      issue(AluOp.LMB, 0x500)
      issue(AluOp.FP_MATMUL)

      for (_ <- 0 until 400) { tick() }

      assert(readBeats == 32 + 16, s"read beats $readBeats, expected 32 (int8) + 16 (bf16)")
      // 判别核心：bf16 的全部读必须发生在 int8 首次读之前 —— 顺序反转即提升
      // 的直接证据（in-order 队列必然 int8 在前）。与 VADD 的首行写无关，无竞态。
      assert(fp32LastReadCycle > 0,
        "the FP.MATMUL never read its operands at all")
      assert(int8FirstReadCycle > fp32LastReadCycle,
        s"no promotion: fp32 last read cycle $fp32LastReadCycle, int8 first read $int8FirstReadCycle")
      assert(int8ReadAfterStore,
        "the int8 MATMUL read before the RAW producer retired -- dependency violated")
    }
  }

  // ---- 累加链不得被扫描换序（2026-09-24 板上 G2 P2 k4 失败的回归锁）----------
  //
  // FP.MATMUL(k0) 与 FP.MATMULA(k1) 的 k-tile 区间不相交 → token 表没有边；
  // 累加顺序只由发射顺序保证，而浮点加法不可交换。slot 出现空洞后，扫描若
  // 允许同引擎类 worker 互相越过，MATMULA 会先于 MATMUL 执行 → 位级错。
  // 正确行为：fp32 类 worker 不得越过下方未发射的 fp32 类 worker。
  it should "keep an FP.MATMULA behind an unissued same-class FP.MATMUL" in {
    test(new Alu(p)).withAnnotations(backendAnnos) { dut =>
      dut.io.req.valid.poke(false.B)
      dut.io.req.bits.op.poke(AluOp.ADDI)
      dut.io.req.bits.addr.poke(0.U)
      dut.io.req.bits.imm.poke(0.U)
      dut.io.rs1.valid.poke(true.B); dut.io.rs2.valid.poke(true.B)
      dut.io.rs1.data.poke(0.U); dut.io.rs2.data.poke(0.U)
      dut.io.rd.ready.poke(true.B); dut.io.vectorWrite.ready.poke(true.B)
      dut.io.dtcmMatrix.ready.poke(true.B); dut.io.dtcmMatrix.rdata.poke(0.U)
      dut.io.dtcmAux.ready.poke(true.B); dut.io.dtcmAux.rdata.poke(0.U)
      dut.io.dmaIdle.poke(true.B)

      var firstFpReadAddr = -1
      def tick(): Unit = {
        val v = dut.io.dtcmMatrix.valid.peek().litToBoolean
        val w = dut.io.dtcmMatrix.write.peek().litToBoolean
        if (v && !w) {
          val a = dut.io.dtcmMatrix.addr.peek().litValue.toInt
          if (firstFpReadAddr < 0) firstFpReadAddr = a
        }
        dut.clock.step()
      }
      def issue(op: AluOp.Type, rs1: Int = 0): Unit = {
        dut.io.rs1.data.poke(rs1.U)
        dut.io.req.bits.op.poke(op)
        dut.io.req.bits.addr.poke(0.U)
        dut.io.req.bits.imm.poke(0.U)
        dut.io.req.valid.poke(true.B)
        var wait = 0
        while (!dut.io.req.ready.peek().litToBoolean) {
          tick(); wait += 1; assert(wait < 100, s"$op could not enqueue")
        }
        tick()
        dut.io.req.valid.poke(false.B)
      }

      // 生产者挡住 FP.MATMUL(k0)；FP.MATMULA(k1) 区间不相交、无 token 边
      issue(AluOp.VSETA, 0x000)
      issue(AluOp.VSETB, 0x100)
      issue(AluOp.VSETD, 0x000)
      issue(AluOp.VSETLEN, p.Bf16LanesPerRow)
      issue(AluOp.VADD)
      issue(AluOp.LMA, 0x000)
      issue(AluOp.LMB, 0x100)
      issue(AluOp.FP_MATMUL)
      issue(AluOp.LMA, 0x400)
      issue(AluOp.LMB, 0x500)
      issue(AluOp.FP_MATMULA)

      for (_ <- 0 until 300) { tick() }

      // 累加序锁：fp32 的首次读必须是 MATMUL 的 k0（0x000 区间），不是 k1。
      assert(firstFpReadAddr == 0x000,
        s"fp32 accumulator chain reordered: first fp32 operand read at 0x${firstFpReadAddr.toHexString} (want 0x0)")
    }
  }

  // ---- DMA 租约退休死锁回归（DMA死锁验证/）-----------------------------------
  //
  // 描述符 FIFO 下背靠背 DMA 不再等引擎空闲；死锁类来自 token 依赖：与上一笔
  // 区间冲突的第二笔必须等上一笔的租约退休，而退休完全由引擎回传的完成脉冲
  // （dmaDone + 该描述符自己的槽号）驱动 —— 没有脉冲，依赖它的命令永远发不出
  // 去；这正是旧"共享槽号寄存器 + 空闲沿退休"会泄漏的那一类（第二个描述符
  // 会把槽号覆盖，第一笔的槽再无人清）。本用例压住完成脉冲，验证冲突命令
  // 绝不提前发射；回传后才放行。
  // 逐拍推演与真 RTL A/B 证据：DMA死锁验证/README.md。
  it should "hold a conflicting back-to-back DMA until the first descriptor's lease echoes back done" in {
    test(new Alu(p)).withAnnotations(backendAnnos) { dut =>
      dut.io.req.valid.poke(false.B)
      dut.io.req.bits.op.poke(AluOp.ADDI)
      dut.io.req.bits.addr.poke(0.U)
      dut.io.req.bits.imm.poke(0.U)
      dut.io.rs1.valid.poke(true.B); dut.io.rs2.valid.poke(true.B)
      dut.io.rs1.data.poke(0.U); dut.io.rs2.data.poke(0.U)
      dut.io.rd.ready.poke(true.B); dut.io.vectorWrite.ready.poke(true.B)
      dut.io.dtcmMatrix.ready.poke(true.B); dut.io.dtcmMatrix.rdata.poke(0.U)
      dut.io.dtcmAux.ready.poke(true.B); dut.io.dtcmAux.rdata.poke(0.U)
      dut.io.dmaIdle.poke(true.B)        // FIFO room (engine idleness not required)

      def tick(): Unit = dut.clock.step()
      def hold(op: AluOp.Type, rs1: Int, rs2: Int, imm: Int): Unit = {
        dut.io.rs1.data.poke(rs1.U)
        dut.io.rs2.data.poke(rs2.U)
        dut.io.req.bits.op.poke(op)
        dut.io.req.bits.addr.poke(0.U)
        dut.io.req.bits.imm.poke(imm.U)
        dut.io.req.valid.poke(true.B)
        var wait = 0
        while (!dut.io.req.ready.peek().litToBoolean) {
          tick(); wait += 1; assert(wait < 200, s"$op could not issue")
        }
        tick()
        dut.io.req.valid.poke(false.B)
      }

      // 1. DMAST 读 DTCM[0x40, 4行) 搬去 DRAM：发射当拍占一个读租约，完成
      //    脉冲被压住（模拟引擎还在搬）。槽号必须在 fire 当拍采样 —— 它随
      //    描述符进 FIFO，之后不会再出现在 dmaTok 上。
      dut.io.rs1.data.poke(0x1000.U)
      dut.io.rs2.data.poke(0x40.U)
      dut.io.req.bits.op.poke(AluOp.DMAST)
      dut.io.req.bits.addr.poke(0.U)
      dut.io.req.bits.imm.poke(4.U)
      dut.io.req.valid.poke(true.B)
      var wait = 0
      while (!dut.io.req.ready.peek().litToBoolean) {
        tick(); wait += 1; assert(wait < 200, "DMAST could not issue")
      }
      assert(dut.io.dmaStart.peek().litToBoolean, "DMAST start pulse missing on its fire cycle")
      val tokEn = dut.io.dmaTok.en.peek().litToBoolean
      val tokId = dut.io.dmaTok.id.peek().litValue.toInt
      assert(tokEn, "DMAST did not declare a lease")
      tick()
      dut.io.req.valid.poke(false.B)

      // 2. 同区间的 DMALD（WAW/WAR 冲突）必须挂在 EX，等上一笔租约退休。
      dut.io.rs1.data.poke(0x2000.U)
      dut.io.rs2.data.poke(0x40.U)
      dut.io.req.bits.op.poke(AluOp.DMALD)
      dut.io.req.bits.imm.poke(4.U)
      dut.io.req.valid.poke(true.B)
      for (_ <- 0 until 8) {
        assert(!dut.io.req.ready.peek().litToBoolean,
          "conflicting DMALD issued while the DMAST read lease was still live")
        tick()
      }

      // 3. 回传该描述符自己的槽号：租约退休。发射扫描刻意不把 tokenRetire
      //    折进 dmaDepsResolved（防组合环），DMALD 在回传的下一拍就绪。
      dut.io.dmaDone.poke(true.B)
      dut.io.dmaDoneTok.en.poke(tokEn.B)
      dut.io.dmaDoneTok.id.poke(tokId.U)
      tick()
      dut.io.dmaDone.poke(false.B)
      wait = 0
      while (!dut.io.req.ready.peek().litToBoolean) {
        tick(); wait += 1; assert(wait < 4, "DMALD did not issue after the lease retired")
      }
      val tok2En = dut.io.dmaTok.en.peek().litToBoolean
      val tok2Id = dut.io.dmaTok.id.peek().litValue.toInt
      assert(tok2En, "DMALD did not declare a lease")
      tick()
      dut.io.req.valid.poke(false.B)

      // 4. SM 写 DTCM[0x40, 64行)：对 DMALD 的写租约是 WAR。租约未回传时
      //    必须 0 写拍；回传后 64 拍宽写一次跑完。
      hold(AluOp.SM, rs1 = 0x40, rs2 = 0, imm = 0)
      val storeBeats = p.MatrixDim * p.MatrixDim / (p.DtcmRowBits / 32)
      def countWrites(n: Int): Int = {
        var w = 0
        for (_ <- 0 until n) {
          if (dut.io.dtcmMatrix.valid.peek().litToBoolean &&
              dut.io.dtcmMatrix.write.peek().litToBoolean) w += 1
          tick()
        }
        w
      }
      assert(countWrites(200) == 0,
        "SM wrote beats while the DMALD write lease was still live")
      dut.io.dmaDone.poke(true.B)
      dut.io.dmaDoneTok.en.poke(tok2En.B)
      dut.io.dmaDoneTok.id.poke(tok2Id.U)
      tick()
      dut.io.dmaDone.poke(false.B)
      val released = countWrites(200)
      assert(released == storeBeats,
        s"SM wrote $released of $storeBeats beats after the DMA lease retired " +
        s"(leaked slot: the echo did not retire the descriptor's own id)")
    }
  }
}
