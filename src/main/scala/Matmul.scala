// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


object Matmul {
    def apply(p: Parameters): Matmul = {
        return Module(new Matmul(p))
    }
}

// Matrix-unit shell (C3/C5): routes commands between the int8 engine(s) and
// the bf16 tier, arbitrates their shared DTCM port, and reports per-class
// completion. The int8 FSM lives in Int8Engine; p.int8Engines selects one
// (legacy) or two instances.
//
// Command routing:
//   - FP.* commands self-describe their tier and go to the fp32 unit;
//   - LMA/LMB BROADCAST: every int8 engine and the fp32 unit latch the base
//     (engines only latch while idle, so a broadcast fires only when every
//     target is ready -- the shell's req.ready folds that in);
//   - MATMUL/MATMULA/SM/MQCFG/MCFG2 go to io.engine's int8 engine only
//     (io.engine is the enqueue-time engine-select snapshot from Alu).
//
// DTCM port arbitration is FIXED-PRIORITY e0 > e1 > fp32. Correctness holds
// under any priority (every client is an independent FSM that never holds the
// port while waiting on another engine); only sustained simultaneous bursts
// could starve a lower client, which no current workload produces. Swap in a
// round-robin pointer if that ever shows up in a profile.
//
// done is a 3-entry one-hot-per-class pulse vector (0=e0, 1=e1, 2=fp32) so
// Alu can retire each engine's range token by source. It stays an
// unconditional file-tail connection: folding any leg into a when would eat
// a done pulse, leak the token, and hang HALT (the C4-verified iron rule).
class Matmul(p: Parameters) extends Module {
    val nUnits = 3

    val io = IO(new Bundle {
        // Decode cycle.
        val req = Flipped(Decoupled(new AluCmd))
        // Execute cycle.
        val rs1 = Flipped(new RegfileReadDataIO)
        val rs2 = Flipped(new RegfileReadDataIO)
        val rd  = Decoupled(new RegfileWriteDataIO)
        // Dedicated DTCM burst port.
        val dtcm = new DtcmPortIO(p)
        // Snapshot supplied by Alu's MatrixIdex queue for the command on req.
        val quant = Input(new MatrixQuantConfig(p))
        val context = Input(new MatrixContext(p))
        // Int8 engine select for the command on req (0 in single-engine
        // builds; ignored by FP.* commands).
        val engine = Input(UInt(1.W))
        // Per-class issue readiness (0=e0, 1=e1, 2=fp32) for Alu's queue
        // gating; unused classes read true.
        val unitReady = Output(Vec(nUnits, Bool()))
        // Per-class completion pulse, same indexing.
        val done = Output(Vec(nUnits, Bool()))
        // Aggregate completion (any class) -- legacy single-bit view.
        val doneAny = Output(Bool())
        // PERF probe: per-engine busy, readable at PERF 0x64/0x68/0x6C.
        val engBusy = Output(Vec(2, Bool()))
    })

    // ---- Native FP tier (bf16 8x8 when bf16Matmul) ----------------------------
    val fp32Insn = io.req.bits.op.isOneOf(
        AluOp.FP_MATMUL, AluOp.FP_MATMULA, AluOp.FP_SM)
    if (!p.bf16Matmul) {
        // Without an FP unit these commands would be silently dropped and
        // their range token would never retire (deadlock): catch it here.
        assert(!(io.req.valid && fp32Insn),
            "Matmul: FP.* commands need Parameters(bf16Matmul = true)")
    }
    val fp32Unit: Option[FpTier] =
        if (p.bf16Matmul) Some(Module(new Bf16MatmulUnit(p)))
        else None

    // ---- Int8 engines ---------------------------------------------------------
    val eng0 = Module(new Int8Engine(p, 0))
    val eng1 = if (p.int8Engines >= 2) Some(Module(new Int8Engine(p, 1))) else None
    val engines = Seq(Some(eng0), eng1).flatten

    val bcast = io.req.bits.op.isOneOf(AluOp.LMA, AluOp.LMB)
    val int8Insn = !fp32Insn
    // LMA/LMB addressing by build: with a single int8 engine they keep the
    // historical all-latch broadcast (engines + fp tier) -- fp32-tier programs
    // rely on it and the all-idle fire condition is unobservable there.  With
    // two engines the broadcast would force both engines into lockstep (every
    // base load waits for BOTH idle), so they target io.engine's engine only;
    // per-engine operand streams then interleave freely.  A dual-int8 build
    // that also runs the FP tier would need an explicit FP-targeting LMA --
    // none exists today (dual8 has no fp unit).
    val dualInt8 = p.int8Engines >= 2
    val bcastAll = if (dualInt8) false.B else bcast

    for ((e, i) <- engines.zipWithIndex) {
        e.io.req.valid := io.req.valid && int8Insn &&
            (bcastAll || io.engine === i.U)
        e.io.req.bits  := io.req.bits
        e.io.rs1.valid := io.rs1.valid
        e.io.rs1.data  := io.rs1.data
        e.io.rs2.valid := io.rs2.valid
        e.io.rs2.data  := io.rs2.data
        e.io.quant     := io.quant
        e.io.context   := io.context
    }

    fp32Unit.foreach { u =>
        u.io.req.valid := io.req.valid && (fp32Insn || bcastAll)
        u.io.req.bits  := io.req.bits
        u.io.rs1.valid := true.B
        u.io.rs1.data  := io.rs1.data
        u.io.context   := io.context
    }

    io.rd.valid     := false.B   // matrix ops never write a GPR
    io.rd.bits.addr := 0.U
    io.rd.bits.data := 0.U

    // ---- DTCM port arbitration: fixed priority e0 > e1 > fp32 ----------------
    // Every engine keeps at most one read in flight, so rdata can broadcast to
    // all clients: only the granted client samples it.
    val clients = engines.map(_.io.dtcm) ++
        fp32Unit.map(u => Seq(u.io.dtcm)).getOrElse(Seq.empty[DtcmPortIO])
    // index 0..engines.size-1 are the int8 engines, fp32 last.
    val grant = Wire(Vec(clients.size, Bool()))
    var masked = true.B
    for (i <- 0 until clients.size) {
        grant(i) := masked && clients(i).valid
        masked = masked && !clients(i).valid
    }
    io.dtcm.valid := false.B
    io.dtcm.write := false.B
    io.dtcm.addr  := 0.U
    io.dtcm.wdata := 0.U
    io.dtcm.wstrb := 0.U
    for (i <- 0 until clients.size) {
        when (grant(i)) {
            io.dtcm.valid := clients(i).valid
            io.dtcm.write := clients(i).write
            io.dtcm.addr  := clients(i).addr
            io.dtcm.wdata := clients(i).wdata
            io.dtcm.wstrb := clients(i).wstrb
        }
    }
    for (i <- 0 until clients.size) {
        clients(i).ready := io.dtcm.ready && grant(i)
        clients(i).rdata := io.dtcm.rdata
    }

    // ---- Issue readiness ------------------------------------------------------
    // A command fires only when every target can latch it: the selected engine
    // for workers/MQCFG/MCFG2, ALL engines (and the fp32 unit) for the LMA/LMB
    // broadcast in single-int8 builds, or the selected engine for LMA/LMB in
    // dual-int8 builds (see the routing note above).
    val fpReady = fp32Unit.map(_.io.req.ready).getOrElse(true.B)
    val engReady = engines.map(_.io.req.ready)
    val allEngReady = engReady.reduce(_ && _)
    val selEngReady = eng1 match {
        case Some(_) => Mux(io.engine === 0.U, engReady(0), engReady(1))
        case None    => engReady(0)
    }
    // FP.* needs only the fp32 unit; an LMA/LMB broadcast needs every latch
    // (all int8 engines AND the fp32 unit); any other int8 command needs only
    // its selected engine.
    io.req.ready := Mux(fp32Insn, fpReady,
        Mux(bcastAll, allEngReady && fpReady, selEngReady))

    // ---- Per-class readiness / done -------------------------------------------
    io.unitReady(0) := engReady(0)
    io.unitReady(1) := eng1.map(_ => engReady(1)).getOrElse(true.B)
    io.unitReady(2) := fpReady

    io.done(0) := eng0.io.done
    io.done(1) := eng1.map(_.io.done).getOrElse(false.B)
    io.done(2) := fp32Unit.map(_.io.done).getOrElse(false.B)

    // Aggregate pulse: unconditional connection (see class comment).
    io.doneAny := io.done.asUInt.orR

    io.engBusy(0) := eng0.io.busy                              // PERF probe (dual-overlap work)
    io.engBusy(1) := eng1.map(_.io.busy).getOrElse(false.B)    // PERF probe (dual-overlap work)
}
