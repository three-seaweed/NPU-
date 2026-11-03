// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._

object VectorUnit {
  def apply(p: Parameters): VectorUnit = Module(new VectorUnit(p))
}

// Blocking custom-1 vector unit.  A DTCM row holds eight bf16 elements and all
// eight lanes progress together.
//
// The unit is a three-segment row pipeline sharing the single wide DTCM port:
//
//   FETCH   issues the operand reads for row r into one of two input buffers,
//   COMPUTE pops an input buffer, runs the op's datapath, pushes an output row,
//   STORE   pops an output buffer and issues the DTCM write.
//
// The segments run concurrently on different rows, so per-row cost is set by the
// port (reads + writes per row) rather than by the sum of the FSM steps: VSMUL
// costs 2 cycles/row instead of 5, VMUL 3 instead of 7, VDOT 2 instead of 10.
// The port is arbitrated STORE > FETCH, which guarantees drain and so cannot
// deadlock: a full output buffer always wins the port and empties.
//
// Numerics are unchanged.  Expensive fp32 chains are still register-separated:
// reciprocal/rsqrt keep one state per Newton sub-step (they run one row at a
// time, which is all the layer programs need -- they evaluate scales on
// single-element vectors), and reductions still use the 8 -> 4 -> 2 -> 1 tree.
// That tree is now free-running rather than stepped, but rows still enter and
// leave it in order and `acc` is still folded exactly once per row in row order,
// so a reduction produces bit-identical results to the serial version.  This
// matters: fp32 addition is not associative, so any reassociation would change
// the answer.
//
// ALIASING: because fetch of row r+1 may pass the store of row r, a source
// region may alias the destination only if it is *identical* to it (the common
// in-place `VSETA b; VSETD b` form).  A shifted overlap (dst = srcA +/- one row)
// would read pre-store data for some rows and post-store data for others; the
// assertion below catches that in simulation.  See docs/ISA.md 2.5.4.
class VectorUnit(p: Parameters) extends Module {
  val L = p.Bf16LanesPerRow
  require(L == 8, "the pipelined reduction tree is defined for eight bf16 lanes")

  val rowBytes = p.DtcmRowBytes
  val rowShift = log2Ceil(rowBytes)
  val NEG_INF = "hff800000".U(32.W)
  val POS_INF = "h7f800000".U(32.W)
  val QNAN = "h7fc00000".U(32.W)

  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new AluCmd))
    val rs1 = Flipped(new RegfileReadDataIO)
    val rs2 = Flipped(new RegfileReadDataIO)
    val rd = Decoupled(new RegfileWriteDataIO)
    val dtcm = new DtcmPortIO(p)
    // Queue-provided compute snapshot. Direct VectorUnit users keep valid low
    // and continue to use the architectural VSET* registers.
    val context = Input(new VectorContext(p))
    val done = Output(Bool())
    // PERF split: true unit execution (row pipeline running), distinct from
    // queue occupancy -- io.req.ready collapses both.
    val busy = Output(Bool())
  })

  def usesB(op: AluOp.Type): Bool =
    op.isOneOf(AluOp.VADD, AluOp.VSUB, AluOp.VMUL, AluOp.VFMA, AluOp.VDOT)

  def usesD(op: AluOp.Type): Bool = op === AluOp.VFMA

  def writesDst(op: AluOp.Type): Bool =
    op.isOneOf(
      AluOp.VADD, AluOp.VSUB, AluOp.VMUL, AluOp.VFMA, AluOp.VSMUL, AluOp.VSADD,
      AluOp.VEXP, AluOp.VRECIP, AluOp.VRSQRT, AluOp.VSOFTEXP, AluOp.VABS,
    )

  def isReduce(op: AluOp.Type): Bool =
    op.isOneOf(AluOp.VSUM, AluOp.VMAX, AluOp.VDOT, AluOp.VSOFTEXP)

  def isBasic(op: AluOp.Type): Bool =
    op.isOneOf(AluOp.VADD, AluOp.VSUB, AluOp.VMUL, AluOp.VFMA, AluOp.VSMUL, AluOp.VSADD,
      AluOp.VABS)

  // Top level: config ops retire in sIdle; compute ops run the three segments in
  // sRun; reductions then hand their fp32 result to the GPR in sWriteRd.
  val sIdle :: sRun :: sWriteRd :: Nil = Enum(3)
  val state = RegInit(sIdle)

  val srcABase = RegInit(0.U(p.AddrBits.W))
  val srcBBase = RegInit(0.U(p.AddrBits.W))
  val dstBase = RegInit(0.U(p.AddrBits.W))
  val vlen = RegInit(0.U(16.W))
  val scalarOp = RegInit(0.U(32.W))

  val curOp = Reg(AluOp())
  val rdReg = RegInit(0.U(5.W))
  val resultReg = RegInit(0.U(32.W))

  // Row count.  For VCVT.BF16 a row is an *output* row (fed by two int32 input
  // rows); for VCVT.I8 a row is an *input* row (two of which share one int8
  // output row).  Every other op maps rows 1:1.
  val rows = (vlen + (L - 1).U) >> log2Ceil(L)

  // Lane mask for row r: the final partial row masks its unused high lanes.
  def laneMask(r: UInt): Vec[Bool] = {
    val rem = vlen - (r << log2Ceil(L))
    VecInit((0 until L).map(k => k.U < rem))
  }

  // ---- Row buffers ---------------------------------------------------------
  // Two of each, so fetch can be filling row r+1 while compute holds row r.
  val ibA = Reg(Vec(2, UInt(p.DtcmRowBits.W)))
  val ibB = Reg(Vec(2, UInt(p.DtcmRowBits.W)))
  val ibD = Reg(Vec(2, UInt(p.DtcmRowBits.W)))
  val ibRow = Reg(Vec(2, UInt(16.W)))
  val ibValid = RegInit(VecInit(Seq.fill(2)(false.B)))

  // The store side is format-agnostic: compute hands over a finished 128-bit row,
  // its byte strobe and its byte address, so VCVT.I8's half-row packing needs no
  // special case in the store segment.
  val obData = Reg(Vec(2, UInt(p.DtcmRowBits.W)))
  val obStrb = Reg(Vec(2, UInt(p.DtcmRowBytes.W)))
  val obAddr = Reg(Vec(2, UInt(p.AddrBits.W)))
  val obValid = RegInit(VecInit(Seq.fill(2)(false.B)))

  val fRow = RegInit(0.U(16.W))     // input row being fetched
  val fSel = RegInit(0.U(2.W))      // 0 = srcA, 1 = srcB, 2 = dst (VFMA read-modify-write)
  val fPtr = RegInit(0.U(1.W))      // input buffer being filled
  val cPtr = RegInit(0.U(1.W))      // input buffer being consumed
  val cOut = RegInit(0.U(1.W))      // output buffer being produced
  val cRows = RegInit(0.U(16.W))    // rows compute has finished with
  val sPtr = RegInit(0.U(1.W))      // output buffer being written
  val sRows = RegInit(0.U(16.W))    // stores retired

  // Reads return one cycle after they fire, so the in-flight read carries a tag.
  val rdFire = RegInit(false.B)
  val rdSel = Reg(UInt(2.W))
  val rdPtr = Reg(UInt(1.W))
  val rdRow = Reg(UInt(16.W))
  val rdLast = Reg(Bool())

  // ---- Compute sub-states ---------------------------------------------------
  // Long arithmetic chains span multiple states. That is a deliberate trade:
  // the DTCM port needs 2-3 cycles per row (2 for VSMUL, 3 for VMUL -- see the
  // throughput regression in VectorUnitSpec), so the compute segment had a spare
  // cycle, and spending it buys a register between the two long operations that
  // used to sit back to back in one period. Throughput is set by the port, so it
  // barely moves; only per-op latency grows by one cycle.
  val cStates = Enum(13)
  val cIdle = cStates(0); val cSoftE = cStates(1)
  val cRecipC1 = cStates(2); val cRecipY1 = cStates(3); val cRecipP2 = cStates(4)
  val cRecipC2 = cStates(5); val cRecipY2 = cStates(6)
  val cRsqrtT = cStates(7); val cRsqrtC = cStates(8); val cRsqrtY = cStates(9)
  val cBasic2 = cStates(10)   // addF32 + narrow, after stage 1's mulF32
  val cExp2 = cStates(11)     // 2^f table + interpolate, after stage 1's x*log2e
  val cSoftE2 = cStates(12)   // VSOFTEXP: second half of exp, emit and fold
  val cState = RegInit(cIdle)

  val pipeX = Reg(Vec(L, UInt(32.W)))
  val pipeY = Reg(Vec(L, UInt(32.W)))
  val pipeP = Reg(Vec(L, UInt(32.W)))
  val pipeC = Reg(Vec(L, UInt(32.W)))
  val recipSign = Reg(Vec(L, Bool()))
  val recipZero = Reg(Vec(L, Bool()))
  val recipInf = Reg(Vec(L, Bool()))
  val rsqrtSign = Reg(Vec(L, Bool()))
  val rsqrtZero = Reg(Vec(L, Bool()))
  // exp's range-reduction result, held between the two halves of the chain.
  val expMid = Reg(Vec(L, new ExpMid))
  // The mask, row index and last-row flag of the row a chain is working on. These
  // are latched because stage 1 releases the input buffer immediately, so cPtr has
  // already moved on by the time the chain emits.
  val cMaskReg = Reg(Vec(L, Bool()))
  val cLastReg = Reg(Bool())
  val cRowReg = Reg(UInt(16.W))

  // ---- Reduction tree ------------------------------------------------------
  // Free-running 4-stage pipeline: load -> 8:4 -> 4:2 -> 2:1 -> fold into acc.
  // Rows enter in order and never overtake, so `acc` is folded in row order.
  //
  // Masked-off lanes of a partial final row are filled with the operation's
  // identity at load time (+0 for the sums, -Inf for max) rather than carried as
  // per-node valid bits. That is exactly equivalent -- x+0 == x and
  // max(x,-Inf) == x, both bit-precise -- and it removes two levels of mux from
  // every tree node.  The resulting cone is smaller; whether it meets the target
  // period is intentionally left to the post-route report.
  val terms = Reg(Vec(L, UInt(32.W)))
  val red1 = Reg(Vec(4, UInt(32.W)))
  val red2 = Reg(Vec(2, UInt(32.W)))
  val rowSum = Reg(UInt(32.W))
  val acc = RegInit(0.U(32.W))
  val t0Valid = RegInit(false.B); val t0Last = Reg(Bool())
  val r1Valid = RegInit(false.B); val r1Last = Reg(Bool())
  val r2Valid = RegInit(false.B); val r2Last = Reg(Bool())
  val r3Valid = RegInit(false.B); val r3Last = Reg(Bool())
  val treeEmpty = !t0Valid && !r1Valid && !r2Valid && !r3Valid

  // Ops that hand a finished row to the store segment. Both conversions do, even
  // though neither is in writesDst (their output format/stride differs from the
  // bf16 elementwise ops).
  def producesRows(op: AluOp.Type): Bool =
    writesDst(op) || op === AluOp.VCVT_I8 || op === AluOp.VCVT_BF16

  val nStores = Mux(producesRows(curOp), rows, 0.U)
  val allDone = (fRow === rows) && (cRows === rows) && (sRows === nStores) && treeEmpty

  io.dtcm.valid := false.B
  io.dtcm.write := false.B
  io.dtcm.addr := 0.U
  io.dtcm.wdata := 0.U
  io.dtcm.wstrb := 0.U
  // Back-to-back dispatch: on the cycle the last row's store retires
  // (allDone), the port is idle and the unit state is fully drained, so the
  // next command may start immediately instead of paying a sIdle round trip.
  // Reductions are excluded -- their fp32 result still needs the sWriteRd
  // handoff to the GPR.  Alu's retire path pairs the same-cycle done+start
  // (it retires the outgoing token from the still-current tokenId register).
  io.req.ready := (state === sIdle) ||
    (state === sRun && allDone && !isReduce(curOp))
  io.busy := state =/= sIdle
  io.rd.valid := state === sWriteRd
  io.rd.bits.addr := rdReg
  io.rd.bits.data := resultReg
  io.done := false.B
  // Tree-entry default. This must be declared BEFORE the compute segment: Chisel
  // last-connect semantics mean whichever assignment appears later in source order
  // wins, so putting it after loadTree() would silently swallow the pulse.
  t0Valid := false.B

  // ---- Datapath views of the row compute currently holds --------------------
  val cA = ibA(cPtr)
  val cB = ibB(cPtr)
  val cD = ibD(cPtr)
  val aF = VecInit((0 until L).map(k => Bf16.toF32(cA(16 * k + 15, 16 * k))))
  val bF = VecInit((0 until L).map(k => Bf16.toF32(cB(16 * k + 15, 16 * k))))
  val dF = VecInit((0 until L).map(k => Bf16.toF32(cD(16 * k + 15, 16 * k))))

  // VCVT.BF16 reads 4 int32 values per DTCM row, so one 8-lane bf16 output row
  // spans two input rows: srcA row 2r lands in buffer slot A, row 2r+1 in slot B.
  val i32Lane = VecInit((0 until L).map { k =>
    if (k < 4) cA(32 * k + 31, 32 * k) else cB(32 * (k - 4) + 31, 32 * (k - 4))
  })

  // ---- Shared per-lane fp32 datapaths (2026-11-02 area diet) -----------------
  // The compute chains are FSM-serialized -- at most one chain state runs per
  // cycle -- so the per-lane addF32 / mulF32 / expStage / f32ToBf16 cones that
  // used to be instantiated per call site (9 mul + 5 add + 2x2 exp + 6 narrow
  // sites x 8 lanes, ~52K LUT) were never simultaneously live.  Each now
  // exists ONCE per lane; call sites drive the operand busses (last-connect
  // override; the driving conditions are mutually exclusive states/ops) and
  // read the shared result in the same cycle.  Bit-exact by construction:
  // identical functions on identical operand values, FSM cycle structure
  // untouched.  The reduction tree keeps its own reduceCombine nodes: its four
  // stages pipeline different rows concurrently, so its 8 adders ARE
  // simultaneously live and cannot share anything.  Measured unit costs:
  // addF32 = 429, mulF32 = 109 (+2 DSP), exp pair = 645 LUT per lane.
  val sMulX = WireDefault(VecInit(Seq.fill(L)(0.U(32.W))))
  val sMulY = WireDefault(VecInit(Seq.fill(L)(0.U(32.W))))
  val sAddX = WireDefault(VecInit(Seq.fill(L)(0.U(32.W))))
  val sAddY = WireDefault(VecInit(Seq.fill(L)(0.U(32.W))))
  val sNarX = WireDefault(VecInit(Seq.fill(L)(0.U(32.W))))
  val sExpX = WireDefault(VecInit(Seq.fill(L)(0.U(32.W))))
  val sMul  = Wire(Vec(L, UInt(32.W)))
  val sAdd  = Wire(Vec(L, UInt(32.W)))
  val sNar  = Wire(Vec(L, UInt(16.W)))
  val sExp1 = Wire(Vec(L, new ExpMid))
  val sExp2 = Wire(Vec(L, UInt(32.W)))
  for (k <- 0 until L) {
    sMul(k)  := Bf16.mulF32(sMulX(k), sMulY(k))
    sAdd(k)  := Bf16.addF32(sAddX(k), sAddY(k))
    sNar(k)  := Bf16.f32ToBf16(sNarX(k))
    sExp1(k) := Activation.expStage1(sExpX(k))
    sExp2(k) := Activation.expStage2(expMid(k)) // both s2 sites read expMid: no mux
  }

  // fp32 -> signed int8 with round-to-nearest-even and native signed saturation
  // [-128, +127]. NaN maps to zero; scale and zero-point arithmetic are deliberately
  // outside this conversion primitive.
  def f32ToI8RneSat(f: UInt): UInt = {
    val sign = f(31)
    val exp  = f(30, 23)
    val sig  = Cat(1.U(1.W), f(22, 0))                 // 24b significand (implicit 1)
    val isNaN = exp === "hff".U && f(22, 0) =/= 0.U
    val tooSmall = exp < 126.U                          // |f| < 0.5
    val tooBig   = exp >= 135.U                         // |f| >= 256
    // Only exponents 126..134 can reach the RNE path: they need right shifts
    // 24..16 respectively. Keeping the old 8-bit shift amount made firtool
    // emit a 288-bit variable shift per lane, which Vivado expanded into a
    // multi-million-mux network despite all other exponent cases being clamped.
    val inRoundRange = exp >= 126.U && exp <= 134.U
    val sh = Mux(inRoundRange, (150.U(8.W) - exp)(4, 0), 0.U(5.W))
    val shMinusOne = Mux(sh === 0.U, 0.U(5.W), sh - 1.U)
    val intMag = (sig >> sh)(8, 0)
    val guard  = (sig >> shMinusOne)(0)
    // The RNE range has only nine possible sticky masks (bits 15..23). A
    // literal lookup keeps both the mask and its mux inputs at 24 bits; a
    // variable left shift would still widen its intermediate result in FIRRTL.
    val stickyMask = MuxLookup(shMinusOne, 0.U(24.W))(Seq(
      15.U(5.W) -> "h007fff".U(24.W),
      16.U(5.W) -> "h00ffff".U(24.W),
      17.U(5.W) -> "h01ffff".U(24.W),
      18.U(5.W) -> "h03ffff".U(24.W),
      19.U(5.W) -> "h07ffff".U(24.W),
      20.U(5.W) -> "h0fffff".U(24.W),
      21.U(5.W) -> "h1fffff".U(24.W),
      22.U(5.W) -> "h3fffff".U(24.W),
      23.U(5.W) -> "h7fffff".U(24.W)))
    val sticky = (sig & stickyMask).orR
    val roundUp = guard && (sticky || intMag(0))
    val mag = intMag +& roundUp                          // can reach 256
    val maxMag = Mux(sign, 128.U(10.W), 127.U(10.W))
    val clamped = Mux(tooBig || mag > maxMag, maxMag, mag)
    val signed = Mux(sign, (~clamped(7, 0)) + 1.U, clamped(7, 0))
    Mux(tooSmall || isNaN, 0.U(8.W), signed)(7, 0)
  }

  // Elementwise arithmetic, split across two cycles. Stage 1 does the multiply
  // and picks the adder's operands; stage 2 does the add and the narrow to bf16.
  //
  // The mux that selects between `prod` and `aF` used to sit before the adder, so
  // the unsplit implementation exposed mulF32 -> mux -> addF32 -> f32ToBf16 as
  // one timing cone.  The register here separates multiply from add/narrow;
  // post-route STA, not functional output patterns, determines whether it meets.
  val basicAddA = Wire(Vec(L, UInt(32.W)))
  val basicAddB = Wire(Vec(L, UInt(32.W)))
  val basicThru = Wire(Vec(L, UInt(32.W)))   // ops whose result skips the adder
  for (k <- 0 until L) {
    sMulX(k) := aF(k)   // unconditional drive; every other mul site overrides
    sMulY(k) := Mux(curOp === AluOp.VSMUL, scalarOp, bF(k))  // under its condition
    val prod = sMul(k)
    basicAddA(k) := Mux(curOp === AluOp.VFMA, prod, aF(k))
    basicAddB(k) := MuxLookup(curOp.asUInt, 0.U(32.W))(Seq(
      AluOp.VADD.asUInt -> bF(k),
      AluOp.VSUB.asUInt -> Bf16.negF32(bF(k)),
      AluOp.VSADD.asUInt -> scalarOp,
      AluOp.VFMA.asUInt -> dF(k),
    ))
    basicThru(k) := Mux(curOp === AluOp.VABS, Cat(0.U(1.W), aF(k)(30, 0)), prod)
  }

  val basicOut = Wire(Vec(L, UInt(32.W)))
  for (k <- 0 until L) {
    sAddX(k) := pipeX(k)  // unconditional drive; cSoftE/C-step states override
    sAddY(k) := pipeY(k)
    val sum = sAdd(k)
    basicOut(k) := MuxLookup(curOp.asUInt, sum)(Seq(
      AluOp.VMUL.asUInt -> pipeP(k),
      AluOp.VSMUL.asUInt -> pipeP(k),
      AluOp.VABS.asUInt -> pipeP(k),
    ))
  }

  // VSUM/VMAX fold srcA directly; VDOT folds the lane products.  (VSOFTEXP feeds
  // the tree from its own exp results, so it is not in this mux.)
  val reduceInput = Wire(Vec(L, UInt(32.W)))
  for (k <- 0 until L) {
    when(curOp === AluOp.VDOT) { sMulX(k) := aF(k); sMulY(k) := bF(k) }
    reduceInput(k) := Mux(curOp === AluOp.VDOT, sMul(k), aF(k))
  }

  // VMAX selects a larger fp32 value; all other reductions add. The tree only
  // combines valid leaves, so a partial final DTCM row needs no fake identity.
  val isMaxReduce = curOp === AluOp.VMAX
  def reduceCombine(x: UInt, y: UInt): UInt =
    Mux(isMaxReduce, Mux(Bf16.gtF32(y, x), y, x), Bf16.addF32(x, y))

  // ---- FETCH segment -------------------------------------------------------
  // Operands per row: VFMA needs srcA/srcB/dst, the two-source ops and VCVT.BF16
  // need two, everything else one.
  val nSel = Mux(curOp === AluOp.VFMA, 3.U(2.W),
             Mux(usesB(curOp) || curOp === AluOp.VCVT_BF16, 2.U(2.W), 1.U(2.W)))
  val fLastSel = fSel === (nSel - 1.U)
  val fRowOff = fRow << rowShift
  val fetchAddr = Mux(curOp === AluOp.VCVT_BF16,
    srcABase + (((fRow << 1) + fSel) << rowShift),
    MuxLookup(fSel, srcABase + fRowOff)(Seq(
      1.U -> (srcBBase + fRowOff),
      2.U -> (dstBase + fRowOff))))

  // Only start filling a buffer once compute has released it.  Two buffers plus
  // at least one issue cycle per row means the previous read into a buffer has
  // always been captured by the time fetch wraps back round to it.
  val fetchReq = (state === sRun) && (fRow < rows) && !ibValid(fPtr)

  // ---- STORE segment -------------------------------------------------------
  val storeReq = obValid(sPtr)

  // ---- Shared DTCM port: store wins, so output buffers always drain ---------
  io.dtcm.valid := storeReq || fetchReq
  io.dtcm.write := storeReq
  io.dtcm.addr := Mux(storeReq, obAddr(sPtr), fetchAddr)
  io.dtcm.wdata := obData(sPtr)
  io.dtcm.wstrb := obStrb(sPtr)
  val storeFire = storeReq && io.dtcm.ready
  val fetchFire = !storeReq && fetchReq && io.dtcm.ready

  when(storeFire) {
    obValid(sPtr) := false.B
    sPtr := ~sPtr
    sRows := sRows + 1.U
  }

  rdFire := fetchFire
  when(fetchFire) {
    rdSel := fSel
    rdPtr := fPtr
    rdRow := fRow
    rdLast := fLastSel
    when(fLastSel) {
      fSel := 0.U
      fRow := fRow + 1.U
      fPtr := ~fPtr
    }.otherwise {
      fSel := fSel + 1.U
    }
  }

  // Capture the returning read one cycle later, into the buffer it was tagged
  // with (fetch may already have moved on to the next buffer).
  when(rdFire) {
    switch(rdSel) {
      is(0.U) { ibA(rdPtr) := io.dtcm.rdata }
      is(1.U) { ibB(rdPtr) := io.dtcm.rdata }
      is(2.U) { ibD(rdPtr) := io.dtcm.rdata }
    }
    when(rdLast) {
      ibRow(rdPtr) := rdRow
      ibValid(rdPtr) := true.B
    }
  }

  // ---- COMPUTE segment -----------------------------------------------------
  val cRow = ibRow(cPtr)
  val cMask = laneMask(cRow)
  val cLast = cRow === (rows - 1.U)
  def strbOf(m: Vec[Bool]): UInt =
    Cat((0 until L).reverse.map(k => Mux(m(k), "b11".U(2.W), "b00".U(2.W))))

  // producesRows/nStores/allDone live above (treeEmpty), where req.ready can
  // see them for back-to-back dispatch.
  val cNeedsOut = producesRows(curOp)
  // A chain may only start when its output slot is free; once started it runs to
  // completion without further gating (nothing else can claim that slot).
  val cStart = (state === sRun) && (cState === cIdle) && ibValid(cPtr) &&
    (!cNeedsOut || !obValid(cOut))

  def emit(data: UInt, strb: UInt, addr: UInt): Unit = {
    obData(cOut) := data
    obStrb(cOut) := strb
    obAddr(cOut) := addr
    obValid(cOut) := true.B
    cOut := ~cOut
  }
  def emitLanes(v: Seq[UInt], m: Vec[Bool], r: UInt): Unit =
    emit(Cat(v.reverse), strbOf(m), dstBase + (r << rowShift))
  def release(): Unit = {
    ibValid(cPtr) := false.B
    cPtr := ~cPtr
    cRows := cRows + 1.U
  }
  // Masked-off lanes take the operation's identity, so the tree needs no valid bits.
  def loadTree(src: Seq[UInt], m: Vec[Bool], last: Bool): Unit = {
    val identity = Mux(isMaxReduce, NEG_INF, 0.U(32.W))
    for (k <- 0 until L) { terms(k) := Mux(m(k), src(k), identity) }
    t0Valid := true.B
    t0Last := last
  }

  when(cStart) {
    // Stage 1 for every op. It captures everything the rest of the chain needs
    // into registers, so the input buffer is released here rather than at emit:
    // fetch can start the next row while this one is still finishing.
    cMaskReg := cMask
    cLastReg := cLast
    cRowReg := cRow
    when(isBasic(curOp)) {
      for (k <- 0 until L) {
        pipeX(k) := basicAddA(k)
        pipeY(k) := basicAddB(k)
        pipeP(k) := basicThru(k)
      }
      cState := cBasic2
    }.elsewhen(curOp === AluOp.VEXP) {
      for (k <- 0 until L) { sExpX(k) := aF(k); expMid(k) := sExp1(k) }
      cState := cExp2
    }.elsewhen(curOp === AluOp.VCVT_BF16) {
      // Keep conversion and narrowing together.  Besides avoiding an otherwise
      // unnecessary fp32 register, firtool can simplify this composition across
      // the function boundary.  Splitting it introduced a reproducible board-only
      // VCVT.BF16 regression while the original implementation passed.
      // (i32ToF32 stays inline: it is tiny and feeds only this site's operand.)
      for (k <- 0 until L) { sNarX(k) := Bf16.i32ToF32(i32Lane(k)) }
      emitLanes((0 until L).map(k => sNar(k)), cMask, cRow)
    }.elsewhen(curOp === AluOp.VCVT_I8) {
      // Eight bf16 lanes pack into one half of an int8 row: even input rows fill
      // the low half, odd rows the high half of the row at dstBase + (r/2)*16.
      // Left as a single cycle -- no multiplier, and the board confirms it.
      val packed = Cat((0 until L).reverse.map(k => f32ToI8RneSat(aF(k))))
      val strb8 = Cat((0 until L).reverse.map(k => cMask(k)))
      val half = cRow(0)
      emit(Mux(half, Cat(packed, 0.U(64.W)), Cat(0.U(64.W), packed)),
           Mux(half, Cat(strb8, 0.U(8.W)), Cat(0.U(8.W), strb8)),
           dstBase + ((cRow >> 1) << rowShift))
    }.elsewhen(curOp === AluOp.VSOFTEXP) {
      for (k <- 0 until L) {
        sAddX(k) := aF(k); sAddY(k) := Bf16.negF32(scalarOp)
        pipeX(k) := sAdd(k)
      }
      cState := cSoftE
    }.elsewhen(curOp === AluOp.VRECIP) {
      // recip(x) = y * (2 - x*y), twice.  One multiply/add per state.
      for (k <- 0 until L) {
        val x = aF(k)
        val xAbs = Cat(0.U(1.W), x(30, 0))
        val y0 = "h7eeeeeee".U(32.W) - xAbs
        pipeX(k) := xAbs
        pipeY(k) := y0
        sMulX(k) := xAbs; sMulY(k) := y0
        pipeP(k) := sMul(k)
        recipSign(k) := x(31)
        recipZero(k) := x(30, 23) === 0.U
        recipInf(k) := x(30, 23) === "hff".U && x(22, 0) === 0.U
      }
      cState := cRecipC1
    }.elsewhen(curOp === AluOp.VRSQRT) {
      for (k <- 0 until L) {
        val x = aF(k)
        val xAbs = Cat(0.U(1.W), x(30, 0))
        val xHalf = Cat(0.U(1.W), x(30, 23) - 1.U, x(22, 0))
        val y0 = "h5f3759df".U(32.W) - (xAbs >> 1)
        pipeX(k) := xHalf
        pipeY(k) := y0
        sMulX(k) := y0; sMulY(k) := y0
        pipeP(k) := sMul(k)
        rsqrtSign(k) := x(31)
        rsqrtZero(k) := x(30, 23) === 0.U
      }
      cState := cRsqrtT
    }.otherwise {
      // VSUM / VMAX / VDOT: one row per cycle into the tree. Stage 1 only, since
      // its longest operation is the single VDOT multiply.
      loadTree(reduceInput, cMask, cLast)
    }
    release()
  }

  // ---- Second halves of the split chains ------------------------------------
  when(cState === cBasic2) {
    for (k <- 0 until L) { sNarX(k) := basicOut(k) }
    emitLanes((0 until L).map(k => sNar(k)), cMaskReg, cRowReg)
    cState := cIdle
  }
  when(cState === cExp2) {
    for (k <- 0 until L) { sNarX(k) := sExp2(k) }
    emitLanes((0 until L).map(k => sNar(k)), cMaskReg, cRowReg)
    cState := cIdle
  }
  // VSOFTEXP: stage 1 subtracted the scalar, so the two halves of exp follow.
  when(cState === cSoftE) {
    for (k <- 0 until L) { sExpX(k) := pipeX(k); expMid(k) := sExp1(k) }
    cState := cSoftE2
  }
  when(cState === cSoftE2) {
    val e = sExp2 // the shared stage-2 outputs ARE this row's exp results
    for (k <- 0 until L) { sNarX(k) := sExp2(k) }
    emitLanes((0 until L).map(k => sNar(k)), cMaskReg, cRowReg)
    loadTree(e, cMaskReg, cLastReg)
    cState := cIdle
  }

  when(cState === cRecipC1) {
    for (k <- 0 until L) {
      sAddX(k) := "h40000000".U(32.W); sAddY(k) := Bf16.negF32(pipeP(k))
      pipeC(k) := sAdd(k)
    }
    cState := cRecipY1
  }
  when(cState === cRecipY1) {
    for (k <- 0 until L) { sMulX(k) := pipeY(k); sMulY(k) := pipeC(k); pipeY(k) := sMul(k) }
    cState := cRecipP2
  }
  when(cState === cRecipP2) {
    for (k <- 0 until L) { sMulX(k) := pipeX(k); sMulY(k) := pipeY(k); pipeP(k) := sMul(k) }
    cState := cRecipC2
  }
  when(cState === cRecipC2) {
    for (k <- 0 until L) {
      sAddX(k) := "h40000000".U(32.W); sAddY(k) := Bf16.negF32(pipeP(k))
      pipeC(k) := sAdd(k)
    }
    cState := cRecipY2
  }
  when(cState === cRecipY2) {
    val out = (0 until L).map { k =>
      sMulX(k) := pipeY(k); sMulY(k) := pipeC(k)
      val y2 = sMul(k)
      val normal = Cat(recipSign(k), y2(30, 0))
      val infS = Cat(recipSign(k), POS_INF(30, 0))
      sNarX(k) := Mux(recipZero(k), infS,
        Mux(recipInf(k), Cat(recipSign(k), 0.U(31.W)), normal))
      sNar(k)
    }
    emitLanes(out, cMaskReg, cRowReg)
    cState := cIdle
  }

  when(cState === cRsqrtT) {
    for (k <- 0 until L) { sMulX(k) := pipeX(k); sMulY(k) := pipeP(k); pipeP(k) := sMul(k) }
    cState := cRsqrtC
  }
  when(cState === cRsqrtC) {
    for (k <- 0 until L) {
      sAddX(k) := "h3fc00000".U(32.W); sAddY(k) := Bf16.negF32(pipeP(k))
      pipeC(k) := sAdd(k)
    }
    cState := cRsqrtY
  }
  when(cState === cRsqrtY) {
    val out = (0 until L).map { k =>
      sMulX(k) := pipeY(k); sMulY(k) := pipeC(k)
      val y = sMul(k)
      sNarX(k) := Mux(rsqrtZero(k), POS_INF, Mux(rsqrtSign(k), QNAN, y))
      sNar(k)
    }
    emitLanes(out, cMaskReg, cRowReg)
    cState := cIdle
  }

  // ---- Reduction tree: advance every cycle, valid bits ride along -----------
  // Identical arithmetic and identical ordering to the stepped version, so the
  // fp32 result is bit-for-bit the same. (t0Valid's default is set above, before
  // the compute segment, so loadTree's pulse survives last-connect.)
  r1Valid := t0Valid; r1Last := t0Last
  when(t0Valid) {
    for (i <- 0 until 4) red1(i) := reduceCombine(terms(2 * i), terms(2 * i + 1))
  }
  r2Valid := r1Valid; r2Last := r1Last
  when(r1Valid) {
    for (i <- 0 until 2) red2(i) := reduceCombine(red1(2 * i), red1(2 * i + 1))
  }
  r3Valid := r2Valid; r3Last := r2Last
  when(r2Valid) {
    rowSum := reduceCombine(red2(0), red2(1))
  }
  when(r3Valid) {
    val nextAcc = reduceCombine(acc, rowSum)
    acc := nextAcc
    when(r3Last) { resultReg := nextAcc }
  }

  // ---- Retire --------------------------------------------------------------
  when(state === sRun) {
    when(allDone) {
      when(isReduce(curOp)) {
        state := sWriteRd
      }.otherwise {
        io.done := true.B
        state := sIdle
      }
    }
    // Shifted overlap between a source and the destination is not expressible in
    // a row pipeline (see the header comment); identical bases are fine.
    when(writesDst(curOp) && !isReduce(curOp)) {
      val span = rows << rowShift
      val aliases = (srcABase =/= dstBase) &&
        (srcABase < (dstBase + span)) && (dstBase < (srcABase + span))
      assert(!aliases,
        "VectorUnit: srcA overlaps dst without being identical to it; " +
        "row-pipelined execution makes that ordering undefined (docs/ISA.md 2.5.4)")
    }
  }

  when(state === sWriteRd) {
    when(io.rd.ready) {
      io.done := true.B
      state := sIdle
    }
  }

  // ---- Dispatch ------------------------------------------------------------
  // Placed last so its reset of the segment state overrides the assignments
  // above (all of which are gated on sRun and therefore inactive here anyway).
  // Gated on fire, not sIdle: req.ready also admits a back-to-back start on
  // the sRun/allDone cycle, and startCompute's resets must win over the retire
  // block's state := sIdle in exactly that case (last-connect, source order).
  def startCompute(op: AluOp.Type): Unit = {
    val startVlen = Mux(io.context.valid, io.context.vlen, vlen)
    curOp := op
    when(io.context.valid) {
      srcABase := io.context.srcA
      srcBBase := io.context.srcB
      dstBase := io.context.dst
      vlen := io.context.vlen
      scalarOp := io.context.scalar
    }
    rdReg := io.req.bits.addr
    fRow := 0.U; fSel := 0.U; fPtr := 0.U
    cPtr := 0.U; cOut := 0.U; cRows := 0.U
    sPtr := 0.U; sRows := 0.U
    cState := cIdle
    rdFire := false.B
    for (i <- 0 until 2) { ibValid(i) := false.B; obValid(i) := false.B }
    t0Valid := false.B; r1Valid := false.B; r2Valid := false.B; r3Valid := false.B
    acc := Mux(op === AluOp.VMAX, NEG_INF, 0.U(32.W))
    resultReg := Mux(op === AluOp.VMAX, NEG_INF, 0.U(32.W))
    when(startVlen === 0.U) {
      when(isReduce(op)) {
        state := sWriteRd
      }.otherwise {
        io.done := true.B
        state := sIdle
      }
    }.otherwise {
      state := sRun
    }
  }

  when(io.req.fire) {
    switch(io.req.bits.op) {
        is(AluOp.VSETA) { srcABase := io.rs1.data; io.done := true.B }
        is(AluOp.VSETB) { srcBBase := io.rs1.data; io.done := true.B }
        is(AluOp.VSETD) { dstBase := io.rs1.data; io.done := true.B }
        is(AluOp.VSETLEN) { vlen := io.rs1.data(15, 0); io.done := true.B }
        is(AluOp.VSETS) { scalarOp := io.rs1.data; io.done := true.B }
        is(AluOp.VADD) { startCompute(AluOp.VADD) }
        is(AluOp.VSUB) { startCompute(AluOp.VSUB) }
        is(AluOp.VMUL) { startCompute(AluOp.VMUL) }
        is(AluOp.VFMA) { startCompute(AluOp.VFMA) }
        is(AluOp.VSMUL) { startCompute(AluOp.VSMUL) }
        is(AluOp.VSADD) { startCompute(AluOp.VSADD) }
        is(AluOp.VEXP) { startCompute(AluOp.VEXP) }
        is(AluOp.VRECIP) { startCompute(AluOp.VRECIP) }
        is(AluOp.VRSQRT) { startCompute(AluOp.VRSQRT) }
        is(AluOp.VSUM) { startCompute(AluOp.VSUM) }
        is(AluOp.VMAX) { startCompute(AluOp.VMAX) }
        is(AluOp.VDOT) { startCompute(AluOp.VDOT) }
        is(AluOp.VSOFTEXP) { startCompute(AluOp.VSOFTEXP) }
        is(AluOp.VABS) { startCompute(AluOp.VABS) }
        is(AluOp.VCVT_I8) { startCompute(AluOp.VCVT_I8) }
        is(AluOp.VCVT_BF16) { startCompute(AluOp.VCVT_BF16) }
      }
  }
}
