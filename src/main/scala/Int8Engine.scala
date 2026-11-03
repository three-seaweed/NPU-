// Copyright 2026 BUPT

package xnpu

import chisel3._
import chisel3.util._


// One int8 matrix engine: the load / compute / store FSM driving an NxN
// output-stationary (OS) systolic array with the dormant WS datapath,
// extracted verbatim from Matmul.scala (C3 refactor). Matmul is now a shell
// that routes commands to engines and arbitrates the shared DTCM port; a
// second instance arrives with the C5 dual-engine commit.
//
// Implements five instructions, one matrix operation at a time:
//   LMA     - latch matrix A's DTCM base address (rs1); 1 cycle, no DTCM traffic
//   LMB     - latch matrix B's DTCM base address (rs1); 1 cycle, no DTCM traffic
//   MATMUL  - stream A and B in from DTCM *while* running the systolic feed
//             (load overlaps compute), clearing the accumulators first (C = A x B)
//   MATMULA - same fused load+compute but *keeps* the accumulators (C += A x B),
//             so a long dot product split into K-tiles accumulates in the array
//             and is read back once (SM), instead of once per tile
//   SM      - burst-store the N*N int32 results to DTCM[rs1], row-major
//
// Wide DTCM: one fixed 128-bit beat moves per cycle. An N-int8 array row spans
// BeatsPerArrayRow such beats (1 at N=16, 2 at N=32), so the array width is
// decoupled from the physical DTCM/AXI beat. A can optionally arrive as a 2N-byte
// bf16 activation row. The loader assembles its physical beats, applies an
// N-lane fp32-scale/RNE/saturating quantizer and feeds the existing int8 array
// buffer without writing an intermediate DTCM scratch tile.
//
// req is Decoupled: req.ready is high only when idle. LMA/LMB return to idle the
// same cycle (cheap latches); MATMUL and SM back-pressure the pipeline until done.
class Int8Engine(p: Parameters, engineId: Int = 0) extends Module {
    val n            = p.MatrixDim
    val accPerWord   = p.DtcmRowBits / p.AccBits    // int32 results per wide DTCM word (=4)
    val storeGroups  = n * n / accPerWord           // wide writes to store the result
    val groupsPerRow = n / accPerWord               // wide result-words per matrix row
    val beatsPerRow  = p.BeatsPerArrayRow            // physical beats per array row (1 or 2)
    require(p.ArrayRowBits % p.DtcmRowBits == 0,
        "an int8 array row (MatrixDim*ElemBits) must be a whole number of DTCM beats")
    require(n % accPerWord == 0,
        "matrix row must be a whole number of wide result-words")

    val io = IO(new Bundle {
        // Decode cycle. The Matmul shell already filtered FP.* commands.
        val req = Flipped(Decoupled(new AluCmd))
        // Execute cycle.
        val rs1 = Flipped(new RegfileReadDataIO)
        val rs2 = Flipped(new RegfileReadDataIO)   // MCFG2 mode bits ride rs2
        // Dedicated DTCM burst port (arbitrated by the Matmul shell).
        val dtcm = new DtcmPortIO(p)
        // Snapshot supplied by Alu's MatrixIdex queue for the command on req.
        val quant = Input(new MatrixQuantConfig(p))
        val context = Input(new MatrixContext(p))
        // Completion pulse consumed by the shell's tagged done merge.
        val done = Output(Bool())
        // PERF probe: engine busy, readable at PERF 0x64/0x68/0x6C.
        val busy = Output(Bool())
    })

    val sIdle :: sLoadCompute :: sStore :: sWLoad :: sWFeed :: Nil = Enum(5)
    val state = RegInit(sIdle)

    // Operand storage, row-major signed int8: reg(row)(col).
    val aReg = Reg(Vec(n, Vec(n, SInt(p.ElemBits.W))))
    val bReg = Reg(Vec(n, Vec(n, SInt(p.ElemBits.W))))

    // Counters must span the largest burst (storeGroups to store, row indices to load).
    val cntW    = log2Up(n * n + 1)
    val aBase   = RegInit(0.U(p.AddrBits.W))  // A base byte-address (latched by LMA)
    val bBase   = RegInit(0.U(p.AddrBits.W))  // B base byte-address (latched by LMB)
    val base    = RegInit(0.U(p.AddrBits.W))  // store base byte-address (SM)
    val quantReg = RegInit(0.U.asTypeOf(new MatrixQuantConfig(p)))
    // Tracks K tiles accumulated since the last group partial was spilled.
    val groupTilesDone = RegInit(0.U(16.W))
    val reqCnt  = RegInit(0.U(cntW.W))        // store: next result-group index
    val loadIsA = RegInit(true.B)             // load: alternate complete A/B rows
    val loadRow = RegInit(0.U(cntW.W))        // next logical A/B row
    val loadBeat = RegInit(0.U(2.W))          // beat within a row (A bf16 uses 4)
    val aRows   = RegInit(0.U(cntW.W))        // A rows loaded so far
    val bRows   = RegInit(0.U(cntW.W))        // B rows loaded so far
    val step    = RegInit(0.U(cntW.W))        // systolic feed-cycle counter

    // ---- WS mode state (latched by MCFG2; OS defaults keep C3 behaviour) ----
    // modeReg: bit0 = dataflow (0=OS 1=WS), bit1 = ws_keep. wLoaded marks
    // weights resident in the PE array since the last waterfall. MCFG2 clears
    // it: accepting any config invalidates resident weights (gemmini §6.3.1).
    val modeReg = RegInit(0.U(2.W))
    val wLoaded = RegInit(false.B)
    val wCnt    = RegInit(0.U(log2Up(2 * n).W))  // waterfall cycle count
    // WS output buffer: completed column sums land here from the bottom ps
    // edge (column j delivers row m on cycle m+n+j -- the collection is
    // column-diagonal), sign-extended to int32. SM drains it exactly like the
    // OS accumulators. outAcc selects MATMULA-style accumulation into it.
    val outBuf = Reg(Vec(n, Vec(n, SInt(p.AccBits.W))))
    val outAcc = RegInit(false.B)

    val array = Module(new SystolicArray(p))

    // ---- Defaults ------------------------------------------------------------
    // int8 侧驱动线（FSM 各状态覆写；端口胶合在文件末尾）
    val m8Valid = WireDefault(false.B)
    val m8Write = WireDefault(false.B)
    val m8Addr  = WireDefault(0.U(p.AddrBits.W))
    val m8Wdata = WireDefault(0.U(p.DtcmRowBits.W))
    val m8Wstrb = WireDefault(0.U(p.DtcmRowBytes.W))
    val m8Ready = WireDefault(false.B)
    val m8Rdata = WireDefault(0.U(p.DtcmRowBits.W))

    // int8 FSM 的完成脉冲。
    val int8Done = WireDefault(false.B)

    array.io.en    := false.B
    array.io.clear := false.B
    array.io.flush := false.B
    for (i <- 0 until n) {
        array.io.aIn(i) := 0.S
        array.io.bIn(i) := 0.S
    }
    // WS datapath stays dormant (OS-constant) until the C4 driver lands;
    // these defaults keep the PE wiring bit-identical to the OS-only build.
    array.io.mode  := false.B
    array.io.wLoad := false.B
    array.io.wClr  := false.B
    array.io.wCnt  := 0.U
    array.io.psIn  := VecInit(Seq.fill(n)(0.S(p.PsBits.W)))

    // ready 互锁：FSM 空闲才接受命令（shell 在此之上再叠加其它引擎的空闲）。
    io.req.ready := (state === sIdle)

    val groupSizeValid = io.quant.groupSize === 0.U ||
        io.quant.groupSize(log2Ceil(n) - 1, 0) === 0.U
    val requestedGroupTiles = Mux(io.quant.groupSize === 0.U, 1.U(16.W),
        io.quant.groupSize >> log2Ceil(n))
    val enforceGroup = io.quant.enable && io.quant.groupSize =/= 0.U

    // Narrow a wide counter to the exact width needed to index an n-entry Vec
    // (callers guarantee the value is already in [0, n)).
    def vi(x: UInt): UInt = x(log2Up(n) - 1, 0)

    // Unpack an assembled ArrayRowBits word into a full matrix row of n signed int8
    // elements (element k in bits [8k+7:8k]). All indices are static.
    def unpackFullRow(word: UInt): Vec[SInt] = {
        val row = Wire(Vec(n, SInt(p.ElemBits.W)))
        for (k <- 0 until n) {
            row(k) := word(p.ElemBits * (k + 1) - 1, p.ElemBits * k).asSInt
        }
        row
    }

    // ---- Idle / dispatch -----------------------------------------------------
    // LMA/LMB just latch the operand base addresses (1 cycle, no DTCM traffic);
    // MATMUL then streams A and B in *and* computes in one fused state, so the
    // load overlaps the systolic feed. SM stores the result.
    when (state === sIdle) {
        when (io.req.fire) {
            quantReg := io.quant
            switch (io.req.bits.op) {
                is (AluOp.LMA) { aBase := io.rs1.data }   // latch A base, stay idle
                is (AluOp.LMB) { bBase := io.rs1.data }   // latch B base, stay idle
                is (AluOp.MCFG2) {
                    // Engine config: rs2(2,1) = {ws_keep, dataflow}. A DATAFLOW
                    // switch invalidates resident weights and wipes the
                    // pipelines -- stale weights under a new dataflow are
                    // silent corruption (gemmini §6.3.1 flush-before-switch).
                    // Toggling ws_keep alone must NOT invalidate, else keep
                    // could never be armed after the initial waterfall load.
                    // Configs reach the engine only from the queue bottom, so
                    // the array is idle here and a 1-cycle pulse is safe.
                    val dfSwitch = io.rs2.data(1) =/= modeReg(0)
                    modeReg := io.rs2.data(2, 1)
                    when (dfSwitch) {
                        wLoaded := false.B
                        array.io.clear := true.B
                        array.io.flush := true.B
                        array.io.wClr  := true.B
                        // outBuf is the WS C accumulator; sStore drains but
                        // never zeroes it, and no other command touches it.
                        // Without this, a fresh WS chain after any earlier WS
                        // work silently sums the stale C (board wsbench
                        // 2026-10-25: residue survived even OS roundtrips).
                        for (m <- 0 until n; j <- 0 until n) outBuf(m)(j) := 0.S
                    }
                }
                is (AluOp.SM)  {
                    val activeGroup = quantReg.enable && quantReg.groupSize =/= 0.U
                    val activeGroupTiles = Mux(quantReg.groupSize === 0.U, 1.U(16.W),
                        quantReg.groupSize >> log2Ceil(n))
                    assert(!activeGroup || groupTilesDone === activeGroupTiles,
                        "Matmul: SM must write the int32 partial at a quant group boundary")
                    when (activeGroup) { groupTilesDone := 0.U }
                    base   := io.rs1.data
                    reqCnt := 0.U
                    state  := sStore
                }
                is (AluOp.MATMUL) {
                    assert(!enforceGroup || groupSizeValid,
                        "Matmul: quant groupSize must be a multiple of MatrixDim")
                    assert(!enforceGroup || groupTilesDone === 0.U,
                        "Matmul: previous quant group requires SM before a new MATMUL")
                    when (enforceGroup) { groupTilesDone := 1.U }
                    when (io.context.valid) {
                        aBase := io.context.aBase
                        bBase := io.context.bBase
                    }
                    loadIsA := true.B
                    loadRow := 0.U
                    loadBeat := 0.U
                    aRows  := 0.U
                    bRows  := 0.U
                    step   := 0.U
                    wCnt   := 0.U
                    when (modeReg(0)) {
                        // WS: stage tiles first (sWLoad), then waterfall + feed.
                        assert(!quantReg.aBf16,
                            "WS matmul takes int8 A rows (bf16-A loader is OS-only)")
                        when (modeReg(1)) {
                            assert(wLoaded, "WS-keep requires weights resident (MCFG2 invalidated or never loaded)")
                            state := sWLoad   // A only: B neither read nor captured
                        } .otherwise {
                            state := sWLoad   // A + B, then weight waterfall
                        }
                        outAcc := false.B
                    } .otherwise {
                        array.io.clear := true.B   // wipe accumulators before streaming
                        array.io.flush := true.B  // and start from a clean operand pipeline
                        state  := sLoadCompute
                    }
                }
                is (AluOp.MATMULA) {
                    // Accumulating matmul: same fused load+compute, but keep the
                    // accumulators (C += A x B). Still flush the operand pipeline
                    // so the previous tile's operands don't leak into this feed.
                    assert(!enforceGroup || groupSizeValid,
                        "Matmul: quant groupSize must be a multiple of MatrixDim")
                    assert(!enforceGroup || (groupTilesDone =/= 0.U &&
                        groupTilesDone < requestedGroupTiles),
                        "Matmul: MATMULA crossed a quant group boundary; spill with SM first")
                    when (enforceGroup) { groupTilesDone := groupTilesDone + 1.U }
                    when (io.context.valid) {
                        aBase := io.context.aBase
                        bBase := io.context.bBase
                    }
                    loadIsA := true.B
                    loadRow := 0.U
                    loadBeat := 0.U
                    aRows  := 0.U
                    bRows  := 0.U
                    step   := 0.U
                    wCnt   := 0.U
                    when (modeReg(0)) {
                        assert(!quantReg.aBf16,
                            "WS matmul takes int8 A rows (bf16-A loader is OS-only)")
                        when (modeReg(1)) {
                            assert(wLoaded, "WS-keep requires weights resident (MCFG2 invalidated or never loaded)")
                            state := sWLoad
                        } .otherwise {
                            state := sWLoad
                        }
                        outAcc := true.B
                    } .otherwise {
                        array.io.flush := true.B
                        state  := sLoadCompute
                    }
                }
            }
        }
    }

    // ---- Fused load + compute ------------------------------------------------
    // Stream A and B rows from DTCM (one wide read per cycle, interleaved) while
    // running the systolic feed gated on how many rows have arrived. The DTCM
    // load (port) thus overlaps the array compute.
    when (state === sLoadCompute) {
        // A/B rows stay interleaved. A is either the normal int8 row or a
        // bf16 row; B is always an int8 row. Keep the physical BF16 row width
        // derived from MatrixDim: 16 lanes use two DTCM beats, 32 lanes use four.
        // Keeping a row/beat cursor avoids a dynamic divider in the issue path.
        val bf16BeatsPerRow = p.MatrixDim * p.Bf16Bits / p.DtcmRowBits
        require((p.MatrixDim * p.Bf16Bits) % p.DtcmRowBits == 0,
            "a bf16 array row must be a whole number of DTCM beats")
        val aBeats = Mux(quantReg.aBf16,
            bf16BeatsPerRow.U(3.W), beatsPerRow.U(3.W))
        val curBeats = Mux(loadIsA, aBeats, beatsPerRow.U(3.W))
        val lastBeat = loadBeat === (curBeats - 1.U)
        val aRowBytes = Mux(quantReg.aBf16,
            (p.MatrixDim * (p.Bf16Bits / p.ByteBits)).U(p.AddrBits.W),
            p.ArrayRowBytes.U(p.AddrBits.W))
        val aAddr = aBase + loadRow * aRowBytes + loadBeat * p.DtcmRowBytes.U
        val bAddr = bBase + loadRow * p.ArrayRowBytes.U + loadBeat * p.DtcmRowBytes.U
        when (loadRow < n.U) {
            m8Valid := true.B
            m8Write := false.B
            m8Addr  := Mux(loadIsA, aAddr, bAddr)
            when (m8Ready) {
                when (lastBeat) {
                    loadBeat := 0.U
                    when (loadIsA) {
                        loadIsA := false.B
                    } .otherwise {
                        loadIsA := true.B
                        loadRow := loadRow + 1.U
                    }
                } .otherwise {
                    loadBeat := loadBeat + 1.U
                }
            }
        }
        // Capture returns one cycle after a read fires (tag what was issued). The
        // beats of one row return consecutively, low-to-high. Int8 rows assemble
        // to ArrayRowBits; bf16 A rows assemble to 2*ArrayRowBits then quantize.
        val fFired = RegNext(m8Valid && m8Ready && !m8Write, false.B)
        val fIsA   = RegNext(loadIsA, false.B)
        val fBf16  = RegNext(loadIsA && quantReg.aBf16, false.B)
        val fRow   = RegNext(loadRow, 0.U)
        val fLast  = RegNext(lastBeat, false.B)
        val asmWord = Reg(UInt(p.ArrayRowBits.W))
        val bf16AsmWord = Reg(UInt((p.MatrixDim * p.Bf16Bits).W))
        val mergedWord =
            if (beatsPerRow == 1) m8Rdata
            else Cat(m8Rdata, asmWord(p.ArrayRowBits - 1, p.DtcmRowBits))
        val bf16RowBits = p.MatrixDim * p.Bf16Bits
        val mergedBf16 = Cat(m8Rdata, bf16AsmWord(bf16RowBits - 1, p.DtcmRowBits))

        // Same conversion contract as VectorUnit.VCVT_I8: RNE, signed native
        // saturation and NaN -> zero. A zero dynamic/static scale naturally
        // produces a zero activation row, avoiding an amax==0 divide-by-zero.
        def f32ToI8RneSat(f: UInt): UInt = {
            val sign = f(31)
            val exp = f(30, 23)
            val sig = Cat(1.U(1.W), f(22, 0))
            val isNaN = exp === "hff".U && f(22, 0) =/= 0.U
            val tooSmall = exp < 126.U
            val tooBig = exp >= 135.U
            // Only exponents 126..134 need a round-to-nearest-even shift.
            // Narrowing this amount to five bits keeps the emitted variable
            // shift and sticky-mask logic bounded to the 24-bit significand.
            val inRoundRange = exp >= 126.U && exp <= 134.U
            val sh = Mux(inRoundRange, (150.U(8.W) - exp)(4, 0), 0.U(5.W))
            val shMinusOne = Mux(sh === 0.U, 0.U(5.W), sh - 1.U)
            val intMag = (sig >> sh)(8, 0)
            val guard = (sig >> shMinusOne)(0)
            // The RNE range has only nine possible masks (bits 15..23). Keep
            // every mask and mux input at the 24-bit significand width.
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
            val mag = intMag +& roundUp
            val maxMag = Mux(sign, 128.U(10.W), 127.U(10.W))
            val clamped = Mux(tooBig || mag > maxMag, maxMag, mag)
            val signed = Mux(sign, (~clamped(7, 0)) + 1.U, clamped(7, 0))
            Mux(tooSmall || isNaN, 0.U(8.W), signed)(7, 0)
        }
        val quantized = Wire(Vec(n, UInt(p.ElemBits.W)))
        for (k <- 0 until n) {
            val bf16 = mergedBf16(p.Bf16Bits * (k + 1) - 1, p.Bf16Bits * k)
            quantized(k) := f32ToI8RneSat(Bf16.mulF32(Bf16.toF32(bf16), quantReg.scale))
        }
        when (fFired) {
            when (fBf16) {
                bf16AsmWord := mergedBf16
                when (fLast) {
                    aReg(vi(fRow)) := unpackFullRow(quantized.asUInt)
                    aRows := aRows + 1.U
                }
            } .otherwise {
                asmWord := mergedWord
                when (fLast) {
                    val row = unpackFullRow(mergedWord)
                    when (fIsA) { aReg(vi(fRow)) := row; aRows := aRows + 1.U }
                    .otherwise  { bReg(vi(fRow)) := row; bRows := bRows + 1.U }
                }
            }
        }
        assert(!quantReg.aBf16 || quantReg.enable,
            "Matmul: bf16 A rows require the activation quantizer to be enabled")

        // Gated systolic feed: step s needs A and B rows 0..min(s,n-1) loaded.
        val need    = Mux(step >= (n - 1).U, n.U, step + 1.U)
        val canFeed = (aRows >= need) && (bRows >= need)
        when (canFeed) {
            array.io.en := true.B
            for (i <- 0 until n) {
                val wa = step - i.U
                val aValid = step >= i.U && wa < n.U
                array.io.aIn(i) := Mux(aValid, aReg(i)(vi(wa)), 0.S)

                val wb = step - i.U
                val bValid = step >= i.U && wb < n.U
                array.io.bIn(i) := Mux(bValid, bReg(vi(wb))(i), 0.S)
            }
            step := step + 1.U
            when (step === (p.SystolicRunCycles - 1).U) {
                int8Done := true.B
                state := sIdle
            }
        }
    }

    // ---- WS: stage, waterfall, feed -----------------------------------------
    // sWLoad has two phases over one row cursor: (1) sequential staging reads
    // -- A rows always, B rows too unless ws_keep left them resident (int8 A
    // only; the bf16-A loader stays an OS path); (2) the weight waterfall:
    // W row wCnt drives the top of every column, one row per cycle, riding
    // the PE b-chain (1 cycle/row) so W[i][j] reaches row i on count 2i --
    // exactly when PE(i,j) captures (wLoad && wCnt === 2i). en=1 with aIn=0
    // keeps the product zero, so the ps chain stays clean throughout.
    when (state === sWLoad) {
        val wantB = !modeReg(1)
        val totalRows = Mux(wantB, (2 * n).U, n.U)
        when (loadRow < totalRows) {
            m8Valid := true.B
            m8Write := false.B
            val isA = loadRow < n.U
            m8Addr := Mux(isA,
                aBase + loadRow * p.ArrayRowBytes.U,
                bBase + (loadRow - n.U) * p.ArrayRowBytes.U)
            when (m8Ready) { loadRow := loadRow + 1.U }
        } .elsewhen (wantB) {
            // Phase 2: waterfall the staged B tile into the PE weights.
            array.io.en    := true.B
            array.io.mode  := true.B
            array.io.wLoad := true.B
            array.io.wCnt  := wCnt
            for (j <- 0 until n) {
                array.io.bIn(j) := Mux(wCnt < n.U, bReg(vi(wCnt))(j), 0.S)
            }
            wCnt := wCnt + 1.U
            when (wCnt === (2 * n - 2).U) {
                wLoaded := true.B
                wCnt := 0.U
                state := sWFeed
            }
        } .otherwise {
            // ws_keep: weights already resident, straight to the feed.
            state := sWFeed
        }
        // Read data returns one cycle later; stage into the tile registers
        // (same RegNext alignment trick as the OS capture path).
        val wfFired = RegNext(m8Valid && m8Ready && !m8Write, false.B)
        val wfRow   = RegNext(loadRow, 0.U)
        when (wfFired) {
            when (wfRow < n.U) {
                aReg(vi(wfRow)) := unpackFullRow(m8Rdata)
            } .otherwise {
                bReg(vi(wfRow - n.U)) := unpackFullRow(m8Rdata)
            }
        }
    }

    // sWFeed: skew-feed A (transposed wave mapping: aIn(k) at cycle s carries
    // A[s-k][k], i.e. element A[m][k] enters row k on cycle m+k -- the same
    // ramp shape as OS with the staging indices swapped) while injecting ps
    // zeros at every column top. The zero injection needs no column skew: by
    // superposition, psOut(j) on cycle t equals C[t-n-j][j], so the bottom
    // edge delivers output row m of column j on cycle m+n+j (verified by
    // hand on n=2). Collection therefore indexes per column; the last element
    // (m=j=n-1) lands on cycle 3n-2.
    when (state === sWFeed) {
        array.io.en   := true.B
        array.io.mode := true.B
        for (k <- 0 until n) {
            val wa = step - k.U
            val aValid = step >= k.U && wa < n.U
            array.io.aIn(k) := Mux(aValid, aReg(vi(wa))(k), 0.S)
        }
        for (j <- 0 until n) {
            when (step >= (n + j).U && step < (2 * n + j).U) {
                val m = step - (n + j).U
                val ps = array.io.psOut(j)
                outBuf(vi(m))(j) := Mux(outAcc, outBuf(vi(m))(j) + ps, ps)
            }
        }
        step := step + 1.U
        when (step === (3 * n - 2).U) {
            int8Done := true.B
            state := sIdle
        }
    }

    // ---- Store ---------------------------------------------------------------
    // Wide DTCM: write accPerWord int32 results per cycle, row-major. Group g of
    // row i holds C(i, g*accPerWord .. +accPerWord-1).
    when (state === sStore) {
        when (reqCnt < storeGroups.U) {
            val row = reqCnt / groupsPerRow.U
            val grp = reqCnt % groupsPerRow.U
            val col0 = grp * accPerWord.U
            val wvec = Wire(Vec(accPerWord, UInt(p.AccBits.W)))
            for (e <- 0 until accPerWord) {
                // OS drains the PE accumulators; WS drains the output buffer
                // the bottom ps edge collected into (identical row-major
                // layout, so SM's contract is mode-agnostic).
                wvec(e) := Mux(modeReg(0),
                    outBuf(vi(row))(vi(col0 + e.U)),
                    array.io.acc(vi(row))(vi(col0 + e.U))).asUInt
            }
            m8Valid := true.B
            m8Write := true.B
            m8Addr  := base + (row * n.U + col0) * 4.U
            m8Wdata := wvec.asUInt
            m8Wstrb := ~0.U(p.DtcmRowBytes.W)
            when (m8Ready) {
                reqCnt := reqCnt + 1.U
                when (reqCnt === (storeGroups - 1).U) {
                    int8Done := true.B
                    state := sIdle
                }
            }
        }
    }

    // ---- Port glue (was Matmul's no-fp32-unit case) --------------------------
    io.dtcm.valid := m8Valid
    io.dtcm.write := m8Write
    io.dtcm.addr  := m8Addr
    io.dtcm.wdata := m8Wdata
    io.dtcm.wstrb := m8Wstrb
    m8Ready := io.dtcm.ready
    m8Rdata := io.dtcm.rdata

    io.done := int8Done
    io.busy := state =/= sIdle   // PERF probe (dual-overlap work)
}
