// Copyright 2026 BUPT

package xnpu

import chiseltest._
import firrtl2.AnnotationSeq

// Simulation-backend selection for the test suite.
//
// Treadle (chiseltest's default) interprets the FIRRTL graph, which is fine for
// single-module specs but brutal on anything holding a Core: the 1024-PE array is
// evaluated every cycle even when the test never issues a MATMUL, so a full-Core
// spec runs at roughly one cycle per second (see GemvKernelSpec's header) and the
// end-to-end QuantLayerSpec case takes hours.
//
// Verilator compiles the design to C++ instead, which is far faster per cycle at
// the cost of a one-off build per module. Selection is a JVM system property so
// no spec has to hardcode a backend:
//
//   mill xnpu.test.testOnly '*VectorUnitSpec'                      # treadle (default)
//   XNPU_BACKEND=verilator mill xnpu.test.testOnly '*VectorUnitSpec'
//
// build.mill forwards XNPU_BACKEND into the forked test JVM as -Dxnpu.backend
// (via Task.env, so Mill actually invalidates on it -- sys.env would be cached
// and silently keep whichever backend the first evaluation picked).
//
// WHICH BACKEND TO USE:
//
//   * DtcmDualPortSpec MUST run under Verilator. Its DTCM bank is a BlackBox and
//     Treadle cannot simulate one: it warns "external module xnpu_dtcm_bank was
//     not matched with an implementation" and every read returns zero, so the
//     spec fails with plausible rdata=0 mismatches instead of an obvious error.
//
//   * The Core/Alu-sized specs depend on which Verilator is installed:
//
//       - 4.028 (what Ubuntu 20.04 ships) runs them. CoreSpec takes ~100s
//         including the one-off C++ build, against roughly a cycle per second
//         under Treadle, so prefer it. Verified on 4.028: CoreSpec, CtrlFlowSpec,
//         DmaSpec, LayerSpec.
//
//       - 5.x breaks them. chiseltest 6.0.0 drives Verilator's generated
//         makefile without building the precompiled headers 5.x emits, so every
//         split __Slow object dies on "VCore__pch.h.slow: No such file or
//         directory", and 5.020 has no --no-pch to turn that off. Fall back to
//         Treadle there. Small designs (DtcmRam) are unaffected either way
//         because they are not split into PCH-dependent objects.
//
// An unset/unrecognised value keeps chiseltest's default (Treadle).
trait XnpuBackend {
  // Pass to `test(...).withAnnotations(backendAnnos)`. Empty means "leave the
  // default alone" rather than "explicitly select Treadle", so chiseltest's own
  // default continues to apply.
  def backendAnnos: AnnotationSeq = {
    val backend = sys.props.get("xnpu.backend").map(_.toLowerCase) match {
      case Some("verilator") => Seq(VerilatorBackendAnnotation)
      case Some("treadle")   => Seq(TreadleBackendAnnotation)
      case _                 => Seq()
    }
    // XNPU_VCD=1 attaches a waveform writer to whatever backend was picked
    // (build.mill forwards it the same way as XNPU_BACKEND). Profiling-only:
    // specs stay unchanged and CI runs without it.
    val vcd = sys.props.get("xnpu.vcd").filter(_ != "0").map(_ => WriteVcdAnnotation).toSeq
    backend ++ vcd
  }
}
