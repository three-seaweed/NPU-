// Copyright 2026 BUPT

package xnpu

import circt.stage.ChiselStage

import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters._

// Verilog/SystemVerilog generator entry point.
//
//   mill xnpu.runMain xnpu.Elaborate          # generates the Npu top (default)
//   mill xnpu.runMain xnpu.Elaborate Core      # generates a different module
//
// Output (.sv) is written under generated/.  A design containing a BlackBox (the
// DTCM bank) emits more than one source file, so see splitEmitted below.
object Elaborate extends App {
    val targetDir = "generated"

    // Parameters variant: the second CLI arg selects the build shape, so all
    // stay elaboration-checked.
    //   mill xnpu.runMain xnpu.Elaborate                # NpuTop, 单 int8 + bf16 8x8（默认）
    //   mill xnpu.runMain xnpu.Elaborate NpuTop int8    # int8-only 对照
    // fp32 4x4 tier 已删除（2026-11-01）：FP.* 一律由 bf16 8x8 核承担。
    // 2026-11-02 起默认上板形状 = 单 int8 + bf16 8x8（OOC ~246K LUT / 90%）；
    // 双 int8 + bf16 超 ZCU102（290K / 106%），只留 dual8 作纯双 int8 对照。
    def paramsFor(variant: String): Parameters = variant match {
        case "int8"  => Parameters()                          // int8-only 可综合对照
        case "dual8" => Parameters(int8Engines = 2)           // 纯双 int8：qwen3 性能位流
                                                              // （FP.* 全休眠面积/时序都不占）
        case "full"  => Parameters(bf16Matmul = true)         // 上板形状：单 int8 + bf16 8x8
        case "bf16"  => Parameters(bf16Matmul = true)         // 同上（面积探针别名）
        case other   => sys.error(s"unknown variant: $other")
    }

    // NOTE: the module must be constructed inside ChiselStage's elaboration
    // context, so build it lazily in a thunk rather than a pre-evaluated val.
    def build(): chisel3.RawModule = {
        val variant = args.drop(1).headOption.getOrElse("full")
        val p = paramsFor(variant)
        args.headOption.getOrElse("NpuTop") match {
            case "NpuTop"  => new NpuTop(p)
            case "Npu"     => new Npu(p)
            case "Core"    => new Core(p)
            case "Alu"        => new Alu(p)
            case "Matmul"     => new Matmul(p)
            case "VectorUnit" => new VectorUnit(p)
            case "Control"    => new Control(p)
            case "PipeS0"     => new fp32matmul.PipeS0
            case "PipeS1"     => new fp32matmul.PipeS1
            case "PipeS2"     => new fp32matmul.PipeS2
            case "PipeS2Exact" => new fp32matmul.PipeS2Exact
            case "PipeFull"   => new fp32matmul.PipeFull
            case "Bf16Store"   => new fp32matmul.Bf16Store(p)
            case "Bf16Feed"    => new fp32matmul.Bf16Feed(p)
            case "Bf16Load"    => new fp32matmul.Bf16Load(p)
            case "Bf16TierProbe" => new fp32matmul.Bf16TierProbe(p)
            case "VecAdd8"  => new fp32matmul.VecAdd8
            case "VecMul8"  => new fp32matmul.VecMul8
            case "VecExp8"  => new fp32matmul.VecExp8
            case "Bf16Shell" => new fp32matmul.Bf16Shell(p)
            case other        => sys.error(s"unknown top module: $other")
        }
    }

    // When the design pulls in BlackBox sources, firtool concatenates them into
    // the single output file, separated by "// ----- 8< ----- FILE ..." markers,
    // and appends a firrtl_black_box_resource_files.f *filelist*. That filelist is
    // a bare filename, not Verilog, so the combined file does not parse. Split the
    // sections back into real files and drop the filelist; Vivado then reads each
    // source directly, and the top file is exactly the module it claims to be.
    val Marker = """^// ----- 8< ----- FILE "(.*)" ----- 8< -----$""".r

    def splitEmitted(topFile: Path): Seq[Path] = {
        val lines = Files.readAllLines(topFile).asScala.toVector
        val sections = scala.collection.mutable.LinkedHashMap[String, Vector[String]]()
        var current = topFile.getFileName.toString
        sections(current) = Vector.empty
        for (line <- lines) line match {
            case Marker(name) =>
                current = Paths.get(name).getFileName.toString
                sections(current) = Vector.empty
            case other =>
                sections(current) = sections(current) :+ other
        }
        val dir = topFile.getParent
        sections.toSeq.collect {
            case (name, body) if !name.endsWith(".f") =>
                val out = dir.resolve(name)
                Files.write(out, (body.mkString("\n").trim + "\n").getBytes("UTF-8"))
                out
        }
    }

    val topName = args.headOption.getOrElse("NpuTop")
    ChiselStage.emitSystemVerilogFile(
        build(),
        args = Array("--target-dir", targetDir),
        firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info"),
    )
    val written = splitEmitted(Paths.get(targetDir, s"$topName.sv"))
    println(s"Elaborate: wrote ${written.map(_.toString).mkString(", ")}")
}
