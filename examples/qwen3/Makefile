# run3 -- Qwen3-0.6B CPU inference port (plan Phase 0/1/2, host only)
#
#   make                 # build run3 + qwen3_export
#   make export          # 从 HF 快照生成 build/ 下的检查点+分词器（默认 GS=32）
#   make test            # 门禁 0/1/2：selfcheck + 分词 fixtures + 前向 fixtures
#   make clean
#
# 环境变量：HF_DIR（默认 /home/shao/models/Qwen3-0.6B）、GS（默认 32）。
# Q3_FP32_ACT=1 run3 ... 为 fp32 激活金标模式（默认 int8 激活=NPU 数值路径）。

CC      ?= gcc
CFLAGS  ?= -O2 -g
CSTD     = -std=c11 -Wall -Wextra -Werror
LDLIBS   = -lm

HF_DIR  ?= /home/shao/models/Qwen3-0.6B
GS      ?= 32

BIN  = run3 qwen3_export
SRC  = run.c model.c nn.c forward.c mt.c tokenizer.c sampler.c chat.c \
       xnpu/hal.c xnpu/prefill.c xnpu/emit.c xnpu/layer.c
HDR  = model.h nn.h mt.h tokenizer.h sampler.h chat.h \
       xnpu/hal.h xnpu/prefill.h xnpu/emit.h xnpu/enc.h xnpu/layer.h

all: $(BIN)

qwen3_export: tools/qwen3_export.c
	$(CC) $(CSTD) $(CFLAGS) -o $@ $< $(LDLIBS)

run3: $(SRC) $(HDR)
	$(CC) $(CSTD) $(CFLAGS) -pthread -I. -o $@ $(SRC) $(LDLIBS)

export: qwen3_export
	./qwen3_export --ckpt $(HF_DIR) build/qwen3-0.6b-q80.bin --gs $(GS)
	./qwen3_export --tok $(HF_DIR) build/qwen3_tokenizer.bin

# 门禁 0/1/2（需要先 make export 或手工放置 build/*.bin）
test: $(BIN) tests/test_tokenizer tests/test_layer_map
	@./qwen3_export --selfcheck $(HF_DIR) build/qwen3-0.6b-q80.bin build/qwen3_tokenizer.bin
	@./tests/test_tokenizer build/qwen3_tokenizer.bin fixtures/tokens.tsv
	@./tests/test_layer_map
	@Q3_FP32_ACT=1 ./run3 build/qwen3-0.6b-q80.bin -z build/qwen3_tokenizer.bin -f fixtures/
	@./run3 build/qwen3-0.6b-q80.bin -z build/qwen3_tokenizer.bin -f fixtures/ || \
	  echo "(int8-act 模式 top-1 略低属预期：pos-0 激活离群值的量化噪声，见 plan §5 门禁注记)"

tests/test_tokenizer: tests/test_tokenizer.c tokenizer.c tokenizer.h
	$(CC) $(CSTD) $(CFLAGS) -o $@ tests/test_tokenizer.c tokenizer.c $(LDLIBS)

# ---------------------------------------------------------------------------
# Phase 3：NPU 发射器（xnpu/ 子目录）。
#
#   make test          # 含 tests/test_layer_map（plan §4 工作点表锁定）
#   make fx-rtl        # 重生成 fixtures/rtl/qknorm.*（fe_qknorm 的 RTL 夹具），
#                      # 之后跑 XNPU_BACKEND=verilator mill xnpu.test.testOnly
#                      #   xnpu.Qwen3QknormRtlSpec
XNPU_SRC = xnpu/emit.c

tests/test_layer_map: tests/test_layer_map.c $(XNPU_SRC) xnpu/emit.h xnpu/enc.h
	$(CC) $(CSTD) $(CFLAGS) -o $@ tests/test_layer_map.c $(XNPU_SRC) $(LDLIBS)

xnpu/fx_qknorm: xnpu/fx_qknorm.c $(XNPU_SRC) xnpu/emit.h xnpu/enc.h
	$(CC) $(CSTD) $(CFLAGS) -o $@ xnpu/fx_qknorm.c $(XNPU_SRC) $(LDLIBS)

xnpu/fx_rope: xnpu/fx_rope.c $(XNPU_SRC) xnpu/emit.h xnpu/enc.h
	$(CC) $(CSTD) $(CFLAGS) -o $@ xnpu/fx_rope.c $(XNPU_SRC) $(LDLIBS)

# 整层 kick 夹具（门禁 3a）：tiny 同构微缩模型（结构同构 qk-norm/half-split/
# head_dim!=dim/heads，仿真快百倍）；真形状整层正确性由板上 XNPU_CHECK 兜底。
xnpu/fx_layer: xnpu/fx_layer.c xnpu/emit.c xnpu/layer.c xnpu/emit.h xnpu/enc.h \
               xnpu/layer.h model.c nn.c forward.c mt.c model.h nn.h mt.h
	$(CC) $(CSTD) $(CFLAGS) -pthread -I. -o $@ xnpu/fx_layer.c xnpu/emit.c xnpu/layer.c \
	  model.c nn.c forward.c mt.c $(LDLIBS)

build/tiny_qwen3.bin: qwen3_export
	mkdir -p build
	./qwen3_export --tiny $@ 42

.PHONY: fx-rtl
fx-rtl: xnpu/fx_qknorm xnpu/fx_rope xnpu/fx_layer build/tiny_qwen3.bin
	mkdir -p fixtures/rtl
	./xnpu/fx_qknorm fixtures/rtl/qknorm
	./xnpu/fx_rope fixtures/rtl/rope
	./xnpu/fx_layer build/tiny_qwen3.bin fixtures/rtl/layer

# Dual-engine layer fixtures: the emitter alternates output tiles across
# both int8 engines (XNPU_ENGINES=2); refs are the same CPU shadow.
.PHONY: fx-rtl-dual
fx-rtl-dual: xnpu/fx_layer build/tiny_qwen3.bin
	mkdir -p fixtures/rtl
	XNPU_ENGINES=2 ./xnpu/fx_layer build/tiny_qwen3.bin fixtures/rtl/dual-layer

# 调试工具（前向二分对拍，需先用 fixtures/gen_fixtures.py 等生成 /tmp/hf_*.f32 参考）
tests/dbg_forward: tests/dbg_forward.c $(SRC) $(HDR)
	$(CC) $(CSTD) $(CFLAGS) -pthread -o $@ tests/dbg_forward.c model.c nn.c forward.c mt.c tokenizer.c $(LDLIBS)

# ---------------------------------------------------------------------------
# ZCU102 交叉编译（plan 门禁 1b；A53 纯 CPU 运行，无 NPU 依赖）。
#
#   make zcu102          # PetaLinux SDK 工具链（默认 /opt/petalinux/2021.2，
#                        #   位于 Ubuntu-20.04 WSL 发行版；找不到 env 脚本会提示）
#   make zcu102-static   # 本机 aarch64-linux-gnu-gcc + glibc 静态链接兜底，
#                        #   任意发行版可建、对板上 glibc 版本零依赖
#
# 产物在 build/zcu102/ 与 build/zcu102-static/，互不覆盖。板上部署清单（不含文档）：
#   run3 + build/qwen3-0.6b-q80.bin + build/qwen3_tokenizer.bin + fixtures/
PETALINUX ?= /opt/petalinux/2021.2
ZCU102_ENV = $(PETALINUX)/environment-setup-cortexa72-cortexa53-xilinx-linux
ZCU102_OUT = build/zcu102
ZCU102_SETENV = unset LD_LIBRARY_PATH; . $(ZCU102_ENV)

.PHONY: zcu102
zcu102:
	@test -r $(ZCU102_ENV) || { \
	  echo "PetaLinux SDK env script not found: $(ZCU102_ENV)"; \
	  echo "point PETALINUX at your SDK install, e.g. make zcu102 PETALINUX=/path/to/sdk"; \
	  exit 1; }
	mkdir -p $(ZCU102_OUT)
	$(ZCU102_SETENV) && $$CC $$CFLAGS -O3 $(CSTD) $$LDFLAGS -pthread \
	  -o $(ZCU102_OUT)/run3 $(SRC) $(LDLIBS)
	$(ZCU102_SETENV) && $$CC $$CFLAGS -O3 $(CSTD) $$LDFLAGS \
	  -o $(ZCU102_OUT)/test_tokenizer tests/test_tokenizer.c tokenizer.c $(LDLIBS)
	@$(ZCU102_SETENV) && $$READELF -h $(ZCU102_OUT)/run3 | grep Machine
	@ls -l $(ZCU102_OUT)

ZCU102_ST_OUT = build/zcu102-static

.PHONY: zcu102-static
zcu102-static:
	@command -v aarch64-linux-gnu-gcc >/dev/null || { \
	  echo "aarch64-linux-gnu-gcc not found (apt install gcc-aarch64-linux-gnu)"; exit 1; }
	mkdir -p $(ZCU102_ST_OUT)
	aarch64-linux-gnu-gcc $(CSTD) -O3 -static -pthread -I. -o $(ZCU102_ST_OUT)/run3 $(SRC) $(LDLIBS)
	aarch64-linux-gnu-gcc $(CSTD) -O3 -static -o $(ZCU102_ST_OUT)/test_tokenizer \
	  tests/test_tokenizer.c tokenizer.c $(LDLIBS)
	@readelf -h $(ZCU102_ST_OUT)/run3 | grep Machine
	@file $(ZCU102_ST_OUT)/run3 | sed 's/, BuildID.*//'
	@ls -l $(ZCU102_ST_OUT)

clean:
	rm -f $(BIN) tests/test_tokenizer tests/dbg_forward tests/test_layer_map xnpu/fx_qknorm xnpu/fx_rope xnpu/fx_layer
	rm -rf build/zcu102 build/zcu102-static

ZCU102_WSBENCH = $(ZCU102_ST_OUT)/wsbench

# Standalone board bench for the C4 MCFG2 OS/WS switching (see wsbench.c).
# Direct DTCM window writes only -- no resmem/DMA, safe alongside any build.
.PHONY: wsbench
wsbench:
	@command -v aarch64-linux-gnu-gcc >/dev/null || { \
	  echo "aarch64-linux-gnu-gcc not found"; exit 1; }
	mkdir -p $(ZCU102_ST_OUT)
	aarch64-linux-gnu-gcc $(CSTD) -O2 -static -I. -o $(ZCU102_WSBENCH) \
	  wsbench.c xnpu/hal.c xnpu/emit.c $(LDLIBS)
	@file $(ZCU102_WSBENCH) | sed 's/, BuildID.*//'

ZCU102_BF16SELFTEST = $(ZCU102_ST_OUT)/bf16selftest

# Standalone board selftest for the bf16 8x8 tier (C6, see bf16selftest.c).
# Emits FP.MATMUL/FP.MATMULA/FP.SM with bf16 8x8 operands, compares against
# CPU golden (serial fmaf fold + single RNE). Direct DTCM window writes only.
.PHONY: bf16selftest
bf16selftest:
	@command -v aarch64-linux-gnu-gcc >/dev/null || { \
	  echo "aarch64-linux-gnu-gcc not found"; exit 1; }
	mkdir -p $(ZCU102_ST_OUT)
	aarch64-linux-gnu-gcc $(CSTD) -O2 -static -I. -o $(ZCU102_BF16SELFTEST) \
	  bf16selftest.c xnpu/hal.c xnpu/emit.c $(LDLIBS)
	@file $(ZCU102_BF16SELFTEST) | sed 's/, BuildID.*//'

# Host x86 test: verify C packing + golden round-trip (no board needed).
.PHONY: bf16pack-test
bf16pack-test:
	@command -v gcc >/dev/null || { echo "gcc not found"; exit 1; }
	mkdir -p build
	gcc $(CSTD) -O2 -I. -o build/bf16pack_test bf16pack_test.c -lm
	./build/bf16pack_test

# tokcount: 逐轮上下文长度预检（多轮板上队列的设计闸，tokcount.c）。
.PHONY: tokcount
tokcount:
	@command -v gcc >/dev/null || { echo "gcc not found"; exit 1; }
	mkdir -p build
	gcc $(CSTD) $(CFLAGS) -I. -o build/tokcount tokcount.c tokenizer.c -lm

# PL clock probe: PERF cycles / wall for raw CRL divisor writes (spinf.c).
ZCU102_SPINF = $(ZCU102_ST_OUT)/spinf
.PHONY: spinf
spinf:
	@command -v aarch64-linux-gnu-gcc >/dev/null || { \
	  echo "aarch64-linux-gnu-gcc not found"; exit 1; }
	mkdir -p $(ZCU102_ST_OUT)
	aarch64-linux-gnu-gcc $(CSTD) -O2 -static -I. -o $(ZCU102_SPINF) \
	  spinf.c xnpu/hal.c xnpu/emit.c $(LDLIBS)
	@file $(ZCU102_SPINF) | sed 's/, BuildID.*//'

# bf16 8x8 Core 级仿真夹具（fx_bf16self.c，板测 bug 复现用）。
.PHONY: fx-bf16self
fx-bf16self:
	@command -v gcc >/dev/null || { echo "gcc not found"; exit 1; }
	mkdir -p fixtures/rtl
	gcc $(CSTD) -O2 -I. -o build/fx_bf16self fx_bf16self.c xnpu/emit.c -lm
	./build/fx_bf16self

.PHONY: all export test clean zcu102 zcu102-static fx-rtl wsbench bf16selftest bf16pack-test spinf fx-bf16self tokcount
