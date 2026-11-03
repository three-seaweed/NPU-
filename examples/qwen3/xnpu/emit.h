/* emit.h -- 发射器与 DTCM 布局（plan §5 Phase 3；Appendix D 纪律：
 * Emit/LayerMap 骨架从 examples/llama2.c/runq_xnpu.c :561-700 移植）。
 *
 * 与 runq layer_map 的差异（plan §4/Phase 3 实施记录第 0 级）：
 *   1. q/atto/s1/s2/cosq/sinq 按 q_dim 分配（Qwen3 q_dim=2048 ≠ dim）；
 *   2. sw/aq/gscale 的 max 公式显式纳入 q_dim 项（wq out=q_dim、o_proj in=q_dim）；
 *   3. 新增 qkg 区：q/k_norm 增益 2×head_dim×bf16（rmsw 被 attention 前的
 *      dim 尺寸增益占满，qk 增益必须独立成区——runq 无此算子故无此区）；
 *   4. zero 区扩到 max(dim,q_dim)：wq 输出行是 q_dim 长，越界零读会串区；
 *   5. GS 是参数（q3k4 头部 group_size 字段），不是全局变量。
 *
 * eps 同理是参数（Qwen3 rms_eps=1e-6，非 runq 硬编码的 1e-5）。 */
#ifndef Q3_EMIT_H
#define Q3_EMIT_H

#include <stdint.h>

/* 硬件常数（runq_xnpu.c:65-74，与 Parameters.scala 一致） */
#define NTILE           16     /* MatrixDim */
#define DTCM_ROW        16     /* DtcmRowBytes, one AXI beat */
#define VLANES          8      /* bf16 lanes per DTCM row */
#define TILE_BYTES      (NTILE * NTILE)        /* int8 weight tile = 256 B */
#define TILE_ROWS       (TILE_BYTES / DTCM_ROW) /* DMALD beats per int8 tile = 16 */
#define BF16_TILE_BYTES (NTILE * NTILE * 2)    /* bf16 A tile = 512 B */
#define RES_TILE_BYTES  (NTILE * NTILE * 4)    /* int32 result tile = 1024 B */
#define DTCM_BYTES      0x80000u               /* 512 KiB */
#define ITCM_WORDS      8192                   /* 32 KiB */
#define PREFILL_MAXB    NTILE                  /* tokens per prefill kick */

/* ---- program emitter (runq_xnpu.c:561) ---- */
typedef struct { uint32_t *p; int n, cap, bad; } Emit;

void ep(Emit *e, uint32_t insn);
void eli(Emit *e, int rd, uint32_t v);   /* li rd, v -> addi or lui+addi */
int  ehole(Emit *e);
void epatch_b(Emit *e, int at, int f3, int rs1, int rs2);
void epatch_j(Emit *e, int at);
void eb_back(Emit *e, int f3, int rs1, int rs2, int label);

/* ---- DTCM layout for one layer program (runq_xnpu.c:606, +qkg) ----
 * Computed at runtime from (B, dim, hidden, q_dim, kv_dim, T, gs). */
typedef struct {
    uint32_t desc, zero, sc, sw, rmsw, gscale;
    uint32_t cosq, sinq, cosk, sink;
    uint32_t x, s1, s2, q, atto;
    uint32_t aq, cb, ci, wb;
    uint32_t cb2, ci2;   /* XNPU_SCHED double buffers */
    uint32_t qkg;        /* NEW: q/k_norm gains, 2*head_dim bf16 */
    uint32_t kvh, vvh, att, attf, tmph;
    uint32_t h1, h2, h3;
    uint32_t total;
    int ok;
} LayerMap;

/* Build the DTCM map (q_dim-aware).  Returns ok=0 if the shape does not fit
 * 512 KiB, in which case the caller must stay on the CPU.  head_dim sizes the
 * qkg region (2 * head_dim bf16) and feeds the rope/qknorm emitters. */
LayerMap layer_map_qwen3(int B, int dim, int hidden, int q_dim, int kv_dim,
                         int head_dim, int Tmax, int gs, int sched);

/* QK-Norm: y = gain * x / sqrt(mean(x^2) + eps), per head of len=head_dim,
 * iterated B*n_heads times.  The gain pointer is FIXED across iterations
 * (Qwen3 q/k_norm weight is (head_dim,) broadcast over heads) exactly like
 * fe_rmsnorm's (runq_xnpu.c:1043); only the length, iteration count and eps
 * differ.  dst == src (in-place) is supported: each vector op reads a row
 * before writing it. */
void fe_qknorm(Emit *e, const LayerMap *m, uint32_t src, uint32_t dst,
               uint32_t gain, int B, int n_heads, int head_dim, float eps);

/* Half-split RoPE (Qwen3: theta=1e6, j pairs with j+head_dim/2 -- NOT llama2
 * interleave).  vec = B*L bf16 (L = q_dim or kv_dim, n_heads*L/head_dim heads);
 * cos_t/sin_t are the per-token replicated 128-entry tables with sin
 * pre-signed (first half -sin, second half +sin) so the vector part keeps the
 * v*cos + swap(v)*sin_signed decomposition.  swap() is a scalar 32-bit word
 * exchange loop (no lane permute in the vector unit; pre-arranging on the
 * host would break the single-kick property).  In-place on vec, s1/s2 used. */
void fe_rope_half(Emit *e, const LayerMap *m, uint32_t vec, uint32_t cos_t,
                  uint32_t sin_t, int B, int L, int head_dim);

/* ---- 以下六个为 fe_rmsnorm/fe_linear/fe_attention/fe_swiglu/fe_residual/
 * fe_dma_{in,out} 的移植（runq_xnpu.c :1043-1510），差异仅三处：
 *   1. fe_rmsnorm 的 eps 从硬编码 1e-5 改参数（Qwen3 rms_eps=1e-6）；
 *   2. fe_linear 的 GS 从全局变量改参数（q3k4 头 group_size 字段）；
 *   3. fe_attention 的 q 行步进 dim*2 改 q_dim*2（Qwen3 q_dim != dim），
 *      签名里的 dim 相应更名 q_dim。---- */
void fe_rmsnorm_eps(Emit *e, const LayerMap *m, uint32_t src, uint32_t dst,
                    int B, int dim, float eps);
void fe_dma_in(Emit *e, const LayerMap *m, int desc_word, uint32_t dtcm, uint32_t rows);
void fe_dma_out(Emit *e, const LayerMap *m, int desc_word, uint32_t dtcm, uint32_t rows);
void fe_linear(Emit *e, const LayerMap *m, uint32_t srctok, uint32_t dst,
               int B, int in, int out, int w_word, int sw_word, int gs, int sched);
void fe_attention(Emit *e, const LayerMap *m, int B, int n_heads, int head_dim,
                  int kv_mul, int q_dim, int kv_dim, int start_pos);
void fe_swiglu(Emit *e, const LayerMap *m, int B, int hidden);
void fe_residual(Emit *e, uint32_t dst, uint32_t src, int n);

#endif /* Q3_EMIT_H */
