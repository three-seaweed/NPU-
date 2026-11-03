/* layer.h -- 层程序组装与权重打包（plan §5 Phase 3 第 2 级）。
 *
 * build_layer_program_qwen3 从 runq_xnpu.c :1515-1611 移植，Qwen3 适配五处：
 *   1. cosq/sinq 的 batch DMA 行数按 B*q_dim（runq 按 B*dim）；
 *   2. zero 向量按 max(dim,q_dim) 行；
 *   3. QKV 三发射后插入 q/k 增益装载（DW_QKG/DW_KKG）+ fe_qknorm×2；
 *   4. fe_rope -> fe_rope_half（q 段 L=q_dim、k 段 L=kv_dim）；
 *   5. wq out=q_dim、o_proj in=q_dim、fe_attention 的 q 行步进 q_dim*2。
 * pack_weight 从 :920-937 移植（GS 参数化）。 */
#ifndef Q3_LAYER_H
#define Q3_LAYER_H

#include <stdint.h>

#include "emit.h"

/* Descriptor word indices (byte offset = index*4 from m->desc).
 * runq :1030-1035 + DW_QKG/DW_KKG（Qwen3 qk 增益，head_dim bf16 各一）。 */
enum {
    DW_WQ = 0, DW_SWQ, DW_WK, DW_SWK, DW_WV, DW_SWV, DW_WO, DW_SWO,
    DW_W1, DW_SW1, DW_W2, DW_SW2, DW_W3, DW_SW3,
    DW_RMSA, DW_RMSF, DW_KC, DW_VC,
    DW_XBUF, DW_COSQ, DW_SINQ, DW_COSK, DW_SINK, DW_ZERO, DW_FLAGS,
    DW_QKG, DW_KKG, DW_NWORDS
};

#define DF_LOAD_BATCH  0x1u
#define DF_STORE_BATCH 0x2u

/* B_tile[r=in][c=out] = w_q[(ot*NTILE+c)*n + (kt*NTILE+r)]，tiles 按 (ot,kt)
 * 序；scales 按 [ot][g][c] bf16 序（每组 16 个连续，DTCM 32B 切片一读即得）。
 * d=out 行数，n=in 列数（w->q 是 out*n 行主）。 */
void pack_weight(const int8_t *q, const float *s, int d, int n, int gs,
                 int8_t *tiles, uint16_t *scales);

/* Assemble the whole layer: one kick, one HALT.  Returns program words, or
 * -1 on emitter overflow.  n_heads/n_kv derive from q_dim/kv_dim over
 * head_dim.  eps 传 rms_eps（Qwen3 1e-6）。 */
int build_layer_program_qwen3(uint32_t *prog, int cap, const LayerMap *m,
                              int B, int dim, int hidden, int q_dim, int kv_dim,
                              int head_dim, int start_pos, int gs, float eps,
                              int sched);

#endif /* Q3_LAYER_H */
