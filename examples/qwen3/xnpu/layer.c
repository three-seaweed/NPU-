/* layer.c -- pack_weight + build_layer_program_qwen3（见 layer.h 头注的
 * 移植来源与五处 Qwen3 适配）。 */
#include "layer.h"
#include "enc.h"

#include <string.h>

/* (runq_xnpu.c:920-937 verbatim，GS->gs 参数) */
void pack_weight(const int8_t *q, const float *s, int d, int n, int gs,
                 int8_t *tiles, uint16_t *scales) {
    int nrt = d / NTILE, nkt = n / NTILE, ngrp = n / gs;
    for (int ot = 0; ot < nrt; ot++) {
        for (int kt = 0; kt < nkt; kt++) {
            int8_t *tile = tiles + ((size_t)(ot * nkt + kt)) * TILE_BYTES;
            for (int c = 0; c < NTILE; c++) {
                size_t row = (size_t)(ot * NTILE + c) * n + (size_t)kt * NTILE;
                for (int r = 0; r < NTILE; r++)
                    tile[r * NTILE + c] = q[row + r];   /* verbatim int8 */
            }
        }
        for (int g = 0; g < ngrp; g++)
            for (int c = 0; c < NTILE; c++)
                scales[((size_t)ot * ngrp + g) * NTILE + c] =
                    f2bf16(s[(size_t)(ot * NTILE + c) * ngrp + g]);
    }
}

int build_layer_program_qwen3(uint32_t *prog, int cap, const LayerMap *m,
                              int B, int dim, int hidden, int q_dim, int kv_dim,
                              int head_dim, int start_pos, int gs, float eps,
                              int sched) {
    Emit e = { prog, 0, cap, 0 };
    int n_heads = q_dim / head_dim, n_kv = kv_dim / head_dim;
    uint32_t krow = m->kvh + (uint32_t)start_pos * (uint32_t)kv_dim * 2;
    uint32_t vrow = m->vvh + (uint32_t)start_pos * (uint32_t)kv_dim * 2;
    int zrows = (dim > q_dim ? dim : q_dim) * 2 / DTCM_ROW;

    /* Batch inputs arrive through the 128-bit master DMA once, in layer 0.
     * Later layers keep residual/RoPE data resident in DTCM and only patch the
     * per-layer descriptor before their single kick. */
    ep(&e, enc_lw(30, 0, (int)m->desc + DW_FLAGS * 4));
    ep(&e, enc_andi(29, 30, DF_LOAD_BATCH));
    int no_batch_load = ehole(&e);
    fe_dma_in(&e, m, DW_XBUF, m->x, (uint32_t)B * dim * 2u / DTCM_ROW);
    fe_dma_in(&e, m, DW_COSQ, m->cosq, (uint32_t)B * q_dim * 2u / DTCM_ROW);
    fe_dma_in(&e, m, DW_SINQ, m->sinq, (uint32_t)B * q_dim * 2u / DTCM_ROW);
    fe_dma_in(&e, m, DW_COSK, m->cosk, (uint32_t)B * kv_dim * 2u / DTCM_ROW);
    fe_dma_in(&e, m, DW_SINK, m->sink, (uint32_t)B * kv_dim * 2u / DTCM_ROW);
    fe_dma_in(&e, m, DW_ZERO, m->zero, (uint32_t)zrows);
    epatch_b(&e, no_batch_load, BR_BEQ, 29, 0);

    /* 0. Pull this layer's K/V history [0, start_pos) in from the DDR mirror. */
    {
        uint32_t hrows = (uint32_t)start_pos * (uint32_t)kv_dim * 2u / DTCM_ROW;
        fe_dma_in(&e, m, DW_KC, m->kvh, hrows);
        fe_dma_in(&e, m, DW_VC, m->vvh, hrows);
    }

    /* 1. attention RMSNorm: load this layer's gain, then normalise x -> s1 */
    ep(&e, enc_lw(1, 0, (int)m->desc + DW_RMSA * 4));
    eli(&e, 2, m->rmsw);
    ep(&e, enc_dmald(1, 2, dim / VLANES));
    ep(&e, enc_dmawait());
    fe_rmsnorm_eps(&e, m, m->x, m->s1, B, dim, eps);

    /* 2. Q, K, V. Each Linear quantizes the token rows in-kick.  Qwen3:
     * wq 的 out 是 q_dim（不是 dim）。 */
    fe_linear(&e, m, m->s1, m->q, B, dim, q_dim, DW_WQ, DW_SWQ, gs, sched);
    fe_linear(&e, m, m->s1, krow, B, dim, kv_dim, DW_WK, DW_SWK, gs, sched);
    fe_linear(&e, m, m->s1, vrow, B, dim, kv_dim, DW_WV, DW_SWV, gs, sched);

    /* 2b. QK-Norm（Qwen3 独有）：装载本层 q/k 增益（各 head_dim bf16，
     * 复用 rmsw 之后的 qkg 区），逐头归一，原地写回。 */
    ep(&e, enc_lw(1, 0, (int)m->desc + DW_QKG * 4));
    eli(&e, 2, m->qkg);
    ep(&e, enc_dmald(1, 2, head_dim * 2 / DTCM_ROW));
    ep(&e, enc_dmawait());
    ep(&e, enc_lw(1, 0, (int)m->desc + DW_KKG * 4));
    eli(&e, 2, m->qkg + (uint32_t)head_dim * 2);
    ep(&e, enc_dmald(1, 2, head_dim * 2 / DTCM_ROW));
    ep(&e, enc_dmawait());
    fe_qknorm(&e, m, m->q, m->q, m->qkg, B, n_heads, head_dim, eps);
    fe_qknorm(&e, m, krow, krow, m->qkg + (uint32_t)head_dim * 2, B, n_kv, head_dim, eps);

    /* 3. half-split RoPE on Q and on the K rows just written into the cache */
    fe_rope_half(&e, m, m->q, m->cosq, m->sinq, B, q_dim, head_dim);
    fe_rope_half(&e, m, krow, m->cosk, m->sink, B, kv_dim, head_dim);

    /* Publish this batch's K/V to the DDR mirror so later batches and the CPU
     * decode path see them.  DMAST is fire-and-forget; HALT drains it. */
    ep(&e, enc_lw(1, 0, (int)m->desc + DW_KC * 4));
    eli(&e, 2, krow);
    eli(&e, 3, (uint32_t)start_pos * (uint32_t)kv_dim * 2);
    ep(&e, enc_add(1, 1, 3));
    ep(&e, enc_dmast(1, 2, B * kv_dim * 2 / DTCM_ROW));
    ep(&e, enc_dmawait());
    ep(&e, enc_lw(1, 0, (int)m->desc + DW_VC * 4));
    eli(&e, 2, vrow);
    ep(&e, enc_add(1, 1, 3));
    ep(&e, enc_dmast(1, 2, B * kv_dim * 2 / DTCM_ROW));
    ep(&e, enc_dmawait());

    /* 4-8. QK^T, scale, causal mask, softmax, Attn@V -> atto */
    fe_attention(&e, m, B, n_heads, head_dim, n_heads / n_kv, q_dim, kv_dim,
                 start_pos);

    /* 9-10. output projection (in=q_dim) and the first residual */
    fe_linear(&e, m, m->atto, m->s2, B, q_dim, dim, DW_WO, DW_SWO, gs, sched);
    fe_residual(&e, m->x, m->s2, B * dim);

    /* 11. FFN RMSNorm */
    ep(&e, enc_lw(1, 0, (int)m->desc + DW_RMSF * 4));
    eli(&e, 2, m->rmsw);
    ep(&e, enc_dmald(1, 2, dim / VLANES));
    ep(&e, enc_dmawait());
    fe_rmsnorm_eps(&e, m, m->x, m->s1, B, dim, eps);

    /* 12. up and gate */
    fe_linear(&e, m, m->s1, m->h1, B, dim, hidden, DW_W3, DW_SW3, gs, sched); /* up   = w3(x) */
    fe_linear(&e, m, m->s1, m->h2, B, dim, hidden, DW_W1, DW_SW1, gs, sched); /* gate = w1(x) */

    /* 13. SwiGLU */
    fe_swiglu(&e, m, B, hidden);

    /* 14-15. down projection and the second residual */
    fe_linear(&e, m, m->h2, m->s2, B, hidden, dim, DW_W2, DW_SW2, gs, sched);
    fe_residual(&e, m->x, m->s2, B * dim);

    /* Only the last layer publishes the residual back to reserved DDR. HALT
     * then drains all units before STATUS.done becomes visible to the host. */
    ep(&e, enc_lw(30, 0, (int)m->desc + DW_FLAGS * 4));
    ep(&e, enc_andi(29, 30, DF_STORE_BATCH));
    int no_batch_store = ehole(&e);
    fe_dma_out(&e, m, DW_XBUF, m->x, (uint32_t)B * dim * 2u / DTCM_ROW);
    epatch_b(&e, no_batch_store, BR_BEQ, 29, 0);

    ep(&e, ENC_HALT);
    return e.bad ? -1 : e.n;
}
