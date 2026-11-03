/* fx_layer -- 整层程序 RTL 仿真夹具生成器（Phase 3 第 2 级，门禁 3a）。
 *
 * 用真实 run3 CPU 前向（forward.c 逐 stage 调试钩子）产出参考值，用真实
 * pack_weight/build_layer_program_qwen3 产出 kick 内容——被测的就是发射器
 * 组合本身（描述符接线/级序/token 依赖/KV 发布/标志位），对拍基准与板上
 * XNPU_CHECK 相同（run3 自己的 CPU 影子，int8-act 档）。
 *
 * 用法：fx_layer <ckpt.bin> <prefix> [layer] [B] [start_pos] [T]
 *   默认 tiny 检查点（--tiny 导出）：layer=0 B=2 start_pos=3 T=start_pos+B。
 *   检查点须与发射器 gs 一致（q3k4 头部 group_size 自动读取）。
 *
 * 产物（<prefix> 默认 fixtures/rtl/layer）：
 *   .prog    u32：词数 + ITCM 词
 *   .ddr     DDR 字节镜像（x/cos/sin/zero/增益/KV 历史/7 矩阵 packed tiles）
 *   .mem_in  DTCM 字节镜像 [0, window)：仅描述符非零（其余全走 DMA）
 *   .txt     元数据（dims/offsets/ddr 布局/参考区位置）
 *   .ref.q/.ref.k/.ref.v/.ref.attn/.ref.swiglu/.ref.x   fp32 参考逐 (b) 行
 *
 * 参考采集：先跑 start_pos 个 warm-up token 填 KV 历史，再逐 token 跑
 * B 个 batch 位置，钩子抓 layer=LFIX 的 stage 2(q)/3(k)/4(attn)/6(层末 x)/
 * 8(swiglu)；v 行直接读 value_cache。warm-up 与 batch 的 token id 来自固定
 * LCG。参考是 fp32 CPU 影子，RTL 侧 bf16/int8——spec 用相对容差对拍。
 * 再生成：make fx-rtl。 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "nn.h"
#include "enc.h"
#include "emit.h"
#include "layer.h"

/* forward.c 的逐 stage 调试钩子（头文件未声明，见 forward.c:17-18） */
extern void q3_layer_dbg_set(void (*fn)(int, int, const float *, int));

#define MAXB 8

static uint32_t rng = 0xfa17c0deu;
static int tok_rng(int vocab) {
    rng = rng * 1664525u + 1013904223u;
    return (int)((rng >> 8) % (uint32_t)vocab);
}

/* ---- 参考采集 ---- */
static int g_layer, g_capturing;
static float *g_emb, *g_q, *g_k, *g_attn, *g_xend, *g_ffn;
static int g_b;

static void hook(int stage, int layer, const float *x, int n) {
    if (layer != g_layer || !g_capturing)
        return;
    float *dst = NULL;
    switch (stage) {
    case -1: dst = g_emb; break;      /* 嵌入 = 层 0 输入 */
    case 2:  dst = g_q; break;        /* q（rope 后） */
    case 3:  dst = g_k; break;        /* k（rope 后） */
    case 4:  dst = g_attn; break;     /* 注意力输出 */
    case 8:  dst = g_ffn; break;      /* silu(gate)*up */
    case 6:  dst = g_xend; break;     /* 层末残差 */
    default: return;
    }
    memcpy(dst + (size_t)g_b * n, x, (size_t)n * sizeof(float));
}

static void put_bf16(uint8_t *img, uint32_t off, const float *src, int n) {
    for (int i = 0; i < n; i++) {
        uint16_t h = f2bf16(src[i]);
        memcpy(img + off + (size_t)i * 2, &h, 2);
    }
}

static void dump(const char *path, const void *buf, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    if (n && fwrite(buf, 1, n, f) != n) { perror(path); exit(1); }
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <ckpt> <prefix> [layer] [B] [start_pos]\n", argv[0]); return 1; }
    const char *ckpt = argv[1], *prefix = argv[2];
    int lfix = argc > 3 ? atoi(argv[3]) : 0;
    int B = argc > 4 ? atoi(argv[4]) : 2;
    int start_pos = argc > 5 ? atoi(argv[5]) : 3;
    if (B < 1 || B > MAXB || lfix < 0 || start_pos < 0) { fprintf(stderr, "bad args\n"); return 1; }

    Transformer t;
    if (build_transformer(&t, ckpt, 0)) { fprintf(stderr, "load failed\n"); return 1; }
    const Config *c = &t.cfg;
    int dim = c->dim, hidden = c->hidden_dim, hd = c->head_dim, gs = c->group_size;
    int q_dim = XN_Q_DIM(c), kv_dim = XN_KV_DIM(c);
    int n_heads = c->n_heads, n_kv = c->n_kv_heads;
    int T = start_pos + B;
    float eps = c->rms_eps;
    printf("ckpt: dim=%d hidden=%d L=%d q=%d kv=%d hd=%d gs=%d theta=%g eps=%g\n",
           dim, hidden, c->n_layers, q_dim, kv_dim, hd, gs,
           (double)c->rope_theta, (double)eps);

    /* ---- CPU 影子：warm-up + B 个 batch 位置，钩子抓层 lfix ---- */
    size_t bqd = (size_t)B * q_dim, bkd = (size_t)B * kv_dim;
    g_emb = calloc((size_t)B * dim, 4); g_q = calloc(bqd, 4);
    g_k = calloc(bkd, 4); g_attn = calloc(bqd, 4);
    g_ffn = calloc((size_t)B * hidden, 4); g_xend = calloc((size_t)B * dim, 4);
    float *vref = calloc(bkd, 4);
    g_layer = lfix;
    q3_layer_dbg_set(hook);
    for (int p = 0; p < T; p++) {
        int tok = tok_rng(c->vocab_size);
        g_capturing = (p >= start_pos);
        g_b = p - start_pos;
        forward(&t, tok, p);
        if (g_capturing) {
            float *vrow = t.s.value_cache + ((size_t)lfix * c->seq_len + p) * kv_dim;
            memcpy(vref + (size_t)g_b * kv_dim, vrow, (size_t)kv_dim * 4);
        }
    }
    q3_layer_dbg_set(NULL);

    /* ---- layer map / 程序 ---- */
    LayerMap m = layer_map_qwen3(B, dim, hidden, q_dim, kv_dim, hd, T, gs, 0);
    if (!m.ok) { fprintf(stderr, "shape does not fit DTCM\n"); return 1; }
    static uint32_t prog[ITCM_WORDS];
    int nwords = build_layer_program_qwen3(prog, ITCM_WORDS, &m, B, dim, hidden,
                                           q_dim, kv_dim, hd, start_pos, gs, eps, 0);
    if (nwords < 0) { fprintf(stderr, "emitter overflow\n"); return 1; }
    printf("program: %d words\n", nwords);

    /* ---- DDR 镜像 ---- */
    uint32_t off = 0;
    uint32_t o_xbuf, o_cosq, o_sinq, o_cosk, o_sink, o_zero, o_rmsa, o_rmsf,
             o_qkg, o_kkg, o_kc, o_vc;
    uint32_t o_wt[7], o_ws[7];
#define ALLOC16(bytes) (off = (off + 15u) & ~15u, off += (bytes), off - (uint32_t)(bytes))
    o_xbuf = ALLOC16((uint32_t)B * dim * 2);
    o_cosq = ALLOC16((uint32_t)B * q_dim * 2);
    o_sinq = ALLOC16((uint32_t)B * q_dim * 2);
    o_cosk = ALLOC16((uint32_t)B * kv_dim * 2);
    o_sink = ALLOC16((uint32_t)B * kv_dim * 2);
    o_zero = ALLOC16((uint32_t)(dim > q_dim ? dim : q_dim) * 2);
    o_rmsa = ALLOC16((uint32_t)dim * 2);
    o_rmsf = ALLOC16((uint32_t)dim * 2);
    o_qkg = ALLOC16((uint32_t)hd * 2);
    o_kkg = ALLOC16((uint32_t)hd * 2);
    o_kc = ALLOC16((uint32_t)T * kv_dim * 2);
    o_vc = ALLOC16((uint32_t)T * kv_dim * 2);
    {   /* 7 矩阵：wq wk wv wo w1 w2 w3（d=out,n=in） */
        int ds[7] = { q_dim, kv_dim, kv_dim, dim, hidden, dim, hidden };
        int ns[7] = { dim, dim, dim, q_dim, dim, hidden, dim };
        for (int i = 0; i < 7; i++) {
            o_wt[i] = ALLOC16((uint32_t)(ds[i] / NTILE) * (ns[i] / NTILE) * TILE_BYTES);
            o_ws[i] = ALLOC16((uint32_t)(ds[i] / NTILE) * (ns[i] / gs) * NTILE * 2);
        }
    }
    uint8_t *ddr = calloc(off, 1);
    if (!ddr) { perror("calloc"); return 1; }
    put_bf16(ddr, o_xbuf, g_emb, B * dim);
    {   /* cos/sin 复制表：per (b,head) 128 项，角度用 batch 位置 */
        float *ct = malloc((size_t)hd * 4), *st = malloc((size_t)hd * 4);
        for (int b = 0; b < B; b++) {
            for (int j = 0; j < hd / 2; j++) {
                float w = powf(c->rope_theta, -(float)(2 * j) / (float)hd);
                float a = (float)(start_pos + b) * w;
                ct[j] = cosf(a); ct[j + hd / 2] = cosf(a);
                st[j] = -sinf(a); st[j + hd / 2] = sinf(a);
            }
            for (int h = 0; h < n_heads; h++) {
                put_bf16(ddr, o_cosq + (uint32_t)((b * n_heads + h) * hd) * 2, ct, hd);
                put_bf16(ddr, o_sinq + (uint32_t)((b * n_heads + h) * hd) * 2, st, hd);
            }
            for (int h = 0; h < n_kv; h++) {
                put_bf16(ddr, o_cosk + (uint32_t)((b * n_kv + h) * hd) * 2, ct, hd);
                put_bf16(ddr, o_sink + (uint32_t)((b * n_kv + h) * hd) * 2, st, hd);
            }
        }
        free(ct); free(st);
    }
    put_bf16(ddr, o_rmsa, t.w.rms_att_weight + (size_t)lfix * dim, dim);
    put_bf16(ddr, o_rmsf, t.w.rms_ffn_weight + (size_t)lfix * dim, dim);
    put_bf16(ddr, o_qkg, t.w.q_norm_weight + (size_t)lfix * hd, hd);
    put_bf16(ddr, o_kkg, t.w.k_norm_weight + (size_t)lfix * hd, hd);
    {   /* KV 历史（层 lfix 的 [0,start_pos) 行） */
        for (int p = 0; p < start_pos; p++) {
            const float *krow = t.s.key_cache + ((size_t)lfix * c->seq_len + p) * kv_dim;
            const float *vrow = t.s.value_cache + ((size_t)lfix * c->seq_len + p) * kv_dim;
            put_bf16(ddr, o_kc + (uint32_t)p * kv_dim * 2, krow, kv_dim);
            put_bf16(ddr, o_vc + (uint32_t)p * kv_dim * 2, vrow, kv_dim);
        }
    }
    {   /* packed tiles */
        const QTensor *ws[7] = { &t.w.wq[lfix], &t.w.wk[lfix], &t.w.wv[lfix], &t.w.wo[lfix],
                                 &t.w.w1[lfix], &t.w.w2[lfix], &t.w.w3[lfix] };
        int ds[7] = { q_dim, kv_dim, kv_dim, dim, hidden, dim, hidden };
        int ns[7] = { dim, dim, dim, q_dim, dim, hidden, dim };
        for (int i = 0; i < 7; i++)
            pack_weight(ws[i]->q, ws[i]->s, ds[i], ns[i], gs,
                        (int8_t *)(ddr + o_wt[i]), (uint16_t *)(ddr + o_ws[i]));
    }

    /* ---- DTCM 镜像（仅描述符非零）---- */
    uint8_t *dtcm = calloc(m.total, 1);
    uint32_t desc[DW_NWORDS] = { 0 };
    desc[DW_WQ] = o_wt[0]; desc[DW_SWQ] = o_ws[0];
    desc[DW_WK] = o_wt[1]; desc[DW_SWK] = o_ws[1];
    desc[DW_WV] = o_wt[2]; desc[DW_SWV] = o_ws[2];
    desc[DW_WO] = o_wt[3]; desc[DW_SWO] = o_ws[3];
    desc[DW_W1] = o_wt[4]; desc[DW_SW1] = o_ws[4];
    desc[DW_W2] = o_wt[5]; desc[DW_SW2] = o_ws[5];
    desc[DW_W3] = o_wt[6]; desc[DW_SW3] = o_ws[6];
    desc[DW_RMSA] = o_rmsa; desc[DW_RMSF] = o_rmsf;
    desc[DW_KC] = o_kc; desc[DW_VC] = o_vc;
    desc[DW_XBUF] = o_xbuf; desc[DW_COSQ] = o_cosq; desc[DW_SINQ] = o_sinq;
    desc[DW_COSK] = o_cosk; desc[DW_SINK] = o_sink; desc[DW_ZERO] = o_zero;
    desc[DW_FLAGS] = DF_LOAD_BATCH | DF_STORE_BATCH;
    desc[DW_QKG] = o_qkg; desc[DW_KKG] = o_kkg;
    memcpy(dtcm + m.desc, desc, DW_NWORDS * 4);

    /* ---- dump ---- */
    char path[512];
    FILE *f;
    uint32_t nw = (uint32_t)nwords;
    snprintf(path, sizeof path, "%s.prog", prefix);
    f = fopen(path, "wb");
    if (!f || fwrite(&nw, 4, 1, f) != 1 || fwrite(prog, 4, nwords, f) != (size_t)nwords) {
        perror(path); return 1;
    }
    fclose(f);
    snprintf(path, sizeof path, "%s.ddr", prefix);      dump(path, ddr, off);
    snprintf(path, sizeof path, "%s.mem_in", prefix);   dump(path, dtcm, m.total);
    snprintf(path, sizeof path, "%s.ref.q", prefix);    dump(path, g_q, bqd * 4);
    snprintf(path, sizeof path, "%s.ref.k", prefix);    dump(path, g_k, bkd * 4);
    snprintf(path, sizeof path, "%s.ref.v", prefix);    dump(path, vref, bkd * 4);
    snprintf(path, sizeof path, "%s.ref.attn", prefix); dump(path, g_attn, bqd * 4);
    snprintf(path, sizeof path, "%s.ref.swiglu", prefix); dump(path, g_ffn, (size_t)B * hidden * 4);
    snprintf(path, sizeof path, "%s.ref.x", prefix);    dump(path, g_xend, (size_t)B * dim * 4);
    snprintf(path, sizeof path, "%s.txt", prefix);
    f = fopen(path, "w");
    if (!f) { perror(path); return 1; }
    fprintf(f,
        "# layer kick RTL fixture (regen: make fx-rtl)\n"
        "layer %d\nB %d\nstart_pos %d\ngs %d\neps %g\ntheta %g\n"
        "dim %d\nhidden %d\nq_dim %d\nkv_dim %d\nhead_dim %d\nn_heads %d\nn_kv %d\n"
        "window %u\nddr_bytes %u\nn_words %u\n"
        "desc_off %u\nx_off %u\nq_off %u\natto_off %u\nh2_off %u\n"
        "ddr_xbuf %u\nddr_kc %u\nddr_vc %u\nkv_row_bytes %d\n",
        lfix, B, start_pos, gs, (double)eps, (double)c->rope_theta,
        dim, hidden, q_dim, kv_dim, hd, n_heads, n_kv,
        m.total, off, nw, m.desc, m.x, m.q, m.atto, m.h2,
        o_xbuf, o_kc, o_vc, kv_dim * 2);
    fclose(f);
    printf("fixture written to %s.{prog,ddr,mem_in,ref.*,txt} "
           "(window %u B, ddr %u B, refs: q/k/v/attn/swiglu/x)\n",
           prefix, m.total, off);
    return 0;
}
