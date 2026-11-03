/* forward.c -- Qwen3 CPU 前向（plan Phase 1）。
 * 逐条对齐 modeling_qwen3.py（transformers v4.51）与 conduct_docs/qwen3_0.6b_plan.md §1：
 *   坑1 head_dim 显式（q_dim=n_heads*head_dim=2048, kv_dim=n_kv*head_dim=1024）
 *   坑2 RoPE half-split：头内 j 与 j+hd/2 配对，freq_j = theta^(-2j/hd)，RoPE 前
 *   坑3 QK-Norm：wq/wk 输出逐头 rmsnorm(eps=1e-6)，RoPE 前
 * GQA q 头 h -> kv 头 h/kv_mul；score / sqrt(head_dim)；SwiGLU；tied logits。 */
#include "model.h"
#include "mt.h"
#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------- 相位计时（XNPU_MT_PROF=1；cpu_multicore_neon_decode_plan.md §11 余项 M2）
 * 只测墙钟，不碰数值路径；prof_on 为负表示未初始化，首次调用时读环境变量。
 * 目的：区分"仍可并行化的相位"与"已撞带宽/串行胶水"两类，再决定 M2 是否值得做。 */
enum { P_EMBED, P_QKV, P_QKROPE, P_ATTN, P_WO, P_FFN13, P_W2, P_FNORM, P_LMHEAD, P_N };
static const char *g_prof_name[P_N] = {
    "embed", "qkv-mm", "qknorm+rope+kv", "attn", "wo+resid",
    "ffn+w13+silu", "w2+resid", "finalnorm", "lm_head"};
static double g_prof_acc[P_N];
static long g_prof_tokens;
static int g_prof_on = -1;

static int prof_on(void) {
    if (g_prof_on < 0) {
        g_prof_on = getenv("XNPU_MT_PROF") ? 1 : 0;
        if (g_prof_on)
            atexit(q3_prof_dump); /* 首个 token 时注册，退出自动汇总 */
    }
    return g_prof_on;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double prof_mark(int k, double t0) {
    if (!g_prof_on)
        return t0;
    double t = now_s();
    g_prof_acc[k] += t - t0;
    return t;
}

void q3_prof_dump(void) {
    if (!prof_on())
        return;
    double tot = 0;
    for (int i = 0; i < P_N; i++)
        tot += g_prof_acc[i];
    fprintf(stderr, "[prof] tokens=%ld wall=%.3fs (%.4f s/tok)\n", g_prof_tokens, tot,
            g_prof_tokens ? tot / g_prof_tokens : 0.0);
    for (int i = 0; i < P_N; i++)
        fprintf(stderr, "[prof]   %-16s %8.3fs %5.1f%%\n", g_prof_name[i], g_prof_acc[i],
                tot > 0 ? 100.0 * g_prof_acc[i] / tot : 0.0);
}

/* debug hook（tests/dbg_forward.c 用）：stage: -1=embed 1=xnorm 2=q 3=k
 * 4=attnout 5=attn-resid 6=layer-end 7=ffn-norm 8=ffn-prod */
static void (*g_layer_dbg)(int stage, int layer, const float *x, int n) = NULL;
void q3_layer_dbg_set(void (*fn)(int, int, const float *, int)) { g_layer_dbg = fn; }

static float *g_rope_cos = NULL; /* [seq_len][hd/2] */
static float *g_rope_sin = NULL;
static int g_rope_len = 0;

static void rope_tables_build(const Config *c) {
    int half = c->head_dim / 2;
    if (g_rope_len == c->seq_len)
        return;
    free(g_rope_cos);
    free(g_rope_sin);
    g_rope_cos = malloc((size_t)c->seq_len * half * sizeof(float));
    g_rope_sin = malloc((size_t)c->seq_len * half * sizeof(float));
    if (!g_rope_cos || !g_rope_sin) {
        fprintf(stderr, "run3: oom rope tables\n");
        exit(1);
    }
    for (int pos = 0; pos < c->seq_len; pos++)
        for (int j = 0; j < half; j++) {
            float inv = powf(c->rope_theta, -(float)(2 * j) / (float)c->head_dim);
            float f = (float)pos * inv;
            g_rope_cos[(size_t)pos * half + j] = cosf(f);
            g_rope_sin[(size_t)pos * half + j] = sinf(f);
        }
    g_rope_len = c->seq_len;
}

/* half-split 旋转：vec 按 [n_heads][head_dim] 排列 */
static void rope_halfsplit(float *vec, int n_heads, int head_dim, int pos,
                           const float *cos_t, const float *sin_t) {
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float *v = vec + (size_t)h * head_dim;
        for (int j = 0; j < half; j++) {
            float c = cos_t[j], s = sin_t[j];
            float a = v[j], b = v[j + half];
            v[j] = a * c - b * s;
            v[j + half] = b * c + a * s;
        }
    }
    (void)pos;
}

static void dequant_row(float *dst, const QTensor *w, size_t row, int in, int gs) {
    const int8_t *q = w->q + row * (size_t)in;
    const float *s = w->s + row * (size_t)(in / gs);
    for (int i = 0; i < in; i++)
        dst[i] = (float)q[i] * s[i / gs];
}

/* ---------- 并行相位（cpu_multicore_neon_decode_plan.md §3） ----------
 * 切分维度：matmul 按输出行、attention 按 q 头——片内表达式与串行版逐字符
 * 一致 ⇒ 逐位不变（plan §2 定理 1）。量化 scratch 由主线程在 dispatch 前
 * 写一次、worker 只读（静态上界 in<=4096，同 nn.c 注记）。 */

static int8_t g_xq[4096];
static float g_xs[4096];

struct mm_job {
    float *y;
    const QTensor *w;
    const int8_t *xq;
    const float *xs;
    int out, in, gs;
};

static void mm_fn(int tid, void *ud) {
    struct mm_job *j = ud;
    int r0, r1;
    mt_range(tid, j->out, &r0, &r1);
    matmul_q8_rows(j->y, j->w, j->xq, j->xs, r0, r1, j->in, j->gs);
}

/* 单张量并行相位：量化一次 + 行分片；金标档回退原串行核（plan 非目标） */
static void mm(float *y, const QTensor *w, const float *x, int out, int in, int gs) {
    if (nn_fp32_act()) {
        matmul_q8(y, w, x, out, in, gs);
        return;
    }
    quantize_gs(g_xq, g_xs, x, in, gs);
    struct mm_job j = { y, w, g_xq, g_xs, out, in, gs };
    mt_run(mm_fn, &j);
}

/* wq/wk/wv 共享同一输入：量化一次，按 [q | k | v] 拼接输出行空间分片，
 * 每行仍由 matmul_q8_rows 以原表达式计算 ⇒ 与三笔独立串行调用逐位同 */
struct qkv_job {
    const QTensor *wq, *wk, *wv;
    float *q, *k, *v;
    const int8_t *xq;
    const float *xs;
    int q_dim, kv_dim, in, gs;
};

static void qkv_fn(int tid, void *ud) {
    struct qkv_job *j = ud;
    int r0, r1;
    mt_range(tid, j->q_dim + 2 * j->kv_dim, &r0, &r1);
    if (r0 < j->q_dim)
        matmul_q8_rows(j->q, j->wq, j->xq, j->xs, r0,
                       r1 < j->q_dim ? r1 : j->q_dim, j->in, j->gs);
    if (r1 > j->q_dim && r0 < j->q_dim + j->kv_dim) {
        int a = r0 > j->q_dim ? r0 : j->q_dim;
        int e = r1 < j->q_dim + j->kv_dim ? r1 : j->q_dim + j->kv_dim;
        matmul_q8_rows(j->k, j->wk, j->xq, j->xs, a - j->q_dim, e - j->q_dim,
                       j->in, j->gs);
    }
    if (r1 > j->q_dim + j->kv_dim) {
        int a = r0 > j->q_dim + j->kv_dim ? r0 : j->q_dim + j->kv_dim;
        matmul_q8_rows(j->v, j->wv, j->xq, j->xs, a - j->q_dim - j->kv_dim,
                       r1 - j->q_dim - j->kv_dim, j->in, j->gs);
    }
}

/* w1/w3 共享同一输入，同法拼接分片 */
struct w13_job {
    const QTensor *w1, *w3;
    float *hb, *hb2;
    const int8_t *xq;
    const float *xs;
    int hidden, in, gs;
};

static void w13_fn(int tid, void *ud) {
    struct w13_job *j = ud;
    int r0, r1;
    mt_range(tid, 2 * j->hidden, &r0, &r1);
    if (r0 < j->hidden)
        matmul_q8_rows(j->hb, j->w1, j->xq, j->xs, r0,
                       r1 < j->hidden ? r1 : j->hidden, j->in, j->gs);
    if (r1 > j->hidden) {
        int a = r0 > j->hidden ? r0 : j->hidden;
        matmul_q8_rows(j->hb2, j->w3, j->xq, j->xs, a - j->hidden, r1 - j->hidden,
                       j->in, j->gs);
    }
}

/* attention 按 q 头切分：头内 tpos 序不变，GQA 只读共享 K/V ⇒ 逐位同 */
struct attn_job {
    const Transformer *t;
    int l, pos;
};

static void attn_fn(int tid, void *ud) {
    struct attn_job *j = ud;
    const Transformer *t = j->t;
    const Config *c = &t->cfg;
    const RunState *s = &t->s;
    int hd = c->head_dim, kv_dim = XN_KV_DIM(c), kv_mul = XN_KV_MUL(c);
    int h0, h1;
    mt_range(tid, c->n_heads, &h0, &h1);
    float scale = 1.f / sqrtf((float)hd);
    for (int h = h0; h < h1; h++) {
        const float *qh = s->q + (size_t)h * hd;
        const float *kh_base =
            s->key_cache + ((size_t)j->l * c->seq_len) * kv_dim + (size_t)(h / kv_mul) * hd;
        float *att = s->att + (size_t)h * c->seq_len;
        for (int tpos = 0; tpos <= j->pos; tpos++) {
            const float *kh = kh_base + (size_t)tpos * kv_dim;
            float sc = 0.f;
            for (int i = 0; i < hd; i++)
                sc += qh[i] * kh[i];
            att[tpos] = sc * scale;
        }
        softmax_inplace(att, j->pos + 1);
        float *oh = s->xb2 + (size_t)h * hd; /* 复用 xb2 装注意力输出 */
        memset(oh, 0, hd * sizeof(float));
        for (int tpos = 0; tpos <= j->pos; tpos++) {
            const float *vh = s->value_cache +
                              ((size_t)j->l * c->seq_len + tpos) * kv_dim +
                              (size_t)(h / kv_mul) * hd;
            float a = att[tpos];
            for (int i = 0; i < hd; i++)
                oh[i] += a * vh[i];
        }
    }
}

/* 单 token 前向：写入 KV cache 于 pos，返回 logits（t->s.logits） */
float *forward(Transformer *t, int token, int pos) {
    const Config *c = &t->cfg;
    RunState *s = &t->s;
    Weights *w = &t->w;
    int dim = c->dim, hd = c->head_dim, gs = c->group_size;
    int q_dim = XN_Q_DIM(c), kv_dim = XN_KV_DIM(c);
    rope_tables_build(c);

    double t0 = prof_on() ? now_s() : 0.0;
    if (g_prof_on)
        g_prof_tokens++;
    dequant_row(s->x, &w->tokens, (size_t)token, dim, gs);
    t0 = prof_mark(P_EMBED, t0);
    if (g_layer_dbg)
        g_layer_dbg(-1, 0, s->x, dim);

    for (int l = 0; l < c->n_layers; l++) {
        /* attention block */
        rmsnorm_eps(s->xb, s->x, w->rms_att_weight + (size_t)l * dim, dim, c->rms_eps);
        if (g_layer_dbg)
            g_layer_dbg(1, l, s->xb, dim);
        if (nn_fp32_act()) { /* 金标档不并行（plan 非目标） */
            matmul_q8(s->q, &w->wq[l], s->xb, q_dim, dim, gs);
            matmul_q8(s->k, &w->wk[l], s->xb, kv_dim, dim, gs);
            matmul_q8(s->v, &w->wv[l], s->xb, kv_dim, dim, gs);
        } else {
            quantize_gs(g_xq, g_xs, s->xb, dim, gs);
            struct qkv_job j = { &w->wq[l], &w->wk[l], &w->wv[l], s->q, s->k, s->v,
                                 g_xq, g_xs, q_dim, kv_dim, dim, gs };
            mt_run(qkv_fn, &j);
        }
        t0 = prof_mark(P_QKV, t0);

        /* 坑3：QK-Norm 逐头（RoPE 前） */
        for (int h = 0; h < c->n_heads; h++)
            rmsnorm_eps(s->q + (size_t)h * hd, s->q + (size_t)h * hd,
                        w->q_norm_weight + (size_t)l * hd, hd, c->rms_eps);
        for (int h = 0; h < c->n_kv_heads; h++)
            rmsnorm_eps(s->k + (size_t)h * hd, s->k + (size_t)h * hd,
                        w->k_norm_weight + (size_t)l * hd, hd, c->rms_eps);

        /* 坑2：half-split RoPE */
        const float *cos_t = g_rope_cos + (size_t)pos * (hd / 2);
        const float *sin_t = g_rope_sin + (size_t)pos * (hd / 2);
        rope_halfsplit(s->q, c->n_heads, hd, pos, cos_t, sin_t);
        rope_halfsplit(s->k, c->n_kv_heads, hd, pos, cos_t, sin_t);
        if (g_layer_dbg) {
            g_layer_dbg(2, l, s->q, q_dim);
            g_layer_dbg(3, l, s->k, kv_dim);
        }

        /* KV cache 追加 */
        float *krow = s->key_cache + ((size_t)l * c->seq_len + pos) * kv_dim;
        float *vrow = s->value_cache + ((size_t)l * c->seq_len + pos) * kv_dim;
        memcpy(krow, s->k, kv_dim * sizeof(float));
        memcpy(vrow, s->v, kv_dim * sizeof(float));
        t0 = prof_mark(P_QKROPE, t0);

        /* attention：逐 q 头，GQA 共享 kv 头（按头切分并行） */
        struct attn_job aj = { t, l, pos };
        mt_run(attn_fn, &aj);
        t0 = prof_mark(P_ATTN, t0);
        if (g_layer_dbg)
            g_layer_dbg(4, l, s->xb2, q_dim);
        mm(s->xb, &w->wo[l], s->xb2, dim, q_dim, gs);
        for (int i = 0; i < dim; i++)
            s->x[i] += s->xb[i];
        t0 = prof_mark(P_WO, t0);
        if (g_layer_dbg)
            g_layer_dbg(5, l, s->x, dim);

        /* ffn block: down(silu(gate(x)) * up(x)) */
        rmsnorm_eps(s->xb, s->x, w->rms_ffn_weight + (size_t)l * dim, dim, c->rms_eps);
        if (g_layer_dbg)
            g_layer_dbg(7, l, s->xb, dim);
        if (nn_fp32_act()) {
            matmul_q8(s->hb, &w->w1[l], s->xb, c->hidden_dim, dim, gs);
            matmul_q8(s->hb2, &w->w3[l], s->xb, c->hidden_dim, dim, gs);
        } else {
            quantize_gs(g_xq, g_xs, s->xb, dim, gs);
            struct w13_job j = { &w->w1[l], &w->w3[l], s->hb, s->hb2, g_xq, g_xs,
                                 c->hidden_dim, dim, gs };
            mt_run(w13_fn, &j);
        }
        for (int i = 0; i < c->hidden_dim; i++) {
            float g = s->hb[i];
            s->hb[i] = g / (1.f + expf(-g)) * s->hb2[i]; /* silu(g)·up */
        }
        t0 = prof_mark(P_FFN13, t0);
        if (g_layer_dbg)
            g_layer_dbg(8, l, s->hb, c->hidden_dim);
        mm(s->xb, &w->w2[l], s->hb, dim, c->hidden_dim, gs);
        for (int i = 0; i < dim; i++)
            s->x[i] += s->xb[i];
        t0 = prof_mark(P_W2, t0);
        if (g_layer_dbg)
            g_layer_dbg(6, l, s->x, dim);
    }

    rmsnorm_eps(s->xb, s->x, w->rms_final_weight, dim, c->rms_eps);
    t0 = prof_mark(P_FNORM, t0);
    if (g_layer_dbg)
        g_layer_dbg(9, 0, s->xb, dim); /* 最终归一化后（对应 HF hs_28） */
    mm(s->logits, &w->tokens, s->xb, c->vocab_size, dim, gs);
    t0 = prof_mark(P_LMHEAD, t0);
    (void)t0;
    return s->logits;
}
