/* prefill.c -- 见 prefill.h 头注。kick/描述符/ITCM 补丁机制自 runq_xnpu.c
 * :3130-3260 移植；窗口/主表/双 arena 为 Qwen3 新增（28 层 426MiB 重排
 * 超 256MiB resmem）。
 *
 * 顺序契约（板侧缓存一致性，违反即静默错数）：
 *   ① CPU 写 resmem（打包/表/嵌入）→ ② xn_evict() → ③ kick（DMA 读）
 *   ④ kick 完成 → ⑤ xn_evict() → ⑥ CPU 读回（DMA 写的结果）
 * arena 惰性重打包发生在 kick 循环内，故 stage_window 末尾自带 evict。 */
#include "prefill.h"
#include "hal.h"
#include "enc.h"
#include "emit.h"
#include "layer.h"
#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NARENA 2
#define MAXL 64

typedef struct {
    int staged;
    int tcap;                /* NPU 上下文上限（位置数），XNPU_TCAP 只能调低 */
    int minb;                /* prefill 接受的最小批 B（小于则回 CPU，见 prompt 守卫） */
    int check;               /* XNPU_CHECK：CPU 影子对拍 */
    float check_tol;
    int cur_B, cur_start, cur_T, itcm_valid;
    LayerMap map;
    uint32_t prog[ITCM_WORDS], loaded[ITCM_WORDS];
    /* 常驻 resmem */
    uint64_t cosq_p, sinq_p, cosk_p, sink_p, zero_p, xbuf_p, gains_p, kc_p, vc_p;
    uint8_t *cosq_v, *sinq_v, *cosk_v, *sink_v, *xbuf_v, *gains_v, *kc_v, *vc_v;
    size_t kv_layer_bytes;
    size_t gain_layer_bytes;
    /* 双 arena */
    struct {
        uint8_t *v;
        uint64_t p;
        int first_layer;     /* 当前内容的首层；-1 = 空 */
    } arena[NARENA];
    int win_layers;
    uint64_t wt_p[MAXL][7], ws_p[MAXL][7];
} Pf;

static Pf g_pf;

static void put_bf16(uint8_t *dst, const float *src, int n) {
    uint16_t *d = (uint16_t *)dst;
    for (int i = 0; i < n; i++) d[i] = f2bf16(src[i]);
}
static float get_bf16(const uint8_t *src, int idx) {
    uint16_t h;
    memcpy(&h, src + (size_t)idx * 2, 2);
    return bf162f(h);
}

/* cos/sin 主表一行（n_heads 个头，每头 hd 项，j 配 j+hd/2，sin 预签名） */
static void rope_table_row(uint8_t *cq, uint8_t *sq, int n_heads, int hd,
                           double theta, int pos) {
    for (int h = 0; h < n_heads; h++) {
        uint16_t *c = (uint16_t *)(cq + (size_t)h * hd * 2);
        uint16_t *s = (uint16_t *)(sq + (size_t)h * hd * 2);
        for (int j = 0; j < hd / 2; j++) {
            float w = (float)pow(theta, -2.0 * (double)j / (double)hd);
            float a = (float)pos * w;
            float fc = cosf(a), fs = sinf(a);
            c[j] = f2bf16(fc);  c[j + hd / 2] = f2bf16(fc);
            s[j] = f2bf16(-fs); s[j + hd / 2] = f2bf16(fs);
        }
    }
}

int xnpu_prefill_ready(void) {
    return g_xn.enabled && g_xn.prefill && g_pf.staged;
}

static int shape_ok(const Config *c, int B, int T) {
    int q_dim = XN_Q_DIM(c), kv_dim = XN_KV_DIM(c);
    uint64_t kv_rows = (uint64_t)B * kv_dim * 2u / DTCM_ROW;
    if ((uint64_t)B * c->dim > 65528u || (uint64_t)B * c->hidden_dim > 65528u ||
        (uint64_t)B * q_dim > 65528u || (uint64_t)T > 65528u || kv_rows > 2047u)
        return 0;
    return 1;
}

int xnpu_selftest_report(const Config *c) {
    int q_dim = XN_Q_DIM(c), kv_dim = XN_KV_DIM(c), hd = c->head_dim, gs = c->group_size;
    printf("[selftest] dim=%d hidden=%d q_dim=%d kv_dim=%d hd=%d gs=%d；"
           "B/T 工作点（layer_map 实算，DTCM %uB）\n",
           c->dim, c->hidden_dim, q_dim, kv_dim, hd, gs, DTCM_BYTES);
    for (int B = 1; B <= PREFILL_MAXB; B++) {
        int maxT = 0;
        for (int T = c->seq_len; T >= 1; T--) {
            if (!shape_ok(c, B, T)) continue;
            LayerMap m = layer_map_qwen3(B, c->dim, c->hidden_dim, q_dim, kv_dim,
                                         hd, T, gs, g_xn.sched);
            if (m.ok) { maxT = T; break; }
        }
        printf("[selftest]   B=%2d  maxT=%d%s\n", B, maxT,
               maxT ? "" : "  (放不下：回 CPU)");
        if (!maxT) break;
    }
    {
        int B = 4, T = c->seq_len < 40 ? c->seq_len : 40;
        LayerMap m = layer_map_qwen3(B, c->dim, c->hidden_dim, q_dim, kv_dim, hd,
                                     T, gs, g_xn.sched);
        int n = build_layer_program_qwen3(g_pf.prog, ITCM_WORDS, &m, B, c->dim,
                                          c->hidden_dim, q_dim, kv_dim, hd,
                                          0, gs, c->rms_eps, g_xn.sched);
        printf("[selftest] B=%d/T=%d: DTCM %u/%uB (%s)，层程序 %d/%d 词"
               "（ITCM 8192 是 selftest 硬门）\n",
               B, T, m.total, DTCM_BYTES, m.ok ? "fits" : "TOO LARGE",
               n, ITCM_WORDS);
        if (!m.ok || n < 0) return 1;
    }
    return 0;
}

int xnpu_prefill_setup(Transformer *t) {
    if (!g_xn.enabled || !g_xn.prefill) return -1;
    if (g_pf.staged) return 0;
    const Config *c = &t->cfg;
    if (c->n_layers > MAXL) return -1;
    int q_dim = XN_Q_DIM(c), kv_dim = XN_KV_DIM(c), hd = c->head_dim, gs = c->group_size;

    /* tcap 从工作点表实算（B=1 的 map 最大上下文），不再硬编码：旧的 64
     * 是 B=2 档，既截断了 B=1 档（~76），也与表脱钩（表变了不会跟着走）。
     * 注意 B=1 只是"放得下"，不是"跑得快"——kick 成本由矩阵 tile 数主导
     * （M<=16 时阵列时间与 B 无关），每 token = 28 kick x ~77ms / B：
     * B=4 0.64s < CPU 1.1s < B=2 1.08s < B=1 2.16s。XNPU_TCAP 只允许
     * 调低（调试用），调高会被物理上限钳住——防止又出现静默假通过。 */
    g_pf.tcap = 0;
    for (int T = c->seq_len; T >= 8; T--) {
        LayerMap m = layer_map_qwen3(1, c->dim, c->hidden_dim, q_dim, kv_dim,
                                     c->head_dim, T, gs, g_xn.sched);
        if (m.ok) { g_pf.tcap = T; break; }
    }
    if (!g_pf.tcap) return -1;   /* 连 B=1/T=8 都放不下：形状不支持 */
    {
        const char *tc = getenv("XNPU_TCAP");
        if (tc) {
            int v = atoi(tc);
            if (v >= 8 && v < g_pf.tcap) g_pf.tcap = v;
        }
    }
    /* 最小批策略：B=1 的 NPU prefill 吞吐只有 CPU 的一半（上注的账），
     * 低于 minb 的提示词回 CPU 并留声。XNPU_BFORCE 可越过（调试用）。 */
    g_pf.minb = 2;
    {
        const char *mb = getenv("XNPU_MINB");
        if (mb) {
            int v = atoi(mb);
            if (v >= 1 && v <= PREFILL_MAXB) g_pf.minb = v;
        }
    }
    g_pf.check = getenv("XNPU_CHECK") != NULL;
    g_pf.check_tol = 5e-2f;
    {
        const char *ct = getenv("XNPU_CHECK_TOL");
        if (ct) {
            float v = strtof(ct, NULL);
            if (v > 1e-4f && v < 1.0f) {
                g_pf.check_tol = v;
                fprintf(stderr, "[xnpu] XNPU_CHECK_TOL=%.4g（调整已打印留档）\n", (double)v);
            }
        }
    }

    size_t trow_q = (size_t)q_dim * 2, trow_k = (size_t)kv_dim * 2;
    size_t kv_bytes = (size_t)c->n_layers * g_pf.tcap * trow_k;
    g_pf.gain_layer_bytes = (size_t)(2 * c->dim + 2 * hd) * 2;
    uint8_t *zero_v;
    if (rm_alloc((size_t)g_pf.tcap * trow_q, &g_pf.cosq_v, &g_pf.cosq_p) ||
        rm_alloc((size_t)g_pf.tcap * trow_q, &g_pf.sinq_v, &g_pf.sinq_p) ||
        rm_alloc((size_t)g_pf.tcap * trow_k, &g_pf.cosk_v, &g_pf.cosk_p) ||
        rm_alloc((size_t)g_pf.tcap * trow_k, &g_pf.sink_v, &g_pf.sink_p) ||
        rm_alloc((size_t)q_dim * 2, &zero_v, &g_pf.zero_p) ||
        rm_alloc((size_t)g_pf.tcap * c->dim * 2, &g_pf.xbuf_v, &g_pf.xbuf_p) ||
        rm_alloc((size_t)c->n_layers * g_pf.gain_layer_bytes, &g_pf.gains_v, &g_pf.gains_p) ||
        rm_alloc(kv_bytes, &g_pf.kc_v, &g_pf.kc_p) ||
        rm_alloc(kv_bytes, &g_pf.vc_v, &g_pf.vc_p)) {
        fprintf(stderr, "[xnpu] resmem 布局失败（池 %lluKB 不够常驻区）\n",
                (unsigned long long)(g_xn.rm_size / 1024));
        return -1;
    }
    g_pf.kv_layer_bytes = (size_t)g_pf.tcap * trow_k;
    memset(zero_v, 0, (size_t)q_dim * 2);

    for (int p = 0; p < g_pf.tcap; p++) {
        rope_table_row(g_pf.cosq_v + (size_t)p * trow_q, g_pf.sinq_v + (size_t)p * trow_q,
                       c->n_heads, hd, (double)c->rope_theta, p);
        rope_table_row(g_pf.cosk_v + (size_t)p * trow_k, g_pf.sink_v + (size_t)p * trow_k,
                       c->n_kv_heads, hd, (double)c->rope_theta, p);
    }
    for (int l = 0; l < c->n_layers; l++) {
        uint8_t *g = g_pf.gains_v + (size_t)l * g_pf.gain_layer_bytes;
        put_bf16(g, t->w.rms_att_weight + (size_t)l * c->dim, c->dim);
        put_bf16(g + (size_t)c->dim * 2, t->w.rms_ffn_weight + (size_t)l * c->dim, c->dim);
        put_bf16(g + (size_t)(2 * c->dim) * 2, t->w.q_norm_weight + (size_t)l * hd, hd);
        put_bf16(g + (size_t)(2 * c->dim + hd) * 2, t->w.k_norm_weight + (size_t)l * hd, hd);
    }

    {
        int ds[7] = { q_dim, kv_dim, kv_dim, c->dim, c->hidden_dim, c->dim, c->hidden_dim };
        int ns[7] = { c->dim, c->dim, c->dim, q_dim, c->dim, c->hidden_dim, c->dim };
        size_t per_layer = 0;
        for (int i = 0; i < 7; i++)
            per_layer += (size_t)(ds[i] / NTILE) * (ns[i] / NTILE) * TILE_BYTES +
                         (size_t)(ds[i] / NTILE) * (ns[i] / gs) * NTILE * 2;
        /* stage_window 的矩阵间 64B 对齐垫片（每矩阵 ≤63B）不计入上式：
         * 不加余量则尾层 w3 尺度溢出到相邻 arena 头部、被邻居打包覆盖
         * （2026-10-18 板上 CHECK 实测炸点，worst rel 0.68 @L15p6）。 */
        per_layer += 7u * 64u;
        size_t avail = g_xn.rm_size - g_xn.rm_used - (1u << 20); /* 1MiB 余量 */
        int win = (int)(avail / NARENA / per_layer);
        if (win < 1) {
            fprintf(stderr, "[xnpu] resmem 剩余 %lluKB 装不下单层（需 %lluKB/层）\n",
                    (unsigned long long)(avail / 1024), (unsigned long long)(per_layer / 1024));
            return -1;
        }
        if (win > c->n_layers) win = c->n_layers;
        g_pf.win_layers = win;
        size_t ab = (size_t)win * per_layer;
        for (int a = 0; a < NARENA; a++) {
            if (rm_alloc(ab, &g_pf.arena[a].v, &g_pf.arena[a].p)) {
                fprintf(stderr, "[xnpu] arena %d 分配失败\n", a);
                return -1;
            }
            g_pf.arena[a].first_layer = -1;
        }
        fprintf(stderr, "[xnpu] staging: tcap=%d, %lluKB/层, win=%dx%d 层%s, "
                        "KV 镜像 %lluKB\n",
                g_pf.tcap, (unsigned long long)(per_layer / 1024), win, NARENA,
                win >= c->n_layers ? "（全量驻存）" : "",
                (unsigned long long)(kv_bytes * 2 / 1024));
    }
    xnpu_evict();   /* 常驻表/增益写入后、任何 DMA 读之前 */
    g_pf.staged = 1;
    return 0;
}

static int stage_window(int a, int l0, Transformer *t) {
    const Config *c = &t->cfg;
    int q_dim = XN_Q_DIM(c), kv_dim = XN_KV_DIM(c), gs = c->group_size;
    int l1 = l0 + g_pf.win_layers;
    if (l1 > c->n_layers) l1 = c->n_layers;
    uint64_t t0 = g_xn.prof ? xn_now_ns() : 0;
    uint8_t *base = g_pf.arena[a].v;
    uint64_t base_p = g_pf.arena[a].p;
    size_t off = 0;
    const QTensor *ws[7];
    int ds[7] = { q_dim, kv_dim, kv_dim, c->dim, c->hidden_dim, c->dim, c->hidden_dim };
    int ns[7] = { c->dim, c->dim, c->dim, q_dim, c->dim, c->hidden_dim, c->dim };
    for (int l = l0; l < l1; l++) {
        ws[0] = &t->w.wq[l]; ws[1] = &t->w.wk[l]; ws[2] = &t->w.wv[l];
        ws[3] = &t->w.wo[l]; ws[4] = &t->w.w1[l]; ws[5] = &t->w.w2[l];
        ws[6] = &t->w.w3[l];
        for (int i = 0; i < 7; i++) {
            size_t tb = (size_t)(ds[i] / NTILE) * (ns[i] / NTILE) * TILE_BYTES;
            size_t sb = (size_t)(ds[i] / NTILE) * (ns[i] / gs) * NTILE * 2;
            off = (off + 63u) & ~(size_t)63u;
            pack_weight(ws[i]->q, ws[i]->s, ds[i], ns[i], gs,
                        (int8_t *)(base + off), (uint16_t *)(base + off + tb));
            g_pf.wt_p[l][i] = base_p + off;
            g_pf.ws_p[l][i] = base_p + off + tb;
            off += tb + sb;
        }
    }
    g_pf.arena[a].first_layer = l0;
    xnpu_evict();   /* 重打包的权重是 kick 的 DMA 源 */
    if (g_xn.prof) g_xn.p_stage_ns += xn_now_ns() - t0;
    return 0;
}

static void stage_window_ensure(int l, Transformer *t) {
    int widx = l / g_pf.win_layers;
    int a = widx % NARENA;
    int l0 = widx * g_pf.win_layers;
    if (g_pf.arena[a].first_layer != l0) stage_window(a, l0, t);
}

/* ITCM 程序装载（仅补丁变更词，runq :3131-3154）。几何 (B,start,T) 任一
 * 变化都重建——map 的区域地址随 Tmax 变。 */
static int load_program(Transformer *t, int B, int start_pos, int T) {
    const Config *c = &t->cfg;
    if (g_pf.itcm_valid && g_pf.cur_B == B && g_pf.cur_start == start_pos &&
        g_pf.cur_T == T)
        return 0;
    g_pf.map = layer_map_qwen3(B, c->dim, c->hidden_dim, XN_Q_DIM(c), XN_KV_DIM(c),
                               c->head_dim, T, c->group_size, g_xn.sched);
    if (!g_pf.map.ok) return -1;
    int n = build_layer_program_qwen3(g_pf.prog, ITCM_WORDS, &g_pf.map, B, c->dim,
                                      c->hidden_dim, XN_Q_DIM(c), XN_KV_DIM(c),
                                      c->head_dim, start_pos, c->group_size,
                                      c->rms_eps, g_xn.sched);
    if (n < 0) {
        fprintf(stderr, "[xnpu] 层程序超 ITCM；CPU 路径\n");
        return -1;
    }
    /* 全量重写（2026-10-18 板上实测：变更词补丁会让批>=1 全错——补丁的
     * loaded[]=0 假设与 ITCM 加电残留/前程序尾词不符，值为 0 的词（HALT
     * 即 0）被跳写。全量 2K 词/kick 的 MMIO 开销相对 15.8MB 权重流可忽
     * 略；n 之后补一个 HALT 覆盖程序变短时的旧尾）。 */
    int writes = 0;
    for (int i = 0; i < n; i++) {
        xn_wr(XN_ITCM + (uint32_t)i * 4, g_pf.prog[i]);
        g_pf.loaded[i] = g_pf.prog[i];
        writes++;
    }
    xn_wr(XN_ITCM + (uint32_t)n * 4, ENC_HALT);
    g_xn.p_itcm_words += (uint64_t)writes;
    g_pf.itcm_valid = 1;
    g_pf.cur_B = B;
    g_pf.cur_start = start_pos;
    g_pf.cur_T = T;
    if (g_xn.prof)
        fprintf(stderr, "[xnpu] 层程序 %d 词，%d ITCM 写（B=%d start=%d T=%d）\n",
                n, writes, B, start_pos, T);
    return 0;
}

/* CHECK 影子：快照取自 CPU cache（须在 NPU 回灌覆盖之前调用）。 */
static void check_against_shadow(Transformer *t, int n) {
    const Config *c = &t->cfg;
    int kv_dim = XN_KV_DIM(c);
    float *snap = malloc((size_t)c->n_layers * n * kv_dim * 2 * sizeof(float));
    if (!snap) return;
    for (int l = 0; l < c->n_layers; l++)
        for (int p = 0; p < n; p++) {
            size_t loff = ((size_t)l * c->seq_len + p) * kv_dim;
            memcpy(snap + ((size_t)l * n + p) * kv_dim * 2,
                   t->s.key_cache + loff, (size_t)kv_dim * sizeof(float));
            memcpy(snap + ((size_t)l * n + p) * kv_dim * 2 + kv_dim,
                   t->s.value_cache + loff, (size_t)kv_dim * sizeof(float));
        }
    xnpu_evict();
    int bad = 0;
    double worst_rel = 0.0, worst_nrmse = 0.0;
    int worst_l = -1, worst_p = -1;
    int shown = 0;
    for (int l = 0; l < c->n_layers; l++) {
        const uint8_t *kc = g_pf.kc_v + (size_t)l * g_pf.kv_layer_bytes;
        const uint8_t *vc = g_pf.vc_v + (size_t)l * g_pf.kv_layer_bytes;
        double lrel = 0.0;
        int lp = -1;
        for (int p = 0; p < n; p++) {
            const float *ref = snap + ((size_t)l * n + p) * kv_dim * 2;
            for (int half = 0; half < 2; half++) {
                const uint8_t *mirror = (half ? vc : kc) + (size_t)p * kv_dim * 2;
                double mx = 0, se = 0, sn = 0;
                for (int i = 0; i < kv_dim; i++) {
                    double e = ref[half * kv_dim + i], g = get_bf16(mirror, i);
                    if (fabs(e) > mx) mx = fabs(e);
                    se += (g - e) * (g - e);
                    sn += e * e;
                }
                double rel = mx > 1e-6 ? sqrt(se / kv_dim) / mx : 0;
                double nrmse = sn > 1e-12 ? sqrt(se / kv_dim) / sqrt(sn / kv_dim) : 0;
                if (rel > worst_rel) { worst_rel = rel; worst_l = l; worst_p = p; }
                if (nrmse > worst_nrmse) worst_nrmse = nrmse;
                if (rel > lrel) { lrel = rel; lp = p; }
                if (rel > g_pf.check_tol) {
                    bad++;
                    if (shown < 10) {
                        shown++;
                        fprintf(stderr, "[xnpu-check]   bad L%02d p%02d %s rel=%.3e "
                                        "nrmse=%.3e\n",
                                l, p, half ? "V" : "K", rel, nrmse);
                    }
                    if (shown <= 2) {   /* 前两个超差点常开元素级 dump（sudo 会清 env，不做开关） */
                        fprintf(stderr, "[xnpu-check]   dump L%02d p%02d %s："
                                        "NPU[i] vs CPU[i]（前 12 元素）\n    ",
                                l, p, half ? "V" : "K");
                        for (int i = 0; i < 12; i++)
                            fprintf(stderr, "%8.4f/%8.4f ", get_bf16(mirror, i),
                                    ref[half * kv_dim + i]);
                        fprintf(stderr, "\n");
                    }
                }
            }
        }
        fprintf(stderr, "[xnpu-check]   L%02d worst rel=%.3e (p%02d)%s\n",
                l, lrel, lp, lrel > g_pf.check_tol ? " <-- 超" : "");
    }
    fprintf(stderr, "[xnpu-check] %d 层 x %d pos KV 对拍：worst rel=%.3e (L%d p%d) "
                    "worst nrmse=%.3e，超容差点 %d/%d，tol=%.3g -> %s\n",
            c->n_layers, n, worst_rel, worst_l, worst_p, worst_nrmse, bad,
            c->n_layers * n * 2, (double)g_pf.check_tol, bad ? "FAIL" : "PASS");
    free(snap);
}

/* ---- 层内 stage 二分（XNPU_CHECK 档：CPU 影子逐 stage 钩子 + L00 kick 后
 * MMIO 读 DTCM 的 q/atto/h2/x 对拍——一步定位层内坏级）---- */
extern void q3_layer_dbg_set(void (*fn)(int, int, const float *, int));

static float *g_sg_q, *g_sg_attn, *g_sg_ffn, *g_sg_x, *g_sg_emb;
static int g_sg_pos, g_sg_n;

static void stage_hook(int st, int layer, const float *x, int n_) {
    if (layer != 0 || g_sg_pos >= g_sg_n) return;
    float *dst;
    switch (st) {
    case -1: dst = g_sg_emb; break;  /* 嵌入（层 0 输入），宽 dim */
    case 2:  dst = g_sg_q; break;    /* q（rope 后），宽 XN_Q_DIM */
    case 4:  dst = g_sg_attn; break; /* 注意力输出，宽 XN_Q_DIM */
    case 8:  dst = g_sg_ffn; break;  /* silu(gate)*up，宽 hidden */
    case 6:  dst = g_sg_x; break;    /* 层末残差，宽 dim */
    default: return;
    }
    memcpy(dst + (size_t)g_sg_pos * n_, x, (size_t)n_ * sizeof(float));
}

/* L00 的 kick 完成后立即调用（DTCM 尚未被后续层覆写）。p0 = 本批首位置。
 * s2 参考用 CPU 影子的 h2 现算（w2 GEMM 隔离对拍）。 */
typedef struct { const char *name; uint32_t off; int width; const float *ref; int local; } StageReg;

static void stage_compare(Transformer *t, const LayerMap *m, int B, int p0) {
    const Config *c = &t->cfg;
    StageReg regs[5];
    int nr = 0;
    regs[nr++] = (StageReg){ "q(rope后)", m->q, XN_Q_DIM(c), g_sg_q, 0 };
    regs[nr++] = (StageReg){ "atto(@V)", m->atto, XN_Q_DIM(c), g_sg_attn, 0 };
    regs[nr++] = (StageReg){ "h2(swiglu)", m->h2, c->hidden_dim, g_sg_ffn, 0 };
    regs[nr++] = (StageReg){ "x(层末)", m->x, c->dim, g_sg_x, 0 };
    float *w2ref = malloc((size_t)B * c->dim * 4);
    if (w2ref) {
        for (int b = 0; b < B; b++)
            matmul_q8(w2ref + (size_t)b * c->dim, &t->w.w2[0],
                      g_sg_ffn + (size_t)(p0 + b) * c->hidden_dim, c->dim,
                      c->hidden_dim, c->group_size);
        regs[nr++] = (StageReg){ "s2(w2出)", m->s2, c->dim, w2ref, 1 };
    }
    for (size_t r = 0; r < sizeof regs / sizeof regs[0]; r++) {
        double worst = 0, scale = 0;
        int wp = 0;
        for (int b = 0; b < B; b++)
            for (int i = 0; i < regs[r].width; i++) {
                uint32_t el = (uint32_t)((size_t)b * regs[r].width + i) * 2;
                uint32_t w = xn_rd(XN_DTCM + regs[r].off + (el & ~3u));
                int lane = ((size_t)b * regs[r].width + i) & 1;
                uint16_t h = (uint16_t)(w >> (lane ? 16 : 0));
                float got = bf162f(h);
                float e = regs[r].ref[(size_t)(regs[r].local ? b : p0 + b) * regs[r].width + i];
                if (fabs(e) > scale) scale = fabs(e);
                if (fabs(got - e) > worst) { worst = fabs(got - e); wp = b; }
            }
        fprintf(stderr, "[xnpu-stage] L00 p%02d %-11s worst=%.4e scale=%.4e rel=%.3e (b%d)\n",
                p0, regs[r].name, worst, scale, scale > 1e-6 ? worst / scale : 0, wp);
        if (w2ref && g_sg_emb && !strcmp(regs[r].name, "x(层末)")) {
            /* 恒等式核验（逐 token 行首 6 元素）：got ?= emb+op+s2w ?= ref */
            float *opref = malloc((size_t)B * c->dim * 4);
            if (opref) {
                for (int b = 0; b < B; b++)
                    matmul_q8(opref + (size_t)b * c->dim, &t->w.wo[0],
                              g_sg_attn + (size_t)(p0 + b) * XN_Q_DIM(c), c->dim,
                              XN_Q_DIM(c), c->group_size);
                for (int b = 0; b < B; b++) {
                    fprintf(stderr, "[xnpu-stage]    x 恒等式 b%d i0-5: got/重构/ref\n     ", b);
                    for (int i = 0; i < 6; i++) {
                        uint32_t el = (uint32_t)((size_t)b * c->dim + i) * 2;
                        uint32_t w = xn_rd(XN_DTCM + regs[r].off + (el & ~3u));
                        uint16_t h = (uint16_t)(w >> ((((size_t)b * c->dim + i) & 1) ? 16 : 0));
                        float recon = g_sg_emb[(size_t)(p0 + b) * c->dim + i] +
                                      opref[(size_t)b * c->dim + i] +
                                      w2ref[(size_t)b * c->dim + i];
                        fprintf(stderr, "%8.4f/%8.4f/%8.4f ", bf162f(h), recon,
                                regs[r].ref[(size_t)(p0 + b) * c->dim + i]);
                    }
                    fprintf(stderr, "\n");
                }
                free(opref);
            }
        }
    }
    free(w2ref);
}

int xnpu_prefill_prompt(Transformer *t, const int *tokens, int n) {
    if (!xnpu_prefill_ready() || g_xn.unhealthy) return -1;
    const Config *c = &t->cfg;
    int dim = c->dim, q_dim = XN_Q_DIM(c), kv_dim = XN_KV_DIM(c), gs = c->group_size;
    if (n < 2 || n > c->seq_len) return -1;
    if (n > g_pf.tcap) {
        fprintf(stderr, "[xnpu] prefill→CPU：n=%d 超 tcap=%d（DTCM 工作点；"
                "更长提示需分块注意力重构，见 plan §7.3）\n", n, g_pf.tcap);
        return -1;
    }

    /* CHECK 档：先跑 CPU 影子（填 fp32 cache 供稍后对拍；logits 弃用）；
     * 同时挂 stage 钩子抓 L00 的 q/atto/ffn/x 供层内二分。 */
    if (g_pf.check) {
        g_sg_n = n;
        g_sg_q = malloc((size_t)n * XN_Q_DIM(c) * 4);
        g_sg_emb = malloc((size_t)n * c->dim * 4);
        g_sg_attn = malloc((size_t)n * XN_Q_DIM(c) * 4);
        g_sg_ffn = malloc((size_t)n * c->hidden_dim * 4);
        g_sg_x = malloc((size_t)n * c->dim * 4);
        q3_layer_dbg_set(stage_hook);
        for (int p = 0; p < n; p++) {
            g_sg_pos = p;
            forward(t, tokens[p], p);
        }
        q3_layer_dbg_set(NULL);
    }

    /* 嵌入懒反量化进 xbuf */
    {
        float *row = malloc((size_t)dim * sizeof(float));
        if (!row) return -1;
        for (int p = 0; p < n; p++) {
            const int8_t *q = t->w.tokens.q + (size_t)tokens[p] * dim;
            const float *s = t->w.tokens.s + (size_t)tokens[p] * (dim / gs);
            for (int i = 0; i < dim; i++)
                row[i] = (float)q[i] * s[i / gs];
            put_bf16(g_pf.xbuf_v + (size_t)p * dim * 2, row, dim);
        }
        free(row);
    }
    xnpu_evict();
    __sync_synchronize();

    int B = 0, forced = 0;
    {
        const char *bf = getenv("XNPU_BFORCE");
        if (bf) {
            int v = atoi(bf);
            if (v >= 1 && v <= PREFILL_MAXB && v <= n && shape_ok(c, v, n)) {
                LayerMap m = layer_map_qwen3(v, c->dim, c->hidden_dim, q_dim, kv_dim,
                                             c->head_dim, n, gs, g_xn.sched);
                if (m.ok) { B = v; forced = 1; }
            }
        }
    }
    for (int b = PREFILL_MAXB; b >= 1 && !B; b--) {
        if (b > n) continue;
        if (!shape_ok(c, b, n)) continue;
        LayerMap m = layer_map_qwen3(b, c->dim, c->hidden_dim, q_dim, kv_dim,
                                     c->head_dim, n, gs, g_xn.sched);
        if (m.ok) { B = b; break; }
    }
    if (!B) {
        fprintf(stderr, "[xnpu] prefill→CPU：n=%d 无可行 B（map 放不下）\n", n);
        return -1;
    }
    if (!forced && B < g_pf.minb) {
        fprintf(stderr, "[xnpu] prefill→CPU：最大可行 B=%d < minb=%d"
                "（B=1 kick 与 B=4 同价，吞吐仅 CPU 一半；XNPU_MINB/BFORCE 可改）\n",
                B, g_pf.minb);
        return -1;
    }

    /* 窗口外层化（2026-10-18）：for 窗口 { for 批 { for 层 } }。旧批外层实现
     * 每批走全 28 层 → 双 arena 每批换装 4 次，5 批提示词重复 repack ~2.2GB
     * （staging 9.3s 的主因）；本序每提示词每窗口只打包一次（共 4 次 ≈442MB）。
     * 正确性：层 l 的注意力只读层 l 自己的 K/V 镜像，因果性（位置 p 只看
     * ≤p）下，窗口内"批×层"的字典序遍历与逐层批序等价；批间 x 状态经
     * xbuf 由窗口首层 DF_LOAD / 末层 DF_STORE 传递，描述符语义不变。 */
    int nwin = (c->n_layers + g_pf.win_layers - 1) / g_pf.win_layers;
    for (int w = 0; w < nwin; w++) {
        int wfirst = w * g_pf.win_layers;
        int wlast = wfirst + g_pf.win_layers - 1;
        if (wlast >= c->n_layers) wlast = c->n_layers - 1;
        for (int l = wfirst; l <= wlast; l++)
            stage_window_ensure(l, t);   /* 幂等：同窗口仅首次实际打包 */
        for (int p0 = 0; p0 < n; p0 += B) {
            int bb = n - p0 < B ? n - p0 : B;
            if (load_program(t, bb, p0, n)) return -1;
            LayerMap *m = &g_pf.map;
            for (int l = wfirst; l <= wlast; l++) {
                size_t gl = (size_t)l * g_pf.gain_layer_bytes;
                uint32_t desc[DW_NWORDS];
                for (int i = 0; i < 7; i++) {
                    desc[DW_WQ + i * 2] = (uint32_t)g_pf.wt_p[l][i];
                    desc[DW_SWQ + i * 2] = (uint32_t)g_pf.ws_p[l][i];
                }
                desc[DW_RMSA] = (uint32_t)(g_pf.gains_p + gl);
                desc[DW_RMSF] = (uint32_t)(g_pf.gains_p + gl + (size_t)dim * 2);
                desc[DW_QKG] = (uint32_t)(g_pf.gains_p + gl + (size_t)(2 * dim) * 2);
                desc[DW_KKG] = (uint32_t)(g_pf.gains_p + gl + (size_t)(2 * dim + c->head_dim) * 2);
                desc[DW_KC] = (uint32_t)(g_pf.kc_p + (size_t)l * g_pf.kv_layer_bytes);
                desc[DW_VC] = (uint32_t)(g_pf.vc_p + (size_t)l * g_pf.kv_layer_bytes);
                desc[DW_XBUF] = (uint32_t)(g_pf.xbuf_p + (size_t)p0 * dim * 2);
                desc[DW_COSQ] = (uint32_t)(g_pf.cosq_p + (size_t)p0 * q_dim * 2);
                desc[DW_SINQ] = (uint32_t)(g_pf.sinq_p + (size_t)p0 * q_dim * 2);
                desc[DW_COSK] = (uint32_t)(g_pf.cosk_p + (size_t)p0 * kv_dim * 2);
                desc[DW_SINK] = (uint32_t)(g_pf.sink_p + (size_t)p0 * kv_dim * 2);
                desc[DW_ZERO] = (uint32_t)g_pf.zero_p;
                desc[DW_FLAGS] = (l == wfirst ? DF_LOAD_BATCH : 0u) |
                                 (l == wlast ? DF_STORE_BATCH : 0u);
                for (int i = 0; i < DW_NWORDS; i++)
                    xn_wr(XN_DTCM + m->desc + (uint32_t)i * 4, desc[i]);
                g_xn.p_desc_words += DW_NWORDS;
                if (xnpu_kick("layer")) return -1;
                if (g_pf.check && l == 0)
                    stage_compare(t, m, bb, p0);   /* 每个 batch 的 L00 kick 后立即二分 */
            }
        }
    }

    /* 先对拍（快照在 CPU cache 尚未被回灌覆盖时取），再回灌。 */
    if (g_pf.check) check_against_shadow(t, n);

    xnpu_evict();
    __sync_synchronize();
    for (int l = 0; l < c->n_layers; l++) {
        const uint8_t *kc = g_pf.kc_v + (size_t)l * g_pf.kv_layer_bytes;
        const uint8_t *vc = g_pf.vc_v + (size_t)l * g_pf.kv_layer_bytes;
        size_t loff = (size_t)l * c->seq_len * kv_dim;
        for (int p = 0; p < n; p++)
            for (int i = 0; i < kv_dim; i++) {
                t->s.key_cache[loff + (size_t)p * kv_dim + i] =
                    get_bf16(kc + (size_t)p * kv_dim * 2, i);
                t->s.value_cache[loff + (size_t)p * kv_dim + i] =
                    get_bf16(vc + (size_t)p * kv_dim * 2, i);
            }
    }
    return 0;
}
