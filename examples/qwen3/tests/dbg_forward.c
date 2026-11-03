/* dbg_forward -- 前向二分对拍工具。
 * usage: tests/dbg_forward ckpt.bin tok.bin "prompt" /tmp/hf_ prefix pos
 *   在 layer 0 的各 stage 与 HF 参考比余弦（embed/xnorm/q/k/attnout/attnresid/end）。 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../model.h"
#include "../nn.h"
#include "../tokenizer.h"

void q3_layer_dbg_set(void (*fn)(int, int, const float *, int));

static double cosine(const float *a, const float *b, int n) {
    double d = 0, x = 0, y = 0;
    for (int i = 0; i < n; i++) {
        d += (double)a[i] * b[i];
        x += (double)a[i] * a[i];
        y += (double)b[i] * b[i];
    }
    return (x > 0 && y > 0) ? d / sqrt(x * y) : -2;
}

static float *g_ref[16]; /* stage -> [T][width] */
static const char *g_names[9] = {"",    "xnorm", "q",   "k",          "",
                                 "attnresid", "end", "ffnnorm", "ffnprod"};
static int g_width[16];
static int g_pos, g_layer = 0;
static int g_print = 0; /* 只在最后一趟 forward 打印 */

static float *g_ref_deep[32]; /* layer -> h_end 参考（hmid/hfinal） */
static int g_has_deep = 0;
static void hook(int stage, int layer, const float *x, int n) {
    if (!g_print)
        return;
    if (stage == 6 && g_has_deep && g_ref_deep[layer] && layer != g_layer) {
        printf("pos %d layer %2d end   cos=%.4f (deep ref)\n", g_pos, layer,
               cosine(x, g_ref_deep[layer] + (size_t)g_pos * 1024, 1024));
        return;
    }
    if (stage == 9) {
        printf("pos %d FINAL-NORMED  cos=%.4f (vs hs_28)\n", g_pos,
               cosine(x, g_ref_deep[27] + (size_t)g_pos * 1024, 1024));
        return;
    }
    if (layer != g_layer || stage < 1 || stage > 8 || !g_ref[stage])
        return;
    const float *row = g_ref[stage] + (size_t)g_pos * g_width[stage];
    printf("pos %d layer %d stage %-9s cos=%.4f (n=%d ref_w=%d)\n", g_pos, layer,
           g_names[stage], cosine(x, row, n < g_width[stage] ? n : g_width[stage]), n,
           g_width[stage]);
}

static float *loadf(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    float *p = malloc((size_t)sz);
    if (fread(p, 1, (size_t)sz, f) != (size_t)sz) {
        free(p);
        p = NULL;
    }
    fclose(f);
    *n = (size_t)sz / 4;
    return p;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s ckpt tok prompt prefix pos layer\n", argv[0]);
        return 1;
    }
    Transformer t;
    build_transformer(&t, argv[1], 0);
    Tokenizer tok;
    build_tokenizer(&tok, argv[2]);
    int n;
    int *ids = tok_encode(&tok, argv[3], strlen(argv[3]), &n);
    int pos = atoi(argv[5]);
    g_layer = argc > 6 ? atoi(argv[6]) : 0;
    g_pos = pos;
    const char *prefix = argv[4];
    char path[1024];
    size_t cnt;
    char lt[8];
    snprintf(lt, sizeof lt, "l%d", g_layer);
    char fn[8][32];
    snprintf(fn[0], 32, "%s_xnorm.f32", lt);
    snprintf(fn[1], 32, "%s_q.f32", lt);
    snprintf(fn[2], 32, "%s_k.f32", lt);
    snprintf(fn[3], 32, "%s_attnresid.f32", lt);
    snprintf(fn[4], 32, "%s_end.f32", lt);
    snprintf(fn[5], 32, "%s_ffnnorm.f32", lt);
    snprintf(fn[6], 32, "%s_downin.f32", lt);
    struct { int st, w; const char *f; } tabs[] = {
        {1, t.cfg.dim, fn[0]}, {2, XN_Q_DIM(&t.cfg), fn[1]},
        {3, XN_KV_DIM(&t.cfg), fn[2]}, {5, t.cfg.dim, fn[3]},
        {6, t.cfg.dim, fn[4]}, {7, t.cfg.dim, fn[5]},
        {8, t.cfg.hidden_dim, fn[6]},
    };
    for (size_t i = 0; i < sizeof tabs / sizeof tabs[0]; i++) {
        snprintf(path, sizeof path, "%s%s", prefix, tabs[i].f);
        g_ref[tabs[i].st] = loadf(path, &cnt);
        g_width[tabs[i].st] = tabs[i].w;
        if (g_ref[tabs[i].st] && cnt % (size_t)tabs[i].w != 0)
            printf("warn: %s size %zu not multiple of %d\n", path, cnt, tabs[i].w);
    }
    {
        size_t cnt2;
        char hp[64];
        for (int li = 0; li < 29; li++) {
            snprintf(hp, sizeof hp, "/tmp/hf_hs_%d.f32", li);
            g_ref_deep[li == 0 ? 0 : li - 1] = loadf(hp, &cnt2); /* hs_{L+1}（注意 hs_28=post-norm） */
        }
        g_ref_deep[27] = loadf("/tmp/hf_prenorm27.f32", &cnt2); /* 覆盖为真 pre-norm */
        g_has_deep = 1;
    }
    q3_layer_dbg_set(hook);
    for (int i = 0; i <= pos; i++) {
        g_print = (i == pos);
        forward(&t, ids[i], i);
    }
    q3_layer_dbg_set(NULL);
    return 0;
}
