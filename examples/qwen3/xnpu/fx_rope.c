/* fx_rope -- fe_rope_half 的 RTL 仿真夹具生成器（Phase 3 第 1 级之二）。
 *
 * 协议与 fx_qknorm 相同（.prog/.mem_in/.mem_out/.txt），spec 侧
 * Qwen3RopeRtlSpec 重放。本夹具的关键判别面：
 *   - half-split 几何（j 配 j+64）：若发射器退回 interleave 会大错；
 *   - 表的 per-token 复制结构（start_pos=3，token 位置 3/4，角度不同——
 *     若所有 token 用了同一张表立刻暴露）；
 *   - cos 表前后半相同、sin_signed 前半负后半正的宿主契约。
 *
 * 数值契约：vmul/vadd 每步回写 bf16，参考模型按
 *   out = bf16( bf16(v*cos) + bf16(swap(v)*sin_signed) ) 计算。
 * 再生成：make fx-rtl。 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "enc.h"
#include "emit.h"

#define DIM 1024
#define HID 3072
#define QD 2048
#define KVD 1024
#define HD 128
#define NH 16
#define NKV 8
#define B 2
#define T 8
#define THETA 1000000.0
#define START_POS 3
#define GS 64

static uint32_t rng = 0x5eed1234u;
static float frand(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((float)(rng >> 8) / (float)(1 << 24)) * 2.0f - 1.0f;
}

static void put_bf16(uint8_t *img, uint32_t off, const float *src, int n) {
    for (int i = 0; i < n; i++) {
        uint16_t h = f2bf16(src[i]);
        memcpy(img + off + (size_t)i * 2, &h, 2);
    }
}
static float get_bf16(const uint8_t *img, uint32_t off, int idx) {
    uint16_t h;
    memcpy(&h, img + off + (size_t)idx * 2, 2);
    return bf162f(h);
}

/* half-split 表：每 token 每头 128 项。cos[j]=cos[j+64]=cos(p*w_j)；
 * sin_signed[j]=-sin(p*w_j)，sin_signed[j+64]=+sin(p*w_j)。 */
static void gen_tables(float *cos_t, float *sin_t, int pos) {
    for (int j = 0; j < HD / 2; j++) {
        float w = (float)pow(THETA, -2.0 * (double)j / (double)HD);
        float a = (float)((double)pos * (double)w);
        float c = cosf(a), s = sinf(a);
        cos_t[j] = c;         cos_t[j + HD / 2] = c;
        sin_t[j] = -s;        sin_t[j + HD / 2] = s;
    }
}

/* 参考模型：out = bf16(bf16(v*cos) + bf16(swap(v)*sin_signed))，逐头。 */
static void rope_ref(uint8_t *img, uint32_t off, uint32_t cos_off, uint32_t sin_off,
                     int n_heads) {
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < n_heads; h++) {
            uint32_t voff = off + (uint32_t)((b * n_heads + h) * HD) * 2;
            uint32_t toff_c = cos_off + (uint32_t)((b * n_heads + h) * HD) * 2;
            uint32_t toff_s = sin_off + (uint32_t)((b * n_heads + h) * HD) * 2;
            float out[HD];
            for (int j = 0; j < HD; j++) {
                float v = get_bf16(img, voff, j);
                float sw = get_bf16(img, voff, (j < HD / 2) ? j + HD / 2 : j - HD / 2);
                float c = get_bf16(img, toff_c, j);
                float s = get_bf16(img, toff_s, j);
                uint16_t z1 = f2bf16(v * c), z2 = f2bf16(sw * s);
                out[j] = bf162f(f2bf16(bf162f(z1) + bf162f(z2)));
            }
            put_bf16(img, voff, out, HD);
        }
    }
}

static void dump(const char *path, const void *buf, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    if (fwrite(buf, 1, n, f) != n) { perror(path); exit(1); }
    fclose(f);
}

int main(int argc, char **argv) {
    const char *prefix = argc > 1 ? argv[1] : "fixtures/rtl/rope";

    LayerMap m = layer_map_qwen3(B, DIM, HID, QD, KVD, HD, T, GS, 0);
    if (!m.ok) { fprintf(stderr, "shape does not fit DTCM\n"); return 1; }

    uint8_t *in = calloc(m.total, 1);
    if (!in) { perror("calloc"); return 1; }

    /* ---- 输入：q @ m.q（B×q_dim）、k @ m.kvh 首行（B×kv_dim）、
     * cos/sin 复制表（per token per head 128 项，位置 START_POS+b）。---- */
    float *tmp = malloc((size_t)QD * sizeof(float));
    float *ct = malloc((size_t)QD * sizeof(float));
    float *st = malloc((size_t)QD * sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < QD; i++) tmp[i] = frand();
        put_bf16(in, m.q + (uint32_t)b * QD * 2, tmp, QD);
        for (int i = 0; i < KVD; i++) tmp[i] = frand();
        put_bf16(in, m.kvh + (uint32_t)b * KVD * 2, tmp, KVD);
        gen_tables(ct, st, START_POS + b);          /* per-token 表 */
        for (int h = 0; h < NH; h++) {              /* 复制到每头 */
            put_bf16(in, m.cosq + (uint32_t)((b * NH + h) * HD) * 2, ct, HD);
            put_bf16(in, m.sinq + (uint32_t)((b * NH + h) * HD) * 2, st, HD);
        }
        for (int h = 0; h < NKV; h++) {
            put_bf16(in, m.cosk + (uint32_t)((b * NKV + h) * HD) * 2, ct, HD);
            put_bf16(in, m.sink + (uint32_t)((b * NKV + h) * HD) * 2, st, HD);
        }
    }
    free(tmp); free(ct); free(st);

    /* ---- 发射：rope(q) + rope(k) + HALT ---- */
    static uint32_t prog[ITCM_WORDS];
    Emit e = { prog, 0, ITCM_WORDS, 0 };
    fe_rope_half(&e, &m, m.q, m.cosq, m.sinq, B, QD, HD);
    fe_rope_half(&e, &m, m.kvh, m.cosk, m.sink, B, KVD, HD);
    ep(&e, ENC_HALT);
    if (e.bad) { fprintf(stderr, "emitter overflow\n"); return 1; }
    printf("program: %d words\n", e.n);

    /* ---- 参考（in-place 覆写）---- */
    uint8_t *out = malloc(m.total);
    memcpy(out, in, m.total);
    rope_ref(out, m.q, m.cosq, m.sinq, NH);
    rope_ref(out, m.kvh, m.cosk, m.sink, NKV);

    /* ---- dump ---- */
    char path[512];
    uint32_t n = (uint32_t)e.n;
    snprintf(path, sizeof path, "%s.prog", prefix);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return 1; }
    if (fwrite(&n, 4, 1, f) != 1 || fwrite(prog, 4, e.n, f) != (size_t)e.n) {
        perror(path); return 1;
    }
    fclose(f);
    snprintf(path, sizeof path, "%s.mem_in", prefix); dump(path, in, m.total);
    snprintf(path, sizeof path, "%s.mem_out", prefix); dump(path, out, m.total);
    snprintf(path, sizeof path, "%s.txt", prefix);
    f = fopen(path, "w");
    if (!f) { perror(path); return 1; }
    fprintf(f,
        "# fe_rope_half RTL fixture (regen: make fx-rtl)\n"
        "head_dim %d\nn_heads %d\nn_kv %d\nB %d\nstart_pos %d\ntheta %g\n"
        "q_off %u\nk_off %u\ncosq_off %u\nsinq_off %u\ncosk_off %u\nsink_off %u\n"
        "s1_off %u\ns2_off %u\nwindow %u\nn_words %u\n",
        HD, NH, NKV, B, START_POS, THETA,
        m.q, m.kvh, m.cosq, m.sinq, m.cosk, m.sink, m.s1, m.s2,
        m.total, n);
    fclose(f);
    printf("fixture written to %s.{prog,mem_in,mem_out,txt} (window %u B)\n",
           prefix, m.total);
    return 0;
}
