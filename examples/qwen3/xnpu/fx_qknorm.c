/* fx_qknorm -- fe_qknorm 的 RTL 仿真夹具生成器（Phase 3 第 1 级）。
 *
 * 产物四件（前缀默认 fixtures/rtl/qknorm）：
 *   <p>.prog    u32：词数 N，后跟 N 个指令字（fe_qknorm(q)+fe_qknorm(k)+HALT）
 *   <p>.mem_in  DTCM 字节镜像 [0, map.total)：随机 q/k + 两份不同增益
 *   <p>.mem_out 参考镜像：bf16 边界、fp32 内积的逐头 rmsnorm 期望值
 *   <p>.txt     键值元数据（offsets/dims/eps/window）
 *
 * 数值契约（与 src/test/scala/Qwen3QknormRtlSpec.scala 对齐）：RTL 向量核
 * bf16 存储、fp32 计算，VRSQRT 是 0x5f3759df 种子 + 1 次 Newton 而非精确
 * rsqrt，标量中间量（VSMUL/VSADD 回写）按 bf16 舍入——故 spec 侧用相对
 * 容差对拍，本参考用纯 fp32 链（1/sqrtf）即可。
 *
 * 增益负对照：q/k 增益取不同值，spec 侧据此检查"Q 增益没串进 K"。
 * 再生成：make fx-rtl（或本文件 -o 后直接运行，输出前缀可 -p 覆盖）。 */
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
#define EPS 1e-6f
#define GS 64

static uint32_t rng = 0x1234abcdu;
static float frand(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((float)(rng >> 8) / (float)(1 << 24)) * 2.0f - 1.0f; /* [-1,1) */
}

static void put_bf16(uint8_t *img, uint32_t off, const float *src, int n) {
    for (int i = 0; i < n; i++) {
        uint16_t h = f2bf16(src[i]);
        memcpy(img + off + (size_t)i * 2, &h, 2);
    }
}

/* 参考模型：y = gain * bf16(bf16(x * rsqrt(mean(x^2)+eps)))，逐头。 */
static void qknorm_ref(uint8_t *img, uint32_t off, uint32_t goff, int blocks) {
    for (int b = 0; b < blocks; b++) {
        const uint8_t *xp = img + off + (size_t)b * HD * 2;
        const uint8_t *gp = img + goff;
        float ss = 0.0f;
        for (int i = 0; i < HD; i++) {
            uint16_t xh, gh;
            memcpy(&xh, xp + (size_t)i * 2, 2);
            memcpy(&gh, gp + (size_t)i * 2, 2);
            float xf = bf162f(xh);
            ss += xf * xf;
        }
        float scale = 1.0f / sqrtf(ss / (float)HD + EPS);
        for (int i = 0; i < HD; i++) {
            uint16_t xh, gh;
            memcpy(&xh, xp + (size_t)i * 2, 2);
            memcpy(&gh, gp + (size_t)i * 2, 2);
            float y = bf162f(f2bf16(bf162f(xh) * scale)); /* vsmul 回写 bf16 */
            float z = bf162f(f2bf16(y * bf162f(gh)));      /* vmul 回写 bf16 */
            uint16_t zh = f2bf16(z);
            memcpy((uint8_t *)img + off + (size_t)b * HD * 2 + (size_t)i * 2, &zh, 2);
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
    const char *prefix = argc > 1 ? argv[1] : "fixtures/rtl/qknorm";

    LayerMap m = layer_map_qwen3(B, DIM, HID, QD, KVD, HD, T, GS, 0);
    if (!m.ok) { fprintf(stderr, "shape does not fit DTCM\n"); return 1; }

    /* ---- 输入：q @ m.q（B×q_dim），k @ m.kvh 首行（B×kv_dim，模拟
     * 刚写进 KV cache 的 k 行），两份增益 @ m.qkg / m.qkg+256。
     * 幅值随头号变化（0.2~2.3），让不同头的 scale 显著不同。---- */
    uint8_t *in = calloc(m.total, 1);
    if (!in) { perror("calloc"); return 1; }
    float *tmp = malloc((size_t)QD * sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            float amp = 0.2f + 0.13f * (float)h;
            for (int i = 0; i < HD; i++)
                tmp[(size_t)h * HD + i] = frand() * amp;
        }
        put_bf16(in, m.q + (uint32_t)b * QD * 2, tmp, QD);
        for (int h = 0; h < NKV; h++) {
            float amp = 0.3f + 0.21f * (float)h;
            for (int i = 0; i < HD; i++)
                tmp[(size_t)h * HD + i] = frand() * amp;
        }
        put_bf16(in, m.kvh + (uint32_t)b * KVD * 2, tmp, KVD);
    }
    for (int i = 0; i < HD; i++) tmp[i] = 0.5f + 0.5f * frand();   /* q gain */
    put_bf16(in, m.qkg, tmp, HD);
    for (int i = 0; i < HD; i++) tmp[i] = 0.6f + 0.4f * frand() + 0.4f; /* k gain, 不同分布 */
    put_bf16(in, m.qkg + HD * 2, tmp, HD);
    free(tmp);

    /* ---- 发射：qknorm(q) + qknorm(k) + HALT，全部 in-place ---- */
    static uint32_t prog[ITCM_WORDS];
    Emit e = { prog, 0, ITCM_WORDS, 0 };
    fe_qknorm(&e, &m, m.q, m.q, m.qkg, B, NH, HD, EPS);
    fe_qknorm(&e, &m, m.kvh, m.kvh, m.qkg + HD * 2, B, NKV, HD, EPS);
    ep(&e, ENC_HALT);
    if (e.bad) { fprintf(stderr, "emitter overflow\n"); return 1; }
    printf("program: %d words\n", e.n);

    /* ---- 参考 ---- */
    uint8_t *out = malloc(m.total);
    memcpy(out, in, m.total);
    qknorm_ref(out, m.q, m.qkg, B * NH);
    qknorm_ref(out, m.kvh, m.qkg + HD * 2, B * NKV);

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
        "# fe_qknorm RTL fixture (regen: make fx-rtl)\n"
        "head_dim %d\nn_heads %d\nn_kv %d\nB %d\n"
        "q_off %u\nk_off %u\nqg_off %u\nkg_off %u\n"
        "zero_off %u\nsc_off %u\neps %g\nwindow %u\nn_words %u\n",
        HD, NH, NKV, B, m.q, m.kvh, m.qkg, m.qkg + HD * 2, m.zero, m.sc,
        (double)EPS, m.total, n);
    fclose(f);
    printf("fixture written to %s.{prog,mem_in,mem_out,txt} (window %u B)\n",
           prefix, m.total);
    return 0;
}
