/* fx_bf16self -- bf16 8x8 tier Core 级仿真夹具生成器（2026-11-03）。
 *
 * 板上 bf16selftest.c 的程序逐字 dump：同样的 LCG 激励、同样的
 * emit_k8/emit_k16 指令序列、同样的 DTCM 基址（A1=0x000 A2=0x200
 * B=0x400 C=0x600）。产物供 Qwen3Bf16CoreSpec 在
 * Parameters(bf16Matmul=true) 的真 Core 上重放（板测发现 C 行1低64位
 * 出错——Core 级 FP.* 路径此前无任何仿真覆盖，此夹具补上该缺口）。
 *
 * 产物（前缀 fixtures/rtl/bf16k8 与 bf16k16，两套共享同一激励）：
 *   <p>.prog    u32：词数 N + N 个指令字（emit_k8/emit_k16 + HALT）
 *   <p>.mem_in  DTCM 字节镜像 [0, 0x680)：A1/A2/B 打包，C 区全零
 *   <p>.mem_out 参考镜像：同窗口 + C 区 golden（bit-exact 对拍）
 *   <p>.txt     键值元数据
 *
 * 黄金 = 与 MultiEngineSpec/板上自测同构：bf16 宽化 + 严格 k 序 fmaf
 * 折叠 + FP.SM 出口单次 RNE（bf16_golden/bf16_golden_2t，bf16test.h）。
 * 再生成：make fx-bf16self。
 */
#include "xnpu/enc.h"
#include "xnpu/emit.h"
#include "bf16test.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OFF_A1 0x0000u
#define OFF_A2 0x0200u
#define OFF_B  0x0400u
#define OFF_C  0x0600u
#define WIN    (OFF_C + BF16T_TILE_BYTES)   /* 0x680 */

static void wr(const char *prefix, const char *suf, const void *buf, size_t n) {
    char path[320];
    snprintf(path, sizeof path, "%s.%s", prefix, suf);
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(buf, 1, n, f) != n) { perror(path); exit(1); }
    fclose(f);
}

/* 与 bf16selftest.c 的 emit_k8/emit_k16 保持一致（寄存器/顺序勿改） */
static int emit_k8(uint32_t *p) {
    Emit e = { p, 0, 64, 0 };
    eli(&e, 5, OFF_A1);
    eli(&e, 6, OFF_B);
    eli(&e, 7, OFF_C);
    ep(&e, enc_lma(5));
    ep(&e, enc_lmb(6));
    ep(&e, enc_fp_matmul());
    ep(&e, enc_fp_sm(7));
    ep(&e, ENC_HALT);
    return e.n;
}

static int emit_k16(uint32_t *p) {
    Emit e = { p, 0, 64, 0 };
    eli(&e, 5, OFF_A1);
    eli(&e, 6, OFF_B);
    eli(&e, 7, OFF_C);
    eli(&e, 9, OFF_A2);
    ep(&e, enc_lma(5));
    ep(&e, enc_lmb(6));
    ep(&e, enc_fp_matmul());       /* C = A1 x B (clears) */
    ep(&e, enc_lma(9));            /* re-latch A base to A2 */
    ep(&e, enc_fp_matmul_acc());   /* C += A2 x B (no clear) */
    ep(&e, enc_fp_sm(7));
    ep(&e, ENC_HALT);
    return e.n;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "fixtures/rtl";

    uint64_t lcg = BF16_LCG_INIT;
    static uint16_t A1[BF16_DIM][BF16_DIM], A2[BF16_DIM][BF16_DIM],
                    B[BF16_DIM][BF16_DIM];
    static uint16_t g1[BF16_DIM][BF16_DIM], g2[BF16_DIM][BF16_DIM];
    for (int i = 0; i < BF16_DIM; i++)
        for (int j = 0; j < BF16_DIM; j++)
            A1[i][j] = bf16_bfTile_elem(&lcg, i, j, 11);
    for (int i = 0; i < BF16_DIM; i++)
        for (int j = 0; j < BF16_DIM; j++)
            A2[i][j] = bf16_bfTile_elem(&lcg, i, j, 17);
    for (int i = 0; i < BF16_DIM; i++)
        for (int j = 0; j < BF16_DIM; j++)
            B[i][j] = bf16_bfTile_elem(&lcg, i, j, 13);
    for (int i = 0; i < BF16_DIM; i++)
        for (int j = 0; j < BF16_DIM; j++) {
            g1[i][j] = bf16_golden(A1, B, i, j);
            g2[i][j] = bf16_golden_2t(A1, A2, B, i, j);
        }

    static uint8_t img[WIN], ref1[WIN], ref2[WIN];
    uint32_t words[BF16T_TILE_WORDS];
    bf16_pack_words(words, A1); memcpy(img + OFF_A1, words, sizeof words);
    bf16_pack_words(words, A2); memcpy(img + OFF_A2, words, sizeof words);
    bf16_pack_words(words, B);  memcpy(img + OFF_B,  words, sizeof words);
    memcpy(ref1, img, WIN);
    memcpy(ref2, img, WIN);
    bf16_pack_words(words, g1); memcpy(ref1 + OFF_C, words, sizeof words);
    bf16_pack_words(words, g2); memcpy(ref2 + OFF_C, words, sizeof words);

    static uint32_t prog[64];
    struct { const char *tag; int (*emit)(uint32_t *); } cases[] = {
        { "bf16k8", emit_k8 }, { "bf16k16", emit_k16 },
    };
    for (unsigned k = 0; k < sizeof cases / sizeof cases[0]; k++) {
        char prefix[256];
        int n = cases[k].emit(prog);
        FILE *f;
        snprintf(prefix, sizeof prefix, "%s/%s", dir, cases[k].tag);
        char path[320];
        snprintf(path, sizeof path, "%.240s.prog", prefix);
        f = fopen(path, "wb");
        if (!f || fwrite(&(uint32_t){ (uint32_t)n }, 4, 1, f) != 1 ||
            fwrite(prog, 4, (size_t)n, f) != (size_t)n) { perror(path); exit(1); }
        fclose(f);
        wr(prefix, "mem_in", img, sizeof img);
        wr(prefix, "mem_out", k == 0 ? ref1 : ref2, WIN);
        snprintf(path, sizeof path, "%.240s.txt", prefix);
        f = fopen(path, "w");
        fprintf(f, "# bf16 8x8 core-level fixture (board bf16selftest replica)\n"
                   "dim %d\na1 %u\na2 %u\nb %u\nc %u\nwindow %u\nmode %s\n",
                BF16_DIM, OFF_A1, OFF_A2, OFF_B, OFF_C, (unsigned)WIN, cases[k].tag);
        fclose(f);
        printf("%s: prog=%d words, window=%u B\n", cases[k].tag, n, (unsigned)WIN);
    }
    return 0;
}
