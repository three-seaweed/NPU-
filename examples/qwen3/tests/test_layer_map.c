/* test_layer_map -- plan §4 工作点表的可执行锁定（Phase 3 第 0 级）。
 *
 * 断言三件事：
 *   1. 逐 token / 逐 context 行的边际字节 == §4 的 49,344 / 4,102（GS=64）；
 *   2. GS=64 工作点表（B=1/76,B=2/64,B=4/40,B>=8 放不下；规划 §4 近似值为
 *      77/64/40，B=1 差 1 因 qkg+512B 与 zero 扩到 q_dim 的 +2,048B——规划
 *      自己声明精确值以运行时为准）与 GS=32 回退表（sw 区几乎翻倍 → 工作点
 *      腰斩；2026-10-17 发现现有 build/ 检查点是 GS=32 导出的，NPU 线必须按
 *      规划首选重导出 GS=64）；
 *   3. 结构不变量：全区域 16B 对齐、total<=DTCM_BYTES、qkg=2*head_dim*2。
 *
 * 失败即改 emit.c 或改 plan —— 两边必须一致，不许静默漂移。 */
#include <stdio.h>
#include <stdlib.h>

#include "../xnpu/emit.h"

#define DIM 1024
#define HID 3072
#define QD 2048
#define KVD 1024
#define HD 128
#define SEQ 512

static int max_T(int B, int gs) {
    for (int T = SEQ; T >= 1; T--) {
        LayerMap m = layer_map_qwen3(B, DIM, HID, QD, KVD, HD, T, gs, 0);
        if (m.ok) return T;
    }
    return 0;
}

static int fails = 0;
static void chk(const char *what, long got, long want) {
    if (got != want) {
        printf("FAIL %-34s got %ld want %ld\n", what, got, want);
        fails++;
    } else {
        printf("ok   %-34s %ld\n", what, got);
    }
}

int main(void) {
    /* ---- 边际字节（§4 复现；GS 只影响 gscale 边际）---- */
    {
        LayerMap m1 = layer_map_qwen3(1, DIM, HID, QD, KVD, HD, 16, 64, 0);
        LayerMap m2 = layer_map_qwen3(2, DIM, HID, QD, KVD, HD, 16, 64, 0);
        LayerMap m1b = layer_map_qwen3(1, DIM, HID, QD, KVD, HD, 32, 64, 0);
        chk("per-token marginal (GS=64)", (long)m2.total - (long)m1.total, 49344);
        chk("per-T marginal   (GS=64)", (long)m1b.total - (long)m1.total, 16 * 4102);
        LayerMap s1 = layer_map_qwen3(1, DIM, HID, QD, KVD, HD, 16, 32, 0);
        LayerMap s2 = layer_map_qwen3(2, DIM, HID, QD, KVD, HD, 16, 32, 0);
        chk("per-token marginal (GS=32)", (long)s2.total - (long)s1.total, 49536);
    }

    /* ---- 工作点表 ---- */
    printf("GS=64 working points (DTCM %u B):\n", DTCM_BYTES);
    chk("B=1 max T (GS=64)", max_T(1, 64), 76);
    chk("B=2 max T (GS=64)", max_T(2, 64), 64);
    chk("B=4 max T (GS=64)", max_T(4, 64), 40);
    chk("B=8 max T (GS=64)", max_T(8, 64), 0);
    printf("GS=32 working points (fallback; sw region ~doubles):\n");
    chk("B=1 max T (GS=32)", max_T(1, 32), 52);
    chk("B=2 max T (GS=32)", max_T(2, 32), 40);
    chk("B=4 max T (GS=32)", max_T(4, 32), 16);
    chk("B=8 max T (GS=32)", max_T(8, 32), 0);

    /* ---- 结构不变量 ---- */
    {
        LayerMap m = layer_map_qwen3(4, DIM, HID, QD, KVD, HD, 40, 64, 0);
        uint32_t offs[] = { m.desc, m.zero, m.sc, m.sw, m.rmsw, m.qkg,
                            m.cosq, m.sinq, m.cosk, m.sink, m.x, m.q, m.atto,
                            m.s1, m.s2, m.aq, m.cb, m.tmph, m.att, m.attf,
                            m.h1, m.h2, m.h3, m.kvh, m.vvh, m.wb, m.ci };
        const char *names[] = { "desc","zero","sc","sw","rmsw","qkg",
                                "cosq","sinq","cosk","sink","x","q","atto",
                                "s1","s2","aq","cb","tmph","att","attf",
                                "h1","h2","h3","kvh","vvh","wb","ci" };
        int n = (int)(sizeof offs / sizeof offs[0]);
        for (int i = 0; i < n; i++) {
            char buf[64];
            snprintf(buf, sizeof buf, "%s 16B-aligned", names[i]);
            chk(buf, (long)(offs[i] % 16), 0);
        }
        chk("ci 4096-aligned", (long)(m.ci % 4096), 0);
        LayerMap m0 = layer_map_qwen3(4, DIM, HID, QD, KVD, HD, 40, 64, 0);
        /* 相邻区差顺带锁前区尺寸：rmsw 是 dimb=2048，qkg 是 2*head_dim*2=512 */
        chk("rmsw size (qkg-rmsw)", (long)(m0.qkg - m0.rmsw), 2048);
        chk("qkg size (cosq-qkg)", (long)(m0.cosq - m0.qkg), 512);
        chk("B=4/T=40 fits", (long)m0.ok, 1);
        LayerMap m41 = layer_map_qwen3(4, DIM, HID, QD, KVD, HD, 41, 64, 0);
        chk("B=4/T=41 must NOT fit", (long)m41.ok, 0);
    }

    if (fails) { printf("%d FAILURES\n", fails); return 1; }
    printf("layer_map_qwen3: all checks passed\n");
    return 0;
}
