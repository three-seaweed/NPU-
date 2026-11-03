/* bf16selftest.c -- standalone board selftest for the bf16 8x8 tier (C6).
 *
 * Verifies FP.MATMUL / FP.MATMULA / FP.SM on real silicon via the `full`
 * variant bitstream (Parameters(bf16Matmul=true)).  Mirrors MultiEngineSpec's
 * bf16 tier tests on the board, at the full Core level (ITCM program through
 * the scalar core, so FP.* rides the matrix queue exactly as a layer program
 * would):
 *
 *   Test 1 (K=8):  LMA(a1) LMB(b) FP.MATMUL FP.SM(c)     -> C = A1 x B
 *   Test 2 (K=16): LMA(a1) LMB(b) FP.MATMUL
 *                  LMA(a2)          FP.MATMULA FP.SM(c) -> C = A1 x B + A2 x B
 *
 * Golden = serial fp32 fmaf fold of widened bf16 operands, single final RNE
 * (f2bf16).  Bit-exact for well-scaled inputs (exp 110..125).
 *
 * Tiles live in DTCM (direct host window writes, no DMA/resmem). Per-kick
 * cycle counts from the PERF cycles counter (XN_CTRL+0x44).
 *
 * Board: sudo -n npurun ./bf16selftest_run.sh
 *   (wrapper re-exports XNPU_RESMEM_PHYS/SIZE/XNPU_BASE; sudo env_reset
 *    scrubs XNPU_* and npurun doesn't accept `env` as a command arg)
 */
#define _POSIX_C_SOURCE 200809L   /* nanosleep under -std=c11 */
#include "xnpu/hal.h"
/* BF16_CACHE block needs these directly */
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "xnpu/emit.h"
#include "xnpu/enc.h"
#include "bf16test.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* DTCM byte offsets (each bf16 8x8 tile = 128 B = 8 DTCM rows; 0x200 stride
 * keeps windows far apart, DTCM is 512 KB) */
#define OFF_A1  0x0000u
#define OFF_A2  0x0200u
#define OFF_B   0x0400u
#define OFF_C   0x0600u

static uint16_t A1[BF16_DIM][BF16_DIM], A2[BF16_DIM][BF16_DIM], B[BF16_DIM][BF16_DIM];
static uint16_t gold1[BF16_DIM][BF16_DIM], gold2[BF16_DIM][BF16_DIM];

static uint32_t rd32(uint32_t off) { return xn_rd(off); }
static uint32_t prog_words[64];

/* pack a bf16 tile to DTCM via MMIO */
static void pack_mmio(uint32_t off, const uint16_t t[BF16_DIM][BF16_DIM]) {
    uint32_t words[BF16T_TILE_WORDS];
    bf16_pack_words(words, t);
    for (int w = 0; w < BF16T_TILE_WORDS; w++)
        xn_wr(XN_DTCM + off + (uint32_t)w * 4u, words[w]);
}

/* read element (r,j) from DTCM via MMIO */
static uint16_t unpack_mmio(uint32_t off, int r, int j) {
    /* fetch the one beat holding (r, j), decode in place -- do NOT re-index a
     * 4-word window with whole-tile math (the 2026-11-03 board-chase: the old
     * r*2 window + bf16_unpack_word(r) double-decode read stack OOB for every
     * row >= 1 and masqueraded as a silicon bug for five hours). */
    int wi = (r * (BF16_DIM / 8) + j / 8) * 4;   /* beat (r*bpr + j/8) */
    uint32_t words[4];
    for (int w = 0; w < 4; w++)
        words[w] = xn_rd(XN_DTCM + off + (uint32_t)(wi + w) * 4u);
    int e = j % 8;                               /* 2 bf16 per 32-bit word */
    return (uint16_t)(words[e / 2] >> (16 * (e % 2)));
}

/* BF16_DUMP=1: print tile words (pack verification + C-region forensics) */
static void dump_words(const char *tag, uint32_t off, int nwords) {
    printf("dump %s @0x%03x:", tag, off);
    for (int w = 0; w < nwords; w++) {
        if (w % 4 == 0) printf("\n  +%03x:", off + w * 4);
        printf(" %08x", xn_rd(XN_DTCM + off + (uint32_t)w * 4u));
    }
    printf("\n");
}

/* compare RTL output against golden; returns mismatch count */
static int check_bf16(const char *tag, uint32_t off, const uint16_t g[BF16_DIM][BF16_DIM]) {
    int bad = 0;
    for (int i = 0; i < BF16_DIM && bad < 4; i++)
        for (int j = 0; j < BF16_DIM && bad < 4; j++) {
            uint16_t got = unpack_mmio(off, i, j);
            if (got != g[i][j]) {
                printf("  %s C[%d][%d] got 0x%04x want 0x%04x\n", tag, i, j, got, g[i][j]);
                bad++;
            }
        }
    printf("%s: %s\n", tag, bad ? "FAIL" : "PASS");
    return bad;
}

/* write ITCM program, kick, return PERF cycles */
static uint32_t run_prog(const char *what, int n_insns) {
    for (int i = 0; i < n_insns; i++) xn_wr(XN_ITCM + (uint32_t)i * 4u, prog_words[i]);
    xn_wr(XN_ITCM + (uint32_t)n_insns * 4u, ENC_HALT);
    if (xnpu_kick(what) != 0) { printf("%s: kick failed\n", what); exit(1); }
    return rd32(XN_CTRL + 0x44);
}

/* K=8: LMA(a1) LMB(b) FP.MATMUL FP.SM(c) HALT */
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

/* K=16 (two tiles): LMA(a1) LMB(b) FP.MATMUL LMA(a2) FP.MATMULA FP.SM(c) HALT */
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

int main(void) {
    if (xnpu_init() != 0) { printf("xnpu_init failed\n"); return 1; }

    /* generate stimulus (same LCG as MultiEngineSpec.bfTile) */
    uint64_t lcg = BF16_LCG_INIT;
    for (int i = 0; i < BF16_DIM; i++)
        for (int j = 0; j < BF16_DIM; j++)
            A1[i][j] = bf16_bfTile_elem(&lcg, i, j, 11);
    for (int i = 0; i < BF16_DIM; i++)
        for (int j = 0; j < BF16_DIM; j++)
            A2[i][j] = bf16_bfTile_elem(&lcg, i, j, 17);
    for (int i = 0; i < BF16_DIM; i++)
        for (int j = 0; j < BF16_DIM; j++)
            B[i][j]  = bf16_bfTile_elem(&lcg, i, j, 13);

    /* CPU golden */
    for (int i = 0; i < BF16_DIM; i++)
        for (int j = 0; j < BF16_DIM; j++) {
            gold1[i][j] = bf16_golden(A1, B, i, j);
            gold2[i][j] = bf16_golden_2t(A1, A2, B, i, j);
        }

    /* pack operands to DTCM */
    pack_mmio(OFF_A1, A1);
    pack_mmio(OFF_A2, A2);
    pack_mmio(OFF_B,  B);

    int dump = getenv("BF16_DUMP") != NULL;
    if (dump) {
        dump_words("A1", OFF_A1, BF16T_TILE_WORDS);
        dump_words("B",  OFF_B,  BF16T_TILE_WORDS);
        dump_words("C-before", OFF_C, BF16T_TILE_WORDS);
    }

    int bad = 0;
    uint32_t c;

    if (getenv("BF16_TRACE")) {
        /* C[1][0..3] (words OFF_C+0x10/+0x14) forensics: when does the
         * engine's write actually land relative to the CPU seeing HALT?
         * golden k8: +0x10 = b7a8b8aa (C[1][0]=b8aa, C[1][1]=b7a8). */
        struct timespec ms300 = { 0, 300 * 1000 * 1000 };
        c = run_prog("k8t", emit_k8(prog_words));
        printf("trace: cycles=%u\n", c);
        printf("trace: imm    w600=%08x w610=%08x w614=%08x\n",
               rd32(XN_DTCM + OFF_C), rd32(XN_DTCM + OFF_C + 0x10),
               rd32(XN_DTCM + OFF_C + 0x14));
        nanosleep(&ms300, NULL);
        printf("trace: +300ms w610=%08x w614=%08x\n",
               rd32(XN_DTCM + OFF_C + 0x10), rd32(XN_DTCM + OFF_C + 0x14));
        xn_wr(XN_ITCM, ENC_HALT);           /* empty kick: HALT only */
        xnpu_kick("empty");
        printf("trace: empty  w610=%08x w614=%08x\n",
               rd32(XN_DTCM + OFF_C + 0x10), rd32(XN_DTCM + OFF_C + 0x14));
        nanosleep(&ms300, NULL);
        printf("trace: +300ms w610=%08x w614=%08x\n",
               rd32(XN_DTCM + OFF_C + 0x10), rd32(XN_DTCM + OFF_C + 0x14));
        return 0;
    }

    if (getenv("BF16_POLL")) {
        /* Late-write vs lying-read discriminator: hammer w610 after the kick.
         * CHANGED => the DTCM content itself flips late (a real delayed write);
         * never   => content was stable and the read path returns garbage. */
        c = run_prog("k8p", emit_k8(prog_words));
        uint32_t first = rd32(XN_DTCM + OFF_C + 0x10);
        uint32_t cur = first;
        long i = 0;
        const long N = 5000000L;
        while (i < N) {
            cur = rd32(XN_DTCM + OFF_C + 0x10);
            if (cur != first) break;
            i++;
        }
        printf("poll: first=%08x %s cur=%08x after %ld re-reads\n",
               first, cur != first ? "CHANGED" : "never", cur, i);
        return 0;
    }

    if (getenv("BF16_CACHE")) {
        /* CPU-cache discriminator: is the DTCM window mapping cacheable?
         * 1) kick k8, read C row 0/1 via the hal window (caches the lines);
         * 2) perturb B[0][0], repack, kick k8 again (C row 0 must change);
         * 3) read again via the hal window -- STALE means the engine write
         *    did not invalidate the CPU's cached lines;
         * 4) read the same words through a FRESH private mmap of the same
         *    phys range -- that is what actually lives in DTCM (if the pack
         *    writes never left the CPU cache, this shows the OLD data). */
        c = run_prog("k8c1", emit_k8(prog_words));
        uint32_t a600 = rd32(XN_DTCM + OFF_C), a610 = rd32(XN_DTCM + OFF_C + 0x10);
        printf("cache: kick1 hal-read  w600=%08x w610=%08x\n", a600, a610);

        B[0][0] = 0x3c00;              /* any value != LCG original */
        pack_mmio(OFF_B, B);
        c = run_prog("k8c2", emit_k8(prog_words));
        uint32_t b600 = rd32(XN_DTCM + OFF_C), b610 = rd32(XN_DTCM + OFF_C + 0x10);
        printf("cache: kick2 hal-read  w600=%08x w610=%08x\n", b600, b610);

        int fd = open("/dev/mem", O_RDWR | O_SYNC);
        volatile uint32_t *w = MAP_FAILED;
        if (fd >= 0)
            w = mmap(NULL, 0x1000, PROT_READ, MAP_SHARED, fd,
                     (off_t)0x400000000ULL + XN_DTCM + 0x600);
        if (w == MAP_FAILED) {
            printf("cache: private mmap failed\n");
        } else {
            printf("cache: fresh-mmap     w600=%08x w610=%08x\n", w[0], w[4]);
            munmap((void *)w, 0x1000);
        }
        if (fd >= 0) close(fd);
        printf("cache: verdict read-stale=%s (w600 changed %d, w610 changed %d)\n",
               (b600 == a600) ? "YES" : "no",
               b600 != a600, b610 != a610);
        return 0;
    }

    c = run_prog("k8", emit_k8(prog_words));
    if (dump) dump_words("C-after-k8", OFF_C, BF16T_TILE_WORDS);
    printf("k8: %u cycles\n", c);
    bad += check_bf16("k8", OFF_C, gold1);

    c = run_prog("k16", emit_k16(prog_words));
    printf("k16: %u cycles\n", c);
    bad += check_bf16("k16", OFF_C, gold2);
    if (dump) dump_words("C-after-k16", OFF_C, BF16T_TILE_WORDS);

    printf("bf16selftest: %s (%d mismatches)\n", bad ? "FAIL" : "ALL PASS", bad);
    return bad ? 1 : 0;
}
