/* wsbench.c -- standalone board bench for the C4 MCFG2 OS/WS switching.
 *
 * Mirrors MultiEngineSpec's WS protocol on real silicon, at the full Core
 * level (ITCM program through the scalar core, so MCFG2 rides the matrix
 * queue exactly as a layer program would):
 *
 *   OS      : LMA(a1) LMB(b) MATMUL SM(c_os)
 *   WS      : LMA(a1) LMB(b) MCFG2(WS) MATMULA SM(c_ws1)      C = A1 x B
 *   WS-keep : LMA(a1) LMB(b) MCFG2(WS) MATMULA LMA(a2)
 *             MCFG2(WS_KEEP) MATMULA SM(c_ws2)                C = (A1+A2) x B
 *   OS-back : MCFG2(OS) LMA(a1) LMB(b) MATMUL SM(c_os2)       switch-back check
 *             (a1 now holds A2 -- P_WSK's copy loop overwrote it -- so the
 *              golden here is A2 x B, not A1 x B)
 *
 * Needs the outBuf-clear-on-dfSwitch RTL fix (Int8Engine MCFG2): on earlier
 * bitstreams the WS outBuf survives HALT and the MCFG2 invalidate does not
 * cover it, so WS chains accumulate every earlier WS chain's C. On such a
 * bitstream reload the PL (fresh reset) before this bench.
 *
 * Tiles live in DTCM (direct host window writes, no DMA/resmem). Per-kick
 * cycle counts come from the PERF cycles counter (XN_CTRL+0x44).
 *
 * Modes (argv[1], default "all"): os | ws | wskeep | osback | all.  A single
 * mode runs just that chain and prints `SUMMARY mode=<m> cycles=<c> pass=<01>`
 * (webui /osws/run parses that line).  Every chain now starts with an
 * MCFG2(OS) whose dataflow bits match reset default: it never pulses the
 * invalidate, but the instruction itself costs ~2 cycles (os-baseline 142 ->
 * 144, board-measured 2026-11-02, same on repeat runs) -- and after any WS
 * chain it restores OS, making single-mode invocations order-independent
 * despite modeReg persisting across HALT.  `osback` packs
 * A2 over the A1 window first (the ws-keep copy loop does that on board, and
 * its golden here is A2 x B).
 *
 * Board: sudo -n npurun ./wsbench [os|ws|wskeep|osback|all]
 *        (env same as run3n-v1.sh carrier; osws_run.sh wraps claim+release).
 */
#include "xnpu/hal.h"
#include "xnpu/emit.h"
#include "xnpu/enc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* DTCM byte offsets */
#define OFF_A1   0x0000u
#define OFF_A2   0x0100u
#define OFF_B    0x0200u
#define OFF_C_OS 0x0300u
#define OFF_C_WS1 0x0700u
#define OFF_C_WS2 0x0B00u
#define OFF_C_OS2 0x0F00u

#define N 16

static int8_t A1[N][N], A2[N][N], Bm[N][N];
static int32_t golden1[N][N], golden2[N][N], goldenA2[N][N];

static uint32_t rd32(uint32_t off) { return xn_rd(off); }

static uint32_t prog_words[64];

static void pack_tile(uint32_t off, const int8_t t[N][N]) {
    for (int i = 0; i < N; i++) {
        uint32_t row = 0;
        for (int j = 0; j < N; j++) {
            row |= ((uint32_t)(uint8_t)t[i][j]) << (8u * (j % 4u));
            if (j % 4u == 3u) { xn_wr(XN_DTCM + off + (uint32_t)i * 16u + (j / 4u) * 4u, row); row = 0; }
        }
    }
}

static int32_t read_c_word(uint32_t off, int i, int j) {
    return (int32_t)xn_rd(XN_DTCM + off + (uint32_t)(i * N + j) * 4u);
}

static int check(const char *tag, uint32_t off, const int32_t g[N][N]) {
    int bad = 0;
    for (int i = 0; i < N && bad < 4; i++)
        for (int j = 0; j < N && bad < 4; j++) {
            int32_t got = read_c_word(off, i, j);
            if (got != g[i][j]) {
                printf("  %s C[%d][%d] got %d want %d\n", tag, i, j, got, g[i][j]);
                bad++;
            }
        }
    printf("%s: %s\n", tag, bad ? "FAIL" : "PASS");
    return bad;
}

/* one program, one kick; returns elapsed cycles */
static uint32_t run_prog(const char *what, int n_insns) {
    for (int i = 0; i < n_insns; i++) xn_wr(XN_ITCM + (uint32_t)i * 4u, prog_words[i]);
    xn_wr(XN_ITCM + (uint32_t)n_insns * 4u, ENC_HALT);
    if (xnpu_kick(what) != 0) { printf("%s: kick failed\n", what); exit(1); }
    return rd32(XN_CTRL + 0x44);   /* PERF counters are per-kick (startPulse) */
}



static int emit_prog(uint32_t *p, int which) {
    Emit e = { p, 0, 64, 0 };
    uint32_t cfg_ws = 2, cfg_keep = 6, cfg_os = 0;
    if (which == 0) {            /* OS baseline */
        eli(&e, 5, OFF_A1); eli(&e, 6, OFF_B); eli(&e, 7, OFF_C_OS);
        eli(&e, 12, cfg_os);
        ep(&e, enc_mcfg2(12));   /* bits==reset default: fresh-PL no-op, after
                                  * WS restores OS (single-mode robustness) */
        ep(&e, enc_lma(5)); ep(&e, enc_lmb(6));
        ep(&e, enc_matmul());
        ep(&e, enc_sm(7));
    } else if (which == 1) {     /* WS: waterfall once */
        eli(&e, 5, OFF_A1); eli(&e, 6, OFF_B); eli(&e, 7, OFF_C_WS1);
        eli(&e, 10, cfg_ws); eli(&e, 12, cfg_os);
        /* modeReg persists across HALT: force a dataflow toggle so the
         * MCFG2 invalidate (clear) fires -- ws->ws changes clear nothing. */
        ep(&e, enc_mcfg2(12));
        ep(&e, enc_lma(5)); ep(&e, enc_lmb(6));
        ep(&e, enc_mcfg2(10));
        ep(&e, enc_matmul_acc());
        ep(&e, enc_sm(7));
    } else if (which == 2) {     /* WS-keep, spec shape: DTCM A rewritten mid-
                                  * program by scalar LW/SW (V1 row tokens gate
                                  * the SW against MATMULA#1's read tokens) */
        eli(&e, 5, OFF_A1); eli(&e, 8, OFF_A2); eli(&e, 6, OFF_B);
        eli(&e, 7, OFF_C_WS2);
        eli(&e, 10, cfg_ws); eli(&e, 11, cfg_keep); eli(&e, 12, cfg_os);
        ep(&e, enc_mcfg2(12));
        ep(&e, enc_lma(5)); ep(&e, enc_lmb(6));
        ep(&e, enc_mcfg2(10));
        ep(&e, enc_matmul_acc());
        /* copy 256B = 64 words from A2 to A1 */
        ep(&e, enc_addi(20, 5, 0));            /* dst = A1 */
        ep(&e, enc_addi(21, 8, 0));            /* src = A2 */
        ep(&e, enc_addi(22, 0, 64));           /* count */
        int Lcp = e.n;
        ep(&e, enc_lw(23, 21, 0));
        ep(&e, enc_sw(23, 20, 0));
        ep(&e, enc_addi(20, 20, 4));
        ep(&e, enc_addi(21, 21, 4));
        ep(&e, enc_addi(22, 22, -1));
        eb_back(&e, 1, 22, 0, Lcp);            /* BNE x22, x0, Lcp */
        ep(&e, enc_mcfg2(11));
        ep(&e, enc_matmul_acc());
        ep(&e, enc_sm(7));
    } else {                     /* OS after WS (switch-back) */
        eli(&e, 5, OFF_A1); eli(&e, 6, OFF_B); eli(&e, 7, OFF_C_OS2);
        eli(&e, 10, cfg_os);
        ep(&e, enc_mcfg2(10));
        ep(&e, enc_lma(5)); ep(&e, enc_lmb(6));
        ep(&e, enc_matmul());
        ep(&e, enc_sm(7));
    }
    ep(&e, ENC_HALT);
    return e.n;
}

struct mode_spec {
    const char *name;   /* argv spelling */
    const char *tag;    /* print/summary tag */
    int which;          /* emit_prog chain selector */
    uint32_t coff;      /* C window to check */
    const int32_t (*golden)[N];
};

int main(int argc, char **argv) {
    const char *m = argc > 1 ? argv[1] : "all";
    const struct mode_spec single[] = {
        { "os",     "os",            0, OFF_C_OS,  golden1 },
        { "ws",     "ws",            1, OFF_C_WS1, golden1 },
        { "wskeep", "ws-keep",       2, OFF_C_WS2, golden2 },
        { "osback", "os-after-ws",   3, OFF_C_OS2, goldenA2 },
    };
    int single_mode = -1;
    for (int i = 0; i < 4; i++)
        if (!strcmp(m, single[i].name)) single_mode = i;
    if (single_mode < 0 && strcmp(m, "all")) {
        printf("usage: wsbench [os|ws|wskeep|osback|all]\n");
        return 2;
    }

    if (xnpu_init() != 0) { printf("xnpu_init failed\n"); return 1; }

    srand(42);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A1[i][j] = (int8_t)(rand() % 15 - 7);
            A2[i][j] = (int8_t)(rand() % 15 - 7);
            Bm[i][j] = (int8_t)(rand() % 15 - 7);
        }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            long s1 = 0, s2 = 0;
            for (int k = 0; k < N; k++) { s1 += (long)A1[i][k] * Bm[k][j]; s2 += (long)A2[i][k] * Bm[k][j]; }
            golden1[i][j] = (int32_t)s1;
            golden2[i][j] = (int32_t)(s1 + s2);
            goldenA2[i][j] = (int32_t)s2;
        }

    pack_tile(OFF_A1, A1); pack_tile(OFF_A2, A2); pack_tile(OFF_B, Bm);
    if (single_mode == 3)
        pack_tile(OFF_A1, A2); /* osback standalone: A1 window holds A2 (same
                                * DTCM state the in-suite chain-3 assumes) */

    int bad = 0;
    uint32_t c;

    if (single_mode >= 0) {
        const struct mode_spec *s = &single[single_mode];
        c = run_prog(s->tag, emit_prog(prog_words, s->which));
        printf("%s: %u cycles\n", s->tag, c);
        bad += check(s->tag, s->coff, s->golden);
        printf("SUMMARY mode=%s cycles=%u pass=%d\n", s->name, c, bad ? 0 : 1);
        return bad ? 1 : 0;
    }

    c = run_prog("os-baseline", emit_prog(prog_words, 0));
    printf("os-baseline: %u cycles\n", c);
    int b = check("os-baseline", OFF_C_OS, golden1);
    bad += b;
    printf("SUMMARY mode=os cycles=%u pass=%d\n", c, b ? 0 : 1);

    c = run_prog("ws-waterfall", emit_prog(prog_words, 1));
    printf("ws-waterfall: %u cycles\n", c);
    b = check("ws-waterfall", OFF_C_WS1, golden1);
    bad += b;
    printf("SUMMARY mode=ws cycles=%u pass=%d\n", c, b ? 0 : 1);

    c = run_prog("ws-keep", emit_prog(prog_words, 2));
    printf("ws-keep (2x MATMULA): %u cycles\n", c);
    b = check("ws-keep", OFF_C_WS2, golden2);
    bad += b;
    printf("SUMMARY mode=wskeep cycles=%u pass=%d\n", c, b ? 0 : 1);

    c = run_prog("os-after-ws", emit_prog(prog_words, 3));
    printf("os-after-ws: %u cycles\n", c);
    b = check("os-after-ws (A2 x B: P_WSK copied A2 over A1)",
              OFF_C_OS2, goldenA2);
    bad += b;
    printf("SUMMARY mode=osback cycles=%u pass=%d\n", c, b ? 0 : 1);

    printf("wsbench: %s (%d mismatches)\n", bad ? "FAIL" : "ALL PASS", bad);
    return bad ? 1 : 0;
}
