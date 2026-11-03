/* spinf.c -- PL clock frequency probe (2026-11-03, f60bf16 bring-up).
 *
 * Kicks a long scalar loop through the real core and reports
 * PERF cycles / wall time. This measures the ACTUAL PL clock, independent
 * of the kernel's clk_summary belief (raw CRL_APB divisor writes made via
 * /dev/mem do not propagate into the kernel clock tree).
 *
 * The loop's cycle count is deterministic, so the reported cycles must stay
 * constant across runs; only the wall time (and thus the ratio) moves when
 * the FCLK0 divisor changes.
 *
 * Build/run: make spinf; on board: sudo npurun ./spinf 20000000
 * (arg = loop iterations; 20M is ~0.6 s at 100 MHz).
 */
#define _POSIX_C_SOURCE 200809L   /* clock_gettime under -std=c11 */
#include "xnpu/hal.h"
#include "xnpu/emit.h"
#include "xnpu/enc.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint32_t prog_words[16];
static uint32_t rd32(uint32_t off) { return xn_rd(off); }

int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 20000000L;
    if (n <= 0 || n > 100000000L) { printf("bad n\n"); return 2; }

    if (xnpu_init() != 0) { printf("xnpu_init failed\n"); return 1; }

    /* x5 = 0 ; x6 = n ; loop: addi x5,x5,1 ; bne x5,x6,-4 ; halt
     * (x5 MUST be zeroed in-program: registers survive across kicks, so a
     * second run would otherwise start from n and spin 2^32 iterations) */
    Emit e = { prog_words, 0, 16, 0 };
    ep(&e, 0x00000293u);   /* addi x5, x0, 0        */
    eli(&e, 6, (uint32_t)n);
    ep(&e, 0x00128293u);   /* addi x5, x5, 1        */
    ep(&e, 0xFE629EE3u);   /* bne  x5, x6, -4       */
    ep(&e, ENC_HALT);
    for (int i = 0; i < e.n; i++)
        xn_wr(XN_ITCM + (uint32_t)i * 4u, prog_words[i]);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (xnpu_kick("spinf") != 0) { printf("kick failed\n"); return 1; }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    uint32_t cyc = rd32(XN_CTRL + 0x44);

    double dt = (double)(t1.tv_sec - t0.tv_sec) +
                (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("spinf: n=%ld cycles=%u wall=%.6fs freq=%.4f MHz\n",
           n, cyc, dt, (double)cyc / dt / 1e6);
    return 0;
}
