/* bf16pack_test.c -- host x86 verification of the C packing helpers
 * (bf16test.h, bf16 8x8) against themselves.
 *
 * Generates A=bfTile(11), B=bfTile(13) with the same LCG as MultiEngineSpec,
 * packs to word arrays, computes golden, and does a pack->unpack round-trip
 * self-check.  The old Bf16Dump cross-check went away with the 2026-11-01
 * 8x8 rework (main deleted); the authoritative contract check now runs on
 * silicon via bf16selftest.
 *
 * Build: gcc -O2 -I. -o build/bf16pack_test bf16pack_test.c -lm
 */
#include "bf16test.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    uint64_t lcg = BF16_LCG_INIT;
    uint16_t A[BF16_DIM][BF16_DIM], B[BF16_DIM][BF16_DIM];

    /* Generate all of A first, then all of B (matches Scala order) */
    for (int i = 0; i < BF16_DIM; i++)
        for (int j = 0; j < BF16_DIM; j++)
            A[i][j] = bf16_bfTile_elem(&lcg, i, j, 11);
    for (int i = 0; i < BF16_DIM; i++)
        for (int j = 0; j < BF16_DIM; j++)
            B[i][j] = bf16_bfTile_elem(&lcg, i, j, 13);

    /* quick sanity: print first 4 A and B elements */
    printf("A[0][0..3]: %04x %04x %04x %04x\n", A[0][0], A[0][1], A[0][2], A[0][3]);
    printf("B[0][0..3]: %04x %04x %04x %04x\n", B[0][0], B[0][1], B[0][2], B[0][3]);

    /* pack->unpack round-trip self-check */
    uint32_t wordsA[BF16T_TILE_WORDS], wordsB[BF16T_TILE_WORDS];
    bf16_pack_words(wordsA, A);
    bf16_pack_words(wordsB, B);

    int rt_bad = 0;
    for (int i = 0; i < BF16_DIM; i++)
        for (int j = 0; j < BF16_DIM; j++) {
            uint16_t ua = bf16_unpack_word(wordsA, i, j);
            uint16_t ub = bf16_unpack_word(wordsB, i, j);
            if (ua != A[i][j] || ub != B[i][j]) {
                if (rt_bad < 5)
                    printf("RT mismatch A[%d][%d]: pack=%04x unpack=%04x | B[%d][%d]: pack=%04x unpack=%04x\n",
                           i, j, A[i][j], ua, i, j, B[i][j], ub);
                rt_bad++;
            }
        }
    printf("round-trip: %s (%d bad)\n", rt_bad ? "FAIL" : "PASS", rt_bad);
    if (rt_bad) return 1;

    /* golden */
    uint16_t gold[BF16_DIM][BF16_DIM];
    for (int i = 0; i < BF16_DIM; i++)
        for (int j = 0; j < BF16_DIM; j++)
            gold[i][j] = bf16_golden(A, B, i, j);

    /* print for diff with Bf16Dump */
    printf("=== A words (%d) ===\n", BF16T_TILE_WORDS);
    for (int w = 0; w < BF16T_TILE_WORDS; w++)
        printf("%08x\n", wordsA[w]);
    printf("=== B words (%d) ===\n", BF16T_TILE_WORDS);
    for (int w = 0; w < BF16T_TILE_WORDS; w++)
        printf("%08x\n", wordsB[w]);
    printf("=== golden C (%d bf16) ===\n", BF16_DIM * BF16_DIM);
    for (int i = 0; i < BF16_DIM; i++)
        for (int j = 0; j < BF16_DIM; j++)
            printf("%04x\n", gold[i][j]);

    printf("bf16pack_test: done\n");
    return 0;
}
