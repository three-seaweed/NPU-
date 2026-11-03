/* bf16test.h -- portable bf16 8x8 tile packing + golden (shared by host test
 * and board program).  No MMIO -- pure computation on word arrays.
 *
 * Mirrors MultiEngineSpec.scala bf16 8x8 tier (C6):
 *   bfPackTile: an n-bf16 row is n/8 beats (one at n=8); beat (r*bpr+half) at
 *     byte base+(r*bpr+half)*16 holds elements [half*8..half*8+7], element e
 *     in bits [16e+15:16e] of the 128-bit beat.
 *   golden: acc = fold fma(bf162f(A[i][k]), bf162f(B[k][j])) over k=0..n-1;
 *     result = f2bf16(acc)  (single RNE at FP.SM exit, no intermediate rounding)
 *   stimulus: LCG bfTile with exp 110..125 (well-scaled) + sprinkled zeros.
 *
 * 2026-11-01 the tier shrank 16x16 -> 8x8 (Bf16MatrixDim, LUT diet) and the
 * Bf16Dump main was deleted; the host test now self-checks via pack->unpack
 * round-trip, the board program carries its own golden.
 */
#ifndef BF16TEST_H
#define BF16TEST_H

#include "xnpu/enc.h"   /* f2bf16, bf162f (portable, no board deps) */

#include <stdint.h>
#include <math.h>

#define BF16_DIM         8
#ifndef BF16T_TILE_BYTES
#define BF16T_TILE_BYTES  (BF16_DIM * BF16_DIM * 2)   /* 128; renamed: emit.h BF16_TILE_BYTES is the vector-unit tile (512) */
#endif
#define BF16T_TILE_WORDS  (BF16T_TILE_BYTES / 4)       /* 32 */

/* ---- LCG stimulus (mirrors MultiEngineSpec.scala:376-386) ---- */
#define BF16_LCG_INIT   0x243F6A8885A308D3ULL
#define BF16_LCG_MULT   6364136223846793005ULL
#define BF16_LCG_INC    1442695040888963407ULL

static inline uint16_t bf16_rnd16(uint64_t *lcg) {
    *lcg = *lcg * BF16_LCG_MULT + BF16_LCG_INC;
    return (uint16_t)((*lcg >> 40) & 0xFFFFULL);
}

/* bfTile element (i,j): exp 110..125, sprinkled zeros.
 * Mirrors: if ((i*7+j*13+zeroEvery)%zeroEvery==0) 0x0000
 *          else ((rnd16()&1)<<15) | ((110+(rnd16()&0xf))<<7) | (rnd16()&0x7f) */
static inline uint16_t bf16_bfTile_elem(uint64_t *lcg, int i, int j, int zeroEvery) {
    if (((i * 7 + j * 13 + zeroEvery) % zeroEvery) == 0) return 0x0000;
    uint16_t sign = (uint16_t)((bf16_rnd16(lcg) & 1) << 15);
    uint16_t exp  = (uint16_t)((110 + (bf16_rnd16(lcg) & 0xf)) << 7);
    uint16_t man  = (uint16_t)(bf16_rnd16(lcg) & 0x7f);
    return (uint16_t)(sign | exp | man);
}

/* ---- packing (mirrors MultiEngineSpec.bfPackTile) ----
 * words[] = BF16T_TILE_WORDS uint32 words of the 128-byte tile.
 * Beat (r*bpr+half) at words[(r*bpr+half)*4 .. +3]; bpr = BF16_DIM/8 beats
 * per row (one at n=8).  Element e (0..7) in bits [16e+15:16e] of the beat.
 * A 128-bit beat = 8 bf16 elements; split into lo (elem 0-3) + hi (elem 4-7)
 * to avoid __int128 / UB shifts. */
static inline void bf16_pack_words(uint32_t *words,
                                   const uint16_t t[BF16_DIM][BF16_DIM]) {
    const int bpr = BF16_DIM / 8;
    for (int r = 0; r < BF16_DIM; r++) {
        for (int half = 0; half < bpr; half++) {
            uint64_t lo = 0, hi = 0;
            for (int e = 0; e < 8; e++) {
                uint64_t v = t[r][half * 8 + e] & 0xFFFFu;
                if (e < 4) lo |= v << (16 * e);
                else       hi |= v << (16 * (e - 4));
            }
            int wi = (r * bpr + half) * 4;
            words[wi + 0] = (uint32_t)(lo & 0xFFFFFFFFu);
            words[wi + 1] = (uint32_t)(lo >> 32);
            words[wi + 2] = (uint32_t)(hi & 0xFFFFFFFFu);
            words[wi + 3] = (uint32_t)(hi >> 32);
        }
    }
}

/* Unpack element (r,j) from word array. Mirrors bfUnpack. */
static inline uint16_t bf16_unpack_word(const uint32_t *words, int r, int j) {
    int wi = (r * (BF16_DIM / 8) + j / 8) * 4;
    uint64_t lo = (uint64_t)words[wi] | ((uint64_t)words[wi + 1] << 32);
    uint64_t hi = (uint64_t)words[wi + 2] | ((uint64_t)words[wi + 3] << 32);
    int e = j % 8;
    if (e < 4) return (uint16_t)((lo >> (16 * e)) & 0xFFFFu);
    else       return (uint16_t)((hi >> (16 * (e - 4))) & 0xFFFFu);
}

/* ---- golden (mirrors MultiEngineSpec golden) ----
 * acc = foldLeft(0, k=0..n-1) -> fma(bf162f(A[i][k]), bf162f(B[k][j]), acc)
 * result = f2bf16(acc)  (single RNE, no intermediate rounding) */
static inline uint16_t bf16_golden(const uint16_t A[BF16_DIM][BF16_DIM],
                                   const uint16_t B[BF16_DIM][BF16_DIM],
                                   int i, int j) {
    float acc = 0.0f;
    for (int k = 0; k < BF16_DIM; k++)
        acc = fmaf(bf162f(A[i][k]), bf162f(B[k][j]), acc);
    return f2bf16(acc);
}

/* Two-tile (K=2n) golden: C = A0 x B + A1 x B (continuous fp32 fold, no
 * tile-boundary rounding). Mirrors MultiEngineSpec test 2 golden. */
static inline uint16_t bf16_golden_2t(const uint16_t A0[BF16_DIM][BF16_DIM],
                                      const uint16_t A1[BF16_DIM][BF16_DIM],
                                      const uint16_t B[BF16_DIM][BF16_DIM],
                                      int i, int j) {
    float acc = 0.0f;
    for (int k = 0; k < BF16_DIM; k++)
        acc = fmaf(bf162f(A0[i][k]), bf162f(B[k][j]), acc);
    for (int k = 0; k < BF16_DIM; k++)
        acc = fmaf(bf162f(A1[i][k]), bf162f(B[k][j]), acc);
    return f2bf16(acc);
}

#endif /* BF16TEST_H */
