/* enc.h -- 指令编码器（plan 附录 D 纪律：从 examples/llama2.c/runq_xnpu.c
 * :434-607 移植，不 fork 文件。编码必须与 src/main/scala/Decode.scala 的
 * Encode 对象逐位一致；改这里必须同步改四处（Decode/Encode、Alu op/token、
 * Control、本编码器），见 CLAUDE.md 硬件变更检查单第 2 条）。
 *
 * Phase 3 新增发射器（fe_qknorm / fe_rope half-split / o_proj fe_linear）
 * 全部复用本文件既有指令，不引入新 opcode。 */
#ifndef Q3_ENC_H
#define Q3_ENC_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* bf16 <-> fp32.  Narrowing is round-to-nearest-even, matching the RTL
 * (docs/UserGuide.md 4.3); widening is a zero-extend of the mantissa.
 * (runq_xnpu.c:436-443) */
static inline uint16_t f2bf16(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    if (((b >> 23) & 0xFFu) == 0xFFu) return (uint16_t)(b >> 16);  /* Inf/NaN pass through */
    b += 0x7FFFu + ((b >> 16) & 1u);
    return (uint16_t)(b >> 16);
}
static inline float bf162f(uint16_t h) {
    uint32_t b = (uint32_t)h << 16; float f; memcpy(&f, &b, 4); return f;
}
static inline uint32_t f2bits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }

/* ---- opcode 常量 (runq_xnpu.c:450-459) ---- */
#define OPC_OPIMM   0x13u
#define OPC_OP      0x33u
#define OPC_LUI     0x37u
#define OPC_LOAD    0x03u
#define OPC_STORE   0x23u
#define OPC_BRANCH  0x63u
#define OPC_JAL     0x6Fu
#define OPC_CUSTOM0 0x0Bu
#define OPC_CUSTOM1 0x2Bu
#define ENC_HALT    0x00000000u

/* I-type immediates are sign-extended by the core, so anything outside
 * [-2048, 2047] silently wraps when packed with & 0xFFF -- an andi mask of
 * 0x800 (2*RES_TILE_BYTES) becomes & ~0x7FF and corrupted the dual cb-slot
 * select (dual replay, 2026-10-26). Abort at generation time instead. */
static inline uint32_t enc_imm12(int imm) {
    if (imm < -2048 || imm > 2047) {
        fprintf(stderr, "enc: immediate %d does not fit a signed 12-bit field\n", imm);
        abort();
    }
    return ((uint32_t)imm & 0xFFFu) << 20;
}

/* ---- 标量 (runq_xnpu.c:461-498) ---- */
static inline uint32_t enc_addi(int rd, int rs1, int imm) {
    return enc_imm12(imm) | ((rs1 & 0x1F) << 15) | ((rd & 0x1F) << 7) | OPC_OPIMM; }
static inline uint32_t enc_andi(int rd, int rs1, int imm) {
    return enc_imm12(imm) | ((rs1 & 0x1F) << 15) | (7u << 12) |
           ((rd & 0x1F) << 7) | OPC_OPIMM; }
static inline uint32_t enc_xori(int rd, int rs1, int imm) {
    return enc_imm12(imm) | ((rs1 & 0x1F) << 15) | (4u << 12) |
           ((rd & 0x1F) << 7) | OPC_OPIMM; }
static inline uint32_t enc_lui(int rd, int imm20) {
    return (((uint32_t)imm20 & 0xFFFFFu) << 12) | ((rd & 0x1F) << 7) | OPC_LUI; }
static inline uint32_t enc_slli(int rd, int rs1, int sh) {
    return (((uint32_t)sh & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | (1u << 12) | ((rd & 0x1F) << 7) | OPC_OPIMM; }
static inline uint32_t enc_srli(int rd, int rs1, int sh) {
    return (((uint32_t)sh & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | (5u << 12) | ((rd & 0x1F) << 7) | OPC_OPIMM; }
static inline uint32_t enc_add(int rd, int rs1, int rs2) {
    return ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | ((rd & 0x1F) << 7) | OPC_OP; }
static inline uint32_t enc_or(int rd, int rs1, int rs2) {
    return ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | (6u << 12) | ((rd & 0x1F) << 7) | OPC_OP; }
static inline uint32_t enc_lw(int rd, int rs1, int imm) {
    return enc_imm12(imm) | ((rs1 & 0x1F) << 15) | (2u << 12) | ((rd & 0x1F) << 7) | OPC_LOAD; }
static inline uint32_t enc_sw(int rs2, int rs1, int imm) {
    uint32_t i = (uint32_t)enc_imm12(imm) >> 20;   /* re-extract the checked low 12 bits */
    return (((i >> 5) & 0x7Fu) << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) |
           (2u << 12) | ((i & 0x1Fu) << 7) | OPC_STORE; }
static inline uint32_t enc_branch(int f3, int rs1, int rs2, int off) {
    uint32_t i = (uint32_t)off;
    return (((i >> 12) & 1u) << 31) | (((i >> 5) & 0x3Fu) << 25) | ((rs2 & 0x1F) << 20) |
           ((rs1 & 0x1F) << 15) | ((f3 & 7) << 12) | (((i >> 1) & 0xFu) << 8) |
           (((i >> 11) & 1u) << 7) | OPC_BRANCH; }
#define BR_BLT 4
#define BR_BGE 5
#define BR_BEQ 0
#define BR_BNE 1
static inline uint32_t enc_jal(int rd, int off) {
    uint32_t i = (uint32_t)off;
    return (((i >> 20) & 1u) << 31) | (((i >> 1) & 0x3FFu) << 21) | (((i >> 11) & 1u) << 20) |
           (((i >> 12) & 0xFFu) << 12) | ((rd & 0x1F) << 7) | OPC_JAL; }

/* ---- Matrix / DMA (custom-0, runq_xnpu.c:500-531) ---- */
static inline uint32_t enc_matrix(int f3, int rs1) { return ((rs1 & 0x1F) << 15) | ((f3 & 7) << 12) | OPC_CUSTOM0; }
static inline uint32_t enc_lma(int rs1) { return enc_matrix(0, rs1); }
static inline uint32_t enc_lmb(int rs1) { return enc_matrix(1, rs1); }
static inline uint32_t enc_matmul(void) { return enc_matrix(2, 0); }
static inline uint32_t enc_sm(int rs1) { return enc_matrix(3, rs1); }
static inline uint32_t enc_matmul_acc(void) { return enc_matrix(4, 0); }
/* MQCFG: rs1 = fp32 activation multiplier bits, rs2 = cfg
 *   cfg[0] enable bf16-A loader, cfg[15:1] group index, cfg[31:16] group size
 *   (0 = do not check group boundaries, which is what full-K MATMUL.ACC needs). */
static inline uint32_t enc_mqcfg(int scale_reg, int cfg_reg) {
    return (1u << 25) | ((cfg_reg & 0x1F) << 20) | ((scale_reg & 0x1F) << 15) | (2u << 12) | OPC_CUSTOM0; }
static inline uint32_t enc_matrix_f7(int f3, int f7, int rs1) {
    return (((uint32_t)f7 & 0x7Fu) << 25) | ((rs1 & 0x1F) << 15) | ((f3 & 7) << 12) | OPC_CUSTOM0; }
static inline uint32_t enc_fp_matmul(void)     { return enc_matrix_f7(2, 2, 0); }
static inline uint32_t enc_fp_matmul_acc(void) { return enc_matrix_f7(4, 1, 0); }
static inline uint32_t enc_fp_sm(int rs1)      { return enc_matrix_f7(3, 1, rs1); }
/* MCFG2: queued matrix config riding the MATMUL slot (f3=2, f7=3; R-type).
 * cfg bits: [0]=int8 engine select (0 until dual-engine builds), [1]=dataflow
 * (0=OS 1=WS), [2]=ws_keep (weights stay resident across MATMULA; B neither
 * re-read nor re-captured). Accepting MCFG2 always invalidates resident
 * weights (clear+flush+wClr) -- reload after every config. Must stay
 * bit-identical with Decode.Encode.mcfg2. */
static inline uint32_t enc_mcfg2(int cfg_reg) {
    return (3u << 25) | ((cfg_reg & 0x1Fu) << 20) | (2u << 12) | OPC_CUSTOM0; }
/* FP.GEMV/GEMVA/GSMV removed 2026-10-18 (GemvUnit reverted): their encodings
 * would silently decode as legacy int8 MATMUL/MATMULA/SM -- do not re-add. */
static inline uint32_t enc_dma(int f3, int rs1, int rs2, int rows) {
    return ((((uint32_t)rows >> 5) & 0x7Fu) << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) |
           ((f3 & 7) << 12) | (((uint32_t)rows & 0x1Fu) << 7) | OPC_CUSTOM0; }
static inline uint32_t enc_dmald(int rs1, int rs2, int rows) { return enc_dma(5, rs1, rs2, rows); }
static inline uint32_t enc_dmast(int rs1, int rs2, int rows) { return enc_dma(6, rs1, rs2, rows); }
static inline uint32_t enc_dmawait(void) { return enc_dma(7, 0, 0, 0); }

/* ---- Vector (custom-1, runq_xnpu.c:534-558) ---- */
static inline uint32_t enc_vec(int f3, int f7, int rd, int rs1, int rs2) {
    return (((uint32_t)f7 & 0x7Fu) << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) |
           ((f3 & 7) << 12) | ((rd & 0x1F) << 7) | OPC_CUSTOM1; }
static inline uint32_t enc_vseta(int rs1)   { return enc_vec(0, 0, 0, rs1, 0); }
static inline uint32_t enc_vsetb(int rs1)   { return enc_vec(0, 1, 0, rs1, 0); }
static inline uint32_t enc_vsetd(int rs1)   { return enc_vec(0, 2, 0, rs1, 0); }
static inline uint32_t enc_vsetlen(int rs1) { return enc_vec(0, 3, 0, rs1, 0); }
static inline uint32_t enc_vsets(int rs1)   { return enc_vec(0, 4, 0, rs1, 0); }
static inline uint32_t enc_vadd(void)   { return enc_vec(1, 0, 0, 0, 0); }
static inline uint32_t enc_vmul(void)   { return enc_vec(1, 2, 0, 0, 0); }
/* VFMA is read-modify-write on dst (dst = srcA*srcB + dst), so its destination
 * must already hold a defined value before the first accumulation. */
static inline uint32_t enc_vfma(void)   { return enc_vec(1, 3, 0, 0, 0); }
static inline uint32_t enc_vsmul(void)  { return enc_vec(1, 4, 0, 0, 0); }
static inline uint32_t enc_vsadd(void)  { return enc_vec(1, 5, 0, 0, 0); }
static inline uint32_t enc_vexp(void)   { return enc_vec(1, 6, 0, 0, 0); }
static inline uint32_t enc_vrecip(void) { return enc_vec(1, 7, 0, 0, 0); }
static inline uint32_t enc_vrsqrt(void) { return enc_vec(1, 8, 0, 0, 0); }
static inline uint32_t enc_vabs(void)   { return enc_vec(1, 9, 0, 0, 0); }
static inline uint32_t enc_vcvt_i8(void)   { return enc_vec(1, 10, 0, 0, 0); }
static inline uint32_t enc_vcvt_bf16(void) { return enc_vec(1, 11, 0, 0, 0); }
static inline uint32_t enc_vmax(int rd) { return enc_vec(2, 1, rd, 0, 0); }
static inline uint32_t enc_vdot(int rd) { return enc_vec(2, 2, rd, 0, 0); }
static inline uint32_t enc_vsum(int rd) { return enc_vec(2, 0, rd, 0, 0); }
static inline uint32_t enc_vsoftexp(int rd) { return enc_vec(3, 0, rd, 0, 0); }

#endif /* Q3_ENC_H */
