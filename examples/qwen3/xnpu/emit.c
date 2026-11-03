/* emit.c -- Emit 助手 + q_dim 版 layer_map + fe_qknorm。
 * Emit 助手逐行移植自 examples/llama2.c/runq_xnpu.c :561-607；
 * layer_map 骨架移植自 :627-700，q_dim 改动见 emit.h 头注与 plan Phase 3
 * 实施记录第 0 级；fe_qknorm 以 :1043-1071 fe_rmsnorm 为母版克隆。 */
#include "emit.h"
#include "enc.h"
#include "layer.h"   /* DW_NWORDS: the desc reservation must track the enum */

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* XNPU_ENGINES=2: alternate output tiles across both int8 engines (dual
 * bitstream). Cached; anything below 2 means the single-engine emission.
 * Requires a bitstream built with Parameters(int8Engines = 2) and the
 * scoped LMA/LMB shadows. */
int emit_engines(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("XNPU_ENGINES");
        cached = (e && atoi(e) >= 2) ? 2 : 1;
    }
    return cached;
}

/* ---- Emit helpers (runq_xnpu.c:563-607, verbatim) ---- */
void ep(Emit *e, uint32_t insn) {
    if (e->n >= e->cap) { e->bad = 1; return; }
    e->p[e->n++] = insn;
}
/* li rd, v -> addi (small) or lui+addi.  The lui immediate is biased by 0x800 so
 * the addi's sign extension cancels out. */
void eli(Emit *e, int rd, uint32_t v) {
    int32_t sv = (int32_t)v;
    if (sv >= -2048 && sv < 2048) { ep(e, enc_addi(rd, 0, sv)); return; }
    uint32_t hi = (v + 0x800u) >> 12;
    int32_t lo = (int32_t)(v << 20) >> 20;
    ep(e, enc_lui(rd, (int)(hi & 0xFFFFFu)));
    ep(e, enc_addi(rd, rd, lo));
}
/* Reserve a slot for a forward branch, to be patched once the target is known. */
int ehole(Emit *e) { int at = e->n; ep(e, 0); return at; }
static void epatch_b_to(Emit *e, int at, int target, int f3, int rs1, int rs2) {
    int off = (target - at) * 4;
    if (at < 0 || at >= e->cap || off < -4096 || off > 4094 || (off & 1)) {
        e->bad = 1; return;
    }
    e->p[at] = enc_branch(f3, rs1, rs2, off); }
void epatch_b(Emit *e, int at, int f3, int rs1, int rs2) {
    epatch_b_to(e, at, e->n, f3, rs1, rs2); }
void epatch_j(Emit *e, int at) {
    int off = (e->n - at) * 4;
    if (at < 0 || at >= e->cap || off < -1048576 || off > 1048574 || (off & 1)) {
        e->bad = 1; return;
    }
    e->p[at] = enc_jal(0, off); }
/* Backward branch to a recorded label. */
void eb_back(Emit *e, int f3, int rs1, int rs2, int label) {
    int self = e->n;
    int off = (label - self) * 4;
    if (label < 0 || off < -4096 || off > 4094 || (off & 1)) {
        e->bad = 1; return;
    }
    ep(e, enc_branch(f3, rs1, rs2, off)); }

/* ---- layer_map, q_dim-aware (runq_xnpu.c:627-700 + plan §4 改动) ---- */
static uint32_t lm_alloc(uint32_t *off, uint32_t bytes, uint32_t align) {
    *off = (*off + align - 1u) & ~(align - 1u);
    uint32_t at = *off;
    *off += bytes;
    return at;
}

LayerMap layer_map_qwen3(int B, int dim, int hidden, int q_dim, int kv_dim,
                         int head_dim, int Tmax, int gs, int sched) {
    LayerMap m; uint32_t o = 0;
    memset(&m, 0, sizeof m);

    /* The descriptor MUST be the first allocation: every desc read in the
     * layer program is `lw rd, x0, desc+field*4` -- a 12-bit signed immediate
     * (max 2047). A map that pushes desc past 2048 silently truncates the
     * immediate and every DMA source address reads back zero (dual build,
     * 2026-10-25: desc at 4096 -> weight loads from DDR 0x0/0x200/...).
     * And it must be a REAL reservation: desc used to ride on the memset-0
     * while the ci tiles owned [0, 4096) -- every ot=0 SM then stomped the
     * descriptor mid-kick and later desc lws returned fp32 result words
     * (garbage x1 -> unaligned dmald -> DMA wedge; dual replay, 2026-10-26).
     * ci follows, still 4096-aligned so the dual/sched slot toggles (x13
     * bit11 = engine pair 2048 B, bit10 = sched 1024 B) are untouched; the
     * cost is the ~3.9 KiB pad between desc and ci. */
    m.desc = lm_alloc(&o, DW_NWORDS * 4u, 4);
    m.ci   = lm_alloc(&o, (emit_engines() > 1 ? 4u : (sched ? 2u : 1u)) * RES_TILE_BYTES, 4096);
    m.ci2  = m.ci + RES_TILE_BYTES;
    uint32_t dimb = (uint32_t)dim * 2, hidb = (uint32_t)hidden * 2, kvb = (uint32_t)kv_dim * 2;
    uint32_t qb = (uint32_t)q_dim * 2;                    /* bf16 bytes per q row */
    uint32_t bdim = (uint32_t)B * dimb, bhid = (uint32_t)B * hidb, bkv = (uint32_t)B * kvb;
    uint32_t bq = (uint32_t)B * qb;

    /* A zero vector sized for the widest extent zeroed in the layer: an
     * attention head (head_dim), one Linear output row (dim), or a wq row
     * (q_dim -- runq sized this dim-only because q_dim==dim there).  Reading
     * past the region would bleed into sc, so size for max(dim, q_dim). */
    m.zero = lm_alloc(&o, (dim > q_dim ? dimb : qb), DTCM_ROW);
    m.sc   = lm_alloc(&o, 4 * DTCM_ROW, DTCM_ROW);        /* amax / 1-amax / inv / s_x */
    /* Group scales, [ot][g][c] bf16: sized for the widest linear in the layer.
     * runq's max was over the {dim,hidden} pair; Qwen3 adds the q_dim pair
     * (wq out=q_dim in=dim; o_proj out=dim in=q_dim).  kv pairs are strictly
     * covered by the dim pair (kv_dim <= dim when n_kv <= n_heads).  With
     * GS=64 the hidden pair (3072) still dominates (q pair 2048) -- the terms
     * must be present anyway because GS is a runtime knob. */
    {
        uint32_t g = (uint32_t)(gs > 0 ? gs : NTILE);
        uint32_t pairs[4] = {
            (uint32_t)(dim / NTILE) * (uint32_t)(hidden / g),
            (uint32_t)(hidden / NTILE) * (uint32_t)(dim / g),
            (uint32_t)(q_dim / NTILE) * (uint32_t)(dim / g),
            (uint32_t)(dim / NTILE) * (uint32_t)(q_dim / g),
        };
        uint32_t mx = pairs[0];
        for (int i = 1; i < 4; i++) if (pairs[i] > mx) mx = pairs[i];
        m.sw = lm_alloc(&o, mx * NTILE * 2u + DTCM_ROW, DTCM_ROW);
        /* fp32 s_x[g][token] for the widest group count. */
        uint32_t ng = 0;
        { uint32_t cand[3] = { (uint32_t)hidden, (uint32_t)dim, (uint32_t)q_dim };
          for (int i = 0; i < 3; i++) if (cand[i] > ng) ng = cand[i];
          ng /= g; }
        m.gscale = lm_alloc(&o, (uint32_t)B * ng * 4u + DTCM_ROW, DTCM_ROW);
    }
    m.rmsw = lm_alloc(&o, dimb, DTCM_ROW);                /* attention/ffn/final norm gain (dim) */
    m.qkg  = lm_alloc(&o, 2u * (uint32_t)head_dim * 2u, DTCM_ROW);  /* qk gains: 2*head_dim bf16 */
    m.cosq = lm_alloc(&o, bq, DTCM_ROW);
    m.sinq = lm_alloc(&o, bq, DTCM_ROW);
    m.cosk = lm_alloc(&o, bkv, DTCM_ROW);
    m.sink = lm_alloc(&o, bkv, DTCM_ROW);
    m.x    = lm_alloc(&o, bdim, DTCM_ROW);                /* residual, persists across kicks */
    /* s1/s2 carry q-extent for the rope/norm stages (Qwen3 q_dim > dim), so
     * they are q-sized now -- runq's "dim-extent only" comment no longer holds. */
    m.q    = lm_alloc(&o, bq, DTCM_ROW);
    m.atto = lm_alloc(&o, bq, DTCM_ROW);
    m.s1   = lm_alloc(&o, bq, DTCM_ROW);                  /* norm output / rope swap scratch */
    m.s2   = lm_alloc(&o, bq, DTCM_ROW);                  /* rope product / group-amax scratch */
    /* Tile-major int8 activation (VSMUL+VCVT.I8 in-kick); widest extent now
     * includes q_dim (o_proj input). */
    {
        uint32_t cand[3] = { (uint32_t)hidden, (uint32_t)dim, (uint32_t)q_dim };
        uint32_t mx = cand[0];
        for (int i = 1; i < 3; i++) if (cand[i] > mx) mx = cand[i];
        m.aq = lm_alloc(&o, mx / NTILE * TILE_BYTES, DTCM_ROW);
    }
    /* XNPU_SCHED: contiguous C-scratch pair; sched=0 keeps the historical size. */
    m.cb   = lm_alloc(&o, (emit_engines() > 1 ? 4u : (sched ? 2u : 1u)) * BF16_TILE_BYTES, DTCM_ROW);
    m.cb2  = m.cb + BF16_TILE_BYTES;
    m.tmph = lm_alloc(&o, dimb, DTCM_ROW);
    m.att  = lm_alloc(&o, (uint32_t)Tmax * 2 + DTCM_ROW, DTCM_ROW);
    m.attf = lm_alloc(&o, (uint32_t)Tmax * 4 + DTCM_ROW, DTCM_ROW);
    m.h1   = lm_alloc(&o, bhid, DTCM_ROW);
    m.h2   = lm_alloc(&o, bhid, DTCM_ROW);
    m.h3   = lm_alloc(&o, bhid, DTCM_ROW);
    m.kvh  = lm_alloc(&o, (uint32_t)Tmax * kvb, DTCM_ROW);
    m.vvh  = lm_alloc(&o, (uint32_t)Tmax * kvb, DTCM_ROW);
    m.wb   = lm_alloc(&o, 2u * (uint32_t)(gs / NTILE) * TILE_BYTES, DTCM_ROW);
    m.total = o;
    m.ok = (o <= DTCM_BYTES);
    return m;
}

/* ---- fe_qknorm: fe_rmsnorm (runq_xnpu.c:1043-1071) 的逐头克隆 ----
 * 差异只有三处：r4/r9/r16 用 head_dim；r15 的迭代数是 B*n_heads；eps 参数化。
 * 增益指针 r12 在循环外固定——Qwen3 的 q/k_norm 增益是 (head_dim,) 对所有头
 * 广播，与 fe_rmsnorm 中 dim 增益对所有 token 复用同构。 */
void fe_qknorm(Emit *e, const LayerMap *m, uint32_t src, uint32_t dst,
               uint32_t gain, int B, int n_heads, int head_dim, float eps) {
    eli(e, 4, (uint32_t)head_dim);
    ep(e, enc_addi(8, 0, 1));
    eli(e, 9, (uint32_t)head_dim * 2);
    eli(e, 11, src);
    eli(e, 13, dst);
    eli(e, 12, gain);
    eli(e, 14, m->zero);
    eli(e, 18, m->sc);
    eli(e, 15, (uint32_t)B * (uint32_t)n_heads);
    eli(e, 16, f2bits(1.0f / (float)head_dim));
    eli(e, 17, f2bits(eps));
    ep(e, enc_addi(10, 0, 0));
    int L = e->n;
    ep(e, enc_vseta(11)); ep(e, enc_vsetb(11)); ep(e, enc_vsetlen(4)); ep(e, enc_vdot(5));
    /* scr = 0 + sum   (materialise the fp32 GPR into a 1-lane vector) */
    ep(e, enc_vsets(5)); ep(e, enc_vseta(14)); ep(e, enc_vsetd(18)); ep(e, enc_vsetlen(8));
    ep(e, enc_vsadd());
    ep(e, enc_vsets(16)); ep(e, enc_vseta(18)); ep(e, enc_vsetd(18)); ep(e, enc_vsmul());
    ep(e, enc_vsets(17)); ep(e, enc_vsadd());
    ep(e, enc_vrsqrt());
    ep(e, enc_vsum(6));
    ep(e, enc_vsets(6)); ep(e, enc_vseta(11)); ep(e, enc_vsetd(13)); ep(e, enc_vsetlen(4));
    ep(e, enc_vsmul());
    ep(e, enc_vseta(13)); ep(e, enc_vsetb(12)); ep(e, enc_vsetd(13)); ep(e, enc_vmul());
    ep(e, enc_add(11, 11, 9)); ep(e, enc_add(13, 13, 9)); ep(e, enc_addi(10, 10, 1));
    eb_back(e, BR_BLT, 10, 15, L);
}

/* ---- fe_rope_half: runq fe_rope (:1315-1350) 的 half-split 变体 ----
 * Qwen3 布局（theta=1e6，j 配 j+head_dim/2，非 llama2 interleave）：
 *   new[j]        = v[j]*cos_j - v[j+64]*sin_j
 *   new[j+64]     = v[j+64]*cos_j + v[j]*sin_j
 * 同样分解为  v*cos + swap(v)*sin_signed：swap 是头内 j<->j+64 交换，
 * sin_signed 前半 -sin、后半 +sin，cos 表前后两半相同（角度只依赖 j mod 64）。
 * 宿主按此生成 cos/sin 表（复制表：每 token 每头 128 项 ×B）。
 *
 * swap() 实现为标量 32-bit 字交换循环：word w（元素 2w,2w+1）与 word
 * w+head_dim/4（元素 2w+64,2w+65）整体互换——无移位掩码需求，比母版的
 * 相邻交换多一倍 LW/SW（每 2 元素 4 次访问 vs 2 次），向量部分三条指令
 * 与母版逐字相同。 */
void fe_rope_half(Emit *e, const LayerMap *m, uint32_t vec, uint32_t cos_t,
                  uint32_t sin_t, int B, int L, int head_dim) {
    int heads = B * L / head_dim;
    uint32_t half_b = (uint32_t)head_dim;      /* bytes per half head (128) */
    int wpr = head_dim / 4;                    /* 32-bit words per half head */
    /* ---- swap loop: s1 = swap(v)，每头内 word w <-> w+head_dim/4 ---- */
    eli(e, 1, vec);
    eli(e, 2, m->s1);
    eli(e, 3, (uint32_t)heads);
    ep(e, enc_addi(4, 0, 0));                  /* head counter */
    int Lh = e->n;
    eli(e, 9, (uint32_t)wpr);
    ep(e, enc_addi(5, 1, (int)half_b));        /* v high-half cursor */
    ep(e, enc_addi(6, 2, (int)half_b));        /* s1 high-half cursor */
    ep(e, enc_addi(7, 0, 0));                  /* word counter */
    int Lw = e->n;
    ep(e, enc_lw(20, 1, 0));
    ep(e, enc_lw(21, 5, 0));
    ep(e, enc_sw(21, 2, 0));
    ep(e, enc_sw(20, 6, 0));
    ep(e, enc_addi(1, 1, 4)); ep(e, enc_addi(2, 2, 4));
    ep(e, enc_addi(5, 5, 4)); ep(e, enc_addi(6, 6, 4));
    ep(e, enc_addi(7, 7, 1));
    eb_back(e, BR_BLT, 7, 9, Lw);
    ep(e, enc_addi(1, 1, (int)half_b));        /* skip to next head's low half */
    ep(e, enc_addi(2, 2, (int)half_b));
    ep(e, enc_addi(4, 4, 1));
    eb_back(e, BR_BLT, 4, 3, Lh);
    /* ---- v = v*cos + s1*sin_signed（与母版 :1340-1349 相同）---- */
    eli(e, 10, m->s1);
    eli(e, 11, sin_t);
    eli(e, 12, m->s2);
    eli(e, 13, (uint32_t)(B * L));
    ep(e, enc_vseta(10)); ep(e, enc_vsetb(11)); ep(e, enc_vsetd(12)); ep(e, enc_vsetlen(13));
    ep(e, enc_vmul());
    eli(e, 14, vec);
    eli(e, 15, cos_t);
    ep(e, enc_vseta(14)); ep(e, enc_vsetb(15)); ep(e, enc_vsetd(14)); ep(e, enc_vsetlen(13));
    ep(e, enc_vmul());
    ep(e, enc_vseta(14)); ep(e, enc_vsetb(12)); ep(e, enc_vsetd(14)); ep(e, enc_vadd());
}

/* ---- 以下移植自 runq_xnpu.c，见 emit.h 尾注的三处差异 ---- */

/* RMSNorm over B token rows (runq_xnpu.c:1043-1071)，eps 参数化。 */
void fe_rmsnorm_eps(Emit *e, const LayerMap *m, uint32_t src, uint32_t dst,
                    int B, int dim, float eps) {
    eli(e, 4, (uint32_t)dim);
    ep(e, enc_addi(8, 0, 1));
    eli(e, 9, (uint32_t)dim * 2);
    eli(e, 11, src);
    eli(e, 13, dst);
    eli(e, 12, m->rmsw);
    eli(e, 14, m->zero);
    eli(e, 18, m->sc);
    eli(e, 15, (uint32_t)B);
    eli(e, 16, f2bits(1.0f / (float)dim));
    eli(e, 17, f2bits(eps));
    ep(e, enc_addi(10, 0, 0));
    int L = e->n;
    ep(e, enc_vseta(11)); ep(e, enc_vsetb(11)); ep(e, enc_vsetlen(4)); ep(e, enc_vdot(5));
    /* scr = 0 + sum   (materialise the fp32 GPR into a 1-lane vector) */
    ep(e, enc_vsets(5)); ep(e, enc_vseta(14)); ep(e, enc_vsetd(18)); ep(e, enc_vsetlen(8));
    ep(e, enc_vsadd());
    ep(e, enc_vsets(16)); ep(e, enc_vseta(18)); ep(e, enc_vsetd(18)); ep(e, enc_vsmul());
    ep(e, enc_vsets(17)); ep(e, enc_vsadd());
    ep(e, enc_vrsqrt());
    ep(e, enc_vsum(6));
    ep(e, enc_vsets(6)); ep(e, enc_vseta(11)); ep(e, enc_vsetd(13)); ep(e, enc_vsetlen(4));
    ep(e, enc_vsmul());
    ep(e, enc_vseta(13)); ep(e, enc_vsetb(12)); ep(e, enc_vsetd(13)); ep(e, enc_vmul());
    ep(e, enc_add(11, 11, 9)); ep(e, enc_add(13, 13, 9)); ep(e, enc_addi(10, 10, 1));
    eb_back(e, BR_BLT, 10, 15, L);
}

/* Stream `rows` 16 B rows from a descriptor-supplied DDR address into DTCM
 * (runq_xnpu.c:1077-1114 verbatim)。A single core DMA descriptor is capped at
 * 2047 rows by the 12-bit S-type immediate, so a long transfer is emitted as a
 * fixed chain of descriptors. */
void fe_dma_in(Emit *e, const LayerMap *m, int desc_word, uint32_t dtcm, uint32_t rows) {
    if (!rows) return;
    ep(e, enc_lw(1, 0, (int)m->desc + desc_word * 4));
    eli(e, 2, dtcm);
    while (rows) {
        uint32_t r = rows > 2047u ? 2047u : rows;
        ep(e, enc_dmald(1, 2, (int)r));
        /* The current RTL owns one DMA range token per accepted descriptor.
         * Retire it before issuing another descriptor; otherwise an immediately
         * following command can replace the token id on the dma-idle boundary. */
        ep(e, enc_dmawait());
        rows -= r;
        if (rows) {                       /* advance both ends for the next chunk */
            eli(e, 3, r * DTCM_ROW);
            ep(e, enc_add(1, 1, 3));
            ep(e, enc_add(2, 2, 3));
        }
    }
}

void fe_dma_out(Emit *e, const LayerMap *m, int desc_word, uint32_t dtcm, uint32_t rows) {
    if (!rows) return;
    ep(e, enc_lw(1, 0, (int)m->desc + desc_word * 4));
    eli(e, 2, dtcm);
    while (rows) {
        uint32_t r = rows > 2047u ? 2047u : rows;
        ep(e, enc_dmast(1, 2, (int)r));
        ep(e, enc_dmawait());
        rows -= r;
        if (rows) {
            eli(e, 3, r * DTCM_ROW);
            ep(e, enc_add(1, 1, 3));
            ep(e, enc_add(2, 2, 3));
        }
    }
}

/* y[B][out] = x[B][in] * W^T, W streamed from DDR as (ot,kt)-ordered int8 tiles
 * (runq_xnpu.c:1119-1313 verbatim，GS->gs 参数)。`srctok` is token-major bf16;
 * the program quantizes and packs it into the tile-major int8 A operand at
 * m->aq in-kick, then one full-B GEMM per (output tile, group) with per-row
 * activation scales and per-output-channel weight scales. */
/* One group's k-tile chain: LMA/LMB + MATMUL first tile then MATMULA per
 * further k-tile, walking the A/B tile pointers r20/r14 (ktpg in r26).
 * Clobbers r20/r14/r22/r23. */
static void emit_gemm_chain(Emit *e) {
    ep(e, enc_addi(23, 0, 0));                        /* kt = 0 */
    int Lkt = e->n;
    ep(e, enc_lma(20)); ep(e, enc_lmb(14));
    int bacc = ehole(e);
    ep(e, enc_matmul());
    int jaf = ehole(e);
    int Ldo = e->n; ep(e, enc_matmul_acc());
    epatch_b_to(e, bacc, Ldo, 1, 23, 0);
    epatch_j(e, jaf);
    ep(e, enc_addi(20, 20, TILE_BYTES));
    ep(e, enc_addi(14, 14, TILE_BYTES));
    ep(e, enc_addi(23, 23, 1));
    eb_back(e, BR_BLT, 23, 26, Lkt);
}

/* GEMM group epilogue for the ci slot x13 currently points at: spill was
 * already issued; convert the group's int32 partial to bf16 (VCVT) and fold
 * the per-row activation scale (r17 walk) and the per-ot weight-scale slice
 * (r16) into the output rows.  Clobbers r14/r20/r22. */
static void emit_gemm_epilogue(Emit *e, const LayerMap *m, int eng, int sched) {
    if (eng > 1) ep(e, enc_add(14, 18, 0)); else eli(e, 14, m->cb);
    if (eng > 1) {
        /* Engine pair rides x13's bit11, extracted by position: 0x800 is not
         * a representable positive andi mask (sign-extends to ~0x7FF, dragging
         * the ci base bits in -- cb rows landed in tmph/h2 and trashed the
         * map; dual replay, 2026-10-26). Base-agnostic form: ((x13>>11)&1)<<10. */
        ep(e, enc_srli(22, 13, 11));
        ep(e, enc_andi(22, 22, 1));
        ep(e, enc_slli(22, 22, 10));
        ep(e, enc_add(14, 14, 22));                   /* cb + engine*1024 */
    }
    if (sched) {
        /* cb pair select rides the ci toggle's bit10 (x13 holds m->ci or
         * m->ci + RES_TILE_BYTES): 0 or 512 bytes onto the cb base. */
        ep(e, enc_andi(22, 13, RES_TILE_BYTES));
        ep(e, enc_srli(22, 22, 1));
        ep(e, enc_add(14, 14, 22));
    }
    ep(e, enc_vseta(13)); ep(e, enc_vsetd(14)); ep(e, enc_vsetlen(2)); ep(e, enc_vcvt_bf16());
    ep(e, enc_addi(10, 0, 0));                        /* b = 0 */
    ep(e, enc_addi(5, 12, 0));                        /* output row */
    int Lb2 = e->n;
    ep(e, enc_lw(20, 17, 0));
    ep(e, enc_vsets(20)); ep(e, enc_vseta(14)); ep(e, enc_vsetd(14)); ep(e, enc_vsetlen(4));
    ep(e, enc_vsmul());
    ep(e, enc_vseta(14)); ep(e, enc_vsetb(16)); ep(e, enc_vsetd(5)); ep(e, enc_vfma());
    ep(e, enc_addi(17, 17, 4));
    ep(e, enc_addi(14, 14, NTILE * 2));
    ep(e, enc_add(5, 5, 31));
    ep(e, enc_addi(10, 10, 1));
    eb_back(e, BR_BLT, 10, 6, Lb2);
}

void fe_linear(Emit *e, const LayerMap *m, uint32_t srctok, uint32_t dst,
               int B, int in, int out, int w_word, int sw_word, int gs, int sched) {
    int nrt = out / NTILE;             /* output tiles */
    int ngrp = in / gs;                /* quantization groups along K */
    int ktpg = gs / NTILE;             /* K-tiles inside one group */
    uint32_t swrows = ((uint32_t)nrt * ngrp * NTILE * 2u) / DTCM_ROW;
    uint32_t group_wbytes = (uint32_t)ktpg * TILE_BYTES;

    /* The checkpoint's group scales for this weight (bf16, [ot][g][c] order). */
    fe_dma_in(e, m, sw_word, m->sw, swrows);

    /* ---- per-token/group scale + int8 A tile ----------------------------- */
    eli(e, 3, srctok);                                /* token 0, group 0 */
    eli(e, 18, m->gscale);                            /* fp32 s_x[g][b] */
    eli(e, 28, m->aq);                                /* int8 group 0, token row 0 */
    eli(e, 25, (uint32_t)ngrp);
    eli(e, 24, (uint32_t)B);
    eli(e, 11, (uint32_t)gs);
    ep(e, enc_addi(4, 0, NTILE));                     /* one K tile row */
    eli(e, 16, (uint32_t)ktpg);
    eli(e, 6, f2bits(127.0f));
    eli(e, 7, f2bits(1.0f / 127.0f));
    eli(e, 26, 0x7fffu);                              /* fp32->bf16 RNE bias */
    eli(e, 31, (uint32_t)in * 2u);                    /* token row stride */
    ep(e, enc_addi(21, 0, 0));                        /* g = 0 */
    int Lgs = e->n;
    ep(e, enc_addi(20, 3, 0));                        /* token source */
    ep(e, enc_addi(27, 28, 0));                       /* int8 dst row in tile 0 */
    ep(e, enc_addi(10, 0, 0));                        /* b = 0 */
    int Lbs = e->n;
    eli(e, 2, m->s2);
    ep(e, enc_vseta(20)); ep(e, enc_vsetd(2)); ep(e, enc_vsetlen(11)); ep(e, enc_vabs());
    ep(e, enc_vseta(2)); ep(e, enc_vsetlen(11)); ep(e, enc_vmax(30));
    /* amax==0 has a defined all-zero quantization path. Do not feed 0 into
     * VRECIP and rely on later NaN handling. */
    int nonzero = ehole(e);                           /* bne amax, x0, nonzero */
    ep(e, enc_addi(22, 0, 0));                        /* quant multiplier = 0 */
    ep(e, enc_sw(0, 18, 0));
    int quantize = ehole(e);                          /* jump over nonzero path */
    int Lnonzero = e->n;
    epatch_b_to(e, nonzero, Lnonzero, 1, 30, 0);
    /* Convert the fp32 reduction to bf16 with round-to-nearest-even. */
    ep(e, enc_srli(22, 30, 16));
    ep(e, enc_andi(22, 22, 1));
    ep(e, enc_add(22, 22, 26));
    ep(e, enc_add(22, 30, 22));
    ep(e, enc_srli(22, 22, 16));
    eli(e, 29, m->sc);
    ep(e, enc_sw(22, 29, 0));
    ep(e, enc_addi(23, 0, 1));                        /* 1-lane vectors below */
    eli(e, 12, m->sc + 16);
    ep(e, enc_vseta(29)); ep(e, enc_vsetlen(23)); ep(e, enc_vsetd(12)); ep(e, enc_vrecip());
    eli(e, 13, m->sc + 32);
    ep(e, enc_vsets(6)); ep(e, enc_vseta(12)); ep(e, enc_vsetd(13)); ep(e, enc_vsmul());
    eli(e, 14, m->sc + 48);
    ep(e, enc_vsets(7)); ep(e, enc_vseta(29)); ep(e, enc_vsetd(14)); ep(e, enc_vsmul());
    ep(e, enc_lw(22, 13, 0)); ep(e, enc_slli(22, 22, 16));
    ep(e, enc_lw(15, 14, 0)); ep(e, enc_slli(15, 15, 16)); ep(e, enc_sw(15, 18, 0));
    epatch_j(e, quantize);

    /* Quantize this token's gs elements a 16-element tile at a time. Source is
     * token-major bf16; destination rows form the tile-major int8 Matrix A. */
    ep(e, enc_addi(8, 20, 0));
    ep(e, enc_addi(9, 27, 0));
    ep(e, enc_addi(19, 0, 0));
    ep(e, enc_vsets(22)); ep(e, enc_vsetlen(4));
    int Lq = e->n;
    eli(e, 2, m->s2);
    ep(e, enc_vseta(8)); ep(e, enc_vsetd(2)); ep(e, enc_vsmul());
    ep(e, enc_vseta(2)); ep(e, enc_vsetd(9)); ep(e, enc_vcvt_i8());
    ep(e, enc_addi(8, 8, NTILE * 2));
    ep(e, enc_addi(9, 9, TILE_BYTES));
    ep(e, enc_addi(19, 19, 1));
    eb_back(e, BR_BLT, 19, 16, Lq);

    ep(e, enc_add(20, 20, 31));                       /* next token, same group */
    ep(e, enc_addi(27, 27, NTILE));
    ep(e, enc_addi(18, 18, 4));
    ep(e, enc_addi(10, 10, 1));
    eb_back(e, BR_BLT, 10, 24, Lbs);
    ep(e, enc_addi(3, 3, gs * 2));                    /* token 0, next group */
    eli(e, 22, group_wbytes);
    ep(e, enc_add(28, 28, 22));
    ep(e, enc_addi(21, 21, 1));
    eb_back(e, BR_BLT, 21, 25, Lgs);

    /* ---- GEMM: one full-B int32 partial per (output tile, group) ---------- */
    ep(e, enc_lw(1, 0, (int)m->desc + w_word * 4));   /* weight tile stream */
    eli(e, 13, m->ci);
    eli(e, 14, m->cb);
    int eng = emit_engines();
    if (eng > 1) {
        /* Dual: r2/r18 hold the ci/cb bases (the quantize constants in
         * r6/r7 are dead, but r6 is the Lz row bound -- r2/r18 are the
         * dead ones here). r18 doubles as the cb base for the per-group
         * engine offset below; r2 re-bases the ci cursor per ot. */
        eli(e, 19, m->ci);                    /* r19: dead in GEMM (quantize counter) */
        eli(e, 18, m->cb);
    }
    eli(e, 15, m->sw);
    eli(e, 27, group_wbytes);
    eli(e, 28, (uint32_t)nrt);
    eli(e, 26, (uint32_t)ktpg);
    eli(e, 25, (uint32_t)ngrp);
    eli(e, 31, (uint32_t)out * 2);                    /* dst row stride */
    eli(e, 30, (uint32_t)ngrp * NTILE * 2u);          /* scale bytes per output tile */
    ep(e, enc_addi(4, 0, NTILE));
    eli(e, 6, (uint32_t)B);
    eli(e, 12, dst);
    eli(e, 2, (uint32_t)(B * NTILE));                 /* valid C elements */
    ep(e, enc_addi(11, 0, 0));                        /* int8 A, no MQCFG loader */
    ep(e, enc_mqcfg(0, 11));
    ep(e, enc_addi(24, 0, 0));                        /* ot = 0 */
    int Lot = e->n;
    /* VFMA reads old dst, so initialise from the dedicated zero vector rather
     * than multiplying uninitialised data (which is unsafe for NaNs). */
    eli(e, 11, f2bits(0.0f));
    eli(e, 22, m->zero);
    ep(e, enc_vsets(11)); ep(e, enc_vseta(22)); ep(e, enc_vsetlen(4));
    ep(e, enc_addi(7, 0, 0)); ep(e, enc_addi(5, 12, 0));
    int Lz = e->n;
    ep(e, enc_vsetd(5)); ep(e, enc_vsadd());
    ep(e, enc_add(5, 5, 31)); ep(e, enc_addi(7, 7, 1));
    eb_back(e, BR_BLT, 7, 6, Lz);

    ep(e, enc_addi(21, 0, 0));                        /* g = 0 */
    if (eng > 1) {
        /* Engine = (g+ot)&1, recomputed per group at the loop top: with
         * ot-level alternation one engine ran a whole ot's groups before the
         * other was ever fed -- zero overlap, board A/B cycles bit-identical.
         * Per-group alternation keeps both engines' chains queued together;
         * the per-engine LMA/LMB routing lets each base load fire on its own
         * engine's idle.  r7 carries the C-scratch pair toggle (folded into
         * x13 at each group top; a direct xori of x13 cannot also carry the
         * engine bit: 0x800 is not a legal I-type immediate). */
        ep(e, enc_addi(7, 0, 0));                     /* scratch pair toggle */
    }
    eli(e, 17, m->gscale);                            /* s_x[g][b] */
    ep(e, enc_addi(16, 15, 0));                       /* scale slice for (ot, 0) */
    eli(e, 3, m->aq);                                 /* A tiles, walked in order */
    eli(e, 29, group_wbytes);
    eli(e, 8, m->wb);
    eli(e, 9, m->wb + group_wbytes);
    ep(e, enc_dmald(1, 8, ktpg * TILE_ROWS));         /* w0 -> r8 buffer */
    ep(e, enc_add(1, 1, 27));
    /* w1 -> r9 buffer: the loop body's prefetch covers w[g+1] for g >= 1,
     * so chain(1)'s weights must be started here (skipped when ngrp == 1). */
    ep(e, enc_addi(22, 21, 1));
    int pf0 = ehole(e);
    ep(e, enc_dmald(1, 9, ktpg * TILE_ROWS));
    ep(e, enc_add(1, 1, 27));
    epatch_b(e, pf0, BR_BGE, 22, 25);

    ep(e, enc_addi(20, 3, 0));                        /* A group tile 0 */
    ep(e, enc_addi(14, 8, 0));                        /* B group tile 0 */
    if (eng > 1) {
        /* Group 0's engine + ci cursor: parity = (g+ot)&1 (add bit0 is xor
         * bit0; r21 = g = 0 here).  r6 stays B throughout: it is the b-loop
         * bound inside the epilogue and Lz's bound in the next ot.  x13 =
         * ci + toggle (bit10, from r7) + engine pair (bit11, parity<<11). */
        ep(e, enc_add(22, 21, 24));
        ep(e, enc_andi(22, 22, 1));
        ep(e, enc_mcfg2(22));
        ep(e, enc_add(13, 19, 7));
        ep(e, enc_slli(22, 22, 11));
        ep(e, enc_add(13, 13, 22));
    }
    emit_gemm_chain(e);
    ep(e, enc_sm(13));

    /* Software-pipelined group loop: at Lg top x13/r7 still describe group
     * g-1's slots, so epilogue(g-1) issues first and THIS group's chain is
     * enqueued before its own epilogue can throttle the scalar on the vector
     * queue.  That one-deep deferral is what lets the two int8 engines'
     * chains overlap (the deeper root fix was the matrix queue: at depth 4
     * the next group's MCFG2/LMA could not even enqueue before the running
     * chain drained, forcing lockstep -- iqDepth is 12 in dual builds).
     * Bodies run ngrp-1 times (chains 1..ngrp-1 after the prologue chain 0);
     * skipped entirely when ngrp == 1. */
    if (ngrp > 1) {
        int Lg = e->n;
        emit_gemm_epilogue(e, m, eng, sched);         /* epilogue(g-1) */
        ep(e, enc_addi(16, 16, NTILE * 2));           /* next group's scale slice */
        ep(e, enc_dmawait());                         /* w[g] landed */
        /* Swap BEFORE this body's prefetch: r8 then points at the buffer w[g]
         * landed in (old r9) for chain(g), and the prefetch below targets the
         * just-freed buffer for w[g+1].  The DMA captures r9 at issue, so the
         * swap must precede the dmald. */
        ep(e, enc_addi(22, 8, 0)); ep(e, enc_addi(8, 9, 0)); ep(e, enc_addi(9, 22, 0));
        ep(e, enc_addi(22, 21, 2));
        int pf = ehole(e);                            /* last body skips the prefetch */
        ep(e, enc_dmald(1, 9, ktpg * TILE_ROWS));     /* prefetch w[g+1] */
        ep(e, enc_add(1, 1, 27));
        epatch_b(e, pf, BR_BGE, 22, 25);

        if (eng > 1) {
            /* Group g's engine + ci cursor (g = r21+1); the r7 toggle flips
             * here once per group. */
            ep(e, enc_addi(22, 21, 1));
            ep(e, enc_add(22, 22, 24));
            ep(e, enc_andi(22, 22, 1));
            ep(e, enc_mcfg2(22));
            ep(e, enc_xori(7, 7, RES_TILE_BYTES));
            ep(e, enc_add(13, 19, 7));
            ep(e, enc_slli(22, 22, 11));
            ep(e, enc_add(13, 13, 22));
        } else if (sched) {
            /* Single-engine builds toggle x13's pair bit directly. */
            ep(e, enc_xori(13, 13, RES_TILE_BYTES));
        }
        ep(e, enc_add(3, 3, 29));                     /* next activation group: chain(g) reads aq + g*group */
        ep(e, enc_addi(20, 3, 0));                    /* A group tile 0 */
        ep(e, enc_addi(14, 8, 0));                    /* B group tile 0 */
        emit_gemm_chain(e);
        ep(e, enc_sm(13));
        ep(e, enc_addi(21, 21, 1));
        /* Back edge with an extra +1 on the compare: bodies run for entered
         * r21 = 0..ngrp-2.  A plain BLT r21,ngrp would run one body too
         * many -- an extra chain on garbage weights whose SM overwrites the
         * real result. */
        ep(e, enc_addi(22, 21, 1));
        eb_back(e, BR_BLT, 22, 25, Lg);
    }

    emit_gemm_epilogue(e, m, eng, sched);             /* final group's epilogue */
    ep(e, enc_addi(16, 16, NTILE * 2));               /* next group's scale slice */

    ep(e, enc_addi(12, 12, NTILE * 2));               /* next output column block */
    ep(e, enc_add(15, 15, 30));                       /* skip this ot's scale rows */
    ep(e, enc_addi(24, 24, 1));
    eb_back(e, BR_BLT, 24, 28, Lot);
    /* r17 (s_x) 与 r16 (weight scale slice) 不在此重置：s_x 是 [g][b] 连续
     * 布局，走完一组恰好落在下一组首元素；跨 ot 的重置在 Lot 循环体内
     * （eli 17, gscale 与 addi 16, 15, 0——runq :1239-1241）。 */
}

/* Causal multi-head attention entirely on the vector core
 * (runq_xnpu.c:1359-1483 verbatim；q 行步进 dim*2 -> q_dim*2)。
 *   scores  : VDOT(q_head, K[t] head slice)   -- both contiguous, no transpose
 *   softmax : VMAX + VSOFTEXP + VRECIP + VSMUL
 *   @V      : per t, VSMUL(V[t]*att[t]) + VADD into the output head slice
 * Scores are parked as fp32 in m->attf, packed to bf16 for the softmax, then
 * unpacked again so the @V loop can read each weight as an fp32 scalar for
 * VSETS.  Causality is structural: token b only iterates t <= start_pos + b. */
void fe_attention(Emit *e, const LayerMap *m, int B, int n_heads,
                  int head_dim, int kv_mul, int q_dim, int kv_dim,
                  int start_pos) {
    eli(e, 1, (uint32_t)B);
    ep(e, enc_addi(2, 0, 0));            /* b = 0 */
    eli(e, 3, (uint32_t)(start_pos + 1));/* Tb for b = 0 */
    eli(e, 28, (uint32_t)kv_dim * 2);    /* KV row stride */
    eli(e, 29, (uint32_t)head_dim);
    eli(e, 30, f2bits(1.0f / sqrtf((float)head_dim)));
    ep(e, enc_addi(24, 0, 0));           /* token byte offset into q / atto */
    ep(e, enc_addi(25, 0, 0));           /* head byte offset within a token row */
    int Lb = e->n;
    eli(e, 4, (uint32_t)n_heads);
    ep(e, enc_addi(5, 0, 0));            /* h = 0 */
    ep(e, enc_addi(26, 0, 0));           /* kv head byte offset */
    ep(e, enc_addi(27, 0, 0));           /* h mod kv_mul */
    int Lh = e->n;

    /* ---- q/out head pointers: base + b*q_dim*2 + h*head_dim*2 ---- */
    eli(e, 6, m->q);
    eli(e, 7, m->atto);
    /* b*q_dim*2 accumulated in x24, h*head_dim*2 in x25 */
    ep(e, enc_add(6, 6, 24)); ep(e, enc_add(6, 6, 25));
    ep(e, enc_add(7, 7, 24)); ep(e, enc_add(7, 7, 25));

    /* ---- phase 1: raw scores -> attf (fp32) ---- */
    eli(e, 8, m->kvh);
    ep(e, enc_add(8, 8, 26));            /* K row 0 + kv head offset */
    eli(e, 9, m->attf);
    ep(e, enc_addi(10, 0, 0));           /* t = 0 */
    ep(e, enc_vseta(6)); ep(e, enc_vsetlen(29));
    int Lt1 = e->n;
    ep(e, enc_vsetb(8)); ep(e, enc_vdot(11));
    ep(e, enc_sw(11, 9, 0));
    ep(e, enc_add(8, 8, 28));
    ep(e, enc_addi(9, 9, 4));
    ep(e, enc_addi(10, 10, 1));
    eb_back(e, BR_BLT, 10, 3, Lt1);

    /* ---- phase 2: pack fp32 -> bf16, scale, softmax, unpack back to fp32 ---- */
    eli(e, 9, m->attf);
    eli(e, 12, m->att);
    eli(e, 23, 0x7fffu);                 /* fp32->bf16 RNE bias */
    ep(e, enc_addi(10, 0, 0));
    int Lp = e->n;                       /* pack two fp32 lanes into one bf16 word */
    ep(e, enc_lw(13, 9, 0));
    ep(e, enc_srli(21, 13, 16)); ep(e, enc_andi(21, 21, 1));
    ep(e, enc_add(21, 21, 23)); ep(e, enc_add(13, 13, 21)); ep(e, enc_srli(13, 13, 16));
    ep(e, enc_lw(14, 9, 4));
    ep(e, enc_srli(21, 14, 16)); ep(e, enc_andi(21, 21, 1));
    ep(e, enc_add(21, 21, 23)); ep(e, enc_add(14, 14, 21)); ep(e, enc_srli(14, 14, 16));
    ep(e, enc_slli(14, 14, 16));
    ep(e, enc_or(13, 13, 14));
    ep(e, enc_sw(13, 12, 0));
    ep(e, enc_addi(9, 9, 8)); ep(e, enc_addi(12, 12, 4)); ep(e, enc_addi(10, 10, 2));
    eb_back(e, BR_BLT, 10, 3, Lp);

    eli(e, 12, m->att);
    ep(e, enc_vsets(30)); ep(e, enc_vseta(12)); ep(e, enc_vsetd(12)); ep(e, enc_vsetlen(3));
    ep(e, enc_vsmul());                                  /* *= 1/sqrt(head_dim) */
    ep(e, enc_vseta(12)); ep(e, enc_vsetlen(3)); ep(e, enc_vmax(13));
    ep(e, enc_vsets(13)); ep(e, enc_vseta(12)); ep(e, enc_vsetd(12));
    ep(e, enc_vsoftexp(14));                             /* dst = exp(x-max), x14 = sum */
    /* inv = 1/sum through a one-element vector, then normalise */
    eli(e, 15, m->sc);
    eli(e, 16, m->zero);
    ep(e, enc_addi(17, 0, 1));
    ep(e, enc_vsets(14)); ep(e, enc_vseta(16)); ep(e, enc_vsetd(15)); ep(e, enc_vsetlen(17));
    ep(e, enc_vsadd());
    ep(e, enc_vseta(15)); ep(e, enc_vsetd(15)); ep(e, enc_vrecip());
    ep(e, enc_vsum(18));
    ep(e, enc_vsets(18)); ep(e, enc_vseta(12)); ep(e, enc_vsetd(12)); ep(e, enc_vsetlen(3));
    ep(e, enc_vsmul());

    eli(e, 9, m->attf);                                  /* unpack bf16 -> fp32 */
    eli(e, 12, m->att);
    ep(e, enc_addi(10, 0, 0));
    int Lu = e->n;
    ep(e, enc_lw(13, 12, 0));
    ep(e, enc_slli(14, 13, 16));  ep(e, enc_sw(14, 9, 0));
    ep(e, enc_srli(14, 13, 16));  ep(e, enc_slli(14, 14, 16)); ep(e, enc_sw(14, 9, 4));
    ep(e, enc_addi(9, 9, 8)); ep(e, enc_addi(12, 12, 4)); ep(e, enc_addi(10, 10, 2));
    eb_back(e, BR_BLT, 10, 3, Lu);

    /* ---- phase 3: out_head = sum_t att[t] * V[t] head slice ---- */
    eli(e, 19, f2bits(0.0f));
    eli(e, 21, m->zero);
    ep(e, enc_vsets(19)); ep(e, enc_vseta(21)); ep(e, enc_vsetd(7)); ep(e, enc_vsetlen(29));
    ep(e, enc_vsadd());                                  /* zero the accumulator */
    eli(e, 8, m->vvh);
    ep(e, enc_add(8, 8, 26));
    eli(e, 9, m->attf);
    eli(e, 20, m->tmph);
    ep(e, enc_addi(10, 0, 0));
    int Lt2 = e->n;
    ep(e, enc_lw(13, 9, 0));
    ep(e, enc_vsets(13)); ep(e, enc_vseta(8)); ep(e, enc_vsetd(20)); ep(e, enc_vsetlen(29));
    ep(e, enc_vsmul());
    ep(e, enc_vseta(7)); ep(e, enc_vsetb(20)); ep(e, enc_vsetd(7)); ep(e, enc_vadd());
    ep(e, enc_add(8, 8, 28));
    ep(e, enc_addi(9, 9, 4));
    ep(e, enc_addi(10, 10, 1));
    eb_back(e, BR_BLT, 10, 3, Lt2);

    /* next head: advance the q/out head offset, and the KV head offset every
     * kv_mul heads (grouped-query attention shares one KV head across kv_mul Q heads) */
    eli(e, 13, (uint32_t)head_dim * 2);
    ep(e, enc_add(25, 25, 13));
    ep(e, enc_addi(27, 27, 1));
    eli(e, 14, (uint32_t)kv_mul);
    int nk = ehole(e);                                   /* blt hmod, kv_mul -> skip */
    ep(e, enc_addi(27, 0, 0));
    ep(e, enc_add(26, 26, 13));
    epatch_b(e, nk, BR_BLT, 27, 14);
    ep(e, enc_addi(5, 5, 1));
    eb_back(e, BR_BLT, 5, 4, Lh);

    /* next token: reset the head offset, advance the token offset, grow Tb */
    ep(e, enc_addi(25, 0, 0));
    eli(e, 13, (uint32_t)q_dim * 2);
    ep(e, enc_add(24, 24, 13));
    ep(e, enc_addi(3, 3, 1));
    ep(e, enc_addi(2, 2, 1));
    eb_back(e, BR_BLT, 2, 1, Lb);
}

/* hb = silu(gate) * up, computed in place on the gate buffer
 * (runq_xnpu.c:1486-1500 verbatim)。 */
void fe_swiglu(Emit *e, const LayerMap *m, int B, int hidden) {
    eli(e, 1, m->h2);                    /* gate */
    eli(e, 2, m->h1);                    /* up */
    eli(e, 3, m->h3);                    /* temp */
    eli(e, 4, (uint32_t)(B * hidden));
    eli(e, 5, f2bits(-1.0f));
    eli(e, 6, f2bits(1.0f));
    ep(e, enc_vsetlen(4));
    ep(e, enc_vsets(5)); ep(e, enc_vseta(1)); ep(e, enc_vsetd(3)); ep(e, enc_vsmul());
    ep(e, enc_vseta(3)); ep(e, enc_vsetd(3)); ep(e, enc_vexp());
    ep(e, enc_vsets(6)); ep(e, enc_vsadd());
    ep(e, enc_vrecip());
    ep(e, enc_vseta(1)); ep(e, enc_vsetb(3)); ep(e, enc_vsetd(1)); ep(e, enc_vmul());
    ep(e, enc_vseta(1)); ep(e, enc_vsetb(2)); ep(e, enc_vsetd(1)); ep(e, enc_vmul());
}

/* dst += src, elementwise (runq_xnpu.c:1504-1510 verbatim).  dst aliases srcA
 * exactly, which is the only overlap form the vector unit permits. */
void fe_residual(Emit *e, uint32_t dst, uint32_t src, int n) {
    eli(e, 1, dst);
    eli(e, 2, src);
    eli(e, 3, (uint32_t)n);
    ep(e, enc_vseta(1)); ep(e, enc_vsetb(2)); ep(e, enc_vsetd(1)); ep(e, enc_vsetlen(3));
    ep(e, enc_vadd());
}
