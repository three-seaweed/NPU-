/* nn.c -- 数值核实现。
 * rmsnorm/softmax 形状抄自 qwen3.c runq.c:366/:381（eps 改为参数）；
 * 量化/matmul 与 export 端 quantize_q80 及 llama2.c matmul 同式：
 *   scale = maxabs/127, q = round(x/scale), y = Σ int8*int8 * sx*sw 逐组。 */
#include "nn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void rmsnorm_eps(float *o, const float *x, const float *gain, int size, float eps) {
    float ss = 0.f;
    for (int j = 0; j < size; j++)
        ss += x[j] * x[j];
    ss = 1.f / sqrtf(ss / (float)size + eps);
    for (int j = 0; j < size; j++)
        o[j] = x[j] * ss * gain[j];
}

void softmax_inplace(float *x, int size) {
    float mx = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > mx)
            mx = x[i];
    float sum = 0.f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    for (int i = 0; i < size; i++)
        x[i] /= sum;
}

void quantize_gs(int8_t *q, float *s, const float *x, int n, int gs) {
    for (int g = 0; g < n / gs; g++) {
        const float *grp = x + g * gs;
        float amax = 0.f;
        for (int i = 0; i < gs; i++)
            amax = fmaxf(amax, fabsf(grp[i]));
        float scale = amax / 127.f;
        if (scale == 0.f)
            scale = 1.f;
        s[g] = scale;
        for (int i = 0; i < gs; i++)
            q[g * gs + i] = (int8_t)lrintf(grp[i] / scale);
    }
}

/* Q3_FP32_ACT=1: 权重反量化、激活保持 fp32（金标模式，隔离激活量化噪声） */
int g_q3_fp32_act = -1;
static int fp32_act(void) {
    if (g_q3_fp32_act < 0) {
        const char *e = getenv("Q3_FP32_ACT");
        g_q3_fp32_act = (e && e[0] == '1') ? 1 : 0;
    }
    return g_q3_fp32_act;
}
int nn_fp32_act(void) { return fp32_act(); }

/* ---------- 并行分片核（plan §3/§4） ---------- */

static int32_t dot_i8_scalar(const int8_t *w, const int8_t *x, int gs) {
    int32_t ival = 0;
    for (int i = 0; i < gs; i++)
        ival += (int32_t)w[i] * (int32_t)x[i];
    return ival;
}

#if defined(__aarch64__)
#include <arm_neon.h>

static int g_neon = -1;
static int neon_on(void) {
    if (g_neon < 0) {
        const char *e = getenv("XNPU_NEON");
        g_neon = (e && e[0] == '0') ? 0 : 1;
    }
    return g_neon;
}

/* 组内 int8 点积（A53 无 SDOT；plan §4 定稿写法）：
 * vmull_s8 乘积 ≤ 127*127=16129 精确落 int16，vpadalq_s16 加宽累加 + vaddvq_s32
 * 水平归约均为整数精确、次序无关 ⇒ 与标量同 ival（plan §2 定理 2）；
 * 尾段 gs%16 走标量。 */
static int32_t dot_i8_neon(const int8_t *w, const int8_t *x, int gs) {
    int32x4_t acc = vdupq_n_s32(0);
    int k = 0;
    for (; k + 16 <= gs; k += 16) {
        int8x16_t w16 = vld1q_s8(w + k);
        int8x16_t x16 = vld1q_s8(x + k);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(w16), vget_low_s8(x16)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(w16), vget_high_s8(x16)));
    }
    int32_t ival = vaddvq_s32(acc);
    for (; k < gs; k++)
        ival += (int32_t)w[k] * (int32_t)x[k];
    return ival;
}
#endif

void matmul_q8_rows(float *y, const QTensor *w, const int8_t *xq, const float *xs,
                    int r0, int r1, int in, int gs) {
#if defined(__aarch64__)
    const int neon = neon_on();
#endif
    for (int r = r0; r < r1; r++) {
        const int8_t *wr = w->q + (size_t)r * in;
        const float *sr = w->s + (size_t)r * (in / gs);
        float acc = 0.f;
        for (int g = 0; g < in / gs; g++) {
            const int8_t *wg = wr + (size_t)g * gs;
            const int8_t *xg = xq + (size_t)g * gs;
            int32_t ival;
#if defined(__aarch64__)
            ival = neon ? dot_i8_neon(wg, xg, gs) : dot_i8_scalar(wg, xg, gs);
#else
            ival = dot_i8_scalar(wg, xg, gs);
#endif
            acc += (float)ival * sr[g] * xs[g]; /* 表达式树冻结：不许 fmaf/预乘 */
        }
        y[r] = acc;
    }
}

void matmul_q8(float *y, const QTensor *w, const float *x, int out, int in, int gs) {
    if (fp32_act()) {
        for (int r = 0; r < out; r++) {
            const int8_t *wr = w->q + (size_t)r * in;
            const float *sr = w->s + (size_t)r * (in / gs);
            float acc = 0.f;
            for (int g = 0; g < in / gs; g++)
                for (int i = 0; i < gs; i++)
                    acc += (float)wr[g * gs + i] * sr[g] * x[g * gs + i];
            y[r] = acc;
        }
        return;
    }
    /* x 量化一次，供所有输出行复用 */
    int8_t xq[4096];
    float xs[4096 / 1];
    /* 静态上界：in <= 4096（dim 1024, q_dim 2048, hidden 3072 均满足） */
    if (in > 4096) {
        /* 回退堆分配路径（正常模型不会走到） */
        int8_t *dq = malloc(in);
        float *ds = malloc((size_t)(in / gs) * sizeof(float));
        quantize_gs(dq, ds, x, in, gs);
        for (int r = 0; r < out; r++) {
            const int8_t *wr = w->q + (size_t)r * in;
            const float *sr = w->s + (size_t)r * (in / gs);
            float acc = 0.f;
            for (int g = 0; g < in / gs; g++) {
                int32_t ival = 0;
                for (int i = 0; i < gs; i++)
                    ival += (int32_t)wr[g * gs + i] * (int32_t)dq[g * gs + i];
                acc += (float)ival * sr[g] * ds[g];
            }
            y[r] = acc;
        }
        free(dq);
        free(ds);
        return;
    }
    quantize_gs(xq, xs, x, in, gs);
    for (int r = 0; r < out; r++) {
        const int8_t *wr = w->q + (size_t)r * in;
        const float *sr = w->s + (size_t)r * (in / gs);
        float acc = 0.f;
        for (int g = 0; g < in / gs; g++) {
            int32_t ival = 0;
            for (int i = 0; i < gs; i++)
                ival += (int32_t)wr[g * gs + i] * (int32_t)xq[g * gs + i];
            acc += (float)ival * sr[g] * xs[g];
        }
        y[r] = acc;
    }
}
