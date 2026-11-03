/* nn.h -- 数值核：rmsnorm(eps)/softmax/量化/q8 matvec。
 * 与 runq_xnpu.c CPU 回退路径同数值：激活沿输入维按 GS 分组量化，int8 x int8 累加。 */
#ifndef Q3_NN_H
#define Q3_NN_H

#include "model.h"

void rmsnorm_eps(float *o, const float *x, const float *gain, int size, float eps);
void softmax_inplace(float *x, int size);

/* y[out] = W(out x in, q8) · x(in, fp32)，内部对 x 按 GS 分组量化 */
void matmul_q8(float *y, const QTensor *w, const float *x, int out, int in, int gs);

/* 并行分片核（cpu_multicore_neon_decode_plan.md §3）：x 由调用方一次量化，
 * 本核只算行区间 [r0,r1)。行内表达式与 matmul_q8 串行路径逐字符一致；
 * aarch64 上组内点积走 NEON（整数段精确 ⇒ 逐位一致，XNPU_NEON=0 关闭）。 */
void matmul_q8_rows(float *y, const QTensor *w, const int8_t *xq, const float *xs,
                    int r0, int r1, int in, int gs);

/* 金标模式（Q3_FP32_ACT）查询：forward 的并行相位据此回退串行核 */
int nn_fp32_act(void);

/* 显式量化（rope/残差前共享一次量化结果时用） */
void quantize_gs(int8_t *q, float *s, const float *x, int n, int gs);

#endif
