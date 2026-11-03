/* sampler.h -- temp/top-k/top-p 采样器（自 qwen3.c runq.c:949 移植并紧凑化）。
 * 语义：先 top-k 截断，再在截断表内 top-p，最后按温度分布采样。 */
#ifndef Q3_SAMPLER_H
#define Q3_SAMPLER_H

#include <stdint.h>

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex *probindex; /* [vocab_size] 工作区 */
    float temperature;
    float topp;
    int topk;
    uint64_t rng_state;
} Sampler;

void sampler_init(Sampler *s, int vocab_size, float temperature, float topp, int topk,
                  uint64_t seed);
int sampler_sample(Sampler *s, float *logits); /* 会原地改写 logits */

#endif
