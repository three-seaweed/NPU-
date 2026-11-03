/* sampler.c -- 见 sampler.h。数值语义与 qwen3.c runq.c 的
 * sample_argmax/sample_mult/sample_topp 一致，改为单次排序的紧凑表实现。 */
#include "sampler.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int cmp_probindex(const void *a, const void *b) {
    float x = ((const ProbIndex *)a)->prob, y = ((const ProbIndex *)b)->prob;
    if (x > y)
        return -1;
    if (x < y)
        return 1;
    return ((const ProbIndex *)a)->index - ((const ProbIndex *)b)->index;
}

void sampler_init(Sampler *s, int vocab_size, float temperature, float topp, int topk,
                  uint64_t seed) {
    s->vocab_size = vocab_size;
    s->probindex = malloc((size_t)vocab_size * sizeof(ProbIndex));
    s->temperature = temperature;
    s->topp = topp;
    s->topk = topk;
    s->rng_state = seed ? seed : 0x853c49e6748fea9bull;
}

static float rng_next_f(Sampler *s) {
    s->rng_state = s->rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)(s->rng_state >> 33) / (float)(1u << 31); /* [0,1) */
}

int sampler_sample(Sampler *s, float *logits) {
    int n = s->vocab_size;
    if (s->temperature <= 0.f) {
        int mi = 0;
        for (int i = 1; i < n; i++)
            if (logits[i] > logits[mi])
                mi = i;
        return mi;
    }
    /* temp + softmax */
    for (int i = 0; i < n; i++)
        logits[i] /= s->temperature;
    float mx = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > mx)
            mx = logits[i];
    float sum = 0.f;
    for (int i = 0; i < n; i++) {
        logits[i] = expf(logits[i] - mx);
        sum += logits[i];
    }
    for (int i = 0; i < n; i++)
        logits[i] /= sum;
    /* 排序取 top-k */
    for (int i = 0; i < n; i++) {
        s->probindex[i].prob = logits[i];
        s->probindex[i].index = i;
    }
    qsort(s->probindex, (size_t)n, sizeof(ProbIndex), cmp_probindex);
    int k = (s->topk > 0 && s->topk < n) ? s->topk : n;
    /* 截断表内 top-p：累积到超 topp 即止（至少保留 1 个） */
    float cum = 0.f;
    int last = k;
    for (int i = 0; i < k; i++) {
        cum += s->probindex[i].prob;
        if (cum > s->topp) {
            last = i + 1;
            break;
        }
    }
    float coin = rng_next_f(s);
    float c = 0.f;
    for (int i = 0; i < last; i++) {
        c += s->probindex[i].prob;
        if (coin * cum < c)
            return s->probindex[i].index;
    }
    return s->probindex[last - 1].index;
}
