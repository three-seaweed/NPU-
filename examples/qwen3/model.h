/* model.h -- q3k4 v3 checkpoint: Config, weights view over mmap, run state.
 * run3 (plan Phase 1).  All shapes derive from Config via XN_* macros. */
#ifndef Q3_MODEL_H
#define Q3_MODEL_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int dim;         /* hidden_size */
    int hidden_dim;  /* intermediate_size */
    int n_layers;
    int n_heads;     /* q heads */
    int n_kv_heads;
    int vocab_size;
    int seq_len;     /* 模型侧上限（CPU KV/rope 容量），可被 -c 压 */
    int head_dim;    /* 显式字段（Qwen3 坑 1） */
    int group_size;  /* Q8_0 GS */
    float rope_theta;
    float rms_eps;
    int shared_classifier;
    int qk_norm;
} Config;

/* 派生形状（禁裸数字的唯一出口） */
#define XN_Q_DIM(c)  ((c)->n_heads * (c)->head_dim)
#define XN_KV_DIM(c) ((c)->n_kv_heads * (c)->head_dim)
#define XN_KV_MUL(c) ((c)->n_heads / (c)->n_kv_heads)

/* Q8_0 张量：[i8 q[out*in]][f32 s[out*(in/GS)]]，w->q[r*in+i] 行主、组沿输入维 */
typedef struct {
    int8_t *q;
    float *s;
} QTensor;

typedef struct {
    /* fp32 段（检查点内自然顺序） */
    float *rms_att_weight;   /* [L][dim] */
    float *rms_ffn_weight;   /* [L][dim] */
    float *q_norm_weight;    /* [L][head_dim] */
    float *k_norm_weight;    /* [L][head_dim] */
    float *rms_final_weight; /* [dim] */
    /* q80 段 */
    QTensor tokens; /* [vocab][dim]，tied 分类头 */
    QTensor *wq;    /* [L] q_dim x dim */
    QTensor *wk;    /* [L] kv_dim x dim */
    QTensor *wv;    /* [L] kv_dim x dim */
    QTensor *wo;    /* [L] dim x q_dim */
    QTensor *w1;    /* [L] hidden x dim */
    QTensor *w2;    /* [L] dim x hidden */
    QTensor *w3;    /* [L] hidden x dim */
} Weights;

typedef struct {
    /* activation buffers (fp32) */
    float *x;    /* resid */
    float *xb;   /* normed / matvec out */
    float *xb2;  /* second out */
    float *hb;   /* ffn gate */
    float *hb2;  /* ffn up */
    float *q;    /* [q_dim] */
    float *k;    /* [kv_dim] */
    float *v;    /* [kv_dim] */
    float *att;  /* [n_heads][seq_len] scores */
    float *logits; /* [vocab] */
    /* quantized activations（与 NPU 数值路径一致：GS 同权重） */
    int8_t *xq_q;
    float *xq_s; /* [dim/GS] */
    int8_t *hq_q;
    float *hq_s; /* [hidden/GS] */
    /* KV cache fp32 */
    float *key_cache;   /* [L][seq_len][kv_dim] */
    float *value_cache; /* [L][seq_len][kv_dim] */
} RunState;

typedef struct {
    Config cfg;
    Weights w;
    RunState s;
    void *map;
    size_t map_size;
} Transformer;

int build_transformer(Transformer *t, const char *path, int ctx_len);
void free_transformer(Transformer *t);

/* forward.c：单 token 前向，写 KV 于 pos，返回 logits */
float *forward(Transformer *t, int token, int pos);
/* forward.c：相位墙钟汇总（仅 XNPU_MT_PROF=1 时输出；退出前调用一次） */
void q3_prof_dump(void);

#endif
