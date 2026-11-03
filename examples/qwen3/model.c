/* model.c -- q3k4 v3 checkpoint loader (mmap, no copy).
 * 格式见 conduct_docs/qwen3_0.6b_plan.md 附录 A；布局与 tools/qwen3_export.c 写出端一致。 */
#define _GNU_SOURCE
#include "model.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CKPT_MAGIC 0x346B3371u
#define CKPT_VERSION 3u
#define HDR_SIZE 256

static void die(const char *msg) {
    fprintf(stderr, "run3: %s\n", msg);
    exit(1);
}

int build_transformer(Transformer *t, const char *path, int ctx_len) {
    memset(t, 0, sizeof *t);
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        die("cannot open checkpoint");
    struct stat sb;
    if (fstat(fd, &sb) != 0)
        die("fstat checkpoint");
    t->map_size = (size_t)sb.st_size;
    t->map = mmap(NULL, t->map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (t->map == MAP_FAILED)
        die("mmap checkpoint");
    close(fd);

    const unsigned char *h = t->map;
    uint32_t magic, ver;
    memcpy(&magic, h, 4);
    memcpy(&ver, h + 4, 4);
    if (magic != CKPT_MAGIC || ver != CKPT_VERSION)
        die("not a q3k4 v3 checkpoint");
    Config *c = &t->cfg;
    int32_t v[9];
    memcpy(v, h + 8, sizeof v);
    c->dim = v[0];
    c->hidden_dim = v[1];
    c->n_layers = v[2];
    c->n_heads = v[3];
    c->n_kv_heads = v[4];
    c->vocab_size = v[5];
    c->seq_len = v[6];
    c->head_dim = v[7];
    c->group_size = v[8];
    memcpy(&c->rope_theta, h + 44, 4);
    memcpy(&c->rms_eps, h + 48, 4);
    c->shared_classifier = h[52];
    c->qk_norm = h[53];
    if (!c->shared_classifier)
        die("v3 requires tied embeddings");
    if (!c->qk_norm)
        die("v3 requires qk_norm weights");
    if (ctx_len > 0 && ctx_len < c->seq_len)
        c->seq_len = ctx_len;

    if (c->dim % c->group_size || c->hidden_dim % c->group_size ||
        XN_Q_DIM(c) % c->group_size || XN_KV_DIM(c) % c->group_size)
        die("group_size does not divide dims");

    /* wire pointers over the mmap */
    const unsigned char *p = h + HDR_SIZE;
    size_t L = (size_t)c->n_layers;
    t->w.rms_att_weight = (float *)p;
    p += L * c->dim * 4;
    t->w.rms_ffn_weight = (float *)p;
    p += L * c->dim * 4;
    t->w.q_norm_weight = (float *)p;
    p += L * c->head_dim * 4;
    t->w.k_norm_weight = (float *)p;
    p += L * c->head_dim * 4;
    t->w.rms_final_weight = (float *)p;
    p += (size_t)c->dim * 4;

    size_t nq[7] = {(size_t)XN_Q_DIM(c) * c->dim, (size_t)XN_KV_DIM(c) * c->dim,
                    (size_t)XN_KV_DIM(c) * c->dim, (size_t)c->dim * XN_Q_DIM(c),
                    (size_t)c->hidden_dim * c->dim, (size_t)c->dim * c->hidden_dim,
                    (size_t)c->hidden_dim * c->dim};
    QTensor *slots[8];
    t->w.wq = malloc(L * sizeof(QTensor));
    t->w.wk = malloc(L * sizeof(QTensor));
    t->w.wv = malloc(L * sizeof(QTensor));
    t->w.wo = malloc(L * sizeof(QTensor));
    t->w.w1 = malloc(L * sizeof(QTensor));
    t->w.w2 = malloc(L * sizeof(QTensor));
    t->w.w3 = malloc(L * sizeof(QTensor));
    if (!t->w.wq || !t->w.wk || !t->w.wv || !t->w.wo || !t->w.w1 || !t->w.w2 || !t->w.w3)
        die("oom layer slots");
    slots[0] = &t->w.tokens;
    /* token 表一次，然后每层按 wq wk wv wo w1 w2 w3（写出端同序） */
    t->w.tokens.q = (int8_t *)p;
    p += (size_t)c->vocab_size * c->dim;
    t->w.tokens.s = (float *)p;
    p += (size_t)c->vocab_size * (c->dim / c->group_size) * 4;
    for (size_t l = 0; l < L; l++) {
        slots[1] = &t->w.wq[l];
        slots[2] = &t->w.wk[l];
        slots[3] = &t->w.wv[l];
        slots[4] = &t->w.wo[l];
        slots[5] = &t->w.w1[l];
        slots[6] = &t->w.w2[l];
        slots[7] = &t->w.w3[l];
        for (int k = 1; k < 8; k++) {
            size_t ne = nq[k - 1];
            slots[k]->q = (int8_t *)p;
            p += ne;
            slots[k]->s = (float *)p;
            p += (ne / (size_t)c->group_size) * 4;
        }
    }
    if ((size_t)(p - h) != t->map_size)
        die("checkpoint size mismatch (trailing/short)");

    /* run state */
    RunState *s = &t->s;
    size_t T = (size_t)c->seq_len;
    s->x = calloc(c->dim, sizeof(float));
    s->xb = calloc(c->dim, sizeof(float));
    s->xb2 = calloc((size_t)(c->dim > XN_Q_DIM(c) ? c->dim : XN_Q_DIM(c)), sizeof(float)); /* 容注意力输出 q_dim */
    s->hb = calloc(c->hidden_dim, sizeof(float));
    s->hb2 = calloc(c->hidden_dim, sizeof(float));
    s->q = calloc(XN_Q_DIM(c), sizeof(float));
    s->k = calloc(XN_KV_DIM(c), sizeof(float));
    s->v = calloc(XN_KV_DIM(c), sizeof(float));
    s->att = calloc((size_t)c->n_heads * T, sizeof(float));
    s->logits = calloc(c->vocab_size, sizeof(float));
    s->xq_q = malloc(c->dim);
    s->xq_s = malloc((size_t)(c->dim / c->group_size) * sizeof(float));
    s->hq_q = malloc(c->hidden_dim);
    s->hq_s = malloc((size_t)(c->hidden_dim / c->group_size) * sizeof(float));
    s->key_cache = calloc(L * T * XN_KV_DIM(c), sizeof(float));
    s->value_cache = calloc(L * T * XN_KV_DIM(c), sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v ||
        !s->att || !s->logits || !s->xq_q || !s->xq_s || !s->hq_q || !s->hq_s ||
        !s->key_cache || !s->value_cache)
        die("oom run state");
    return 0;
}

void free_transformer(Transformer *t) {
    free(t->w.wq);
    free(t->w.wk);
    free(t->w.wv);
    free(t->w.wo);
    free(t->w.w1);
    free(t->w.w2);
    free(t->w.w3);
    RunState *s = &t->s;
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->xq_q);
    free(s->xq_s);
    free(s->hq_q);
    free(s->hq_s);
    free(s->key_cache);
    free(s->value_cache);
    if (t->map && t->map != MAP_FAILED)
        munmap(t->map, t->map_size);
    memset(t, 0, sizeof *t);
}
