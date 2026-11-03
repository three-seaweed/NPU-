/* qwen3_export.c -- pure C exporter for Qwen3-0.6B (run3 Phase 0, plan §5).
 *
 * Modes:
 *   --ckpt  <hfdir> <out.bin> [--seq-len N] [--gs N]   v3 "q3k4" checkpoint (附录 A)
 *   --tok   <hfdir> <out.bin>                          q3t4 tokenizer   (附录 B)
 *   --tiny  <out.bin> [seed]                           deterministic micro Qwen3
 *   --selfcheck <hfdir> <ckpt.bin> [tok.bin]           read back + vs safetensors
 *
 * Formats are byte-exact per conduct_docs/qwen3_0.6b_plan.md 附录 A/B:
 *   ckpt: u32 magic 0x346B3371, version 3, 256B header, fp32 norm segment,
 *         then Q8_0 tensors [i8 q[size]; f32 s[size/GS]] with groups along the
 *         input dim, natural layout (no permute).
 *   tok:  u32 magic 0x346B3374, version 1; vocab stored as bytes_to_unicode-
 *         DECODED raw byte strings; score = -merge_rank, primitives -1e10,
 *         added/gap entries -inf.
 *
 * Numeric sources ported: quantize formula == qwen3.c export.py quantize_q80
 * (:54); GPT-2 byte<->unicode == export.py bytes_to_unicode (:183).
 * safetensors container = 8B LE header length + JSON + raw payload.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CKPT_MAGIC 0x346B3371u /* "q3k4" */
#define TOK_MAGIC  0x346B3374u /* "q3t4" */
#define CKPT_VERSION 3u
#define TOK_VERSION  1u
#define HDR_SIZE   256

static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "qwen3_export: ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}
#define DIE(cond, ...)                                                                             \
    do {                                                                                           \
        if (!(cond)) die(__VA_ARGS__);                                                             \
    } while (0)

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    DIE(p, "out of memory (%zu)", n);
    return p;
}
static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n);
    DIE(q, "out of memory (%zu)", n);
    return q;
}

/* ============================ minimal JSON DOM ============================ */
/* Covers config.json, safetensors headers and tokenizer.json (vocab object,
 * merges array of [a,b] pairs, added_tokens array).  Arena-leaked (one-shot). */

typedef enum { JNULL, JBOOL, JNUM, JSTR, JARR, JOBJ } JType;

typedef struct JV JV;
struct JV {
    JType t;
    double num;   /* JNUM / JBOOL(0|1) */
    char *str;    /* JSTR, decoded UTF-8, NUL-terminated */
    size_t len;   /* JSTR byte length */
    JV **items;   /* JARR/JOBJ values */
    char **keys;  /* JOBJ keys (C strings) */
    size_t n;
};

static char *g_src;
static size_t g_depth;

static void utf8_push(char *buf, size_t *n, uint32_t cp) {
    char b[4];
    int k;
    if (cp < 0x80) {
        b[0] = (char)cp;
        k = 1;
    } else if (cp < 0x800) {
        b[0] = (char)(0xC0 | (cp >> 6));
        b[1] = (char)(0x80 | (cp & 0x3F));
        k = 2;
    } else if (cp < 0x10000) {
        b[0] = (char)(0xE0 | (cp >> 12));
        b[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        b[2] = (char)(0x80 | (cp & 0x3F));
        k = 3;
    } else {
        b[0] = (char)(0xF0 | (cp >> 18));
        b[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        b[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        b[3] = (char)(0x80 | (cp & 0x3F));
        k = 4;
    }
    memcpy(buf + *n, b, (size_t)k);
    *n += (size_t)k;
}

static uint32_t hex4(const char *p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9')
            v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f')
            v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            v |= (uint32_t)(c - 'A' + 10);
        else
            die("json: bad \\u escape");
    }
    return v;
}

static char *json_parse_string(size_t *out_len) {
    DIE(*g_src == '"', "json: expected string");
    g_src++;
    size_t cap = 32, n = 0;
    char *buf = xmalloc(cap);
#define PUSH(c)                                                                                    \
    do {                                                                                           \
        if (n + 4 >= cap) {                                                                        \
            cap *= 2;                                                                              \
            buf = xrealloc(buf, cap);                                                              \
        }                                                                                          \
        buf[n++] = (c);                                                                            \
    } while (0)
    while (*g_src != '"') {
        char c = *g_src++;
        if (c == '\0')
            die("json: unterminated string");
        if (c != '\\') {
            PUSH(c);
            continue;
        }
        char e = *g_src++;
        switch (e) {
        case '"': PUSH('"'); break;
        case '\\': PUSH('\\'); break;
        case '/': PUSH('/'); break;
        case 'b': PUSH('\b'); break;
        case 'f': PUSH('\f'); break;
        case 'n': PUSH('\n'); break;
        case 'r': PUSH('\r'); break;
        case 't': PUSH('\t'); break;
        case 'u': {
            uint32_t cp = hex4(g_src);
            g_src += 4;
            if (cp >= 0xD800 && cp <= 0xDBFF && g_src[0] == '\\' && g_src[1] == 'u') {
                uint32_t lo = hex4(g_src + 2);
                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    g_src += 6;
                }
            }
            if (n + 4 >= cap) {
                while (n + 4 >= cap)
                    cap *= 2;
                buf = xrealloc(buf, cap);
            }
            utf8_push(buf, &n, cp);
            break;
        }
        default: die("json: bad escape \\%c", e);
        }
    }
#undef PUSH
    g_src++;
    buf[n] = '\0';
    *out_len = n;
    return buf;
}

static void json_ws(void) {
    while (*g_src == ' ' || *g_src == '\t' || *g_src == '\n' || *g_src == '\r')
        g_src++;
}

static JV *json_parse(void) {
    json_ws();
    JV *v = xmalloc(sizeof(JV));
    memset(v, 0, sizeof(JV));
    if (*g_src == '"') {
        v->t = JSTR;
        v->str = json_parse_string(&v->len);
    } else if (*g_src == '{') {
        DIE(++g_depth < 256, "json: too deep");
        g_src++;
        v->t = JOBJ;
        size_t cap = 8;
        v->keys = xmalloc(cap * sizeof(char *));
        v->items = xmalloc(cap * sizeof(JV *));
        json_ws();
        if (*g_src == '}') {
            g_src++;
            g_depth--;
            return v;
        }
        for (;;) {
            json_ws();
            size_t klen;
            char *key = json_parse_string(&klen);
            (void)klen;
            json_ws();
            DIE(*g_src == ':', "json: expected :");
            g_src++;
            JV *val = json_parse();
            if (v->n == cap) {
                cap *= 2;
                v->keys = xrealloc(v->keys, cap * sizeof(char *));
                v->items = xrealloc(v->items, cap * sizeof(JV *));
            }
            v->keys[v->n] = key;
            v->items[v->n] = val;
            v->n++;
            json_ws();
            if (*g_src == ',') {
                g_src++;
                continue;
            }
            DIE(*g_src == '}', "json: expected }");
            g_src++;
            break;
        }
        g_depth--;
    } else if (*g_src == '[') {
        DIE(++g_depth < 256, "json: too deep");
        g_src++;
        v->t = JARR;
        size_t cap = 8;
        v->items = xmalloc(cap * sizeof(JV *));
        json_ws();
        if (*g_src == ']') {
            g_src++;
            g_depth--;
            return v;
        }
        for (;;) {
            JV *val = json_parse();
            if (v->n == cap) {
                cap *= 2;
                v->items = xrealloc(v->items, cap * sizeof(JV *));
            }
            v->items[v->n++] = val;
            json_ws();
            if (*g_src == ',') {
                g_src++;
                continue;
            }
            DIE(*g_src == ']', "json: expected ]");
            g_src++;
            break;
        }
        g_depth--;
    } else if (!strncmp(g_src, "true", 4)) {
        g_src += 4;
        v->t = JBOOL;
        v->num = 1;
    } else if (!strncmp(g_src, "false", 5)) {
        g_src += 5;
        v->t = JBOOL;
    } else if (!strncmp(g_src, "null", 4)) {
        g_src += 4;
        v->t = JNULL;
    } else {
        char *end;
        v->num = strtod(g_src, &end);
        DIE(end > g_src, "json: bad number at %.20s", g_src);
        g_src = end;
        v->t = JNUM;
    }
    return v;
}

static JV *json_load_file(const char *path) {
    FILE *f = fopen(path, "rb");
    DIE(f, "cannot open %s: %s", path, strerror(errno));
    DIE(fseek(f, 0, SEEK_END) == 0, "seek %s", path);
    long sz = ftell(f);
    DIE(sz > 0, "empty %s", path);
    rewind(f);
    char *buf = xmalloc((size_t)sz + 1);
    DIE(fread(buf, 1, (size_t)sz, f) == (size_t)sz, "read %s", path);
    buf[sz] = '\0';
    fclose(f);
    g_src = buf;
    JV *v = json_parse();
    free(buf);
    return v;
}

static JV *jobj_get(const JV *o, const char *key) {
    DIE(o && o->t == JOBJ, "json: object expected (key %s)", key);
    for (size_t i = 0; i < o->n; i++)
        if (!strcmp(o->keys[i], key))
            return o->items[i];
    return NULL;
}
static long long jint(const JV *v) {
    DIE(v && v->t == JNUM, "json: number expected");
    return (long long)v->num;
}
static double jnum(const JV *v) {
    DIE(v && v->t == JNUM, "json: number expected");
    return v->num;
}

/* ============================ safetensors reader ============================ */

typedef struct {
    char *base, *data;
    size_t size;
    JV *header;
} STFile;

typedef struct {
    const char *dtype;
    size_t shape[4];
    int ndim;
    size_t off, nbytes;
} STensor;

static void st_open(STFile *st, const char *path) {
    int fd = open(path, O_RDONLY);
    DIE(fd >= 0, "cannot open %s: %s", path, strerror(errno));
    struct stat sb;
    DIE(fstat(fd, &sb) == 0, "stat %s", path);
    st->size = (size_t)sb.st_size;
    st->base = mmap(NULL, st->size, PROT_READ, MAP_PRIVATE, fd, 0);
    DIE(st->base != MAP_FAILED, "mmap %s: %s", path, strerror(errno));
    close(fd);
    uint64_t hlen;
    memcpy(&hlen, st->base, 8);
    DIE(hlen > 0 && hlen < st->size, "bad safetensors header length");
    char *hjson = xmalloc(hlen + 1);
    memcpy(hjson, st->base + 8, hlen);
    hjson[hlen] = '\0';
    g_src = hjson;
    st->header = json_parse();
    free(hjson);
    st->data = st->base + 8 + hlen;
}

/* NB: returns pointer into a static; one live lookup at a time. */
static const STensor *st_get(const STFile *st, const char *name) {
    JV *v = jobj_get(st->header, name);
    if (!v)
        return NULL;
    static STensor t;
    JV *d = jobj_get(v, "dtype");
    JV *sh = jobj_get(v, "shape");
    JV *ofs = jobj_get(v, "data_offsets");
    DIE(d && sh && ofs, "tensor %s: missing fields", name);
    DIE(d->t == JSTR, "tensor %s: dtype", name);
    t.dtype = d->str;
    DIE(sh->t == JARR && sh->n <= 4, "tensor %s: shape", name);
    t.ndim = (int)sh->n;
    for (int i = 0; i < t.ndim; i++)
        t.shape[i] = (size_t)jint(sh->items[i]);
    DIE(ofs->t == JARR && ofs->n == 2, "tensor %s: offsets", name);
    size_t beg = (size_t)jint(ofs->items[0]), end = (size_t)jint(ofs->items[1]);
    DIE(end >= beg, "tensor %s: bad offsets", name);
    t.off = beg;
    t.nbytes = end - beg;
    return &t;
}

static inline float bf16_to_f32(const uint8_t *p) {
    uint32_t bits = (uint32_t)p[0] | ((uint32_t)p[1] << 8);
    union { uint32_t u; float f; } u;
    u.u = bits << 16;
    return u.f;
}

/* ============================ Q8_0 quantize ============================ */

typedef struct {
    int8_t *q;
    float *s;
    size_t nelem, out, in, ngrp;
} Q80;

static void q80_alloc(Q80 *w, size_t out, size_t in, int gs) {
    DIE(in % (size_t)gs == 0, "GS %d does not divide input dim %zu", gs, in);
    w->out = out;
    w->in = in;
    w->nelem = out * in;
    w->ngrp = in / (size_t)gs;
    w->q = xmalloc(w->nelem);
    w->s = xmalloc(out * w->ngrp * sizeof(float));
}

static void q80_quant(Q80 *w, const float *x, int gs) {
    size_t gsz = (size_t)gs;
    for (size_t r = 0; r < w->out; r++) {
        const float *row = x + r * w->in;
        int8_t *qr = w->q + r * w->in;
        float *sr = w->s + r * w->ngrp;
        for (size_t g = 0; g < w->ngrp; g++) {
            float amax = 0.f;
            for (size_t i = 0; i < gsz; i++)
                amax = fmaxf(amax, fabsf(row[g * gsz + i]));
            float scale = amax / 127.f;
            if (scale == 0.f)
                scale = 1.f; /* 零组保护（附录 E.2#4） */
            sr[g] = scale;
            for (size_t i = 0; i < gsz; i++)
                qr[g * gsz + i] = (int8_t)lrintf(row[g * gsz + i] / scale);
        }
    }
}

static void q80_from_st(const STFile *st, const char *name, Q80 *w, int gs) {
    const STensor *t = st_get(st, name);
    DIE(t, "tensor %s missing", name);
    DIE(!strcmp(t->dtype, "BF16"), "tensor %s: expected BF16 got %s", name, t->dtype);
    DIE(t->ndim == 2, "tensor %s: expected 2D", name);
    size_t out = t->shape[0], in = t->shape[1]; /* [out,in] row-major */
    q80_alloc(w, out, in, gs);
    size_t gsz = (size_t)gs;
    for (size_t r = 0; r < out; r++) {
        const uint8_t *row = (const uint8_t *)st->data + t->off + r * in * 2;
        int8_t *qr = w->q + r * in;
        float *sr = w->s + r * w->ngrp;
        for (size_t g = 0; g < w->ngrp; g++) {
            const uint8_t *grp = row + g * gsz * 2;
            float amax = 0.f;
            for (size_t i = 0; i < gsz; i++)
                amax = fmaxf(amax, fabsf(bf16_to_f32(grp + 2 * i)));
            float scale = amax / 127.f;
            if (scale == 0.f)
                scale = 1.f;
            sr[g] = scale;
            for (size_t i = 0; i < gsz; i++)
                qr[g * gsz + i] = (int8_t)lrintf(bf16_to_f32(grp + 2 * i) / scale);
        }
    }
}

static void q80_write(FILE *f, const Q80 *w) {
    fwrite(w->q, 1, w->nelem, f);
    fwrite(w->s, 4, w->out * w->ngrp, f);
}

static float *st_read_f32(const STFile *st, const char *name, size_t *n_elem) {
    const STensor *t = st_get(st, name);
    DIE(t, "tensor %s missing", name);
    DIE(!strcmp(t->dtype, "BF16"), "tensor %s: expected BF16", name);
    size_t n = 1;
    for (int i = 0; i < t->ndim; i++)
        n *= t->shape[i];
    DIE(t->nbytes == n * 2, "tensor %s: size mismatch", name);
    float *out = xmalloc(n * sizeof(float));
    const uint8_t *p = (const uint8_t *)st->data + t->off;
    for (size_t i = 0; i < n; i++)
        out[i] = bf16_to_f32(p + 2 * i);
    *n_elem = n;
    return out;
}

/* ============================ GPT-2 byte<->unicode ============================ */

static int b2u_cp2byte(int cp) {
    if (cp >= 33 && cp <= 126) return cp;          /* '!'..'~' */
    if (cp >= 161 && cp <= 172) return cp;         /* 0xA1..0xAC */
    if (cp >= 174 && cp <= 255) return cp;         /* 0xAE..0xFF */
    if (cp >= 256 && cp <= 288) return cp - 256;   /* 0x00..0x20 */
    if (cp >= 289 && cp <= 322) return cp - 289 + 127; /* 0x7F..0xA0 */
    if (cp == 323) return 173;                     /* 0xAD */
    return -1;
}

static int utf8_next(const char *s, size_t n, uint32_t *cp) {
    if (n == 0) return -1;
    uint8_t c = (uint8_t)s[0];
    if (c < 0x80) { *cp = c; return 1; }
    int k;
    uint32_t v;
    if ((c & 0xE0) == 0xC0) { k = 2; v = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { k = 3; v = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { k = 4; v = c & 0x07; }
    else return -1;
    if ((size_t)k > n) return -1;
    for (int i = 1; i < k; i++) {
        if (((uint8_t)s[i] & 0xC0) != 0x80) return -1;
        v = (v << 6) | ((uint8_t)s[i] & 0x3F);
    }
    *cp = v;
    return k;
}

static char *b2u_decode(const char *s, size_t slen, size_t *out_len) {
    char *out = xmalloc(slen + 1);
    size_t n = 0, i = 0;
    while (i < slen) {
        uint32_t cp;
        int k = utf8_next(s + i, slen - i, &cp);
        if (k < 0 || b2u_cp2byte((int)cp) < 0) {
            free(out);
            return NULL;
        }
        out[n++] = (char)b2u_cp2byte((int)cp);
        i += (size_t)k;
    }
    *out_len = n;
    return out;
}

/* ============================ rank hash (FNV-1a, open addressing) ============ */

typedef struct {
    uint32_t *slots; /* rank+1, 0 = empty */
    char **keys;
    size_t *klens;
    size_t cap;
} RankHash;

static uint64_t fnv1a(const char *s, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint8_t)s[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

static void rh_init(RankHash *h, size_t expect) {
    h->cap = 1;
    while (h->cap < expect * 2)
        h->cap <<= 1;
    h->slots = xmalloc(h->cap * sizeof(uint32_t));
    h->keys = xmalloc(h->cap * sizeof(char *));
    h->klens = xmalloc(h->cap * sizeof(size_t));
    memset(h->slots, 0, h->cap * sizeof(uint32_t));
}

static void rh_put(RankHash *h, const char *s, size_t n, uint32_t rank) {
    size_t i = fnv1a(s, n) & (h->cap - 1);
    while (h->slots[i]) {
        if (h->klens[i] == n && !memcmp(h->keys[i], s, n))
            return; /* keep first rank */
        i = (i + 1) & (h->cap - 1);
    }
    h->slots[i] = rank + 1;
    h->keys[i] = (char *)s;
    h->klens[i] = n;
}

static int rh_get(const RankHash *h, const char *s, size_t n, uint32_t *rank) {
    size_t i = fnv1a(s, n) & (h->cap - 1);
    while (h->slots[i]) {
        if (h->klens[i] == n && !memcmp(h->keys[i], s, n)) {
            *rank = h->slots[i] - 1;
            return 1;
        }
        i = (i + 1) & (h->cap - 1);
    }
    return 0;
}

/* ============================ writers / rng ============================ */

static void fput_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void fput_i32(FILE *f, int32_t v) { fwrite(&v, 4, 1, f); }
static void fput_f32(FILE *f, float v) { fwrite(&v, 4, 1, f); }

static float neg_inf_f32(void) {
    union { uint32_t u; float f; } u = {0xFF800000u};
    return u.f;
}

/* xoshiro128** -- deterministic; mirrored in fixtures/gen_tiny_ref.py */
typedef struct { uint32_t s[4]; } Rng;
static uint32_t rng_rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }
static uint32_t rng_next(Rng *r) {
    uint32_t *s = r->s;
    uint32_t result = rng_rotl(s[1] * 5, 7) * 9;
    uint32_t t = s[1] << 9;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rng_rotl(s[3], 11);
    return result;
}
static float rng_uniform(Rng *r, float lo, float hi) {
    return lo + (hi - lo) * ((float)(rng_next(r) >> 8) / (float)(1u << 24));
}

/* ============================ config ============================ */

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
    int head_dim, group_size;
    float rope_theta, rms_eps;
} QConfig;

static void write_ckpt_header(FILE *f, const QConfig *c) {
    char hdr[HDR_SIZE];
    memset(hdr, 0, sizeof hdr);
    uint32_t *h = (uint32_t *)hdr;
    h[0] = CKPT_MAGIC;
    h[1] = CKPT_VERSION;
    int32_t *v = (int32_t *)(hdr + 8);
    v[0] = c->dim;         v[1] = c->hidden_dim; v[2] = c->n_layers; v[3] = c->n_heads;
    v[4] = c->n_kv_heads;  v[5] = c->vocab_size; v[6] = c->seq_len;   v[7] = c->head_dim;
    v[8] = c->group_size;
    union { uint32_t u; float f; } tb, eb;
    tb.f = c->rope_theta;
    eb.f = c->rms_eps;
    h[11] = tb.u;
    h[12] = eb.u;
    hdr[52] = 1; /* shared_classifier */
    hdr[53] = 1; /* qk_norm */
    fwrite(hdr, 1, HDR_SIZE, f);
}

/* ============================ --ckpt ============================ */

static const char *LAYERNORMS[2] = {"input_layernorm", "post_attention_layernorm"};
static const char *QKNORMS[2] = {"q_norm", "k_norm"};
static const char *PROJS[7][2] = {
    {"wq", "self_attn.q_proj.weight"}, {"wk", "self_attn.k_proj.weight"},
    {"wv", "self_attn.v_proj.weight"}, {"wo", "self_attn.o_proj.weight"},
    {"w1", "mlp.gate_proj.weight"},    {"w2", "mlp.down_proj.weight"},
    {"w3", "mlp.up_proj.weight"},
};

static int mode_ckpt(const char *hfdir, const char *out, int seq_len, int gs) {
    char path[1024];
    QConfig c;

    snprintf(path, sizeof path, "%s/config.json", hfdir);
    JV *cfg = json_load_file(path);
    c.dim = (int)jint(jobj_get(cfg, "hidden_size"));
    c.hidden_dim = (int)jint(jobj_get(cfg, "intermediate_size"));
    c.n_layers = (int)jint(jobj_get(cfg, "num_hidden_layers"));
    c.n_heads = (int)jint(jobj_get(cfg, "num_attention_heads"));
    c.n_kv_heads = (int)jint(jobj_get(cfg, "num_key_value_heads"));
    c.vocab_size = (int)jint(jobj_get(cfg, "vocab_size"));
    JV *hd = jobj_get(cfg, "head_dim");
    DIE(hd, "config.json: head_dim missing");
    c.head_dim = (int)jint(hd);
    c.rope_theta = (float)jnum(jobj_get(cfg, "rope_theta"));
    c.rms_eps = (float)jnum(jobj_get(cfg, "rms_norm_eps"));
    JV *tie = jobj_get(cfg, "tie_word_embeddings");
    DIE(tie && tie->t == JBOOL && tie->num == 1, "requires tie_word_embeddings=true");
    JV *abias = jobj_get(cfg, "attention_bias");
    DIE(!abias || (abias->t == JBOOL && abias->num == 0), "attention_bias=true unsupported");
    int q_dim = c.n_heads * c.head_dim;
    int kv_dim = c.n_kv_heads * c.head_dim;
    DIE(q_dim % gs == 0 && kv_dim % gs == 0 && c.dim % gs == 0 && c.hidden_dim % gs == 0,
        "GS %d must divide q%d/kv%d/dim%d/hidden%d", gs, q_dim, kv_dim, c.dim, c.hidden_dim);
    c.seq_len = seq_len;
    c.group_size = gs;
    fprintf(stderr,
            "config: dim=%d hidden=%d L=%d heads=%dkv%d hd=%d q_dim=%d kv_dim=%d vocab=%d "
            "theta=%g eps=%g gs=%d\n",
            c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads, c.head_dim, q_dim, kv_dim,
            c.vocab_size, c.rope_theta, c.rms_eps, gs);

    snprintf(path, sizeof path, "%s/model.safetensors", hfdir);
    STFile st;
    st_open(&st, path);

    /* tied 断言：lm_head 与 embed_tokens 逐字节一致（st_get 返回 static，先拷贝） */
    const STensor *embp = st_get(&st, "model.embed_tokens.weight");
    DIE(embp, "model.embed_tokens.weight missing");
    STensor emb = *embp; /* copy */
    const STensor *lmh = st_get(&st, "lm_head.weight");
    if (lmh)
        DIE(lmh->nbytes == emb.nbytes &&
                !memcmp(st.data + emb.off, st.data + lmh->off, emb.nbytes),
            "lm_head.weight != embed_tokens.weight (unexpected for tied model)");

    FILE *f = fopen(out, "wb");
    DIE(f, "cannot write %s: %s", out, strerror(errno));
    write_ckpt_header(f, &c);

    char key[128];
    size_t n;
    float *buf;
    for (int li = 0; li < 2; li++)
        for (int l = 0; l < c.n_layers; l++) {
            snprintf(key, sizeof key, "model.layers.%d.%s.weight", l, LAYERNORMS[li]);
            buf = st_read_f32(&st, key, &n);
            DIE(n == (size_t)c.dim, "%s: %zu != dim", key, n);
            fwrite(buf, 4, n, f);
            free(buf);
        }
    for (int w = 0; w < 2; w++)
        for (int l = 0; l < c.n_layers; l++) {
            snprintf(key, sizeof key, "model.layers.%d.self_attn.%s.weight", l, QKNORMS[w]);
            buf = st_read_f32(&st, key, &n);
            DIE(n == (size_t)c.head_dim, "%s: %zu != head_dim", key, n);
            fwrite(buf, 4, n, f);
            free(buf);
        }
    buf = st_read_f32(&st, "model.norm.weight", &n);
    DIE(n == (size_t)c.dim, "final norm: %zu != dim", n);
    fwrite(buf, 4, n, f);
    free(buf);

    Q80 w;
    q80_from_st(&st, "model.embed_tokens.weight", &w, gs);
    DIE(w.out == (size_t)c.vocab_size && w.in == (size_t)c.dim, "embed shape");
    q80_write(f, &w);
    free(w.q);
    free(w.s);
    for (int l = 0; l < c.n_layers; l++)
        for (int t = 0; t < 7; t++) {
            snprintf(key, sizeof key, "model.layers.%d.%s", l, PROJS[t][1]);
            q80_from_st(&st, key, &w, gs);
            q80_write(f, &w);
            free(w.q);
            free(w.s);
        }

    DIE(fclose(f) == 0, "close %s: %s", out, strerror(errno));
    munmap(st.base, st.size);
    long long q80_b = (long long)c.vocab_size * c.dim +
                      (long long)c.n_layers * (q_dim * c.dim + 2 * kv_dim * c.dim +
                                               c.dim * q_dim + 2 * c.hidden_dim * c.dim +
                                               c.dim * c.hidden_dim);
    fprintf(stderr, "wrote %s: q80 payload %.1f MiB (incl. scales)\n", out,
            (double)q80_b * (1.0 + 4.0 / gs) / 1048576.0);
    return 0;
}

/* ============================ --tok ============================ */

static int mode_tok(const char *hfdir, const char *out) {
    char path[1024];
    snprintf(path, sizeof path, "%s/tokenizer.json", hfdir);
    JV *tj = json_load_file(path);
    JV *model = jobj_get(tj, "model");
    JV *vocab = jobj_get(model, "vocab");
    JV *merges = jobj_get(model, "merges");
    JV *added = jobj_get(tj, "added_tokens");
    DIE(vocab && vocab->t == JOBJ, "tokenizer.json: model.vocab");
    DIE(merges && merges->t == JARR, "tokenizer.json: model.merges");
    DIE(added && added->t == JARR, "tokenizer.json: added_tokens");
    size_t n_vocab = vocab->n, n_merges = merges->n, n_added = added->n;
    fprintf(stderr, "tokenizer.json: vocab=%zu merges=%zu added=%zu\n", n_vocab, n_merges,
            n_added);
    DIE(n_vocab == n_merges + 256,
        "vocab/merges invariant broke: %zu != %zu+256 (merge-rank scoring needs it)", n_vocab,
        n_merges);

    RankHash rh;
    rh_init(&rh, n_merges);
    size_t max_len = 0;
    for (size_t r = 0; r < n_merges; r++) {
        JV *m = merges->items[r];
        DIE(m->t == JARR && m->n == 2 && m->items[0]->t == JSTR && m->items[1]->t == JSTR,
            "merge %zu: expected [a,b] strings", r);
        size_t al = m->items[0]->len, bl = m->items[1]->len;
        char *cat = xmalloc(al + bl + 1);
        memcpy(cat, m->items[0]->str, al);
        memcpy(cat + al, m->items[1]->str, bl);
        cat[al + bl] = '\0';
        /* 统一在解码域（原始字节）建表，与 vocab 查询同域 */
        size_t dlen;
        char *dec = b2u_decode(cat, al + bl, &dlen);
        DIE(dec, "merge %zu: parts not byte-unicode", r);
        rh_put(&rh, dec, dlen, (uint32_t)r);
        if (dlen > max_len)
            max_len = dlen;
    }

    DIE(n_added <= 256, "unexpected added token count %zu", n_added);
    typedef struct { uint32_t id; uint16_t flags; const char *bytes; size_t blen; } Added;
    Added ad[256];
    for (size_t i = 0; i < n_added; i++) {
        JV *a = added->items[i];
        ad[i].id = (uint32_t)jint(jobj_get(a, "id"));
        JV *sp = jobj_get(a, "special");
        int special = sp && sp->t == JBOOL && sp->num == 1;
        uint16_t flags = 0; /* 1=stop(双 eos), 2=renderable */
        if (ad[i].id == 151645 || ad[i].id == 151643)
            flags |= 1;
        else if (!special)
            flags |= 2;
        ad[i].flags = flags;
        JV *ct = jobj_get(a, "content");
        DIE(ct && ct->t == JSTR, "added token %zu: content", i);
        ad[i].bytes = ct->str; /* added tokens are plain text, not byte-unicode */
        ad[i].blen = ct->len;
        if (ct->len > max_len)
            max_len = ct->len;
    }

    snprintf(path, sizeof path, "%s/config.json", hfdir);
    JV *cfg = json_load_file(path);
    long long vocab_size = jint(jobj_get(cfg, "vocab_size"));
    DIE(vocab_size >= (long long)n_vocab, "config vocab_size < tokenizer vocab");

    /* vocab id -> raw bytes */
    char **vbytes = xmalloc((size_t)vocab_size * sizeof(char *));
    size_t *vblens = xmalloc((size_t)vocab_size * sizeof(size_t));
    memset(vbytes, 0, (size_t)vocab_size * sizeof(char *));
    for (size_t i = 0; i < n_vocab; i++) {
        DIE(vocab->items[i]->t == JNUM, "vocab value not a number");
        uint32_t id = (uint32_t)vocab->items[i]->num;
        DIE(id < (uint32_t)vocab_size && !vbytes[id], "vocab id %u bad/duplicate", id);
        size_t blen;
        char *b = b2u_decode(vocab->keys[i], strlen(vocab->keys[i]), &blen);
        DIE(b, "vocab id %u: not byte-unicode decodable", id);
        vbytes[id] = b;
        vblens[id] = blen;
    }

    FILE *f = fopen(out, "wb");
    DIE(f, "cannot write %s: %s", out, strerror(errno));
    fput_u32(f, TOK_MAGIC);
    fput_u32(f, TOK_VERSION);
    fput_u32(f, (uint32_t)vocab_size);
    fput_u32(f, (uint32_t)max_len);
    fput_i32(f, -1); /* bos */
    fput_u32(f, 2);
    fput_u32(f, 151645);
    fput_u32(f, 151643);
    fput_u32(f, (uint32_t)n_added);
    for (size_t i = 0; i < n_added; i++) {
        fput_u32(f, ad[i].id);
        fwrite(&ad[i].flags, 2, 1, f);
        uint16_t ln = (uint16_t)ad[i].blen;
        fwrite(&ln, 2, 1, f);
        fwrite(ad[i].bytes, 1, ad[i].blen, f);
    }
    size_t n_prim = 0;
    for (uint32_t id = 0; id < (uint32_t)vocab_size; id++) {
        float score;
        if (!vbytes[id])
            score = neg_inf_f32(); /* added/gap ids live in the special table or nowhere */
        else {
            uint32_t rank;
            if (rh_get(&rh, vbytes[id], vblens[id], &rank))
                score = -(float)rank;
            else {
                score = -1e10f; /* 基元哨兵（附录 B） */
                n_prim++;
            }
        }
        uint32_t blen = vbytes[id] ? (uint32_t)vblens[id] : 0;
        fput_f32(f, score);
        fput_u32(f, blen);
        if (blen)
            fwrite(vbytes[id], 1, blen, f);
    }
    DIE(fclose(f) == 0, "close %s", out);
    fprintf(stderr, "wrote %s: vocab_size=%lld max_len=%zu primitives=%zu\n", out, vocab_size,
            max_len, n_prim);
    return 0;
}

/* ============================ --tiny ============================ */
/* 微缩同构模型：head_dim != dim/n_heads、qk-norm gain != 1、GS 与真模型不同，
 * 用于在无量化-官方对拍噪声下验证 C 前向（门禁 0③，fixtures/gen_tiny_ref.py 对拍）。 */

static int mode_tiny(const char *out, uint32_t seed) {
    QConfig c;
    c.dim = 64;
    c.hidden_dim = 192;
    c.n_layers = 2;
    c.n_heads = 4;
    c.n_kv_heads = 2;
    c.vocab_size = 256;
    c.seq_len = 64;
    c.head_dim = 32; /* != dim/n_heads=16，显式字段语义 */
    c.group_size = 32;
    c.rope_theta = 10000.f;
    c.rms_eps = 1e-5f;
    int q_dim = c.n_heads * c.head_dim, kv_dim = c.n_kv_heads * c.head_dim;
    DIE(q_dim % c.group_size == 0 && kv_dim % c.group_size == 0 && c.dim % c.group_size == 0 &&
            c.hidden_dim % c.group_size == 0,
        "tiny shape/GS");

    Rng rng = {{seed, seed * 0x9E3779B9u + 1, seed * 0x85EBCA6Bu + 2, seed * 0xC2B2AE35u + 3}};
    for (int i = 0; i < 16; i++)
        rng_next(&rng);

    FILE *f = fopen(out, "wb");
    DIE(f, "cannot write %s: %s", out, strerror(errno));
    write_ckpt_header(f, &c);

    size_t n_norm = (size_t)c.n_layers * (2 * c.dim + 2 * c.head_dim) + c.dim;
    float *norms = xmalloc(n_norm * sizeof(float));
    for (size_t i = 0; i < n_norm; i++)
        norms[i] = rng_uniform(&rng, 0.5f, 1.5f); /* gain != 1 */
    fwrite(norms, 4, n_norm, f);
    free(norms);

    /* 与 mode_ckpt 相同的段序：token 表一次，随后每层 wq wk wv wo w1 w2 w3 */
    size_t out_dims[8], in_dims[8];
    out_dims[0] = (size_t)c.vocab_size;
    in_dims[0] = (size_t)c.dim;
    out_dims[1] = (size_t)q_dim;
    in_dims[1] = (size_t)c.dim;
    out_dims[2] = (size_t)kv_dim;
    in_dims[2] = (size_t)c.dim;
    out_dims[3] = (size_t)kv_dim;
    in_dims[3] = (size_t)c.dim;
    out_dims[4] = (size_t)c.dim;
    in_dims[4] = (size_t)q_dim;
    out_dims[5] = (size_t)c.hidden_dim;
    in_dims[5] = (size_t)c.dim;
    out_dims[6] = (size_t)c.dim;
    in_dims[6] = (size_t)c.hidden_dim;
    out_dims[7] = (size_t)c.hidden_dim;
    in_dims[7] = (size_t)c.dim;
    float scale_w = 0.5f / sqrtf((float)c.dim);
    for (int l = 0; l < c.n_layers; l++)
        for (int t = 0; t < 8; t++) {
            if (t == 0 && l > 0)
                continue; /* token 表只写一次，放在 layer 0 槽位 */
            Q80 w;
            q80_alloc(&w, out_dims[t], in_dims[t], c.group_size);
            float *x = xmalloc(w.nelem * sizeof(float));
            for (size_t i = 0; i < w.nelem; i++)
                x[i] = rng_uniform(&rng, -scale_w, scale_w);
            q80_quant(&w, x, c.group_size);
            q80_write(f, &w);
            free(x);
            free(w.q);
            free(w.s);
        }
    DIE(fclose(f) == 0, "close %s", out);
    fprintf(stderr, "wrote %s: tiny (dim=%d L=%d q=%d kv=%d hd=%d gs=%d seed=%u)\n", out, c.dim,
            c.n_layers, q_dim, kv_dim, c.head_dim, c.group_size, seed);
    return 0;
}

/* ============================ --selfcheck ============================ */

static int mode_selfcheck(const char *hfdir, const char *ckbin, const char *tokbin) {
    char path[1024];
    unsigned char hdr[HDR_SIZE];
    FILE *f = fopen(ckbin, "rb");
    DIE(f, "open %s", ckbin);
    DIE(fread(hdr, 1, HDR_SIZE, f) == HDR_SIZE, "short header");
    uint32_t magic, ver;
    memcpy(&magic, hdr, 4);
    memcpy(&ver, hdr + 4, 4);
    DIE(magic == CKPT_MAGIC && ver == CKPT_VERSION, "bad magic/version %08x/%u", magic, ver);
    int32_t v[9];
    memcpy(v, hdr + 8, sizeof v);
    QConfig c = {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], 0, 0};
    union { uint32_t u; float fl; } tb, eb;
    memcpy(&tb.u, hdr + 44, 4);
    memcpy(&eb.u, hdr + 48, 4);
    c.rope_theta = tb.fl;
    c.rms_eps = eb.fl;
    int q_dim = c.n_heads * c.head_dim, kv_dim = c.n_kv_heads * c.head_dim;
    printf("ckpt: dim=%d hidden=%d L=%d heads=%d kv=%d vocab=%d seq=%d hd=%d gs=%d "
           "theta=%g eps=%g shared=%d qk=%d\n",
           c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab_size, c.seq_len,
           c.head_dim, c.group_size, c.rope_theta, c.rms_eps, hdr[52], hdr[53]);

    snprintf(path, sizeof path, "%s/model.safetensors", hfdir);
    STFile st;
    st_open(&st, path);

    /* file size check */
    fseek(f, 0, SEEK_END);
    long long expect = HDR_SIZE;
    expect += (long long)c.n_layers * (2 * c.dim + 2 * c.head_dim) * 4 + c.dim * 4;
    long long gw = 4 + c.group_size; /* bytes per group incl. f32 scale */
    expect += ((long long)c.vocab_size * c.dim / c.group_size) * gw;
    long long per_layer =
        ((q_dim * c.dim + 2LL * kv_dim * c.dim + (long long)c.dim * q_dim +
          2LL * c.hidden_dim * c.dim + (long long)c.dim * c.hidden_dim) /
         c.group_size) * gw;
    expect += per_layer * c.n_layers;
    DIE(ftell(f) == expect, "file size %lld != expected %lld", ftell(f), expect);
    fseek(f, HDR_SIZE, SEEK_SET);

    /* spot-check: read fp32 norms & row 0 of each tensor type, compare source */
    char key[128];
    size_t n;
    float *buf = xmalloc((size_t)c.hidden_dim * sizeof(float)); /* max norm width */
    double max_rel = 0;
    for (int li = 0; li < 2; li++)
        for (int l = 0; l < c.n_layers; l++) {
            snprintf(key, sizeof key, "model.layers.%d.%s.weight", l, LAYERNORMS[li]);
            float *src = st_read_f32(&st, key, &n);
            DIE(fread(buf, 4, c.dim, f) == (size_t)c.dim, "norm read");
            DIE(!memcmp(buf, src, c.dim * 4), "%s: fp32 mismatch", key);
            free(src);
        }
    for (int w = 0; w < 2; w++)
        for (int l = 0; l < c.n_layers; l++) {
            snprintf(key, sizeof key, "model.layers.%d.self_attn.%s.weight", l, QKNORMS[w]);
            float *src = st_read_f32(&st, key, &n);
            DIE(fread(buf, 4, c.head_dim, f) == (size_t)c.head_dim, "qknorm read");
            DIE(!memcmp(buf, src, c.head_dim * 4), "%s: fp32 mismatch", key);
            free(src);
        }
    {
        float *src = st_read_f32(&st, "model.norm.weight", &n);
        DIE(fread(buf, 4, c.dim, f) == (size_t)c.dim, "final norm read");
        DIE(!memcmp(buf, src, c.dim * 4), "final norm: fp32 mismatch");
        free(src);
    }
    free(buf);

    /* q80 tensors: 抽查每类张量的 row 0（q 行 0 + s 行 0 位于流头部，其余 fseek 跳过） */
    double worst = 0;
    const char *worst_name = "";
    const char *names[16];
    names[0] = "model.embed_tokens.weight";
    for (int t = 0; t < 7; t++) {
        snprintf(key, sizeof key, "model.layers.0.%s", PROJS[t][1]);
        names[t + 1] = strdup(key);
    }
    for (int t = 0; t < 7; t++) {
        snprintf(key, sizeof key, "model.layers.%d.%s", c.n_layers - 1, PROJS[t][1]);
        names[t + 8] = strdup(key);
    }
    long long per_layer_b =
        ((long long)q_dim * c.dim + 2LL * kv_dim * c.dim + (long long)c.dim * q_dim +
         2LL * c.hidden_dim * c.dim + (long long)c.dim * c.hidden_dim) /
        c.group_size * (4 + c.group_size);
    for (int tt = 0; tt < 14; tt++) {
        if (tt == 7)
            continue; /* 无独立 tokens 副本 */
        int t = tt < 7 ? tt : tt - 7;
        int lay = tt < 7 ? 0 : c.n_layers - 1;
        /* 绝对偏移：norms+tokens 之后，第 lay 层第 t 个张量 */
        /* 段序：tokens, wq, wk, wv, wo, w1, w2, w3（t=0..7） */
        long long outs[8] = {(long long)c.vocab_size, q_dim, kv_dim, kv_dim, c.dim,
                             c.hidden_dim, c.dim, c.hidden_dim};
        long long ins[8] = {c.dim, c.dim, c.dim, c.dim, q_dim, c.dim, c.hidden_dim, c.dim};
        long long before = 0;
        for (int k2 = 0; k2 < t; k2++)
            before += outs[k2] * ins[k2] / c.group_size * (4 + c.group_size);
        long long absq = HDR_SIZE + (long long)c.n_layers * (2 * c.dim + 2 * c.head_dim) * 4 +
                         (long long)c.dim * 4 + (long long)lay * per_layer_b + before;
        /* before 已含 tokens 段（t=0），base 不再重复加 */
        const STensor *ten = st_get(&st, names[tt]);
        DIE(ten, "%s missing in safetensors", names[tt]);
        size_t out = ten->shape[0], in = ten->shape[1];
        size_t ngrp = in / (size_t)c.group_size;
        int8_t *q = xmalloc(in);
        float *s = xmalloc(ngrp * sizeof(float));
        if (fseek(f, (long)absq, SEEK_SET) != 0) die("seek"); if (ftell(f) != (long)absq) die("absq %lld beyond eof?", absq);
        if (fread(q, 1, in, f) != in) { fseek(f,0,SEEK_END); die("%s: q read at %lld (eof %ld lay %d t %d before %lld)", names[tt], absq, ftell(f), lay, t, before); }
        DIE(fseek(f, (long)(absq + (long long)out * in), SEEK_SET) == 0, "seek s");
        DIE(fread(s, 4, ngrp, f) == ngrp, "%s: s read", names[tt]);
        double err = 0;
        for (size_t i = 0; i < in; i++) {
            float sc = s[i / (size_t)c.group_size];
            float dq = (float)q[i] * sc;
            float srcv = bf16_to_f32((const uint8_t *)st.data + ten->off + 2 * i);
            err = fmax(err, fabs(dq - srcv) / sc); /* 以半步长为单位，≤0.5 */
        }
        if (err > worst) {
            worst = err;
            worst_name = names[t];
        }
        free(q);
        free(s);
    }
    printf("ckpt q80 row0: worst err %.4f half-steps (%s) -- must be <= 0.5\n",
           worst, worst_name);
    (void)max_rel;
    fclose(f);

    if (tokbin) {
        FILE *tf = fopen(tokbin, "rb");
        DIE(tf, "open %s", tokbin);
        uint32_t tm, tv, vs, ml;
        DIE(fread(&tm, 4, 1, tf) == 1 && fread(&tv, 4, 1, tf) == 1, "tok header");
        DIE(tm == TOK_MAGIC && tv == TOK_VERSION, "tok bad magic/version");
        DIE(fread(&vs, 4, 1, tf) == 1 && fread(&ml, 4, 1, tf) == 1, "tok header2");
        int32_t bos;
        DIE(fread(&bos, 4, 1, tf) == 1, "bos");
        uint32_t neos, eos[2];
        DIE(fread(&neos, 4, 1, tf) == 1 && neos == 2, "n_eos");
        DIE(fread(eos, 4, 2, tf) == 2 && eos[0] == 151645 && eos[1] == 151643, "eos");
        uint32_t nsp;
        DIE(fread(&nsp, 4, 1, tf) == 1, "n_special");
        for (uint32_t i = 0; i < nsp; i++) {
            uint32_t id;
            uint16_t fl, ln;
            DIE(fread(&id, 4, 1, tf) == 1, "sp id");
            DIE(fread(&fl, 2, 1, tf) == 1 && fread(&ln, 2, 1, tf) == 1, "sp fl/ln");
            DIE(fseek(tf, ln, SEEK_CUR) == 0, "sp skip");
        }
        size_t total_bytes = 0;
        uint32_t nonempty = 0;
        for (uint32_t id = 0; id < vs; id++) {
            float sc;
            uint32_t ln;
            DIE(fread(&sc, 4, 1, tf) == 1, "score");
            DIE(fread(&ln, 4, 1, tf) == 1, "len");
            DIE(fseek(tf, ln, SEEK_CUR) == 0, "skip");
            total_bytes += ln;
            if (ln)
                nonempty++;
        }
        printf("tok: vocab=%u max_len=%u nonempty=%u total_payload=%zu\n", vs, ml, nonempty,
               total_bytes);
        fclose(tf);
    }
    printf("selfcheck OK\n");
    return 0;
}

/* ============================ main ============================ */

int main(int argc, char **argv) {
    if (argc < 2)
        die("usage: qwen3_export --ckpt|--tok|--tiny|--selfcheck ... (see header comment)");
    if (!strcmp(argv[1], "--ckpt")) {
        DIE(argc >= 4, "--ckpt <hfdir> <out.bin> [--seq-len N] [--gs N]");
        int seq_len = 512, gs = 64;
        for (int i = 4; i < argc; i++) {
            if (!strcmp(argv[i], "--seq-len") && i + 1 < argc)
                seq_len = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--gs") && i + 1 < argc)
                gs = atoi(argv[++i]);
            else
                die("unknown arg %s", argv[i]);
        }
        return mode_ckpt(argv[2], argv[3], seq_len, gs);
    }
    if (!strcmp(argv[1], "--tok")) {
        DIE(argc == 4, "--tok <hfdir> <out.bin>");
        return mode_tok(argv[2], argv[3]);
    }
    if (!strcmp(argv[1], "--tiny")) {
        uint32_t seed = 42;
        if (argc == 4)
            seed = (uint32_t)strtoul(argv[3], NULL, 0);
        DIE(argc == 3 || argc == 4, "--tiny <out.bin> [seed]");
        return mode_tiny(argv[2], seed);
    }
    if (!strcmp(argv[1], "--selfcheck")) {
        DIE(argc == 4 || argc == 5, "--selfcheck <hfdir> <ckpt.bin> [tok.bin]");
        return mode_selfcheck(argv[2], argv[3], argc == 5 ? argv[4] : NULL);
    }
    die("unknown mode %s", argv[1]);
    return 1;
}
