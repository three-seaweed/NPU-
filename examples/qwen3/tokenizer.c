/* tokenizer.c -- q3t4 分词器实现。
 * 文件格式 = 附录 B（tools/qwen3_export.c --tok 写出端）。
 * 预分词类表：ASCII/Latin/Greek/Cyrillic/CJK/假名/谚文 归 L，ASCII+常用数字归 N，
 * 类外脚本归符号类（v1 范围外，规划 §5 Phase 2 已声明）。 */
#define _GNU_SOURCE
#include "tokenizer.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define TOK_MAGIC 0x346B3374u
#define TOK_VERSION 1u

static void die(const char *msg) {
    fprintf(stderr, "run3: %s\n", msg);
    exit(1);
}

/* ---------- Unicode 类表（近似，覆盖常用脚本） ---------- */

static int cp_is_letter(uint32_t cp) {
    struct { uint32_t lo, hi; } R[] = {
        {0x41, 0x5A},    {0x61, 0x7A},    {0xAA, 0xAA},    {0xB5, 0xB5},
        {0xBA, 0xBA},    {0xC0, 0xD6},    {0xD8, 0x2FF},   {0x370, 0x373},
        {0x376, 0x377},  {0x37A, 0x37D},  {0x37F, 0x37F},  {0x386, 0x386},
        {0x388, 0x3FF},  {0x400, 0x481},  {0x48A, 0x52F},  {0x531, 0x556},
        {0x559, 0x559},  {0x561, 0x587},  {0x5D0, 0x5EA},  {0x5EF, 0x5F2},
        {0x620, 0x64A},  {0x66E, 0x6D3},  {0x710, 0x72F},  {0x750, 0x77F},
        {0x900, 0x963},  {0x966, 0x97F} /* Devanagari 近似 */, {0xE01, 0xE3A},
        {0xE40, 0xE4E} /* 泰文近似 */,  {0x1100, 0x11FF}, {0x1E00, 0x1FFF},
        {0x2C60, 0x2C7F}, {0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xA000, 0xA48F},
        {0xA720, 0xA7FF}, {0xAC00, 0xD7A3}, {0xF900, 0xFAFF}, {0xFB00, 0xFB06},
        {0xFB13, 0xFB17}, {0xFF21, 0xFF3A}, {0xFF41, 0xFF5A}, {0xFF66, 0xFF9D},
        {0x20000, 0x2A6DF}, {0x2A700, 0x2EBEF}, {0x2F800, 0x2FA1F},
        {0x30000, 0x3134A},
    };
    for (size_t i = 0; i < sizeof R / sizeof R[0]; i++)
        if (cp >= R[i].lo && cp <= R[i].hi)
            return 1;
    return 0;
}

static int cp_is_number(uint32_t cp) {
    struct { uint32_t lo, hi; } R[] = {
        {0x30, 0x39},   {0x660, 0x669}, {0x6F0, 0x6F9}, {0x966, 0x96F},
        {0x9E6, 0x9EF}, {0xFF10, 0xFF19},
    };
    for (size_t i = 0; i < sizeof R / sizeof R[0]; i++)
        if (cp >= R[i].lo && cp <= R[i].hi)
            return 1;
    return 0;
}

/* \s = Unicode White_Space（Rust regex 语义） */
static int cp_is_space(uint32_t cp) {
    return cp == 0x09 || cp == 0x0A || cp == 0x0B || cp == 0x0C || cp == 0x0D ||
           cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

static int cp_is_newline(uint32_t cp) { return cp == 0x0A || cp == 0x0D; }

/* 读一个 UTF-8 码点（容错：非法序列按单字节处理，cp=字节值） */
static int u8_next(const uint8_t *s, size_t n, uint32_t *cp) {
    if (n == 0)
        return -1;
    uint8_t c = s[0];
    if (c < 0x80) {
        *cp = c;
        return 1;
    }
    int k;
    uint32_t v;
    if ((c & 0xE0) == 0xC0) { k = 2; v = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { k = 3; v = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { k = 4; v = c & 0x07; }
    else { *cp = c; return 1; } /* 容错：孤立续字节 */
    if ((size_t)k > n) { *cp = c; return 1; }
    for (int i = 1; i < k; i++) {
        if (((uint8_t)s[i] & 0xC0) != 0x80) { *cp = c; return 1; }
        v = (v << 6) | ((uint8_t)s[i] & 0x3F);
    }
    *cp = v;
    return k;
}

/* ---------- bytes->id 哈希 ---------- */

static uint64_t fnv1a(const uint8_t *s, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) {
        h ^= s[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

static void hash_build(Tokenizer *t) {
    size_t cap = 1;
    while (cap < (size_t)t->vocab_size * 2)
        cap <<= 1;
    t->h_cap = cap;
    t->h_slots = calloc(cap, sizeof(uint32_t));
    if (!t->h_slots)
        die("oom tokenizer hash");
    for (uint32_t id = 0; id < t->vocab_size; id++) {
        if (!t->vocab_len[id])
            continue;
        size_t i = fnv1a(t->vocab_bytes[id], t->vocab_len[id]) & (cap - 1);
        while (t->h_slots[i]) {
            uint32_t j = t->h_slots[i] - 1;
            if (t->vocab_len[j] == t->vocab_len[id] &&
                !memcmp(t->vocab_bytes[j], t->vocab_bytes[id], t->vocab_len[id]))
                break; /* 重复串：保留首个 */
            i = (i + 1) & (cap - 1);
        }
        if (!t->h_slots[i])
            t->h_slots[i] = id + 1;
    }
}

static int vocab_lookup(const Tokenizer *t, const uint8_t *s, size_t n, int *id) {
    size_t i = fnv1a(s, n) & (t->h_cap - 1);
    while (t->h_slots[i]) {
        uint32_t j = t->h_slots[i] - 1;
        if (t->vocab_len[j] == n && !memcmp(t->vocab_bytes[j], s, n)) {
            *id = (int)j;
            return 1;
        }
        i = (i + 1) & (t->h_cap - 1);
    }
    return 0;
}

/* ---------- 加载 ---------- */

int build_tokenizer(Tokenizer *t, const char *path) {
    memset(t, 0, sizeof *t);
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        die("cannot open tokenizer file");
    struct stat sb;
    if (fstat(fd, &sb) != 0)
        die("fstat tokenizer");
    t->map_size = (size_t)sb.st_size;
    t->map = mmap(NULL, t->map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (t->map == MAP_FAILED)
        die("mmap tokenizer");
    close(fd);
    const uint8_t *p = t->map;
    uint32_t magic, ver;
    memcpy(&magic, p, 4);
    p += 4;
    memcpy(&ver, p, 4);
    p += 4;
    if (magic != TOK_MAGIC || ver != TOK_VERSION)
        die("not a q3t4 v1 tokenizer file");
    memcpy(&t->vocab_size, p, 4);
    p += 4;
    memcpy(&t->max_token_length, p, 4);
    p += 4;
    memcpy(&t->bos_id, p, 4);
    p += 4;
    memcpy(&t->n_eos, p, 4);
    p += 4;
    if (t->n_eos != 2)
        die("expect 2 eos");
    for (int i = 0; i < 2; i++) {
        memcpy(&t->eos[i], p, 4);
        p += 4;
    }
    memcpy(&t->n_special, p, 4);
    p += 4;
    t->special = malloc(t->n_special * sizeof(SpecialTok));
    if (!t->special)
        die("oom special");
    for (uint32_t i = 0; i < t->n_special; i++) {
        memcpy(&t->special[i].id, p, 4);
        p += 4;
        memcpy(&t->special[i].flags, p, 2);
        p += 2;
        uint16_t ln;
        memcpy(&ln, p, 2);
        p += 2;
        t->special[i].bytes = p;
        t->special[i].len = ln;
        p += ln;
    }
    t->vocab_bytes = malloc((size_t)t->vocab_size * sizeof(uint8_t *));
    t->vocab_len = malloc((size_t)t->vocab_size * sizeof(uint32_t));
    t->vocab_score = malloc((size_t)t->vocab_size * sizeof(float));
    if (!t->vocab_bytes || !t->vocab_len || !t->vocab_score)
        die("oom vocab tables");
    for (uint32_t id = 0; id < t->vocab_size; id++) {
        memcpy(&t->vocab_score[id], p, 4);
        p += 4;
        uint32_t ln;
        memcpy(&ln, p, 4);
        p += 4;
        t->vocab_bytes[id] = p;
        t->vocab_len[id] = ln;
        p += ln;
    }
    if ((size_t)(p - (const uint8_t *)t->map) != t->map_size)
        die("tokenizer size mismatch");
    hash_build(t);
    return 0;
}

void free_tokenizer(Tokenizer *t) {
    free(t->special);
    free(t->vocab_bytes);
    free(t->vocab_len);
    free(t->vocab_score);
    free(t->h_slots);
    if (t->map && t->map != MAP_FAILED)
        munmap(t->map, t->map_size);
    memset(t, 0, sizeof *t);
}

/* ---------- 预分词：在 s[0..n) 上取一个 chunk，返回长度 ---------- */

/* 大小写不敏感的缩写后缀（branch a） */
static int match_contraction(const uint8_t *s, size_t n) {
    static const char *cands[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    if (n < 2 || s[0] != '\'')
        return 0;
    for (size_t c = 0; c < sizeof cands / sizeof cands[0]; c++) {
        const char *w = cands[c];
        size_t wl = strlen(w);
        if (n < wl)
            continue;
        int ok = 1;
        for (size_t i = 1; i < wl; i++) {
            char a = (char)s[i], b = w[i];
            if (a >= 'A' && a <= 'Z')
                a = (char)(a - 'A' + 'a');
            if (a != b) {
                ok = 0;
                break;
            }
        }
        if (ok)
            return (int)wl;
    }
    return 0;
}

/* 从 s[0..n) 起取一个字母run长度（每步按码点判 L） */
static size_t run_letters(const uint8_t *s, size_t n) {
    size_t i = 0, bytes = 0;
    while (i < n) {
        uint32_t cp;
        int k = u8_next(s + i, n - i, &cp);
        if (k < 0)
            break;
        if (!cp_is_letter(cp))
            break;
        i += (size_t)k;
        bytes = i;
    }
    return bytes;
}

static size_t run_numbers(const uint8_t *s, size_t n) { /* 单个数字（branch c: \p{N}） */
    uint32_t cp;
    int k = u8_next(s, n, &cp);
    if (k > 0 && cp_is_number(cp))
        return (size_t)k;
    return 0;
}

static size_t run_symbols(const uint8_t *s, size_t n) { /* 非空白/字母/数字 的连续 */
    size_t i = 0, bytes = 0;
    while (i < n) {
        uint32_t cp;
        int k = u8_next(s + i, n - i, &cp);
        if (k < 0)
            break;
        if (cp_is_space(cp) || cp_is_letter(cp) || cp_is_number(cp))
            break;
        i += (size_t)k;
        bytes = i;
    }
    return bytes;
}

static size_t run_newlines(const uint8_t *s, size_t n) { /* [\r\n]+ */
    size_t i = 0, bytes = 0;
    while (i < n) {
        uint32_t cp;
        int k = u8_next(s + i, n - i, &cp);
        if (k < 0)
            break;
        if (!cp_is_newline(cp))
            break;
        i += (size_t)k;
        bytes = i;
    }
    return bytes;
}

/* 空白 run（任意 \s） */
static size_t run_space(const uint8_t *s, size_t n) {
    size_t i = 0, bytes = 0;
    while (i < n) {
        uint32_t cp;
        int k = u8_next(s + i, n - i, &cp);
        if (k < 0)
            break;
        if (!cp_is_space(cp))
            break;
        i += (size_t)k;
        bytes = i;
    }
    return bytes;
}

/* 在 pos 处取下一个 chunk（按正则分支次序，左最左-先匹配） */
static size_t pretoken_chunk(const uint8_t *s, size_t n) {
    /* a) (?i:缩写) */
    int c = match_contraction(s, n);
    if (c)
        return (size_t)c;
    /* b) [^\r\n L N]? L+ —— 前缀可有可无，先试有前缀（贪心：正则先吃前缀再要字母） */
    {
        uint32_t cp;
        int k = u8_next(s, n, &cp);
        if (k > 0 && cp != 0x0D && cp != 0x0A && !cp_is_letter(cp) && !cp_is_number(cp)) {
            size_t lr = run_letters(s + k, n - (size_t)k);
            if (lr)
                return (size_t)k + lr;
        }
        /* 无前缀 */
        size_t lr = run_letters(s, n);
        if (lr)
            return lr;
    }
    /* c) \p{N} 单数字 */
    {
        size_t nr = run_numbers(s, n);
        if (nr)
            return nr;
    }
    /* d) ' '? 符号串 [\r\n]* */
    {
        size_t off = 0;
        uint32_t cp;
        int k = u8_next(s, n, &cp);
        if (k > 0 && cp == 0x20) {
            size_t sr = run_symbols(s + 1, n - 1);
            if (sr) {
                off = 1 + sr;
                return off + run_newlines(s + off, n - off);
            }
        }
        size_t sr = run_symbols(s, n);
        if (sr)
            return sr + run_newlines(s + sr, n - sr);
    }
    /* e) \s*[\r\n]+：贪心 \s* 从最长回退，找到使 [\r\n]+ 成立的最大 k，
     * 再贪心延伸换行段到 m，匹配 = [0,m)（可短于整个 ws run） */
    {
        size_t ws = run_space(s, n);
        for (size_t k = ws;; k--) {
            if (k < n && (s[k] == 0x0A || s[k] == 0x0D)) {
                size_t m = k;
                while (m < n && (s[m] == 0x0A || s[m] == 0x0D))
                    m++;
                return m;
            }
            if (k == 0)
                break;
        }
    }
    /* f) \s+(?!\S)：整段 ws；若紧跟非空白则回退最后一个空白字符（按字符回退） */
    {
        size_t ws = run_space(s, n);
        if (ws) {
            size_t take = ws;
            if (ws < n) {
                uint32_t cp;
                int k = u8_next(s + ws, n - ws, &cp);
                if (k > 0 && !cp_is_space(cp)) {
                    /* 回退最后一位空白字符（跳过 UTF-8 续字节） */
                    size_t last = ws - 1;
                    while (last > 0 && (s[last] & 0xC0) == 0x80)
                        last--;
                    take = last;
                }
            }
            if (take > 0)
                return take;
        }
    }
    /* g) \s+ */
    {
        size_t ws = run_space(s, n);
        if (ws)
            return ws;
    }
    /* 兜底：单字节（非法 UTF-8 等） */
    return 1;
}

/* ---------- added-token 最长匹配 ---------- */

static int match_special_at(const Tokenizer *t, const uint8_t *s, size_t n) {
    int best = -1;
    uint32_t best_len = 0;
    for (uint32_t i = 0; i < t->n_special; i++) {
        uint32_t ln = t->special[i].len;
        if (ln <= n && ln > best_len && !memcmp(t->special[i].bytes, s, ln)) {
            best = (int)i;
            best_len = ln;
        }
    }
    return best;
}

/* ---------- BPE 贪心合并（llama2.c 循环形状） ---------- */

#define Q3_MAX_PIECES 1024

static int bpe_chunk(const Tokenizer *t, const uint8_t *s, size_t n, int *out, int *n_out) {
    /* 符号表：偏移+长度。byte-level BPE：初始片段 = 单字节（不是码点！） */
    size_t off[Q3_MAX_PIECES], len[Q3_MAX_PIECES];
    int np = 0;
    for (size_t i = 0; i < n && np < Q3_MAX_PIECES; i++, np++) {
        off[np] = i;
        len[np] = 1;
    }
    while (np > 1) {
        float best_score = -1e30f;
        int best_idx = -1;
        size_t best_len = 0;
        for (int i = 0; i < np - 1; i++) {
            size_t cl = len[i] + len[i + 1];
            uint8_t cat[256];
            if (cl > sizeof cat)
                continue;
            memcpy(cat, s + off[i], len[i]);
            memcpy(cat + len[i], s + off[i + 1], len[i + 1]);
            int id;
            if (vocab_lookup(t, cat, cl, &id)) {
                float sc = t->vocab_score[id];
                if (sc > best_score && sc > -1e9f) { /* 基元哨兵不参与合并 */
                    best_score = sc;
                    best_idx = i;
                    best_len = cl;
                }
            }
        }
        if (best_idx < 0)
            break;
        len[best_idx] = best_len;
        for (int i = best_idx + 1; i < np - 1; i++) {
            off[i] = off[i + 1];
            len[i] = len[i + 1];
        }
        np--;
    }
    if (np >= Q3_MAX_PIECES) {
        *n_out = 0;
        return -1;
    }
    for (int i = 0; i < np; i++) {
        int id = -1;
        if (!vocab_lookup(t, s + off[i], len[i], &id)) {
            fprintf(stderr, "run3: BPE piece not in vocab (len %zu)\n", len[i]);
            id = 0;
        }
        out[i] = id;
    }
    *n_out = np;
    return 0;
}

int *tok_encode(const Tokenizer *t, const char *text, size_t len, int *n_out) {
    /* 第一遍：added tokens 最长匹配整段预切分（HF trie-first 架构），
     * 避免符号分支吞掉 <|im_end|> 的前缀 */
    int cap = 64;
    int *ids = malloc(cap * sizeof(int));
    int n = 0;
    size_t i = 0;
    while (i < len) {
        int sp = match_special_at(t, (const uint8_t *)text + i, len - i);
        if (sp >= 0) {
            if (n == cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
            ids[n++] = (int)t->special[sp].id;
            i += t->special[sp].len;
            continue;
        }
        /* 普通 span：从当前位置到最近的特殊 token 起点（无则到串尾） */
        size_t nearest = len - i;
        for (size_t j = 0; j < len - i; j++) {
            if (match_special_at(t, (const uint8_t *)text + i + j, len - i - j) >= 0) {
                nearest = j;
                break;
            }
        }
        size_t plain_end = i + nearest;
        while (i < plain_end) {
            size_t chunk = pretoken_chunk((const uint8_t *)text + i, plain_end - i);
            if (chunk == 0)
                chunk = 1;
            int pieces[Q3_MAX_PIECES];
            int np = 0;
            if (bpe_chunk(t, (const uint8_t *)text + i, chunk, pieces, &np) != 0)
                die("chunk too large for BPE workspace");
            if (n + np > cap) {
                while (n + np > cap)
                    cap *= 2;
                ids = realloc(ids, cap * sizeof(int));
            }
            memcpy(ids + n, pieces, (size_t)np * sizeof(int));
            n += np;
            i += chunk;
        }
    }
    *n_out = n;
    return ids;
}

const uint8_t *tok_decode(const Tokenizer *t, int id, uint32_t *len) {
    *len = 0;
    if (id < 0 || (uint32_t)id >= t->vocab_size)
        return (const uint8_t *)"";
    for (uint32_t i = 0; i < t->n_special; i++) {
        if ((int)t->special[i].id == id) {
            if (t->special[i].flags & Q3_TOK_FLAG_RENDERABLE) {
                *len = t->special[i].len;
                return t->special[i].bytes;
            }
            return (const uint8_t *)""; /* 特殊 token 不渲染 */
        }
    }
    *len = t->vocab_len[id];
    return t->vocab_bytes[id];
}

int tok_is_stop(const Tokenizer *t, int id) {
    return (uint32_t)id == t->eos[0] || (uint32_t)id == t->eos[1];
}
