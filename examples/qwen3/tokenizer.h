/* tokenizer.h -- q3t4 byte-level BPE 分词器（plan Phase 2，附录 B 格式）。
 * encode 语义镜像 HF tokenizers（Rust regex 左最左-先分支 + ByteLevel BPE）：
 *   1) added tokens 最长匹配预切分（含 <think> 等非 special 的 added）
 *   2) 预分词正则（tokenizer.json 实测）：
 *      (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n L N]? L+ | N | ' '+符号串+换行*
 *      | \s*[\r\n]+ | \s+(?!\S) | \s+
 *   3) 逐字节初始切分 + score 贪心合并（llama2.c 循环形状）
 */
#ifndef Q3_TOKENIZER_H
#define Q3_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#define Q3_TOK_FLAG_STOP 1u       /* 双 eos 之一 */
#define Q3_TOK_FLAG_RENDERABLE 2u /* 非 special 的 added token（如 <think>） */

typedef struct {
    uint32_t id;
    uint16_t flags;
    const uint8_t *bytes;
    uint32_t len;
} SpecialTok;

typedef struct {
    uint32_t vocab_size;
    uint32_t max_token_length;
    int32_t bos_id; /* -1 */
    uint32_t n_eos;
    uint32_t eos[2];
    SpecialTok *special;
    uint32_t n_special;
    /* vocab 表（mmap 指针） */
    const uint8_t **vocab_bytes; /* [vocab_size] */
    uint32_t *vocab_len;
    float *vocab_score;
    void *map;
    size_t map_size;
    /* bytes -> id 哈希（FNV-1a 开放寻址） */
    uint32_t *h_slots; /* id+1, 0=空 */
    size_t h_cap;
} Tokenizer;

int build_tokenizer(Tokenizer *t, const char *path);
void free_tokenizer(Tokenizer *t);

/* 编码：返回新 malloc 的 id 数组，*n_out 为长度；文本为原始 UTF-8 字节 */
int *tok_encode(const Tokenizer *t, const char *text, size_t len, int *n_out);

/* id -> 字节串；特殊 token 依 flags（不可渲染的返回 len=0） */
const uint8_t *tok_decode(const Tokenizer *t, int id, uint32_t *len);

int tok_is_stop(const Tokenizer *t, int id);

#endif
