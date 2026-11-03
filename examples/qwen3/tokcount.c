/* tokcount.c -- 逐轮上下文长度预检工具（2026-11-03 多轮板上测试配套）。
 *
 * 读入文本（stdin 或文件参数），按 run.c 同款 tokenizer 精确计 token 数。
 * 用途：多轮队列上板前，逐轮渲染全对话（chat.c 模板）并断言每轮
 * n_ids <= NPU prefill 上限（tcap，板测工作点 75）——把"上下文不足
 * 回答不了/静默回 CPU"的情况在设计阶段排除，而不是上了板才发现。
 *
 * 用法: tokcount <tokenizer.bin> [file]  （缺 file 读 stdin）
 * 输出: 唯一一行 "<n> tokens"，脚本按数字断言。
 */
#include <stdio.h>
#include <stdlib.h>

#include "tokenizer.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: tokcount <tokenizer.bin> [file]\n");
        return 2;
    }
    Tokenizer t;
    if (build_tokenizer(&t, argv[1]) != 0) {
        fprintf(stderr, "build_tokenizer failed\n");
        return 1;
    }
    size_t cap = 1 << 20;
    unsigned char *buf = malloc(cap);
    size_t len = 0;
    if (argc > 2) {
        FILE *f = fopen(argv[2], "rb");
        if (!f) { perror(argv[2]); return 1; }
        len = fread(buf, 1, cap, f);
        fclose(f);
    } else {
        len = fread(buf, 1, cap, stdin);
    }
    int n_ids;
    int *ids = tok_encode(&t, (const char *)buf, len, &n_ids);
    printf("%d\n", n_ids);
    free(ids);
    free(buf);
    free_tokenizer(&t);
    return 0;
}
