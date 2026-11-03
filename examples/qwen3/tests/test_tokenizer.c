/* test_tokenizer.c -- 门禁 2：分词 fixtures 100% id 精确匹配。
 * tokens.tsv 记录格式：name \t len(text) \t raw-text-bytes \t id,id,... \n */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tokenizer.h"

int main(int argc, char **argv) {
    const char *tok_path = argc > 1 ? argv[1] : "qwen3_tokenizer.bin";
    const char *fx_path = argc > 2 ? argv[2] : "fixtures/tokens.tsv";
    Tokenizer t;
    if (build_tokenizer(&t, tok_path) != 0)
        return 1;
    FILE *f = fopen(fx_path, "rb");
    if (!f) {
        perror(fx_path);
        return 1;
    }
    /* slurp */
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    unsigned char *buf = malloc(sz);
    if (fread(buf, 1, sz, f) != (size_t)sz)
        return 1;
    fclose(f);

    int cases = 0, pass = 0;
    size_t i = 0;
    while (i < (size_t)sz) {
        /* name */
        size_t ns = i;
        while (buf[i] != '\t')
            i++;
        char name[128];
        size_t nl = i - ns;
        if (nl >= sizeof name)
            nl = sizeof name - 1;
        memcpy(name, buf + ns, nl);
        name[nl] = '\0';
        i++;
        /* len */
        size_t ls = i;
        while (buf[i] != '\t')
            i++;
        long text_len = strtol((char *)buf + ls, NULL, 10);
        i++;
        const unsigned char *text = buf + i;
        i += (size_t)text_len;
        if (buf[i] != '\t') {
            fprintf(stderr, "case %s: bad record\n", name);
            return 1;
        }
        i++;
        size_t is = i;
        while (i < (size_t)sz && buf[i] != '\n')
            i++;
        /* parse expected ids */
        int exp[4096];
        int ne = 0;
        size_t p = is;
        while (p < i) {
            size_t q = p;
            while (q < i && buf[q] != ',')
                q++;
            exp[ne++] = (int)strtol((char *)buf + p, NULL, 10);
            p = q + 1;
        }
        i++; /* \n */

        int got_n = 0;
        int *got = tok_encode(&t, (const char *)text, (size_t)text_len, &got_n);
        int ok = got_n == ne;
        for (int k = 0; ok && k < ne; k++)
            if (got[k] != exp[k])
                ok = 0;
        if (ok) {
            pass++;
        } else {
            printf("FAIL %-22s exp(%d):", name, ne);
            for (int k = 0; k < ne; k++)
                printf(" %d", exp[k]);
            printf("\n     %-22s got(%d):", "", got_n);
            for (int k = 0; k < got_n; k++)
                printf(" %d", got[k]);
            printf("\n");
        }
        free(got);
        cases++;
    }
    printf("tokenizer: %d/%d cases exact\n", pass, cases);
    free_tokenizer(&t);
    return pass == cases ? 0 : 1;
}
