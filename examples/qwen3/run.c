/* run.c -- run3 主程序（附录 C CLI）。
 *   ./run3 ckpt.bin -z tok.bin [-t T] [-p P] [-k K] [-e 0|1] [-c ctx] [-n steps]
 *                  [-m "one-shot message"] [-i] [-f fixtures_dir]
 * -f：跑门禁（分词 fixtures 100% + 前向 fixtures top-1/余弦/贪心16）后退出。 */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "chat.h"
#include "model.h"
#include "nn.h"
#include "sampler.h"
#include "tokenizer.h"
#include "xnpu/prefill.h"
#include "xnpu/hal.h"

/* ---------- fixtures 门禁 ---------- */

static int gate_tokenizer(const Tokenizer *tok, const char *dir) {
    char path[1024];
    snprintf(path, sizeof path, "%s/tokens.tsv", dir);
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    unsigned char *buf = malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz)
        return -1;
    fclose(f);
    int cases = 0, pass = 0;
    size_t i = 0;
    while (i < (size_t)sz) {
        size_t ns = i;
        while (buf[i] != '\t')
            i++;
        char name[128];
        size_t nl = i - ns < sizeof name ? i - ns : sizeof name - 1;
        memcpy(name, buf + ns, nl);
        name[nl] = '\0';
        i++;
        size_t ls = i;
        while (buf[i] != '\t')
            i++;
        long text_len = strtol((char *)buf + ls, NULL, 10);
        i++;
        const unsigned char *text = buf + i;
        i += (size_t)text_len;
        i++; /* \t */
        size_t is = i;
        while (i < (size_t)sz && buf[i] != '\n')
            i++;
        int exp[4096], ne = 0;
        size_t p = is;
        while (p < i) {
            size_t q = p;
            while (q < i && buf[q] != ',')
                q++;
            exp[ne++] = (int)strtol((char *)buf + p, NULL, 10);
            p = q + 1;
        }
        i++;
        int got_n;
        int *got = tok_encode(tok, (const char *)text, (size_t)text_len, &got_n);
        int ok = got_n == ne;
        for (int k2 = 0; ok && k2 < ne; k2++)
            ok = got[k2] == exp[k2];
        if (ok)
            pass++;
        else {
            printf("FAIL tok %-22s exp(%d) got(%d)\n", name, ne, got_n);
            printf("  exp:");
            for (int k2 = 0; k2 < ne; k2++)
                printf(" %d", exp[k2]);
            printf("\n  got:");
            for (int k2 = 0; k2 < got_n; k2++)
                printf(" %d", got[k2]);
            printf("\n");
        }
        free(got);
        cases++;
    }
    free(buf);
    printf("[gate2] tokenizer: %d/%d exact\n", pass, cases);
    return pass == cases ? 0 : 1;
}

/* 门禁 1：逐位 argmax + 末位 logits 余弦 + 贪心 16 */
static int gate_forward(Transformer *t, const char *dir) {
    char path[1024];
    snprintf(path, sizeof path, "%s/prompts.tsv", dir);
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return -1;
    }
    char line[16384];
    int total_pos = 0, total_top1 = 0;
    int rc = 0;
    while (fgets(line, sizeof line, f)) {
        char *save = NULL;
        char *name = strtok_r(line, "\t", &save);
        char *ids_s = strtok_r(NULL, "\t", &save);
        char *arg_s = strtok_r(NULL, "\t\n", &save);
        if (!name || !ids_s || !arg_s)
            continue;
        int ids[512], ni = 0, arg[512], na = 0;
        {
            char tmp[8192];
            strncpy(tmp, ids_s, sizeof tmp - 1);
            char *sv = NULL;
            for (char *tk = strtok_r(tmp, ",", &sv); tk && ni < 512; tk = strtok_r(NULL, ",", &sv))
                ids[ni++] = atoi(tk);
            strncpy(tmp, arg_s, sizeof tmp - 1);
            sv = NULL;
            for (char *tk = strtok_r(tmp, ",", &sv); tk && na < 512; tk = strtok_r(NULL, ",", &sv))
                arg[na++] = atoi(tk);
        }
        /* 跑前向 */
        float *logits = NULL;
        int top1 = 0;
        for (int p = 0; p < ni; p++) {
            logits = forward(t, ids[p], p);
            int mi = 0;
            for (int q2 = 1; q2 < t->cfg.vocab_size; q2++)
                if (logits[q2] > logits[mi])
                    mi = q2;
            if (mi == arg[p]) {
                top1++;
            } else {
                /* 找期望 id 的 logits 与差值 */
                float mip = logits[mi], ep = logits[arg[p]];
                int rk = 0;
                for (int q3 = 0; q3 < t->cfg.vocab_size; q3++)
                    if (logits[q3] > ep) rk++;
                printf("  MISS %s pos %d: got %d(%.4f) want %d(%.4f) rank %d gap %.4f\n",
                       name, p, mi, mip, arg[p], ep, rk, mip - ep);
            }
        }
        /* 末位 logits vs fixtures */
        snprintf(path, sizeof path, "%s/logits_%s.f32", dir, name);
        FILE *lf = fopen(path, "rb");
        double dot = 0, na2 = 0, nb = 0;
        if (lf) {
            float *ref = malloc((size_t)t->cfg.vocab_size * 4);
            if (fread(ref, 4, (size_t)t->cfg.vocab_size, lf) == (size_t)t->cfg.vocab_size) {
                for (int q2 = 0; q2 < t->cfg.vocab_size; q2++) {
                    dot += (double)ref[q2] * logits[q2];
                    na2 += (double)ref[q2] * ref[q2];
                    nb += (double)logits[q2] * logits[q2];
                }
            }
            free(ref);
            fclose(lf);
        }
        double cosine = (na2 > 0 && nb > 0) ? dot / sqrt(na2 * nb) : -1;
        /* 贪心 16 */
        snprintf(path, sizeof path, "%s/greedy16.tsv", dir);
        FILE *gf = fopen(path, "rb");
        int g16[16], ng = 0;
        int match16 = 0, total16 = 0;
        if (gf) {
            char gl[8192];
            while (fgets(gl, sizeof gl, gf)) {
                char *sv = NULL;
                char *nm = strtok_r(gl, "\t", &sv);
                if (!nm || strcmp(nm, name))
                    continue;
                char *seq = strtok_r(NULL, "\t\n", &sv);
                char *sv2 = NULL;
                for (char *tk = strtok_r(seq, ",", &sv2); tk && ng < 16; tk = strtok_r(NULL, ",", &sv2))
                    g16[ng++] = atoi(tk);
                break;
            }
            fclose(gf);
        }
        if (ng > 0) {
            int cur[512 + 16];
            memcpy(cur, ids, (size_t)ni * sizeof(int));
            /* 注意：KV 已推进到 ni；直接续（前向位置继续） */
            for (int g = 0; g < ng; g++) {
                int mi = 0;
                for (int q2 = 1; q2 < t->cfg.vocab_size; q2++)
                    if (logits[q2] > logits[mi])
                        mi = q2;
                if (mi == g16[g])
                    match16++;
                total16++;
                cur[ni + g] = mi;
                logits = forward(t, mi, ni + g);
            }
        }
        printf("[gate1] %-4s top1 %d/%d cosine %.4f greedy16 %d/%d\n", name, top1, ni, cosine,
               match16, total16);
        total_pos += ni;
        total_top1 += top1;
        if (cosine < 0.98)
            rc = 1;
    }
    fclose(f);
    printf("[gate1] total top1 %d/%d (%.1f%%)\n", total_top1, total_pos,
           100.0 * total_top1 / (double)total_pos);
    return rc;
}

/* ---------- main ---------- */

int main(int argc, char **argv) {
    const char *ckpt_path = NULL, *tok_path = NULL, *fixtures_dir = NULL, *oneshot = NULL;
    float temperature = 0.6f, topp = 0.95f;
    int topk = 20, thinking = 0, ctx_len = 0, steps = 256, interactive = 0;
    uint64_t seed = 42;

    if (argc < 2) {
        fprintf(stderr, "usage: %s ckpt.bin -z tok.bin [flags] (-h for help)\n", argv[0]);
        return 1;
    }
    ckpt_path = argv[1];
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-z") && i + 1 < argc)
            tok_path = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc)
            temperature = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc)
            topp = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc)
            topk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-e") && i + 1 < argc)
            thinking = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc)
            ctx_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc)
            steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-m") && i + 1 < argc)
            oneshot = argv[++i];
        else if (!strcmp(argv[i], "-i"))
            interactive = 1;
        else if (!strcmp(argv[i], "-f") && i + 1 < argc)
            fixtures_dir = argv[++i];
        else if (!strcmp(argv[i], "-s") && i + 1 < argc)
            seed = strtoull(argv[++i], NULL, 0);
        else {
            fprintf(stderr, "unknown flag %s\n", argv[i]);
            return 1;
        }
    }
    if (!tok_path) {
        fprintf(stderr, "missing -z tokenizer.bin\n");
        return 1;
    }

    Transformer t;
    if (build_transformer(&t, ckpt_path, ctx_len) != 0)
        return 1;
    Tokenizer tok;
    if (build_tokenizer(&tok, tok_path) != 0)
        return 1;
    printf("config: dim=%d hidden=%d L=%d heads=%d kv=%d hd=%d vocab=%d seq=%d gs=%d "
           "theta=%g eps=%g\n",
           t.cfg.dim, t.cfg.hidden_dim, t.cfg.n_layers, t.cfg.n_heads, t.cfg.n_kv_heads,
           t.cfg.head_dim, t.cfg.vocab_size, t.cfg.seq_len, t.cfg.group_size, t.cfg.rope_theta,
           t.cfg.rms_eps);

    if (fixtures_dir) {
        int rc = gate_tokenizer(&tok, fixtures_dir);
        if (rc == 0)
            rc |= gate_forward(&t, fixtures_dir);
        free_transformer(&t);
        free_tokenizer(&tok);
        return rc ? 1 : 0;
    }

    /* Phase 3：NPU int8 prefill（设备缺席/失配即静默 CPU 路径，行为与
     * 基线一致——XNPU_DISABLE 是 A/B 的显式基线档）。 */
    g_xn.sched = getenv("XNPU_SCHED") != NULL;   /* selftest 工作点表须同档 */
    if (getenv("XNPU_SELFTEST")) {
        int rc = xnpu_selftest_report(&t.cfg);
        free_transformer(&t);
        free_tokenizer(&tok);
        return rc;
    }
    g_xn.prof = getenv("XNPU_PROF") != NULL;
    if (g_xn.prof) atexit(xnpu_prof_report);
    if (xnpu_init() == 0)
        xnpu_prefill_setup(&t);   /* 失败仅降级，不致命 */

    Sampler s;
    sampler_init(&s, t.cfg.vocab_size, temperature, topp, topk, seed);

    /* 对话缓冲：每轮重编码全对话、从 pos=0 重预填（短对话 O(n^2) 可接受，
     * 换取多轮语义正确：生成文本以 <|im_end|>\n 回填进上下文） */
    char conv[65536];
    size_t conv_len = 0;
    conv[0] = '\0';
    char prompt[8192];
    int turn = 0;
    do {
        const char *user = NULL;
        if (oneshot) {
            user = oneshot;
        } else {
            printf("\n> ");
            fflush(stdout);
            if (!fgets(prompt, sizeof prompt, stdin))
                break;
            size_t pl = strlen(prompt);
            while (pl && (prompt[pl - 1] == '\n' || prompt[pl - 1] == '\r'))
                prompt[--pl] = '\0';
            if (!pl)
                continue;
            if (!strcmp(prompt, "/quit") || !strcmp(prompt, "exit"))
                break;
            user = prompt;
        }
        if (turn == 0)
            conv_len += (size_t)chat_render_message(conv + conv_len,
                                                    (int)(sizeof conv - conv_len), "system",
                                                    "You are a helpful assistant.", 0, 0);
        conv_len += (size_t)chat_render_message(conv + conv_len,
                                                (int)(sizeof conv - conv_len), "user", user, 0, 0);
        conv_len += (size_t)chat_render_message(conv + conv_len,
                                                (int)(sizeof conv - conv_len), "assistant", NULL,
                                                1, thinking);
        int n_ids;
        int *ids = tok_encode(&tok, conv, conv_len, &n_ids);
        if (n_ids >= t.cfg.seq_len - steps - 8) {
            fprintf(stderr, "(context full)\n");
            free(ids);
            break;
        }
        printf("\n");
        /* 生成同时收集文本用于回填 */
        size_t reply_cap = 16384, reply_len = 0;
        char *reply = malloc(reply_cap);
        reply[0] = '\0';
        int gen = 0;
        {
            float *logits = NULL;
            int token = ids[0];
            /* NPU 预填 [0,n)（含 CHECK 影子档）。成功后末 token 的 logits
             * 由一次 CPU 前向产生（读 NPU 回灌的 KV，v1 接受这一个
             * token 的 CPU 重算，Phase 4 再做 lm_head-only 快路）；失败/
             * 超限则函数内回退完整 CPU 重跑。 */
            if (xnpu_prefill_prompt(&t, ids, n_ids) != 0) {
                for (int i2 = 0; i2 < n_ids; i2++) {
                    logits = forward(&t, ids[i2], i2);
                    token = ids[i2];
                }
            } else {
                token = ids[n_ids - 1];
                logits = forward(&t, token, n_ids - 1);
            }
            for (int i2 = 0; i2 < steps; i2++) {
                token = sampler_sample(&s, logits);
                if (tok_is_stop(&tok, token))
                    break;
                uint32_t blen;
                const uint8_t *b = tok_decode(&tok, token, &blen);
                if (blen) {
                    fwrite(b, 1, blen, stdout);
                    fflush(stdout);
                    if (reply_len + blen + 1 > reply_cap) {
                        reply_cap *= 2;
                        reply = realloc(reply, reply_cap);
                    }
                    memcpy(reply + reply_len, b, blen);
                    reply_len += blen;
                    reply[reply_len] = '\0';
                }
                gen++;
                if (n_ids + gen >= t.cfg.seq_len)
                    break;
                logits = forward(&t, token, n_ids + gen - 1);
            }
        }
        conv_len += (size_t)chat_render_message(conv + conv_len,
                                                (int)(sizeof conv - conv_len), "assistant", reply,
                                                0, 0);
        free(reply);
        free(ids);
        printf("\n");
        turn++;
    } while (interactive && !oneshot);

    free_transformer(&t);
    free_tokenizer(&tok);
    return 0;
}
