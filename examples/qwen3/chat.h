/* chat.h -- 对话模板与生成循环（规划 §1.2）。 */
#ifndef Q3_CHAT_H
#define Q3_CHAT_H

#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

/* 生成直到 stop（双 eos）或 steps 上限；逐 token 经 fn 回调（打印）。
 * 返回生成的 token 数（不含 stop）。*/
int chat_generate(Transformer *t, Tokenizer *tok, Sampler *s, const int *prompt_ids,
                  int n_prompt, int steps, int pos_base, void (*emit)(const uint8_t *bytes,
                                                                     uint32_t len));

/* 拼一段消息的模板文本（含尾缀），写入 buf（容量 cap），返回长度。
 * role: "system"/"user"/"assistant"；assistant_start=1 时只拼
 * "<|im_start|>assistant\n" + 可选空 think 预填（不发 <|im_end|>）。 */
int chat_render_message(char *buf, int cap, const char *role, const char *content,
                        int assistant_start, int enable_thinking);

#endif
