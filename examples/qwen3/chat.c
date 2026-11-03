/* chat.c -- 模板渲染（§1.2）。生成循环内联在 run.c（需收集回复文本做回填）。 */
#include "chat.h"

#include <stdio.h>
#include <string.h>

int chat_render_message(char *buf, int cap, const char *role, const char *content,
                        int assistant_start, int enable_thinking) {
    int n = 0;
    n += snprintf(buf + n, (size_t)(cap - n), "<|im_start|>%s\n", role);
    if (assistant_start && !enable_thinking)
        n += snprintf(buf + n, (size_t)(cap - n), "<think>\n\n</think>\n\n");
    if (content)
        n += snprintf(buf + n, (size_t)(cap - n), "%s", content);
    if (!assistant_start)
        n += snprintf(buf + n, (size_t)(cap - n), "<|im_end|>\n");
    return n;
}
