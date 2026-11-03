/* prefill.h -- NPU int8 prefill 的 run3 集成（plan §5 Phase 3）。
 *
 * 数据流（窗口外层、批内层——plan 定案，重排 memcpy 降为每提示词一次）：
 *   for w (层窗口) { for p0 (批) { for l in 窗口 w {
 *       stage_window_ensure(l); 置 DF_LOAD_BATCH/DF_STORE_BATCH 于窗口
 *       首/末层；填描述符（本批 xbuf 槽/cos-sin 主表切片/KV 镜像）；
 *       kick 层 l } } }
 * 窗口首层从 xbuf 槽装载（窗口 0 = 嵌入，后续窗口 = 上一窗口末层回存的
 * 残差），末层回存到同槽。KV 镜像逐层逐批累积（因果性由层程序保证）。
 *
 * cos/sin 主表按位置一次性生成常驻（[T_cap][q_dim/kv_dim] 每头复制），
 * 批的描述符只改切片基址——零重写。嵌入经 dequant_row 懒反量化进 xbuf。
 *
 * XNPU_CHECK=1：先跑 CPU 影子前向（同 token 序列，填 fp32 cache），快照
 * KV；NPU 预填后逐 (layer,pos) 对拍镜像 vs 快照（rel-to-max + nrmse），
 * 阈值 XNPU_CHECK_TOL（默认 5e-2，调整必须带数据记录）。
 * 注意：NPU/CPU 两路文本不保证逐字节一致（bf16 KV 舍入属设计内），逐
 * 字节 A/B 铁律只适用于同数值路径的发射器变更（门禁 3a 已做）。
 *
 * XNPU_SELFTEST=1：打印 B/T 工作点表 + 程序词数（无需设备），exit 0。
 * 失败语义：任何 kick 失败 → 返回 -1，调用方从 pos=0 纯 CPU 重跑
 * （CPU cache 未被污染，重跑即自愈）。 */
#ifndef Q3_PREFILL_H
#define Q3_PREFILL_H

#include "model.h"

int xnpu_prefill_ready(void);                 /* init+setup 完成，可卸载 */
int xnpu_prefill_setup(Transformer *t);       /* 暂存布局 + 常驻表；0 成功 */
int xnpu_prefill_prompt(Transformer *t, const int *tokens, int n);
int xnpu_selftest_report(const Config *c);    /* 工作点表；总是 0 */

#endif /* Q3_PREFILL_H */
