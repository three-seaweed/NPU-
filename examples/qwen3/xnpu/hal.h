/* hal.h -- NPU 设备抽象（plan §5 Phase 3；附录 D 纪律：自 runq_xnpu.c
 * :730-895/:1873-1925 移植，砍掉 fp32/gemv/sched 分支，保留板侧生死机制）。
 *
 * 环境变量（与 runq 同名同义）：
 *   XNPU_BASE        控制窗物理基址（默认 0x400000000，以 Vivado Address
 *                    Editor 为准）
 *   XNPU_RESMEM      u-dma-buf 名（默认 resmem）；或 XNPU_RESMEM_PHYS/
 *                    XNPU_RESMEM_SIZE 显式 carveout（/dev/mem 档）
 *   XNPU_TIMEOUT_MS  kick 超时（默认 30000）
 *   XNPU_DISABLE     存在即走纯 CPU（A/B 对照的基线档）
 *   XNPU_SCHED       存在即开层程序 cb/ci 双缓冲（自 runq 移植的向量侧
 *                    depwait 杠杆；prefill.c 的 map/程序两侧同值取用）
 *   XNPU_PROF / XNPU_CHECK / XNPU_SELFTEST 见 prefill.h
 *
 * 板侧教训（全部继承，勿重蹈）：
 *   - u-dma-buf sync 属性路径内核 oops（2026-10-09）→ 默认走 xn_evict
 *     驱逐行走（1280KB > L2），XNPU_RM_SYNC 仅留档；
 *   - resmem 越过 2GiB M_AXI 硬限直接硬失败；
 *   - kick 超时无 abort CSR，唯一处置 = 停用设备走 CPU。 */
#ifndef Q3_HAL_H
#define Q3_HAL_H

#include <stddef.h>
#include <stdint.h>

#define XN_ITCM         0x00000u
#define XN_DTCM         0x80000u
#define XN_CTRL         0x100000u
#define XN_REG_MAP_BYTES 0x140000u
#define XN_DMA_LIMIT    0x80000000ULL

typedef struct {
    volatile uint32_t *reg;
    uint64_t base;
    uint8_t *rm_virt;
    uint64_t rm_phys;
    size_t rm_size, rm_used;
    int enabled;      /* 设备已映射、probe 通过、resmem 就绪 */
    int unhealthy;    /* kick 超时/失配：停用，不再碰设备 */
    int prof;         /* XNPU_PROF：kick/evict/暂存记账 */
    int check;        /* XNPU_CHECK：CPU 影子对拍 */
    int prefill;      /* XNPU_PREFILL=0 关闭卸载（默认开） */
    int sched;        /* XNPU_SCHED：层程序 cb/ci 双缓冲（fe_linear 的
                         depwaitV 杠杆；map 与程序两侧必须同值） */
    uint32_t timeout_ms;
    /* 记账（XNPU_PROF） */
    uint64_t p_kicks, p_kick_ns, p_stage_ns, p_status_reads, p_evict_ns;
    uint64_t p_itcm_words, p_desc_words;
    uint64_t hw_cycles, hw_matrix, hw_vector, hw_dma, hw_dep_m, hw_dep_v, hw_dmawait;
    /* V1 屏障归因（PERF 0x54/0x58/0x5C/0x60，见 Npu.scala 头注释）：
       memwait−memclear = 全排空等待中"单元其实已闲"以外的开销；
       bothidle = kick 死时间；dep_v_solo = depwaitV 的真气泡份额。 */
    uint64_t hw_memwait, hw_memclear, hw_bothidle, hw_dep_v_solo;
} XnpuHal;

extern XnpuHal g_xn;

uint64_t xn_now_ns(void);
int xnpu_init(void);                 /* 幂等；成功置 g_xn.enabled */
void xnpu_evict(void);               /* 缓存驱逐行走（DMA 前后一致性） */
int xnpu_kick(const char *what);     /* CTRL start + STATUS 轮询 + 记账 */
int rm_alloc(size_t nb, uint8_t **virt, uint64_t *phys);
void xnpu_prof_report(void);

static inline void xn_wr(uint32_t off, uint32_t v) { g_xn.reg[off >> 2] = v; }
static inline uint32_t xn_rd(uint32_t off) { return g_xn.reg[off >> 2]; }

#endif /* Q3_HAL_H */
