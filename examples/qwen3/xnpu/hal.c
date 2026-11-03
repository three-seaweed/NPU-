/* hal.c -- 见 hal.h 头注。移植自 runq_xnpu.c :730-895/:1873-1925。 */
#define _GNU_SOURCE
#include "hal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

XnpuHal g_xn;

uint64_t xn_now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static int xnpu_open_regs(uint64_t base) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return -1;
    void *p = mmap(NULL, XN_REG_MAP_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)base);
    close(fd);
    if (p == MAP_FAILED) return -1;
    g_xn.reg = (volatile uint32_t *)p;
    g_xn.base = base;
    return 0;
}

/* DTCM must behave as RAM and STATUS must be READ-ONLY.  A STATUS write that
 * sticks means the window is plain DDR, not the NPU -- the usual symptom of a
 * wrong base or an unconfigured PL. (runq :750-762) */
static int xnpu_probe(void) {
    const uint32_t pat = 0xA5C3F00Du;
    xn_wr(XN_DTCM + 0, pat);
    xn_wr(XN_DTCM + 4, ~pat);
    if (xn_rd(XN_DTCM + 0) != pat || xn_rd(XN_DTCM + 4) != ~pat) return -1;
    xn_wr(XN_CTRL + 0x00, 0x2u);            /* clear-done (CTRL write-only) */
    xn_wr(XN_CTRL + 0x04, 0xDEADBEEFu);     /* STATUS read-only in the RTL */
    if (xn_rd(XN_CTRL + 0x04) == 0xDEADBEEFu) return -1;
    return 0;
}

static int read_sysfs_u64(const char *path, uint64_t *out) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    long long v;
    int r = fscanf(f, "%lli", &v);
    fclose(f);
    if (r != 1) return -1;
    *out = (uint64_t)v;
    return 0;
}

/* Reserved DMA-reachable memory.  u-dma-buf（O_SYNC 请求非缓存映射）或
 * XNPU_RESMEM_PHYS 显式 carveout。缓存一致性默认靠 xn_evict 行走：
 * 板 2026-10-09 实测 u-dma-buf sync 属性 store 路径内核 oops（驱动 5.4.2
 * on 5.10-xilinx），此后连 open 这些属性都可能挂死——XNPU_RM_SYNC 仅留档，
 * 默认绝不开。 */
static int xnpu_open_resmem(const char *name) {
    const char *ep_ = getenv("XNPU_RESMEM_PHYS");
    if (ep_) {
        uint64_t phys = strtoull(ep_, NULL, 0);
        const char *es = getenv("XNPU_RESMEM_SIZE");
        uint64_t size = es ? strtoull(es, NULL, 0) : (16ull << 20);
        if (phys + size > XN_DMA_LIMIT) {
            fprintf(stderr, "[xnpu] resmem 0x%llx+0x%llx crosses the 2GiB M_AXI limit\n",
                    (unsigned long long)phys, (unsigned long long)size);
            return -1;
        }
        int fd = open("/dev/mem", O_RDWR | O_SYNC);
        if (fd < 0) {
            fprintf(stderr, "[xnpu] open /dev/mem: %s\n", strerror(errno));
            return -1;
        }
        void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)phys);
        close(fd);
        if (p == MAP_FAILED) {
            fprintf(stderr, "[xnpu] mmap resmem: %s\n", strerror(errno));
            return -1;
        }
        g_xn.rm_virt = p;
        g_xn.rm_phys = phys;
        g_xn.rm_size = size;
        g_xn.rm_used = 0;
        fprintf(stderr, "[xnpu] resmem /dev/mem phys=0x%llx size=%lluKB\n",
                (unsigned long long)phys, (unsigned long long)(size / 1024));
        return 0;
    }
    char path[256];
    uint64_t phys, size;
    const char *cls = NULL;
    const char *classes[] = { "u-dma-buf", "udmabuf", NULL };
    for (int i = 0; classes[i]; i++) {
        snprintf(path, sizeof path, "/sys/class/%s/%s/phys_addr", classes[i], name);
        if (read_sysfs_u64(path, &phys) == 0) {
            cls = classes[i];
            break;
        }
    }
    if (!cls) {
        fprintf(stderr, "[xnpu] no u-dma-buf '%s' (set XNPU_RESMEM_PHYS)\n", name);
        return -1;
    }
    snprintf(path, sizeof path, "/sys/class/%s/%s/size", cls, name);
    if (read_sysfs_u64(path, &size)) {
        fprintf(stderr, "[xnpu] read %s failed\n", path);
        return -1;
    }
    if (phys + size > XN_DMA_LIMIT) {
        fprintf(stderr, "[xnpu] resmem 0x%llx+0x%llx crosses the 2GiB M_AXI limit\n",
                (unsigned long long)phys, (unsigned long long)size);
        return -1;
    }
    snprintf(path, sizeof path, "/dev/%s", name);
    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "[xnpu] open %s: %s\n", path, strerror(errno));
        return -1;
    }
    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        fprintf(stderr, "[xnpu] mmap %s: %s\n", path, strerror(errno));
        return -1;
    }
    g_xn.rm_virt = p;
    g_xn.rm_phys = phys;
    g_xn.rm_size = size;
    g_xn.rm_used = 0;
    fprintf(stderr, "[xnpu] resmem %s/%s phys=0x%llx size=%lluKB\n", cls, name,
            (unsigned long long)phys, (unsigned long long)(size / 1024));
    return 0;
}

int rm_alloc(size_t nb, uint8_t **virt, uint64_t *phys) {
    size_t off = (g_xn.rm_used + 63) & ~(size_t)63;
    if (off + nb > g_xn.rm_size) return -1;
    *virt = g_xn.rm_virt + off;
    *phys = g_xn.rm_phys + off;
    g_xn.rm_used = off + nb;
    return 0;
}

/* 驱逐行走：扫一块死缓冲的每 64B 行，把 A53 的 L1(32K)/L2(1M) 挤出去——
 * 脏行下推、DMA 写入的陈旧行失效。1280KB 是实测下限（4MB 是 4 倍过度
 * 且曾是 PS 侧 #1 开销）；XNPU_EVICT_KB 可调。 (runq :795-816) */
void xnpu_evict(void) {
    /* 8MB: the thrash must defeat L2 residency of the u-dma-buf mirror lines
     * for EVERY host binary layout. 1280KB is only 1.25x L2 and loses to
     * conflict misses on some layouts -- the mirror then serves pre-DMA
     * content to both XNPU_CHECK and the generation-path KV readback
     * (2026-10-25: same board, same bitstream, old binary exact / new binary
     * 1671-of-1680 KV points stale; XNPU_EVICT_KB=8192 restores the exact
     * baseline). Board cost: ~40ms of walks per prefill, vs ~18s total. */
    uint32_t want = (8192u << 10);
    const char *ek = getenv("XNPU_EVICT_KB");
    if (ek) {
        long v = atol(ek);
        if (v >= 64 && v <= 8192) want = (uint32_t)v << 10;
    }
    static uint8_t dead[8u << 20];
    static volatile uint8_t *const walk = dead;
    uint64_t t0 = g_xn.prof ? xn_now_ns() : 0;
    for (uint32_t i = 0; i < want; i += 64)
        walk[i] = (uint8_t)i;
    __sync_synchronize();
    if (g_xn.prof) g_xn.p_evict_ns += xn_now_ns() - t0;
}

int xnpu_init(void) {
    if (g_xn.enabled || g_xn.unhealthy) return g_xn.enabled ? 0 : -1;
    if (getenv("XNPU_DISABLE")) return -1;
    g_xn.timeout_ms = 30000;
    {
        const char *to = getenv("XNPU_TIMEOUT_MS");
        if (to) {
            long v = atol(to);
            if (v >= 1 && v <= 3600000) g_xn.timeout_ms = (uint32_t)v;
        }
    }
    uint64_t base = 0x400000000ULL;
    {
        const char *b = getenv("XNPU_BASE");
        if (b) base = strtoull(b, NULL, 0);
    }
    const char *rm = getenv("XNPU_RESMEM");
    if (!rm) rm = "resmem";
    g_xn.prefill = 1;
    {
        const char *pf = getenv("XNPU_PREFILL");
        if (pf) g_xn.prefill = atoi(pf) != 0;
    }

    if (xnpu_open_regs(base)) {
        fprintf(stderr, "[xnpu] /dev/mem mmap @0x%llx failed; CPU path\n",
                (unsigned long long)base);
        return -1;
    }
    if (xnpu_probe()) {
        fprintf(stderr, "[xnpu] no NPU response at 0x%llx (PL not loaded, or wrong base); CPU path\n",
                (unsigned long long)base);
        return -1;
    }
    if (xnpu_open_resmem(rm)) {
        fprintf(stderr, "[xnpu] no reserved DMA memory; CPU path\n");
        return -1;
    }
    g_xn.enabled = 1;
    fprintf(stderr, "[xnpu] device ready: base=0x%llx, DTCM 512KiB, prefill=%s\n",
            (unsigned long long)base, g_xn.prefill ? "on" : "off");
    return 0;
}

int xnpu_kick(const char *what) {
    uint64_t t0 = g_xn.prof ? xn_now_ns() : 0;
    uint32_t kicks0 = xn_rd(XN_CTRL + 0x40);
    __sync_synchronize();
    xn_wr(XN_CTRL + 0x00, 0x2u);   /* clear stale done */
    xn_wr(XN_CTRL + 0x00, 0x1u);   /* start */
    uint64_t deadline = xn_now_ns() + (uint64_t)g_xn.timeout_ms * 1000000ull;
    uint64_t polls = 0;
    struct timespec pause = { .tv_sec = 0, .tv_nsec = 50000 }; /* 50 us */
    for (;;) {
        uint32_t status = xn_rd(XN_CTRL + 0x04);
        polls++;
        if (status & 1u) break;
        if (xn_now_ns() >= deadline) {
            /* 无 abort CSR（UserGuide 6.2）：唯一稳妥处置 = 停用设备走 CPU。 */
            g_xn.p_status_reads += polls;
            fprintf(stderr, "[xnpu] %s kick timed out after %u ms (%llu STATUS reads); "
                            "disabling XNPU for the rest of the run\n",
                    what, g_xn.timeout_ms, (unsigned long long)polls);
            g_xn.unhealthy = 1;
            g_xn.enabled = 0;
            return -1;
        }
        nanosleep(&pause, NULL);
    }
    g_xn.p_status_reads += polls;
    __sync_synchronize();
    uint32_t kicks1 = xn_rd(XN_CTRL + 0x40);
    if (kicks1 != kicks0 + 1u) {
        fprintf(stderr, "[xnpu] %s accepted-kick mismatch: %u -> %u; disabling XNPU\n",
                what, kicks0, kicks1);
        g_xn.unhealthy = 1;
        g_xn.enabled = 0;
        return -1;
    }
    g_xn.p_kicks++;
    if (g_xn.prof) {
        g_xn.p_kick_ns += xn_now_ns() - t0;
        g_xn.hw_matrix += xn_rd(XN_CTRL + 0x24);
        g_xn.hw_vector += xn_rd(XN_CTRL + 0x28);
        g_xn.hw_dma += xn_rd(XN_CTRL + 0x2C);
        g_xn.hw_dep_m += xn_rd(XN_CTRL + 0x4C);
        g_xn.hw_dep_v += xn_rd(XN_CTRL + 0x50);
        g_xn.hw_dmawait += xn_rd(XN_CTRL + 0x3C);
        g_xn.hw_cycles += xn_rd(XN_CTRL + 0x44);
        g_xn.hw_memwait += xn_rd(XN_CTRL + 0x54);
        g_xn.hw_memclear += xn_rd(XN_CTRL + 0x58);
        g_xn.hw_bothidle += xn_rd(XN_CTRL + 0x5C);
        g_xn.hw_dep_v_solo += xn_rd(XN_CTRL + 0x60);
    }
    return 0;
}

void xnpu_prof_report(void) {
    if (!g_xn.prof || !g_xn.p_kicks) return;
    fprintf(stderr, "\n[xnpu-prof] kicks=%llu  kick wall=%.2f ms  staging=%.2f ms  "
                    "evict=%.2f ms  STATUS reads=%.1f/kick\n",
            (unsigned long long)g_xn.p_kicks, g_xn.p_kick_ns * 1e-6,
            g_xn.p_stage_ns * 1e-6, g_xn.p_evict_ns * 1e-6,
            (double)g_xn.p_status_reads / (double)g_xn.p_kicks);
    fprintf(stderr, "[xnpu-prof] PS MMIO: ITCM=%llu words, desc=%llu words "
                    "(%.1f + %.1f /kick)\n",
            (unsigned long long)g_xn.p_itcm_words, (unsigned long long)g_xn.p_desc_words,
            (double)g_xn.p_itcm_words / (double)g_xn.p_kicks,
            (double)g_xn.p_desc_words / (double)g_xn.p_kicks);
    fprintf(stderr, "[xnpu-prof] HW/kick: cycles=%.0f matrix=%llu vector=%llu dma=%llu "
                    "depwaitM=%llu depwaitV=%llu dmawait=%llu\n",
            (double)g_xn.hw_cycles / (double)g_xn.p_kicks,
            (unsigned long long)g_xn.hw_matrix, (unsigned long long)g_xn.hw_vector,
            (unsigned long long)g_xn.hw_dma, (unsigned long long)g_xn.hw_dep_m,
            (unsigned long long)g_xn.hw_dep_v, (unsigned long long)g_xn.hw_dmawait);
    fprintf(stderr, "[xnpu-prof] barrier: memwait=%llu memclear=%llu bothidle=%llu "
                    "depv_solo=%llu\n",
            (unsigned long long)g_xn.hw_memwait, (unsigned long long)g_xn.hw_memclear,
            (unsigned long long)g_xn.hw_bothidle, (unsigned long long)g_xn.hw_dep_v_solo);
}
