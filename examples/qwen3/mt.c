/* mt.c -- 常驻 worker 池实现（plan §3）。
 * 空闲策略：有界自旋（相位间 main 的串行胶水 < 自旋预算，热路径不落锁）
 * 后 mutex+cond park——NPU kick（~55ms×28/题）期间 worker 必须沉睡，
 * 不与 main 的 MMIO 轮询抢核。
 * 同步原语：g_gen 为原子代数（unsigned，!= 比较可回绕），release 发版 /
 * acquire 观察配对，使 g_fn/g_ud 与 worker 的 y[] 写对 main 可见；
 * park 判定与 broadcast 同持 g_mtx，无丢失唤醒。 */
#define _GNU_SOURCE
#include "mt.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#define MT_MAX 16
#define MT_SPIN 4096

static int g_nt = 1;
static int g_inited = 0;
static pthread_t g_th[MT_MAX];

static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static unsigned _Atomic g_gen;
static int _Atomic g_done; /* 已完成 worker 数（不含 main） */
static mt_fn g_fn;
static void *g_ud;

static void *mt_worker_main(void *arg);

void mt_init(void) {
    if (g_inited)
        return;
    g_inited = 1;
    int nt = 1;
    const char *e = getenv("XNPU_THREADS");
    if (e && atoi(e) > 0)
        nt = atoi(e);
    else {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        nt = (n > 0 && n < 4) ? (int)n : 4;
    }
    if (nt > MT_MAX)
        nt = MT_MAX;
    for (int i = 1; i < nt; i++) {
        if (pthread_create(&g_th[i], NULL, mt_worker_main, (void *)(intptr_t)i) != 0) {
            nt = i; /* 部分创建失败：收缩到已创建数，正确性不受影响 */
            break;
        }
    }
    if (nt > 1)
        g_nt = nt;
}

int mt_nthreads(void) {
    if (!g_inited)
        mt_init();
    return g_nt;
}

void mt_range(int tid, int total, int *r0, int *r1) {
    int base = total / g_nt, rem = total % g_nt;
    *r0 = tid * base + (tid < rem ? tid : rem);
    *r1 = *r0 + base + (tid < rem ? 1 : 0);
}

static void *mt_worker_main(void *arg) {
    int tid = (int)(intptr_t)arg;
    /* seen 恒置 0，禁止在启动时读 g_gen：线程首指令可能晚于第一次派发被
     * 调度（实测 WSL 下必现），那样 worker 会把已派发的代数当作自己已完成
     * 的 job 直接 park，main 在 done 等待上永久自旋。mt_init 严格先于首个
     * dispatch（pthread_create 建立 happens-before），代数从 1 起，seen=0
     * 保证首 job 不漏。 */
    unsigned seen = 0;
    for (;;) {
        unsigned spins = 0;
        for (;;) {
            unsigned g = atomic_load_explicit(&g_gen, memory_order_acquire);
            if (g != seen) {
                seen = g;
                break;
            }
            if (++spins < MT_SPIN)
                continue;
            pthread_mutex_lock(&g_mtx);
            if (atomic_load_explicit(&g_gen, memory_order_acquire) == seen)
                pthread_cond_wait(&g_cv, &g_mtx);
            pthread_mutex_unlock(&g_mtx);
            spins = 0;
        }
        g_fn(tid, g_ud);
        atomic_fetch_add_explicit(&g_done, 1, memory_order_acq_rel);
    }
    return NULL;
}

void mt_run(mt_fn fn, void *ud) {
    if (!g_inited)
        mt_init();
    int nt = g_nt;
    if (nt <= 1) {
        fn(0, ud);
        return;
    }
    atomic_store_explicit(&g_done, 0, memory_order_relaxed);
    g_fn = fn;
    g_ud = ud;
    atomic_fetch_add_explicit(&g_gen, 1u, memory_order_release);
    pthread_mutex_lock(&g_mtx); /* 唤醒已 park 的 worker（无 park 时空锁开销） */
    pthread_cond_broadcast(&g_cv);
    pthread_mutex_unlock(&g_mtx);
    fn(0, ud); /* main 分片 */
    while (atomic_load_explicit(&g_done, memory_order_acquire) != nt - 1)
        ; /* 相位级等待（<1ms）不值得 park main */
}
