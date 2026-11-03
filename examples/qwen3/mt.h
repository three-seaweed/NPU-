/* mt.h -- 常驻 worker 池（conduct_docs/cpu_multicore_neon_decode_plan.md §3）。
 * XNPU_THREADS 选择线程数（默认 min(nproc,4)；1 = 纯串行旧路径）。
 * mt_run 分片调用 fn(tid,ud)，main=tid0 参与计算并阻塞至全员完成。
 * 逐位一致由调用方切分方式保证：按输出行 / 按 q 头切分，片内序不变。 */
#ifndef Q3_MT_H
#define Q3_MT_H

typedef void (*mt_fn)(int tid, void *ud);

void mt_init(void);     /* 惰性，幂等；mt_run 首次调用自动触发 */
int  mt_nthreads(void); /* 1..MT_MAX */
void mt_run(mt_fn fn, void *ud);

/* total 分 g_nt 片，tid 取第 tid 片的行区间 [r0,r1)：无缝无交叠 */
void mt_range(int tid, int total, int *r0, int *r1);

#endif
