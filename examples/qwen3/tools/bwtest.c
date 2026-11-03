/* bwtest -- 多线程顺序读带宽（M2 数据驱动的上限参照）。
 * 读同一个文件 N 次，第 2 次起数据在 page cache ⇒ 测的是 DDR 读带宽，
 * 与 run3 decode 的权重流形态同构（大块顺序读、无线程间共享）。 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static const char *g_path;
static long g_size;
static int g_nt;

static void *work(void *a) {
    int tid = (int)(long)a;
    long chunk = g_size / g_nt, off = (long)tid * chunk;
    int fd = open(g_path, O_RDONLY);
    if (fd < 0) return NULL;
    const long BUFSZ = 1 << 20;
    char *buf = malloc(BUFSZ);
    long left = chunk, sum = 0;
    while (left > 0) {
        long n = left < BUFSZ ? left : BUFSZ;
        ssize_t r = pread(fd, buf, (size_t)n, off);
        if (r <= 0) break;
        sum += buf[0];
        off += r;
        left -= r;
    }
    close(fd);
    free(buf);
    return (void *)sum;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: bwtest <file> <threads> [reps]\n"); return 1; }
    g_path = argv[1];
    g_nt = atoi(argv[2]);
    int reps = argc > 3 ? atoi(argv[3]) : 3;
    int fd = open(g_path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    g_size = (long)lseek(fd, 0, SEEK_END);
    close(fd);
    for (int rep = 0; rep < reps; rep++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        pthread_t th[16];
        for (int i = 0; i < g_nt; i++) pthread_create(&th[i], NULL, work, (void *)(long)i);
        for (int i = 0; i < g_nt; i++) pthread_join(th[i], NULL);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dt = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
        fprintf(stderr, "rep=%d threads=%d bytes=%ld dt=%.3fs bw=%.2f GB/s\n", rep, g_nt,
                g_size, dt, (double)g_size / dt / 1e9);
    }
    return 0;
}
