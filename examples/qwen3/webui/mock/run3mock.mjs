#!/usr/bin/env node
/* run3 -i 的字节语法复刻 mock —— webui 开发期的板上模型替身。
 *
 * 契约（与 run.c:214-397 逐字节对齐，server.js 的解析状态机按此写）：
 *   stdout: 启动 "config: ...\n" → "\n> "
 *           [收到一行] → "\n" + 逐 piece 即时写的回复 + "\n" + "\n> " …
 *   stderr: 偶发诊断行；上下文打满时 "(context full)\n" 后直接退出——
 *           注意此时 stdout 一个字节都不再写（真 run3 在 printf("\n") 前 break）。
 *   stdin : 每行一条用户消息；"exit"/"/quit" 或 EOF 退出。
 *   无 PTY、无回显；启动参数 -t/-p/-k/-e 与真 run3 同名同义。
 *
 * 测试钩子（真实模型没有，只给前端联调用）：
 *   /edge  回复里含空行+行首 "> "——触发解析器已知局限（假截断），见 plan §3
 *   /slow  后续 token 间隔放大到 ~800ms——测状态条/心跳
 */
import readline from 'node:readline';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ---- argv（真 run3 的子集） ---- */
let temperature = 0.6, topp = 0.95, topk = 20, thinking = 0;
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i];
  if (a === '-t') temperature = +process.argv[++i];
  else if (a === '-p') topp = +process.argv[++i];
  else if (a === '-k') topk = +process.argv[++i];
  else if (a === '-e') thinking = +process.argv[++i];
}

const REPLIES = [
  '你好！我是运行在 ZCU102 上的 mock 模型——真正的 Qwen3-0.6B 还在联调。等它上板，这里会换成逐 token 的真实流式回复。',
  '端侧问答的关键路径：prefill 走 NPU int8 GEMM，decode 目前在 CPU（A53）上。所以你会看到首字前的停顿比字与字之间长得多——这是真实的硬件行为，不是卡了。',
  '给你列一下这块 NPU 的部件：\n\n- DTCM 512KB，32 bank 真双口\n- int8 16×16 脉动阵列\n- 8-lane bf16 向量核\n- RV32I 标量核发命令\n\n三条数据流异步执行，靠 DTCM range token 保序。',
  'Sure, English works too. I am only a mock, but the streaming you are watching—token by token, with real gaps—is exactly how the real model will arrive on the board.',
  '（mock 词穷了，换个话题吧。比如问我 NPU 的脉动阵列是怎么回事。）',
];
let replyIdx = 0;
let turn = 0;

function makeReply(text) {
  if (text === '/edge')
    return '行首引用边界测试：\n\n> 这一行以 "> " 开头\n\n按 plan §3 的已知局限，解析器会在这里把回合误判为结束——真实模型几乎不会这样输出。';
  if (temperature < 0.2)
    return `（温度 0 确定性档）你输入的是：「${text}」。采样参数是 run3 的启动参数，改参数要重启会话——这正是"重启会话"按钮的语义。`;
  const r = REPLIES[replyIdx++ % REPLIES.length];
  return thinking
    ? `<think>\n用户说：${text}\n（mock 思考：从 canned 列表里轮一个回复即可。）\n</think>\n\n${r}`
    : r;
}

/* 按 UTF-16 码点切 2~4 字符的 piece，模拟 token 粒度（不撕裂代理对） */
function pieces(s) {
  const chars = [...s];
  const out = [];
  for (let i = 0; i < chars.length; ) {
    const n = 2 + Math.floor(Math.random() * 3);
    out.push(chars.slice(i, i + n).join(''));
    i += n;
  }
  return out;
}

async function onLine(line) {
  const text = line.trim();
  if (!text) return; /* run3 对空行 continue，不消耗回合 */
  if (text === 'exit' || text === '/quit') process.exit(0);

  if (++turn > 30) {
    /* 上下文打满：stderr 一行即退出，stdout 不再有任何字节 */
    process.stderr.write('(context full)\n');
    await sleep(50);
    process.exit(0);
  }

  process.stdout.write('\n'); /* 回合分隔（run.c: "printf("\\n")"） */
  const reply = makeReply(text);
  for (const [i, p] of pieces(reply).entries()) {
    process.stdout.write(p);
    /* 首 piece 前 sim prefill 静默；之后 25~60ms —— 板上 decode ~1 tok/s 的缩时版 */
    await sleep(i === 0 ? 400 + Math.random() * 600 : 25 + Math.random() * 35);
  }
  process.stdout.write('\n');   /* printf("\n") */
  process.stdout.write('\n> '); /* 下一回合提示符 */

  if (turn % 5 === 0)
    process.stderr.write(
      `xnpu(mock): prefill ${120 + turn * 37} tok in ${(0.3 + Math.random() * 0.4).toFixed(2)}s, npu_busy=${(88 + Math.random() * 10).toFixed(1)}%\n`,
    );
}

/* ---- 启动序列 ---- */
await sleep(1000 + Math.random() * 800); /* 640MB mmap 加载期的可见 loading */
process.stdout.write(
  `config: dim=1024 hidden=3072 L=28 heads=16 kv=8 hd=64 vocab=151936 seq=40960 gs=8 theta=1e6 eps=1e-5 (mock t=${temperature} p=${topp} k=${topk} e=${thinking})\n`,
);
process.stderr.write('mmap(mock): qwen3-0.6b-q80.bin 640MB, page-cache warm\n');
process.stdout.write('\n> ');

const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on('line', onLine);
rl.on('close', () => process.exit(0));
