#!/usr/bin/env node
/* qwen3 webui 桥 —— 浏览器 ⇄ SSE/POST ⇄ run3 会话进程。
 * 方案、协议与决策记录：conduct_docs/qwen3_webui_plan.md
 *
 * 运载三种（XNPU_WEBUI_MODE）：
 *   mock   默认。spawn mock/run3mock.mjs——复刻 run3 -i 的 stdout/stderr
 *          字节语法，模型未上板期间开发/自测前端；解析状态机与真运载共用。
 *   local  本机 child_process 直跑 run3（模型文件在 host 就绪时用）。
 *   ssh    ssh2 exec 到板卡（终态；需先 npm i ssh2，板上命令联调时对齐）。
 *
 * stdout 解析状态机（契约见 mock/run3mock.mjs 头注，来源 run.c:214-397）：
 *   loading --行残段收在 '> '--> ready --POST /send--> generating
 *   generating --'\n\n> '--> ready；进程退出 → closed。
 *   截断符 '\n\n> ' = run3 生成后 printf("\n") + 下一回合 printf("\n> ")。
 *   已知局限：生成文本里出现"空行+行首 '> '"（markdown 引用块）会被误判
 *   为回合结束，后续字节落进诊断抽屉（stdout-unexpected），可见可排查。
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { StringDecoder } from 'node:string_decoder';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODE = process.env.XNPU_WEBUI_MODE || 'mock';
const PORT = +(process.env.PORT || 8899);
const HOST = process.env.XNPU_WEBUI_HOST || '127.0.0.1'; /* WSL2 会转发给 Windows 浏览器 */
const MOCK_PATH = path.join(__dirname, 'mock', 'run3mock.mjs');
const INDEX_PATH = path.join(__dirname, 'public', 'index.html');

/* ssh 模式（联调时对齐：板侧路径与 XNPU_* 环境以当时的部署纪律为准） */
const SSH_TARGET = process.env.XNPU_SSH_TARGET || 'liangyu@100.64.0.117';
const SSH_KEY = process.env.XNPU_SSH_KEY || path.join(process.env.HOME, '.ssh/id_ed25519');
/* 默认命令按 BOARD_REGRESSION 的 /home/liangyu/deploy_xiangjiang/ 布局猜的，
 * 环境变量沿用 runq_xnpu 全套——上板联调第一步就是核对这一行。 */
const SSH_CMD = process.env.XNPU_WEBUI_CMD ||
  'cd ~/deploy_xiangjiang/qwen3 2>/dev/null || cd ~/deploy_xiangjiang; ' +
  'XNPU_RESMEM_PHYS=0x60000000 XNPU_RESMEM_SIZE=0x10000000 XNPU_BASE=0x400000000 ' +
  './run3 qwen3-0.6b-q80.bin -z qwen3_tokenizer.bin';
/* local 模式默认按 host 侧 Makefile 布局（examples/qwen3/ 下 make 产物） */
const LOCAL_CMD = process.env.XNPU_WEBUI_CMD || './run3 build/qwen3-0.6b-q80.bin -z build/qwen3_tokenizer.bin';
const LOCAL_CWD = process.env.XNPU_WEBUI_CWD || path.resolve(__dirname, '..');

const DEF_PARAMS = { temperature: 0.6, topp: 0.95, topk: 20, thinking: 0 };
const MAX_INPUT = 8000; /* run3 的 prompt 缓冲是 8192 字节，留余量 */

/* ---------------- OS/WS 矩阵核面板（wsbench 桥） ----------------
 * 板上执行链：ssh exec `osws_run.sh <mode>`（自含 plctl claim/release）→
 * sudo npurun wsbench <mode> → SUMMARY 行。单 NPU 纪律：bench 直写 DTCM，
 * 与聊天会话互斥（generating 中拒绝 bench，bench 中拒绝 send）。 */
const OWS_MODES = { os: 'OS 基线', ws: 'WS 瀑布', wskeep: 'WS-keep', osback: 'OS 回切', all: '全链自检' };
const OWS_BOARD_DIR = process.env.XNPU_OWS_DIR || '~/deploy_qwen3';
const OWS_SCRIPT = process.env.XNPU_OWS_SCRIPT || 'osws_run.sh';
const OWS_TIMEOUT = 30000;
const MOCK_OWS_CYCLES = { os: 142, ws: 192, wskeep: 1027, osback: 145 }; /* 板上 10-31 实测 */

/* ---------------- 采样参数（run3 启动参数语义） ---------------- */
function sanitizeParams(p = {}) {
  const num = (v, lo, hi, dflt) => {
    const n = Number(v);
    return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : dflt;
  };
  return {
    temperature: num(p.temperature, 0, 2, DEF_PARAMS.temperature),
    topp: num(p.topp, 0, 1, DEF_PARAMS.topp),
    topk: Math.round(num(p.topk, 1, 200, DEF_PARAMS.topk)),
    thinking: p.thinking ? 1 : 0,
  };
}
function paramFlags(p) {
  return `-t ${p.temperature} -p ${p.topp} -k ${p.topk} -e ${p.thinking} -i`;
}
/* mock 走 spawn 数组 argv（不经 shell），必须拆成独立元素——整串会被
 * argv 精确匹配静默忽略，参数就回落默认值（自测抓到过一次） */
function paramArgs(p) {
  return ['-t', String(p.temperature), '-p', String(p.topp),
          '-k', String(p.topk), '-e', String(p.thinking), '-i'];
}

/* ---------------- run3 stdout 解析状态机 ---------------- */
class Run3Parser {
  constructor(sink) {
    this.sink = sink; /* {token(text), diag(line,kind), status(state)} */
    this.state = 'loading';
    this.dec = new StringDecoder('utf8'); /* 跨 chunk 组 UTF-8，token piece 不会被撕开 */
    this.buf = '';
    this.scan = 0;
    this.leadNl = false; /* generating 的首字节是回合分隔 '\n' */
  }
  push(chunk) {
    this.buf = this.buf.slice(this.scan) + this.dec.write(chunk);
    this.scan = 0;
    this.pump();
  }
  pump() {
    if (this.state === 'loading') {
      /* 整行 drain 到诊断（config 行等）；行残段恰好停在 '> ' 即就绪 */
      let nl;
      while ((nl = this.buf.indexOf('\n')) >= 0) {
        const line = this.buf.slice(0, nl);
        this.buf = this.buf.slice(nl + 1);
        if (line) this.sink.diag(line, 'stdout');
      }
      if (this.buf === '> ') {
        this.buf = '';
        this.state = 'ready';
        this.sink.status('ready');
      }
    } else if (this.state === 'ready') {
      /* 就绪态不应再有 stdout：要么是假截断的余文，要么是异常，都送诊断可见 */
      if (this.buf) {
        this.sink.diag(this.buf, 'stdout-unexpected');
        this.buf = '';
      }
    } else if (this.state === 'generating') {
      if (this.leadNl) {
        if (this.buf.startsWith('\n')) this.buf = this.buf.slice(1);
        this.leadNl = false;
      }
      const i = this.buf.indexOf('\n\n> ', this.scan);
      if (i >= 0) {
        this.emit(this.buf.slice(this.scan, i));
        this.buf = '';
        this.scan = 0;
        this.state = 'ready';
        this.sink.status('ready');
      } else {
        /* 滞留末尾 ≤3 字节（'\n\n>' 是截断符前缀），防跨 chunk 撕裂 */
        const safe = Math.max(this.scan, this.buf.length - 3);
        if (safe > this.scan) this.emit(this.buf.slice(this.scan, safe));
        this.scan = safe;
      }
    }
  }
  emit(t) {
    if (t) this.sink.token(t);
  }
  markSent() {
    this.state = 'generating';
    this.leadNl = true;
    this.buf = '';
    this.scan = 0;
    this.sink.status('generating');
  }
  flush() {
    /* 进程退出：generating 中的残文补发（context-full 就是这条路——一个
     * 字节都不会有，状态直接落 closed，前端靠诊断抽屉解释） */
    const rest = this.buf.slice(this.scan) + this.dec.end();
    this.buf = '';
    this.scan = 0;
    if (this.state === 'generating') this.emit(rest);
    this.state = 'closed';
    this.sink.status('closed');
  }
}

/* stderr 逐行化（诊断事件按行发） */
class LineSplitter {
  constructor(onLine) {
    this.onLine = onLine;
    this.dec = new StringDecoder('utf8');
    this.buf = '';
  }
  push(chunk) {
    this.buf += this.dec.write(chunk);
    let nl;
    while ((nl = this.buf.indexOf('\n')) >= 0) {
      const line = this.buf.slice(0, nl).replace(/\r$/, '');
      this.buf = this.buf.slice(nl + 1);
      if (line) this.onLine(line);
    }
  }
  flush() {
    const rest = (this.buf + this.dec.end()).replace(/\r$/, '');
    this.buf = '';
    if (rest) this.onLine(rest);
  }
}

/* ---------------- 会话：一条常驻进程 = 一个对话 ----------------
 * 运载差异被抹平成统一 IO 把手 {stdin, stdout, stderr, onExit, kill}：
 * child_process 的 stdout 是独立流，ssh2 的 exec stream 本身就是双工
 * （stderr 挂在 .stderr 上）——这里统一掉，解析状态机不感知运载。 */
class Session {
  constructor() {
    this.state = 'closed';
    this.params = { ...DEF_PARAMS };
    this.parser = null;
    this.proc = null;
    this.clients = new Set();
    this.startedAt = 0;
    this.epoch = 0;
  }

  broadcast(type, data) {
    const payload = `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
    for (const res of this.clients) res.write(payload);
  }
  diag(line, kind) { this.broadcast('diag', { line, kind }); }
  token(t) { this.broadcast('token', { text: t }); }
  status(state) {
    this.state = state;
    this.broadcast('status', { state, mode: MODE, params: this.params });
  }

  async start(params) {
    const epoch = ++this.epoch;
    this.params = sanitizeParams(params);
    await this.stop();
    if (epoch !== this.epoch) return; /* 期间又来了次 restart，让最新者赢 */
    this.parser = new Run3Parser({
      token: (t) => this.token(t),
      diag: (l, k) => this.diag(l, k),
      status: (s) => this.status(s),
    });
    this.status('loading');
    try {
      if (MODE === 'mock') this.spawnMock();
      else if (MODE === 'local') this.spawnLocal();
      else if (MODE === 'ssh') await this.spawnSsh();
      else throw new Error(`unknown XNPU_WEBUI_MODE: ${MODE}`);
      this.startedAt = Date.now();
    } catch (e) {
      this.diag(String((e && e.message) || e), 'error');
      this.status('error');
    }
  }

  /* 温退：stdin EOF → run3 的 fgets 返回 NULL → 自然退出；3s 后硬杀兜底 */
  async stop() {
    const proc = this.proc;
    this.proc = null;
    if (!proc) return;
    await new Promise((resolve) => {
      const t1 = setTimeout(() => { try { proc.kill(); } catch { /* 已退 */ } }, 3000);
      const t2 = setTimeout(resolve, 5000);
      proc.notifyExit = () => { clearTimeout(t1); clearTimeout(t2); resolve(); };
      try { proc.stdin.end(); } catch { /* 已断 */ }
    });
    try { proc.kill(); } catch { /* 已退：kill 已退进程是无害 no-op */ }
  }

  wire(io) {
    const stderr = new LineSplitter((l) => this.diag(l, 'stderr'));
    io.stdout.on('data', (c) => this.parser && this.parser.push(c));
    if (io.stderr) io.stderr.on('data', (c) => stderr.push(c));
    const proc = { ...io, notifyExit: null };
    io.onExit(() => {
      stderr.flush();
      if (this.proc === proc) { /* 已被 restart 换下时不动全局状态 */
        this.proc = null;
        this.parser && this.parser.flush();
      }
      if (proc.notifyExit) { const f = proc.notifyExit; proc.notifyExit = null; f(); }
    });
    this.proc = proc;
  }

  spawnMock() {
    const child = spawn(process.execPath, [
      MOCK_PATH, 'mock-ckpt.bin', '-z', 'mock-tokenizer.bin', ...paramArgs(this.params),
    ], { cwd: __dirname, stdio: ['pipe', 'pipe', 'pipe'] });
    this.wire({
      stdin: child.stdin, stdout: child.stdout, stderr: child.stderr,
      onExit: (cb) => child.on('close', cb), kill: () => child.kill('SIGKILL'),
    });
  }

  spawnLocal() {
    const cmd = `${LOCAL_CMD} ${paramFlags(this.params)}`;
    const child = spawn('/bin/sh', ['-c', cmd], { cwd: LOCAL_CWD });
    this.wire({
      stdin: child.stdin, stdout: child.stdout, stderr: child.stderr,
      onExit: (cb) => child.on('close', cb), kill: () => child.kill('SIGKILL'),
    });
    this.diag(`$ (${LOCAL_CWD}) ${cmd}`, 'log');
  }

  async spawnSsh() {
    let Client;
    try {
      const mod = await import('ssh2');
      Client = mod.Client || (mod.default && mod.default.Client);
    } catch {
      throw new Error('ssh 模式需要 ssh2 依赖：cd examples/qwen3/webui && npm install');
    }
    const [username, host] = SSH_TARGET.includes('@')
      ? SSH_TARGET.split('@') : [process.env.USER || 'liangyu', SSH_TARGET];
    const conn = new Client();
    const cmd = `${SSH_CMD} ${paramFlags(this.params)}`;

    await new Promise((resolve, reject) => {
      conn.on('ready', resolve).on('error', reject);
      const opts = { host, username, readyTimeout: 15000, keepaliveInterval: 15000 };
      if (fs.existsSync(SSH_KEY)) opts.privateKey = fs.readFileSync(SSH_KEY);
      if (process.env.SSH_AUTH_SOCK) opts.agent = process.env.SSH_AUTH_SOCK;
      conn.connect(opts);
    });

    const stream = await new Promise((resolve, reject) =>
      conn.exec(cmd, { pty: false }, (err, s) => (err ? reject(err) : resolve(s))));

    this.diag(`$ ssh ${username}@${host} ${cmd}`, 'log');
    /* 每个会话一条新连接，重启即整链拆掉——简单可靠，会话本就低频。
     * conn 层出错/断开兜底走 stream.close()，让 wire 的统一出口收尾。 */
    conn.on('error', () => { try { stream.close(); } catch { /* 已关 */ } });
    conn.on('close', () => { try { stream.close(); } catch { /* 已关 */ } });
    this.wire({
      stdin: stream, stdout: stream, stderr: stream.stderr,
      onExit: (cb) => stream.on('close', cb), kill: () => conn.end(),
    });
  }

  send(text) {
    if (benchBusy)
      return { code: 409, error: 'OS/WS bench 进行中，稍候再发' };
    if (this.state !== 'ready' || !this.proc)
      return { code: 409, error: `当前状态 ${this.state}，不可发送（就绪后才行）` };
    this.broadcast('user', { text });
    this.parser.markSent();
    this.proc.stdin.write(text + '\n');
    return { code: 200 };
  }
}

/* ---------------- OS/WS bench 执行 ---------------- */
let benchBusy = false;

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

/* ssh 一次性 exec（自含连接），返回 {code, out, err}；与 spawnSsh 共用
 * 目标/密钥解析 */
async function sshExec(cmd, timeoutMs = OWS_TIMEOUT) {
  const mod = await import('ssh2');
  const Client = mod.Client || (mod.default && mod.default.Client);
  const [username, host] = SSH_TARGET.includes('@')
    ? SSH_TARGET.split('@') : [process.env.USER || 'liangyu', SSH_TARGET];
  const conn = new Client();
  return new Promise((resolve, reject) => {
    let out = '', err = '', code = null, done = false;
    const finish = (e) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      try { conn.end(); } catch { /* 已断 */ }
      e ? reject(e) : resolve({ code, out, err });
    };
    const timer = setTimeout(() => finish(new Error(`板侧执行超时（>${timeoutMs}ms）`)), timeoutMs);
    conn.on('ready', () => {
      conn.exec(cmd, { pty: false }, (e, s) => {
        if (e) { finish(e); return; }
        s.on('data', (c) => { out += c; });
        s.stderr.on('data', (c) => { err += c; });
        s.on('close', (c) => { code = c; finish(); });
      });
    });
    conn.on('error', finish);
    const opts = { host, username, readyTimeout: 15000, keepaliveInterval: 15000 };
    if (fs.existsSync(SSH_KEY)) opts.privateKey = fs.readFileSync(SSH_KEY);
    if (process.env.SSH_AUTH_SOCK) opts.agent = process.env.SSH_AUTH_SOCK;
    conn.connect(opts);
  });
}

/* 解析 wsbench stdout：SUMMARY 行 + OSWS_RC */
function parseOws(out, exitCode) {
  const rows = [...out.matchAll(/SUMMARY mode=(\S+) cycles=(\d+) pass=(\d)/g)]
    .map((m) => ({ mode: m[1], cycles: +m[2], pass: m[3] === '1' }));
  const pass = exitCode === 0 && rows.length > 0 && rows.every((r) => r.pass);
  const cycles = rows.length === 1 ? rows[0].cycles : null;
  return { rows, cycles, pass };
}

async function runOws(mode) {
  if (benchBusy) return { code: 409, error: '上一个 bench 还在跑' };
  if (sess.state === 'generating') return { code: 409, error: '对话生成中（单 NPU，bench 会踩 DTCM）' };

  if (MODE === 'local')
    return { code: 501, error: 'local 运载没有 NPU——bench 只在 ssh（板卡）模式可用，mock 模式返回仿真数字' };

  benchBusy = true;
  sess.broadcast('osws', { phase: 'start', mode, label: OWS_MODES[mode] });
  try {
    let result;
    if (MODE === 'mock') {
      await sleep(400); /* 走一遍真实前端节奏 */
      const rows = mode === 'all'
        ? Object.entries(MOCK_OWS_CYCLES).map(([m, cycles]) => ({ mode: m, cycles, pass: true }))
        : [{ mode, cycles: MOCK_OWS_CYCLES[mode], pass: true }];
      result = { mode, label: OWS_MODES[mode], rows,
                 cycles: rows.length === 1 ? rows[0].cycles : null,
                 pass: true, simulated: true };
    } else {
      const r = await sshExec(`cd ${OWS_BOARD_DIR} && ./${OWS_SCRIPT} ${mode}`);
      const tail = r.out.trim().split('\n').slice(-12).join('\n');
      if (!r.out.includes('SUMMARY'))
        throw new Error(`板侧输出无 SUMMARY 行（exit=${r.code}）: ${tail || r.err}`);
      const p = parseOws(r.out, r.code);
      result = { mode, label: OWS_MODES[mode], ...p, raw: tail };
    }
    sess.broadcast('osws', { phase: 'done', ...result });
    return { code: 200, ...result };
  } catch (e) {
    const msg = String((e && e.message) || e);
    sess.broadcast('osws', { phase: 'error', mode, label: OWS_MODES[mode], error: msg });
    return { code: 502, error: msg };
  } finally {
    benchBusy = false;
  }
}

/* ---------------- HTTP：静态页 + SSE + 三个 JSON 口 ---------------- */
const sess = new Session();

function readJson(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (c) => {
      body += c;
      if (body.length > 16384) { reject(new Error('body too large')); req.destroy(); }
    });
    req.on('end', () => {
      if (!body) return resolve({});
      try { resolve(JSON.parse(body)); } catch { reject(new Error('invalid json')); }
    });
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  const u = new URL(req.url, 'http://localhost');

  if (req.method === 'GET' && (u.pathname === '/' || u.pathname === '/index.html')) {
    try {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(fs.readFileSync(INDEX_PATH));
    } catch { res.writeHead(500); res.end('index.html missing'); }
    return;
  }

  if (req.method === 'GET' && u.pathname === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({
      mode: MODE, state: sess.state, params: sess.params, benchBusy,
      owsModes: OWS_MODES,
      uptimeSec: sess.startedAt ? Math.round((Date.now() - sess.startedAt) / 1000) : 0,
      clients: sess.clients.size,
    }));
    return;
  }

  if (req.method === 'GET' && u.pathname === '/events') {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    });
    res.write('retry: 3000\n\n');
    /* 新客户端先给快照：状态/模式/参数（无历史重放，气泡只是镜像） */
    res.write(`event: status\ndata: ${JSON.stringify({
      state: sess.state, mode: MODE, params: sess.params,
    })}\n\n`);
    sess.clients.add(res);
    req.on('close', () => sess.clients.delete(res));
    return;
  }

  if (req.method === 'POST' && u.pathname === '/send') {
    let body;
    try { body = await readJson(req); } catch (e) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: e.message }));
      return;
    }
    /* run3 按行读：多行压成单行（内部换行折叠为空格），首尾去空白 */
    const text = String(body.text || '').replace(/\s*\n+\s*/g, ' ').trim();
    if (!text) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: '空消息' }));
      return;
    }
    if ([...text].length > MAX_INPUT) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: `超过单条上限 ${MAX_INPUT} 字` }));
      return;
    }
    const r = sess.send(text);
    res.writeHead(r.code, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(r.code === 200 ? { ok: true } : { error: r.error }));
    return;
  }

  if (req.method === 'POST' && u.pathname === '/restart') {
    let body;
    try { body = await readJson(req); } catch (e) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: e.message }));
      return;
    }
    await sess.start((body && body.params) || body);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true, params: sess.params }));
    return;
  }

  /* OS/WS 矩阵核面板：mode ∈ OWS_MODES；执行经 runOws（互斥 + SSE 广播），
   * 结果同时回 HTTP 与 osws 事件（多标签页看到同一份历史）。 */
  if (req.method === 'POST' && u.pathname === '/osws/run') {
    let body;
    try { body = await readJson(req); } catch (e) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: e.message }));
      return;
    }
    const mode = body && body.mode;
    if (!OWS_MODES[mode]) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: `mode 必须是 ${Object.keys(OWS_MODES).join('/')}` }));
      return;
    }
    const r = await runOws(mode);
    res.writeHead(r.code, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(r.code === 200 ? r : { error: r.error }));
    return;
  }

  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'not found' }));
});

/* prefill 静默期保活：15s 心跳注释行 */
setInterval(() => {
  for (const res of sess.clients) res.write(': ping\n\n');
}, 15000).unref();

server.listen(PORT, HOST, () => {
  console.log(`[webui] http://${HOST}:${PORT}  mode=${MODE}`);
  if (MODE === 'ssh' && !fs.existsSync(SSH_KEY) && !process.env.SSH_AUTH_SOCK)
    console.log('[webui] 注意：未找到 SSH key，ssh 模式可能连不上板');
  sess.start({}); /* 起服务即拉起会话，开浏览器就能看到 loading→ready */
});

process.on('SIGINT', () => {
  server.close();
  sess.epoch++;
  if (sess.proc) { try { sess.proc.kill(); } catch { /* noop */ } }
  process.exit(0);
});
