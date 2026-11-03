#!/bin/sh
# mtnpu_queue.sh -- 多轮对话队列整层 NPU 测试（2026-11-03，板侧自含）。
#
# 8 会话 13 轮：S1-S3 为两轮链（记忆 / 数学上下文携带 / 知识追问），
# S4-S8 为 conduct_docs/题目.md 单轮逻辑题（A4/B3/C1/D4/E2）。
# 上下文预算：逐轮全对话 n_ids 由 host tokcount（同 tokenizer.c+bin）
# 精确预检 <= 70（tcap=75 工作点留 5 余量），超限静默回 CPU 的情况
# 在设计阶段排除；板上复核 = 日志零回退留声 + 每会话 [xnpu-prof] kicks>0。
# 每会话独立 run3-mt 进程 => 独立 [xnpu-prof]（kick 周期数/wall/matrix/
# vector 忙率）。逐轮全量重预填是 run3 设计事实，n^2 可接受。
cd /home/liangyu/deploy_qwen3 || exit 2
mkdir -p logs

sess() {
  sid=$1; shift
  echo ""
  echo "########## SESSION $sid ##########"
  { for q in "$@"; do printf '%s\n' "$q"; done; } | \
    sudo -n npurun ./run3mt.sh --threads 4 --prof \
      qwen3-0.6b-q80-gs64.bin -z qwen3_tokenizer.bin -t 0 -n 64 -i 2>&1
  echo "########## SESSION $sid END ##########"
}

sudo -n plctl claim >/dev/null || { echo CLAIM_FAIL; exit 2; }
sess S1-mem  "我叫小明，请记住我的名字，只回答好的。" "我叫什么名字？只回答名字。"
sess S2-math "3加5等于几？只回答得数。" "再加7呢？只回答得数。"
sess S3-chem "水的化学式是什么？一句话直接回答。" "它由哪两种元素组成？只回答元素。"
sess S4-orbit "一颗卫星轨道高度降低了，它的轨道周期会变长还是变短？一句话，直接给答案。"
sess S5-link "要把5MB图像经1Mbps链路传回，一次过境只有5分钟，能传完吗？一步完成，直接给结论和计算。"
sess S6-order "把步骤排成合理顺序：下传数据、拍摄目标、调整姿态对准、接收任务指令、存储图像。只排顺序，一步完成。"
sess S7-noprop "前提：这颗卫星没有推进系统。问：它能自己抬高轨道吗？一句话直接回答。"
sess S8-safemode "安全模式下只保留通信和电源。安全模式下相机能工作吗？一句话直接回答。"
sudo -n plctl release
echo "ALL_SESSIONS_DONE"
