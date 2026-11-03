#!/bin/sh
# mtnpu_queue4.sh -- 追问轮 NPU 化终版（2026-11-03）。
# 教训链：队列1/2 的三条件第 2 轮回复超预算回退；队列3 按 tcap=75 重排仍
# 回退——B=2 分批工作点窗口实测只到 ~64（63 跑通、67 需 B=1 被 minb 拦）。
# 本队列按 n2<=63 重排（B=2 窗口内）+ 输出约束钉死回复长度：
#   S1'' n=37/60  S3'' n=34/59~60  D3'' n=40/62（r1<=5 由"只答一项"保证）
cd /home/liangyu/deploy_qwen3 || exit 2
mkdir -p logs

sess() {
  sid=$1; shift
  echo ""
  echo "########## SESSION $sid ##########"
  { for q in "$@"; do printf '%s\n' "$q"; done; } | \
    sudo -n npurun ./run3mt.sh --threads 4 --prof \
      qwen3-0.6b-q80-gs64.bin -z qwen3_tokenizer.bin -t 0 -n 96 -i 2>&1
  echo "########## SESSION $sid END ##########"
}

sudo -n plctl claim >/dev/null || { echo CLAIM_FAIL; exit 2; }
sess S1f "我叫小明，请记住我的名字，只回复：好的。" "我叫什么名字？"
sess S3f "水的化学式是什么？只答化学式。" "含哪两种元素？"
sess D3f "电量低时该关哪些设备？只答一项。" "节电时先断哪些？"
sudo -n plctl release
echo "ALL_SESSIONS_DONE"
