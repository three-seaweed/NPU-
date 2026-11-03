#!/bin/sh
# mtnpu_queue3.sh -- 追问轮 NPU 化补跑（2026-11-03）。
# 队列1 的 S1/S3 与队列2 的 D3 第 2 轮因模型回复超预算回退 CPU；本队列
# 用实测回复（贪心 -t 0 确定性复现）精确重算预算并改写追问长度：
#   S1' n=36/70  S3' n=33/69  D3' n=33/67  —— 六轮全部 <=70，全 NPU。
# E1-t2/t3（n=141/195）不在本队列：三连问链物理上超 tcap=75 工作点窗口
# （B=1 最高 T~76，DTCM/KV 镜像预算），压缩措辞后仍 >=80，保持原样作
# 边界实证；唯一根治是 plan §7.3 分块注意力重构。
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
sess S1r "我叫小明，请记住我的名字，只回答好的。" "我叫什么？只答名字。"
sess S3r "水的化学式是什么？一句话直接回答。" "由哪两种元素组成？只答元素。"
sess D3r "电量低时该关哪些设备？一句话。" "节电时先断开哪些负载？"
sudo -n plctl release
echo "ALL_SESSIONS_DONE"
