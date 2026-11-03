#!/bin/sh
# mtnpu_queue2.sh -- conduct_docs/题目.md 补测（19 题制全量，2026-11-03）。
# 首轮队列已测 A4/B3/C1/D4/E2（mtnpu_queue.sh）；本队列补其余 14 题：
# A1-A3/A5、B1/B2/B4/B5、C2/C3、D1/D2 单轮（host tokcount 预检 n<=57，
# NPU 必中），D3 两轮（t2 n~72 边缘，超则留声回 CPU），E1 三轮
# （t2=101/t3=176 天然超 tcap=75 → CPU 留声，答案照出：多轮上下文
# vs NPU 窗口的边界实证）。列表题 -n 96 防截断；逐轮 n 由 tokcount
# 精确预检，context-full 线 440 远离。
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
sess A1 "卫星进入地影区后，太阳能板电流降为0，电池电压开始缓慢下降。这是故障吗？用一句话回答。"
sess A2 "星上温度持续升高，同时姿态角速度也在增大。请给出一个可能把两者联系起来的原因，20字以内。"
sess A3 "地面站连续三次过境都收不到信号，但星上遥测显示一切正常。列出3个可能原因，各一句话。"
sess A5 "为什么相机成像时要求卫星姿态足够稳定？一句话。"
sess B1 "电量只剩15%，下一圈还有一次对地通信和一次成像任务，只能做一个。选哪个？一句话说理由。"
sess B2 "卫星同时出现：载荷过热、电池电量低、通信链路中断。按处理优先级排序，各给一句理由。"
sess B4 "什么情况下应该主动让卫星进入安全模式，而不是继续执行任务？给3条，各一句话。"
sess B5 "如果地面指令和星上自主判断冲突，应该听谁的？一句话说明原则。"
sess C2 "卫星入轨后首次开机，哪一步最不应该最先做：展开太阳翼、检测电池状态、启动大功率载荷、建立通信。一句话。"
sess C3 "故障处置流程：发现异常、进入安全模式、通知地面、分析原因、恢复任务、隔离故障部件。指出顺序不对的步骤并说明。"
sess D1 "遥测：电池电压28V，温度25℃，姿态角速度0.1°/s，通信正常。请判断卫星状态，一句话。"
sess D2 "遥测：电池电压???V，温度25℃，姿态角速度乱码。请判断卫星状态，一句话。"
sess D3 "电量低时该关哪些设备？一句话。" "需要节电时先断开哪些负载？"
sess E1 "一颗卫星轨道高度500km，太阳能板输出100W。" "它进入地影区后，靠什么供电？一句话。" "如果日蚀时负载是80W，要持续35分钟，电池至少要多少Wh？给出计算。"
sudo -n plctl release
echo "ALL_SESSIONS_DONE"
