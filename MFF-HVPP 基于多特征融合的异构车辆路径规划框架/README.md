# MFF-HVPP: 基于多特征融合的异构车辆路径规划框架

##  倪静 谭云洁
##  上海理工大学

本代码库为谭云洁毕业论文，实现了论文《基于多特征融合的异构车辆路径规划问题》中提出的MFF-HVPP框架。该框架通过深度强化学习，结合多特征融合编码器和基于注意力机制的通道特征扩展模块（CAE），实现了异构车辆路径规划的高效求解，支持min-max（均衡负载）和min-sum（总成本）双目标优化。

## 核心创新点
- **多特征融合编码器**：整合节点位置、需求特征与车辆状态
- **基于注意力机制的通道特征扩展模块CAE**：通过通道维度特征扩展解决传统注意力机制的梯度消失问题
- **分层解码策略**：解耦车辆选择与节点选择的序列决策过程
- **动态状态更新**：基于LSTM的车辆状态实时追踪机制
- **双模态奖励函数**：同时适配min-max和min-sum优化目标

## 主要性能
| 场景               | 客户节点 | Gap   | 计算时间 | 优化目标      |
|--------------------|----------|-------|----------|-------------|
| Min-Max HCVRP      | 120      | 1.31% | 8.66s    | 最大行程时间  |
| Min-Sum HCVRP      | 160      | 1.07% | 13.77s   | 总行程时间    |

## 环境依赖
- Python >= 3.8
- PyTorch >= 1.10
- numpy >= 1.21
- tensorboard_logger
- tqdm

## 快速开始
### 安装
```bash
pip install -r requirements.txt
```

## 训练示例-训练120节点min-max场景
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --problem hcvrp \
    --graph_size 120 \
    --optim_target minmax \
    --n_vehicles 5 \
    --embed_dim 256 \
    --trans_cfe_layers 4 \
    --batch_size 512 \
    --max_epoch 200

# 评估160节点min-sum场景
python evaluate.py \
    --ckpt pretrained/min_sum_160.pt \
    --val_dataset datasets/hcvrp_160_test.pkl \
    --val_m 8 \
    --temperature 0.1 \
    --n_vehicles 7

#代码结构
```bash
MFF-HVPP/
├── core/
│   ├── encoder/            # 多特征融合编码器
│   │   ├── trans_cfe.py    # TransCAE模块
│   │   └── cpe.py         # 循环位置编码
│   ├── decoder/            # 分层解码器
│   └── agent.py            # 强化学习智能体
├── datasets/               # 预生成数据集
├── pretrained/             # 预训练模型
├── utils/
│   ├── vrplib.py           # VRP数据工具
│   └── visualize.py        # 可视化工具
└── train.py                # 主训练脚本
```

###关键特性
###1. 异构车辆配置
####支持不同容量/速度的车辆混合调度：
```bash
vehicle_config = [
    {'capacity': 20, 'speed': 1.2},  # 类型A
    {'capacity': 35, 'speed': 1.0},  # 类型B
    {'capacity': 50, 'speed': 0.8}   # 类型C
]
```
####2. 动态需求支持
```bash
节点特征维度 [x, y, demand, dynamic_flag]
node_feats = torch.tensor([
    [0.5, 0.3, 4.0, 0],   # 静态需求
    [0.2, 0.7, 2.5, 1],   # 动态需求
])
```

####3. 多目标切换
通过--optim_target参数指定优化目标：

minmax: 最小化最大车辆行程时间

minsum: 最小化总行程时间


##引用
若本项目对您的研究有帮助，请引用我们的论文：
```bash
@article{tan2025mff,
  title={基于多特征融合的异构车辆路径规划问题},
  author={谭云洁 and 倪静},
  journal={建模与仿真},
  volume={14},
  number={3},
  pages={694--715},
  year={2025}
}
