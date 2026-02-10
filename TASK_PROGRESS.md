# E5 数据驱动模式发现实验 - 进度记录

## 当前任务：运行验证所有 E5 子实验（大乐透）

### 已完成的子实验（大乐透）

| 子实验 | AUC | 最优参数 | 状态 |
|--------|-----|---------|------|
| E5c SAX | 0.7635 | alpha=3, L=3, min_freq=5 | ✅ 完成 |
| E5a Matrix Profile | 0.7099 | m=3, k=50 | ✅ 完成 |
| E5d Shapelet | 0.7115 | shapelet_len=5, k=50 | ✅ 完成 |
| E5b Autoencoder | 0.7196 | W=10, latent=8, k=50, euclidean | ✅ 完成 |
| E5f Dictionary | 运行中 | 当前最优: W=5,K=50,α=0.5,k=50 AUC=0.7183 | ⏳ 后台运行 |
| E5e Contrastive | 运行中 | 当前: W=5,embed=16,tau=0.2 训练中 | ⏳ 后台运行 |

### 后台任务 ID
- E5f: be4a108
- E5e: bd5eef2

### 待完成步骤
1. 等待 E5f 和 E5e 完成大乐透验证
2. 全部 6 个子实验跑双色球 (shuangseqiu)
3. Step 7: 多算法融合实验
4. Step 8: 汇总报告
