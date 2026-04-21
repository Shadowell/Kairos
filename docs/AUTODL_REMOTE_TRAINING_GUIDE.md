# AutoDL 远端训练与回传指南

> 本地 Mac 没有 CUDA 时的标准流程：在本地完成数据采集 + 数据集准备，到 AutoDL GPU 实例上跑微调，再把 checkpoint 拉回本地做回测。
>
> 本文档对齐的代码版本已经包含以下关键修复（确保你用的是最新代码）：
>
> - `kairos/data/collect.py`：CSI300 成分股列名修复、东财失败自动降级到腾讯/新浪
> - `kairos/data/prepare_dataset.py`：`amount` 缺失用 `close*volume` 近似、过滤 macOS `._xxx.parquet` 伴生文件
> - `kairos/training/train_predictor.py`：支持 CPU 模式、`KAIROS_SMOKE=1` 快速冒烟
> - `kairos/training/config.py`：`dataset_path` 指向 `finetune/data/processed_datasets`
> - `kairos/utils/training_utils.py`：`setup_ddp` 自动在无 CUDA 时切换 `gloo` 后端

---

## 0. 路线概览

```
[本地 Mac]                                    [AutoDL GPU 实例]
 1. 采集 K 线 (raw/)         ──打包 scp──▶     4. 安装依赖
 2. 采集指数 (raw/index_)    ──打包 scp──▶     5. 生成 pkl 数据集
 3. （可选）本地 smoke test                    6. 训练 predictor（重头戏）
                              ◀──scp拉回──     7. 打包 artifacts
 8. 本地回测看 IC
```

**时间/费用估算**（RTX 5090 32GB / ¥2.93/h）：

| 步骤 | 时间 | 费用 |
|---|---|---|
| 创建实例 + 传数据 | ~10 min | - |
| 环境准备 + pip install | ~5 min | ¥0.3 |
| kairos-prepare | ~30 s | - |
| 训练 30 epoch | **~1.5–2 h** | **¥5–6** |
| 打包下载 | ~5 min | - |
| **合计** | **~2.5 h** | **≈ ¥8** |

---

## 1. 创建 AutoDL 实例

### 1.1 配置选择

| 项 | 推荐 | 说明 |
|---|---|---|
| 卡型 | **RTX 5090 32GB × 1** | 25.3M 参数的 Kronos-small，单卡足够 |
| CPU/内存 | 25 核 / 90GB | DataLoader num_workers 可开大 |
| 数据盘 | 免费 50GB | 总数据 + HF cache + checkpoint 约 10GB |
| 计费 | **按量计费** | 跑完立刻关机最省 |

### 1.2 镜像选择（**最关键**）

RTX 5090 是 **Blackwell 架构（sm_120）**，需要 **CUDA 12.8+ / PyTorch 2.7+**，老镜像跑不起来（会报 `no kernel image available`）。

**优先顺序**：

1. `PyTorch 2.7.0 / Python 3.12 / CUDA 12.8` ← 首选
2. `PyTorch 2.5.1 / Python 3.12 / CUDA 12.4` ← 次选，进机器后需升级：
   ```bash
   pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
   ```

### 1.3 SSH 信息

实例启动后在 AutoDL 控制台"快捷工具"里看到：

- SSH 命令：`ssh -p <PORT> root@<HOST>`
- 密码：在"实例详情"里复制

本文后续都用 `<HOST>` 和 `<PORT>` 占位，替换成你自己的。

---

## 2. 本地准备数据包（在 Mac 上）

### 2.1 采集 K 线和指数（如果还没做）

```bash
cd /Users/jie.feng/wlb/Kairos
source .venv/bin/activate

# 300 只成分股 × 2018-2026 日线，约 12 分钟
kairos-collect --universe csi300 --freq daily \
    --start 2018-01-01 --end 2026-04-17 \
    --out ./raw/daily --workers 1

# CSI300 指数日线（手动采集，脚本里没暴露）
python -c "
import akshare as ak, pandas as pd
from pathlib import Path
df = ak.stock_zh_index_daily(symbol='sh000300')
df['date'] = pd.to_datetime(df['date'])
df = df.rename(columns={'date': 'datetime'})
df = df[(df['datetime'] >= '2018-01-01') & (df['datetime'] <= '2026-04-17')].reset_index(drop=True)
for col in ['amount', 'turnover', 'pct_chg', 'vwap']:
    df[col] = pd.NA
df['vwap'] = df['close']
df = df[['datetime','open','high','low','close','volume','amount','turnover','pct_chg','vwap']]
out = Path('raw/index_daily/000300.parquet')
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, index=False)
print(f'saved {out}: {len(df)} rows')
"
```

### 2.2 打包代码 + 数据

**关键**：打包时必须屏蔽 macOS 元数据文件 `._xxx`，否则 `kairos-prepare` 会把它们当 parquet 读，报 `Parquet magic bytes not found`。

```bash
cd /Users/jie.feng/wlb

# 代码包（几 MB）
COPYFILE_DISABLE=1 tar --no-xattrs -czf /tmp/kairos_code.tar.gz \
    --exclude='Kairos/.venv' \
    --exclude='Kairos/__pycache__' \
    --exclude='Kairos/**/__pycache__' \
    --exclude='Kairos/.pytest_cache' \
    --exclude='Kairos/*.egg-info' \
    --exclude='Kairos/.git' \
    --exclude='Kairos/raw' \
    --exclude='Kairos/artifacts' \
    --exclude='._*' --exclude='.DS_Store' \
    Kairos/

# 数据包（~20 MB）
cd /Users/jie.feng/wlb/Kairos
COPYFILE_DISABLE=1 tar --no-xattrs -czf /tmp/kairos_raw.tar.gz \
    --exclude='._*' --exclude='.DS_Store' \
    raw/

ls -lh /tmp/kairos_code.tar.gz /tmp/kairos_raw.tar.gz
```

### 2.3 上传到 AutoDL

```bash
scp -P <PORT> /tmp/kairos_code.tar.gz /tmp/kairos_raw.tar.gz \
    root@<HOST>:/root/autodl-tmp/
```

> 网不稳时用 `rsync` 支持断点续传：
> ```bash
> rsync -avzP -e "ssh -p <PORT>" \
>     /tmp/kairos_code.tar.gz /tmp/kairos_raw.tar.gz \
>     root@<HOST>:/root/autodl-tmp/
> ```

---

## 3. AutoDL 环境准备

SSH 登进去：

```bash
ssh -p <PORT> root@<HOST>
```

### 3.1 检查 GPU 和 CUDA

```bash
nvidia-smi                              # 应看到 RTX 5090 32GB, CUDA 13.0
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望: 2.7.0+cu128 True NVIDIA GeForce RTX 5090
```

如果 `torch.cuda.is_available()` 是 `False` 或后续训练报 `sm_120 not compiled`：

```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```

### 3.2 配置 HuggingFace 镜像（国内必做）

```bash
cat >> ~/.bashrc << 'EOF'
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
EOF
source ~/.bashrc
mkdir -p /root/autodl-tmp/hf_cache
```

> `HF_HOME` 必须放在数据盘 `/root/autodl-tmp/`，不要放系统盘（容易爆满且关机可能丢）。

#### ⚠️ 重要：不要同时开 AutoDL 学术加速和 HF 镜像

如果你之前执行过 `source /etc/network_turbo`，会设置 `http_proxy/https_proxy` 指向 AutoDL 的境外代理。这个代理**会把你对 `hf-mirror.com` 的请求也绕到境外**，导致下载卡死（国内镜像被代理成了境外请求再回国内）。

**诊断**：

```bash
env | grep -i proxy
# 如果看到 http_proxy=http://10.x.x.x:xxxxx 就是中招了
```

**修复**（两种策略二选一）：

策略 A：用 hf-mirror 就关掉学术加速（推荐）

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
env | grep -i proxy    # 应无输出
```

策略 B：保留代理，把 hf-mirror 加进 no_proxy 白名单

```bash
export no_proxy="$no_proxy,hf-mirror.com"
export NO_PROXY="$no_proxy"
```

**规则**：国内镜像（hf-mirror、modelscope、清华/阿里 pypi）**不要走代理**；只有境外原站（`huggingface.co`、`github.com`）才需要代理。

### 3.3 解压代码和数据

```bash
cd /root/autodl-tmp
tar xzf kairos_code.tar.gz
cd Kairos
tar xzf ../kairos_raw.tar.gz

# 验证
ls raw/daily | grep -v '^\._' | wc -l    # 应为 300
ls raw/index_daily                        # 应有 000300.parquet
```

如果数字不对，或者 `ls | grep '^\._'` 显示有 `._xxx` 文件：

```bash
find raw -name '._*' -delete
```

### 3.4 安装依赖

```bash
cd /root/autodl-tmp/Kairos
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e '.[train]'
pip install 'numpy<2'        # 避免 numpy 2.x 与 torch 预编译 wheel 不兼容
```

> 如果 `pip install` 到 PyPI 慢，用清华镜像：
> ```bash
> pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

## 4. 生成训练数据集

```bash
cd /root/autodl-tmp/Kairos
source .venv/bin/activate

kairos-prepare \
    --raw ./raw/daily \
    --raw-index ./raw/index_daily/000300.parquet \
    --train 2018-01-01:2023-12-31 \
    --val   2024-01-01:2024-12-31 \
    --test  2025-01-01:2026-04-17 \
    --out   ./finetune/data/processed_datasets
```

预期输出（约 15 秒）：

```
已保存 train_data.pkl: 298 symbols
已保存 val_data.pkl: 299 symbols
已保存 test_data.pkl: 299 symbols
已保存 exog_train.pkl / exog_val.pkl / exog_test.pkl
```

---

## 5. 训练 predictor（主菜）

### 5.1 调整超参（可选但推荐）

针对 AutoDL 机器的 25 核 CPU 和 32GB 显存，微调 `kairos/training/config.py`：

```python
num_workers: int = 8          # 原 2 → 8，喂数据更快
batch_size: int = 64          # 原 50 → 64，利用 32GB 显存
epochs: int = 30              # 保持，约 1.5-2h
```

### 5.2 启动训练（推荐用 tmux）

`tmux` 能让你随时断开 SSH 而训练不停：

```bash
tmux new -s train
cd /root/autodl-tmp/Kairos && source .venv/bin/activate

torchrun --standalone --nproc_per_node=1 \
    -m kairos.training.train_predictor
```

**脱离 tmux 会话**：`Ctrl+B` 然后 `D`。SSH 断开、关电脑都不影响。

**重新连回**：

```bash
ssh -p <PORT> root@<HOST>
tmux attach -t train
```

### 5.3 nohup 替代方案

如果不想装 tmux：

```bash
cd /root/autodl-tmp/Kairos && source .venv/bin/activate
nohup torchrun --standalone --nproc_per_node=1 \
    -m kairos.training.train_predictor \
    > train.log 2>&1 &

tail -f train.log
```

### 5.4 训练监控

开第二个 SSH 窗口：

```bash
# 看 GPU 使用率（应稳定在 >80%）
watch -n 2 nvidia-smi

# 看训练日志
tail -f /root/autodl-tmp/Kairos/train.log

# 看 checkpoint 保存进度
ls -lth /root/autodl-tmp/Kairos/artifacts/checkpoints/predictor/checkpoints/
```

**正常日志片段**：

```
[ep 1/30 step 100/2000] lr=2.3e-05 loss=3.4218 ce=2.8901
...
--- ep 1: val_ce=2.7834 (0:02:45 / total 0:02:45) ---
[save] best → .../best_model
```

### 5.5 先跑一次 smoke 再跑正式训练（强烈建议）

在开长跑之前，先用 1 epoch + 40 step 确认整条链路没问题：

```bash
KAIROS_SMOKE=1 torchrun --standalone --nproc_per_node=1 \
    -m kairos.training.train_predictor
```

预期 2-5 分钟就跑完 1 个 epoch 并存一个 checkpoint。没问题再跑正式训练（把 `KAIROS_SMOKE=1` 去掉）。

---

## 6. 训练完把 checkpoint 拉回本地

### 6.1 AutoDL 上打包

```bash
cd /root/autodl-tmp/Kairos
tar czf /root/autodl-tmp/artifacts.tar.gz artifacts/
ls -lh /root/autodl-tmp/artifacts.tar.gz    # 一般 1-3 GB
```

### 6.2 Mac 上下载

```bash
# 在 Mac 上
scp -P <PORT> root@<HOST>:/root/autodl-tmp/artifacts.tar.gz ~/Downloads/

cd /Users/jie.feng/wlb/Kairos
tar xzf ~/Downloads/artifacts.tar.gz
ls artifacts/checkpoints/predictor/checkpoints/best_model/
```

### 6.3 关机/释放

- **只关机**：数据盘保留，下次开机 5 秒恢复（不再按时收费，但按硬盘收少量存储费）
- **释放实例**：数据盘**彻底清空**，只有不再需要任何数据时才这么做

建议先**关机**保留几天，确认 checkpoint 没问题再释放。

---

## 7. 本地回测（在 Mac 上，CPU 即可）

```bash
cd /Users/jie.feng/wlb/Kairos
source .venv/bin/activate

# 跑 finetune/qlib_test.py 或你自己的回测脚本
python finetune/qlib_test.py
```

观察：

- **IC（信息系数）** > 0.03 即有边际；> 0.05 已相当不错
- **RankIC** 比 IC 稳定
- 分层回测曲线应单调

---

## 8. 常见坑速查

| 错误信息 | 原因 | 解决 |
|---|---|---|
| `Parquet magic bytes not found` | macOS `._xxx` 伴生文件 | `find raw -name '._*' -delete` |
| `no kernel image available for execution on device sm_120` | PyTorch 版本不支持 5090 | 装 torch 2.7+cu128 |
| `ModuleNotFoundError: No module named 'kairos'` | 忘了 `pip install -e .` 或忘了 activate venv | `source .venv/bin/activate` |
| `git clone` 卡住 | AutoDL → GitHub 链路差 | `source /etc/network_turbo` 或用镜像 `gh-proxy.com`，或从本地 scp |
| HF 下载卡住 | 直连 huggingface.co | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| HF 配了镜像仍然卡在 `loading NeoQuasar/...` | 学术加速代理把镜像请求也绕到境外 | `unset http_proxy https_proxy` 或把 `hf-mirror.com` 加入 `no_proxy` |
| `CUDA out of memory` | batch_size 过大 | 调小到 32 或 24 |
| `RuntimeError: NCCL` | CPU 机器误用 nccl 后端 | 已自动处理，无需干预 |
| `torch.compile` 报 sm_120 bug | PyTorch 2.7 对 Blackwell 编译器未完全稳定 | 不启用 torch.compile（默认就没用） |

---

## 9. 后续增量同步

训练期间如果在本地改了代码，只同步改动不用重传数据：

```bash
# 本地 Mac
rsync -avz -e "ssh -p <PORT>" \
    --exclude='.venv' --exclude='__pycache__' --exclude='*.egg-info' \
    --exclude='raw' --exclude='artifacts' --exclude='.git' \
    --exclude='._*' --exclude='.DS_Store' \
    /Users/jie.feng/wlb/Kairos/ \
    root@<HOST>:/root/autodl-tmp/Kairos/
```

反向从 AutoDL 拉回增量 checkpoint 也一样：

```bash
rsync -avzP -e "ssh -p <PORT>" \
    root@<HOST>:/root/autodl-tmp/Kairos/artifacts/ \
    /Users/jie.feng/wlb/Kairos/artifacts/
```

---

## 10. 省钱小技巧

1. **准备工作全在本地做**（采集、打包、代码调试），GPU 机器只负责训练
2. **先 smoke 再正式跑**，省掉"改一个字重启一次"的几小时账单
3. **用按量计费**，不是包日包月；跑完立刻关机
4. **多卡不一定划算**：Kronos-small 25M 参数单卡 5090 就够；多卡只在显存爆或数据极大时才上
5. **留意 AutoDL 的夜间折扣**（部分时段/部分卡有折扣价）
