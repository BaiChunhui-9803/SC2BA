# Codex 实验执行交付说明：SC2BA v10 全 13 场景 × 4 架构 × 固定单对手 vs 多算法池

> **文档用途**：供运行于交互桌面（`WinSta0\Default`）的 Codex 会话直接接管执行 SC2BA `v10` 全流程实机实验、验收、汇总与分析。  
> **编制日期**：2026-11-26  
> **关联工作目录**：`D:\Binich\project\SC2BA-Lab`（主执行仓）｜`E:\Binich\White Night City`（4321 实时 Web 展示面板，已预接入自动同步）  
> **参考历史会话**：Antigravity `5f6c9270-a569-4a63-b724-3bd382e1aa9f`、Codex `01a20c38-eed1-77c1-8a03-0f0824842b33`

---

## 一、实验目标与核心研究问题

本批次（`v10`）承接 `HRL-IMCBS`（`v01`–`v09`，MM 原 6 图四编码全量实证）与 `SC2BA-Lab`（`B00`–`B04`，3m VDN 混池训练、8×8 冻结矩阵、B02a QMIX 单训对照及 `bch/` 移植），在 **SC2BA 多算法对抗平台**上完成两项核心实证：

1. **主实验 `v10-A`（全 13 场景 × 4 种上层状态编码架构）**  
   在 SC2BA 全部 **13 个带预训练对手权重的场景**上完整训练与评测 4 种上层状态编码架构：
   - **已注册对称场景（7）**：`3m`、`8m`、`25m`、`2s3z`、`3s5z`、`1c3s5z`、`MMM`
   - **已注册非对称场景（3）**：`5m_vs_6m`、`10m_vs_11m`、`MMM2`
   - **已补齐注册表的场景（3）**：`3s5z_vs_3s6z`、`3z_vs_3s`、`5z_vs_3s`
   - **4 种上层状态编码架构**：
     1. `im-hash`：原始有符号势力图 → `RdBu` 渲染 → RGB 均衡 24 位十六进制哈希（无界建表）
     2. `im-medoid`：同哈希前端 → 离线加权 Hamming $K$-medoids 固定码本（$K=32$）
     3. `unit-set`：置换不变单位集合编码器（1D Conv + sum/max readout → 32 维嵌入 → $K=32$ 欧氏 medoid）
     4. `im-grid`：单通道有符号势力矩阵（$1\times 128\times 128$）→ 4 层步幅 Conv2d → 32 维嵌入 → $K=32$ 欧氏 medoid
   - **评估核心指标**：9 对手全库冻结胜率（`All-9`）、8 冻结 MARL 等权胜率（`8-Algo`）、最弱 3 算法均胜率（`Worst-3`）、对内置 Bot 胜率、`standard`（规范出生）vs `near`（未见开局）泛化、上层 Q 表规模、冻结未见键率与单步编码 p50 延迟。

2. **核心对照 `v10-B`（固定单冻结算法对抗 vs. 多算法池对抗）**  
   在**严格相同的上层决策步预算（`train_steps`）与相同的冻结状态编码器**下，对比 4 种架构在不同训练对手制度下的表现：
   - **5 种固定单对手制度**（结合 `B01` 强弱与算法家族多样性选取）：
     - `fixed-bot`（内置 AI 难度 7，对应传统 HRL-IM/CBS 设定）
     - `fixed-qmix`（`B01` 8×8 矩阵榜首 54.91%，单调值分解强手）
     - `fixed-dop`（`B01` 第二 48.51%、对 Bot 榜首 72.22%，off-policy Actor-Critic）
     - `fixed-vdn`（经典线性值分解，对 Bot 65.28%）
     - `fixed-coma`（on-policy 反事实策略梯度，打法风格差异大）
   - **2 种多算法池制度**：
     - `pool-algos8`（8 个冻结 MARL 算法轮换，剥离内置 Bot 暴露）
     - `pool-all9`（8 个冻结 MARL 算法 + 内置 Bot 全池轮换）
   - **双轨对照产出**：
     - **训练过程动态曲线**：每隔固定步数（如 `0 / 2k / 4k / 6k / 8k`）冻结 Q 表，在**同一 9 对手验证池**（`diagonal` 布局）评测并记录胜率与未见键演化曲线；
     - **终局池内/池外跨对手矩阵**：在 `standard + near` 9 对手库上拆分报告**池内对手（In-distribution）胜率** vs **池外泛化对手（Out-of-distribution）胜率**，直接检验单对手过拟合与多算法池跨对手泛化增益。

---

## 二、硬性环境配置与仓库安全约束（务必遵守）

1. **Git 推送红线（CRITICAL）**
   - 工作仓库：`D:\Binich\project\SC2BA-Lab`（分支 `main`）。
   - `origin = https://github.com/BaiChunhui-9803/SC2BA.git`；`upstream = https://github.com/dooliu/SC2BA.git`（已禁用推送）。
   - **严禁向 `upstream` 推送**；实验未验收前不要主动 `git push`。
2. **Python / CUDA 环境（不要用旧 `.venv`）**
   - 统一使用：`C:\Users\Binich\.conda\envs\smacv2\python.exe`（Python 3.10，PyTorch `2.5.1+cu121` CUDA 可用，本机 GPU 为 **NVIDIA RTX 4070 12GB**，CPU **24 核**，64GB RAM，已齐备 `pysc2 / cv2 / seaborn / sklearn / scipy / yaml / threadpoolctl`）。
   - **不要使用** `D:\Binich\project\SC2BA\.venv`（缺 `cv2`、CPU-only 且 `sc2ba` 指向旧目录）。
3. **必备环境变量与 PowerShell 坑点**
   - 运行 PowerShell 脚本时**必须加 `-NoProfile`**：用户 Profile 中 `conda activate base` 在默认 `cp1252` 下会因系统 `PATH` 含中文路径 `D:\wechat-web\微信web开发者工具` 报 `UnicodeEncodeError`。
   - 一键脚本 `D:\Binich\project\SC2BA-Lab\bch\run_v10.ps1` 已自动设置 `PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`、`PYTHONPATH=D:\Binich\project\SC2BA-Lab` 并将 `smacv2` 及 `Library\bin`、`Scripts` 前置入 `PATH`。
4. **SC2 引擎前提**
   - SC2 版本：`5.0.16.97425`（`C:\Program Files (x86)\StarCraft II\Versions\Base97425\SC2_x64.exe`），`COMMON.SC2Map` 已就绪。
   - 必须在**活动交互桌面 `WinSta0\Default`** 中运行（不可在锁屏或隐藏桌面启动，否则 D3D9 `CreateDevice` 返回 `D3DERR_NOTAVAILABLE` / 退出码 `536936448`）。
   - 启动前检查无残留 SC2 进程：`Get-Process SC2* -ErrorAction SilentlyContinue`。

---

## 三、已完成的前置代码升级与离线校验（无需重复开发）

以下改动均已落地于 `D:\Binich\project\SC2BA-Lab` 并完成 100% 离线全量验证：

| 文件 | 已完成改动 |
| :--- | :--- |
| `sc2ba/env/starcraft2/maps/sc2ba_maps.py` | 补齐 `3s5z_vs_3s6z`（8v9, PvP, limit=170）、`3z_vs_3s`（3v3, PvP, limit=120）、`5z_vs_3s`（5v3, PvP, limit=120）；修正 `8m` 的 `blue_units` 笔误 `9 -> 8`。 |
| `sc2ba/enemycontrol/enemy_control.py` | 将 `collections.Mapping` 升级为 `collections.abc.Mapping`，兼容 Python 3.10 + PyTorch 2.5 CUDA。 |
| `bch/observation.py` | 向后兼容扩展 `shield / shield_max / unit_type / attack_range / is_medivac / energy`；上层 `margin` 改为有效血量 `health + shield`（人族保持等价）。 |
| `bch/arena.py` | 移除 Marine-only 限制；自动读取 `a_race/b_race` 与完整 `blue_units_config`；新增 `heal: 386`；支持护盾满值开局校验及 `MMM/MMM2` 的 `only_medivac_left` 终局判断。 |
| `bch/lower_action/macros.py` | 分离医疗兵与战斗单位（Medivac 仅对伤员发 `heal (386)` 或 `move`，杜绝 `smart` 误装载）；禁止地面专属单位（Marauder/Zealot/Colossus）锁定空中 Medivac。 |
| `bch/lower_action/reward.py` & `bch/upper_state/geometry.py` | 按各兵种真实射程与有效血盾比计算火力覆盖与 66 维几何监督目标。 |
| `bch/configs/v10_all13.json` | 13 场景 × 9 对手统一配置：每场景固定 4 布局 `standard / offset`（train）、`diagonal`（validation）、`near`（test），含 `mixed` 与 `legacy-nearest` 两种探针行为。 |
| `bch/v10_suite.py` | 多进程并行、Windows `TEMP` 隔离、可断点续跑（`done.json`）的全套实验引擎，内置 `smoke / collect / fit / run / analyze` 五个子命令。 |
| `bch/run_v10.ps1` | 封装环境与阶段调度的一键脚本（支持 `smoke / v10a-collect-fit / v10a-run / v10b-fixed-vs-pool / analyze / all`）。 |
| `bch/verify_v10_offline.py` | 离线校验脚本，已实测通过 **13/13 场景、104/104 `agent.th` 权重前向推理、1,540/1,540 次宏动作校验**，结果位于 `bch/runs/v10_all13/offline_verification.json`。 |

---

## 四、分阶段执行步骤、命令与验收门禁

统一输出根目录：`D:\Binich\project\SC2BA-Lab\bch\runs\v10_all13`

### Stage 0：`v10-0` 全 13 场景实机冒烟与步速标定（必做首关）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\Binich\project\SC2BA-Lab\bch\run_v10.ps1 -Stage smoke -Scenarios all13 -Workers 4
```

- **执行内容**：启动真实 `SC2_x64.exe`，对全部 13 个场景各跑 9 局（`bot` + 8 冻结 MARL，共 117 局），循环覆盖全部 5 分组 × 11 宏动作，测算真实步速（`steps/s`、`eps/min`）和 3 类编码 p50 延迟。
- **验收门禁（Gate 0）**：
  1. `bch/runs/v10_all13/smoke/smoke_summary.json` 中 **`passed == 13` 且 `failed == 0`**；
  2. 所有 13 场景无 `SC2 rejected red macro commands`、无 `Unexpected SC2BA terminal state`、无 `Bad initial roster or HP`、`restarts == 0`；
  3. **若个别异构/新注册场景在实机出现边角报错，必须逐个定位根因并修复后重验至 13/13 PASS，严禁跳过场景**：
     - 重点关注 `MMM` / `MMM2`：如实机对 Medivac 发出 `heal (386)` 偶发 `CantTarget` / `NotEnoughEnergy` 或敌方仅剩 RL Medivac 时终局判定异常，检查 `bch/lower_action/macros.py` 的 `_medivac_support` 与 `bch/arena.py` 的 `step_macros` / `_MixedActions`；
     - 重点关注 `25m`：如 25 单位在 `offset / diagonal / near` 出生挤出可玩边界，微调 `v10_all13.json` 中 `25m` 出生坐标靠近地图中部；
     - 重点关注 `3s5z_vs_3s6z` / `3z_vs_3s` / `5z_vs_3s` / `1c3s5z`：确认神族单位满盾校验及 8 冻结模型每步动作无维度异常。
  4. **吞吐核算**：依据 `smoke_summary.json` 的真实 `steps_per_second` 与 `im-hash` 单步耗时（约 12–17 ms），评估并发数：本机 24 核 64GB RAM，如 4 workers 稳定且 CPU/内存富余，后续 Stage 1–3 可将 `-Workers` 调至 `6` 或 `8` 加速（参考 `B01` 使用 8 SC2 实例 0 重试）。

---

### Stage 1：`v10-A` 探针采集与冻结状态编码器拟合（全 13 场景）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\Binich\project\SC2BA-Lab\bch\run_v10.ps1 -Stage v10a-collect-fit -Scenarios all13 -Workers 6
```

- **执行内容**：
  1. `collect`：每场景 4 布局 × 9 对手 × 2 行为（`mixed` + `legacy-nearest`）× `repeats=2` = 144 局（13 场景共 1,872 局），写入 `probes/<scenario>/probes.jsonl.gz` 与 `manifest.json`（已完成场景自动跳过）。
  2. `fit`：在 **RTX 4070 (`--device cuda`)** 上为全部 13 场景分别拟合 `unit-set`、`im-grid`（`epochs=160, latent_dim=32, codes=32`），并在多核 CPU 上拟合 `im-medoid`（`codes=32`），共 39 个冻结编码器，落地至 `encoders/<scenario>/<method>/encoder.pt` 与 `encoders/fit_summary.csv`。
- **验收门禁（Gate 1）**：
  1. `probes/<sc>/manifest.json` 齐全 13/13，且每场景 `train / validation / test` 三折样本均非空；
  2. `encoders/fit_summary.csv` 包含全部 **39 行**（13 × 3），所有 `medoid_cost`、`test_geometry_mae`、`test_hp_mae` 为有限正数且无 NaN；
  3. 特别核查极短对局场景（如 `3z_vs_3s`、`3m`）训练折唯一状态数是否 $\ge 32$（如某场景因对局过短导致 `im-medoid` 唯一哈希不足 32 个，仅对该场景增补 `--repeats 4` 重新采集探针再拟合）。

---

### Stage 2：`v10-A1` 全 13 场景 × 4 编码架构 × 多算法池（`pool-all9`）完整训练与评测

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\Binich\project\SC2BA-Lab\bch\run_v10.ps1 -Stage v10a-run -Scenarios all13 -Workers 6 -TrainSteps 8000 -EvalEvery 2000 -Seeds "101,202,303"
```

- **规模与协议**：
  - `13 场景 × 4 架构 (im-hash, im-medoid, unit-set, im-grid) × 1 制度 (pool-all9) × 3 配对种子 (101, 202, 303)` = **156 arms**；
  - 每 arm 固定上层决策步预算 `8,000` 步，在 `0 / 2000 / 4000 / 6000 / 8000` 步冻结 Q 表并在 `diagonal` 验证布局对全 9 对手评测（`val_repeats=2`，每检查点 18 局）；
  - 终局在 `standard + near` 两套布局对全 9 对手冻结评测（`test_repeats=5`，共 90 局/arm，自动校验冻结评测不修改 Q 表）。
- **验收门禁（Gate 2）**：
  1. `arms/<sc>/pool-all9/<method>/seed_<seed>/done.json` 齐全 **156/156**；
  2. 运行 `-Stage analyze` 后 `analysis/summary_by_condition.csv` 包含全部 52 组 `(scenario, pool-all9, method)` 统计；
  3. 核查固定码本（`im-medoid / unit-set / im-grid`）上层 Q 表行数严格 $\le 32$ 且 `upper_unseen_rate == 0.00`；对比 `im-hash` 的 Q 表扩张与未见键率；
  4. **及时检查并保存阶段报告**后再进入 Stage 3（若单 arm 耗时快于预期，可考虑将 `25m / MMM2 / 1c3s5z / 3s5z_vs_3s6z` 等复杂场景追加长步数或补至 5 种子作为 `v10-A2`）。

---

### Stage 3：`v10-B1` 固定单冻结算法对抗 vs. 多算法池对抗对照

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\Binich\project\SC2BA-Lab\bch\run_v10.ps1 -Stage v10b-fixed-vs-pool -Scenarios core4 -Workers 6 -TrainSteps 8000 -EvalEvery 2000 -Seeds "101,202,303"
```

- **规模与协议**：
  - **核心 4 类代表性场景**（`core4` = `3m` 小规模对称、`8m` 中规模对称、`2s3z` 神族远程+近战、`MMM` 人族含医疗兵异构）× **4 架构** × **7 训练制度**（`fixed-bot`, `fixed-qmix`, `fixed-dop`, `fixed-vdn`, `fixed-coma`, `pool-algos8`, `pool-all9`）× **3 种子** = 336 arms；
  - `pool-all9` 的 48 arms **自动复用 Stage 2 产物**（检测 `done.json` 直接跳过），实际净增 **288 arms**，且与 Stage 2 共用完全相同的冻结编码器，确保零混淆变量。
- **验收门禁（Gate 3）**：
  1. `core4` 下全部 7 制度 × 4 架构 × 3 种子 = **336/336 arms** 均含 `done.json`；
  2. `analysis/learning_curves.csv` 完整覆盖 7 制度 × 4 架构的 `0/2k/4k/6k/8k` 步 9 对手验证胜率曲线；
  3. `analysis/opponent_matrix.csv` 包含逐对手胜负平与 `in_train_pool` 标记，可精确计算每个固定单对手 vs. 多算法池的 **池内（In-dist）胜率**、**池外（Out-of-dist）胜率**、`8-Algo` 等权胜率与 `Worst-3` 最弱三对手均胜率。
- **可选延伸（视算力与时间决定，非必须一次跑完）**：
  - `v10-B2`：将固定 vs. 多算法池扩展至非对称/新注册场景（`5m_vs_6m,MMM2,3s5z_vs_3s6z,3z_vs_3s` × `fixed-bot,fixed-qmix,pool-algos8,pool-all9`）：
    ```powershell
    Set-Location D:\Binich\project\SC2BA-Lab; $env:PYTHONPATH="D:\Binich\project\SC2BA-Lab"; $env:PYTHONUTF8="1"
    C:\Users\Binich\.conda\envs\smacv2\python.exe -m bch.v10_suite run `
      --scenarios "5m_vs_6m,MMM2,3s5z_vs_3s6z,3z_vs_3s" `
      --methods "im-hash,im-medoid,unit-set,im-grid" `
      --regimes "fixed-bot,fixed-qmix,pool-algos8,pool-all9" `
      --seeds "101,202,303" --train-steps 8000 --eval-every 2000 --eval-init `
      --val-repeats 2 --test-repeats 5 --workers 6 `
      --out-root D:\Binich\project\SC2BA-Lab\bch\runs\v10_all13
    ```

---

### Stage 4：数据汇总、可视化分析与实时 Web 面板核验

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\Binich\project\SC2BA-Lab\bch\run_v10.ps1 -Stage analyze
```

1. **自动汇总产物**（落地于 `D:\Binich\project\SC2BA-Lab\bch\runs\v10_all13\analysis\`）：
   - `arms_per_seed.csv`（逐种子明细）
   - `summary_by_condition.csv` & `summary.json`（按 `scenario × regime × method` 聚合均值 ± 标准差）
   - `learning_curves.csv`（训练过程 9 对手周期验证曲线）
   - `opponent_matrix.csv`（全 9 对手细分胜负平与 `in_train_pool` 标记）
2. **Web 面板自动实时上线（已预对接完成，无需修改前端代码）**：
   - 白夜城本地服务（`http://127.0.0.1:4321`，源码 `E:\Binich\White Night City\src\studio\experiment-hub.ts`）已配置直接读取 `D:\Binich\project\SC2BA-Lab\bch\runs\v10_all13\` 下的 `smoke_summary.json`、`fit_summary.csv`、`summary_by_condition.csv` 与 `learning_curves.csv`。
   - 实验每跑完一批执行一次 `analyze`（或 `run` 结束自动触发），以下页面即刻展示最新柱状图、验证折线图与汇总表：
     - `http://127.0.0.1:4321/experiment-panels/sc2ba-lab/?theme=light&batch=sc2ba-v10-all13-20261126&version=sc2ba-v10-all13-matrix-v1`
     - `http://127.0.0.1:4321/experiment-panels/hrl-imcbs-state-encoding/?theme=light&batch=sc2ba-migration-20261125&version=sc2ba-v10-all13-matrix-v1`
3. **请 Codex 补充交付的分析与图表**（写入 `D:\Binich\project\SC2BA-Lab\bch\runs\v10_all13\analysis\` 并汇总到 `refine-logs\V10_EXPERIMENT_RESULTS.md`）：
   - **图 1（`v10a_all13_four_encoders.png/.pdf`）**：全部 13 场景 × 4 架构的 `All-9` / `8-Algo` 冻结胜率对比、上层 Q 表规模压缩对比与单步编码延迟对比；
   - **图 2（`v10b_fixed_vs_pool_curves.png/.pdf`）**：`core4` 4 场景 × 4 架构下，固定单对手（`bot/qmix/dop/vdn/coma`）vs. 多算法池（`algos8/all9`）的训练过程 9 对手验证胜率演化曲线；
   - **图 3（`v10b_in_vs_out_dist_heatmap.png/.pdf`）**：7 训练制度 × 9 测试对手的跨对手胜率热力矩阵及池内（In-dist）vs. 池外（Out-of-dist）泛化落差对比图；
   - **分析报告（`refine-logs/V10_EXPERIMENT_RESULTS.md`）**：客观归纳（1）4 种编码架构在对称/非对称/神族/异构/25v25 下的表现分化与机制解释；（2）固定单算法训练是否过拟合、多算法池训练对整体与最弱对手的真实提升幅度及各编码架构的适应差异；（3）与 `HRL-IMCBS v09` 及 `SC2BA B01/B02` 的横向对照结论。

---

## 五、异常处理与断点续跑守则

- **断点续跑**：`v10_suite.py` 以每个 arm 目录下的 `done.json` 作为完成标志；若中途发生 SC2 断连或机器重启，只需先清理残留进程 `Stop-Process -Name SC2_x64 -Force -ErrorAction SilentlyContinue`，然后**原样重跑同一阶段命令**即可自动跳过已完成 arm 并重跑未完成项。
- **严谨底线**：
  - 不得伪造任何胜率、延迟或跳过报错场景；
  - 若某并发数下 SC2 出现偶发连接超时，优先降低 `-Workers`（如 `6 -> 4`）并在日志中记录重试情况；
  - 所有实验结束后确认本机无残留 `SC2_x64.exe` 进程，并将最终结果路径与关键指标汇报给用户。
