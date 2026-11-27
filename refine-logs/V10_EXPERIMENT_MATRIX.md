# v10 实验开展矩阵与 SC2BA 全 13 场景推进方案

> 同步副本，完整文档见：[v10_实验开展矩阵与SC2BA推进方案.md](file:///e:/Binich/白春辉/个人/课题/培养环节/论文/第一篇/修改方案/v10_实验开展矩阵与SC2BA推进方案.md)  
> 编制日期：2026-11-26｜离线校验结果：[offline_verification.json](file:///D:/Binich/project/SC2BA-Lab/bch/runs/v10_all13/offline_verification.json)

## 1. SC2BA-Lab 已完成 vs. v10 待启动总览

- **已完成**：`vdn-formal-20261124`（3m 混池 92.36%）、`B01a/b/c`（3m 8×8 冻结矩阵 9,216 局 + Bot 1,152 局）、`B02a`（3m QMIX 单训 vs. 混池首对 200 万步）、`bch` 移植与 `3m/8m` 探针覆盖，以及本轮 **全 13 场景底层升级与 104/104 权重离线验证（ALL PASS）**。
- **待启动 v10 批次**：
  - **`v10-0`**：全 13 场景 × 9 对手实机冒烟与步速标定（~2–3 分钟 @ 4 workers）。
  - **`v10-A1/A2`**：SC2BA 全 13 场景（对称 7 + 非对称 3 + 新注册 3）× 4 上层架构（`im-hash`, `im-medoid`, `unit-set`, `im-grid`）完整训练、过程曲线与 9 对手冻结评测（`standard + near`）。
  - **`v10-B1/B2`**：固定单冻结算法对抗（`fixed-bot`, `fixed-qmix`, `fixed-dop`, `fixed-vdn`, `fixed-coma`）vs. 多算法池对抗（`pool-algos8`, `pool-all9`）同决策步预算对照，输出过程验证曲线与池内/池外跨对手矩阵。
  - **`v10-C`**：高价值补充池（MM 扩展 $K$-敏感性与 100k 步、二值栅格 vs. 势力场消融、异构兵种特征消融、HRL vs. 深度 MARL 同口径横评及 `B02b/B03` VDN 补齐）。

## 2. 本轮新增与升级文件

- [sc2ba_maps.py](file:///D:/Binich/project/SC2BA-Lab/sc2ba/env/starcraft2/maps/sc2ba_maps.py)：注册 `3s5z_vs_3s6z`、`3z_vs_3s`、`5z_vs_3s`，修正 `8m` 蓝方单位数 `9 -> 8`。
- [enemy_control.py](file:///D:/Binich/project/SC2BA-Lab/sc2ba/enemycontrol/enemy_control.py)：修复 `collections.abc.Mapping` 兼容 Python 3.10 + PyTorch 2.5 CUDA 环境。
- [observation.py](file:///D:/Binich/project/SC2BA-Lab/bch/observation.py)、[arena.py](file:///D:/Binich/project/SC2BA-Lab/bch/arena.py)、[macros.py](file:///D:/Binich/project/SC2BA-Lab/bch/lower_action/macros.py)、[reward.py](file:///D:/Binich/project/SC2BA-Lab/bch/lower_action/reward.py)、[geometry.py](file:///D:/Binich/project/SC2BA-Lab/bch/upper_state/geometry.py)：支持全部 13 场景（神族护盾、兵种真实射程、Medivac 治疗/终局与空地目标约束）。
- [v10_all13.json](file:///D:/Binich/project/SC2BA-Lab/bch/configs/v10_all13.json)、[v10_suite.py](file:///D:/Binich/project/SC2BA-Lab/bch/v10_suite.py)、[verify_v10_offline.py](file:///D:/Binich/project/SC2BA-Lab/bch/verify_v10_offline.py)、[run_v10.ps1](file:///D:/Binich/project/SC2BA-Lab/bch/run_v10.ps1)：v10 全量配置、可续跑并行套件、离线校验与一键调度脚本。

## 3. 解锁桌面后运行命令

> 当前 Windows 处于锁屏状态（`LockApp.exe` 运行中），`SC2_x64.exe` 的 Direct3D9 `CreateDevice` 返回 `D3DERR_NOTAVAILABLE`（退出码 `536936448`）。解锁桌面后依次执行：

```powershell
powershell -ExecutionPolicy Bypass -File D:\Binich\project\SC2BA-Lab\bch\run_v10.ps1 -Stage smoke
powershell -ExecutionPolicy Bypass -File D:\Binich\project\SC2BA-Lab\bch\run_v10.ps1 -Stage v10a-collect-fit
powershell -ExecutionPolicy Bypass -File D:\Binich\project\SC2BA-Lab\bch\run_v10.ps1 -Stage v10a-run
powershell -ExecutionPolicy Bypass -File D:\Binich\project\SC2BA-Lab\bch\run_v10.ps1 -Stage v10b-fixed-vs-pool
powershell -ExecutionPolicy Bypass -File D:\Binich\project\SC2BA-Lab\bch\run_v10.ps1 -Stage analyze
```
