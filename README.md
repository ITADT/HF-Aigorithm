# coop_decoy 协同诱饵识别 —— 功能说明

## 1. 简介

赛题二 `coop_decoy`：3 架固定翼无人机在 4km×4km 任务区搜索 3 个真目标
（TargetVehicle）与 15 个诱饵（DecoyVehicle），通过双机协同锁定并摧毁真目标，
同时识别（盯防 20s）诱饵。本版本定位为**先能拿分**的可行版：

- 完成 3 机分区路线规划（螺旋搜索 + 分区轮换）
- 完成识别后的处理（验证 → 主盯防/支援盯防 → 超时释放 → 诱饵宣判）
- 接入 YOLO 视觉识别（官方权重），预留真假判别接口（后续 2 类重训模型直接生效）
- 当前单轮得分 0~6/100、击杀 0~1/3（不稳定，见文末已知限制）

## 2. 文件清单

| 文件 | 作用 |
|---|---|
| `main.py` | 主算法：类 `V1CoopAgent(CoopAgent)`，路线规划 + 状态机 + 决策 |
| `yolo_proc.py` | 共享 YOLO 检测 worker（模块级单例）+ 真假三态接口 |

## 3. 功能架构

### 3.1 路线规划（搜索）
- 任务区分 3 个纵向分区，每架 UAV 按 `uid % 3` 归属一个分区中心；
- 绕分区中心螺旋绕飞（半径 200→600m，线速度 ~24m/s），每 75s 轮换到下一分区中心；
- 云台 pan ±45° 正弦摆动 + tilt 扫掠，光轴落点覆盖半径 ~250m；
- 三机绕飞相位按 uid 错开，避免重叠（罚分）。

### 3.2 识别后的处理（状态机）
```
SEARCH(0) --检出候选--> VERIFY(1) --4s 持续检出--> ENGAGE_MAIN(2)
                              |                        |
                              +--收到 T: 求援--> ENGAGE_HELP(3)
```
- **VERIFY**：飞向候选 + 个人方位 520m 偏移（防多机同点罚分），云台锁定收集样本；
- **ENGAGE_MAIN**：目标侧向摆锤切弦往返（半径 200m），1Hz 广播 `T:`，双机锁定期间
  1Hz 上报；摆锤中心 5s 冻结防噪声漂移；摆锤方向按 UID 错开 120°；
- **ENGAGE_HELP**：目标侧向 450m 站位绕飞，云台始终瞄准共享目标，检出后 2s 广播 `L:`；
- 双 MAIN 让位：同目标双 MAIN 时 uid 大者确定性转为 HELP；
- 超时释放：22s 无支援则拉黑并广播 `D:` 宣判诱饵（HELP 超时静默退出）；
- 收到 `D:` 只放弃不拉黑（防真目标被永久忽略）。

### 3.3 YOLO 视觉识别（v4.2+YOLO 新增）
- `yolo_proc.py` 提供共享 worker：3 机 agent 实例共用同一后台推理线程，各按 uid
  维护最新检测（内存队列，sensor 内零阻塞）；
- `sensor()`：`photo` → 入队 → 取最新假设 → `Detection(detected=True, target_type=类名)`；
- **真假三态接口** `is_real = True / False / None`：
  - 官方单类权重（只有 `TargetVehicle`，未做真假区分）→ 检出即视为真目标候选 `True`；
  - 换 2 类权重（`TargetVehicle`/`DecoyVehicle`）后自动按类名返回 `True`/`False`；
  - 未知类名 → `None`（FSM 保守处理）；
- **降级链**：无照片 / worker 未就绪 / 结果过期 → `sensor()` 返回 None → 走 SDK
  默认识别器（train=AccuracySimulator，eval=YoloDetector，无 UE 再回退
  AccuracySimulator），不影响原有得分行为；
- 附带 per-UAV mini IoU 跟踪（track_id）与 bbox→pan/tilt→lat/lon 投影。

### 3.4 通信
- 全向可达，但 inbox 不清空、消息永久重复投递 → 按 `(sender, payload)` 去重；
- 消息协议：`T:` 目标广播 / `A:` 支援应答 / `L:` 双机锁定 / `D:` 诱饵宣判。

## 4. 运行方式

### 4.1 本地（headless，无需 UE）
```powershell
# 1) 启动 Redis（仿真包 bin\redis-server.exe，端口 6379）
# 2) 运行（把 <代码根> / <仿真包> 换成你的路径；需在仿真包目录执行）
$env:PYTHONPATH = "<代码根>;<仿真包>"
cd <仿真包>
#    train 模式用 AccuracySimulator 快速调参
python\python.exe -u -m competition run --scenario coop_decoy `
    --agent Code.main:V1CoopAgent --duration 300 --seed 1 `
    --mode train --redis-port 6379 --output <代码根>\out_dev1 *> <代码根>\dev1.log
#    eval 模式自动挂官方 YOLO 权重（无 UE 时自动回退 AccuracySimulator）
python\python.exe -u -m competition run --scenario coop_decoy `
    --agent Code.main:V1CoopAgent --duration 600 --seed 1 `
    --mode eval --yolo-model examples/yolotrack/target_vehicle_yolov8s.pt `
    --output <代码根>\out_eval1 *> <代码根>\eval1.log
```
- 结果 JSON 在 `--output` 指定目录的 `<...>.evaluation.json`，stdout 重定向为日志。
- 本地运行前建议设 `$env:YOLO_CONFIG_DIR = "<可写目录>"`（ultralytics 配置目录，
  详见部署说明 2.4）。

### 4.2 可视化平台（UE 渲染）
- 把 `main.py`、`yolo_proc.py` 复制到
  `仿真包/competition/user_algorithms/coop_decoy/`，前端刷新后选中运行；
- 文件名必须为**合法 Python 模块名**（字母/数字/下划线，不能含 `.`、`-`）；
- 前端默认不启用 YOLO：需通过 `http://localhost:8081/api/sim/start` 传
  `mode:"eval"` + `yoloModel:"examples/yolotrack/target_vehicle_yolov8s.pt"`
  + `photoMode:"auto"`（详见部署说明）。

## 5. 已知限制

1. 真目标双机锁定几乎不成立：多轮 run 真目标 coop_ticks=0，击杀 0~1/3 不稳定；
2. 20s 连续双机 dwell 对云台跟踪要求高，dwell 经常被切碎（中断 >2s 归零）；
3. accuracy 维度约 0：双机锁定对象多为诱饵，上报位置承接不到真目标分桶；
4. 官方 YOLO 权重为单类（无真假判别），且对 dataset 小目标（fov50 约 4×8px）0 检出，
   真实 UE 画面下是否可检出未验证——这是后续 2 类重训模型要解决的问题；
5. 残余罚分 4~10（支援机转场穿越主盯防区等瞬态 <200m）。
