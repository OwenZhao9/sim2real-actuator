# sim2real-actuator

## 1. 一句话定位

**量清楚一个执行器有多"脏"（摩擦、偏置、回差），然后在仿真里把这份脏一起复现出来——
这样策略才搬得到真机。**

零仿真器绑定，只依赖 numpy，三件事：**辨识 + 单关节仿真 + 域随机化**。

---

## 2. 为什么存在（要解决的问题）

理想电机是 `力矩 = 想给多少就给多少`。真实减速箱不是。

真实减速箱有四样东西会让"仿真里训好的策略一上真机就废"：

| 脏东西 | 大白话 | 专业名词 |
|---|---|---|
| 推不动才动 | 给一点点力它纹丝不动，力到某个坎才"啪"一下开始转（静摩擦，static / breakaway friction；转起来之后那个恒定阻力叫库仑摩擦，Coulomb friction） |
| 越快越费力 | 转得越快，阻力越大（粘滞摩擦，viscous friction） |
| 一上电就偏 | 什么都没下发，它自己往一边使劲（上电基础力矩 / 偏置，bias torque） |
| 换方向先空转 | 反向时齿轮要先走完那点空隙才咬上（回差，backlash） |

这个库做三件事，**只做这三件**：

1. **辨识（system identification）**：给它一段真实录制（时间、力矩、角速度），
   最小二乘拟合出那四个数字。
2. **仿真（simulation）**：拿这些数字驱动一个单关节模型，纯计算，可在实时环里跑。
3. **域随机化（domain randomization）**：把参数按区间抖一抖，生成 N 个"略有不同的身体"，
   策略在这一堆身体上都能活，才算搬得动。

Pollen Robotics 的 Microduck 就是这个配方：在仿真里把真实执行器的脏东西一起仿真。
**本库是 [Rhoban/bam](https://github.com/Rhoban/bam) 的极简无依赖子集**（详见第 9 节）。

---

## 3. 安装

```bash
uv add "sim2real-actuator @ git+https://github.com/OwenZhao9/sim2real-actuator@v0.1.0"
```

或者从源码：

```bash
git clone https://github.com/OwenZhao9/sim2real-actuator
cd sim2real-actuator
uv sync
uv run pytest
```

Python >= 3.11。运行期唯一依赖是 `numpy`。**不 import 任何硬件相关的东西**，
不打开串口，不访问设备。

---

## 4. 60 秒上手

```python
from sim2real_actuator import ActuatorSim, DomainRanges, identify, load_columns, randomize

# 1) 从任意 CSV 按列名取数 —— 列名由你指定，库里没有硬编码任何表头
cols = load_columns(
    "examples/data/breakaway-measurement.csv",
    t="host_t", tau="cmd_l", vel="ldps", pos="ldeg",
    vel_unit="deg/s",          # 自动转成 rad/s（位置列同时转成 rad）
)

# 2) 辨识：最小二乘拟合摩擦 + 偏置，自动剔除速度饱和点与静止段
p = identify(**cols, fit_inertia=False)
print(p.to_dict())
# {'coulomb_nm': 0.2165, 'viscous_nm_s_per_rad': 0.1660, 'bias_nm': -0.2027, ...}

# 3) 仿真：纯计算，可在实时环里调用
import dataclasses
sim = ActuatorSim(dataclasses.replace(p, inertia_kg_m2=0.05))
sim.reset(pos_rad=0.0, vel_rad_s=0.0)
for _ in range(200):
    s = sim.step(tau_cmd_nm=0.5, dt_s=0.005, external_nm=0.0)
print(f"{s.pos_rad:.4f} rad, {s.vel_rad_s:.4f} rad/s, moving={s.moving}")

# 4) 域随机化：同 seed 同结果，20 个略有不同的身体
bodies = randomize(dataclasses.replace(p, inertia_kg_m2=0.05), DomainRanges(), 20, seed=2024)
```

上下两个方向各测一次"刚开始动"的临界力矩，就能直接分离静摩擦与偏置：

```python
from sim2real_actuator import breakaway
static_nm, bias_nm = breakaway(up_nm=0.29, down_nm=0.51)   # -> (0.40, 0.11)
```

跑现成的两个例子：

```bash
uv run python examples/identify_exoskeleton.py   # 用真实录制辨识
uv run python examples/digital_twin.py           # 双关节数字义体
```

---

## 5. API 参考

导入面就这些，没有别的：

```python
from sim2real_actuator import (
    ActuatorParams, SimState, DomainRanges,
    identify, breakaway, ActuatorSim, randomize, load_columns,
)
```

### 统一符号约定（全库唯一一套）

轴上的运动方程是：

```
inertia_kg_m2 · 角加速度
    = tau_cmd_nm + external_nm + bias_nm
      − viscous_nm_s_per_rad · 角速度
      − coulomb_nm · sign(角速度)
```

也就是说 **`bias_nm` 是执行器自己产生的那份恒定力矩，正数 = 沿正方向推**
（与 `ActuatorParams.bias_nm` 的字段定义"上电基础力矩，带符号（正 = 沿正方向）"一致）。
在这套约定下两个方向的临界力矩是
`up = coulomb_nm − bias_nm`、`down = coulomb_nm + bias_nm`（都取模），
正好反解成 `breakaway(up, down) == ((up+down)/2, (down−up)/2)`。

契约里写的回归式 `τ ≈ J·a + b·ω + c·sign(ω) + bias`，最后一项解出来的是
**调用方需要额外补上的力矩**，正好是 `bias_nm` 的相反数；`identify()` 取负之后再存进
`bias_nm`，所以 `identify()` 和 `breakaway()` 给出的 `bias_nm` 符号一致、可互换。

### `ActuatorParams`（frozen dataclass）

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `coulomb_nm` | `float` | Nm | 库仑摩擦，与速度方向相反的恒定阻力，>= 0 |
| `viscous_nm_s_per_rad` | `float` | Nm/(rad/s) | 粘滞摩擦系数，>= 0 |
| `bias_nm` | `float` | Nm | 上电基础力矩，带符号，正 = 沿正方向 |
| `backlash_rad` | `float` = 0.0 | rad | 齿轮回差**总量**（不是半量），>= 0 |
| `inertia_kg_m2` | `float \| None` = None | kg·m² | 输出端折算转动惯量，> 0；`ActuatorSim` 必须有它 |
| `tau_limit_nm` | `float \| None` = None | Nm | 驱动能实际输出的力矩上限，> 0 |
| `source` | `str` = "" | — | 这组参数来自哪次测量，便于追溯 |

`to_dict() -> dict` / `from_dict(d) -> ActuatorParams`，JSON 往返一致。
未知 key 报 `ValueError`（不静默丢字段）。

### `SimState`（frozen dataclass）

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `pos_rad` | `float` | rad | 输出端位置（含回差死区） |
| `vel_rad_s` | `float` | rad/s | 输出端角速度 |
| `tau_applied_nm` | `float` | Nm | 这一步驱动**实际发出**的力矩 = 限幅后的指令，**不含** `external_nm` / `bias_nm` |
| `moving` | `bool` | — | `False` 表示静摩擦按住了，没动 |

`to_dict()` / `from_dict()`。它在 `step()` 的实时路径上被创建，**刻意不做任何校验**。

### `DomainRanges`（frozen dataclass）

| 字段 | 默认 | 语义 |
|---|---|---|
| `coulomb` | `(0.8, 1.25)` | 乘性区间，作用于 `coulomb_nm` |
| `viscous` | `(0.8, 1.25)` | 乘性区间，作用于 `viscous_nm_s_per_rad` |
| `bias` | `(-0.05, 0.05)` | 加性区间，Nm |
| `backlash` | `(0.0, 0.02)` | 加性区间，rad |

`to_dict()` / `from_dict()`。

### `identify(*, t_s, tau_nm, vel_rad_s, pos_rad=None, fit_inertia=True) -> ActuatorParams`

最小二乘拟合 `τ ≈ J·a + b·ω + c·sign(ω) + bias`。

- `t_s`：时间戳，秒，非递减
- `tau_nm`：每个采样点施加的力矩，Nm
- `vel_rad_s`：输出角速度，rad/s
- `pos_rad`：可选位置，rad。**只用来锐化"静止段"判定**（编码器没跳格就是没动）
- `fit_inertia`：`False` 时丢掉加速度项，`inertia_kg_m2` 留 `None`

自动剔除：

- **速度饱和点**：定点钳位会输出逐位相同的极值，因此"|ω| 的极值被重复 >= 3 次"判为轨道
  （rail），连同前后各一个采样点一起丢掉（导数被污染）。**不写死任何具体饱和值**
  （比如 int16 的 ±3276.7），换单位、换设备都成立。
- **静止段**：`|ω| <= 2% × percentile(|ω|, 99.5)` 的采样点。这些点上 `sign(ω)` 没有意义。
  给了 `pos_rad` 时，位置与前后邻居完全相同的点也算静止。

`backlash_rad` 恒为 `0.0`、`tau_limit_nm` 恒为 `None`：**这两个本函数不估计**，别误会。
`source` 会写上用了多少点、丢了多少点、残差 RMS。

抛 `ValueError` 的情况：序列长度不一致、样本 < 8、时间倒流、可用点不够、
**数据只往一个方向动**（此时 `coulomb_nm` 与 `bias_nm` 数学上不可分——用 `breakaway()`）。

### `breakaway(up_nm, down_nm) -> tuple[float, float]`

从上下两个方向"刚开始动"的临界力矩**模值**，分离出 `(静摩擦, 偏置)`：
返回 `((up+down)/2, (down−up)/2)`。负数或非有限值报 `ValueError`。

### `ActuatorSim`

```python
ActuatorSim(p: ActuatorParams, *, seed: int | None = None)
    .params -> ActuatorParams          # 只读
    .reset(pos_rad: float = 0.0, vel_rad_s: float = 0.0) -> None
    .step(tau_cmd_nm: float, dt_s: float, *, external_nm: float = 0.0) -> SimState
```

- `p.inertia_kg_m2` 必须有值，否则构造期 `ValueError`（积分需要它）。
- `step()` 是**半隐式欧拉**（symplectic Euler）积分：`v += a·dt; pos += v·dt`。
- 静摩擦真的会按住关节：`|tau_cmd + external + bias| <= coulomb` 时 `vel = 0`、`moving=False`，
  一步都不动。反向穿零时同样会重新检查，过不了坎就停在零速。
- `tau_limit_nm` 只钳位**指令**，不钳位 `external_nm`。
- 回差是加在**上报位置**上的死区：`pos_rad` 落后电机端最多 `backlash_rad / 2`，
  换向时切到另一侧；`backlash_rad = 0` 时上报位置与电机端逐位相同。
- **`seed` 在 v0.1.0 不改变任何输出**：`step()` 是确定性的、不消耗任何随机数。
  参数被接受（并真的构造了一个私有 `numpy.random.Generator`），是为了冻结 API 形状
  和将来的随机效应。今天真正生效的"同 seed 同结果"在 `randomize()` 上。
- `step()` 只在**调用方传入非法数值**（`dt_s <= 0`、NaN、inf）时抛 `ValueError`。
  这条路径上没有 IO / 网络 / 超时，所以不存在"降级返回"的场景。

### `randomize(p, ranges, n, seed) -> list[ActuatorParams]`

`coulomb` / `viscous` 乘性扰动，`bias` / `backlash` 加性扰动（`backlash_rad` 钳到 >= 0）。
`inertia_kg_m2` 与 `tau_limit_nm` 原样带过去（`DomainRanges` 没有这两个字段，
本库不擅自新增契约外的 API）。

**同 `(p, ranges, n, seed)` 永远给出完全相同的列表**，且**绝不碰全局 numpy RNG**。

### `load_columns(path, *, t, tau, vel, pos=None, vel_unit="rad/s") -> dict[str, list[float]]`

从任意 CSV 按**调用方给的列名**取数。返回的 key 就是 `identify()` 的关键字参数
（`t_s` / `tau_nm` / `vel_rad_s` / `pos_rad`），所以 `identify(**load_columns(...))` 直接能跑。

`vel_unit="deg/s"` 时，**速度列和位置列一起**从度转成弧度——真机上这两列来自同一个编码器、
同一个单位。列名缺失时报错会同时给出你传的参数名、你要的列名、以及文件里实际有哪些列。

---

## 6. 后端与配置

**没有后端，没有配置文件，没有环境变量，没有网络。**

这个库是一个纯函数 + 一个状态机，全部行为由传进去的参数决定：

| 需要什么 | 从哪来 |
|---|---|
| 时间 | 调用方传 `t_s` / `dt_s`，库内**从不读时钟** |
| 随机性 | 调用方传 `seed`，`randomize()` 用私有 `numpy.random.Generator` |
| 列名 | 调用方传 `t=` / `tau=` / `vel=` / `pos=` |
| 单位 | 调用方传 `vel_unit=` |
| 负载（重力、绑带、限位） | 调用方每步算好，传 `external_nm=` |
| 日志 | 标准 `logging`，logger 名 `sim2real_actuator.*`；**库内不 print** |

### ⚠️ 线程与实时性

- **所有实例都不是线程安全的**。一个实例只在一个线程里用。
  实现里**没有加任何锁**（会拖慢热路径）。多线程请一线程一实例。
- **`ActuatorSim.step()` 可以在实时环里调用**：纯计算、无 IO、无时钟、不阻塞。
  测试里用 monkeypatch 把 `time.time` / `time.monotonic` / `open` / `socket.socket`
  全换成会抛异常的桩，再跑 2000 步——这条承诺是被测出来的，不是写在文档里的。
- **`identify()` / `load_columns()` 不要在实时环里调用**：前者是批量最小二乘（分配大数组），
  后者读文件。放到启动阶段或独立线程。

---

## 7. 边界：不做什么

诚实清单，以下**全部没有实现**，不要指望：

1. **不估计回差**。`identify()` 的回归式里没有回差项，返回的 `backlash_rad` 恒为 `0.0`。
   仿真**会**按你给的 `backlash_rad` 建模，但那个数得你自己量。
2. **不估计力矩上限**。`identify()` 返回的 `tau_limit_nm` 恒为 `None`。
3. **不建模电压跌落 / 温度漂移 / 电流环动力学 / 磁滞（hysteresis）/
   Stribeck 曲线（低速摩擦下凹）/ 传动柔性**。需要这些请用
   [Rhoban/bam](https://github.com/Rhoban/bam)，它就是干这个的重量级实现。
4. **不建模负载**。重力、绑带、限位、人腿——全是**你的 plant**，每步自己算好用
   `external_nm=` 喂进来。`examples/digital_twin.py` 里的 `JointLoad` 就是个示范，
   它**故意**写在 example 里而不是库里。
5. **不做多关节耦合**。一个 `ActuatorSim` 就是一个关节。双关节请开两个实例
   （`examples/digital_twin.py` 就是这么干的）。
6. **不绑任何仿真器**（MuJoCo / Isaac / Bullet 一个都不碰），不提供 mjlab / gym 适配层。
7. **不碰硬件**。不 import `pyserial`，不开串口，不连设备。
8. **不替你判断单位/符号/坐标系**。你喂什么帧进来就得到什么帧的结果——
   左右腿镜像、正方向定义这类项目特有约定，请在调用方处理（见第 8 节实测）。
9. **`ActuatorSim(seed=...)` 在 v0.1.0 不产生任何随机行为**（见第 5 节）。
10. **不做在线辨识 / 递推最小二乘（RLS）**。`identify()` 是一次性批量拟合。

---

## 8. 验证与实测数据

### 8.1 合成数据闭环（契约硬指标：各参数误差 < 5%）

已知参数 → `ActuatorSim` 生成 40 000 步轨迹（dt = 1 ms，三频叠加的正弦激励，
必须来回穿越零速，否则 `coulomb_nm` 与 `bias_nm` 数学上不可分）→ `identify()` 还原：

| 参数 | 真值 | 还原值 | 相对误差 |
|---|---|---|---|
| `coulomb_nm` | 0.35 | 0.348378 | **0.46 %** |
| `viscous_nm_s_per_rad` | 0.08 | 0.080340 | **0.42 %** |
| `bias_nm` | 0.12 | 0.120355 | **0.30 %** |
| `inertia_kg_m2` | 0.006 | 0.005910 | **1.50 %** |

另外还在三组差异很大的执行器上重复同一闭环（高粘滞低摩擦、高摩擦零偏置、大偏置），
全部 < 5%。见 `tests/test_identify.py`。

### 8.2 真实录制的辨识结果

数据：`examples/data/` 里三个 CSV，来自一台真实的 Hypershell 髋关节外骨骼
（约 195 Hz，角速度量化到 0.1 °/s，力矩 Nm）。
跑 `uv run python examples/identify_exoskeleton.py` 复现：

**最小二乘（`breakaway-measurement.csv`，`fit_inertia=False`）**

| 关节 | `coulomb_nm` | `viscous_nm_s_per_rad` | `bias_nm` | 用了多少点 | 残差 RMS |
|---|---|---|---|---|---|
| 左 | 0.2165 | 0.1660 | −0.2027 | 241 / 3815 | 0.0735 Nm |
| 右 | 0.3433 | 0.0109 | +0.1201 | 203 / 3815 | 0.0488 Nm |

**`breakaway()` 分离（同一段斜坡里读出的临界力矩）**

| 关节 | `up_nm` | `down_nm` | 静摩擦 | 偏置 |
|---|---|---|---|---|
| 左 | 0.450 | 0.072 | **0.261** | **−0.189** |
| 右 | 0.277 | 0.479 | **0.378** | **+0.101** |

**与厂商/实测参考值对照**（参考值：静摩擦 左 0.30 / 右 0.40 Nm，
上电基础力矩 左 +0.22 / 右 +0.11 Nm）：

- **右关节全中**：静摩擦 0.378 vs 0.40（差 5.5%），偏置 +0.101 vs +0.11（差 8%）。
  两条完全独立的路径（最小二乘 +0.120、临界力矩 +0.101）互相印证。
- **左关节量级对上，符号相反**：静摩擦 0.261 vs 0.30（差 13%），偏置 −0.189 vs +0.22。
  原因是**这台录制设备的左编码器与右编码器互为镜像**：同一个物理方向，左右两列的符号相反。
  参考值是按穿戴者的解剖方向（左右对称）报的，本库按你喂进来的那一列的方向报。
  **库不替任何项目做镜像**——翻符号是调用方两行代码的事，写进库里就成了外骨骼专有代码。
- 左关节的"向下"临界力矩是松开力矩后靠回弹测到的（不是一条完整的负向斜坡），
  所以左侧的数字比右侧粗糙，这是数据的限制，不是算法的。

**惯量测不出来**：这三段录制的角速度量化到 0.1 °/s、采样约 5 ms，求导基本是噪声，
`fit_inertia=True` 时解出来的 `inertia_kg_m2` 是 1e-5 量级，物理上不可能。
所以例子里一律用 `fit_inertia=False`，数字义体的惯量按斜坡上升时间估了个量级
（0.05 kg·m²）并**在代码里明确标注为假设，不是拟合结果**。

### 8.3 数字义体（硬性验收）

`examples/digital_twin.py` 用上面辨识出的参数驱动一个**双关节数字替身**，
接口与真机 bridge 一致（`open` / `ping` / `version` / `on_sample` / `set_torque` /
`commanded` / `stream_hz` / `enable` / `disable` / `close`，样本字段名与单位也一致），
全程无硬件、不读时钟（时间由调用方传入，所以回放逐位可复现）。

把**真实录制里的力矩指令序列**原样喂进数字义体，输出轨迹与真实录制对比：

| 量 | 真实录制 | 数字义体 | 比值 |
|---|---|---|---|
| 左关节角度跨度 | 10.84 ° | 12.12 ° | 1.12 |
| 右关节角度跨度 | 21.60 ° | 12.90 ° | 0.60 |
| 左关节峰值角速度 | 25.80 °/s | 19.67 °/s | 0.76 |
| 右关节峰值角速度 | 43.60 °/s | 17.56 °/s | 0.40 |

验收标准是"同量级"（比值在 0.1 ~ 10 之间），实测四项全部落在 **0.4 ~ 1.2**。
example 跑完会自己判定并用退出码表态，CI 里由 `tests/test_examples.py` 真的执行一遍。

注意：这个吻合度里有一半功劳属于 example 自己那个 plant（回弹弹簧 0.5 Nm/rad +
绑带阻尼 0.05 Nm·s/rad + ±80° 限位），那几个数是**按物理量级挑的假设，不是测出来的**，
在代码里都标了出来。库负责的是执行器那一半。

### 8.4 性能

`step()` 在 Apple Silicon（macOS 26.2 / arm64）上 **0.46 µs 每次调用**
（约 2.17 M calls/s，500 000 次取三轮最好成绩，含回差与力矩限幅）。
测试里有一条 50 µs 的宽松上界防止意外劣化。

### 8.5 测试覆盖了什么

`uv run pytest` → **143 passed**。

| 文件 | 覆盖点 |
|---|---|
| `test_identify.py` | 合成闭环 < 5%（4 组执行器）；确定性；`pos_rad` 可选；`fit_inertia=False`；速度轨道剔除；静止段剔除；冻结编码器剔除；单方向数据报错；长度不一致；样本太少；时间倒流；时钟不走；全静止；非数值 |
| `test_sim.py` | 牛顿定律；静摩擦按住；偏置移动临界点；库仑摩擦停车距离；粘滞终端速度；外力叠加；力矩限幅只管指令；回差死区（与零回差实例逐位对照）；`reset` 精确；回放逐位一致；多实例互不干扰；非法数值报错；**monkeypatch 掉时钟/文件/socket 后仍能跑 2000 步**；速度下界 |
| `test_breakaway.py` | 对称=无偏置；契约公式；上下反解；符号；非法入参；**与 `ActuatorSim` 实测斜坡互相印证**；与真实参考值对照 |
| `test_randomize.py` | 同 seed 同结果；不同 seed 不同；**全局 numpy RNG 未被污染**；区间边界；未随机化字段带过；退化区间；回差不为负；`n=0`；非法入参 |
| `test_load_columns.py` | 按列名取数；`pos` 可选；`deg/s` 双列换算；key 与 `identify` 对齐；缺列报错含参数名+可用列；多列缺失；非数值定位到行；空单元格；只有表头；空文件；BOM；未知单位；非法列名；文件不存在；**扫描源码确认没有硬编码任何项目表头** |
| `test_params.py` | 三个 dataclass 的 JSON 往返；frozen 不可变；错误基类是 `ValueError` 子类；构造期校验 9 例；未知 key 报错；缺 key 报错；`DomainRanges` 校验 7 例 |
| `test_contract.py` | 公开符号集与契约完全一致；契约那行 import 原样可跑；四个函数签名逐字对照；`ActuatorSim` 方法签名；三个 dataclass 字段顺序；`DomainRanges` 默认值；**AST 扫描：库内无 `print`、无读时钟、无锁、无硬件/兄弟库 import、第三方依赖只有 numpy**；`py.typed` 存在；每个公开符号都有 docstring 与完整标注 |
| `test_examples.py` | 两个 example 真的跑通（子进程 + 退出码）；数字义体接口面完整；回调；回放可复现；未使能时忽略指令 |

---

## 9. 出处与致谢

### 论文

- **[arXiv:2410.08650](https://arxiv.org/abs/2410.08650)** —
  *Extended Friction Models for the Physics Simulation of Servo Actuators*（Rhoban，BAM 的论文）。
  本库的摩擦模型（库仑 + 粘滞 + 静摩擦阈值）就是这篇论文里最基础的那一档。
  已核实：Pollen Robotics 的 `microduck_rl` 在 `pyproject.toml` 里硬依赖
  `better-actuator-models`，并在 `actuator/friction_dr_bam.py` 里
  `from bam.mjlab import BamActuator`。
- **[arXiv:2608.00715](https://arxiv.org/abs/2608.00715)** —
  纯仿真训练的髋外骨骼控制器 + 代谢验证。这篇说明了为什么"把执行器的脏东西量准"
  是纯仿真训练能落到真人身上的前提。

### 参考实现

- **[Rhoban/bam](https://github.com/Rhoban/bam)** —— 重量级参考实现。
- **[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)**
  —— 域随机化与回差建模的工程范例。

### 本库的定位

> **本库是 BAM 的极简无依赖子集。**
> 只做「辨识 + 单关节仿真 + 域随机化」三件事，**不绑任何仿真器**，运行期只依赖 numpy。
> 需要 Stribeck 曲线、磁滞、电压跌落、温度模型、MuJoCo/mjlab 集成的，请直接用
> [Rhoban/bam](https://github.com/Rhoban/bam)——本库不打算长成那样。

数据来自 EvoTavern 黑客松深圳站的 Hypershell 髋关节外骨骼实测录制。

### 待确认

无人值守开发期间按"最保守解释"处理、但值得复核的点：

1. **`bias_nm` 的符号**。契约同时给了两个式子：`breakaway` 返回 `((up+down)/2, (down−up)/2)`，
   回归式写 `τ ≈ J·a + b·ω + c·sign(ω) + bias`。这两者的 `bias` 严格相差一个负号。
   本库以 **`breakaway` 的字面公式 + `ActuatorParams.bias_nm` 的字段定义（"正 = 沿正方向"）**
   为准，`identify()` 把回归截距取负后存入。两条路径因此互相一致（第 8.2 节的右关节
   两条路径都给 +0.10~+0.12 即为佐证）。若下游期望的是回归截距本身，翻个符号即可。
2. **`Sim2RealActuatorError`**。契约 0.4 要求每个库定义自己的 `<Lib>Error` 基类，
   同时要求构造期错误抛 `ValueError`。本库让它**继承 `ValueError`**，两条同时成立，
   并额外导出这一个符号（契约第 2 节的 import 清单里没有它）。
3. **`load_columns(vel_unit="deg/s")` 同时换算位置列**。契约只说了 `vel_unit` 管速度。
   真机上这两列来自同一个编码器、同一个单位，分开处理几乎必然出错，故一起换算。
   如果你的位置列单位与速度列不同，请自己转好再传。
4. **`identify` 没有 `source=` 入参**。契约冻结了签名，不能加。想自定义 provenance
   请用 `dataclasses.replace(p, source="...")`。
5. **速度饱和的判定阈值**（极值重复 >= 3 次、相对容差 1e-12）与**静止段阈值**
   （2% × `percentile(|ω|, 99.5)`）是内部常量，没有做成参数（契约没给入口）。
   实测在合成数据与三段真实录制上都工作正常，但极端数据可能需要调。
6. **`ActuatorSim(seed=...)` 目前不改变任何输出**。见第 5 / 7 节，已如实标注为未实现。
7. **左右关节的镜像符号**。第 8.2 节里左关节偏置符号与参考值相反，判断为录制设备的
   编码器方向约定，不是算法问题。若下游确认参考值本就是编码器帧，那说明这台设备的
   左关节确实反向偏置——两种解释都不影响库本身。

---

## 10. 许可

MIT。见 [LICENSE](LICENSE)。
