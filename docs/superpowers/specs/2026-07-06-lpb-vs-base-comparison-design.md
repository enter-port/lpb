# LPB vs Base Policy 轨迹对比可视化

**Date**: 2026-07-06
**Status**: Approved (pending spec review)

---

## 1. 目标

给定一个初始 state(由 seed 决定),让 LPB 和 base policy 各自独立跑一条完整 rollout,生成**两个对比视频**:

- **Video 1**(base 演化):背景是 base policy 自己跑出来的 rollout,叠加 LPB 的 eef 轨迹(点 + 折线)
- **Video 2**(LPB 演化):背景是 LPB 自己跑出来的 rollout,叠加 base 的 eef 轨迹

每个视频为 2x1 子图(`shouldercamera0` | `shouldercamera1` 并排),scale up 2x → 单帧分辨率 560×280。

折线采用 **累积 + 未来预览** 画法:
- 过去部分(step 0 到 current_step):实线(robot0)/ 虚线(robot1),颜色随时间渐变
- 未来部分(current_step+1 到结束):半透明虚线,颜色渐变
- 当前点:2 个夹爪各一个大圆圈

## 2. 背景

LPB 论文 (arXiv:2508.05941) 在 Transport 任务上报告:base policy 单独使用 ~0.68 成功率,LPB(base + dynamics model + test-time guidance)~0.85。两个方法在**同一个初始 state** 下会产生不同的 action,执行不同 action 后物理状态会分叉。

静态对比图(把两条完整轨迹叠加到初始帧)只能从起点角度看分歧;动态对比视频能看出"在方法 A 实际推进的过程中,方法 B 此刻在哪里、未来会怎么走",更直观地展示分歧的动态过程。

## 3. Non-goals(不做的事)

- **不做**批量评估(一次只跑一个 seed,生成对比视频)。需要批量评估时,用现有的 `eval_test_time_optimization.py` 和 `eval_base_policy.py`。
- **不做**真实机器人部署可视化,只在仿真环境。
- **不修改**现有的 `eval_test_time_optimization.py` / `eval_base_policy.py` / runner 代码。新增独立脚本。
- **不支持**眼在手上的相机视图作为背景(那种相机会随机械臂运动,世界坐标投影无意义)。

## 4. 整体架构

### 新增文件

- `compare_lpb_vs_base.py` — 顶层 Hydra 脚本
- `dyn_model/conf/planner/compare_transport.yaml` — 配置

### 复用代码

- `diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py`:`predict_action` / `predict_action_dyn_guided`
- `diffusion_policy/env/robomimic/robomimic_image_wrapper.py`:`RobomimicImageWrapper`

### 加载代码的处理方式

**复制不重构**:`eval_test_time_optimization.py` 和 `eval_base_policy.py` 里的 policy 加载逻辑是 inline 在各自的 `main()` 函数里。本脚本**复制**这两段加载逻辑(每段约 30 行),不做共享 helper 抽取——避免修改现有 eval 脚本(见 §3 non-goal)。

两段加载逻辑的差异仅在最后是否调用 `policy.initialize_planner(...)`,可以共享前面 90% 的代码(读 ckpt → 建 workspace → load_payload → 取 ema_model → load normalizer)。

### 执行流程

```
1. 加载配置 + 构建 env(单实例 RobomimicImageWrapper)
2. 串行加载策略(避免显存 OOM):
   a. load_base_policy → run base rollout → 存 base_result → 释放 base policy
   b. load_lpb_policy  → run LPB rollout  → 存 lpb_result  → 释放 LPB policy
3. 投影自检(sanity check)
4. 合成 Video 1(driver=base, overlay=lpb)
5. 合成 Video 2(driver=lpb,  overlay=base)
6. 存 mp4 + rollout_data.npz
```

## 5. 详细组件

### 5.1 配置文件 `dyn_model/conf/planner/compare_transport.yaml`

```yaml
hydra:
  run:
    dir: .

# ---- 复用 eval_transport.yaml 的字段 ----
dynamics_model_checkpoint: '/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lpb/data/outputs/2026.06.16/11.15.55_transport/checkpoints/model_60.pth'
policy_checkpoint: '/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lpb/data/outputs/base_policy_6_15/checkpoints/270.ckpt'
guidance_start_timestep: 10
guidance_scale: 0.2
threshold: 2.8
demo_dataset_path: '/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lpb/data/transport/data/expert_demonstration/transport_ph_demo_v141_20_perc.hdf5'
device: "cuda"
planner_target: 'dyn_model.planner.Planner'

# ---- 本脚本特有 ----
compare_seed: 100000              # 初始 state 的 seed
max_steps: 700                    # rollout 长度上限(= task.env_runner.max_steps)
eef_sample_interval: 5            # 每隔几步采一次 eef + 帧(默认 5,~140 个采样点)
render_views:
  - shouldercamera0_image
  - shouldercamera1_image
video_fps: 20                     # 输出 mp4 帧率
video_frame_scale: 2              # 单相机图放大倍数(2 → 280x280)
output_dir: 'data/compare_lpb_vs_base'
```

### 5.2 函数签名

#### `load_lpb_policy(cfg) -> policy`
读 policy ckpt → 建 workspace → load_payload → 取 `ema_model` → 调用 `policy.initialize_planner(...)` 注入 dynamics model + planner。逻辑等价于 `eval_test_time_optimization.py` 的加载段。

#### `load_base_policy(cfg) -> policy`
读同一个 policy ckpt → 建 workspace → load_payload → 取 `ema_model`,**不**调用 `initialize_planner`。逻辑等价于 `eval_base_policy.py` 的加载段。

#### `build_env(cfg, seed) -> env`
从 `payload['cfg'].task` 构建,override `dataset_path` 为 `cfg.demo_dataset_path`(**CLAUDE.md §6.1 必读**:`.ckpt` 里的 stale 绝对路径必须覆盖,否则 `FileNotFoundError`)。返回单实例 `RobomimicImageWrapper`,**显式传 `render_obs_key='shouldercamera0_image'`**(默认 `'agentview_image'` 在 Transport 不存在,见 §10)。调用 `env.seed(seed); env.reset()`。

#### `run_rollout(policy, env, max_steps, use_guidance, sample_interval, view_names) -> dict`

```python
{
  'eef_traj': np.ndarray,    # (T, 2, 3) — 实际步数,robot0 + robot1 的 xyz
  'frames': dict,            # {view_name: np.ndarray (T, H, W, 3) uint8}
  'actions': np.ndarray,     # (T, action_dim)
  'success': bool,
  'success_step': int,       # -1 if not success
  'n_steps': int,
}
```

- `use_guidance=True` → 调 `policy.predict_action_dyn_guided(obs)`(LPB)
- `use_guidance=False` → 调 `policy.predict_action(obs)`(base)
- 每 `sample_interval` 步抓一次 eef + 帧
- 提前 success(`env.get_success_label()` 为 True)则记录 `success_step` 并 break

#### `project_to_camera(points_world, env, camera_name) -> points_px`

世界坐标 (N, 3) → 像素 (N, 2)。实现优先级:

1. 优先尝试 robosuite/MuJoCo 内置 API(如 `sim.camera.world_to_camera_pixels`)
2. Fallback 手算:
   - 从 `env.env.sim` 拿相机外参(`pos`, `mat`)和内参(`fovy`)
   - `p_cam = Rᵀ (p_world − t)`
   - 透视投影:`p_px.x = (p_cam.x / p_cam.z) * fy + cx`,其中 `fy = H / (2 tan(fovy/2))`
3. Sanity check:把 step 0 的 `obs['robot0_eef_pos']` 投到 `shouldercamera0_image`,确认像素位置落在画面中机械臂末端附近

#### `render_video_frame(bg_frame, overlay_traj_world, env, camera_name, current_step, total_overlay_steps, scale, overlay_method) -> np.ndarray (H*scale, W*scale, 3)`

单帧合成(默认用 matplotlib;若性能不足,fallback 见 §10):

1. `ax.imshow(bg_frame)`(背景)
2. 投影 `overlay_traj_world[0..current_step, robot0]` → 实线,时间渐变颜色
3. 投影 `overlay_traj_world[0..current_step, robot1]` → 虚线,时间渐变颜色
4. 投影 `overlay_traj_world[current_step+1..total_overlay_steps-1, robot0/1]` → 半透明虚线,渐变颜色(未来预览)
5. 当前点 `overlay_traj_world[current_step, robot0/1]`:2 个夹爪各一个圆圈
6. 颜色方案(由 `overlay_method` 参数决定):
   - `overlay_method='lpb'` → `plt.cm.plasma`(紫→黄暖色)
   - `overlay_method='base'` → `plt.cm.viridis`(紫→绿冷色)
7. 文字:当前 step / 是否 success
8. 按 `scale` 放大输出

#### `make_video(driver_result, overlay_result, overlay_method, env, view_names, cfg, output_path)`

```python
video_writer = VideoWriter(output_path, fps=cfg.video_fps,
                           frame_size=(W*scale*2, H*scale))  # 2x1 子图

for t in range(driver_result['n_steps']):
    overlay_step = min(t, overlay_result['n_steps']-1)
    left  = render_video_frame(driver_result['frames'][view_names[0]][t],
                                overlay_result['eef_traj'], env,
                                view_names[0], overlay_step,
                                overlay_result['n_steps'], cfg.video_frame_scale,
                                overlay_method)
    right = render_video_frame(driver_result['frames'][view_names[1]][t],
                                overlay_result['eef_traj'], env,
                                view_names[1], overlay_step,
                                overlay_result['n_steps'], cfg.video_frame_scale,
                                overlay_method)
    frame = np.concatenate([left, right], axis=1)  # 2x1 横向并排
    video_writer.write(frame)

video_writer.close()
```

调用:
- `make_video(base_result, lpb_result, 'lpb', env, views, cfg, 'base_driving_lpb_overlay.mp4')`
- `make_video(lpb_result, base_result, 'base', env, views, cfg, 'lpb_driving_base_overlay.mp4')`

### 5.3 数据流

```
cfg ──┬─► load_base_policy ──► run_base_rollout ──► base_result ──┐
      │                       └─► free base policy                 │
      ├─► build_env(seed) ────┤                                    ├─► make_video(driver=base, overlay=lpb)
      │                       │                                    │       → base_driving_lpb_overlay.mp4
      │                       ├─► env.seed(N); reset               │
      │                       │                                    │
      │                       ├─► (reset env to seed N again)      ├─► make_video(driver=lpb,  overlay=base)
      │                       │                                    │       → lpb_driving_base_overlay.mp4
      │                       └─► env used for projection          │
      │                                                            │
      └─► load_lpb_policy ────► run_lpb_rollout ──► lpb_result ───┘
                              └─► free LPB policy
```

## 6. 输出文件

```
{output_dir}/
├── compare_config.yaml                       # 保存的配置
├── base_driving_lpb_overlay.mp4              # Video 1: base 演化,叠加 LPB 轨迹
├── lpb_driving_base_overlay.mp4              # Video 2: LPB 演化,叠加 base 轨迹
└── rollout_data.npz                          # 两条 rollout 的 eef_traj + actions + success
```

## 7. 错误处理 / 边界情况

### 7.1 两个 rollout 长度不一致

- 视频长度 = 驱动方法(driver)的实际步数
- Overlay 轨迹在 `current_step >= overlay_n_steps` 后:
  - 折线不再延长
  - 当前点停在 overlay 的最后一个位置
  - 在画面上叠加文字提示"overlay ended at step X"

### 7.2 驱动方法提前 success

- 视频在 success 那一帧结束
- 最后一帧叠加文字"SUCCESS at step T"

### 7.3 投影点超出画面

- Clip 到图像边界 `[0, W) × [0, H)`
- 折线在出界处**断裂**(不连一条横跨画面的长线),通过检查相邻两点是否都有效来决定是否连线

### 7.4 ckpt 路径 / dataset 路径不存在

- 启动时报错并明确指出哪个路径缺失

### 7.5 GPU OOM

- 策略串行加载(见 §8),每次只占一份 policy 的显存
- 显存仍不够时:`cfg.device='cpu'`(慢但能跑)

## 8. 内存管理(策略串行加载)

两个 policy 同时占显存可能 OOM(base + dynamics model + planner 全在 GPU)。改成串行:

```
1. load_base_policy → run base rollout → 存 base_result → del base_policy; torch.cuda.empty_cache()
2. load_lpb_policy  → run LPB rollout  → 存 lpb_result  → del lpb_policy;  torch.cuda.empty_cache()
3. 合成视频(只需 env.sim 做投影,不再需要 policy)
```

`env` 实例全程保留(`seed_state_map` 缓存初始 state,`sim` 提供投影矩阵)。

## 9. 确定性保证

两个 rollout 必须从**完全相同**的初始 state 出发:

- 用 `seed`(不用 `init_state`)路径:`env.seed(N); env.reset()`
- `RobomimicImageWrapper.seed_state_map` 会缓存 seed→state,第二次 `env.seed(N); env.reset()` 直接从缓存恢复,保证两次 reset 拿到完全相同的世界状态
- **不能用** `init_state` 路径:那条路径每次都 `env.reset()` + 可能存在数值差

## 10. 已知风险

| 风险 | 缓解 |
|------|------|
| **MuJoCo 3D→2D 投影 API 不确定**(robosuite 不同版本接口不同) | §5.2 `project_to_camera` 实现:先试内置 API,fallback 手算;运行时 sanity check 报错 if 投影点远离画面 |
| **`predict_action_dyn_guided` 调用语义** | 直接照抄 `robomimic_image_sequential_runner.run()` 的调用方式,内部维护 `n_action_steps` 缓存 |
| **Transport 双臂物理仿真非线性** | eef_pos 是 3D 笛卡尔,投影简单;task 本身的随机性不是 bug |
| **视频帧渲染慢**(matplotlib 每帧画 4 条 LineCollection + imshow) | 700 帧 × 2 视频。若太慢(<5 fps)改用 opencv 直接画线 + 写帧 |
| **`render_obs_key` 默认值** | RobomimicImageWrapper 默认 `render_obs_key='agentview_image'`,Transport 没这个 key。构建 env 时要传 `render_obs_key='shouldercamera0_image'` |

## 11. 测试方案

### 11.1 脚本内 sanity check

```python
# 加载完两条 rollout 之后,跑一次投影自检
sanity_point_world = base_result['eef_traj'][0, 0]  # base 的 robot0_eef 在 step 0
sanity_point_px = project_to_camera(sanity_point_world.reshape(1, 3), env, 'shouldercamera0_image')[0]
assert 0 <= sanity_point_px[0] < 140 and 0 <= sanity_point_px[1] < 140, \
    f"Projection sanity check failed: {sanity_point_px}"
print(f"[Sanity] world {sanity_point_world} → px {sanity_point_px}")
print("  这个点应该落在 base step 0 帧的 robot0 机械臂末端附近,人工目检视频第一帧确认")
```

### 11.2 人工目检

- 跑 `compare_seed=100000`
- 看两个视频的第一帧:overlay 折线起点应该在画面中机械臂末端附近
- 看折线随时间的渐变:过去部分颜色深,未来部分颜色淡
- 看 success 时视频是否正常结束

## 12. 未决问题(实现阶段再决定)

1. `project_to_camera` 用哪个具体 API — 实现时根据 robosuite 版本试
2. 视频写库用 `imageio` + `imageio-ffmpeg` 还是 `opencv` — 看哪个已安装在 lpb 环境
3. matplotlib 画图慢的话,要不要切换到纯 opencv 画 — 跑一遍看 fps

## 13. 未来扩展(不在本次范围)

- 支持多个 seed 批量生成
- 支持自定义 `init_state`(用 demo 的某一帧作为起点)
- 支持指定 ckpt 路径列表(对比不同 epoch 的差异)
- 输出 3D 轨迹的独立 npz / 轨线图(不依赖相机投影)
