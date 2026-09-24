---
orphan: true
---

# ADR-0010 Experimental Substep Observation

- Status: Experimental (fork only)
- Date: 2026-09-23
- Owners: Cricket-Gym-UniLab fork
- Supersedes: None
- Superseded by: None

## Context

G1 完整接触审计发现：40 次有接触的试验中，9 次接触完全落在 20 ms 控制采样之间，
没有进入分离奖励。任务需要读取每个物理子步的已求解接触及积分后速度，而不是提高策略频率。
当前 UniSim 的控制回调每个子步清空 warm-start；将它用作只读观察器会改变数值执行语义。

## Decision

本决定只覆盖用户授权的 fork 实验分支，不表示上游已接受新接口或生产支持承诺。
在 UniLab adapter 层声明 `SubstepObservationBackend`，通过
`mujoco_observe_substeps` 显式选择本地 MuJoCo 兼容实现。默认 factory 路径保持不变。
长期接口及引擎实现仍归 UniSim 所有；向上游提交前需另行取得批准。

兼容实现使用公开的 `mujoco.rollout.Rollout` 获取每个子步的状态和传感器轨迹，
复用 pool 实际持有的逐环境模型。只进行一次权威物理推进，不通过第二次回放生成训练奖励。
不修改 site-packages、不复制本地依赖源码，也不向第三个仓库提交。

## Stable Contracts

- 观察器在完整物理区间结束后、manager 奖励计算前调用一次。数组顺序为环境、子步、通道。
  每个子步都有样本，包括最后一个；没有初始 reset 样本。
- 命名传感器是该子步 `mj_step` 的求解阶段输出；自由根体的世界线速度是同一子步的积分后值。
  只读数组在回调期间有效；保留数据需要复制。回调不允许修改控制或状态。
- warm-start 只在控制区间入口清零；区间内保留。中间状态不转成 float32 再输入物理引擎。
  最终缓存沿用原来的 dtype。所有未指定的控制量和 pending wrench 遵循原 backend 语义。
- 这个实验路径不支持逐子步控制回调、额外 `mj_forward` 传感器刷新或逐线程 CPU affinity。
  不支持的组合明确拒绝，不静默改变含义。
- 奖励只消费声明的观察能力，不访问 engine model/data/private runtime。
  旧环境、奖励、模型、观测维度及 20 ms 策略频率不变。

## Alternatives Considered

- `BatchEnvPool.control_callback`：每步清空 warm-start，不是只读记录路径。
- 连续调用 `step(nsteps=1)`：改变 warm-start 和中间 dtype 边界。
- 奖励里读取 MuJoCo 私有对象或运行独立回放：违反 owner 边界，且不再是实际训练轨迹。
- 直接修改已安装的依赖：不可复现，也无法在已有授权的两个 fork 中评审。

## Consequences

轨迹记录增加临时内存和复制开销；不能宣称与旧路径等成本。模型、接触参数、策略与控制时序
必须通过真实接触区间的精确状态/传感器比较后，才能把奖励采样作为唯一实验轴。
验证覆盖多环境、外力、局部 reset、最后子步、跨区间接触和单次奖励。
这些验证不构成材料校准或成功击球证据。

## Evidence In Repo

- `src/unilab/base/backend_substeps.py`
- `src/unilab/base/mujoco_substeps.py`
- `src/unilab/base/backend_factory.py`
- `src/unilab/tasks/manipulation/g1_cricket/substep_reward.py`
- `tests/base/test_mujoco_substeps.py`
- `tests/envs/test_g1_cricket_substep_reward.py`
- `g1_cricket_results/impact_v1/impact_reward_audit.json`

## Related Documents

- {doc}`UniSim boundary </adr/ADR-0007-unisim-extraction-boundary>`
- {doc}`Collaboration workflow </en/4-developer_guide/5-contributing_workflow>`
