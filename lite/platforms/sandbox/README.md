# Sandbox 公共能力

工作站负责提交、调度和结果同步；Agent 与环境在 Sandbox 内运行。

| 文件 | 职责 |
|---|---|
| [client.py](/lite/platforms/sandbox/client.py) | 平台 API、AFS 传输、凭据加载、脱敏和飞书配置 |
| [lifecycle.py](/lite/platforms/sandbox/lifecycle.py) | 创建记录、所有权校验、飞书实际配置校验和删除 |
| [batch.py](/lite/platforms/sandbox/batch.py) | 并发队列、进度保存、恢复、超时、续期及删除确认 |
| [monitor.py](/lite/platforms/sandbox/monitor.py) | Agent 停滞判断、批次汇总、飞书折叠卡片及通知节流 |
| `<benchmark>/` | 任务计划、启动配置、环境完成信号、结果读取 |

公共模块不导入具体 benchmark。OSWorld 的参考实现是
[manage.py](/lite/platforms/sandbox/osworld_2/manage.py) 和
[batch.py](/lite/platforms/sandbox/osworld_2/batch.py)，日常运行见
[批量评测示例](/lite/gym/envs/osworld_2/WORKER.md#自动拆分与并发评测)。

## 平台连接

```python
from lite.platforms.sandbox.client import SandboxClient, load_credentials

client = SandboxClient(
    base_url="https://manager.example",
    project="my-project",
    afs_url="https://afs.example",
    credentials=load_credentials(),
)
```

凭据来自显式文件路径，或 `OPEN_SANDBOX_ACCESS_KEY` /
`OPEN_SANDBOX_SECRET_KEY`。客户端初始化不联网、不创建资源。
飞书优先继承模板，或从显式配置文件、`SANDBOX_FEISHU_WEBHOOK` 读取；
不搜索其他 benchmark 的私密目录。只有明确调用 `feishu_test` 才发送测试消息。

## 批量接入

接入者生成任务计划，再传入一个实现 `BatchAdapter` 三个方法的普通对象；
无需继承基类或注册插件：

```python
from lite.platforms.sandbox import batch

state = batch.prepare(args, deployment, build_plan)
# --dry-run 时输出 state；实际运行时才加载平台凭据并构造 client。
result = batch.run(args, client, state, adapter)
```

`args` 提供 `state_dir`、`resume`、`concurrency`、`poll_interval` 和
`task_timeout`；后三项传 `None` 时使用默认值。
`deployment` 包含项目、API/AFS 地址、模板 ID、结果前缀和 benchmark 的版本身份。
恢复时必须匹配原部署，不重新调用 `build_plan`。

`build_plan()` 返回：

```python
{
    "batch_id": "unique-batch-name",
    "job": {"model_config": "/private/model-api.json"},  # 可加 model_profile
    "tasks": [
        {"task_id": "task-a", "experiment": "unique-task-a", "state": "pending"}
    ],
}
```

这是 API Agent 工作流：公共队列冻结选中模型配置，恢复后继续使用该私密快照。
任务选择、release、排除清单等可作为额外字段保留，由 benchmark 自己解释。

| 方法 | benchmark 负责的操作 |
|---|---|
| `submit(item)` | 构造并上传私密作业，创建实例，返回已验证飞书的所有权记录 |
| `observe(item, record)` | 读取执行回执；确认执行结束时返回 `True`，实例消失后仍可读取持久化回执 |
| `collect(item, record)` | 同步产物，写入 `item.workflow_completed`、得分及错误原因 |

适配器可补充自己的进度字段；`item.state` 的流转由公共队列维护。
完成判定不依赖模型得分。正常结束先同步再删除；超时先终止实例再收集；
只有确认 Sandbox 消失才释放并发名额。

轮询、续期或回收时遇到网络超时、限流或临时服务错误，记录 `item.poll_error`，继续处理
其他任务，下一轮核对原 Sandbox 的实际状态后再继续；不重新创建任务。卡片会显示查询
异常。鉴权、所有权或产物校验错误仍停止调度并保留现场，避免错误回收。

### 接入 Agent 进度卡片

入口把 YAML 的 `monitor` 传给 `batch.prepare(..., monitor=config.get("monitor"))`；
实际运行时解析已有飞书配置，构造 `BatchMonitor(webhook, client.secrets)`，通过
`batch.run(..., monitor=monitor)` 传入。未启用时不发送批次卡片，Sandbox 本身的飞书扩展
照常保留。配置示例见 [run.yaml](/lite/platforms/sandbox/osworld_2/configs/run.yaml)。

benchmark 的 `observe` 将自己的产物映射到以下字段即可，不用各写一套通知：

```python
item["progress"] = {
    "turns": 12,                       # 累计已完成的 Agent 步数，跨阶段不清零
    "last_progress_at": 1790244010.0,  # 最近完成步骤的 Unix 时间；查询或重试不能刷新它
    "observed_at": 1790244030.0,       # 本次成功采集时间
    "activity": "waiting_model",      # starting / waiting_model / executing /
                                      # between_phases / finished
    "phase_index": 0, "phase_count": 1,  # 可选，index 从 0 开始
    "phase_turn": 12, "max_steps": 500,  # 可选，本阶段预算
}
```

进度读取须有时限；出错时保留上次数据，设置脱敏后的 `item.progress_error`，恢复后删除
该字段。它只表示监控不可用，不写入流程失败的 `item.error`，也不触发资源回收。
读取现有小型记录即可，避免轮询时下载截图、完整请求或轨迹。OSWorld 的实现直接读取
累计步数、步骤事件尾部及模型日志文件名，无需重建运行镜像。

公共层在开始、恢复、结束和调度异常时通知；运行中定期汇总，异常出现或恢复时提前通知，
这些变化最多每分钟合并一次。Webhooks 发送新卡片快照，不更新旧消息；折叠不需要回调服务。
扫描大批任务时，每处理一项都会检查通知时间，不必等整轮扫描结束。
通知结果保存到 `batch.json.notification`，失败记录脱敏原因，下次轮询按节流规则重试，
不阻断执行。结束或调度停止后没有下一轮，因此发送失败仅额外重试一次；若仍失败，
原因留在 `notification.error`。网络超时后重试可能产生重复快照。模型得分 0 与流程失败分开显示。

通知由调度进程发出。进程被强杀、机器断电或长时间阻塞时不能继续上报；需要用 tmux 或
进程管理器保持调度器运行。这不是监测调度器自身存活的独立服务。

## 提交记录约定

复用 `lifecycle.create`、`verify_feishu` 和 `delete`，不要另写平台删除循环。
每个 benchmark 指定稳定的 `owner`。沿用的 `worker_run` 是每次创建的随机所有权标识。

- 创建前，在 `state_dir/jobs/<experiment>/submission.json` 保存提交记录及
  `worker_run`，然后才上传输入、调用创建 API。创建响应丢失时不自动重试。
- `lifecycle.create(..., details={"submission": submission, ...})` 立即保存
  `worker-<sandbox-id>.json`。其中 `submission` 至少包含
  `experiment`、`batch_run`、`afs_url`、`results_prefix`；`batch_run` 使用计划中的值。
- 保存所有权后调用 `lifecycle.verify_feishu`，成功后才把记录返回队列。
  任务文件格式和上传目录由 benchmark 决定；凭据、Webhook 不进入公开输出。

传输、共享队列及各 benchmark 的离线测试位于
[tests/platforms/sandbox](/tests/platforms/sandbox)。共享队列测试包含不使用 OSWorld
任务、Worker 接口或产物格式的适配器。
