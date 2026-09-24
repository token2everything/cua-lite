# Sandbox 公共能力

工作站负责提交、调度和结果同步；Agent 与环境在 Sandbox 内运行。

| 文件 | 职责 |
|---|---|
| [client.py](/lite/platforms/sandbox/client.py) | 平台 API、AFS 传输、凭据加载、脱敏和飞书配置 |
| [lifecycle.py](/lite/platforms/sandbox/lifecycle.py) | 创建记录、所有权校验、飞书实际配置校验和删除 |
| [batch.py](/lite/platforms/sandbox/batch.py) | 并发队列、进度保存、恢复、超时、续期及删除确认 |
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
