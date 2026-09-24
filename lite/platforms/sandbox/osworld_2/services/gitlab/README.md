# Task026 的 GitLab 服务

GitLab 与 Task026 的 Worker、桌面 VM、TeamChat/Caddy 放在**同一个 Sandbox** 中，
使用本地 Docker volumes 保存这次任务的用户和仓库。无需另建 GitLab Sandbox。
本目录提供服务启动与初始化工具；独立 Agent 镜像方案下的完整 Task026 仍需实机回归，
不能只修改任务 ID 就认为网站、GitLab 与模型链路已就绪。

| 运行部分 | 配置 |
|---|---|
| 外层 Sandbox | Task026 部署预留 `32Gi`，包含所有嵌套容器与 VM |
| GitLab 容器 | launcher 固定 8 GiB，memory 与 memory-plus-swap 上限相同；2 个 Puma worker、Sidekiq 并发 10 |
| Agent 容器 | 通用 Agent 镜像，单独限制为 8 GiB |
| 桌面 VM | `OSWORLD_VM_RAM_SIZE=4G`；不随外层内存调整而改变 |

[launch.sh](/lite/platforms/sandbox/osworld_2/services/gitlab/launch.sh) 支持已有 Docker
daemon，也可在缺少时启动；共享 daemon 的生命周期由外层部署进程管理。
[init.rb](/lite/platforms/sandbox/osworld_2/services/gitlab/init.rb) 完成 GitLab 初始化与健康检查。
[image.json](/lite/platforms/sandbox/osworld_2/services/gitlab/image.json) 保存官方来源与
内部镜像的固定引用；部署时将其中 `pinned_image` 设为 `OSWORLD_GITLAB_IMAGE`。
初始化要求 GitLab `18.7.0`。外层需提供 Bash、Docker/dockerd、wget、timeout 等脚本依赖。

## 维护者接入步骤

1. 按[部署指南](/lite/platforms/sandbox/osworld_2/README.md)准备匹配 release 的
   benchmark 镜像、任务数据与模板，固定 `OSWORLD_WORKER_IMAGE`、`CUA_AGENT_IMAGE`
   digest。保留飞书、Worker/data/results 挂载及本次任务独立的结果位置。
2. 在新的 `benchmarks/osworld/worker-builds/<build-id>/input/` 发布服务文件和外层
   启动脚本。模板把整个 build 目录只读挂到 `/opt/osworld-build`。一个可采用的发布布局是：

   ```text
   input/
     launch_worker.sh           # 当前生产 Worker launcher
     launch_rollout.sh          # 维护者的服务编排入口
     run_worker_agent.sh        # 当前生产 launch_rollout.sh 的原样副本
     services/gitlab/
       launch.sh
       init.rb
       image.json
   ```

3. 编排入口先关闭模板注入的 execd 命令服务，再启动 Docker、匹配 release 的
   TeamChat/Caddy 和 GitLab。设置 `WEBSITE_HOST_SUFFIX`，等待网站可用。
   为 GitLab 创建本次 Sandbox 独有、可写的输出父目录，例如
   `OSWORLD_GITLAB_OUTPUT_ROOT=/results/_workers/<worker-id>/gitlab`，以关闭 shell
   tracing 的方式启动 `/opt/osworld-build/input/services/gitlab/launch.sh` 并记录子进程 PID。
4. 等待 GitLab 的 `run/readiness.json` 为 `ready:true`，且 `run/connection.env` 已发布。
   只在可信编排进程中 source 该文件并 export `GITLAB_URL`、`GITLAB_PRIVATE_TOKEN`，
   再调用 `/opt/osworld-build/input/run_worker_agent.sh`。当前
   [Worker launcher](/lite/platforms/sandbox/osworld_2/launch_worker.sh) 会把这两个变量
   传入 Worker；[生产 rollout launcher](/lite/platforms/sandbox/osworld_2/launch_rollout.sh)
   会继续启动 coordinator 与独立 Agent。
5. 核对 Worker、桌面 VM 到 GitLab 与 TeamChat 的连通性，验证 Task026 的初始化、
   交互、原生评分和清理。通过后，操作者才能在 `run.yaml` 选择该模板和 `job.tasks: ["026"]`，
   沿用 `plan → submit → sync → delete`。

上面的编排脚本由部署维护者交付，仓库的普通 rollout 入口不会自动部署网站或 GitLab。
`manage submit`/带 `--agent-job` 的 create 固定调用
`/opt/osworld-build/input/launch_rollout.sh`，因此服务入口要发布在这个位置；仅修改
Environment 的 entrypoint 不会覆盖它。入口随后调用生产副本，避免递归调用自身。
这些脚本随 benchmark 部署发布，Agent 使用现有 OCI 镜像，不另准备 Python 依赖归档。

GitLab 默认外层 HTTP 端口为 **8929**（`OSWORLD_GITLAB_PORT` 可覆盖），Git-over-SSH 为
2222；TeamChat/Caddy 使用 80。GitLab 对外地址来自 Sandbox 的非 loopback IPv4。
Worker 和桌面 VM 都必须可直连该地址，代理规则需绕过它。

## 初始化与凭据边界

GitLab launcher 为本次服务生成管理员密码与 token，Rails 在嵌套容器内写私密文件；
检查通过后再复制到外层输出目录。`run/` 为 0700，连接文件为 0600。
Task026 使用这些管理接口完成官方 setup/evaluation，并创建任务所需的普通 GitLab 账号。

管理员凭据只交给可信编排进程与 Worker，不挂入 Agent 或桌面 VM，不加入模型 prompt、
轨迹或共享日志。不要公开输出目录或启用 `set -x`。带 Docker socket 的编排进程属于可信
管理进程；容器挂载差异不是抵御该进程访问宿主机数据的安全边界。

初始化检查登录页、管理员 API 与 `/-/readiness?all=1`；launcher 还从外层检查对外地址的
登录页。运行中通过相同 `init.rb --health` 周期性复查，失败时将 readiness 标为 false，
保留诊断且不自动重启；恢复后可重新变为 ready。判断状态同时查看 readiness 与采样时间，
不能只看 `status.json.stage` 是否曾到达 `ready`。

## 输出与收尾

以下文件位于 `$OSWORLD_GITLAB_OUTPUT_ROOT/run/`：

| 文件 | 用途 |
|---|---|
| `connection.json` / `connection.env` | 私密管理员连接信息；不打印、不共享 |
| `readiness.json` / `status.json` | 当前可用性、阶段、退出码与更新时间 |
| `bootstrap.json` / `health.json` | 初始化及最近一次组件检查结果 |
| `runtime.json` / `runtime.jsonl` | 最近状态与运行采样历史，包括资源、OOM 和容器状态 |
| `outer-http.json` | 外层访问 GitLab 地址的检查结果 |
| `image.json` / `daemon.json` | 实际镜像身份与 daemon 所有权 |

`health.json.cgroup_memory` 描述 GitLab 容器，`runtime.json.outer_cgroup_memory`
描述外层 Sandbox，均不是桌面 VM 内存。诊断只保留允许的状态字段，不记录 HTTP headers、
响应正文或私密凭据。

收尾先停止本次 Worker/Agent 流程并等待任务关闭，再向记录的 GitLab launcher PID 发 TERM
并等待退出。launcher 通过唯一 owner label 核对并移除自己的 GitLab 容器、三个数据卷和
连接文件，保留诊断；不停止共享 Docker daemon。外层编排者待所有子进程结束后再回收
自身 daemon 与 Sandbox。不要按通用容器名批量删除，也不要回收别人的服务。

每个 release 分别完成[验证要求](/lite/gym/envs/osworld_2/WORKER.md#verification-and-extension)，
保存协议证据与模型得分。已有镜像发布记录或 GitLab 健康检查成功，不能替代新部署的
Task026 验证。
