# WebArena-Verified 评测：从这里开始

这套流程让模型看桌面截图、操作鼠标和键盘，在购物、地图、GitLab 等网站上
完成 812 个任务，再由官方评分程序判断是否完成。基准版本固定为 1.2.3。
已验证的后端是 Qwen 和 GLM；两者使用同一个 CUA-Lite 原生通用 agent。

## 你应该读哪份文件

| 你的目的 | 入口 |
| --- | --- |
| 第一次接手，想知道准备什么、怎样跑起来 | [快速上手指南](/skills/webarena-verified-eval/references/reproduction.md) |
| 让 Codex 帮你检查环境、运行或分析 | [Codex 执行说明（skill）](/skills/webarena-verified-eval/SKILL.md) |
| 核对配置、镜像、资产、异常或历史结果 | [技术参考](/skills/webarena-verified-eval/references/technical-reference.md) |
| 在线阅读并与同事协作 | [飞书上手指南](https://gcnxubmy1vgu.feishu.cn/docx/YnArdkqJooQ8MOxO4UMcq3wLnCb) |

主线面向复用现有沙盒平台、镜像和共享资产的同事。克隆代码不会自动取得
这些权限或下载大型数据；先按上手指南确认准备条件。换平台部署看技术参考。
飞书权限与代码仓库权限独立。

## 可以直接交给 Codex 的第一句话

```text
请读取 skills/webarena-verified-eval/SKILL.md，按快速上手指南接手评测。
使用 Qwen 配置，先检查本机代码、凭证配置、共享资产和平台挂载参数，
列出缺项并给出实验计划。不要输出密钥，不要申请沙盒或调用模型。
```

没有安装 skill 也可以明确要求 Codex 读取这个文件；安装后可使用
`$webarena-verified-eval` 调用。如何继续授权冒烟和全量，见上手指南。

## 操作顺序与实际入口

准备条件 → 克隆配置 → 离线初始化 → 单沙盒验证 → 全量 → 归档分析 → 清理。

初始化使用 `campaign/initialize.py`，阶段管理使用 `campaign/orchestrate.py`，
逐题执行 `scripts/webarena_verified_generic.py`。本目录的旧 `run.sh`、gateway
和模型专用 runner 不是这条完整桌面评测的入口；不需要另起 GPU 推理或 env-server。

模型预算、图片历史和站点分配由
[配置模板](/devs/exps/eval/browsergym.webarena_verified/EXPERIMENT_CONFIG.generic-desktop-dual-v2.json)
维护；运行时生成的 config 和 manifest 记录本轮实际采用的配置与文件版本。
文档不授权模型费用、资源创建或删除；按本次用户请求执行。
