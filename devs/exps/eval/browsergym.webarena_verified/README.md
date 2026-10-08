# WebArena-Verified 评测：从这里开始

本目录提供 WebArena-Verified 评测支持：准备环境、执行任务、管理并发沙盒、
记录轨迹、调用官方评分程序，以及归档和分析结果，不以重跑某次历史实验为目标。
当前桌面链路使用 CUA-Lite 原生通用 agent，让模型看截图、操作鼠标和键盘，
在购物、地图、GitLab 等网站上完成任务。基准版本为 1.2.3，共 812 题。

Qwen 和 GLM 是已经验证的模型接入与实验配置，供新评测参考，不是必须选择的后端。
先确定本轮模型和配置，再核对当前入口是否支持；当前全量 initializer 仅提供
这两个模型的预设，其他模型或配置变更可能需要单独适配和回归，不能只改模型名就宣称可用。

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
我的目标是用本仓库开展 WebArena-Verified 评测，不是复现历史 Qwen/GLM 实验。
先确认我本轮要测的模型、API 接入和实验要求；已有实验配置只作参考。
在报告本轮计划时直接展示已验证实验的参考配置，再说明本轮建议和差异。
检查本机代码、凭证配置、共享资产、平台挂载参数和当前入口的支持范围，
说明每项填在哪个文件/参数，自主执行本地和远端只读权限检查并给出证据，
区分已验证、失败和待沙盒验证；需要适配的地方单列，不擅自改代码或替换模型。
不要输出密钥，不要申请沙盒或发起模型推理请求。
```

没有安装 skill 也可以明确要求 Codex 读取这个文件；安装后可使用
`$webarena-verified-eval` 调用。如何继续授权冒烟和全量，见上手指南。

## 操作顺序与实际入口

准备条件 → 克隆配置 → 离线初始化 → 单沙盒验证 → 全量 → 归档分析 → 清理。

初始化使用 `campaign/initialize.py`，阶段管理使用 `campaign/orchestrate.py`，
逐题执行 `scripts/webarena_verified_generic.py`。本目录的旧 `run.sh`、gateway
和模型专用 runner 不是这条完整桌面评测的入口；不需要另起 GPU 推理或 env-server。

当前已验证预设的模型预算、图片历史和站点分配由
[配置模板](/devs/exps/eval/browsergym.webarena_verified/EXPERIMENT_CONFIG.generic-desktop-dual-v2.json)
维护；它是参考与现有入口的输入，不替代用户对新评测的选择。
运行时生成的 config 和 manifest 记录本轮已确认的配置与文件版本。
文档不授权模型费用、资源创建或删除；按本次用户请求执行。
