---
name: webarena-verified-v04-generate
description: Generate and verify entity-bound two-site Shopping and GitLab tasks on fixed WebArena-Verified assets, collect independent Qwen GUI trajectories, assess GLM responses, and export audited V3 JSON and absolute-path JSONL.
---

# WebArena-Verified v0.4 双站数据生成

## 输入与执行边界

需要仓库路径、已授权沙盒与模型凭证文件、目标题数、共享盘输出目录。
当前组合为 Shopping + GitLab；每题两站，一条完整 GUI episode。
先交付首批20题及实际回答分析，再设计前向新增30题至累计50题；用户另有明确规模时遵循该规模。
题目、作者参考、教师成功轨迹、学生评分分别计数，不将目标数量称为已交付数量。

固定资产为 WebArena-Verified 1.2.3 seed。作者、教师与学生每次执行各自重建完整两站业务实例，
执行前只做常规认证，不为某题添加对象或预做业务步骤。模型仅看到公共Query、四字段最终答案协议、
1920×1080全桌面截图、computer与response工具。私有答案、作者计划、HAR、trace和DOM不进入模型输入。
官方812题仅供生成后的污染检查；抽象任务族可来自仓库WebHarbor种子，具体实体与GT必须重新取证。

## 环境与冻结

Linux x86_64、uv、Python3.12；业务Docker和GUI运行时在本批次新申请的沙盒中。
主机不启动Docker。沙盒需挂载 /mnt/webarena-verified，输入subPath为 eval/benchmark/webarena-verified/1.2.3。
凭证仅含 BASE、PROJECT、OPEN_SANDBOX_ACCESS_KEY、OPEN_SANDBOX_SECRET_KEY、OPENAI_BASE_URL、
OPENAI_API_KEY、ASSET_SUBPATH，以及按需QWEN_BASE_URL/QWEN_API_KEY/GLM_BASE_URL/GLM_API_KEY。
PROJECT为 ailab-llmit；读取用户已授权文件，不打印密钥，不将其他测试key自动作为覆盖。

```bash
export PYTHONDONTWRITEBYTECODE=1
export UV_PROJECT_ENVIRONMENT=/tmp/wav-v04-host-env
bash setup.sh
uv run --no-sync python -B pipeline.py --help
```

setup安装锁定主机与固定清洗依赖；随后使用 --no-sync，避免同步移除清洗依赖。
init将完整代码、vendor输入、依赖锁、coordinator和清洗profile冻结到 runs/generated/RUN/release。
后续phase使用该release。新作者实现或评分语义变化必须创建新run，旧结果与原handle保留。
探索run用于选题；作者实现完成后，正式任务run再次init冻结。
管线版本v0.4、固定网站资产1.2.3和轨迹格式multimodal_image_text_dialogs_v3分别记录。

## 探索与中间实体选取

```bash
RUN=unique20261012a
ENVFILE=/absolute/authorized/credentials.env
uv run --no-sync python -B pipeline.py init --run "$RUN" --sites '["shopping","gitlab"]' --env "$ENVFILE"
uv run --no-sync python -B pipeline.py probe --run "$RUN" --env "$ENVFILE"
uv run --no-sync python -B pipeline.py prepare --run "$RUN" --env "$ENVFILE"
uv run --no-sync python -B pipeline.py explore --run "$RUN" --env "$ENVFILE"
```

probe验证教师三轮、学生一轮实际身份，保留原生reasoning_content并冻结配置哈希。
模型为qwen3.8-max与glm-5.3-flash，参数在runtime/models.example.json，最大150步。
额度不足、传输或身份错误阻止正式模型采样；保留真实响应，不换成其他模型制造成功。
prepare申请本scope沙盒、准备两站锁定镜像及共享GUI运行时；两站hostname与会话独立。

exploration/explore-01/observations.json保存有界、只读、可见页面、链接与实体；
截图、HAR和trace用于解释探索结论，不直接冻结GT。来源实体保存实际链接、title/SKU、原始属性、
完整窗口与并列规则；目标项目/Issue保存实际可见URL、IID和title。不得猜测ID或伪造跨站天然关系。
读取相关截图确认字段对GUI可见。实体选择需唯一或完整结果集可确定。
来源的多条件筛选、排名、预算和多输入计算应形成真实依赖，不以无意义站点跳转凑跨站长度。

## 正式作者与任务合同

先读取 infra/author_cross_site.py、explore_cross_site.py 当前CLI与config owner。
每题config是独立JSON对象：身份/模板/家族及抽象种子来源；source的真实可见navigation、kind、
expected_count、max_pages；selection.kind 显式选择 rank 或 budget_pair。
rank 指定过滤、rank_by/direction/rank、title_ascending并列规则；
budget_pair 仅用于已声明的两页商品窗口，指定 budget 和 sorted_title_tuple_ascending，
从两页各取一件，最大化不超过预算的价格合计，同价按排序后的完整标题对决定；
target的真实已有issue_url与title；goal_template、公开给定parameters和note_template。
对方站点的已有对象必须来自探索取证；业务写入只发生在正式任务执行阶段。

```bash
uv run --no-sync python -B pipeline.py ground --run "$RUN" --env "$ENVFILE"   --task wavgen-unique20261012a-7000001 --config /absolute/task-config.json
uv run --no-sync python -B pipeline.py check --run "$RUN" --env "$ENVFILE"
```

正式author在新联合基线上重读来源、确定中间实体、计算事实、提交目标，并重新打开确认持久化。
草案使用sites/start_urls与joint baseline.sites资产映射，不保留singular shape别名。
task_definition拥有task_type/goal_template/parameters/answer/completion_events/dependencies。
整体类型RETRIEVE/NAVIGATE/MUTATE；最终答复task_type/status/retrieved_data/error_details，
检索成功为规定同质数组，导航/修改为null。completion events来自本次真实HAR。

每个跨站dependency绑定来源站、来源实体identity/label参数、来源读取URL、fact_readouts与目标事件。
fact_readouts 为每个事实指定实体内CSS selector、text/input_value读取方式及string/integer/decimal类型；
原始HAR HTML与截图可见HTML按同一类型精确匹配，数量不能通过价格中的数字substring推断。
目标预期从同一具名facts编译。派生结果通过field_evidence.rule绑定全部原始输入和typed
format/sum/multiply/ratio_percent算法；派生值无需在源页面原文出现。
所有答案叶、关键规则及实体绑定都有当前正式observation指针或截图区域。
商品 price_share 的分子为所选商品价格，eligible_price_share 的分子为全部合格商品价格合计；
两者分母均为完整声明窗口的全部原价格合计，不能省略被筛除的商品。
预算组合分别绑定真实 anchor 与 counterpart，并在评论中记录 partner 标题、pair_total 和
pair_total 占全窗口价格合计的百分比，保留两位小数、使用 ties to even。
全部计算输入必须在各自原 episode 中有实际 GUI 可见证据，再通过具名派生图进入目标。
末态checks明确site/readback_url_parameter，预期引用同一semantic parameters；保持非目标状态。

check回放正确执行、错误答案/目标/实体/缺失来源/部分完成/必要顺序负例，HAR与trace同判，
合法等价GUI路径通过。随后官方catalog污染、公私投影与哈希检查，全部pass才冻结准入。
public仅generated_task_id/sites/start_urls/intent，私有task_definition/evaluator与grounding不得泄漏。

## 独立采样、首批分析与前向扩量

```bash
uv run --no-sync python -B pipeline.py sample --run "$RUN" --env "$ENVFILE"
uv run --no-sync python -B pipeline.py clean --run "$RUN" --env "$ENVFILE"
uv run --no-sync python -B pipeline.py export --run "$RUN" --env "$ENVFILE" --count 20 --target /absolute/shared/new-delivery
uv run --no-sync python -B pipeline.py report --run "$RUN" --env "$ENVFILE"
uv run --no-sync python -B pipeline.py release --run "$RUN" --env "$ENVFILE"
```

教师每题最多三个独立语义episode，每次重建完整双站基线，首次owner评分PASS即停止。
学生每题一个episode；provider传输retry2，全部历史attempt保留。没有额外的第四次基础设施重评预算。
语义失败、错误实体、超150步为FAIL；只有教师在三次预算内继续下一独立attempt。
fixed46清洗质量单独记录；清洗REJECT/HOLD不改变原始模型PASS，也不触发成功题再次roll。
传输、额度、审核拦截、缺初态、HAR/trace评分分歧为HOLD，不进入有效分母。
原始教师reference_attempts、学生glm_attempts，保留requests、截图、原生轨迹、canonical tools与receipt。
同一browser context录制来源与目标；封存模型HAR/trace后独立audit，审计不能补足模型读取。

首批实际回答分析区分实体选择、条件迁移、计算、站点切换、目标操作、持久化及格式错误。
成功率PASS/(PASS+FAIL)，HOLD及覆盖率另列；教师success@1与三次内成功率分开，报告各题尝试次数。
整体信号只用于后续新任务设计；
新增30题先冻结规则与配额再采样。旧20题保留Query/GT/评分与全部成绩，不按学生失败筛教师样本。
需要补教师/清洗缺口时新增独立候选并保存预声明选择顺序，不复制或微改旧轨迹凑数。

多个source run通过assemble.py plan/build/clean/export/report汇总；plan采样前固定{run,count}，
选择规则为准入顺序的teacher PASS且fixed46 accepted前N条，student只进入report。
不同teacher快照不能混称同一配置。首批与新增批次分别报告，自适应构造集不能称独立盲测。

## 清洗、共享交付与资源

固定InternFlow0.2.0的46操作、InternValidata0.4.17，三阶段格式修复、格式过滤、GUI行为过滤。
只处理teacher PASS，不按某题调整profile。验证完整原生回转、真实请求与返回model、tool配对、图片尺寸/哈希。
每题json_absolute/*.json与jsonl_absolute/*.jsonl内容相同，合并multimodal_image_text_dialogs_v3.jsonl。
全部消息、native_projection、image_aliases和source_media_archive引用转换为存在共享盘绝对路径；
导出目录不能存在，绑定路径后不能随意移动。审核包包含两站初态、作者/模型证据、release与评分哈希、失败attempt。

所有创建与command先记录intent及原handle。确认的API拒绝保留状态；unknown只核对原nonce/handle，不重复POST。
长任务使用后台session，按实际状态续查，不因观察超时重启。
release前确认本scope命令实际terminal，没有未知提交；只释放本run/nonce拥有的沙盒和实例。
释放后查询全部分页确认对应ID消失，保存release.receipt.json及共享证据。
最终报告真实数量、实体/任务族分布、教师步数、学生PASS/FAIL/HOLD及分母、文件核验和资源状态。
机器验收数据仍pending_human_review、not_promoted。将help/import/offline tests与真实新增采样明确区分。
