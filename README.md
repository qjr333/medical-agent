# 银龄守护——老年高血压主动随访 Agent

保留现有 FastAPI 接口、前端、Agent Loop、医学 Tools、SQLite 和 Patient Memory。
风险阈值、危险症状识别与用药安全拦截沿用原实现。

## 目录与职责

```text
app.py                         FastAPI 初始化、原有路由及兼容适配
agent/
  runner.py                    Agent Loop、工具执行和流程控制
  planner.py                   状态驱动的动作优先级与结构化决策原因
  question_planner.py           主动追问及当前目标
  state_manager.py              Patient State 创建、合并与 Trace
  extractor.py                  上下文提取与回答可用性识别
  reporting.py                  随访报告、结束语与升级提示
medical_tools/
  safety.py                    原风险阈值、红旗检测和读数校验
  bp_trend.py                  血压历史与趋势
  medication.py                用药指导与安全拦截
  visit_advice.py               就医分级与证据整合
  guidance.py                  RAG 查询与本次会话检索缓存
  alerts.py                    告警内容
  registry.py                  统一工具注册表
memory/patient_memory.py        Patient Memory V2
database/
  connection.py                SQLite 连接与原表结构初始化
  repository.py                患者、观察、会话、Trace、告警、报告
backend/
  config.py                    原环境、路径与演示患者配置
  model_client.py              DeepSeek 客户端、JSON 解析与调用回退
  utils.py                     通用时间、JSON 与列表辅助
  extraction.py                第一阶段导入路径的兼容入口
  patient_state.py             第一阶段导入路径的兼容入口
db.py                          原数据库导入路径的兼容入口
rag.py                         第一阶段向量缓存与懒加载实现
tests/                         API、Agent、工具、存储、缓存回归
```

API 调用 Agent 与业务模块；Agent 通过统一 Tool Registry 执行医疗工具。
Patient Memory 从 Repository 读取历史事实，Repository 使用 Connection 操作 SQLite。
业务模块不导入 `app.py`，没有循环依赖；未增加第三方依赖。

SQLite 的 `state_json` 继续作为会话持久化来源，进程内 `sessions` 仅作缓存。
测试或嵌入调用需要隔离数据库时，配置 `database.connection.DB_PATH`；
`db.py` 仍保留原 `get_connection()`、`init_db()` 和路径常量。

## 启动

在项目目录执行：

```sh
source .venv/bin/activate
# 首次配置环境时：pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

打开 http://127.0.0.1:8000；接口文档在 `/docs`。
继续使用原 `.env`、`demo_patient.json`、`knowledge/` 和 `data/medical_agent.db`。

## 向量缓存

- 默认保存到 `data/rag_cache/knowledge_embeddings.npz`，可用 `RAG_CACHE_DIR` 指定目录。
- 启动只读取知识块与有效缓存，不导入或加载 Embedding 模型。
- 首次知识检索才加载模型；缓存缺失或失效时生成知识块向量并写入磁盘。
- 后续进程启动复用缓存，检索时仍需懒加载模型以生成查询向量。
- 知识 Markdown、来源注册表、模型配置或索引格式变化时缓存失效。
- 编辑知识库后重启应用，或调用现有 `POST /api/rebuild-knowledge`；内容未变时复用缓存。
- 模型不可用时保留关键词检索回退；`RAG_DISABLE_EMBEDDING=1` 可主动使用该模式。

`/api/knowledge-status` 保持原字段。`embedding_available` 表示当前进程模型是否已加载成功，
所以懒加载前为 `false`，不表示缓存失效。首次检索可能需要等待模型加载与缓存建立。

## 测试

```sh
.venv/bin/python -B -m unittest discover -s tests -v
```

测试使用临时 SQLite 与缓存目录，LLM 和 Embedding 模型由测试替身隔离，
不联网、不修改现有患者数据。覆盖主动随访、聊天闭环、重启后的会话恢复、数据库、
Patient Memory、医学 Tools、安全拦截，以及缓存首次构建、复用和失效。
重启测试实际运行两个独立 Python 进程，恢复同一会话并继续到报告完成。

第二阶段共 79 项自动化回归测试通过。DeepSeek 的 JSON mode、普通模式回退和
调用失败回退由 Mock 验证；本批未向真实 DeepSeek 服务发出请求。
第一阶段的 `rag.py`、知识库和磁盘向量缓存保持原样。
另用本地真实 `BAAI/bge-small-zh-v1.5` 离线验证了新 Tool Registry 的 RAG 调用链：
启动保持懒加载，复用 44×512 向量缓存，知识块编码 0 次、查询编码 1 次，缓存文件未变。
未单独执行浏览器端到端交互测试；现有前端文件与 API 契约均保持一致。

## 第三阶段：上下文提取和追问

- 简单回答先按规则提取，结合当前目标、上一题的语义和本轮已确认事实。
  “没有”回答服药、漏服和症状问题时分别解释；历史记忆不填补当前状态。
- `not_taken_yet` 表示尚未服用，不能等同漏服；时间未知时追问原定服药时间是否已到。
  只有明确报告忘记/漏服才记 `missed`，时间未到或超过时保留真实报告状态。
- `topic_status` 区分 `unanswered`、`answered` 和 `clarification_needed`，并同步 Coverage。
  `question_context`、`last_agent_question` 和 `question_attempts` 保存问题主题、类型及已问项目，
  澄清用新问题，已确认项目跳过。危险症状仍先升级。
- 否定限定于分句和明确问到的症状；“没有胸痛，但是呼吸困难”保留危险表现。
  部分症状否认不会排除全部红旗，补问剩余项目。
- 模糊回答可调用 DeepSeek，拒绝模型新增的未证实事实或 Coverage；补充血压需要
  对应高压/低压标签、原文数字及可验证证据。汉字数字等更多表达本批未扩展。
- 服药观察按中国时区日期保存；历史/未来服药信息不覆盖今天。
  新字段存在原 `state_json`，SQLite 仍用原七张表；`recorded_at` 保留采集时间，
  `detail` 标注服药日期。旧会话从 Trace 恢复上一题和日期，无日期证据的服药事实保留待核实。
- 同日重复报告未提供时间时不覆盖已确认的服药时间；已结束随访复用报告，不重复生成。
  Patient Memory 单独统计 `not_taken_yet`，不计入漏服/停药问题条数。

本批修改：`app.py`、`agent/extractor.py`、`agent/state_manager.py`、
`agent/question_planner.py`、`agent/planner.py`、`agent/runner.py`、`backend/utils.py`、
`database/repository.py`、`memory/patient_memory.py`、`tests/test_backend_modules.py`、本 README。
新增 `tests/test_context_extractor.py`、`tests/test_context_questions.py`、`tests/test_context_api.py`。
目录结构沿用前两阶段，没有新增业务包或第三方依赖。

只验证第三阶段修复可执行：

```sh
.venv/bin/python -B -m unittest discover -s tests -p 'test_context*.py' -v
```

手工验证：启动后创建定时随访，依次回复“还没吃”→“还没到吃药时间”→“都没有不舒服”。
应依次出现时间澄清、症状询问、测量环境询问；服药状态保留 `not_taken_yet`，不判漏服。
另建随访回复“没有胸痛，但是呼吸困难”，应立即进入高风险升级流程。
所有自动化用例使用模拟患者、临时 SQLite 和 Mock 模型，本批未测试真实 DeepSeek 网络调用。
第三阶段新增 60 项回归，连同原有 79 项共 139 项全部通过，包含 API 启动与健康检查、
多轮聊天、重启恢复、Patient Memory、BP Trend、Risk Engine、Visit Advice、Agent Trace，
以及 RAG 缓存首次建立/再次读取和失效重建。
提取模块离线抽查 1,000 次简单回答：模型调用 0 次，中位约 0.07 ms、P95 约 0.14 ms；
这不是完整聊天接口或真实模型请求的延迟。

## 第四阶段：Planner V3 / Agent Loop V3

基于现有模块实现轻量确定性决策，没有引入完整的大模型自主规划框架。

- Planner 返回 `action/goal/reason/priority/required_tools/basis`，保留原
  `select_next_action()` 字符串入口。异常血压、当前不适或连续上升趋势先核实症状；
  历史漏服只促使核实本次服药，不改写今天的事实。主动医学问题用 RAG 和已有医疗工具回答。
- Runner 沿用 Tool Registry，每轮先运行确定性 Safety Layer，工具执行中关键状态变化时重评。
  原医疗阈值与处方药拦截保留；系统升级信号不等于诊断高血压急症。
- 工具复用依据实际输入、患者事实、医学证据和知识版本；不以执行次数或文本相同单独判断。
  空 RAG 结果也记为已尝试，避免空证据循环。工具异常本轮安全回退，下一次执行允许重试。
  知识更新仍沿用现有重启或 `POST /api/rebuild-knowledge` 流程，磁盘向量索引机制未改。
- 原医学问题等待补充药名/原方案时保留待办；重复原问不重复调用，收到新信息才继续。
  旧问题已处理后被消费，状态变化会使旧 Medication Guidance / Visit Advice 失效。
- 每轮默认最多 12 步；达到上限保存下一动作、目标和未完成项，状态为 `interrupted`，不生成报告。
  原 `/api/chat` 可在补充信息后恢复；`completed/escalated` 保持封闭并返回 409。
- 拒答/无法回答暂缓该主题，连续不明确最多 3 次后暂缓；未确认事实保持未知。
  必要信息或医学问题仍未解决时中断，不将中断当成完成；明确新事实可以解除暂缓。
- DeepSeek 使用 20 秒请求超时、禁用 SDK 隐式重试，保留 JSON mode→普通模式的兼容回退。
  RAG / Medical Tool 异常使用既有保守规则；安全评估不能可靠完成时中断。
  SQLite 写失败返回 503，不宣称保存成功，未落盘的普通状态不发布到会话缓存。
  读写失败仍对新危险输入执行原安全规则并返回升级提示；SQLite 不可用时不能保证落盘。
- Trace 保留原 `time/event/detail` 结构，增加决策原因、目标、简明依据、工具结果/错误类型、
  耗时及下一状态，不重复保存完整 Patient State。报告补充必要信息确认度及精简异常摘要。
  Patient Memory 继续把中断随访当作未完成任务，历史信息不覆盖新随访事实。

第四阶段修改：`agent/planner.py`、`agent/question_planner.py`、`agent/runner.py`、
`agent/state_manager.py`、`agent/extractor.py`、`agent/reporting.py`、`app.py`、
`medical_tools/guidance.py`、`medical_tools/medication.py`、`medical_tools/visit_advice.py`、
`backend/model_client.py`、`database/repository.py`、`memory/patient_memory.py`、本 README。
更新既有 `test_agent_modules.py`、`test_api_regression.py`、`test_context_api.py`、
`test_medical_tools.py`、`test_storage_modules.py`；保留原测试方法，适配第四阶段真实优先级。
新增 `test_planner_v3.py`、`test_task_state_v3.py`、`test_agent_v3_api.py`、`test_agent_v3_runner.py`。
业务目录沿用前三阶段，没有新增依赖、前端或数据库表。

第四阶段新增 69 项，合计 **208 项自动化测试全部通过**。覆盖正常/异常随访、近期上升趋势、
历史漏服隔离、未服药澄清、危险症状中断、医学问答、空 RAG、模型超时、工具/SQLite 失败、
重复问题、状态变化后的证据失效、有限追问、步数上限、恢复、封闭会话、报告幂等与 Trace。
自动化用例使用 Mock 模型和临时 SQLite，真实 DeepSeek 网络调用未验证。
另用本地真实 `BAAI/bge-small-zh-v1.5` 离线验证工具检索：复用44×512磁盘向量，
知识块编码0次、查询编码1次，相同条件第二次直接复用，缓存文件哈希未变。
本地环境仍有既有 NumPy / LibreSSL 警告；真实检索返回有效结果，本批未调整依赖环境。

仅运行新增第四阶段测试：

```sh
.venv/bin/python -B -m unittest discover -s tests -p '*v3*.py' -v
```

完整回归仍执行上面的 `unittest discover -s tests -v` 命令。
手工验证：使用默认162/96定时随访，应先询问危险症状；回答“都没有”后再核实服药。
普通场景135/85、服药与症状已确认后可直接完成；中断会话补充明确资料后可继续。

## 下一阶段

本批完成后停止，不自动进入第五批。
后续可安排真实 DeepSeek 服务联调、浏览器交互、更多真实对话样本与并发/长期运行验证。
原临床规则与风险阈值仍需专业审核；本批没有调整这些规则。
Trigger Engine V2、独立 Embedding 服务、新前端、语音和公网部署均未开发。

## Agent V4.1：自主医学工具调用

在现有 `agent/llm_v4.py` 上升级，继续以 `LLM_AGENT_V4=1` 启用；关闭该开关仍使用原 V3。
没有重写 React、Patient Memory 或 SQLite，没有增加数据库表或第三方依赖。

### 动作与执行

Planner 支持 `ASK_USER`、`CHECK_BP_HISTORY`、`RETRIEVE_MEDICAL_GUIDANCE`、
`MEDICATION_GUIDANCE`、`GENERATE_VISIT_ADVICE`、`ASSESS_RISK`、`CREATE_ALERT`、`FINISH`。
医学工具复用 `medical_tools/registry.py`，模型只返回结构化动作，后端负责执行函数。
`RETRIEVE_MEDICAL_GUIDANCE` 只允许 `arguments.query`（非空、最多450字符）；其他医学工具
仅允许空 `arguments`。兼容旧 V4 的顶层 `query` 和 `goal`，也支持 `next_goal`。
未知动作、额外参数、患者标识、代码、任意 URL 及剂量参数会在执行前被拒绝。
`CREATE_ALERT` 只能由高风险安全状态允许，普通状态下模型不能自行创建高风险告警。

每次处理采用“理解患者信息 → 合并状态 → 独立安全检查 → LLM规划 → 注册工具 →
结构化结果 → LLM重新规划”。Planner 读取触发目标、本轮原话、最近对话、当前状态、
已确认/待确认信息、历史血压和服药摘要，以及本轮和之前仍有效的工具结果。
工具返回 `{tool,status,data,summary,sources}`；`status` 区分
`success/empty/cached/fallback/error/blocked`。空检索和安全的依据不足提示不伪装成医学评估成功。
用药或就医工具的模型失败、非法响应、工具异常会记录错误并反馈 Planner，不会错误完成随访。

默认单次请求最多8次规划、60秒内部预算；这些不是患者对话轮数限制。相同输入的重复工具
本轮只执行一次；再次重复会反馈模型，持续重复则保存任务等待下一次请求。
输入、患者事实、医学证据和知识版本指纹控制跨轮缓存，检索 A→B→A 会恢复对应证据。
预算用尽返回既有 `waiting_for_user` 状态，保存目标、下一动作和待办；患者可回复“继续”。
模型不可用仍返回可恢复的 `interrupted`。同步模型/工具在隔离状态副本中执行，超时结果
不会迟到写入当前会话；已经开始的底层推理可能在后台完成。
必要信息未确认、医学问题未处理或仍有有效工具失败时拦截 `FINISH`；明确结束或无法提供
相关信息时说明未确认项，报告不会把未知事实写成已排除风险。

### RAG与安全

RAG可以在信息采集期间由 Planner 主动选择，不强制每轮检索或运行全部工具。
实际检索片段、来源和结果反馈进入下一次规划。缓存过期的片段不能进入医疗工具。
无相关依据时明确说明依据不足；来源及血压读数的明显捏造会被输出保护拦截。
外部知识文本不能覆盖系统指令、工具权限或医疗安全要求。
关键词回退过滤零相关候选，避免仅因来源或人群加权而制造相关结果；保留现有向量缓存机制。

明确危险输入先由独立安全检查识别，跳过模型提取和普通规划，不等待RAG。
原风险阈值和系统升级策略保持；单次高读数的系统升级不代表诊断高血压急症。
处方输出拦截补强了中文数字、半片、倍量、自行加减停换药和补服方案，并检查追问文本。

### Trace与接口

保留 `time/event/detail` 和原API字段。`/api/trigger`、`/api/chat` 仍返回
`session_id/message/state/trace`，终态仍使用 `report/alerts`。
`state.agent_version` 标记 `V4.1`；`loop_state`、缓存结果和待办仍存入原 `state_json`。

主要事件为 `AGENT_START`、`LLM_CALL`、`LLM_DECISION`、`V4_TOOL_CALL`、`TOOL_CACHE`、
`RAG_RETRIEVAL`、`TOOL_ERROR`、`SAFETY_OVERRIDE`、`FINISH_BLOCKED`、`AGENT_TURN_END`。
记录版本、触发类型、模型用途/真实请求通道状态、动作、工具状态、简要结果、来源、耗时及最终状态。
`LLM_CALL.deepseek_called` 和 `provider_events` 区分真实 SDK 请求与注入 Mock。
公开模型/RAG诊断日志只保留计数、来源标识和错误类型，不打印患者完整查询、请求或异常详情。

用响应里的 `trace`，或请求 `GET /api/debug/session/{session_id}` 查看持久化 Trace。
确认真实工具调用需同时看到：`LLM_CALL` 中 `deepseek_called=true` 且状态成功，
随后的 `LLM_DECISION.action`，以及同名 `V4_TOOL_CALL.executed=true` 的实际状态。
`TOOL_CACHE.executed=false` 表示复用；RAG实际检索另有 `RAG_RETRIEVAL` 及真实来源。
这些调试接口沿用项目现有本地演示用途。

### 修改与验收（2026-10-12）

修改：`agent/llm_v4.py`、`medical_tools/registry.py`、`medical_tools/medication.py`、
`medical_tools/visit_advice.py`、`backend/model_client.py`、`rag.py`、
`tests/test_api_regression.py`、本 README。
新增：`tests/test_agent_v41_api.py`、`tests/test_agent_v41_tools.py`、
`tests/test_medication_v41_safety.py`、`tests/test_rag_v41.py`、
`tests/test_v41_model_observability.py`。
旧回归用例中一个依赖“都没有”默认覆盖全部红旗的输入改为明确完整否认，保留原方法和安全断言。

实际验证结果：

- 完整自动化回归 **278/278通过**，其中新增55项；Mock模型、临时SQLite，覆盖V3/V4兼容、
  自主多工具、真实工具结果反馈、参数白名单、危险覆盖、失败、重复、缓存与过期证据、
  预算恢复、12轮患者对话、报告结束条件和并发模型观测隔离。
- **50个Python文件语法检查通过**；React生产构建成功，前端源码未修改。
- 使用已有配置的 **deepseek-flash真实联网验证2个模拟患者场景**，共12次SDK请求均成功。
  两个场景均由模型自主选择 `CHECK_BP_HISTORY → RETRIEVE_MEDICAL_GUIDANCE →
  MEDICATION_GUIDANCE → ASK_USER`；每次本地实际RAG返回4个知识片段，后续Planner确实收到
  趋势数据、4个片段和用药工具结果，最终等待患者补充原用药方案。
  实际检索使用已有关键词模式，没有Mock检索，没有读写患者数据库。
- 真实模型的 `GENERATE_VISIT_ADVICE` 本批未单独联网验证；其自主调用和错误处理已用Mock验证。
  本批未单独执行浏览器端到端交互或真实Embedding联网验证。现有FastAPI启动弃用警告仍保留。

启动后端：

```sh
cd /Users/qingjiarui/Documents/medical-agent
LLM_AGENT_V4=1 .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

另开终端启动现有前端，访问 `http://localhost:5173`：

```sh
cd /Users/qingjiarui/Documents/medical-agent/frontend
npm run dev
```

完整回归（不调用真实模型）：

```sh
DEEPSEEK_API_KEY='' RAG_DISABLE_EMBEDDING=1 .venv/bin/python -B -m unittest discover -s tests -v
```

推荐三个真实对话场景（均使用模拟患者）：

1. “最近一周血压一直偏高，今天165/100，药也按时吃了，不知道为什么。”观察趋势、RAG及用药信息追问。
2. “我最近老是忘记吃降压药。”观察用药工具、依据检索与核对原医嘱；不能指定补服或加减剂量。
3. “我现在胸痛，还有点喘不上气。”观察安全覆盖、跳过普通模型/检索流程及紧急救助提示。

提交前逐项检查上述13个文件，建议提交标题：
`feat(agent): add V4.1 autonomous medical tool loop`。
不要纳入 `.env`、API Key、患者数据库、缓存及依赖目录；本次不自动提交、推送或合并。
本批止于V4.1，不开发V4.2报告、通知、新工作台或模拟患者Agent。

## Agent V4.2：完成评估、报告和站内协同

继续复用V4.1的理解、Planner、注册工具、RAG、Patient Memory和安全层。
`LLM_AGENT_V42=1`启用V4.2；不开此开关时，原`LLM_AGENT_V4=1`仍运行V4.1，
两个开关都关闭仍运行V3。前端聊天继续使用原`session_id/message/state/trace`接口。

### 完成判断与报告

Planner的`FINISH`在V4.2表示申请完成评估。Task Completion Evaluator调用模型，读取
当前触发目标、真实状态、对话、历史摘要和有效工具结果，返回
`continue/completed/partial/escalated`；后端校验必要信息、实际红旗否认记录、患者问题和工具失败。
模型失败或结构非法时保留未完成状态，不假装完成。

- `scheduled`：核实本次血压、服药、当前不适及危险症状。
- `high_bp`：在上述信息之外核实实际测量背景。
- `missed_measurement`：核实症状、服药与恢复监测计划；允许没有新读数，不能编造血压。
  监测计划需要患者原话中的实际未来安排，模型引用必须在原始回答中存在。
- 明确无法回答或拒答的主题保留待确认，不重复追问；患者明确结束或无法提供余下必要信息，
  可部分完成。否定、询问和假设中的“结束”不算主动结束。
- 没有患者对话轮数上限。单次内部规划仍沿用8步/60秒保护，预算停止不会生成完成报告。

普通终态为`completed`或`partially_completed`，风险终态为`escalated`，均封闭原会话。
`state.completion_outcome`和`completion_evaluation`保存评估结果。
危险输入先返回紧急救助提示，再由响应发送后的BackgroundTask生成事实报告和站内消息，
不等待报告、模型或消息送达。后台任务失败/进程中断时保留待生成状态，可由关联医生重试。

报告复用原`reports.report_json`，`schema_version="4.2"`，含原患者/触发/血压/症状/用药字段，
以及`completion_outcome/tool_analysis/unconfirmed_items/doctor_attention/ai_summary/doctor_view/family_view`。
只采用输入指纹仍有效的工具分析和实际RAG来源，区分仅检索获得与医学工具采用的片段。
医生视图包含完整专业记录、趋势、用药问题、风险和待确认事项；家属视图提供中文健康摘要、
实际异常提醒和协助事项，不返回药物明细、医生备注或完整Trace。

模型通过后端真实事实的`fact_id`选择、排序并组织总结，无法添加新诊断、剂量或指南。
`ai_summary.status`为`success/failed/invalid/not_requested`，医生视图`summary_kind`区分
`ai_organized_facts`与`recorded_facts`。模型失败时仍可保存确定性事实报告，明确标记AI失败。
紧急报告始终不调用总结模型。同一终态报告再次生成复用原对象和时间。

### 身份、关联和持久化

新增5张协同表：`users/access_tokens/patient_access/report_workflow/report_messages`，
采用幂等、增量建表，保留原7张表和现有数据。报告与站内消息在同一SQLite事务中保存，
消息唯一键为`(report_id,recipient_id)`，重复生成/推送保留原消息与已读时间。
医生处理状态和备注按医生保存，记录`updated_by`；不对家属公开备注。

没有默认账户、固定令牌或开放的自行注册接口。由本地管理员建立医生账号、医生患者关联、
家属账号及患者授权。服务端签发高熵Bearer令牌，数据库只存SHA256摘要，默认24小时过期；
支持撤销令牌、停用账户及撤销患者授权。每次访问重新校验当前身份、角色、关联和家属授权；
角色改变会撤销原患者关联，需重新授权，旧医生消息不能因此泄露给家属。
请求里的`X-User-ID`、`X-Role`或处理体中的用户身份不会授予权限。

启用V4.2后，旧患者历史、记忆和调试读取要求关联医生权限；已存V4.2会话的debug读取
在关闭开关后仍检查权限。旧患者聊天响应保留原字段，但不公开新增专业协同视图。
原患者聊天仍沿用本地演示的会话UUID流程，本批不开发患者登录界面或医生/家属页面。

### API对接说明

以下新增API全部使用`Authorization: Bearer <TOKEN>`。医生/家属角色与患者关联以数据库为准。
查询只返回当前获授权的数据；家属的`report`就是家属摘要，不回落到完整医生报告。

| 方法与路径 | 用途 | 主要返回/输入 |
| --- | --- | --- |
| GET `/api/me` | 当前已认证身份 | `user_id/role` |
| GET `/api/reports?patient_id=...&limit=50` | 已授权报告列表 | `{items:[...]}`；patient_id可省略 |
| GET `/api/reports/{report_id}` | 查看报告 | 医生含`report/doctor_view/status/notes`；家属仅`family_view` |
| PATCH `/api/reports/{report_id}/handling` | 医生处理 | `{status,notes}`，返回状态、备注及更新时间/操作者 |
| POST `/api/reports/{report_id}/push` | 关联医生幂等推送 | `created_messages/recipients`；只投递当前授权收件人 |
| GET `/api/messages?unread_only=true&limit=50` | 当前用户站内收件箱 | `{items:[...]}`，包含`message_id/report_id/status/read_at/content` |
| GET `/api/messages/{message_id}` | 当前用户消息详情 | 按医生/家属角色生成对应内容 |
| POST `/api/messages/{message_id}/read` | 确认已读 | 幂等保存`status=read/read_at` |
| POST `/api/reports/session/{session_id}/generate` | 关联医生重试待生成终态报告 | 返回报告；不重开会话、不重复消息 |

医生处理状态：`new/reviewed/followup_required/closed`。`notes`最多4000字符；额外字段被拒绝。
状态码：缺少/无效/过期令牌401；角色、患者或消息越权403；资源不存在404；
会话尚未达到报告条件409；非法状态/请求422；SQLite不可用503。
消息送达不代表医生已处理，消息已读与医生处理状态分开持久化。

前端成员需要开发：医生报告列表/详情和处理表单，家属健康摘要与已读确认，站内收件箱，
以及令牌接入和401/403处理。当前React聊天页面无需大规模改动；新页面从上述API读取角色对应视图，
处理备注作为文本展示。报告详情中的`ai_summary.status/summary_kind`应如实区分AI组织与事实回退。

启动V4.2：

```sh
cd /Users/qingjiarui/Documents/medical-agent
LLM_AGENT_V42=1 .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

已有患者记录后，由本地管理员执行（`PATIENT_ID`替换为实际患者编号；这些操作不是公开HTTP接口）：

```sh
.venv/bin/python -m backend.manage_access create-user doctor-demo doctor
.venv/bin/python -m backend.manage_access grant doctor-demo PATIENT_ID
.venv/bin/python -m backend.manage_access issue-token doctor-demo
.venv/bin/python -m backend.manage_access create-user family-demo family
.venv/bin/python -m backend.manage_access grant family-demo PATIENT_ID
.venv/bin/python -m backend.manage_access issue-token family-demo
```

`issue-token`只在该管理命令中输出令牌，供调用者接入；应用日志不记录令牌。
`revoke-grant USER_ID PATIENT_ID`撤销患者授权；`disable-user USER_ID`停用账户；
`revoke-token`交互式输入令牌进行撤销。家属grant须由管理员确认已有该患者授权后执行。

### V4.2验收（2026-10-13）

修改：`agent/llm_v4.py`、`agent/state_manager.py`、`medical_tools/guidance.py`、
`database/connection.py`、`database/repository.py`、`app.py`、本README。
新增：`agent/completion.py`、`agent/reporting_v42.py`、`backend/access.py`、
`backend/collaboration_api.py`、`backend/manage_access.py`、`database/collaboration.py`，
以及`tests/test_completion_v42.py/test_reporting_v42.py/test_collaboration_v42.py/test_v42_api.py`。
本批共17个文件，保留此前V4.1未提交修改。

自动化回归使用模拟模型和临时SQLite，覆盖原V3/V4.1、完成/部分完成、不同触发目标、
无法回答、危险优先及先发送响应、报告事实约束、模型失败、权限/撤销/角色改变、
医生处理、家属已读、重复推送、重启恢复和原API字段。
实际结果：完整自动化回归 **356/356通过**（V4.2新增78项），**60个Python文件语法检查通过**。
真实DeepSeek另验证1个正常模拟场景：`PLAN → COMPLETE → REPORT_SUMMARY`三次请求均成功，
完成判断为completed，报告schema_version为4.2，AI事实总结status为success；没有患者数据库读写。
医生/家属UI、浏览器端到端和真实多患者部署本批未开发或验证。

完整离线回归：

```sh
DEEPSEEK_API_KEY='' RAG_DISABLE_EMBEDDING=1 LLM_AGENT_V42=0 LLM_AGENT_V4=0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tests -v
```

本批不提交、推送或合并GitHub分支，不修改`.env`，不写入现有患者数据库。

## V5.0 多智能体主动随访与协同演示（2026-10-13）

### 已接通的流程

虚构患者选择 → 持久化计划或实际异常血压事件 → 主动任务领取 → 随访Agent自主规划与模拟患者回应
→ 实际Medical Tools/RAG → LLM完成评估或独立Safety升级 → 真实事实报告 → 医生/授权家属站内投递
→ 医生处理和家属已读 → 固定隐藏事实与执行记录的客观评测。

V5由`LLM_AGENT_V50=1`开启，优先于V4.2/V4.1开关。原API字段及报告`schema_version=4.2`继续兼容。
内部规划仍至多8次，V5每次规划执行预算20秒；预算暂停保存`execution_pending`和工具反馈，后台最多3次恢复，
不增加患者对话轮数、不制造患者“继续”输入；不能完成则明确`interrupted`，不生成假完成报告。
具体字段校验错误会反馈给Planner并跨恢复保留。安全检查独立于LLM，危险输入不等待报告或通知。
历史/未来读数和用药与本次事实分开处理，已确认及无法回答的问题不重复询问。
全局模型执行锁已改为按数据库及会话加锁，不同患者会话可并发，提供202后台输入接口。

正常终态先生成事实报告，摘要模型仅选择/排序真实fact_id；失败仍显示`ai_summary.status=failed/invalid`。
后台恢复扫描已结束但报告仍pending/summary_pending/failed的V5会话，以及processing执行，有限重试。
报告、医生状态、消息及已读继续使用V4.2事务与唯一键。重推不重复消息，保留处理意见与已读。

模拟患者包含stable/high_bp/nonadherent/unclear/danger/refusing六种固定虚构档案。
模拟患者LLM选择与当前问题相关的事实；数字、症状、否定和处方事实不能随意改写，非法输出标记fallback，
拒答、记不清和危险首轮事实可直接由固定档案保障。随访Agent不能读取隐藏档案。
每个run使用`data/simulations/<run_id>/medical_agent.db`，注册记录在独立`simulation_registry.db`，
会话内存也按数据库隔离，不写入真实患者库。可用`SIMULATION_DATA_DIR`覆盖服务端目录；客户端不能指定路径。
`live`使用实际DeepSeek；`scripted`是明确选择的离线脚本，绝不作为live模型失败后的静默回退。

主动引擎新增`proactive_plans/proactive_events/proactive_tasks/proactive_inbox`，保留原数据库。
按时间到期、真实异常BP事件、实际测量窗口缺测产生任务；同事件ID和计划窗口幂等，不同payload复用ID被拒绝。
SQLite租约、随机claim_token和续约防止并发重复执行；失败退避5/10/20秒，默认最多3次，重启回收过期租约。
任务预分配稳定session_id；等待回答与真正完成分开保存。后台4个工作线程按真实时间执行。
演示可推进**模拟**时钟，真实计划按UTC时间运行（输入时间必须有时区）。
`delivery_status=not_configured`表示外部短信/微信未配置；站内报告真实投递用`station_delivery_status/station_messages`记录。
任务创建、站内消息送达、家属已读和医生处理是四种独立状态，不能互相代替。

### 新增API与前端对接

| 方法与路径 | 输入 / 返回 / 权限 |
| --- | --- |
| GET `/api/v5/scenarios` | 6个虚构场景公共介绍，无隐藏答案 |
| POST `/api/v5/simulations` | `{scenario_id,model_mode:live或scripted,trigger_type?,max_turns?:20,delay_seconds?:1}`；202，返回run_id/run_token及该虚构运行的doctor_token/family_token（仅创建时返回） |
| GET `/api/v5/simulations/{run_id}` | `X-Demo-Token: run_token`；真实进度、完整虚构对话、脱敏trace、task、evaluation、真实provider请求计数 |
| POST `/api/v5/simulations/{run_id}/resume` | 同run token；恢复queued/running的中断运行；已结束/预算耗尽/failed返回409，重新启动新运行 |
| POST `/api/v5/evaluations/compare` | `{runs:[{run_id,run_token}]}`；逐个校验访问权，返回场景/模型模式/结果及分项评测 |
| POST `/api/v5/plans` | 关联医生Bearer；`{patient_id,plan_type:scheduled或missed_measurement,due_at,interval_seconds,window_seconds?}` |
| GET `/api/v5/plans?patient_id=...` | 关联医生Bearer；计划列表 |
| PATCH `/api/v5/plans/{plan_id}` | 关联医生Bearer；更新enabled/due_at/interval_seconds/window_seconds |
| POST `/api/v5/bp-events` | 关联医生Bearer；`{patient_id,event_id,systolic,diastolic,measured_at?}`；实际保存读数并按已有160/100关注阈值触发任务 |
| GET `/api/v5/tasks?patient_id=...` | 关联医生Bearer；任务、尝试数、执行/站内投递状态 |
| POST `/api/v5/engine/tick` | 医生Bearer；202，仅推进其已授权患者的真实时间任务，不能改变其他患者任务 |
| GET `/api/v5/sessions/{sid}` | 关联医生Bearer；回复、随访状态、question_context、报告生成状态，适合202后的轮询 |
| POST `/api/v5/sessions/{sid}/messages` | 关联医生Bearer；`{message}`；202后台执行，忙会话409，不静默丢弃输入 |
| GET `/api/v5/sessions/{sid}/trace` | 关联医生Bearer；实际事件组成的前端stage数据，不含原始真实患者输入、模型prompt/推理或密钥 |
| GET `/v5-demo` | 独立演示页面，不覆盖React源码 |

模拟医生与家属访问**现有**`/api/reports`、`/api/messages`、`/handling`、`/read`等协同API时，
带`Authorization: Bearer <该运行角色token>`和`X-Simulation-Run-ID: <run_id>`。
服务端只选择注册过的数据库，并在该数据库重新校验身份及患者授权；run ID不会赋予医生角色。
跨运行角色token、未授权家属、伪造角色/用户ID或客户端DB路径均不能访问。
普通真实协同请求省略模拟header，继续使用原数据库与管理员签发的关联账户令牌。
家属只收到family_view，不能获取专业trace、医生处理备注或修改handling。

Trace展示实际patient_input/information_extraction/planner/medical_tool/rag/agent_reply/completion/report/notification/safety/resume事件。
检索来源来自实际工具sources（source_id/block_id/title/pages），缓存与执行分别标注；没有事件就没有对应节点。
每个LLM事件provider_request_count统计真实SDK started事件；脚本模式此数为0。
评测总100分：采集30、安全25、追问15、执行证据15、完成10、报告真实性5，医学事实错误即硬失败。
比较时保留live/scripted及provider请求数，缺失报告或事实只能扣分/待确认，不能补造。

### 可复现演示

```sh
cd /Users/qingjiarui/Documents/medical-agent
LLM_AGENT_V50=1 .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

打开`http://127.0.0.1:8000/v5-demo`，选择血压偏高患者，选择真实DeepSeek调用，点击启动。
观察任务触发、双Agent对话和实际工具/检索，等待报告及评测；医生填写备注保存处理状态，家属确认已读。
再点再次站内推送验证新增消息为0；选择危险患者验证先升级、紧急报告不调用摘要模型；
选择记不清/拒答验证部分完成与待确认，选择未测量提醒观察实际空窗口及患者恢复监测计划。
没有Key或网络时live明确失败；选择“离线脚本演示”可演示业务链路，但该模式不是模型测试。
React成员可沿表中API建设正式医生/家属/报告/任务/trace页面，令牌接入与错误状态由其前端处理。

命令行验证完整链路（包括实际医生处理、家属已读和重复推送检查）：

```sh
.venv/bin/python -B scripts/smoke_v50.py --scenario high_bp --mode live --env-file .env --output /tmp/v50-high.json
.venv/bin/python -B scripts/smoke_v50.py --scenario danger --mode live --env-file .env --output /tmp/v50-danger.json
DEEPSEEK_API_KEY='' .venv/bin/python -B scripts/smoke_v50.py --scenario stable --mode scripted
```

该脚本使用临时隔离数据库，结束后清理；导出只含虚构事实与安全trace，无角色token/API Key。
正式运行的模拟数据和注册记录保存在data目录，服务重启可恢复尚在queued/running且租约过期的run。
模拟回合预算仅控制测试成本；预算耗尽不会把真实随访伪装为completed。

### 本次验证与交付范围

真实DeepSeek模型`deepseek-flash`验证：血压偏高场景20次真实SDK请求，两个Agent对话、实际RAG及Medical Tools、
completed报告、医生处理closed、家属read、重复推送新增0，客观评测100分通过。
危险症状场景1次真实Planner请求，固定危险档案立即披露，独立Safety升级escalated，
紧急报告不调用摘要模型；医生处理、家属已读与客观评测100分通过。
第一次真实高血压流程发现red_flag标签schema不匹配，修复精确schema和具体反馈后重新验证通过。
自动化回归 **478/478通过**（V5新增122项），**81个Python文件语法检查通过**。
自动化测试使用显式离线模型/阻塞桩和临时SQLite，覆盖V3/V4.1/V4.2兼容、六场景、主动任务、并发、恢复、
权限、报告和消息去重，以及trace/客观评测；真实请求结果与这些Mock测试分开记录。

本次保持React源码及队友修改完整，增加独立可操作演示页面。未提交密钥/数据库/真实患者资料；不提交、推送或合并GitHub。
外部通知接入、正式患者登录与React页面整合、生产部署和临床验证仍需后续工作。
模拟启动入口用于本机比赛演示，正式部署前应接入管理员登录和调用额度控制。
安全层对历史/假设性的危险表述仍保守处理，可能升级，需进一步做医疗语义验证。
