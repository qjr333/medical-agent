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
