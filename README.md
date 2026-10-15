# 银龄守护——老年高血压主动随访 Agent（V5.0）

基于 FastAPI、DeepSeek、RAG、SQLite 和 React 的主动随访演示项目。
随访 Agent 理解患者回答、自主选择工具和追问；Python 校验患者事实、医学证据、处方安全及最低结束条件。
模拟患者 Agent 使用独立的虚构隐藏档案，支持完整双 Agent 对话、Trace 和场景评估。

## 核心功能

- 定时随访、异常血压、缺失测量三类主动任务，支持任务去重、执行恢复与会话持久化。
- 上下文信息提取、动态追问、语义验证失败后的安全暂停及恢复。
- 血压趋势、用药辅助评估、就医建议、危险症状检查及风险升级。
- Patient Memory V2：历史血压、服药、上次随访和关注点摘要。
- 医学知识检索及来源追溯，磁盘向量缓存和 Embedding 懒加载。
- 医生/家属分视图报告、站内消息、已读状态、处理状态及患者授权控制。
- 六类虚构患者的双 Agent 模拟，区分真实模型与离线脚本模式。

## 当前架构

```text
前端 / API → V5 Runtime → Followup Agent → Tool Registry → 医疗工具 / RAG
                              ↓                         ↓
                      Patient State / Memory       Safety Layer
                              ↓
                       SQLite Repository
                              ↓
                      Clinical Report / 站内消息
```

`agent/followup_agent.py` 负责模型提取、规划和执行流程；`agent/clinical_report.py` 组织有事实依据的医生及家属报告。
V5 模拟和主动任务通过 `backend/runtime_v5.py` 执行，明确启用 V5。
`/api/trigger`、`/api/chat` 保留旧路径及响应字段，使用下方 V5 启动配置运行。
医学风险判断、数值有效性、工具参数、处方限制和风险升级由后端独立执行。

V5 在返回输入已接收前保存患者原话，提取成功后先保存待验证候选，独立语义验证通过后再原子写入事实与记录。
中断恢复复用原始输入或候选，不重复应用已确认事实。对话与工具结果在规划上下文中去重，审计 Trace 保留。
相同事实下已通过验证的“继续随访”评估可以复用；新回答、风险、工具结果或结束意图变化即失效。
追问、工具选择及结束申请仍由 LLM 决定。单次规划与内部恢复共享时间窗口；同一次患者输入最多累计24次内部 PLAN，
耗尽后安全暂停。这是执行成本限制，不是患者问答轮数或固定问卷。

## 目录结构

```text
app.py                           FastAPI 应用、路由和兼容适配
agent/
  followup_agent.py              V5 模型随访；保留旧会话模式
  clinical_report.py             医生/家属报告；兼容 schema 4.2
  completion.py                 最低结束条件和事实校验
  extractor.py                  提取入口、上下文辅助及旧模式提取
  state_manager.py              患者状态、合并和 Trace
  question_planner.py            提问上下文及旧模式追问
  trace_view.py                  对外 Trace 展示和敏感信息裁剪
  planner.py / runner.py         仍在使用的 V3 兼容链
  reporting.py                  基础报告及安全升级提示
backend/
  config.py / model_client.py    环境配置和 DeepSeek 调用
  runtime_v5.py                  V5 执行、恢复和报告流程
  simulation_v5.py / v5_api.py   双 Agent 模拟及主动任务 API
  proactive.py                  主动任务调度
  access.py / collaboration_api.py  权限、报告及消息 API
  manage_access.py               本地账号、授权和令牌管理
  patient_state.py / extraction.py / utils.py  兼容入口和辅助
medical_tools/                   安全、血压趋势、用药、就医建议、RAG 和注册表
memory/patient_memory.py         Patient Memory V2
simulation/                     虚构患者、脚本模型和场景评估
database/                       SQLite 连接、业务存储、权限、任务和模拟存储
db.py                           旧数据库入口和可覆盖路径兼容
rag.py                          RAG 检索、磁盘缓存和懒加载
frontend/                       React 随访页面
static/index.html               原随访页面（/）
static/v5-demo.html              双 Agent 与医生/家属演示（/v5-demo）
knowledge/                      8 份医学知识 Markdown
knowledge_meta/source_registry.json  知识来源注册表
knowledge_source/               10 份来源 PDF
demo_patient.json               虚构演示患者及触发配置
data/                           本地数据库、缓存和运行记录；Git 忽略
scripts/smoke_v50.py             隔离数据库的业务链验证
tests/                          自动化回归
artifacts/                      历史虚构场景的运行证据
docs/                           本次结构审查、依赖表、计划和验证结果
requirements.txt                Python 依赖
```

React 当前提供随访交互；两个 HTML 页面仍承载实际功能，尤其是双 Agent 和协同演示，因此保留。
完整的文件分类、引用证据和待确认项见 [结构审查](docs/structure-audit.md)、[依赖关系](docs/module-dependencies.json) 和 [执行计划](docs/structure-plan.md)。

## 环境配置

在项目根目录使用现有 `.venv`。React 当前依赖的 Vite 要求 Node.js `20.19+` 或 `22.12+`（本机验证使用 Node 24）。首次安装时执行：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cd frontend
npm ci
```

在项目根目录的 `.env` 中配置 DeepSeek，已有文件应保留。以下为配置示例：

```dotenv
DEEPSEEK_API_KEY=填写自己的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
LLM_AGENT_V50=1
DEEPSEEK_TIMEOUT_SECONDS=50
V5_REQUEST_TIMEOUT_SECONDS=120
```

`DEEPSEEK_MODEL` 应选择账号实际可用的模型。模型调用超时可配置，临时错误最多有限重试一次；
语义验证仍失败时保留上下文并暂停，不将未验证医学事实写入已确认状态。
`.env`、SQLite、运行数据、向量缓存、`node_modules` 和构建产物已由 Git 忽略。

RAG 默认使用 `BAAI/bge-small-zh-v1.5`，可通过 `RAG_EMBEDDING_MODEL` 修改，
缓存目录可通过 `RAG_CACHE_DIR` 修改（默认 `data/rag_cache/`）。
启动读取知识块和有效缓存；首次检索才加载 Embedding 模型。
知识文件、来源注册、模型配置或索引格式变化会使缓存失效。第二次启动复用知识块向量，查询向量仍需计算。
模型不可用时回退关键词检索；`RAG_DISABLE_EMBEDDING=1` 可明确选择此模式。
知识更新后可重启应用或调用 `POST /api/rebuild-knowledge`；`/api/knowledge-status` 返回缓存和模型状态。

## V5 启动

在项目根目录执行：

```sh
LLM_AGENT_V50=1 .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

- 原随访页面：http://127.0.0.1:8000/
- 双 Agent/协同页面：http://127.0.0.1:8000/v5-demo
- API 文档：http://127.0.0.1:8000/docs
- 健康检查：`GET /api/health`

React 页面在另一个终端启动：

```sh
cd frontend
npm run dev
```

打开 http://localhost:5173/；开发代理将 `/api` 请求转到 `127.0.0.1:8000`。
生产构建使用 `npm run build`，产物在 `frontend/dist/`；当前 FastAPI 不自动托管该构建产物。

为兼容历史接口，`LLM_AGENT_V50`、`LLM_AGENT_V42`、`LLM_AGENT_V4` 的选择顺序仍为 V5 → V4.2 → V4.1 → V3。
全部关闭时原聊天入口使用 V3；V5 Runtime 始终使用 V5。启用 V5 同时开启主动任务 worker 和恢复流程。
本次未改变旧会话的版本标识、无版本会话处理、历史 Trace 字段或报告 `schema_version=4.2`。

## 双 Agent 模拟

在 `/v5-demo` 选择稳定、血压异常、漏服/不规律、回答不明确、危险症状或拒答场景。
真实模式需要有效 DeepSeek 配置；离线脚本模式可重复运行，不发出模型请求。
页面提供完整对话、Trace、评估，以及独立演示医生/家属的报告和消息入口。

每个运行使用 `data/simulations/<run_id>/` 下独立存储，不写入真实患者数据库。
运行凭证只用于对应模拟；隐藏患者档案不会作为随访 Agent 的输入。
API 创建入口为 `POST /api/v5/simulations`，主要字段为 `scenario_id`、`model_mode`、`max_turns` 和 `delay_seconds`。
读取结果使用 `GET /api/v5/simulations/{run_id}` 和创建时返回的 `X-Demo-Token`。

可恢复的模型中断会显示“已暂停”和恢复按钮；点击后从已保存阶段继续，不伪造患者输入。
暂停时停止轮询，已记录为可恢复暂停的模拟不会在程序重启后被自动重试。不可恢复、已结束及预算耗尽的运行不能强行恢复。
真实模式必须验证自然回答及其事实证据；只有明确的离线脚本模式可以使用预设事实渲染。

命令行验证完整业务链，使用临时数据库并输出虚构对话：

```sh
DEEPSEEK_API_KEY='' .venv/bin/python -B scripts/smoke_v50.py --scenario stable --mode scripted --output /tmp/medical-stable.json
.venv/bin/python -B scripts/smoke_v50.py --scenario high_bp --mode live --output /tmp/medical-high-bp-live.json
```

`--mode live` 会调用真实 DeepSeek；`scripted` 属于离线验证。脚本默认使用关键词 RAG，
输出包含对话、报告、任务状态及医生处理/家属已读结果；不会导出运行凭证。

医生/家属真实账号通过本地管理入口维护，无开放注册 API：

```sh
.venv/bin/python -m backend.manage_access create-user doctor_demo doctor
.venv/bin/python -m backend.manage_access grant doctor_demo P001
.venv/bin/python -m backend.manage_access issue-token doctor_demo --hours 24
```

`P001` 是当前虚构演示患者编号，使用真实数据时应改为实际编号，家属账号使用角色 `family`。令牌用于 `Authorization: Bearer ...`，应妥善保管。
医生可处理报告，家属只读取授权范围内的家属视图；撤权后历史报告和消息也不可继续访问。
消息是项目内站内消息，目前不发送短信、邮件或外部通知。

## 测试方法

完整回归使用 Mock/脚本模型、临时 SQLite 和临时缓存，不需要真实模型凭证：

```sh
DEEPSEEK_API_KEY='' .venv/bin/python -B -m unittest discover -s tests -v
node --test tests/frontend_regression.test.mjs
cd frontend
npm run build
npm run lint
```

测试保留旧版本兼容用例，覆盖 API、状态更新、多轮交互、安全升级、报告幂等、权限、任务恢复、
旧会话在新进程恢复、历史报告、Patient Memory、医疗工具、RAG 缓存建立/复用/失效和模型超时恢复。
真实模型验证应另行执行 `--mode live`，不能用 Mock 结果代替真实模型通过结论。
最新真实调试：稳定患者正常完成、危险症状正确升级；异常血压场景仍在工具与回复验证中达到时间上限并暂停，尚未全部通过。
真实短答、错误Trace、耗时和剩余问题见 [本次调试结果](docs/v5-debug-2026-10-15.md)。
此前离线优化的测试结果保留在 [优化交付记录](docs/v5-optimization-result.md)、[完整离线对话](artifacts/v5-optimization-2026-10-15/conversations.md)。
前一批结构精简记录保留在 [历史交付记录](docs/cleanup-result.md)。

## 已知限制

- 项目用于虚构患者演示，尚未完成临床验证；医学知识版本和来源需要持续维护。
- LLM 可能超时或返回无法验证的内容，此时随访安全暂停，需要恢复或人工处理。
- SQLite 和进程内后台线程适合当前本地演示；多进程服务共享模型、任务调度和生产部署未在本次范围内。
- React 尚未完整覆盖原 HTML 的双 Agent 和医生/家属页面。
- V3 兼容链、部分 API 导出与 CSS 重叠仍有实际依赖，详见审查表；本次未擅自删除或合并。
