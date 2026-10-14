# V5 模拟回退与验证恢复修复

本批代码和测试直接修改在现有 medical-agent 内。完整对话见 [conversation.md](conversation.md)，原始数据见 [stable-mock-faults.json](stable-mock-faults.json)。

## 修改与原因

- `backend/simulation_v5.py`、`simulation/patient.py`：患者接收实际问题和 question_context；完整回复保留在对话中。回退按当前主题选隐藏档案事实，未知问题安全澄清，不任取第一个事实。模拟恢复先验证保存候选。
- `backend/config.py`、`backend/model_client.py`、`agent/llm_v4.py`：HTTP 默认每次 50 秒，V5 工具和规划窗口默认 120 秒，均可配置。临时错误最多重试一次；关闭 SDK 自动重试；普通模式超时后在原模式重试。
- `agent/llm_v4.py`、`backend/runtime_v5.py`、`backend/v5_api.py`：提取与验证有独立时限。未验证候选保留于 extraction_pending；SQLite、后台和模拟恢复先重做验证，不重新生成。日期或问题语境已变化时暂停等待新输入，避免将旧回答写成今日事实。
- `agent/trace_view.py`、`scripts/smoke_v50.py`：区分生成失败、验证超时、验证拒绝、重试、重试耗尽、安全暂停和恢复。公开 Trace 不暴露候选原文或模型提示词。
- `agent/llm_v4.py`：reply 简短确认，question 实际提问；相同句子及双问候给予模型明确反馈重新生成，语义重复继续由 VERIFY_RESPONSE 审核。

## 测试结果

- 基线 564 项通过；修复后 **603 项全部通过**，新增 39 项。完整输出见 [regression.log](regression.log)。
- 覆盖症状和血压问题下的回退、服药总结污染当前问题、验证首次超时后成功、持续失败不落库、SQLite/后台/模拟恢复、日期和问题过期、双问候与重复确认，以及原医疗安全与 API 回归。
- 故障注入完整双 Agent 模拟中，第 3 轮患者生成失败、第 4 轮患者验证失败，第一次 VERIFY_EXTRACT 超时；随访仍正常 completed，评估通过并生成报告。
- 上述完整对话使用 Mock/脚本模型，真实请求为 0；患者条目 mode=llm 只表示模型适配器路径。
- 真实 DeepSeek：项目现有配置已发起请求，本环境返回 APIConnectionError，没有取得有效模型回答，也没有真实对话。失败证据见 [stable-live.json](stable-live.json) 和 [live-connectivity.json](live-connectivity.json)。本轮真实稳定患者完整随访尚未验证通过。
- 原患者数据库、.env、患者档案、危险规则、RAG 和静态文件共 34 个受保护文件哈希未变。原 API 路径兼容。

## 启动和复现

从 medical-agent 根目录运行：

```sh
DEEPSEEK_API_KEY='' .venv/bin/python -B -m unittest discover -s tests -v
LLM_AGENT_V50=1 DEEPSEEK_TIMEOUT_SECONDS=50 V5_REQUEST_TIMEOUT_SECONDS=120 .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
.venv/bin/python -B scripts/smoke_v50.py --scenario stable --mode live --env-file .env --output artifacts/v5-reliability-2026-10-14/stable-live.json
```

真实模拟需要执行环境能够连接已配置的 DeepSeek 地址。模拟使用临时 SQLite 隔离数据库。
