# V5 自主随访重构交付记录

日期：2026-10-14

## 已修改

- V5 患者信息提取改为 DeepSeek 结构化 claims 加独立 VERIFY_EXTRACT；Python 检查原话位置、字段类型、当前/历史分离、数字及危险症状边界。缺失/无效语义结构会暂停，不使用旧正则回填。
- Agent 可自主选择追问、补问已完成主题、调用工具及提出结束。Completion 保留最低业务目标和当前证据检查，不要求填完固定六项危险症状。
- 对 Agent 回复和医学工具文字做独立 VERIFY_RESPONSE。处方越界、未证血压、虚构处方和错误症状结论会带具体反馈重新生成；验证失败的候选不进入回复或报告。
- 模拟患者可从完整虚构事实档案中选择披露并组织自然语言；两层验证限制原话证据、事实子集及血压/剂量数值。危险档案首轮继续立即触发。
- 修改同时将遗漏的问题范围、非空隐藏症状伪报为空列表等边界纳入测试。原医疗风险阈值、`.env`、既有数据库、患者档案、知识库和API路径未改。

## 验证

- 全部回归：564 项通过（`DEEPSEEK_API_KEY='' .venv/bin/python -B -m unittest discover -s tests -v`，Mock/脚本测试）。
- 模拟患者与V5对话评估：47 项通过；V5 API另有16项测试通过，Agent V5集成12项通过。
- V5语义提取31项、回复/状态安全边界19项通过；已有V5流程28项通过。
- 48个受保护文件哈希未变，包括 `.env`、`demo_patient.json`、RAG索引源码、风险规则、SQLite源码、静态前端及知识文件。`git diff --check` 通过。
- 真实 DeepSeek：危险场景完成安全升级与报告；稳定、高血压场景因当前 APIConnectionError/部分语义验真超时中断，系统没有将其伪报为完成。逐轮对话和诊断详见 [conversation-log.md](conversation-log.md)。

## 复现

从项目根目录执行：

```sh
DEEPSEEK_API_KEY='' .venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B scripts/smoke_v50.py --scenario stable --mode live --env-file .env --output artifacts/v5-autonomy-2026-10-14/stable.json
.venv/bin/python -B scripts/smoke_v50.py --scenario high_bp --mode live --env-file .env --output artifacts/v5-autonomy-2026-10-14/high_bp.json
.venv/bin/python -B scripts/smoke_v50.py --scenario danger --mode live --env-file .env --output artifacts/v5-autonomy-2026-10-14/danger.json
```

模拟使用各自临时SQLite数据库，不操作项目患者库。
