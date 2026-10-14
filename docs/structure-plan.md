# V5 结构精简执行计划

本计划在源码修改前生成。已逐一读取 HEAD `05c4ce1` 的全部 146 个受控文件，并结合当前引用、动态导入、测试、前端资源、知识来源注册和 PDF 元数据审查。

- [逐文件 A–E 审查表](structure-audit.md)
- [完整依赖和文件元数据](module-dependencies.json)

> 执行校正：初始 A 清单包含 `backend/extraction.py`，但兼容测试仍消费该入口。完整回归暴露后原样恢复，最终列 C；本表反映最终执行范围。

## A：执行有充分证据的清理

| 文件 | 证据 | 处理 |
|---|---|---|
| `.DS_Store` | 系统元数据，用户已暂存删除 | 保留用户原删除状态 |
| `data/.DS_Store`、`knowledge_meta/.DS_Store` | 已跟踪的系统元数据，无业务依赖 | 删除文件；不操作其他数据 |
| `frontend/src/assets/hero.png`、`react.svg`、`vite.svg` | 全仓受控文本中的路径/文件名引用为零，当前组件无图片使用 | 删除 |
| `frontend/public/icons.svg` | 无 HTML、React、CSS 或脚本引用 | 删除；保留实际使用的 favicon |
| `frontend/README.md` | 未使用的 Vite 模板说明 | 主 README 完整覆盖 React 命令后删除 |

## B：执行安全改名

| 原路径/符号 | 新路径/符号 |
|---|---|
| `agent/llm_v4.py` | `agent/followup_agent.py` |
| `agent/reporting_v42.py` | `agent/clinical_report.py` |
| `run_agent_v4` | `run_followup_agent` |
| `extract_user_patch_v4` | `extract_patient_information` |
| `build_report_v42` | `build_clinical_report` |

统一更新全部生产代码、局部/动态导入、Mock 路径、测试引用和当前文档。不新增旧文件名的转发模块。历史工件是当时的记录，原样保留。

SQLite 使用 JSON，没有保存 Python 模块名或函数名。此次改名不改既有 `agent_version`、报告 `schema_version=4.2`、`llm_v4_*` 状态键、历史 Trace 事件及医疗风险规则。

## C、E：保留待确认，不删除或合并

- `agent/planner.py`、`agent/runner.py`：旧 API 的规则链仍可实际执行；旧状态恢复和安全回归依赖它们。
- `app.py`：仍承担 API 和测试兼容导出。减少适配层是后续行为变更，暂不改路由或其语义。
- `backend/patient_state.py`、`backend/extraction.py`：导入兼容入口仍被测试使用；保留。
- `App.css`、`index.css`：全局样式有重叠，但都被导入；不擅自合并样式。
- 15 个历史模拟/诊断工件：属于运行证据；保留，不按“未 import”判为无用。
- V3/V4.1/V4.2 开关：不能仅改环境变量默认值。旧无版本会话、worker 启动和调试权限均受影响；此批保留，当前 README 明确 V5 标准启动命令。
- 两个 HTML 页面：均有真实路由和测试消费，且 React 未覆盖 V5 协同能力；保留。

## 文档和验证

按用户明确要求将主 README 重写为当前 V5 说明；历史开发记录留在 Git 历史中，不再复制一份累计日志。

依次检查 Python 导入、FastAPI 生命周期/health/demo、全部自动化测试、React 构建、V5 稳定/异常/危险/漏服双 Agent、主动任务、报告、权限与历史会话兼容。使用临时隔离数据库；不删除失败测试、不修改 `.env`、数据库或知识库，不提交、推送或合并。
