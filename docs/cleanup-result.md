# V5 结构精简交付记录

基准提交：`05c4ce1`。本次先生成逐文件审查和计划，再执行证据充分的 A/B 修改；没有提交、推送或合并。

- [146 文件审查表](structure-audit.md)
- [完整依赖及清理前后文件清单](module-dependencies.json)
- [执行计划与纠正记录](structure-plan.md)
- [当前 V5 使用说明](../README.md)

## 修改前后目录对比

| 修改前 | 修改后 | 原因 |
|---|---|---|
| `agent/llm_v4.py` | `agent/followup_agent.py` | 实际承担 V5 主流程，去掉误导性的模块版本名 |
| `agent/reporting_v42.py` | `agent/clinical_report.py` | 当前报告模块；历史 schema 仍为 `"4.2"` |
| `frontend/src/assets/` 的 3 个模板素材 | 不再存在这些文件 | 全仓零引用，删除前后构建 hash 相同 |
| `frontend/public/icons.svg` | 删除；保留 `favicon.svg` | icons 未被使用，favicon 有实际引用 |
| `frontend/README.md` | 内容由主 README 的 React 使用说明覆盖 | 删除通用 Vite 模板说明 |
| `.DS_Store`（根、data、knowledge_meta） | 受控清单不再保留这些系统文件 | 保留用户已有根目录删除；本批删除另外两处 |
| README 544 行阶段记录 | README 194 行当前项目说明 | 覆盖架构、环境、V5、模拟、测试及限制 |
| 无结构审查文档 | `docs/` 下 4 份审查/交付资料 | 不增加业务模块或依赖 |

```text
修改前                            修改后
agent/                            agent/
  llm_v4.py                         followup_agent.py
  reporting_v42.py                  clinical_report.py
  planner.py / runner.py             planner.py / runner.py（保留兼容）
backend/                          backend/
  extraction.py                     extraction.py（仍被测试使用）
  patient_state.py                  patient_state.py（保留兼容）
frontend/                         frontend/
  README.md                         删除模板说明
  public/icons.svg                  删除未用图标
  src/assets/{hero,react,vite}       删除模板素材
README.md（阶段记录）             README.md（当前 V5）
                                  docs/{structure-audit,structure-plan,
                                        module-dependencies,cleanup-result}
```

清理前 HEAD 清单 146 个文件；清理后的预期受控清单 142 个（移除 8 个，其中根 `.DS_Store` 是用户已有删除，新增 4 份文档，改名不改变文件数）。87 个 Python 文件和 39 份测试文件均保留。

## 实际删除和改名

本批新增删除 **7 个文件**：

- `data/.DS_Store`
- `knowledge_meta/.DS_Store`
- `frontend/README.md`
- `frontend/public/icons.svg`
- `frontend/src/assets/hero.png`
- `frontend/src/assets/react.svg`
- `frontend/src/assets/vite.svg`

根 `.DS_Store` 在开始工作前已经暂存删除，本批没有覆盖用户的暂存状态。Finder 后续可重建本地同名文件，该文件由 `.gitignore` 忽略，不影响已暂存的删除。

两个模块改名见上表；三个公共符号同步改名：

- `run_agent_v4` → `run_followup_agent`
- `extract_user_patch_v4` → `extract_patient_information`
- `build_report_v42` → `build_clinical_report`

20 个 Python 文件统一更新普通/局部导入、动态导入、函数调用和 Mock 引用，另重写主 README。
AST 名称归一化核对确认代码变更仅为改名和模块说明，无额外业务逻辑变化。
不新增旧名称转发文件；测试中保留的 V3/V4 命名描述的是仍需验证的兼容场景。

## 保留旧代码的原因

| 保留项 | 实际用途 / 不执行 C、E 清理的原因 |
|---|---|
| `agent/planner.py`、`agent/runner.py` | 旧 API 在全部 LLM 开关关闭时仍执行 V3；旧无版本会话和安全回归依赖 |
| `agent/extractor.py`、`question_planner.py` | V5 的提取入口与提问上下文仍引用它们，规则路径也有兼容用途 |
| `agent/reporting.py` | V5 急诊升级提示及临床报告基础字段仍复用 |
| `backend/extraction.py`、`patient_state.py` | 兼容测试实际消费原导入路径，不能作为无用文件删除 |
| `db.py` | CLI 初始化及可覆盖 `DB_PATH` 的历史调用契约 |
| `app.py` 的适配导出 | 原 API 和测试存在直接消费者；本批只更新新名称引用 |
| V3/V4.1/V4.2 开关 | 还影响旧会话、worker 启动和调试权限；默认值切换属于待确认行为变更 |
| 两个 HTML 页面 | 均有现行路由/测试，React 尚未覆盖双 Agent 和协同功能 |
| 两份 CSS | 都实际导入；重叠样式合并需单独确认视觉结果 |
| 15 个历史工件 | 虚构场景的真实运行证据；保留本地记录，不按无 import 删除 |
| 知识 Markdown、注册表、来源 PDF | 8 份知识及 10 份来源全部有效关联，必须保留追溯 |

执行中发现初始 A 分类错误：`backend/extraction.py` 的依赖图已有测试引用，但分类没有落实该证据。
完整回归因此出现 1 个导入错误；已恢复原文件、改列 C，并重新运行全部 603 项通过。没有删除测试或改变断言来绕过失败。

## 验证结果

| 检查 | 实际结果 |
|---|---|
| 修改前完整基线 | 603 项通过；React 构建通过 |
| 改名专项 | 提取 7 项、报告 17 项通过 |
| 最终完整自动化测试 | **603 项全部通过**（8.320 秒；恢复工作后再次复核）；保留全部原测试 |
| Python 语法 | 87/87 通过 |
| 实际 Python 导入 | 47 个生产模块及 smoke 脚本均通过 |
| FastAPI 生命周期 | TestClient 启动/关闭正常，health/demo/首页均 200 |
| 实际 V5 启动链 | 4 个原 worker 启动、执行临时库 tick 并关闭，模型调用 0 次 |
| React | build 与 lint 均通过；构建资源 hash 与基线相同 |
| 主动触发 | 定时、异常血压、缺失测量任务及授权 API 回归通过 |
| 安全升级 | 危险症状、血压校验、处方拦截和安全暂停回归通过 |
| 随访及报告 | 完成、部分完成、升级、幂等报告和重复推送回归通过 |
| 医生/家属 | 授权、撤权、报告视图、处理状态和站内消息回归通过 |
| 旧数据兼容 | V3 新进程 SQLite 恢复、历史报告/schema 4.2、旧家属视图保护和 V5 待验候选恢复通过 |
| RAG / Memory / 医疗工具 | 缓存首次构建/读取/失效、记忆、血压趋势、就医建议与 Trace 回归通过 |
| 数据保护 | `.env`、真实 SQLite、缓存、患者 JSON、知识、HTML 和历史工件等 50 项哈希均未变化 |
| Git | diff 空白检查通过；未新增跟踪密钥或数据库，未执行 add/commit/push/merge |

Python 和 API 启动检查使用临时 SQLite，额外拦截数据库连接以禁止写入真实数据目录。
完整回归使用 Mock/脚本模型；本批没有向真实 DeepSeek 发出请求，也没有运行浏览器手工端到端操作。
RAG 业务链烟测使用关键词检索，向量缓存行为由自动化模型替身测试验证；本次没有重新下载或加载真实 Embedding 模型。

六个独立临时数据库的完整双 Agent 业务链：

| 场景 | 最终随访状态 | 评估 | 医生处理 / 家属已读 | 重复推送新消息 | 真实模型请求 |
|---|---|---|---|---|---|
| stable | completed | 100 / 通过 | 均通过 | 0 | 0 |
| high_bp | completed | 100 / 通过 | 均通过 | 0 | 0 |
| nonadherent | completed | 100 / 通过 | 均通过 | 0 | 0 |
| unclear | partially_completed | 89 / 通过 | 均通过 | 0 | 0 |
| danger | escalated | 100 / 通过 | 均通过 | 0 | 0 |
| refusing | partially_completed | 73 / 通过 | 均通过 | 0 | 0 |

上述“通过”包含完整报告和消息链，危险场景按原安全规则升级，拒答/不明确场景保持部分完成。
这些是 scripted 模式的结果，不能代表真实模型效果。临时对话 JSON 存在 `/tmp/medical-structure-<scenario>.json`，没有新增患者运行数据到仓库。

## 运行和复核

从项目根目录执行：

```sh
LLM_AGENT_V50=1 .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
DEEPSEEK_API_KEY='' .venv/bin/python -B -m unittest discover -s tests -v
DEEPSEEK_API_KEY='' .venv/bin/python -B scripts/smoke_v50.py --scenario stable --mode scripted --output /tmp/medical-stable.json
```

在 `frontend/` 执行 `npm run build`、`npm run lint`；开发使用 `npm run dev`。
完整安装、模拟及账号配置见主 README。

## 尚未解决的结构问题

C、E 项仍保留：旧规则链及版本选择、API 兼容导出、两个提取/状态兼容入口、重叠 CSS、历史工件是否归档。
这些需要确认兼容保留策略后单独处理，本批没有擅自删除或合并。
健康接口的原 `version="1.6.1-demo-json"` 显示值仍保留，当前 FastAPI 应用版本为 5.0；没有为了结构改名改变 API 字段值。
本批到此结束，没有自动进入新功能开发。
