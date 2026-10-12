# 04 用药安全（Medication Safety）

> 用途：老年高血压主动随访 Agent 的 RAG 知识文件（V1）。
> 原则：以下内容为对上传指南/标准的结构化摘要，不替代临床诊疗；涉及紧急转诊、处方调整等高风险动作，应由 Safety Layer / 医疗专业人员约束，而不是只依赖 RAG。

本文件用于患者提出“能不能加药/减药/停药”“吃药后不舒服”等问题时，为 Agent 提供安全边界与转诊依据。

---
block_id: MEDSAFE_001
source_id: CN_ELDERLY_HTN_2023
source_title: 中国老年高血压管理指南2023
year: 2023
source_pages: PDF p.2, p.5及治疗章节
population: ["age>=65"]
topic: medication_safety
subtopic: polypharmacy
priority: 105
safety_layer_candidate: true
---

### 知识要点
- 老年高血压患者常伴多种慢性疾病，多重用药较常见。
- 多重用药增加药物相互作用和不良反应管理的复杂性。
- 合并衰弱、高龄或认知问题时，治疗更强调安全性、耐受性和个体化。

### Safety Layer 候选
当患者报告多种处方药、明显不良反应、晕厥/跌倒或认知问题时，Agent 不应自行做处方增减，应进入 `MEDICATION_GUIDANCE + PROFESSIONAL_REVIEW` 路径。

---
block_id: MEDSAFE_002
source_id: CN_NHC_WS_T_872_2025
source_title: 基层医疗卫生机构高血压防治管理标准 WS/T 872-2025
year: 2025
source_pages: PDF p.5
population: ["adult", "hypertension"]
topic: medication_safety
subtopic: adverse_effect_referral
priority: 100
safety_layer_candidate: true
---

### 知识要点
在随访过程中，如果怀疑与降压药有关且不良反应难以处理，属于宜转诊至上级医疗卫生机构的情况之一。

### Safety Layer 候选
`medication_adverse_effect = severe_or_unmanageable` -> 禁止继续由聊天模型给出药物调整建议 -> 进入线下专业评估/转诊建议。

---
block_id: MEDSAFE_003
source_id: CN_PRIMARY_HTN_GUIDE_2025
source_title: 国家基层高血压防治管理指南2025版
year: 2025
source_pages: PDF p.9-10（5.4.4/5.4.5）及 p.13
population: ["adult", "hypertension"]
topic: medication_safety
subtopic: medication_review
priority: 98
safety_layer_candidate: true
---

### 知识要点
已用药患者的管理需要结合当前血压、既往方案、药物不良反应及是否存在需要转诊的情况进行调整。指南中的药物调整是面向医疗人员的临床管理行为。

### Agent 边界
患者端 Agent 可以：识别漏服/不规律、解释为什么需要专业评估、提醒携带药物清单；不应：自行决定处方药加量、减量、停药或换药。

---
block_id: MEDSAFE_004
source_id: WHO_HTN_PHARM_2021
source_title: WHO Guideline for the pharmacological treatment of hypertension in adults
year: 2021
source_pages: Section 3.1, 3.4, 3.5, 3.7
population: ["adult", "non_pregnant"]
topic: medication_safety
subtopic: clinical_pharmacotherapy_reference
priority: 80
safety_layer_candidate: false
---

### 知识要点
WHO 指南提供了开始药物治疗、一线药物类别、联合治疗、目标血压和再评估频率的系统推荐。这些内容适合作为临床知识补充和交叉验证来源。

### Agent 使用方式
该来源不应在中国老年患者场景中自动覆盖现行国内指南；涉及具体处方选择时只作为专业知识背景，不直接转成面向患者的个体处方指令。

---
block_id: MEDSAFE_005
source_id: SYSTEM_SAFETY_POLICY
source_title: Agent产品安全约束（由项目范围定义）
year: 2026
source_pages: N/A
population: ["all"]
topic: medication_safety
subtopic: prescription_boundary
priority: 120
safety_layer_candidate: true
---

### 系统硬规则
- Agent 不诊断疾病。
- Agent 不自行启动、停用、更换处方药，也不决定具体加减量。
- 当患者询问“要不要多吃/少吃/停药/换药”时，应先检索权威知识用于安全解释，并根据状态给出联系医生/药师等专业人员进一步评估的行动建议。
- 高风险症状或严重不良反应优先走 Safety Layer，不得为了完成 RAG 检索而延迟升级。

> 说明：这条不是从某一本指南直接摘录的医学推荐，而是本项目为了限制生成式模型权限而设置的系统安全边界。

