# 03 服药依从性（Medication Adherence）

> 用途：老年高血压主动随访 Agent 的 RAG 知识文件（V1）。
> 原则：以下内容为对上传指南/标准的结构化摘要，不替代临床诊疗；涉及紧急转诊、处方调整等高风险动作，应由 Safety Layer / 医疗专业人员约束，而不是只依赖 RAG。

本文件只解决“有没有按既定处方规律服药、为什么没有、如何记录和升级”，不用于让 AI 自行改处方。

---
block_id: MEDADH_001
source_id: CN_PRIMARY_HTN_GUIDE_2025
source_title: 国家基层高血压防治管理指南2025版
year: 2025
source_pages: PDF p.13
population: ["adult", "hypertension"]
topic: medication_adherence
subtopic: routine_assessment
priority: 98
safety_layer_candidate: false
---

### 知识要点
随访时应了解服药依从性以及是否出现药物不良反应。血压未达标时，不能只看血压数值，还应检查生活方式、用药执行情况及是否需要进一步专业评估。

### Agent 使用方式
如果 `medication_status=unknown` 且本次触发与血压异常有关，`ASK_USER` 可优先确认今天/近期是否按既定方案服药。

---
block_id: MEDADH_002
source_id: CN_BASIC_PUBLIC_HEALTH_2017
source_title: 国家基本公共卫生服务规范（第三版）
year: 2017
source_pages: 健康体检表/高血压随访记录相关字段
population: ["community", "hypertension"]
topic: medication_adherence
subtopic: structured_status
priority: 82
safety_layer_candidate: false
---

### 知识要点
公卫健康档案把主要用药情况作为结构化字段记录，并把服药依从性区分为规律、间断、不服药等状态。

### Agent 使用方式
Patient State 可使用结构化值：`taken / missed / irregular / stopped / unknown`，同时保留 `medication_detail` 记录原因和时间。不要只保存一段自由文本。

---
block_id: MEDADH_003
source_id: CN_NHC_WS_T_872_2025
source_title: 基层医疗卫生机构高血压防治管理标准 WS/T 872-2025
year: 2025
source_pages: PDF p.5-6
population: ["adult", "hypertension"]
topic: medication_adherence
subtopic: patient_education
priority: 100
safety_layer_candidate: false
---

### 知识要点
健康教育应覆盖高血压患者的自我管理、坚持长期治疗的必要性，以及对降压药不良反应的正确认识。

### Agent 使用方式
用药依从性 Tool 的目标不仅是识别“漏服”，还应识别影响长期坚持治疗的障碍，并把需要医生处理的不良反应与普通遗忘分开。

---
block_id: MEDADH_004
source_id: CN_ELDERLY_HTN_2023
source_title: 中国老年高血压管理指南2023
year: 2023
source_pages: PDF p.3, p.7及治疗章节
population: ["age>=65"]
topic: medication_adherence
subtopic: elderly_context
priority: 105
safety_layer_candidate: false
---

### 知识要点
老年患者评估需要了解既往降压治疗及依从性、正在使用的药物、既往药物不良反应，以及可能影响执行治疗的认知和生活状况。

### Agent 使用方式
对于老年患者，漏服不能只记录“是/否”；可以在必要时继续了解：是否忘记、是否因不适自行停药、是否存在认知/自理困难、是否需要家属协助。

---
block_id: MEDADH_005
source_id: CN_HTN_GUIDE_2024
source_title: 中国高血压防治指南（2024年修订版）
year: 2024
source_pages: 治疗与管理相关章节（PDF约 p.55-71）
population: ["adult", "hypertension"]
topic: medication_adherence
subtopic: uncontrolled_bp_check
priority: 94
safety_layer_candidate: false
---

### 知识要点
当血压控制不理想时，应把服药依从性、用药相关不良反应、生活方式和测量情况纳入整体评估，不能把“数值高”简单等同于“药量不够”。

### Agent 使用方式
这支持 Agent 的决策顺序：异常数据 -> 核实测量/趋势 -> 了解依从性和症状 -> 必要时检索知识/建议专业评估，而不是直接进入药量调整。

