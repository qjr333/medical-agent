# 02 主动随访流程（Follow-up Workflow）

> 用途：老年高血压主动随访 Agent 的 RAG 知识文件（V1）。
> 原则：以下内容为对上传指南/标准的结构化摘要，不替代临床诊疗；涉及紧急转诊、处方调整等高风险动作，应由 Safety Layer / 医疗专业人员约束，而不是只依赖 RAG。

本文件定义随访应该收集什么、多久随访、何时年度评估、如何衔接转诊，以及“主动联系”在公卫管理中的依据。

---
block_id: FOLLOW_001
source_id: CN_NHC_WS_T_872_2025
source_title: 基层医疗卫生机构高血压防治管理标准 WS/T 872-2025
year: 2025
source_pages: PDF p.5
population: ["adult", "hypertension"]
topic: followup_workflow
subtopic: followup_frequency
priority: 100
safety_layer_candidate: false
---

### 知识要点
- 已确诊高血压患者需要长期随访。
- 血压达标者：至少每 3 个月随访 1 次。
- 血压未达标者：至少每月随访 1 次，直至达标；符合转诊条件时应进入转诊流程。

### Agent 使用方式
这可用于数据库的 `next_followup_due` 和 Trigger Engine 的计划随访触发，不应用作对个体处方的自动调整依据。

---
block_id: FOLLOW_002
source_id: CN_NHC_WS_T_872_2025
source_title: 基层医疗卫生机构高血压防治管理标准 WS/T 872-2025
year: 2025
source_pages: PDF p.5
population: ["adult", "hypertension"]
topic: followup_workflow
subtopic: followup_content
priority: 100
safety_layer_candidate: false
---

### 知识要点
每次随访应围绕以下方面更新状态：
- 治疗效果与当前血压；
- 基本体格检查；
- 当前用药情况；
- 自我管理情况；
- 个体化健康教育与生活方式；
- 有条件时指导规范家庭血压监测。
同时，每年应进行一次较全面的年度评估。

### Agent 使用方式
Patient State 不需要每轮把所有字段问一遍，而是根据缺失项和当前风险动态选择最重要的信息目标。

---
block_id: FOLLOW_003
source_id: CN_PRIMARY_HTN_GUIDE_2025
source_title: 国家基层高血压防治管理指南2025版
year: 2025
source_pages: PDF p.13
population: ["adult", "hypertension"]
topic: followup_workflow
subtopic: dynamic_followup_fields
priority: 98
safety_layer_candidate: false
---

### 知识要点
随访除血压外，还应了解上次随访后的新诊断或合并症，进行血压、心率等检查，评估生活方式，并了解服药依从性和药物不良反应。

### Agent 使用方式
建议 Patient State 至少具备：`current_bp`、`new_comorbidities`、`medication_status`、`adverse_effects`、`symptoms`、`lifestyle_flags`、`risk_level`。Agent 根据触发原因和当前状态选择需要补充的字段，而非固定问卷。

---
block_id: FOLLOW_004
source_id: CN_BASIC_PUBLIC_HEALTH_2017
source_title: 国家基本公共卫生服务规范（第三版）
year: 2017
source_pages: PDF p.59-60（高血压患者健康管理服务规范）
population: ["community", "hypertension"]
topic: followup_workflow
subtopic: active_contact
priority: 82
safety_layer_candidate: false
---

### 知识要点
- 规范要求对未能按管理要求接受随访的患者主动联系，以保持管理连续性。
- 随访方式可以包括预约门诊、电话追踪和家庭访视。
- 每次服务后应及时更新健康档案。

### Agent 使用方式
这为“主动随访”产品逻辑提供业务依据：系统可以因到期、失访/未测量或异常数据主动创建随访任务，而不仅等待患者发起聊天。

---
block_id: FOLLOW_005
source_id: CN_PRIMARY_HTN_GUIDE_2025
source_title: 国家基层高血压防治管理指南2025版
year: 2025
source_pages: PDF p.13
population: ["adult", "hypertension"]
topic: followup_workflow
subtopic: annual_assessment
priority: 96
safety_layer_candidate: false
---

### 知识要点
所有患者每年应进行一次较全面的年度评估；可与随访结合。内容包括体格指标以及必要的血液、尿液、生化和心电图等检查，有条件时可根据情况增加其他检查。

### Agent 使用方式
年度评估应设计为独立的“任务类型”，不应在普通每日/每月主动随访中一次性询问所有年度项目。

---
block_id: FOLLOW_006
source_id: CN_PRIMARY_HTN_GUIDE_2025
source_title: 国家基层高血压防治管理指南2025版
year: 2025
source_pages: PDF p.12-13
population: ["adult", "hypertension"]
topic: followup_workflow
subtopic: referral_closed_loop
priority: 98
safety_layer_candidate: false
---

### 知识要点
患者转诊后，基层端仍需要追踪上级医院诊断/治疗结果；治疗稳定后可回到基层继续长期随访管理。

### Agent 使用方式
产品流程应形成 `发现异常 -> 转诊/就医 -> 结果回收 -> 恢复长期随访` 的闭环，而不是在生成“建议就医”后结束所有管理。

