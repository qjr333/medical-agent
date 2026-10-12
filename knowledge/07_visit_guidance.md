# 07 就医与行动建议（Visit Guidance）

> 用途：老年高血压主动随访 Agent 的 RAG 知识文件（V1）。
> 原则：以下内容为对上传指南/标准的结构化摘要，不替代临床诊疗；涉及紧急转诊、处方调整等高风险动作，应由 Safety Layer / 医疗专业人员约束，而不是只依赖 RAG。

本文件不用于诊断，而用于把 Patient State + 风险 + RAG 证据转成下一步行动：继续家庭监测、基层/专科评估、转诊以及需要携带的信息。

---
block_id: VISIT_001
source_id: CN_NHC_WS_T_872_2025
source_title: 基层医疗卫生机构高血压防治管理标准 WS/T 872-2025
year: 2025
source_pages: PDF p.5-6
population: ["adult", "hypertension"]
topic: visit_guidance
subtopic: care_level_selection
priority: 105
safety_layer_candidate: false
---

### 知识要点
稳定的原发性高血压可以在基层持续管理；出现标准所列的复杂、严重或难以处理的问题时，应进入上级医疗机构评估，并在稳定后回归基层长期管理。

### Agent 使用方式
`GENERATE_VISIT_ADVICE` 至少输出：`urgency`、`recommended_care_level`、`reason`、`what_to_prepare`，而不是只说“去医院看看”。

---
block_id: VISIT_002
source_id: CN_PRIMARY_HTN_GUIDE_2025
source_title: 国家基层高血压防治管理指南2025版
year: 2025
source_pages: PDF p.12-13
population: ["adult", "hypertension"]
topic: visit_guidance
subtopic: referral_followup
priority: 102
safety_layer_candidate: false
---

### 知识要点
转诊后基层端应主动了解上级医院诊断或治疗结果；若血压达标/病情稳定，可恢复基层常规随访；若问题仍未解决，则继续上级治疗评估。

### Agent 使用方式
报告应保存 `referral_status` 和 `followup_after_referral_due`，用于下一次主动触发。

---
block_id: VISIT_003
source_id: CN_PRIMARY_HTN_GUIDE_2025
source_title: 国家基层高血压防治管理指南2025版
year: 2025
source_pages: PDF p.13
population: ["adult", "hypertension"]
topic: visit_guidance
subtopic: information_for_review
priority: 96
safety_layer_candidate: false
---

### 知识要点
随访和年度评估都依赖连续血压、当前用药、依从性、不良反应、症状/新发合并症和相关检查信息。

### Agent 使用方式
由此可将就医准备整理为：近期家庭血压记录、当前用药清单/服药情况、本次症状时间线、近期检查/就诊资料。该“准备清单”是对指南要求评估信息的产品化整理，不是新的医学诊断建议。

---
block_id: VISIT_004
source_id: CN_BASIC_PUBLIC_HEALTH_2017
source_title: 国家基本公共卫生服务规范（第三版）
year: 2017
source_pages: PDF p.59-60
population: ["community", "hypertension"]
topic: visit_guidance
subtopic: active_referral_tracking
priority: 82
safety_layer_candidate: false
---

### 知识要点
公卫规范要求在转诊后主动追踪结果，并把相关信息纳入健康档案，强调管理连续性。

### Agent 使用方式
家属/医护提醒不应只是一次通知；可以创建“等待上级就诊结果”的后续任务，由 Trigger Engine 在规定时间再次检查状态。

---
block_id: VISIT_005
source_id: CN_ELDERLY_HTN_2023
source_title: 中国老年高血压管理指南2023
year: 2023
source_pages: PDF p.3-6
population: ["age>=65"]
topic: visit_guidance
subtopic: elderly_review_context
priority: 105
safety_layer_candidate: false
---

### 知识要点
老年患者的进一步评估不仅看血压，还应综合共病、体位性变化、药物、不良反应、衰弱和认知等情况。

### Agent 使用方式
72 岁患者的就医摘要应优先带上：近期血压趋势、体位/跌倒相关症状、完整药物清单、依从性、不良反应以及家属支持信息（如存在）。

