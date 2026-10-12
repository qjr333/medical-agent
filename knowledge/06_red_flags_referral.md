# 06 危险信号与转诊（Red Flags & Referral）

> 用途：老年高血压主动随访 Agent 的 RAG 知识文件（V1）。
> 原则：以下内容为对上传指南/标准的结构化摘要，不替代临床诊疗；涉及紧急转诊、处方调整等高风险动作，应由 Safety Layer / 医疗专业人员约束，而不是只依赖 RAG。

本文件中的高风险条目应优先实现为确定性 Safety Layer。RAG 负责提供依据和解释，不负责决定“是否忽略危险”。

---
block_id: RED_001
source_id: CN_NHC_WS_T_872_2025
source_title: 基层医疗卫生机构高血压防治管理标准 WS/T 872-2025
year: 2025
source_pages: PDF p.5-6（8.4）
population: ["adult", "hypertension"]
topic: red_flags_referral
subtopic: general_referral
priority: 110
safety_layer_candidate: true
---

### 知识要点
现行基层标准将以下情况纳入转诊范围：起病急/症状重、怀疑继发性高血压、新发严重并发症或严重合并症、难治性高血压，以及妊娠/哺乳期高血压等。

### Safety Layer 候选
当状态满足明确的高风险转诊条件时，Action Selector 不应继续普通信息收集循环，应进入 `ESCALATE` 或专业转诊路径。

---
block_id: RED_002
source_id: CN_NHC_WS_T_872_2025
source_title: 基层医疗卫生机构高血压防治管理标准 WS/T 872-2025
year: 2025
source_pages: PDF p.5（8.4.2）
population: ["adult", "hypertension"]
topic: red_flags_referral
subtopic: initial_referral
priority: 110
safety_layer_candidate: true
---

### 知识要点
初诊时，明显升高且短期处理仍不能控制的血压、怀疑新发心/脑/肾严重问题、蛋白尿/血尿、特定低钾情况、阵发性血压升高伴明显症状、双上臂血压差异显著以及需要进一步鉴别继发原因等，均属于应考虑上级评估的情况。

### Agent 使用方式
这些条件更适合作为结构化规则或 clinician-facing warning，而不是由对话模型自由解释后才决定是否升级。

---
block_id: RED_003
source_id: CN_NHC_WS_T_872_2025
source_title: 基层医疗卫生机构高血压防治管理标准 WS/T 872-2025
year: 2025
source_pages: PDF p.5（8.4.3）
population: ["adult", "hypertension"]
topic: red_flags_referral
subtopic: followup_referral
priority: 110
safety_layer_candidate: true
---

### 知识要点
随访中出现以下类型问题，应进入上级专业评估：多种降压药充分使用仍未达标、血压明显波动且难控制、怀疑难处理的药物不良反应、发现严重临床疾病或心脑肾损害等。

### Agent 使用方式
将这些状态设计成 `referral_reason[]`，支持报告和医护提醒，不要只返回一个 `risk_level=high`。

---
block_id: RED_004
source_id: CN_PRIMARY_HTN_GUIDE_2025
source_title: 国家基层高血压防治管理指南2025版
year: 2025
source_pages: PDF p.12（第7节转诊）
population: ["adult", "hypertension"]
topic: red_flags_referral
subtopic: emergency_transfer
priority: 112
safety_layer_candidate: true
---

### 知识要点
指南列出的严重情况包括：意识丧失/模糊，高血压伴剧烈头痛呕吐或突发神经功能异常，高血压伴持续剧烈胸背痛，血压升高伴呼吸困难/不能平卧，以及持续胸痛伴明显急性心血管事件线索等。此类情况需要快速线下急诊评估。

### Safety Layer 候选
关键词只是后备检测手段。正式实现应结合患者明确表述、否认信息和当前状态，命中时立即中止普通随访并提示当地急救/急诊评估。

---
block_id: RED_005
source_id: CN_PRIMARY_HTN_GUIDE_2025
source_title: 国家基层高血压防治管理指南2025版
year: 2025
source_pages: PDF p.4（4.2诊断流程附近）
population: ["adult", "hypertension"]
topic: red_flags_referral
subtopic: very_high_bp_with_symptoms
priority: 110
safety_layer_candidate: true
---

### 知识要点
当血压达到明显升高范围且伴急性症状时，指南强调及时转诊；如果无明显急性症状，需要排除诱因、休息后复测并结合进一步评估，而不是只按一个数字处理。

### Agent 使用方式
Risk Engine 应区分“数值异常”与“数值异常 + 急性危险表现”。后者优先升级，前者进入复测/趋势/症状/用药评估。

---
block_id: RED_006
source_id: CN_BASIC_PUBLIC_HEALTH_2017
source_title: 国家基本公共卫生服务规范（第三版）
year: 2017
source_pages: PDF p.59（高血压随访评估）
population: ["community", "hypertension"]
topic: red_flags_referral
subtopic: historical_public_health_emergency_flow
priority: 78
safety_layer_candidate: true
---

### 知识要点
公卫规范的随访流程同样把危急情况评估放在普通随访之前，并要求紧急处理后转诊、之后主动追踪转诊结果。

### 冲突说明
该文件年代较早；具体阈值和临床标准若与 2024/2025 资料存在差异，以更新标准/指南为优先。它更适合作为“安全优先 + 转诊后追踪”的流程依据。

