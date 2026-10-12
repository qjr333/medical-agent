# 05 老年高血压特殊管理（Elderly Management）

> 用途：老年高血压主动随访 Agent 的 RAG 知识文件（V1）。
> 原则：以下内容为对上传指南/标准的结构化摘要，不替代临床诊疗；涉及紧急转诊、处方调整等高风险动作，应由 Safety Layer / 医疗专业人员约束，而不是只依赖 RAG。

本文件用于年龄 ≥65 岁患者的特殊路由，包括血压波动、体位变化、多重用药、衰弱、认知和生活支持。

---
block_id: ELDERLY_001
source_id: CN_ELDERLY_HTN_2023
source_title: 中国老年高血压管理指南2023
year: 2023
source_pages: PDF p.1-2
population: ["age>=65"]
topic: elderly_management
subtopic: elderly_characteristics
priority: 105
safety_layer_candidate: false
---

### 知识要点
老年高血压与一般成人相比具有更明显的收缩压升高、血压波动和多病共存等特点。管理时应更重视安全性、诊室外血压和个体化评估。

### Agent 使用方式
年龄 ≥65 岁时自动增加老年知识源权重；不应完全套用普通成人固定问卷或单一阈值判断。

---
block_id: ELDERLY_002
source_id: CN_ELDERLY_HTN_2023
source_title: 中国老年高血压管理指南2023
year: 2023
source_pages: PDF p.2-3
population: ["age>=65"]
topic: elderly_management
subtopic: bp_variability_and_orthostasis
priority: 105
safety_layer_candidate: true
---

### 知识要点
老年患者血压容易受体位、进餐、情绪、季节/温度等因素影响；直立性血压异常和餐后低血压在老年人中更值得关注，并与晕厥、跌倒等不良事件相关。

### Agent 使用方式
若患者描述“站起来头晕”“饭后发晕”“最近跌倒”，应把体位/进餐与症状时间关系作为高优先级追问目标，而不是仅询问一般头晕。

---
block_id: ELDERLY_003
source_id: CN_ELDERLY_HTN_2023
source_title: 中国老年高血压管理指南2023
year: 2023
source_pages: PDF p.3
population: ["age>=65"]
topic: elderly_management
subtopic: history_and_medications
priority: 105
safety_layer_candidate: false
---

### 知识要点
老年患者评估应覆盖：高血压病程和既往治疗/依从性、心脑肾等并存疾病、正在服用的药物及既往不良反应、生活方式，以及家庭/独居等心理社会因素。

### Agent 使用方式
Patient State 应允许长期保存共病、药物清单、既往不良反应、独居/家庭支持等信息；日常随访按需读取，避免反复询问。

---
block_id: ELDERLY_004
source_id: CN_ELDERLY_HTN_2023
source_title: 中国老年高血压管理指南2023
year: 2023
source_pages: PDF p.5
population: ["age>=65"]
topic: elderly_management
subtopic: frailty
priority: 105
safety_layer_candidate: true
---

### 知识要点
体力活动能力下降或高龄老年高血压患者应关注衰弱。衰弱评估结果会影响治疗强度和目标制定，并强调维持功能与生活质量、谨慎用药及监测不良反应。

### Agent 使用方式
比赛第一版不必让 Agent 自主“诊断衰弱”，但可以加入 `frailty_flag`：活动明显下降、近期非计划体重下降、疲乏或需要他人协助时，提示需要专业评估。

---
block_id: ELDERLY_005
source_id: CN_ELDERLY_HTN_2023
source_title: 中国老年高血压管理指南2023
year: 2023
source_pages: PDF p.5-6
population: ["age>=65"]
topic: elderly_management
subtopic: cognitive_function
priority: 105
safety_layer_candidate: false
---

### 知识要点
认知功能下降会影响老年高血压的长期管理和自我照护。指南建议关注认知筛查，严重认知下降或独居患者的管理需要更个体化并考虑支持体系。

### Agent 使用方式
若对话中出现“反复忘记吃药、记不清测量、独居且认知困难”等线索，可触发 `family_support_needed` 或建议人工/家属协助，而不是简单把所有问题归为“依从性差”。

---
block_id: ELDERLY_006
source_id: CN_ELDERLY_HTN_2023
source_title: 中国老年高血压管理指南2023
year: 2023
source_pages: PDF p.3-4
population: ["age>=65"]
topic: elderly_management
subtopic: home_bp_monitoring
priority: 105
safety_layer_candidate: false
---

### 知识要点
家庭血压监测有助于长期了解老年患者日常血压状态；初始治疗、血压不稳定或方案调整阶段需要更密集的连续监测，稳定后可降低频率，并应详细记录测量日期、时间和读数。

### Agent 使用方式
老年患者的趋势分析优先使用连续记录，不宜只根据一次读数生成结论。

