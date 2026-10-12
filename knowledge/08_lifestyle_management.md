# 08 生活方式管理（Lifestyle Management）

> 用途：老年高血压主动随访 Agent 的 RAG 知识文件（V1）。
> 原则：以下内容为对上传指南/标准的结构化摘要，不替代临床诊疗；涉及紧急转诊、处方调整等高风险动作，应由 Safety Layer / 医疗专业人员约束，而不是只依赖 RAG。

本文件用于健康教育和长期随访中的生活方式建议；不替代营养师/医生对复杂合并症的个体化方案。

---
block_id: LIFE_001
source_id: CN_NHC_WS_T_872_2025
source_title: 基层医疗卫生机构高血压防治管理标准 WS/T 872-2025
year: 2025
source_pages: PDF p.5（8.2）
population: ["adult", "hypertension"]
topic: lifestyle_management
subtopic: core_lifestyle
priority: 100
safety_layer_candidate: false
---

### 知识要点
高血压长期管理应持续提供健康生活方式指导，包括合理膳食、减少钠盐/增加适宜的钾摄入、控制体重、规律运动、戒烟、戒酒、减轻精神压力和保持健康睡眠。

### Agent 使用方式
Lifestyle Tool 应按患者当前问题选择 1~2 个最相关建议，不要每次把整套健康教育一次性输出。

---
block_id: LIFE_002
source_id: CN_PRIMARY_HTN_GUIDE_2025
source_title: 国家基层高血压防治管理指南2025版
year: 2025
source_pages: PDF p.5-6（5.3生活方式干预）
population: ["adult", "hypertension"]
topic: lifestyle_management
subtopic: salt_and_exercise
priority: 98
safety_layer_candidate: false
---

### 知识要点
2025 基层指南强调限盐、体重管理和规律运动等非药物干预，并把生活方式作为高血压长期管理的基础部分。

### Agent 使用方式
如患者问“平时怎么控制”，优先检索生活方式知识；如同时存在高风险症状，则安全升级优先于健康教育。

---
block_id: LIFE_003
source_id: CN_NHC_LIFESTYLE_HTN_2024
source_title: 高血压营养和运动指导原则（2024年版）
year: 2024
source_pages: PDF p.4附近（营养原则）
population: ["adult", "hypertension"]
topic: lifestyle_management
subtopic: nutrition
priority: 101
safety_layer_candidate: false
---

### 知识要点
- 通过能量摄入与活动平衡维持健康体重；超重/肥胖者需要控制总能量。
- 建议逐步把每日食盐摄入降至 5 g 以下，并减少高钠食物；适当增加富钾食物。
- 低钠盐并非人人适合：肾病、高钾血症等情况需要特别谨慎。
- 鼓励新鲜蔬菜、水果及更健康的脂肪结构，并限制或避免饮酒。

### Agent 使用方式
营养建议必须结合共病和当前用药背景；存在肾病/高钾风险时，不应简单输出“多吃钾/换低钠盐”。

---
block_id: LIFE_004
source_id: CN_ELDERLY_HTN_2023
source_title: 中国老年高血压管理指南2023
year: 2023
source_pages: PDF p.7附近（生活方式干预）
population: ["age>=65"]
topic: lifestyle_management
subtopic: elderly_exercise
priority: 105
safety_layer_candidate: false
---

### 知识要点
老年高血压患者适合以可耐受的有氧活动为主，并结合抗阻、平衡、柔韧等训练；指南强调老年人不宜进行剧烈运动，同时应结合功能状态和安全性个体化。

### Agent 使用方式
老人端运动建议必须先考虑跌倒风险、衰弱、当前症状和既往运动能力，不应机械给出“每天跑步”等统一建议。

---
block_id: LIFE_005
source_id: CN_ELDERLY_HTN_2023
source_title: 中国老年高血压管理指南2023
year: 2023
source_pages: PDF p.7
population: ["age>=65"]
topic: lifestyle_management
subtopic: elderly_diet_and_environment
priority: 105
safety_layer_candidate: false
---

### 知识要点
老年高血压生活方式管理强调适度限盐、合理膳食，同时关注寒冷环境、睡眠和情绪等可能导致血压波动的因素。

### Agent 使用方式
对老年患者的健康教育可以与 Trigger Engine 结合，例如季节性/天气相关提醒未来可作为扩展功能，但不应在当前 Demo 中扩大到无关场景。

