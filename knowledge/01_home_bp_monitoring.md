# 01 家庭血压监测（Home BP Monitoring）

> 用途：老年高血压主动随访 Agent 的 RAG 知识文件（V1）。
> 原则：以下内容为对上传指南/标准的结构化摘要，不替代临床诊疗；涉及紧急转诊、处方调整等高风险动作，应由 Safety Layer / 医疗专业人员约束，而不是只依赖 RAG。

本文件回答：怎么量、何时量、量几次、如何记录、老年患者有哪些特殊注意点，以及哪些数据适合触发主动随访。

---
block_id: HBPM_001
source_id: CN_HOME_BP_2019
source_title: 2019中国家庭血压监测指南
year: 2019
source_pages: PDF p.3
population: ["general", "hypertension"]
topic: home_bp_monitoring
subtopic: measurement_conditions
priority: 90
safety_layer_candidate: false
---

### 知识要点
- 家庭测压应在相对安静、放松的状态下进行；测量前先坐位休息约 5 分钟。
- 测量时上臂应得到支撑，并尽量保持袖带位置与心脏水平一致。
- 家庭血压结果容易受近期活动、情绪、进食、排尿等因素影响，因此 Agent 在出现单次异常值时可以询问测量背景，而不是直接把单次读数等同于病情恶化。

### Agent 使用方式
当出现异常读数且测量情境未知时，可生成 `ASK_USER` 目标：确认是否刚活动、是否充分休息、测量姿势是否规范等。

---
block_id: HBPM_002
source_id: CN_HOME_BP_2019
source_title: 2019中国家庭血压监测指南
year: 2019
source_pages: PDF p.3
population: ["general", "hypertension"]
topic: home_bp_monitoring
subtopic: device_selection
priority: 90
safety_layer_candidate: false
---

### 知识要点
- 优先使用经过准确性验证的上臂式全自动电子血压计。
- 血压计应定期校准；该指南建议至少每年校准一次。
- 袖带尺寸应与上臂周径匹配。
- 手指式设备不适合作为家庭血压监测的常规选择；无袖带技术在该指南发布时尚不建议作为标准家庭测压依据。

### Agent 使用方式
如果患者报告的读数与平时明显不一致，可将“设备类型、袖带、设备校准情况”作为后续信息，而不是立刻根据异常值给出治疗结论。

---
block_id: HBPM_003
source_id: CN_HOME_BP_2019
source_title: 2019中国家庭血压监测指南
year: 2019
source_pages: PDF p.3-4
population: ["general", "hypertension"]
topic: home_bp_monitoring
subtopic: frequency_and_timing
priority: 92
safety_layer_candidate: false
---

### 知识要点
- 家庭监测可安排早、晚测量；每次测量前先安静坐位休息，并进行多次读数后综合记录。
- 该指南建议每次测量 2~3 次、间隔约 1 分钟；初诊、治疗早期或血压尚未达标者，在就诊前可连续监测数日；控制稳定后可降低监测频率。
- 早晨测量通常安排在起床后、早餐和服用降压药物之前；晚间测量宜在睡前附近进行，并注意测量前排空膀胱。
- 应同时记录起床、睡觉、进餐和服药时间，以便解释血压变化。

### Agent 使用方式
Agent 可以基于这些信息生成“规范复测”或“补充测量背景”的建议，但不应仅凭一次异常读数自行调整处方。

---
block_id: HBPM_004
source_id: CN_HOME_BP_2019
source_title: 2019中国家庭血压监测指南
year: 2019
source_pages: PDF p.2
population: ["general", "hypertension"]
topic: home_bp_monitoring
subtopic: data_record_and_upload
priority: 92
safety_layer_candidate: false
---

### 知识要点
- 具有通信功能的电子血压设备可以把血压数据传到管理网站、手机应用或程序中进行存储和分析。
- 如果不能自动上传，也应系统记录测量日期、时间、收缩压、舒张压和脉搏等信息。
- 远程数据并不是替代临床判断，而是为长期趋势判断和随访提供更连续的数据基础。

### Agent 使用方式
这条知识直接支持本项目的 `家庭设备 -> 数据上传 -> Trigger Engine -> 主动随访` 架构。

---
block_id: HBPM_005
source_id: CN_ELDERLY_HTN_2023
source_title: 中国老年高血压管理指南2023
year: 2023
source_pages: PDF p.3-4
population: ["age>=65"]
topic: home_bp_monitoring
subtopic: elderly_specific
priority: 105
safety_layer_candidate: false
---

### 知识要点
- 老年高血压患者血压波动更明显，家庭血压和动态血压有助于了解日常状态下的变化。
- 老年患者应特别关注不同体位下的血压变化；必要时需测量卧位/坐位转为站立后的血压。
- 对老年患者进行长期家庭监测时，要结合跌倒、晕厥、餐后不适、认知状态等背景解释读数，不能只看一个数字。

### Agent 使用方式
患者年龄 ≥65 岁时，本知识块应获得额外人群权重。出现“起身头晕”等描述时，应优先路由到 `elderly_management` / `orthostatic_bp` 相关知识。

---
block_id: HBPM_006
source_id: CN_HTN_GUIDE_2024
source_title: 中国高血压防治指南（2024年修订版）
year: 2024
source_pages: PDF p.6-8
population: ["adult", "hypertension"]
topic: home_bp_monitoring
subtopic: remote_monitoring
priority: 96
safety_layer_candidate: false
---

### 知识要点
- 2024 版指南强调诊室外血压测量的价值，包括家庭血压监测和动态血压监测。
- 指南将基于互联网的远程实时家庭血压监测视为血压管理的新模式之一。
- 家庭血压可用于长期观察，并可帮助识别诊室血压与日常状态不一致的情况。

### Agent 使用方式
远程家庭血压数据可作为主动触发来源，但触发后的 Agent 仍需结合症状、服药情况、趋势和测量背景再决策。

