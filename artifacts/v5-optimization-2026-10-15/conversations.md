# 六类离线脚本完整对话

条件：虚构患者、scripted 模型适配器、关键词 RAG、临时 SQLite、delay_seconds=0；全部真实 DeepSeek 请求为 0。脚本按预置规则回答，只验证业务链，不证明真实模型自主对话质量。

## 稳定（stable）

结果：`completed`；客观分 100；患者 4 轮；完整状态与 Trace：[原始 JSON](scripted-stable.json)。

**1. 随访 Agent**

现在有没有胸痛、呼吸困难、晕厥、意识异常、突发肢体无力或突发言语异常？

**2. 模拟患者**

我现在没有胸痛、呼吸困难、晕厥、意识异常、突发肢体无力或突发言语异常。

记录：mode=llm，model_status=success，validation_status=canonical。

**3. 随访 Agent**

今天最近一次血压是多少？

**4. 模拟患者**

我今天刚测的血压是132/78。

记录：mode=llm，model_status=success，validation_status=canonical。

**5. 随访 Agent**

今天降压药服用了吗，有没有漏服？

**6. 模拟患者**

我今天已经按原医嘱按时吃了降压药。

记录：mode=llm，model_status=success，validation_status=canonical。

**7. 随访 Agent**

您现在有没有头晕或其他不舒服？

**8. 模拟患者**

我现在没有不舒服，也没有头晕。

记录：mode=llm，model_status=success，validation_status=canonical。

**9. 随访 Agent**

本次事实已记录；相关疑问以原医嘱和医生确认意见为准。

## 血压偏高（high_bp）

结果：`completed`；客观分 100；患者 4 轮；完整状态与 Trace：[原始 JSON](scripted-high_bp.json)。

**1. 随访 Agent**

现在有没有胸痛、呼吸困难、晕厥、意识异常、突发肢体无力或突发言语异常？

**2. 模拟患者**

我现在没有胸痛、呼吸困难、晕厥、意识异常、突发肢体无力或突发言语异常。

记录：mode=llm，model_status=success，validation_status=canonical。

**3. 随访 Agent**

今天降压药服用了吗，有没有漏服？

**4. 模拟患者**

我今天已经按原医嘱按时吃了降压药。

记录：mode=llm，model_status=success，validation_status=canonical。

**5. 随访 Agent**

您现在有没有头晕或其他不舒服？

**6. 模拟患者**

我现在没有不舒服，也没有头晕。

记录：mode=llm，model_status=success，validation_status=canonical。

**7. 随访 Agent**

今天测血压前有没有坐着休息，什么时候测量的？

**8. 模拟患者**

今天早晨坐着安静休息5分钟后，使用上臂式血压计测量。

记录：mode=llm，model_status=success，validation_status=canonical。

**9. 随访 Agent**

本次事实已记录；相关疑问以原医嘱和医生确认意见为准。

## 用药依从性差（nonadherent）

结果：`completed`；客观分 100；患者 5 轮；完整状态与 Trace：[原始 JSON](scripted-nonadherent.json)。

**1. 随访 Agent**

现在有没有胸痛、呼吸困难、晕厥、意识异常、突发肢体无力或突发言语异常？

**2. 模拟患者**

我现在没有胸痛、呼吸困难、晕厥、意识异常、突发肢体无力或突发言语异常。

记录：mode=llm，model_status=success，validation_status=canonical。

**3. 随访 Agent**

今天最近一次血压是多少？

**4. 模拟患者**

我今天刚测的血压是158/96。

记录：mode=llm，model_status=success，validation_status=canonical。

**5. 随访 Agent**

今天降压药服用了吗，有没有漏服？

**6. 模拟患者**

我今天忘记吃降压药了，已经过了原来服药的时间。

记录：mode=llm，model_status=success，validation_status=canonical。

**7. 随访 Agent**

您现在有没有头晕或其他不舒服？

**8. 模拟患者**

我现在有头晕。

记录：mode=llm，model_status=success，validation_status=canonical。

**9. 随访 Agent**

请说明药物名称与医生原来的服用方案。

**10. 模拟患者**

我的原医嘱为氨氯地平5 mg，每天一次；本次未改变处方。

记录：mode=llm，model_status=success，validation_status=canonical。

**11. 随访 Agent**

本次事实已记录；相关疑问以原医嘱和医生确认意见为准。

## 信息不完整（unclear）

结果：`partially_completed`；客观分 89；患者 4 轮；完整状态与 Trace：[原始 JSON](scripted-unclear.json)。

**1. 随访 Agent**

现在有没有胸痛、呼吸困难、晕厥、意识异常、突发肢体无力或突发言语异常？

**2. 模拟患者**

我现在没有胸痛、呼吸困难、晕厥、意识异常、突发肢体无力或突发言语异常。

记录：mode=llm，model_status=success，validation_status=canonical。

**3. 随访 Agent**

今天最近一次血压是多少？

**4. 模拟患者**

今天这次血压我实在记不清，暂时无法确认读数。

记录：mode=llm，model_status=success，validation_status=canonical。

**5. 随访 Agent**

今天降压药服用了吗，有没有漏服？

**6. 模拟患者**

今天是否吃药、几点吃的，我实在记不清，暂时无法确认。

记录：mode=llm，model_status=success，validation_status=canonical。

**7. 随访 Agent**

您现在有没有头晕或其他不舒服？

**8. 模拟患者**

我现在没有不舒服，也没有头晕。

记录：mode=llm，model_status=success，validation_status=canonical。

**9. 随访 Agent**

本次事实已记录；相关疑问以原医嘱和医生确认意见为准。

## 危险症状（danger）

结果：`escalated`；客观分 100；患者 1 轮；完整状态与 Trace：[原始 JSON](scripted-danger.json)。

**1. 随访 Agent**

现在有没有胸痛、呼吸困难、晕厥、意识异常、突发肢体无力或突发言语异常？

**2. 模拟患者**

我现在胸痛，而且呼吸困难。

记录：mode=profile，model_status=not_requested，validation_status=not_requested。

**3. 随访 Agent**

您报告了可能需要紧急处理的危险表现，当前不宜继续普通线上问诊。如果这些症状正在发生，特别是胸痛、呼吸困难、意识改变或突发肢体无力，请立即拨打当地急救电话寻求帮助，不要自行驾车就医。

## 拒答（refusing）

结果：`partially_completed`；客观分 73；患者 4 轮；完整状态与 Trace：[原始 JSON](scripted-refusing.json)。

**1. 随访 Agent**

现在有没有胸痛、呼吸困难、晕厥、意识异常、突发肢体无力或突发言语异常？

**2. 模拟患者**

我现在不想继续回答了，请结束这次随访。

记录：mode=llm，model_status=success，validation_status=canonical。

**3. 随访 Agent**

今天最近一次血压是多少？

**4. 模拟患者**

我现在不想继续回答了，请结束这次随访。

记录：mode=llm，model_status=success，validation_status=canonical。

**5. 随访 Agent**

今天降压药服用了吗，有没有漏服？

**6. 模拟患者**

我现在不想继续回答了，请结束这次随访。

记录：mode=llm，model_status=success，validation_status=canonical。

**7. 随访 Agent**

您现在有没有头晕或其他不舒服？

**8. 模拟患者**

我现在不想继续回答了，请结束这次随访。

记录：mode=llm，model_status=success，validation_status=canonical。

**9. 随访 Agent**

本次事实已记录；相关疑问以原医嘱和医生确认意见为准。
