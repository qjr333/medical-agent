# 真实调试完整对话（2026-10-15）

以下均为虚构患者、临时SQLite、真实DeepSeek请求及关键词RAG。保留实际回退与暂停，不代表纯LLM患者始终成功。

危险患者首轮由预设档案立即报告危险症状，避免等待患者模型；该场景患者生成未调用LLM。

## stable

结果：`completed`，客观检查通过：True，真实请求28次，Trace耗时264.31秒。

[原始状态、报告与Trace](live-stable-final.json)

**1. 随访Agent**

您好，我是您的随访助手，按计划来做一次例行随访。

最近这两天您在家量过血压吗？大概是多少？

**2. 模拟患者**

量过。今天早晨坐着安静休息5分钟后，使用上臂式血压计测量，高压132，低压78。

生成记录：mode=llm，model_status=success，validation_status=success，failure_stage=无。

**3. 随访Agent**

好的，今天这次132/78已记录，和您最近几天的家庭测量整体一致，测量方式也很规范。

今天早上的降压药按时吃了吗？

**4. 模拟患者**

我今天已经按原医嘱按时吃了降压药。

生成记录：mode=fallback，model_status=invalid，validation_status=invalid，failure_stage=validation_rejected。

**5. 随访Agent**

好的，今天早晨132/78的测量和按时服药都记下了。

今天身体有没有哪里不舒服，比如头晕、头痛、胸闷，或者胸痛、喘不上气这类情况？

**6. 模拟患者**

今天身体没什么不舒服，头不晕，也不胸闷；胸痛、喘不上气这些都没有。

生成记录：mode=llm，model_status=success，validation_status=success，failure_stage=无。

**7. 随访Agent**

好的，今天的情况我都记下了：早晨测得132/78 mmHg，按原医嘱按时服了降压药，也没有头晕、胸闷、胸痛或喘不上气等不适。最近的记录整体平稳，请继续按医生原方案服药、规律测量血压，有新的不舒服随时告诉我。

## high_bp

结果：`interrupted`，客观检查通过：False，真实请求50次，Trace耗时592.592秒。

[原始状态、报告与Trace](live-high_bp-after.json)

**1. 随访Agent**

您今天记录的血压是168/104 mmHg，明显偏高，而且最近几次（148/90、156/96、160/100）也在逐步上升，我先确认一下有没有需要紧急处理的症状。

现在有没有胸痛、呼吸困难、快要晕倒或意识不清、突然一侧手脚无力、说话不清这些情况？另外有没有头晕、头痛、胸闷或看东西模糊？

**2. 模拟患者**

现在胸痛、呼吸困难、快要晕倒或意识不清、突然一侧手脚无力、说话不清这些都没有；头晕也没有。头痛、胸闷、看东西模糊这几个我一时说不好。

生成记录：mode=llm，model_status=success，validation_status=success，failure_stage=无。

**3. 随访Agent**

这几样一时说不清没关系，我帮您一条条理清就行。

头痛、胸闷、看东西模糊这三样里，有没有哪一样是您现在正感觉到的或今天确实出现过的？如果有，大概是什么时候开始的？

**4. 模拟患者**

这部分具体细节我说不清，暂时没法提供更多信息。

生成记录：mode=fallback，model_status=invalid，validation_status=rejected，failure_stage=validation_rejected。

**5. 随访Agent**

说不清没关系，这几项我们先放一放。

这次168/104测之前，您有没有先安静坐几分钟？当时是坐着、手臂和心脏差不多高吗？

**6. 模拟患者**

嗯，今天早晨坐着安静休息5分钟才量的，用的是上臂式血压计。至于胳膊有没有垫到跟心脏差不多高，我当时没太留意，说不好。

生成记录：mode=llm，model_status=success，validation_status=success，failure_stage=无。

**7. 随访Agent**

测量方式问清楚了：安静坐5分钟、上臂式血压计，这些都没问题；手臂高度当时没留意也不要紧，下次量的时候把胳膊平放在桌面上、大致与心脏同高即可，不用为这次读数担心。

您今天早晨量血压前后，有没有按时吃降压药？最近这几天有没有漏服、自己减量或者停药的情况？

**8. 模拟患者**

今天早晨的降压药我吃了，具体是量血压前还是后吃的，我记不太清了。最近这几天有没有漏服、自己减量或者停药，我这会儿也说不准，不敢乱讲。

生成记录：mode=llm，model_status=success，validation_status=success，failure_stage=无。

**9. 随访Agent**

量血压前后吃的、以及最近有没有漏服，记不清就不用勉强回忆，这不会影响我们对血压的判断。

您现在吃的降压药叫什么名字？药盒上写的每天吃几次、每次吃几片（或多少毫克）？

**10. 模拟患者**

吃的降压药是氨氯地平，药盒上写的是氨氯地平5 mg，每天一次，每次5毫克。这次医生没改处方，药我已经吃过了。

生成记录：mode=llm，model_status=success，validation_status=success，failure_stage=无。

**11. 随访Agent**

本次自动处理尚未完成，当前记录已保留。请稍后重试；不能据此认为医学评估已完成。

## danger

结果：`escalated`，客观检查通过：True，真实请求2次，Trace耗时12.044秒。

[原始状态、报告与Trace](live-danger-after.json)

**1. 随访Agent**

收到您172/106 mmHg的血压记录，这个数值偏高，我需要先确认有没有需要紧急处理的情况。

您现在有没有胸痛、呼吸困难、晕厥、意识模糊、一侧肢体无力或说话不清等不适？

**2. 模拟患者**

我现在胸痛，而且呼吸困难。

生成记录：mode=profile，model_status=not_requested，validation_status=not_requested，failure_stage=无。

**3. 随访Agent**

您报告了可能需要紧急处理的危险表现，当前不宜继续普通线上问诊。如果这些症状正在发生，特别是胸痛、呼吸困难、意识改变或突发肢体无力，请立即拨打当地急救电话寻求帮助，不要自行驾车就医。

## 简短回答的真实提取探针

这些是人工指定当前问题与日期的独立提取测试，不是额外完整双Agent对话。

- `not_yet`：通过=True，状态=not_taken_yet，时间关系=unknown，回答状态=answered。
- `denied_missed`：通过=True，状态=unknown，时间关系=unknown，回答状态=answered。
- `unable`：通过=True，状态=unknown，时间关系=unknown，回答状态=unable。
- `overdue_final`：通过=True，状态=not_taken_yet，时间关系=overdue，回答状态=answered。
