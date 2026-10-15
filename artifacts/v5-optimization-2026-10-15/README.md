# V5 优化验证证据（2026-10-15）

所有患者为虚构资料。脚本模拟使用临时数据库、关键词 RAG，真实模型请求为0，不写入项目患者数据库。

- [完整六类离线对话](conversations.md)：稳定、异常血压、用药不规律、模糊回答、危险症状、拒答。
- [机器可读结果](summary.json)：测试条件、各场景结果、调用次数、对比及验证限制。
- `scripted-*.json`：完整对话、最终状态、Trace、报告、医生处理和家属已读记录。
- [修改前上下文](context-before.json)、[修改后上下文](context-after.json)：相同确定性样本，比较序列化字符数，不是模型 token/延迟测量。
- [真实调用首次失败](stable-live-attempt-1.json)、[真实调用网络重试失败](stable-live-network-retry.json)、[最小连通性检查](connectivity.json)：保留失败证据，没有成功的真实模型完整对话。

脚本记录中的旧字段 `mode=llm` 仅表示模型适配器路径返回成功，不等于发出了真实模型请求。
应结合顶层 `model_mode=scripted`、`validation_status=canonical` 和实际 provider 请求计数理解。
真实模式已禁止利用仅事实ID输出或脚本事实渲染绕过自然语言验证。

按用户“DeepSeek服务器崩掉了，不用测了”的要求，已停止后续真实调用，没有安排自动重测。
单元/Mock、脚本链路与真实模型结果分别报告，详见 [交付记录](../../docs/v5-optimization-result.md)。
