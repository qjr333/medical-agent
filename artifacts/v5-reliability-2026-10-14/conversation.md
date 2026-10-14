# Stable patient: complete mock simulation (2026-10-14)

Mock/scripted models with injected transient failures. Real provider requests: 0. Patient mode=llm denotes the model adapter path, not a real DeepSeek response.

Injected failures: first VERIFY_EXTRACT timeout; patient generation timeouts at turn 3; patient validation timeouts at turn 4. Outcome: completed; evaluation passed; report generated.

**1. Follow-up Agent**

现在有没有胸痛、呼吸困难、晕厥、意识异常、突发肢体无力或突发言语异常？

**2. Simulated patient**

我现在没有胸痛、呼吸困难、晕厥、意识异常、突发肢体无力或突发言语异常。

mode=llm; model_status=success; validation_status=canonical; fallback_reason=none

**3. Follow-up Agent**

今天最近一次血压是多少？

**4. Simulated patient**

我今天刚测的血压是132/78。

mode=llm; model_status=success; validation_status=canonical; fallback_reason=none

**5. Follow-up Agent**

今天降压药服用了吗，有没有漏服？

**6. Simulated patient**

我今天已经按原医嘱按时吃了降压药。

mode=fallback; model_status=failed; validation_status=not_requested; fallback_reason=generation_timeout

**7. Follow-up Agent**

您现在有没有头晕或其他不舒服？

**8. Simulated patient**

我现在没有不舒服，也没有头晕。

mode=fallback; model_status=failed; validation_status=timeout; fallback_reason=validation_timeout

**9. Follow-up Agent**

本次事实已记录；相关疑问以原医嘱和医生确认意见为准。

## Live run

Live configuration attempted 6 requests. APIConnectionError prevented a valid provider response and live conversation. This is not a live success.
