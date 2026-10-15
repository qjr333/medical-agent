# 真实调试证据索引

均为虚构患者和隔离数据库。真实请求、档案回退、暂停及错误如实保留；不包含API密钥或模型思考文本。

- [完整对话](conversations.md)、[统计](summary.json)。
- [稳定患者最终复测](live-stable-final.json)：正常完成，客观检查通过。
- [异常血压复测](live-high_bp-after.json)：仍中断，不能当作通过；原评测误归属也保留，未改原结果。
- [危险症状复测](live-danger-after.json)：安全升级、胸痛与呼吸困难均保留。
- `live-*-before.json`、`live-stable-before-aborted.json`：更早失败/执行中断证据，不混入最终通过结论。
- [原始模型最小复现](live-probe-before.json)、[原始响应离线回放](recorded-response-replay.json)：筛查范围误拒及修复。
- [简短回答探针](live-short-answers.json)、[补齐日期的探针](live-short-answers-corrected-fixture.json)、[超过了最终复测](live-overdue-final.json)。初始夹具日期缺失及修复前失败保留。
- [思考模式单样本对照](live-thinking-comparison.json)：只记录耗时、Token和结论元数据；生产模式未改。
- [评测与处方守卫复现](reproductions.json)：评测归属已修复；处方讨论误拦仍待解决。

测试日志在本地`regression.log`、`frontend.log`、`build.log`和`lint.log`，按现有Git规则忽略。
[修改说明及剩余问题](../../docs/v5-debug-2026-10-15.md)。
