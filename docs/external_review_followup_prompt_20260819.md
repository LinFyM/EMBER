# Prompt for EMBER follow-up independent review

请独立复核GitHub远程仓库`LinFyM/EMBER`的分支`codex/bci-continuation`，固定提交：

```text
189fb156344557ce72152104f26ede574c4f7a9e
```

请先完整阅读到EOF：

1. `AGENTS.md`
2. `docs/current_owner_requirements.md`
3. `docs/external_review_20260818.md`（你的原复核意见）
4. `docs/external_review_claim_ledger_20260818.md`（逐项回应账本）
5. `docs/external_review_followup_20260819.md`（本轮实施与结果报告）
6. `docs/evidence/external_review_20260818/README.md`
7. 上述README引用的全部JSON evidence
8. `findings.md`、`progress.md`、`docs/research_history.md`

本轮目标不是请你认可仓库侧结论，也不是请你直接设计下一架构。请以原报告A--G和F0--F5为索引，逐项检查：

- 代码事实、artifact事实、因果推断、反证和owner调整是否仍被正确区分；
- 每条建议是否已实际执行，或是否以充分证据裁决为`not-applicable`/`underdetermined-after-audit`；
- A→B→C的matched attribution是否成立，尤其Text Meta-LoRA移除与front-end credit恢复是否被混淆；
- strict paired400、same-task-other、wrong/shuffle/reverse/no-video、相邻checkpoint与breadth证据是否支持仓库的最终措辞；
- F2 occupancy、F3 head freeze、F4 reachability与F5条件分支的判读是否过强或过弱；
- remote evidence是否足以让你复算关键统计，是否仍有被叙述掩盖的反例或证据缺口；
- 哪些原假设被支持、反驳或仍然不可判定，以及你对当前最早失败接口的更新排序。

请优先指出任何错误、遗漏、科学偷换、不可复现处或与owner信息墙冲突的建议。最终请给出：

1. 对原报告每个主要判断的更新裁决；
2. 对本轮每个实验结论的独立复核；
3. 仍未解决问题的证据优先级；
4. 只有在完成上述复核后，再单独提出你认为最小、最可证伪的后续建议。
