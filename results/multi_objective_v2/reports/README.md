# Multi-objective campaign reports

The production suite exports transparent PNGs at 300 dpi. Source CSVs and metadata accompany each figure.

- [A — observed trade-off](plots/observed_tradeoff.png)
- [B — campaign timeline](prospective/plots/campaign_timeline.png)
- [C — frozen-prediction trust](prospective/plots/surrogate_trust.png)
- [D — current Round 9 decisions](../rounds/ROUND_009/proposal/plots/candidate_decisions.png)
- [Diagnostics 1 — model validation](plots/diagnostics_1.png)
- [Diagnostics 2 — paired-objective evaluation](plots/diagnostics_2.png)
- [E — optional publication summary](prospective/plots/publication_summary.png)

D is a frozen Round 9 proposal view. Diagnostic CV points here reuse the latest stored Round 8 evaluation, without refitting. Mechanical evidence is currently missing; the unavailable panels are intentional.

Historical single-round plots are under each round's `reports/plots/`. Proposals lacking sufficient stored decision evidence and diagnostics lacking archived paired model evaluations retain their original files. See the [backfill manifest](plot_backfill_manifest.json) and [migration verification](plot_migration_verification.json).

[Plot definitions, scientific references and regeneration commands](../../../src/08_multi_objective/04_report_campaign/PLOTTING.md)
