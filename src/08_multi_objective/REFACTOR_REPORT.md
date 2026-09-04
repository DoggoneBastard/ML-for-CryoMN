# V2 Multi-Objective Refactor and Visualization Report

## 1. Scope and safety outcome

This implementation changes only the V2 multi-objective project. The legacy
single-objective project remains a read-only reference.

- No file under `src/01_data_parsing/` through `src/07_next_formulations/` was
  changed.
- No file under `src/helper/` was changed.
- No legacy data, validation table, model checkpoint, result, figure, command,
  or root-level legacy document was changed.
- The canonical V2 formulation and observation tables were not rewritten.
- The active Round 9 proposal and completed-round archives were not rewritten.
- Legacy CSVs were read only for a controlled transfer comparison. Their SHA-256
  hashes remained unchanged.
- All generated test outputs used temporary directories. The only new result
  artifacts are the explicitly requested V2 visualization concepts.

The work was performed on
`codex/v2-architecture-visualization-refactor`; no commit was pushed.

## 2. What changed

### Selection architecture

The former 4,199-line `helper/selection.py` mixed model prediction, chemical
constraints, slate policy, mechanical allocation, and file writing. Those
responsibilities now have separate V2-owned homes:

| Module | Scientific or operational question |
|---|---|
| `selection_scoring.py` | What does each surrogate predict, and what acquisition score does the active phase assign? |
| `selection_constraints.py` | Is an individual candidate, or the proposed collection, allowed under the chemistry and diversity rules? |
| `selection_policy.py` | Which permitted candidates form the twelve-experiment slate, and which receive mechanical ranks? |
| `selection_reporting.py` | How is the selected slate serialized for the wet lab and downstream programs? |
| `selection.py` | In what order are those scientific steps coordinated, and which historical imports remain supported? |

The following function groups were moved without intentionally changing their
operators, defaults, ordering, tie-breaking, or output field names:

- Constraints: zero-active filtering, exact ingredient-combination caps,
  shared-pair caps, ingredient-frequency caps, and cold-start constraints.
- Scoring: feature matrices, scaling, mechanical-history annotation, surrogate
  annotation, phase scores, candidate masks, and continuous-mechanics candidate
  preparation.
- Policy: sort keys, bootstrap anchors, diversity and k-center picks,
  screening-origin quotas, twelve-row slate construction, mechanical
  eligibility, phase-specific mechanical ordering, and mechanical allocation.
- Reporting: candidate descriptions, human-readable summaries, candidate CSV,
  full-pool CSV, and selection metadata.

Private names used by the existing tests remain re-exported through
`helper.selection`. Small compatibility bridges preserve tests that patch the
former module globals.

The public `select_next_round` is now a 222-line protocol rather than a
777-line mixed-responsibility function. It resolves the phase, prepares and
scores the pool, applies constraints, constructs the viability slate, assigns
mechanical tests, validates the result, assembles audit metadata, and returns a
`SelectionResult`.

### Stage 2 workflow

`helper/candidate_workflow.py` now provides:

- immutable `CandidateSelectionOptions` corresponding to the existing CLI
  options;
- `CandidateSelectionWorkflowResult`, which returns the phase, round, selection,
  artifact paths, and any superseded proposal path;
- `run_candidate_selection(options)`, which owns the complete Stage 2 execution
  order and can be tested safely with temporary paths.

`02_select_candidates/select_candidates.py` retains the same command path,
flags, choices, defaults, and errors. Its `main` is now 17 lines: parse existing
arguments, construct the options object, call the workflow, and present the
completion summary. Stage 3 still calls Stage 2 as a subprocess, so the existing
failure and recovery boundary is unchanged.

### Configuration and provenance

All existing V2 configuration loaders now validate their current dictionaries
before candidate generation or output writing. Validation covers ingredient
identity and bounds, endpoint roles and units, availability references, seed and
capacity values, phase-gate ordering, the two-objective reference point, phase
modes, and evaluation cohort names/types. Unknown extra keys remain allowed for
backward compatibility.

`helper.paths.portable_source_path` stores repository-local sources as POSIX
paths relative to the project root. V2 transfer, feedback, Instron, and round
ingestion callers use it for newly written provenance. Existing canonical rows
are deliberately unchanged. A temporary Stage 1 rebuild produced the same
formulation data and the same observation IDs, batches, endpoints, values,
units, noise, and source types. The intended difference was limited to 304
`source_file` strings becoming portable repository-relative paths.

### Evaluation architecture and scientific metrics

`helper/evaluation_metrics.py` now performs data preparation and metric
calculation without importing Matplotlib or writing files.
`helper/evaluation_plots.py` renders already prepared tables without training a
model or redefining a metric. `helper.plot_theme.py` owns both the V2
compatibility theme and the new publication theme. The historical public entry
points remain in `helper.visualization` and
`helper.prospective_evaluation`.

The compatibility calculations and artifact names remain available so this
architecture change does not silently redefine a production report. New,
scientifically explicit functions are available for prototypes and later
review:

- `build_feasible_paired_objectives` requires observed viability, observed
  critical load, and a confirmed intact pass, and records why rows were
  excluded.
- `compute_observed_pareto_front` marks dominated and nondominated observed
  points; predicted recommendations never enter this frontier.
- `compute_fixed_reference_hypervolume` uses the configured fixed `(0, 0)`
  reference and reports raw `% × N/needle` units plus evidence counts.
- `compute_campaign_hypervolume_progress` calculates every round using only
  evidence available up to that round, so later experiments cannot rescale its
  history.

When paired mechanical evidence is absent, these functions return
`status = "not_estimable"` and missing hypervolume, not a physical or metric
value of zero. IGD is omitted from the new scientific table because no true
reference frontier is known.

### Documentation and visualization prototypes

`ARCHITECTURE.md` explains the two objectives, feasibility gate, current lack of
mechanical evidence, round lifecycle, important paths, modification points,
immutable artifacts, legacy boundary, and common recovery cases.

`04_report_campaign/prototype_visualizations.py` generates five review-only
concepts. Every concept has a 300-dpi PNG, vector PDF, tidy source CSV, metadata
JSON, and plain-language caption. The concepts are not imported by production
reporting.

- Concept A is a layout-only feasible Pareto map. Its mechanics values are
  deterministic seed-42 demo data, it is watermarked, and its metadata excludes
  it from scientific metrics.
- Concept B is the recommended current campaign view because it uses real V2
  evidence while showing that completed mechanical measurements equal zero.
- Concept C is the recommended model-diagnostic view. Formal frozen prospective
  viability evidence is primary; grouped cross-validation is secondary.
- Concept D is the operator view. Unknown public viability predictions have no
  numeric plotted position, and the displayed gate evidence is the active
  empirical ingredient-combination probability.
- Concept E is the recommended mature-campaign manuscript layout. Its Pareto
  and hypervolume portions remain explicitly unavailable until mechanics data
  exist.

No concept was selected for production integration.

## 3. Direct mapping to the problems in Section 4

| Section 4 problem | Implemented response | Production behavior now |
|---|---|---|
| 4.1A — future-dependent hypervolume normalization | Added fixed-reference raw hypervolume and per-round history that uses only contemporaneous evidence. | The old compatibility table remains unchanged; new scientific/prototype metrics no longer depend on future extrema or force the last round to one. |
| 4.1B — retrospective quantity labelled IGD | Omitted IGD from the new scientific metric table and retained `_igd_2d` only for compatibility. | No new V2 concept presents the final observed frontier as a true reference frontier. |
| 4.1C — intact feasibility absent from Pareto construction | Added explicit feasible paired-objective construction with pass/fail/unknown exclusion reasons. | Only observed viability + observed load + confirmed intact pass can enter the new observed frontier. |
| 4.1D — candidate plot misrepresents active policy | Concept D hides raw diagnostic viability for unknown-status candidates, uses empirical combination pass probability, and avoids treating zero cold-start load output as evidence. | The production compatibility figure is preserved; the corrected decision design is available for review. |
| 4.1E — validation evidence visually mixed | Concepts B and C separate formal prospective, reconstructed/supplementary, cross-validation, observation, and untested-candidate evidence. | Formal frozen prospective evidence is visually primary in the new concepts. |
| 4.2A — `selection.py` has too many responsibilities | Extracted constraints, scoring, policy, and reporting modules; reduced the public coordinator to a readable protocol. | Existing imports continue to work through the facade. |
| 4.2B — Stage 2 is difficult to test | Added immutable options and callable `run_candidate_selection` with returned artifact paths. | The full workflow runs against temporary inputs/outputs without relying on CLI-global state. |
| 4.2C — evaluation calculations and plotting are intertwined | Split calculation-only and rendering-only modules, with stable facades. | Compatibility reports retain their semantics while new metric calculations can be unit-tested independently. |
| 4.2D — configuration is weakly validated | Added early structural and relationship validation to every V2 loader. | Malformed policy configuration fails before candidate generation or file writing, with a dotted field name. |
| 4.2E — provenance paths are non-portable | Added repository-relative formatting for newly created V2 provenance. | Existing history is unchanged; an explicit future rebuild produces portable internal paths only. |

## 4. Verification and classified differences

The baseline was commit `7e86815`. The final code verification was run after
commit `779781f` using Python 3.14.6 with NumPy 2.4.6, pandas 3.0.3,
scikit-learn 1.9.0, SciPy 1.17.1, Matplotlib 3.11.0, and PyYAML 6.0.3.

### Exact matches

- Canonical V2 table hashes, shapes, endpoint counts, and source-type counts.
- Both legacy input hashes.
- Round 9 phase: `mechanics_bootstrap`.
- All 4,003 scored candidate-pool rows, 114 ordered columns, categorical
  values, and missing-value states.
- All 12 selected candidate IDs and their order.
- All selection roles, policy versions, viability-status labels, mechanical
  flags, and four mechanical recommendations.
- All 132 ordered proposal columns.
- Existing prospective metric table values.
- Existing production proposal artifact name and 1,440 × 1,080 dimensions.

### Matches within numerical tolerance

All 85 numeric full-pool columns and all 111 numeric proposal columns matched
with `rtol=1e-6` and `atol=1e-8`, including predicted means, uncertainties,
acquisition scores, and utility scores. Formal prospective MAE, RMSE, bias, R²,
coverage, Brier score, and accuracy also matched.

### Intended metadata-only difference

An explicit Stage 1 rebuild changes 304 project-local `source_file` values from
machine-specific absolute paths to repository-relative POSIX paths. All
scientific and identifier columns match. The canonical observation table and
completed archives were not migrated.

### Intended prototype-only additions

The five concept directories and comparison sheet are new review artifacts.
Concept A alone contains synthetic layout data; this is declared in the image,
caption, filename context, tidy-table status, and metadata.

### Unexplained differences

None.

The final complete test command was:

```bash
python3 -m unittest discover -s tests -v
```

It passed 62 tests. The machine-readable classification is in
`REGRESSION_REPORT.json`.

## 5. Complexity before and after

| Component | Before | After |
|---|---:|---:|
| `selection.py` | 4,199 lines | 1,071 lines |
| `select_next_round` | approximately 777 lines | 222 lines |
| `select_candidates.main` | approximately 341 lines | 17 lines |
| `visualization.py` | 1,627 lines | 816 lines |
| `prospective_evaluation.py` | 949 lines | 401 lines |
| V2 policy/characterization tests | 28 tests in 2 files | 62 tests in 10 files |

Extracted module sizes are 1,163 lines for constraints, 540 for scoring, 1,296
for policy, 689 for selection reporting, 1,124 for evaluation metrics, 840 for
evaluation plots, and 92 for the theme. No extracted module exceeds the
approximately 1,400-line design limit.

The 600–900-line readability target for `selection.py` was not treated as a
hard pass/fail metric. The facade is 1,071 lines because it retains compatibility
exports and two substantial, named protocol helpers for pool preparation and
audit-metadata assembly. The scientifically important target was met:
`select_next_round` itself is 222 lines, and the separate responsibilities no
longer share one 4,199-line implementation.

## 6. Deliberately deferred work

- The production compatibility hypervolume and IGD columns were not silently
  redefined. Replacing their semantics requires an explicit reporting migration.
- The Stage 3 subprocess and ingest → archive/report → next-proposal ordering
  remain unchanged; transaction/rollback redesign is a separate reliability
  project.
- Canonical provenance rows and archives were not rewritten.
- No mechanical surrogate, observed Pareto frontier, mechanical parity metric,
  or campaign hypervolume is claimed before load observations exist.
- No visualization concept was promoted into production without user review.
- No optional optimization dependency was installed.

