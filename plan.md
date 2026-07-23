# Repurpose the disentanglement architecture for cancer-conditioned MACE modeling

## Context

**Current architecture** ([disentangle.py.py:307-332](disentangle.py.py#L307-L332), [disentangle.py.py:387-429](disentangle.py.py#L387-L429)):

```
X -> shared_encoder      -> E1 (shared embedding)
X -> private_encoder_1   -> P1 (task-1-private embedding)
X -> private_encoder_2   -> P2 (task-2-private embedding)
[E1;P1] -> decoder_1 -> classifier_1 -> Y1 logit   (currently: cancer)
[E1;P2] -> decoder_2 -> classifier_2 -> Y2 logit   (currently: MACE)
```

`difference_loss` (the "orthogonality" term, [disentangle.py.py:436-446](disentangle.py.py#L436-L446)) is a DSN-style (Bousmalis et al. 2016) soft constraint: row-normalize `E1` and each private embedding, then penalize the squared cross-correlation between them. It pushes `E1` and `P1`/`P2` toward encoding *non-redundant* information — it does **not** directly optimize any AUC, and it does not push `P1` vs `P2` apart from each other (only each vs. the shared `E1`).

**Why this doesn't match the stated goal.** The goal is two MACE representations — one specialized for cancer patients, one for non-cancer patients — with performance genuinely differing between them. But today `Y1` (cancer) is a second *predicted task*, not a conditioning variable, so the architecture disentangles "cancer-predictive signal" from "MACE-predictive signal" — a different question (do these two label spaces share mechanism?) than "does the MACE mechanism itself differ by cancer status?" Two pieces of evidence from `logs.md` confirm this mismatch is actually biting:
- `P2->Y1(leak)` sits at 0.7–0.88 the whole run — the "MACE-private" embedding still predicts cancer status well, i.e. real disentanglement isn't being achieved, likely because cancer-cohort membership and MACE risk are genuinely correlated in this data, and forcing orthogonality fights that.
- The subgroup breakdown added last turn shows train `MACE AUC[cancer+]` climbing to 0.909 while test stays at 0.631 (vs. `[cancer-]` train 0.807/test 0.616) — i.e. the apparent train-time subgroup gap is mostly overfitting on the smaller cancer+ group (~2.2k train rows vs. ~14k non-cancer), not evidence of a real effect-modification signal.

**Decisions made with the user** (via AskUserQuestion this turn):
1. Cancer status will be treated as a **known grouping variable**, always available before MACE is assessed (matches the "MACE only in cancer patients" framing) — not a task the model must predict. The standalone cancer classifier is dropped.
2. The two group-specific MACE representations will use a **shared trunk + group-private residual** (reusing the existing DSN-style machinery) rather than two fully independent networks — important because the cancer+ subgroup is much smaller and benefits from pooling through the shared encoder.

## Recommended approach

Reframe the *single* target (`Y_mace`) as two masked "branches" routed by the known cancer-group label, reusing the existing missing-label masking machinery verbatim — no changes needed to `MultiTaskDataset`, `DisentangledMultiTaskNet`, `compute_losses`, `difference_loss`, or `masked_bce`. Only the data-prep call site and print/diagnostic labels change:

- **Branch A** (`Y1`/`classifier_1`/`P1`): MACE label for cancer-positive rows, `NaN` elsewhere.
- **Branch B** (`Y2`/`classifier_2`/`P2`): MACE label for cancer-negative rows, `NaN` elsewhere.
- `MultiTaskDataset` already derives its mask from `~isnan(y)` ([disentangle.py.py:346-363](disentangle.py.py#L346-L363)), so setting the label to `NaN` outside a branch's group automatically excludes those rows from that branch's loss/AUC — this is the exact same mechanism already used for genuinely missing labels, just repurposed to encode group membership.
- `E1` becomes "the common MACE mechanism," `P1`/`P2` become "cancer-specific" / "non-cancer-specific" residuals on top of it. `orth1`/`orth2` keep their current meaning (shared vs. each group-private) — **no `P1` vs `P2` orthogonality term is being added**, since that would regularize representational redundancy, not the thing actually being measured (a performance gap is an empirical outcome, not a target to directly optimize).
- The subgroup-AUC feature added last turn becomes redundant: `auc1`/`auc2` from `evaluate_task_performance` *are already* "MACE AUC in cancer patients" / "MACE AUC in non-cancer patients" once the masks encode group membership, so the extra `subgroup_auc` machinery gets removed rather than kept as dead weight.
- `Y_mace` has no missing values in this data (`np.bincount` on it already succeeds in the current logs), so the branch masks reduce to pure group membership — no interaction with real missingness to worry about.

### Concrete edits (all in `disentangle.py.py`)

1. **Docstring** ([disentangle.py.py:307-332](disentangle.py.py#L307-L332)): rewrite to describe the new framing — one target (MACE), two group-conditioned branches routed by known cancer status, shared trunk + group-private residual — instead of "two label sets Y1, Y2."

2. **Data prep** ([disentangle.py.py:763-794](disentangle.py.py#L763-L794)): replace the `train_Y1`/`train_Y2`/`test_Y1`/`test_Y2` construction. Keep `train_eid`, `train_X`, `test_eid`, `test_X` as-is. Add:
   ```python
   train_cancer_group = np.array([1] * len(d1_coriPreProcess) + [0] * len(d3_coriPreProcess))
   train_mace = np.array(d1_coriPreProcess['Y_mace'].tolist() + d3_coriPreProcess['Y_mace'].tolist(), dtype=float)
   train_Y1 = np.where(train_cancer_group == 1, train_mace, np.nan)   # branch A: MACE | cancer
   train_Y2 = np.where(train_cancer_group == 0, train_mace, np.nan)   # branch B: MACE | non-cancer
   ```
   Mirror for `test_*`. Update the `np.bincount` sanity prints to report per-branch positive counts (via `np.nansum`/masked counts, since the arrays now contain `NaN`) instead of `Task 1(cancer)`/`Task 2(mace)`.

3. **`evaluate_task_performance`** ([disentangle.py.py:547-585](disentangle.py.py#L547-L585)): revert to returning just `(auc1, auc2)` — drop `subgroup_auc` and the `auc1_by_y2`/`auc2_by_y1` computation, since they're now redundant with `auc1`/`auc2` themselves. Remove the now-unused `subgroup_auc` function.

4. **`log_disentanglement_metrics`** ([disentangle.py.py:588-608](disentangle.py.py#L588-L608)): no logic changes needed (the probes already operate correctly against the new masks), just relabel the printed strings, e.g. `E1->Y1` becomes `E1->MACE|cancer`, `P1->Y2(leak)` becomes `P_cancer->MACE|noncancer(leak)`, etc.

5. **`train()`** ([disentangle.py.py:626-699](disentangle.py.py#L626-L699)): remove the "subgroup" print block added last turn (now redundant per point 3); relabel the remaining AUC/loss print strings from `task1`/`task2` to `mace_cancer`/`mace_noncancer` for clarity. `history` dict keys can stay as-is structurally.

6. **`plot_training_history`** ([disentangle.py.py:702-745](disentangle.py.py#L702-L745)): relabel subplot titles ("Task 1 AUC" → "MACE AUC (cancer subgroup)", "Task 2 AUC" → "MACE AUC (non-cancer subgroup)", "Task losses (BCE)" → "MACE BCE loss (cancer / non-cancer branch)").

### Known risk to watch, not to fix preemptively

The cancer+ branch trains on far fewer rows (~2.2k vs ~14k) and `logs.md` already shows it overfitting hard (train AUC 0.91 vs test 0.63) even under the old framing. After this change, watch the new `mace_cancer` test AUC curve in the rerun — if the gap is still large, the fast follow-ups are (in order of effort): fewer epochs / early stopping keyed on that branch's test AUC, then dropout or smaller `private_dim`/hidden width specifically for the small branch. Not doing this now since it's a tuning step that should follow from the reran plot, not be guessed in advance.

## Verification

1. Rerun `disentangle.py.py` in the `hest` conda env (`conda activate hest && python disentangle.py.py`).
2. Confirm it runs to completion without errors and the per-branch positive counts printed at the top look right (cancer branch count ≈ 2.2k, non-cancer branch ≈ 14k, matching the original cohort sizes).
3. Read the new `mace_cancer` / `mace_noncancer` AUC lines and the regenerated `disentangled_training_history.png` — check train/test gap per branch, and whether the two branches' test AUCs actually diverge (the original research question) or stay statistically indistinguishable given the smaller subgroup's noise.
4. Skim the relabeled `log_disentanglement_metrics` line — `E1->MACE|cancer` and `E1->MACE|noncancer` should both stay reasonably high (shared mechanism still predictive in both groups); leakage terms are a secondary check, not the primary readout anymore.
