# Metrics and Prediction Format

## Prediction format

Predictions are JSONL rows keyed by manifest `id`:

```json
{"id":"example-id","predicted_action":"respond","predicted_answer":"short answer"}
```

Allowed actions: `respond`, `ignore`, `wait`, `clarify`, `incorporate`.

For `ignore` and `wait`, `predicted_answer` should be empty or omitted. If a model emits substantive answer text while predicting a non-engagement action, downstream evaluators may count this as a policy violation.

## Primary metrics

### Engagement Accuracy

Accuracy over expected assistant action:

```text
engagement_accuracy = correct predicted_action / total examples
```

### False Engagement Rate

Rate at which the model responds when the expected action is `ignore` or `wait`:

```text
false_engagement_rate = false responds / examples expected to ignore or wait
```

This is the critical metric for side conversations and delivery-person-style interruptions.

### Missed Engagement Rate

Rate at which the model fails to respond when the expected action is `respond` or `incorporate`:

```text
missed_engagement_rate = non-response predictions / examples expected to respond or incorporate
```

### Answer Exact Match

Simple normalized exact-match over examples where the expected action requires an answer and a reference answer exists. This is intentionally minimal; future versions can add semantic equivalence judges.

### Robustness Drop

Compare a clean/control subset to perturbed variants sharing the same `scenario.base_id`:

```text
robustness_drop = clean_score - perturbed_score
```

The seed manifest may not always include paired clean/perturbed examples; newly constructed data should include `scenario.base_id` for this analysis.

## Required report slices

Report metrics by:

- expected action;
- interaction class;
- scenario domain;
- perturbation family;
- source dataset;
- noise level / environment where available.
