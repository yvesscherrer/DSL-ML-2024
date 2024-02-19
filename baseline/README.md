# Baselines

The `classify_svm.py` script provides a baseline system. It uses tf-idf-weighted character-level and word-level n-gram features in a linear SVM classifier. The classifier follows a multi-class (but not multi-label!) setup with the following two options to transform the multi-label annotations:
- `atomic`: Label combinations are added as distinct atomic labels. For example, the English task would have three distinct labels `EN-GB`, `EN-US` and `EN-GB,EN-US`. This corresponds to the DSL-TL setup where the `EN` label is renamed `EN-GB,EN-US`. This setup produces multi-label annotations.
- `expand`: Training instances with several labels are duplicated, once with each label. For example, an instance labeled with `EN-GB,EN-US` would appear twice, once with the label `EN-GB` and once with the label `EN-US`. This setup produces only single-label annotations.

The baseline script is called as follows:

```
python3 classify_svm.py <TASK> <MODE>
```

where `<TASK>` is one of `BCMS, EN, ES, FR, PT` and `<MODE>` is one of `atomic, expand`. Note that both modes produce identical results for BCMS since the training data is single-labeled.

# Evaluation

The `evaluate.py` script is the official scoring script. It provides per-class F1-scores, weighted and macro-averaged F1-scores. It also provides the same scores for the subset of the evaluation set with multiple labels. Usage:

```
python3 evaluate.py goldfile systemfile
```
# Baseline results

| Task | Mode | Macro F1 | Weighted F1 | Amb. Macro F1 | Amb. Weighted F1 |
| ---- | ---- | :------: | :---------: | :-----------: | :--------------: |
| BCMS | atomic | 66.93%   | 79.64%      | 48.44%        | 51.18%         |
| EN   | atomic | 76.51%   | 77.32%      | 72.43%        | 72.43%         |
|      | expand | 74.80%   | 75.39%      | 66.57%        | 66.57%         |
| ES   | atomic | 77.12%   | 78.18%      | 82.27%        | 82.27%         |
|      | expand | 69.66%   | 70.68%      | 66.65%        | 66.65%         |
| FR   | atomic | 63.78%   | 64.41%      | 45.84%        | 68.27%         |
|      | expand | 63.33%   | 64.04%      | 30.82%        | 53.83%         |
| PT   | atomic | 67.55%   | 71.05%      | 68.60%        | 68.60%         |
|      | expand | 66.24%   | 70.00%      | 66.50%        | 66.50%         |
