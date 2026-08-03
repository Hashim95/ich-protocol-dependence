# Paper 1 — Tables

## Table 1

**Internal performance, pooled out-of-fold across development studies (patient-disjoint five-fold cross-validation).**

| Class   |   n pos | AUC [95% CI]         | AP [95% CI]          |   Brier |   Sens@0.5 |   Spec@0.5 |   Prec@0.5 |   F1@0.5 |
|:--------|--------:|:---------------------|:---------------------|--------:|-----------:|-----------:|-----------:|---------:|
| EDH     |     284 | 0.888 [0.868, 0.908] | 0.206 [0.164, 0.257] |   0.017 |      0.134 |      0.996 |      0.336 |    0.191 |
| IPH     |    4272 | 0.969 [0.966, 0.972] | 0.927 [0.920, 0.934] |   0.061 |      0.863 |      0.954 |      0.86  |    0.862 |
| IVH     |    2936 | 0.984 [0.982, 0.986] | 0.939 [0.931, 0.945] |   0.039 |      0.891 |      0.97  |      0.858 |    0.875 |
| SAH     |    3134 | 0.937 [0.932, 0.941] | 0.819 [0.808, 0.831] |   0.085 |      0.708 |      0.947 |      0.747 |    0.727 |
| SDH     |    3027 | 0.944 [0.939, 0.949] | 0.818 [0.805, 0.831] |   0.078 |      0.726 |      0.953 |      0.765 |    0.745 |
| Any     |    7106 | 0.971 [0.969, 0.974] | 0.964 [0.961, 0.967] |   0.065 |      0.915 |      0.938 |      0.911 |    0.913 |
| MACRO   |   20759 | 0.949 [0.945, 0.952] | 0.779                |   0.058 |      0.706 |      0.96  |      0.746 |    0.719 |

*Point estimates are plug-in values; intervals are percentile bootstrap (1,000 resamples). Macro-AUC intervals resample studies once per draw and average all class AUCs within that draw. Threshold-dependent metrics use a fixed 0.5 threshold.*

---

## Table 2

**Rare-subtype precision across architecture families under identical training conditions (final-epoch models, five folds each).**

| Backbone        |   Folds | Macro-AUC       | EDH AUC         | EDH AP          |   EDH AP (best epoch) |
|:----------------|--------:|:----------------|:----------------|:----------------|----------------------:|
| ConvNeXt-tiny   |       5 | 0.9442 ± 0.0016 | 0.8609 ± 0.0127 | 0.1287 ± 0.0300 |                0.1497 |
| ResNet50        |       5 | 0.9587 ± 0.0031 | 0.9044 ± 0.0181 | 0.1332 ± 0.0324 |                0.1447 |
| EfficientNet-B4 |       5 | 0.9295 ± 0.0033 | 0.8057 ± 0.0246 | 0.1162 ± 0.0239 |                0.1296 |
| DenseNet-161    |       1 | 0.9545          | 0.9149          | 0.1127          |                0.1649 |

*All architectures share identical folds, preprocessing, loss, sampler, schedule, and learning rates; hyperparameters were selected for ConvNeXt-tiny and not re-tuned, so comparator performance is a lower bound. Between-fold variance in EDH AP (6.82e-04) exceeds between-architecture variance (7.69e-05) by 8.9x. All architectures performed worst on fold 4.*

---

## Table 3

**The published base method compared with the same architecture re-implemented and trained under patient-disjoint evaluation.**

| Aspect               | Original (as published)           | Re-implementation (this work)   |
|:---------------------|:----------------------------------|:--------------------------------|
| Data partitioning    | Not described as patient-disjoint | Patient-disjoint, 5-fold        |
| Evaluation level     | Slice, fixed threshold 0.5        | Slice, fixed threshold 0.5      |
| Confidence intervals | Not reported                      | Bootstrap 95%                   |
| External validation  | None                              | Two independent cohorts         |
| EDH F1               | 0.467                             | 0.166 (folds 0.105–0.229)       |
| Average F1           | 0.746                             | 0.664 ± 0.005                   |
| EDH AUC              | Not reported per class            | 0.905 ± 0.025                   |
| EDH AP               | Not reported                      | 0.153 ± 0.054                   |

*Per-fold EDH F1 varied by a factor of 2.2 (0.105–0.229) while average F1 remained stable at 0.664 ± 0.005, indicating that single-partition rare-class estimates carry variance far larger than aggregate metrics. Original values are as published; the re-implementation substitutes standard acute-hemorrhage intensity thresholds for the original's unpublished values and uses a matched backbone.*

---

## Table 4

**External validation on two independent cohorts, neither used in development.**

| Cohort              | Class   |   n pos | AUC [95% CI]         |    AP |   F1@0.5 |
|:--------------------|:--------|--------:|:---------------------|------:|---------:|
| CQ500 (scan level)  | EDH     |      12 | 0.805 [0.613, 0.978] | 0.648 |    0.5   |
| CQ500 (scan level)  | IPH     |     129 | 0.805 [0.747, 0.856] | 0.765 |    0.695 |
| CQ500 (scan level)  | IVH     |      26 | 0.873 [0.758, 0.968] | 0.727 |    0.689 |
| CQ500 (scan level)  | SAH     |      57 | 0.867 [0.790, 0.932] | 0.731 |    0.661 |
| CQ500 (scan level)  | SDH     |      49 | 0.775 [0.677, 0.871] | 0.593 |    0.535 |
| CQ500 (scan level)  | Any     |     197 | 0.807 [0.761, 0.853] | 0.845 |    0.776 |
| BHSD (volume level) | EDH     |      23 | 0.831 [0.738, 0.913] | 0.56  |    0.16  |
| BHSD (volume level) | IPH     |     127 | 0.917 [0.870, 0.957] | 0.951 |    0.901 |
| BHSD (volume level) | IVH     |     104 | 0.884 [0.835, 0.927] | 0.911 |    0.747 |
| BHSD (volume level) | SAH     |     109 | 0.786 [0.716, 0.844] | 0.855 |    0.749 |
| BHSD (volume level) | SDH     |      70 | 0.821 [0.757, 0.876] | 0.765 |    0.659 |
| BHSD (volume level) | MACRO   |     433 | 0.848 [0.783, 0.903] | 0.808 |    0.643 |

*Average precision is not comparable across cohorts: EDH prevalence differs by an order of magnitude (1.6% internally, 2.6% in CQ500, 12.0% in BHSD). BHSD contains no hemorrhage-negative volumes and therefore evaluates subtype discrimination among positive scans rather than detection. EDH positive counts are small in both cohorts; intervals are reported throughout.*

---

## Table 5

**Operating characteristics at 95% sensitivity, pooled out-of-fold.**

| Class   |   n pos | Specificity @95% sens   |   PPV |   Alerts per 100 studies |   Reviews per true positive |
|:--------|--------:|:------------------------|------:|-------------------------:|----------------------------:|
| EDH     |     284 | 0.572 [0.409, 0.663]    | 0.036 |                     43.6 |                        28.1 |
| IPH     |    4272 | 0.846 [0.821, 0.866]    | 0.668 |                     34.9 |                         1.5 |
| IVH     |    2936 | 0.931 [0.912, 0.943]    | 0.737 |                     21.8 |                         1.4 |
| SAH     |    3134 | 0.711 [0.690, 0.731]    | 0.419 |                     40.8 |                         2.4 |
| SDH     |    3027 | 0.736 [0.701, 0.765]    | 0.432 |                     38.3 |                         2.3 |
| Any     |    7106 | 0.881 [0.866, 0.897]    | 0.847 |                     45.8 |                         1.2 |

*Reviews per true positive is the reciprocal of positive predictive value. Decision curve analysis showed positive net benefit over both flag-all and flag-none strategies for every subtype.*
