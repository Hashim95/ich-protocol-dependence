**The published base method compared with the same architecture re-implemented and trained under patient-disjoint evaluation.**

| Aspect               | Original (as published)           | Re-implementation (this work)   |
|:---------------------|:----------------------------------|:--------------------------------|
| Data partitioning    | Not described as patient-disjoint | Patient-disjoint, 5-fold        |
| Evaluation level     | Slice, fixed threshold 0.5        | Slice, fixed threshold 0.5      |
| Confidence intervals | Not reported                      | Bootstrap 95%                   |
| External validation  | None                              | Two independent cohorts         |
| EDH F1               | 0.467                             | 0.166 (folds 0.105–0.229)       |
| Average F1           | 0.746                             | 0.664 ± 0.005                   |
| EDH AUC              | 0.998                             | 0.905 ± 0.025                   |
| EDH AP               | Not reported                      | 0.153 ± 0.054                   |

*Original values are as published: average F1 and epidural F1 from the original's class-wise F1 table, epidural AUC from its class-wise AUC table. Re-implementation values are pre-smoothing; the post-smoothing comparison is given in the Discussion. Per-fold EDH F1 varied by a factor of 2.2 (0.105–0.229) while average F1 remained stable at 0.664 ± 0.005, indicating that single-partition rare-class estimates carry variance far larger than aggregate metrics. The re-implementation substitutes standard acute-hemorrhage intensity thresholds for the original's unpublished values and uses a matched backbone.*
