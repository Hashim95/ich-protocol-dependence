**Supplementary Table S3. Threshold-selection criteria compared, pooled out-of-fold.**

| Class   |   n pos |   Youden thr |   Youden prec |   Youden F1 |   F2 thr |   F2 F1 |   Fixed 0.5 prec |   Fixed 0.5 F1 |
|:--------|--------:|-------------:|--------------:|------------:|---------:|--------:|-----------------:|---------------:|
| EDH     |     284 |      1e-05   |         0.102 |       0.177 |  0.0001  |   0.23  |            0.336 |          0.191 |
| IPH     |    4272 |      0.028   |         0.792 |       0.845 |  0.0061  |   0.832 |            0.86  |          0.862 |
| IVH     |    2936 |      0.0086  |         0.771 |       0.846 |  0.0086  |   0.848 |            0.858 |          0.875 |
| SAH     |    3134 |      0.0013  |         0.553 |       0.674 |  0.00037 |   0.653 |            0.747 |          0.727 |
| SDH     |    3027 |      0.00059 |         0.594 |       0.708 |  0.00059 |   0.703 |            0.765 |          0.745 |
| Any     |    7106 |      0.67    |         0.911 |       0.913 |  0.0024  |   0.895 |            0.911 |          0.913 |

*At epidural prevalence, maximising Youden's J drives the threshold to the search-grid floor and yields an F1 below that of an untuned 0.5 threshold; maximising F2 fails similarly. All main-text results use a fixed 0.5 threshold.*
