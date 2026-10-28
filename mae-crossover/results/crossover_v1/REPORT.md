# Results: when does target-domain MAE overtake ImageNet initialisation?

Generated automatically from `runs.csv` (240 fine-tuning runs, seeds [0, 1, 2]). Every number below comes from
the logs; nothing is typed by hand. Test sets were evaluated once per run after restoring the best-validation weights.

## 1. Domain distance from ImageNet (computed before training)
| domain | dataset | fid | otdd | imagenet_confidence | linear_probe_acc | n_train | n_classes |
|---|---|---|---|---|---|---|---|
| plant | plantvillage | 433.328 | 30.621 | 0.298 | 0.899 | 4999 | 38 |
| objects | cifar10 | 266.532 | 28.251 | 0.294 | 0.833 | 5000 | 10 |
| satellite | eurosat | 318.090 | 26.423 | 0.139 | 0.858 | 5000 | 10 |
| animals | pets | 272.932 | 29.301 | 0.510 | 0.861 | 5000 | 37 |

## 2. Primary result: macro-F1 gap, target-MAE (E) minus ImageNet (C), paired by seed
| dataset | budget_pct | gap | ci_lo | ci_hi | n_seeds | p_value |
|---|---|---|---|---|---|---|
| cifar10 | 5 | -0.443 | -0.513 | -0.374 | 3 | 0.001 |
| cifar10 | 10 | -0.380 | -0.403 | -0.357 | 3 | 0.000 |
| cifar10 | 20 | -0.323 | -0.373 | -0.273 | 3 | 0.001 |
| cifar10 | 50 | -0.255 | -0.279 | -0.231 | 3 | 0.000 |
| cifar10 | 100 | -0.196 | -0.241 | -0.152 | 3 | 0.003 |
| eurosat | 5 | -0.183 | -0.198 | -0.169 | 3 | 0.000 |
| eurosat | 10 | -0.155 | -0.191 | -0.119 | 3 | 0.003 |
| eurosat | 20 | -0.111 | -0.141 | -0.082 | 3 | 0.004 |
| eurosat | 50 | -0.066 | -0.087 | -0.045 | 3 | 0.005 |
| eurosat | 100 | -0.049 | -0.059 | -0.039 | 3 | 0.002 |
| pets | 5 | -0.620 | -0.695 | -0.545 | 3 | 0.001 |
| pets | 10 | -0.604 | -0.634 | -0.574 | 3 | 0.000 |
| pets | 20 | -0.548 | -0.572 | -0.525 | 3 | 0.000 |
| pets | 50 | -0.378 | -0.437 | -0.318 | 3 | 0.001 |
| pets | 100 | -0.252 | -0.280 | -0.223 | 3 | 0.001 |
| plantvillage | 5 | -0.239 | -0.316 | -0.163 | 3 | 0.005 |
| plantvillage | 10 | -0.167 | -0.242 | -0.093 | 3 | 0.011 |
| plantvillage | 20 | -0.127 | -0.202 | -0.053 | 3 | 0.018 |
| plantvillage | 50 | -0.054 | -0.082 | -0.026 | 3 | 0.014 |
| plantvillage | 100 | -0.020 | -0.037 | -0.003 | 3 | 0.036 |

### Crossover budget and mean gap per dataset
| dataset | mean_gap_log | crossover_budget | censored | direction | crossover_supported_by_ci |
|---|---|---|---|---|---|
| cifar10 | -0.317 | nan | >100 | C ahead at every budget | False |
| eurosat | -0.110 | nan | >100 | C ahead at every budget | False |
| pets | -0.489 | nan | >100 | C ahead at every budget | False |
| plantvillage | -0.118 | nan | >100 | C ahead at every budget | False |

### Does distance predict the gap? (exploratory: one point per dataset)
| distance | n_datasets | spearman_rho | exact_p | loo_abs_error | loo_baseline_abs_error |
|---|---|---|---|---|---|
| fid | 4 | 0.600 | 0.417 | 0.288 | 0.193 |
| otdd | 4 | -0.400 | 0.750 | 0.350 | 0.193 |
| imagenet_confidence | 4 | -0.800 | 0.333 | 0.124 | 0.193 |
| linear_probe_acc | 4 | 0.000 | 1.000 | 0.205 | 0.193 |

With n datasets the smallest attainable exact p-value is 2/n! (= 0.083 for n = 4); treat this as descriptive.

## 3. Test macro-F1 (mean ± sd over seeds)
| dataset | budget_pct | A_cnn_scratch | C_vit_imagenet | D_vit_scratch | E_vit_mae |
|---|---|---|---|---|---|
| cifar10 | 5 | 0.382 ± 0.012 | 0.868 ± 0.020 | 0.314 ± 0.030 | 0.425 ± 0.017 |
| cifar10 | 10 | 0.452 ± 0.026 | 0.887 ± 0.006 | 0.365 ± 0.032 | 0.506 ± 0.004 |
| cifar10 | 20 | 0.537 ± 0.017 | 0.904 ± 0.008 | 0.419 ± 0.007 | 0.581 ± 0.014 |
| cifar10 | 50 | 0.674 ± 0.004 | 0.926 ± 0.008 | 0.501 ± 0.005 | 0.672 ± 0.005 |
| cifar10 | 100 | 0.746 ± 0.006 | 0.932 ± 0.010 | 0.565 ± 0.011 | 0.736 ± 0.018 |
| eurosat | 5 | 0.764 ± 0.016 | 0.934 ± 0.009 | 0.714 ± 0.032 | 0.751 ± 0.005 |
| eurosat | 10 | 0.813 ± 0.016 | 0.954 ± 0.010 | 0.784 ± 0.004 | 0.799 ± 0.020 |
| eurosat | 20 | 0.851 ± 0.014 | 0.964 ± 0.004 | 0.846 ± 0.011 | 0.853 ± 0.010 |
| eurosat | 50 | 0.913 ± 0.010 | 0.977 ± 0.003 | 0.895 ± 0.006 | 0.910 ± 0.007 |
| eurosat | 100 | 0.949 ± 0.006 | 0.982 ± 0.003 | 0.918 ± 0.007 | 0.933 ± 0.003 |
| pets | 5 | 0.117 ± 0.010 | 0.771 ± 0.008 | 0.073 ± 0.006 | 0.151 ± 0.033 |
| pets | 10 | 0.179 ± 0.013 | 0.815 ± 0.008 | 0.102 ± 0.013 | 0.210 ± 0.020 |
| pets | 20 | 0.283 ± 0.014 | 0.853 ± 0.007 | 0.122 ± 0.007 | 0.305 ± 0.009 |
| pets | 50 | 0.466 ± 0.001 | 0.880 ± 0.006 | 0.196 ± 0.004 | 0.502 ± 0.018 |
| pets | 100 | 0.659 ± 0.015 | 0.890 ± 0.009 | 0.286 ± 0.014 | 0.638 ± 0.017 |
| plantvillage | 5 | 0.548 ± 0.019 | 0.806 ± 0.012 | 0.500 ± 0.038 | 0.567 ± 0.020 |
| plantvillage | 10 | 0.697 ± 0.033 | 0.861 ± 0.039 | 0.607 ± 0.019 | 0.693 ± 0.019 |
| plantvillage | 20 | 0.810 ± 0.016 | 0.940 ± 0.006 | 0.700 ± 0.010 | 0.812 ± 0.024 |
| plantvillage | 50 | 0.930 ± 0.007 | 0.979 ± 0.010 | 0.821 ± 0.008 | 0.925 ± 0.014 |
| plantvillage | 100 | 0.970 ± 0.003 | 0.982 ± 0.006 | 0.911 ± 0.005 | 0.962 ± 0.005 |

## 4. Does the winner replicate across datasets?
| metric | budget_pct | n_datasets | kendall_w | friedman_p | same_winner_everywhere |
|---|---|---|---|---|---|
| test_macro_f1 | 5 | 4 | 0.925 | 0.011 | True |
| test_macro_f1 | 10 | 4 | 0.900 | 0.013 | True |
| test_macro_f1 | 20 | 4 | 1.000 | 0.007 | True |
| test_macro_f1 | 50 | 4 | 0.925 | 0.011 | True |
| test_macro_f1 | 100 | 4 | 1.000 | 0.007 | True |
| test_ece | 5 | 4 | 0.775 | 0.026 | True |
| test_ece | 10 | 4 | 0.900 | 0.013 | True |
| test_ece | 20 | 4 | 0.375 | 0.212 | False |
| test_ece | 50 | 4 | 0.700 | 0.038 | True |
| test_ece | 100 | 4 | 0.550 | 0.086 | False |
| test_ts_ece | 5 | 4 | 0.050 | 0.896 | False |
| test_ts_ece | 10 | 4 | 0.075 | 0.825 | False |
| test_ts_ece | 20 | 4 | 0.525 | 0.098 | False |
| test_ts_ece | 50 | 4 | 0.450 | 0.145 | False |
| test_ts_ece | 100 | 4 | 0.225 | 0.440 | False |

## Figures
`plots/learning_curves.png`, `plots/gap_E_minus_C.png`, `plots/distance_vs_gap.png`, `plots/winner_map.png`,
`plots/ece_vs_budget.png`, `plots/ece_after_temperature_vs_budget.png`, `plots/nll_vs_budget.png`;
MAE reconstructions in `checkpoints/<experiment>/<dataset>/mae_seed*/reconstructions.png`.
