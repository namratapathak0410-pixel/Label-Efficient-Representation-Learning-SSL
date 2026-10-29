# Results of experiment 2: bigger MAE pool, ImageNet + MAE, ResNet-34

Generated automatically from `runs.csv` (405 fine-tuning runs, 3 seeds, arms: A, A2, C, D, E, E2, F).
Pre-registered in `PREREGISTRATION_V2.md` before any of this ran. E2 = MAE on a 20,000-image pool (not Pets), F =
ImageNet ViT + continued MAE on the pool, A2 = ResNet-34. Arms A, C, D, E use the v1 runs wherever the learning rate
was unchanged by the wider grid.

## Hypotheses (H6-H11)
| id | hypothesis | result | verdict |
|---|---|---|---|
| H6 | E2 > E (budget- and seed-averaged) in each of PlantVillage, CIFAR-10, EuroSAT | cifar10 -0.007 [-0.026, +0.011]; eurosat +0.007 [-0.007, +0.022]; plantvillage +0.006 [-0.018, +0.031]; E2>E in 8/15 cells | not supported |
| H7 | E2-C > E-C at every budget; E2 overtakes C (CI excludes 0) somewhere | E2-C > E-C in 8/15 cells; cells with E2 significantly above C: none; best E2-C gap -0.014 (plantvillage, 100%) | not supported |
| H8 | F > C in more than half of the 20 cells; gain larger in farther domains | F>C in 3/20 cells (0 with CI above 0, 10 with CI below 0); Spearman of mean F-C gap with distance: fid +0.60 (p=0.42), otdd -0.40 (p=0.75), imagenet_confidence -0.80 (p=0.33), linear_probe_acc +0.00 (p=1.00) | not supported |
| H9 | F >= E2 (budget- and seed-averaged) in the three pooled datasets | cifar10 +0.158 [+0.083, +0.233]; eurosat +0.090 [+0.077, +0.103]; plantvillage +0.082 [+0.076, +0.089] | supported |
| H10 | A2-A within +-0.03 in most cells; MAE-vs-ResNet verdict unchanged | |A2-A| < 0.03 in 18/20 cells (range -0.015 to +0.042); best MAE arm beats ResNet-18 in 14/20 cells, ResNet-34 in 13/20; verdict flips in 1 cells | partly supported |
| H11 | Wider lr grid does not change the winner among A, C, D, E (v1: C won 20/20) | C wins 20/20 cells | supported |

## Learning-rate grid
The wider grid changed the chosen learning rate in 9 of 16 v1 (dataset, arm) cells; those cells were re-run.

| dataset | arm | lr_v1 | lr_v2 | changed |
|---|---|---|---|---|
| plantvillage | A_cnn_scratch | 0.002 | 0.004 | True |
| plantvillage | C_vit_imagenet | 0.001 | 0.002 | True |
| plantvillage | D_vit_scratch | 0.002 | 0.002 | False |
| plantvillage | E_vit_mae | 0.001 | 0.002 | True |
| cifar10 | A_cnn_scratch | 0.002 | 0.002 | False |
| cifar10 | C_vit_imagenet | 0.000 | 0.000 | True |
| cifar10 | D_vit_scratch | 0.001 | 0.000 | True |
| cifar10 | E_vit_mae | 0.001 | 0.001 | False |
| eurosat | A_cnn_scratch | 0.001 | 0.004 | True |
| eurosat | C_vit_imagenet | 0.001 | 0.002 | True |
| eurosat | D_vit_scratch | 0.001 | 0.001 | False |
| eurosat | E_vit_mae | 0.001 | 0.001 | False |
| pets | A_cnn_scratch | 0.002 | 0.002 | False |
| pets | C_vit_imagenet | 0.001 | 0.001 | False |
| pets | D_vit_scratch | 0.001 | 0.000 | True |
| pets | E_vit_mae | 0.001 | 0.002 | True |

## Mean test macro-F1 (± sd over seeds)
| dataset | budget_pct | A | A2 | C | D | E | E2 | F |
|---|---|---|---|---|---|---|---|---|
| cifar10 | 5 | 0.382 ± 0.012 | 0.384 ± 0.017 | 0.865 ± 0.014 | 0.334 ± 0.018 | 0.425 ± 0.017 | 0.433 ± 0.015 | 0.515 ± 0.053 |
| cifar10 | 10 | 0.452 ± 0.026 | 0.456 ± 0.018 | 0.888 ± 0.010 | 0.378 ± 0.023 | 0.506 ± 0.004 | 0.489 ± 0.012 | 0.639 ± 0.034 |
| cifar10 | 20 | 0.537 ± 0.017 | 0.544 ± 0.014 | 0.911 ± 0.012 | 0.436 ± 0.021 | 0.581 ± 0.014 | 0.577 ± 0.012 | 0.756 ± 0.024 |
| cifar10 | 50 | 0.674 ± 0.004 | 0.672 ± 0.014 | 0.928 ± 0.005 | 0.513 ± 0.011 | 0.672 ± 0.005 | 0.652 ± 0.004 | 0.862 ± 0.005 |
| cifar10 | 100 | 0.746 ± 0.006 | 0.788 ± 0.005 | 0.938 ± 0.010 | 0.586 ± 0.014 | 0.736 ± 0.018 | 0.730 ± 0.003 | 0.900 ± 0.011 |
| eurosat | 5 | 0.738 ± 0.028 | 0.740 ± 0.038 | 0.930 ± 0.004 | 0.714 ± 0.032 | 0.751 ± 0.005 | 0.760 ± 0.007 | 0.895 ± 0.007 |
| eurosat | 10 | 0.800 ± 0.012 | 0.792 ± 0.028 | 0.949 ± 0.003 | 0.784 ± 0.004 | 0.799 ± 0.020 | 0.814 ± 0.003 | 0.932 ± 0.011 |
| eurosat | 20 | 0.854 ± 0.012 | 0.850 ± 0.016 | 0.962 ± 0.010 | 0.846 ± 0.011 | 0.853 ± 0.010 | 0.865 ± 0.000 | 0.954 ± 0.009 |
| eurosat | 50 | 0.916 ± 0.004 | 0.919 ± 0.006 | 0.970 ± 0.001 | 0.895 ± 0.006 | 0.910 ± 0.007 | 0.906 ± 0.011 | 0.973 ± 0.005 |
| eurosat | 100 | 0.955 ± 0.001 | 0.957 ± 0.000 | 0.978 ± 0.007 | 0.918 ± 0.007 | 0.933 ± 0.003 | 0.938 ± 0.005 | 0.978 ± 0.003 |
| pets | 5 | 0.117 ± 0.010 | 0.120 ± 0.003 | 0.771 ± 0.008 | 0.078 ± 0.007 | 0.142 ± 0.017 | nan | 0.229 ± 0.013 |
| pets | 10 | 0.179 ± 0.013 | 0.168 ± 0.041 | 0.815 ± 0.008 | 0.105 ± 0.014 | 0.214 ± 0.017 | nan | 0.373 ± 0.024 |
| pets | 20 | 0.283 ± 0.014 | 0.278 ± 0.024 | 0.853 ± 0.007 | 0.137 ± 0.007 | 0.308 ± 0.008 | nan | 0.596 ± 0.025 |
| pets | 50 | 0.466 ± 0.001 | 0.482 ± 0.017 | 0.880 ± 0.006 | 0.258 ± 0.018 | 0.493 ± 0.021 | nan | 0.828 ± 0.007 |
| pets | 100 | 0.659 ± 0.015 | 0.693 ± 0.013 | 0.890 ± 0.009 | 0.415 ± 0.003 | 0.623 ± 0.021 | nan | 0.870 ± 0.010 |
| plantvillage | 5 | 0.544 ± 0.027 | 0.529 ± 0.012 | 0.796 ± 0.044 | 0.500 ± 0.038 | 0.547 ± 0.031 | 0.560 ± 0.020 | 0.700 ± 0.012 |
| plantvillage | 10 | 0.678 ± 0.049 | 0.685 ± 0.014 | 0.847 ± 0.017 | 0.607 ± 0.019 | 0.686 ± 0.023 | 0.694 ± 0.023 | 0.805 ± 0.042 |
| plantvillage | 20 | 0.808 ± 0.011 | 0.814 ± 0.013 | 0.935 ± 0.012 | 0.700 ± 0.010 | 0.819 ± 0.030 | 0.814 ± 0.037 | 0.912 ± 0.009 |
| plantvillage | 50 | 0.932 ± 0.005 | 0.934 ± 0.009 | 0.975 ± 0.006 | 0.821 ± 0.008 | 0.909 ± 0.012 | 0.924 ± 0.009 | 0.965 ± 0.013 |
| plantvillage | 100 | 0.961 ± 0.015 | 0.970 ± 0.004 | 0.974 ± 0.008 | 0.911 ± 0.005 | 0.961 ± 0.003 | 0.959 ± 0.002 | 0.980 ± 0.005 |

## Calibration and mean F1 by arm (mean over all cells)
| arm | ece | ece_after_ts | nll | macro_f1 |
|---|---|---|---|---|
| A_cnn_scratch | 0.179 | 0.032 | 1.679 | 0.634 |
| A2_cnn34_scratch | 0.176 | 0.030 | 1.634 | 0.639 |
| C_vit_imagenet | 0.052 | 0.024 | 0.373 | 0.903 |
| D_vit_scratch | 0.121 | 0.033 | 1.589 | 0.547 |
| E_vit_mae | 0.151 | 0.032 | 1.449 | 0.643 |
| E2_vit_mae_big | 0.116 | 0.033 | 0.906 | 0.741 |
| F_vit_imagenet_mae | 0.146 | 0.031 | 1.185 | 0.783 |

## Crossover and mean gap to the ImageNet ViT
| pair | dataset | mean_gap_log | crossover_budget | censored | direction | crossover_supported_by_ci |
|---|---|---|---|---|---|---|
| E-C | cifar10 | -0.320 | nan | >100 | C ahead at every budget | False |
| E-C | eurosat | -0.106 | nan | >100 | C ahead at every budget | False |
| E-C | pets | -0.493 | nan | >100 | C ahead at every budget | False |
| E-C | plantvillage | -0.116 | nan | >100 | C ahead at every budget | False |
| E2-C | cifar10 | -0.330 | nan | >100 | C ahead at every budget | False |
| E2-C | eurosat | -0.099 | nan | >100 | C ahead at every budget | False |
| E2-C | plantvillage | -0.110 | nan | >100 | C ahead at every budget | False |
| F-C | cifar10 | -0.162 | nan | >100 | C ahead at every budget | False |
| F-C | eurosat | -0.009 | 38.867 |  | E overtakes C | False |
| F-C | pets | -0.250 | nan | >100 | C ahead at every budget | False |
| F-C | plantvillage | -0.029 | 76.585 |  | E overtakes C | False |

### Distance association of the mean gap (descriptive; n = 3 or 4 datasets)
| pair | distance | n_datasets | spearman_rho | exact_p | loo_abs_error | loo_baseline_abs_error |
|---|---|---|---|---|---|---|
| E-C | fid | 4 | 0.600 | 0.417 | 0.295 | 0.197 |
| E-C | otdd | 4 | -0.400 | 0.750 | 0.358 | 0.197 |
| E-C | imagenet_confidence | 4 | -0.800 | 0.333 | 0.123 | 0.197 |
| E-C | linear_probe_acc | 4 | 0.000 | 1.000 | 0.206 | 0.197 |
| E2-C | fid | 3 | 0.500 | 1.000 | 0.309 | 0.150 |
| E2-C | otdd | 3 | -0.500 | 1.000 | 0.382 | 0.150 |
| E2-C | imagenet_confidence | 3 | -0.500 | 1.000 | 3.060 | 0.150 |
| E2-C | linear_probe_acc | 3 | 0.500 | 1.000 | 0.259 | 0.150 |
| F-C | fid | 4 | 0.600 | 0.417 | 0.189 | 0.125 |
| F-C | otdd | 4 | -0.400 | 0.750 | 0.227 | 0.125 |
| F-C | imagenet_confidence | 4 | -0.800 | 0.333 | 0.062 | 0.125 |
| F-C | linear_probe_acc | 4 | 0.000 | 1.000 | 0.110 | 0.125 |

## Winners and ranking agreement
| metric | budget_pct | n_datasets | n_arms | kendall_w | friedman_p |
|---|---|---|---|---|---|
| test_macro_f1 | 5 | 4 | 6 | 0.979 | 0.002 |
| test_macro_f1 | 10 | 4 | 6 | 0.936 | 0.002 |
| test_macro_f1 | 20 | 4 | 6 | 0.936 | 0.002 |
| test_macro_f1 | 50 | 4 | 6 | 0.893 | 0.003 |
| test_macro_f1 | 100 | 4 | 6 | 0.950 | 0.002 |
| test_ece | 5 | 4 | 6 | 0.464 | 0.098 |
| test_ece | 10 | 4 | 6 | 0.521 | 0.064 |
| test_ece | 20 | 4 | 6 | 0.757 | 0.010 |
| test_ece | 50 | 4 | 6 | 0.750 | 0.010 |
| test_ece | 100 | 4 | 6 | 0.714 | 0.014 |
| test_ts_ece | 5 | 4 | 6 | 0.164 | 0.656 |
| test_ts_ece | 10 | 4 | 6 | 0.143 | 0.722 |
| test_ts_ece | 20 | 4 | 6 | 0.300 | 0.306 |
| test_ts_ece | 50 | 4 | 6 | 0.764 | 0.009 |
| test_ts_ece | 100 | 4 | 6 | 0.543 | 0.054 |

## All paired gaps (macro-F1, paired by seed)

### E2 minus E  (ViT MAE (20k pool) vs ViT target-MAE)
| dataset | budget_pct | gap | ci_lo | ci_hi | n_seeds | p_value |
|---|---|---|---|---|---|---|
| cifar10 | 5 | 0.009 | -0.002 | 0.019 | 3 | 0.073 |
| cifar10 | 10 | -0.017 | -0.041 | 0.007 | 3 | 0.093 |
| cifar10 | 20 | -0.004 | -0.030 | 0.022 | 3 | 0.570 |
| cifar10 | 50 | -0.019 | -0.042 | 0.003 | 3 | 0.065 |
| cifar10 | 100 | -0.005 | -0.056 | 0.046 | 3 | 0.700 |
| eurosat | 5 | 0.009 | -0.003 | 0.021 | 3 | 0.083 |
| eurosat | 10 | 0.015 | -0.026 | 0.056 | 3 | 0.256 |
| eurosat | 20 | 0.013 | -0.012 | 0.037 | 3 | 0.162 |
| eurosat | 50 | -0.005 | -0.036 | 0.026 | 3 | 0.566 |
| eurosat | 100 | 0.004 | -0.003 | 0.012 | 3 | 0.128 |
| plantvillage | 5 | 0.013 | -0.033 | 0.060 | 3 | 0.343 |
| plantvillage | 10 | 0.008 | -0.039 | 0.056 | 3 | 0.534 |
| plantvillage | 20 | -0.005 | -0.023 | 0.013 | 3 | 0.346 |
| plantvillage | 50 | 0.016 | -0.024 | 0.055 | 3 | 0.231 |
| plantvillage | 100 | -0.002 | -0.012 | 0.008 | 3 | 0.496 |

### E2 minus C  (ViT MAE (20k pool) vs ViT ImageNet-1k)
| dataset | budget_pct | gap | ci_lo | ci_hi | n_seeds | p_value |
|---|---|---|---|---|---|---|
| cifar10 | 5 | -0.432 | -0.501 | -0.363 | 3 | 0.001 |
| cifar10 | 10 | -0.399 | -0.450 | -0.347 | 3 | 0.001 |
| cifar10 | 20 | -0.335 | -0.379 | -0.290 | 3 | 0.001 |
| cifar10 | 50 | -0.275 | -0.297 | -0.253 | 3 | 0.000 |
| cifar10 | 100 | -0.208 | -0.226 | -0.190 | 3 | 0.000 |
| eurosat | 5 | -0.170 | -0.189 | -0.151 | 3 | 0.001 |
| eurosat | 10 | -0.135 | -0.150 | -0.121 | 3 | 0.001 |
| eurosat | 20 | -0.097 | -0.123 | -0.071 | 3 | 0.004 |
| eurosat | 50 | -0.065 | -0.096 | -0.033 | 3 | 0.012 |
| eurosat | 100 | -0.040 | -0.046 | -0.034 | 3 | 0.001 |
| plantvillage | 5 | -0.236 | -0.314 | -0.159 | 3 | 0.006 |
| plantvillage | 10 | -0.153 | -0.252 | -0.054 | 3 | 0.022 |
| plantvillage | 20 | -0.121 | -0.184 | -0.058 | 3 | 0.014 |
| plantvillage | 50 | -0.050 | -0.073 | -0.027 | 3 | 0.011 |
| plantvillage | 100 | -0.014 | -0.036 | 0.007 | 3 | 0.101 |

### F minus C  (ViT ImageNet+MAE vs ViT ImageNet-1k)
| dataset | budget_pct | gap | ci_lo | ci_hi | n_seeds | p_value |
|---|---|---|---|---|---|---|
| cifar10 | 5 | -0.351 | -0.463 | -0.239 | 3 | 0.005 |
| cifar10 | 10 | -0.249 | -0.323 | -0.175 | 3 | 0.005 |
| cifar10 | 20 | -0.155 | -0.190 | -0.121 | 3 | 0.003 |
| cifar10 | 50 | -0.065 | -0.077 | -0.054 | 3 | 0.002 |
| cifar10 | 100 | -0.039 | -0.087 | 0.009 | 3 | 0.074 |
| eurosat | 5 | -0.034 | -0.052 | -0.016 | 3 | 0.015 |
| eurosat | 10 | -0.017 | -0.051 | 0.018 | 3 | 0.174 |
| eurosat | 20 | -0.008 | -0.047 | 0.030 | 3 | 0.446 |
| eurosat | 50 | 0.003 | -0.011 | 0.017 | 3 | 0.434 |
| eurosat | 100 | 0.000 | -0.023 | 0.024 | 3 | 0.981 |
| pets | 5 | -0.542 | -0.559 | -0.524 | 3 | 0.000 |
| pets | 10 | -0.441 | -0.512 | -0.371 | 3 | 0.001 |
| pets | 20 | -0.257 | -0.335 | -0.178 | 3 | 0.005 |
| pets | 50 | -0.052 | -0.055 | -0.048 | 3 | 0.000 |
| pets | 100 | -0.020 | -0.049 | 0.009 | 3 | 0.099 |
| plantvillage | 5 | -0.096 | -0.219 | 0.027 | 3 | 0.078 |
| plantvillage | 10 | -0.042 | -0.181 | 0.097 | 3 | 0.326 |
| plantvillage | 20 | -0.023 | -0.037 | -0.009 | 3 | 0.019 |
| plantvillage | 50 | -0.010 | -0.044 | 0.024 | 3 | 0.328 |
| plantvillage | 100 | 0.006 | -0.018 | 0.031 | 3 | 0.378 |

### F minus E2  (ViT ImageNet+MAE vs ViT MAE (20k pool))
| dataset | budget_pct | gap | ci_lo | ci_hi | n_seeds | p_value |
|---|---|---|---|---|---|---|
| cifar10 | 5 | 0.081 | -0.057 | 0.220 | 3 | 0.127 |
| cifar10 | 10 | 0.149 | 0.035 | 0.264 | 3 | 0.030 |
| cifar10 | 20 | 0.179 | 0.100 | 0.258 | 3 | 0.010 |
| cifar10 | 50 | 0.210 | 0.189 | 0.231 | 3 | 0.001 |
| cifar10 | 100 | 0.169 | 0.135 | 0.204 | 3 | 0.002 |
| eurosat | 5 | 0.136 | 0.102 | 0.169 | 3 | 0.003 |
| eurosat | 10 | 0.119 | 0.094 | 0.143 | 3 | 0.002 |
| eurosat | 20 | 0.088 | 0.067 | 0.110 | 3 | 0.003 |
| eurosat | 50 | 0.068 | 0.050 | 0.086 | 3 | 0.004 |
| eurosat | 100 | 0.041 | 0.021 | 0.060 | 3 | 0.012 |
| plantvillage | 5 | 0.140 | 0.064 | 0.217 | 3 | 0.016 |
| plantvillage | 10 | 0.111 | 0.054 | 0.168 | 3 | 0.014 |
| plantvillage | 20 | 0.098 | 0.024 | 0.173 | 3 | 0.030 |
| plantvillage | 50 | 0.040 | 0.029 | 0.051 | 3 | 0.004 |
| plantvillage | 100 | 0.021 | 0.013 | 0.028 | 3 | 0.007 |

### F minus E  (ViT ImageNet+MAE vs ViT target-MAE)
| dataset | budget_pct | gap | ci_lo | ci_hi | n_seeds | p_value |
|---|---|---|---|---|---|---|
| cifar10 | 5 | 0.090 | -0.039 | 0.219 | 3 | 0.096 |
| cifar10 | 10 | 0.132 | 0.041 | 0.223 | 3 | 0.025 |
| cifar10 | 20 | 0.175 | 0.082 | 0.269 | 3 | 0.015 |
| cifar10 | 50 | 0.190 | 0.189 | 0.192 | 3 | 0.000 |
| cifar10 | 100 | 0.164 | 0.135 | 0.193 | 3 | 0.002 |
| eurosat | 5 | 0.145 | 0.117 | 0.172 | 3 | 0.002 |
| eurosat | 10 | 0.134 | 0.089 | 0.178 | 3 | 0.006 |
| eurosat | 20 | 0.101 | 0.086 | 0.116 | 3 | 0.001 |
| eurosat | 50 | 0.063 | 0.047 | 0.079 | 3 | 0.003 |
| eurosat | 100 | 0.045 | 0.030 | 0.060 | 3 | 0.006 |
| pets | 5 | 0.087 | 0.027 | 0.146 | 3 | 0.025 |
| pets | 10 | 0.159 | 0.059 | 0.259 | 3 | 0.021 |
| pets | 20 | 0.289 | 0.241 | 0.337 | 3 | 0.001 |
| pets | 50 | 0.336 | 0.265 | 0.407 | 3 | 0.002 |
| pets | 100 | 0.248 | 0.195 | 0.300 | 3 | 0.002 |
| plantvillage | 5 | 0.154 | 0.046 | 0.261 | 3 | 0.026 |
| plantvillage | 10 | 0.119 | 0.015 | 0.223 | 3 | 0.039 |
| plantvillage | 20 | 0.093 | 0.036 | 0.150 | 3 | 0.020 |
| plantvillage | 50 | 0.056 | 0.006 | 0.106 | 3 | 0.041 |
| plantvillage | 100 | 0.019 | 0.002 | 0.035 | 3 | 0.039 |

### A2 minus A  (ResNet-34 scratch vs ResNet-18 scratch)
| dataset | budget_pct | gap | ci_lo | ci_hi | n_seeds | p_value |
|---|---|---|---|---|---|---|
| cifar10 | 5 | 0.002 | -0.012 | 0.015 | 3 | 0.650 |
| cifar10 | 10 | 0.005 | -0.062 | 0.071 | 3 | 0.794 |
| cifar10 | 20 | 0.007 | -0.011 | 0.025 | 3 | 0.252 |
| cifar10 | 50 | -0.002 | -0.038 | 0.035 | 3 | 0.863 |
| cifar10 | 100 | 0.042 | 0.034 | 0.050 | 3 | 0.002 |
| eurosat | 5 | 0.002 | -0.082 | 0.085 | 3 | 0.934 |
| eurosat | 10 | -0.008 | -0.049 | 0.033 | 3 | 0.483 |
| eurosat | 20 | -0.003 | -0.027 | 0.020 | 3 | 0.601 |
| eurosat | 50 | 0.003 | -0.011 | 0.016 | 3 | 0.474 |
| eurosat | 100 | 0.002 | -0.001 | 0.006 | 3 | 0.102 |
| pets | 5 | 0.003 | -0.025 | 0.032 | 3 | 0.661 |
| pets | 10 | -0.011 | -0.090 | 0.068 | 3 | 0.614 |
| pets | 20 | -0.005 | -0.030 | 0.020 | 3 | 0.476 |
| pets | 50 | 0.016 | -0.026 | 0.058 | 3 | 0.239 |
| pets | 100 | 0.034 | -0.031 | 0.100 | 3 | 0.151 |
| plantvillage | 5 | -0.015 | -0.058 | 0.028 | 3 | 0.273 |
| plantvillage | 10 | 0.006 | -0.147 | 0.160 | 3 | 0.876 |
| plantvillage | 20 | 0.006 | -0.009 | 0.022 | 3 | 0.221 |
| plantvillage | 50 | 0.002 | -0.019 | 0.024 | 3 | 0.691 |
| plantvillage | 100 | 0.009 | -0.022 | 0.040 | 3 | 0.339 |

### E2 minus A2  (ViT MAE (20k pool) vs ResNet-34 scratch)
| dataset | budget_pct | gap | ci_lo | ci_hi | n_seeds | p_value |
|---|---|---|---|---|---|---|
| cifar10 | 5 | 0.050 | 0.006 | 0.093 | 3 | 0.039 |
| cifar10 | 10 | 0.033 | -0.015 | 0.081 | 3 | 0.098 |
| cifar10 | 20 | 0.033 | 0.004 | 0.062 | 3 | 0.040 |
| cifar10 | 50 | -0.020 | -0.053 | 0.014 | 3 | 0.129 |
| cifar10 | 100 | -0.057 | -0.077 | -0.038 | 3 | 0.006 |
| eurosat | 5 | 0.020 | -0.092 | 0.131 | 3 | 0.526 |
| eurosat | 10 | 0.022 | -0.042 | 0.086 | 3 | 0.279 |
| eurosat | 20 | 0.015 | -0.024 | 0.054 | 3 | 0.238 |
| eurosat | 50 | -0.013 | -0.037 | 0.010 | 3 | 0.136 |
| eurosat | 100 | -0.020 | -0.033 | -0.007 | 3 | 0.023 |
| plantvillage | 5 | 0.031 | -0.027 | 0.089 | 3 | 0.146 |
| plantvillage | 10 | 0.009 | -0.062 | 0.080 | 3 | 0.638 |
| plantvillage | 20 | -0.001 | -0.064 | 0.063 | 3 | 0.975 |
| plantvillage | 50 | -0.010 | -0.033 | 0.013 | 3 | 0.194 |
| plantvillage | 100 | -0.011 | -0.025 | 0.003 | 3 | 0.082 |

### E minus C  (ViT target-MAE vs ViT ImageNet-1k)
| dataset | budget_pct | gap | ci_lo | ci_hi | n_seeds | p_value |
|---|---|---|---|---|---|---|
| cifar10 | 5 | -0.441 | -0.509 | -0.372 | 3 | 0.001 |
| cifar10 | 10 | -0.381 | -0.416 | -0.347 | 3 | 0.000 |
| cifar10 | 20 | -0.331 | -0.392 | -0.270 | 3 | 0.002 |
| cifar10 | 50 | -0.256 | -0.267 | -0.245 | 3 | 0.000 |
| cifar10 | 100 | -0.203 | -0.258 | -0.148 | 3 | 0.004 |
| eurosat | 5 | -0.179 | -0.198 | -0.160 | 3 | 0.001 |
| eurosat | 10 | -0.150 | -0.205 | -0.096 | 3 | 0.007 |
| eurosat | 20 | -0.109 | -0.158 | -0.061 | 3 | 0.010 |
| eurosat | 50 | -0.060 | -0.077 | -0.042 | 3 | 0.005 |
| eurosat | 100 | -0.045 | -0.054 | -0.035 | 3 | 0.002 |
| pets | 5 | -0.628 | -0.671 | -0.585 | 3 | 0.000 |
| pets | 10 | -0.600 | -0.633 | -0.568 | 3 | 0.000 |
| pets | 20 | -0.545 | -0.579 | -0.512 | 3 | 0.000 |
| pets | 50 | -0.387 | -0.455 | -0.320 | 3 | 0.002 |
| pets | 100 | -0.268 | -0.298 | -0.237 | 3 | 0.001 |
| plantvillage | 5 | -0.249 | -0.364 | -0.135 | 3 | 0.011 |
| plantvillage | 10 | -0.161 | -0.254 | -0.068 | 3 | 0.018 |
| plantvillage | 20 | -0.116 | -0.162 | -0.071 | 3 | 0.008 |
| plantvillage | 50 | -0.066 | -0.084 | -0.048 | 3 | 0.004 |
| plantvillage | 100 | -0.012 | -0.038 | 0.013 | 3 | 0.168 |

## Figures
`plots/learning_curves_v2.png`, `plots/gap_vs_imagenet_v2.png`, `plots/calibration_v2.png`, `plots/winner_map_v2.png`,
`plots/distance_vs_gap_FC_v2.png`.
