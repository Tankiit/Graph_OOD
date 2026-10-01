# Steering / OOD results

## Static held-out OOD detection

| model | rep | detector | id_acc | auroc | fpr95 | aupr |
|---|---|---|---|---|---|---|
| mpnet | full | knn | 0.964 | 0.947 | 0.243 | 0.808 |
| mpnet | full | mahalanobis | 0.964 | 0.938 | 0.244 | 0.711 |
| mpnet | full | energy_T1 | 0.964 | 0.976 | 0.089 | 0.894 |
| mpnet | full | energy_T1000 | 0.964 | 0.228 | 0.995 | 0.113 |
| mpnet | full | msp_T1 | 0.964 | 0.962 | 0.155 | 0.860 |
| mpnet | full | msp_T1000 | 0.964 | 0.978 | 0.091 | 0.915 |
| mpnet | pca64 | knn | 0.942 | 0.906 | 0.309 | 0.623 |
| mpnet | pca64 | mahalanobis | 0.942 | 0.928 | 0.214 | 0.686 |
| mpnet | pca64 | energy_T1 | 0.942 | 0.973 | 0.105 | 0.898 |
| mpnet | pca64 | energy_T1000 | 0.942 | 0.457 | 0.865 | 0.153 |
| mpnet | pca64 | msp_T1 | 0.942 | 0.956 | 0.184 | 0.842 |
| mpnet | pca64 | msp_T1000 | 0.942 | 0.971 | 0.111 | 0.885 |

## Crossed steering (probes pushed along z + alpha*v)

| model | rep | direction | detector | crossing_prob | initial_rejection | median_alpha_star_radii | power_at_horizon | return_prob | var_reference | var_direction | var_interaction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mpnet | full | pca | knn | 1.000 | 0.052 | 0.841 | 1.000 | 0.009 | 0.986 | 0.010 | 0.004 |
| mpnet | full | pca | mahalanobis | 0.351 | 0.000 | 1.175 | 0.351 | 0.000 | 0.979 | 0.015 | 0.006 |
| mpnet | full | pca | energy_T1 | 0.198 | 0.031 | 0.693 | 0.047 | 0.151 | 0.000 | 1.000 | 0.000 |
| mpnet | full | pca | msp_T1 | 0.531 | 0.031 | 0.959 | 0.500 | 0.062 | 0.000 | 1.000 | 0.000 |
| mpnet | full | random | knn | 1.000 | 0.052 | 0.799 | 1.000 | 0.000 | 0.683 | 0.252 | 0.065 |
| mpnet | full | random | mahalanobis | 1.000 | 0.000 | 0.013 | 1.000 | 0.000 | 0.039 | 0.960 | 0.002 |
| mpnet | full | random | energy_T1 | 0.078 | 0.031 | 0.812 | 0.057 | 0.021 | 0.000 | 1.000 | 0.000 |
| mpnet | full | random | msp_T1 | 0.125 | 0.031 | 0.715 | 0.109 | 0.016 | 0.000 | 1.000 | 0.000 |
| mpnet | pca64 | pca | knn | 1.000 | 0.042 | 0.581 | 1.000 | 0.010 | 0.992 | 0.006 | 0.002 |
| mpnet | pca64 | pca | mahalanobis | 1.000 | 0.031 | 0.582 | 1.000 | 0.007 | 0.984 | 0.015 | 0.001 |
| mpnet | pca64 | pca | energy_T1 | 0.089 | 0.031 | 0.172 | 0.016 | 0.073 | 0.000 | 1.000 | 0.000 |
| mpnet | pca64 | pca | msp_T1 | 0.359 | 0.062 | 1.179 | 0.312 | 0.062 | 0.000 | 1.000 | 0.000 |
| mpnet | pca64 | random | knn | 1.000 | 0.042 | 0.564 | 1.000 | 0.002 | 0.340 | 0.493 | 0.167 |
| mpnet | pca64 | random | mahalanobis | 1.000 | 0.031 | 0.514 | 1.000 | 0.003 | 0.235 | 0.708 | 0.058 |
| mpnet | pca64 | random | energy_T1 | 0.083 | 0.031 | 0.195 | 0.000 | 0.083 | 0.000 | 1.000 | 0.000 |
| mpnet | pca64 | random | msp_T1 | 0.474 | 0.062 | 1.111 | 0.281 | 0.208 | 0.000 | 1.000 | 0.000 |

## Steering checks

| model | rep | direction | detector | max_abs_recompute_error | alpha0_matches_unsteered | mean_score_change_over_horizon_in_cal_sd | frac_paths_score_increases | probe_rejection_at_alpha0 | probe_rejection_at_horizon |
|---|---|---|---|---|---|---|---|---|---|
| mpnet | full | pca | knn | 0.000 | 0.000 | 4.066 | 1.000 | 0.052 | 1.000 |
| mpnet | full | pca | mahalanobis | 0.000 | 0.000 | 1.807 | 1.000 | 0.000 | 0.351 |
| mpnet | full | pca | energy_T1 | 0.000 | 0.000 | -0.786 | 0.328 | 0.031 | 0.047 |
| mpnet | full | pca | msp_T1 | 0.000 | 0.000 | 2.311 | 0.922 | 0.031 | 0.500 |
| mpnet | full | random | knn | 0.000 | 0.000 | 4.792 | 1.000 | 0.052 | 1.000 |
| mpnet | full | random | mahalanobis | 0.000 | 0.000 | 28573.559 | 1.000 | 0.000 | 1.000 |
| mpnet | full | random | energy_T1 | 0.000 | 0.000 | -0.045 | 0.484 | 0.031 | 0.057 |
| mpnet | full | random | msp_T1 | 0.000 | 0.000 | 0.232 | 0.646 | 0.031 | 0.109 |
| mpnet | pca64 | pca | knn | 0.000 | 0.000 | 6.378 | 1.000 | 0.042 | 1.000 |
| mpnet | pca64 | pca | mahalanobis | 0.000 | 0.000 | 11.529 | 1.000 | 0.031 | 1.000 |
| mpnet | pca64 | pca | energy_T1 | 0.000 | 0.000 | -0.433 | 0.401 | 0.031 | 0.016 |
| mpnet | pca64 | pca | msp_T1 | 0.000 | 0.000 | 1.157 | 0.875 | 0.062 | 0.312 |
| mpnet | pca64 | random | knn | 0.000 | 0.000 | 7.383 | 1.000 | 0.042 | 1.000 |
| mpnet | pca64 | random | mahalanobis | 0.000 | 0.000 | 16.766 | 1.000 | 0.031 | 1.000 |
| mpnet | pca64 | random | energy_T1 | 0.000 | 0.000 | -0.870 | 0.250 | 0.031 | 0.000 |
| mpnet | pca64 | random | msp_T1 | 0.000 | 0.000 | 1.076 | 0.771 | 0.062 | 0.281 |
