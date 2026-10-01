# Steering / OOD results

## Static held-out OOD detection

| model | rep | detector | id_acc | auroc | fpr95 | aupr | auroc_cifar100 | auroc_svhn |
|---|---|---|---|---|---|---|---|---|
| bge_base | full | knn | 0.961 | 0.939 | 0.260 | 0.782 |  |  |
| bge_base | full | mahalanobis_shrinkage | 0.961 | 0.949 | 0.235 | 0.811 |  |  |
| bge_base | full | energy_T1 | 0.961 | 0.970 | 0.116 | 0.867 |  |  |
| bge_base | full | energy_T1000 | 0.961 | 0.261 | 0.999 | 0.117 |  |  |
| bge_base | full | msp_T1 | 0.961 | 0.960 | 0.170 | 0.858 |  |  |
| bge_base | full | msp_T1000 | 0.961 | 0.977 | 0.092 | 0.910 |  |  |
| bge_large | full | knn | 0.969 | 0.949 | 0.219 | 0.807 |  |  |
| bge_large | full | mahalanobis_shrinkage | 0.969 | 0.952 | 0.211 | 0.819 |  |  |
| bge_large | full | energy_T1 | 0.969 | 0.975 | 0.101 | 0.891 |  |  |
| bge_large | full | energy_T1000 | 0.969 | 0.290 | 0.995 | 0.122 |  |  |
| bge_large | full | msp_T1 | 0.969 | 0.966 | 0.146 | 0.880 |  |  |
| bge_large | full | msp_T1000 | 0.969 | 0.981 | 0.084 | 0.923 |  |  |
| dinov2_s | full | knn | 0.964 | 0.890 | 0.399 | 0.932 | 0.923 | 0.857 |
| dinov2_s | full | mahalanobis_shrinkage | 0.964 | 0.958 | 0.239 | 0.979 | 0.936 | 0.980 |
| dinov2_s | full | energy_T1 | 0.964 | 0.977 | 0.117 | 0.987 | 0.957 | 0.996 |
| dinov2_s | full | energy_T1000 | 0.964 | 0.518 | 0.898 | 0.638 | 0.493 | 0.543 |
| dinov2_s | full | msp_T1 | 0.964 | 0.954 | 0.155 | 0.969 | 0.931 | 0.977 |
| dinov2_s | full | msp_T1000 | 0.964 | 0.977 | 0.110 | 0.987 | 0.958 | 0.996 |
| minilm | full | knn | 0.944 | 0.944 | 0.240 | 0.790 |  |  |
| minilm | full | mahalanobis_shrinkage | 0.944 | 0.946 | 0.195 | 0.776 |  |  |
| minilm | full | energy_T1 | 0.944 | 0.965 | 0.147 | 0.852 |  |  |
| minilm | full | energy_T1000 | 0.944 | 0.213 | 0.996 | 0.111 |  |  |
| minilm | full | msp_T1 | 0.944 | 0.954 | 0.208 | 0.839 |  |  |
| minilm | full | msp_T1000 | 0.944 | 0.971 | 0.116 | 0.889 |  |  |
| mpnet | full | knn | 0.964 | 0.947 | 0.243 | 0.808 |  |  |
| mpnet | full | mahalanobis_shrinkage | 0.964 | 0.947 | 0.218 | 0.770 |  |  |
| mpnet | full | energy_T1 | 0.964 | 0.976 | 0.089 | 0.894 |  |  |
| mpnet | full | energy_T1000 | 0.964 | 0.228 | 0.995 | 0.113 |  |  |
| mpnet | full | msp_T1 | 0.964 | 0.962 | 0.155 | 0.860 |  |  |
| mpnet | full | msp_T1000 | 0.964 | 0.978 | 0.091 | 0.915 |  |  |
| resnet18 | full | knn | 0.809 | 0.418 | 0.949 | 0.653 | 0.613 | 0.224 |
| resnet18 | full | mahalanobis_shrinkage | 0.809 | 0.413 | 0.993 | 0.677 | 0.649 | 0.177 |
| resnet18 | full | energy_T1 | 0.809 | 0.851 | 0.494 | 0.885 | 0.796 | 0.905 |
| resnet18 | full | energy_T1000 | 0.809 | 0.234 | 1.000 | 0.529 | 0.436 | 0.032 |
| resnet18 | full | msp_T1 | 0.809 | 0.811 | 0.561 | 0.873 | 0.741 | 0.881 |
| resnet18 | full | msp_T1000 | 0.809 | 0.876 | 0.509 | 0.930 | 0.790 | 0.961 |
| resnet50 | full | knn | 0.890 | 0.829 | 0.585 | 0.886 | 0.766 | 0.892 |
| resnet50 | full | mahalanobis_shrinkage | 0.890 | 0.790 | 0.639 | 0.869 | 0.785 | 0.795 |
| resnet50 | full | energy_T1 | 0.890 | 0.855 | 0.392 | 0.875 | 0.824 | 0.885 |
| resnet50 | full | energy_T1000 | 0.890 | 0.238 | 0.999 | 0.525 | 0.374 | 0.101 |
| resnet50 | full | msp_T1 | 0.890 | 0.869 | 0.401 | 0.913 | 0.829 | 0.908 |
| resnet50 | full | msp_T1000 | 0.890 | 0.922 | 0.372 | 0.958 | 0.868 | 0.977 |
| vit_b16 | full | knn | 0.925 | 0.887 | 0.406 | 0.924 | 0.854 | 0.920 |
| vit_b16 | full | mahalanobis_shrinkage | 0.925 | 0.862 | 0.496 | 0.917 | 0.869 | 0.855 |
| vit_b16 | full | energy_T1 | 0.925 | 0.951 | 0.200 | 0.970 | 0.926 | 0.976 |
| vit_b16 | full | energy_T1000 | 0.925 | 0.726 | 0.883 | 0.849 | 0.556 | 0.896 |
| vit_b16 | full | msp_T1 | 0.925 | 0.910 | 0.264 | 0.935 | 0.896 | 0.923 |
| vit_b16 | full | msp_T1000 | 0.925 | 0.948 | 0.204 | 0.967 | 0.928 | 0.969 |

## Crossed steering (probes pushed along z + alpha*v)

| model | rep | direction | detector | crossing_prob | initial_rejection | median_alpha_star_radii | power_at_horizon | return_prob | var_reference | var_direction | var_interaction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bge_base | full | origin | knn | 1.000 | 0.052 | 1.087 | 1.000 | 0.039 | 1.000 | 0.000 | 0.000 |
| bge_base | full | origin | mahalanobis_shrinkage | 1.000 | 0.057 | 1.080 | 1.000 | 0.057 | 1.000 | 0.000 | 0.000 |
| bge_base | full | origin | energy_T1 | 1.000 | 0.078 | 0.530 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| bge_base | full | origin | msp_T1 | 1.000 | 0.047 | 0.707 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| bge_base | full | pca | mahalanobis_shrinkage | 1.000 | 0.057 | 1.219 | 1.000 | 0.000 | 0.995 | 0.004 | 0.001 |
| bge_base | full | radial | knn | 1.000 | 0.052 | 0.539 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| bge_base | full | radial | mahalanobis_shrinkage | 1.000 | 0.057 | 0.455 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| bge_base | full | radial | energy_T1 | 0.078 | 0.078 |  | 0.000 | 0.078 | nan | nan | nan |
| bge_base | full | radial | msp_T1 | 0.047 | 0.047 |  | 0.000 | 0.047 | nan | nan | nan |
| bge_base | full | random | mahalanobis_shrinkage | 1.000 | 0.057 | 0.354 | 1.000 | 0.000 | 0.843 | 0.136 | 0.021 |
| bge_base | full | sphere_pca | knn | 1.000 | 0.052 | 0.999 | 1.000 | 0.036 | 0.894 | 0.047 | 0.059 |
| bge_base | full | sphere_pca | mahalanobis_shrinkage | 1.000 | 0.057 | 1.459 | 1.000 | 0.028 | 0.980 | 0.007 | 0.014 |
| bge_base | full | sphere_pca | energy_T1 | 0.286 | 0.078 | 0.732 | 0.000 | 0.286 | 0.000 | 1.000 | 0.000 |
| bge_base | full | sphere_pca | msp_T1 | 1.000 | 0.047 | 0.986 | 1.000 | 0.031 | 0.000 | 1.000 | 0.000 |
| bge_base | full | sphere_random | knn | 1.000 | 0.052 | 0.870 | 1.000 | 0.001 | 0.659 | 0.245 | 0.096 |
| bge_base | full | sphere_random | mahalanobis_shrinkage | 1.000 | 0.057 | 0.367 | 1.000 | 0.000 | 0.835 | 0.141 | 0.023 |
| bge_base | full | sphere_random | energy_T1 | 1.000 | 0.078 | 1.276 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| bge_base | full | sphere_random | msp_T1 | 1.000 | 0.047 | 1.406 | 1.000 | 0.009 | 0.000 | 1.000 | 0.000 |
| bge_large | full | origin | knn | 1.000 | 0.065 | 1.078 | 1.000 | 0.042 | 1.000 | 0.000 | 0.000 |
| bge_large | full | origin | mahalanobis_shrinkage | 1.000 | 0.060 | 1.040 | 1.000 | 0.060 | 1.000 | 0.000 | 0.000 |
| bge_large | full | origin | energy_T1 | 1.000 | 0.031 | 0.556 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| bge_large | full | origin | msp_T1 | 1.000 | 0.062 | 0.705 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| bge_large | full | pca | mahalanobis_shrinkage | 1.000 | 0.060 | 1.301 | 1.000 | 0.000 | 0.995 | 0.004 | 0.001 |
| bge_large | full | radial | knn | 1.000 | 0.065 | 0.552 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| bge_large | full | radial | mahalanobis_shrinkage | 1.000 | 0.060 | 0.475 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| bge_large | full | radial | energy_T1 | 0.031 | 0.031 |  | 0.000 | 0.031 | nan | nan | nan |
| bge_large | full | radial | msp_T1 | 0.062 | 0.062 |  | 0.000 | 0.062 | nan | nan | nan |
| bge_large | full | random | mahalanobis_shrinkage | 1.000 | 0.060 | 0.328 | 1.000 | 0.000 | 0.845 | 0.139 | 0.016 |
| bge_large | full | sphere_pca | knn | 1.000 | 0.065 | 1.010 | 1.000 | 0.032 | 0.891 | 0.042 | 0.067 |
| bge_large | full | sphere_pca | mahalanobis_shrinkage | 1.000 | 0.060 | 1.464 | 1.000 | 0.021 | 0.998 | 0.002 | 0.000 |
| bge_large | full | sphere_pca | energy_T1 | 0.247 | 0.031 | 0.687 | 0.000 | 0.247 | 0.000 | 1.000 | 0.000 |
| bge_large | full | sphere_pca | msp_T1 | 1.000 | 0.062 | 1.008 | 0.996 | 0.013 | 0.000 | 1.000 | 0.000 |
| bge_large | full | sphere_random | knn | 1.000 | 0.065 | 0.874 | 1.000 | 0.000 | 0.743 | 0.185 | 0.072 |
| bge_large | full | sphere_random | mahalanobis_shrinkage | 1.000 | 0.060 | 0.338 | 1.000 | 0.000 | 0.834 | 0.149 | 0.017 |
| bge_large | full | sphere_random | energy_T1 | 1.000 | 0.031 | 1.296 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| bge_large | full | sphere_random | msp_T1 | 1.000 | 0.062 | 1.416 | 1.000 | 0.003 | 0.000 | 1.000 | 0.000 |
| dinov2_s | full | origin | knn | 0.031 | 0.031 |  | 0.000 | 0.031 | 1.000 | 0.000 | 0.000 |
| dinov2_s | full | origin | mahalanobis_shrinkage | 0.016 | 0.016 |  | 0.000 | 0.016 | 1.000 | 0.000 | 0.000 |
| dinov2_s | full | origin | energy_T1 | 1.000 | 0.094 | 0.595 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| dinov2_s | full | origin | msp_T1 | 1.000 | 0.062 | 0.844 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| dinov2_s | full | pca | mahalanobis_shrinkage | 1.000 | 0.016 | 0.721 | 1.000 | 0.004 | 0.118 | 0.848 | 0.034 |
| dinov2_s | full | radial | knn | 1.000 | 0.031 | 0.272 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| dinov2_s | full | radial | mahalanobis_shrinkage | 1.000 | 0.016 | 0.238 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| dinov2_s | full | radial | energy_T1 | 0.094 | 0.094 |  | 0.000 | 0.094 | nan | nan | nan |
| dinov2_s | full | radial | msp_T1 | 0.062 | 0.062 |  | 0.016 | 0.047 | nan | nan | nan |
| dinov2_s | full | random | mahalanobis_shrinkage | 1.000 | 0.016 | 0.472 | 1.000 | 0.001 | 0.571 | 0.367 | 0.062 |
| dinov2_s | full | sphere_pca | knn | 1.000 | 0.031 | 1.575 | 1.000 | 0.125 | 0.042 | 0.868 | 0.089 |
| dinov2_s | full | sphere_pca | mahalanobis_shrinkage | 1.000 | 0.016 | 2.098 | 1.000 | 0.039 | 0.313 | 0.522 | 0.165 |
| dinov2_s | full | sphere_pca | energy_T1 | 0.216 | 0.094 | 0.547 | 0.051 | 0.210 | 0.000 | 1.000 | 0.000 |
| dinov2_s | full | sphere_pca | msp_T1 | 1.000 | 0.062 | 0.732 | 0.501 | 0.938 | 0.000 | 1.000 | 0.000 |
| dinov2_s | full | sphere_random | knn | 1.000 | 0.031 | 0.871 | 1.000 | 0.003 | 0.476 | 0.373 | 0.150 |
| dinov2_s | full | sphere_random | mahalanobis_shrinkage | 1.000 | 0.016 | 0.617 | 1.000 | 0.001 | 0.516 | 0.383 | 0.101 |
| dinov2_s | full | sphere_random | energy_T1 | 0.993 | 0.094 | 1.227 | 0.471 | 0.560 | 0.000 | 1.000 | 0.000 |
| dinov2_s | full | sphere_random | msp_T1 | 1.000 | 0.062 | 1.272 | 0.729 | 0.509 | 0.000 | 1.000 | 0.000 |
| minilm | full | origin | knn | 0.078 | 0.078 |  | 0.000 | 0.078 | 1.000 | 0.000 | 0.000 |
| minilm | full | origin | mahalanobis_shrinkage | 0.060 | 0.060 |  | 0.000 | 0.060 | 1.000 | 0.000 | 0.000 |
| minilm | full | origin | energy_T1 | 1.000 | 0.062 | 0.383 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| minilm | full | origin | msp_T1 | 1.000 | 0.062 | 0.593 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| minilm | full | pca | mahalanobis_shrinkage | 1.000 | 0.060 | 1.075 | 1.000 | 0.002 | 0.561 | 0.392 | 0.047 |
| minilm | full | radial | knn | 1.000 | 0.078 | 0.415 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| minilm | full | radial | mahalanobis_shrinkage | 1.000 | 0.060 | 0.431 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| minilm | full | radial | energy_T1 | 0.062 | 0.062 |  | 0.000 | 0.062 | nan | nan | nan |
| minilm | full | radial | msp_T1 | 0.062 | 0.062 |  | 0.000 | 0.062 | nan | nan | nan |
| minilm | full | random | mahalanobis_shrinkage | 1.000 | 0.060 | 0.498 | 1.000 | 0.000 | 0.729 | 0.245 | 0.025 |
| minilm | full | sphere_pca | knn | 1.000 | 0.078 | 1.309 | 1.000 | 0.092 | 0.059 | 0.889 | 0.052 |
| minilm | full | sphere_pca | mahalanobis_shrinkage | 0.730 | 0.060 | 2.698 | 0.727 | 0.066 | 0.685 | 0.243 | 0.072 |
| minilm | full | sphere_pca | energy_T1 | 0.835 | 0.062 | 2.172 | 0.661 | 0.427 | 0.000 | 1.000 | 0.000 |
| minilm | full | sphere_pca | msp_T1 | 1.000 | 0.062 | 0.899 | 1.000 | 0.062 | 0.000 | 1.000 | 0.000 |
| minilm | full | sphere_random | knn | 1.000 | 0.078 | 0.967 | 1.000 | 0.000 | 0.577 | 0.315 | 0.108 |
| minilm | full | sphere_random | mahalanobis_shrinkage | 1.000 | 0.060 | 0.581 | 0.968 | 0.032 | 0.695 | 0.271 | 0.033 |
| minilm | full | sphere_random | energy_T1 | 1.000 | 0.062 | 0.923 | 0.786 | 0.215 | 0.000 | 1.000 | 0.000 |
| minilm | full | sphere_random | msp_T1 | 1.000 | 0.062 | 1.101 | 0.996 | 0.025 | 0.000 | 1.000 | 0.000 |
| mpnet | full | origin | knn | 0.055 | 0.055 |  | 0.000 | 0.055 | 1.000 | 0.000 | 0.000 |
| mpnet | full | origin | mahalanobis_shrinkage | 0.076 | 0.076 |  | 0.000 | 0.076 | 1.000 | 0.000 | 0.000 |
| mpnet | full | origin | energy_T1 | 1.000 | 0.062 | 0.362 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| mpnet | full | origin | msp_T1 | 1.000 | 0.031 | 0.569 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| mpnet | full | pca | mahalanobis_shrinkage | 1.000 | 0.076 | 1.314 | 1.000 | 0.000 | 0.985 | 0.013 | 0.002 |
| mpnet | full | radial | knn | 1.000 | 0.055 | 0.471 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| mpnet | full | radial | mahalanobis_shrinkage | 1.000 | 0.076 | 0.529 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| mpnet | full | radial | energy_T1 | 0.062 | 0.062 |  | 0.000 | 0.062 | nan | nan | nan |
| mpnet | full | radial | msp_T1 | 0.031 | 0.031 |  | 0.031 | 0.000 | nan | nan | nan |
| mpnet | full | random | mahalanobis_shrinkage | 1.000 | 0.076 | 0.419 | 1.000 | 0.000 | 0.768 | 0.205 | 0.027 |
| mpnet | full | sphere_pca | knn | 1.000 | 0.055 | 1.503 | 1.000 | 0.056 | 0.881 | 0.027 | 0.092 |
| mpnet | full | sphere_pca | mahalanobis_shrinkage | 0.463 | 0.076 | 2.745 | 0.453 | 0.089 | 0.967 | 0.007 | 0.026 |
| mpnet | full | sphere_pca | energy_T1 | 1.000 | 0.062 | 1.006 | 0.982 | 0.353 | 0.000 | 1.000 | 0.000 |
| mpnet | full | sphere_pca | msp_T1 | 1.000 | 0.031 | 0.907 | 1.000 | 0.001 | 0.000 | 1.000 | 0.000 |
| mpnet | full | sphere_random | knn | 1.000 | 0.055 | 1.014 | 1.000 | 0.000 | 0.683 | 0.224 | 0.092 |
| mpnet | full | sphere_random | mahalanobis_shrinkage | 1.000 | 0.076 | 0.455 | 0.984 | 0.016 | 0.736 | 0.233 | 0.032 |
| mpnet | full | sphere_random | energy_T1 | 1.000 | 0.062 | 0.891 | 0.993 | 0.007 | 0.000 | 1.000 | 0.000 |
| mpnet | full | sphere_random | msp_T1 | 1.000 | 0.031 | 1.121 | 1.000 | 0.008 | 0.000 | 1.000 | 0.000 |
| resnet18 | full | origin | knn | 0.044 | 0.044 |  | 0.000 | 0.044 | 1.000 | 0.000 | 0.000 |
| resnet18 | full | origin | mahalanobis_shrinkage | 0.021 | 0.021 |  | 0.000 | 0.021 | 1.000 | 0.000 | 0.000 |
| resnet18 | full | origin | energy_T1 | 0.375 | 0.047 | 0.873 | 0.375 | 0.000 | 1.000 | 0.000 | 0.000 |
| resnet18 | full | origin | msp_T1 | 0.203 | 0.031 | 0.703 | 0.203 | 0.000 | 1.000 | 0.000 | 0.000 |
| resnet18 | full | pca | mahalanobis_shrinkage | 0.979 | 0.021 | 1.953 | 0.979 | 0.001 | 0.914 | 0.070 | 0.015 |
| resnet18 | full | radial | knn | 1.000 | 0.044 | 0.359 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| resnet18 | full | radial | mahalanobis_shrinkage | 1.000 | 0.021 | 0.393 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| resnet18 | full | radial | energy_T1 | 0.047 | 0.047 |  | 0.000 | 0.047 | nan | nan | nan |
| resnet18 | full | radial | msp_T1 | 0.031 | 0.031 |  | 0.016 | 0.016 | nan | nan | nan |
| resnet18 | full | random | mahalanobis_shrinkage | 1.000 | 0.021 | 0.345 | 1.000 | 0.001 | 0.512 | 0.410 | 0.078 |
| resnet18 | full | sphere_pca | knn | 1.000 | 0.044 | 1.279 | 1.000 | 0.042 | 0.901 | 0.052 | 0.046 |
| resnet18 | full | sphere_pca | mahalanobis_shrinkage | 0.507 | 0.021 | 2.605 | 0.492 | 0.024 | 0.883 | 0.026 | 0.091 |
| resnet18 | full | sphere_pca | energy_T1 | 0.240 | 0.047 | 0.254 | 0.000 | 0.240 | 0.000 | 1.000 | 0.000 |
| resnet18 | full | sphere_pca | msp_T1 | 0.914 | 0.031 | 0.883 | 0.098 | 0.888 | 0.000 | 1.000 | 0.000 |
| resnet18 | full | sphere_random | knn | 1.000 | 0.044 | 0.800 | 1.000 | 0.001 | 0.686 | 0.227 | 0.087 |
| resnet18 | full | sphere_random | mahalanobis_shrinkage | 1.000 | 0.021 | 0.359 | 1.000 | 0.001 | 0.494 | 0.422 | 0.084 |
| resnet18 | full | sphere_random | energy_T1 | 0.082 | 0.047 | 0.288 | 0.000 | 0.082 | 0.000 | 1.000 | 0.000 |
| resnet18 | full | sphere_random | msp_T1 | 0.889 | 0.031 | 1.179 | 0.069 | 0.882 | 0.000 | 1.000 | 0.000 |
| resnet50 | full | origin | knn | 0.039 | 0.039 |  | 0.000 | 0.039 | 1.000 | 0.000 | 0.000 |
| resnet50 | full | origin | mahalanobis_shrinkage | 0.005 | 0.005 |  | 0.000 | 0.005 | 1.000 | 0.000 | 0.000 |
| resnet50 | full | origin | energy_T1 | 0.031 | 0.031 |  | 0.000 | 0.031 | nan | nan | nan |
| resnet50 | full | origin | msp_T1 | 0.484 | 0.047 | 0.820 | 0.484 | 0.000 | 1.000 | 0.000 | 0.000 |
| resnet50 | full | pca | mahalanobis_shrinkage | 0.920 | 0.005 | 2.211 | 0.920 | 0.001 | 0.958 | 0.031 | 0.011 |
| resnet50 | full | radial | knn | 1.000 | 0.039 | 0.388 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| resnet50 | full | radial | mahalanobis_shrinkage | 1.000 | 0.005 | 0.427 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| resnet50 | full | radial | energy_T1 | 0.078 | 0.031 | 1.337 | 0.078 | 0.000 | 1.000 | 0.000 | 0.000 |
| resnet50 | full | radial | msp_T1 | 0.047 | 0.047 |  | 0.016 | 0.031 | nan | nan | nan |
| resnet50 | full | random | mahalanobis_shrinkage | 1.000 | 0.005 | 0.440 | 1.000 | 0.000 | 0.702 | 0.248 | 0.050 |
| resnet50 | full | sphere_pca | knn | 1.000 | 0.039 | 1.674 | 1.000 | 0.069 | 0.869 | 0.068 | 0.063 |
| resnet50 | full | sphere_pca | mahalanobis_shrinkage | 0.053 | 0.005 | 2.581 | 0.035 | 0.020 | 0.998 | 0.002 | 0.000 |
| resnet50 | full | sphere_pca | energy_T1 | 0.195 | 0.031 | 0.348 | 0.000 | 0.195 | 0.000 | 1.000 | 0.000 |
| resnet50 | full | sphere_pca | msp_T1 | 1.000 | 0.047 | 0.808 | 0.266 | 0.924 | 0.000 | 1.000 | 0.000 |
| resnet50 | full | sphere_random | knn | 1.000 | 0.039 | 0.932 | 1.000 | 0.000 | 0.864 | 0.087 | 0.049 |
| resnet50 | full | sphere_random | mahalanobis_shrinkage | 1.000 | 0.005 | 0.484 | 0.984 | 0.016 | 0.700 | 0.239 | 0.061 |
| resnet50 | full | sphere_random | energy_T1 | 0.049 | 0.031 | 0.295 | 0.000 | 0.049 | 0.000 | 1.000 | 0.000 |
| resnet50 | full | sphere_random | msp_T1 | 0.997 | 0.047 | 1.377 | 0.324 | 0.895 | 0.000 | 1.000 | 0.000 |
| vit_b16 | full | origin | knn | 0.073 | 0.073 |  | 0.000 | 0.073 | 1.000 | 0.000 | 0.000 |
| vit_b16 | full | origin | mahalanobis_shrinkage | 0.031 | 0.031 |  | 0.000 | 0.031 | 1.000 | 0.000 | 0.000 |
| vit_b16 | full | origin | energy_T1 | 0.984 | 0.094 | 0.558 | 0.984 | 0.000 | 1.000 | 0.000 | 0.000 |
| vit_b16 | full | origin | msp_T1 | 0.438 | 0.062 | 0.738 | 0.438 | 0.000 | 1.000 | 0.000 | 0.000 |
| vit_b16 | full | pca | mahalanobis_shrinkage | 1.000 | 0.031 | 1.506 | 1.000 | 0.009 | 0.939 | 0.049 | 0.013 |
| vit_b16 | full | radial | knn | 1.000 | 0.073 | 0.394 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| vit_b16 | full | radial | mahalanobis_shrinkage | 1.000 | 0.031 | 0.375 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| vit_b16 | full | radial | energy_T1 | 0.094 | 0.094 |  | 0.000 | 0.094 | nan | nan | nan |
| vit_b16 | full | radial | msp_T1 | 0.062 | 0.062 |  | 0.000 | 0.062 | nan | nan | nan |
| vit_b16 | full | random | mahalanobis_shrinkage | 1.000 | 0.031 | 0.525 | 1.000 | 0.000 | 0.484 | 0.429 | 0.087 |
| vit_b16 | full | sphere_pca | knn | 0.704 | 0.073 | 1.887 | 0.662 | 0.130 | 0.776 | 0.087 | 0.136 |
| vit_b16 | full | sphere_pca | mahalanobis_shrinkage | 0.108 | 0.031 | 2.347 | 0.066 | 0.042 | 0.999 | 0.001 | 0.000 |
| vit_b16 | full | sphere_pca | energy_T1 | 0.271 | 0.094 | 0.601 | 0.023 | 0.263 | 0.000 | 1.000 | 0.000 |
| vit_b16 | full | sphere_pca | msp_T1 | 1.000 | 0.062 | 0.877 | 0.448 | 0.824 | 0.000 | 1.000 | 0.000 |
| vit_b16 | full | sphere_random | knn | 0.971 | 0.073 | 1.081 | 0.970 | 0.002 | 0.524 | 0.293 | 0.183 |
| vit_b16 | full | sphere_random | mahalanobis_shrinkage | 1.000 | 0.031 | 0.645 | 0.967 | 0.033 | 0.440 | 0.432 | 0.128 |
| vit_b16 | full | sphere_random | energy_T1 | 0.987 | 0.094 | 1.181 | 0.740 | 0.268 | 0.000 | 1.000 | 0.000 |
| vit_b16 | full | sphere_random | msp_T1 | 1.000 | 0.062 | 1.294 | 0.621 | 0.681 | 0.000 | 1.000 | 0.000 |

## Steering sensitivity vs static detection

Spearman correlation across model x representation cells, per detector. crossing = fraction of steered ID probes rejected within the horizon; alpha = median crossing distance in ID radii.

| detector | n | rho(crossing_pca,auroc) | rho(crossing_random,auroc) | rho(alpha_radii_pca,auroc) | rho(alpha_radii_random,auroc) |
|---|---|---|---|---|---|
| energy_T1 | 8 |  |  |  |  |
| knn | 8 |  |  |  |  |
| mahalanobis_shrinkage | 8 | 0.73 (p=0.039) | nan (p=nan) | -0.81 (p=0.015) | -0.12 (p=0.78) |
| msp_T1 | 8 |  |  |  |  |

## Steering checks

| model | rep | direction | detector | max_abs_recompute_error | alpha0_matches_unsteered | mean_score_change_over_horizon_in_cal_sd | frac_paths_score_increases | probe_rejection_at_alpha0 | probe_rejection_at_horizon |
|---|---|---|---|---|---|---|---|---|---|
| bge_base | full | origin | knn | 0.000 | 0.000 | 2.949 | 1.000 | 0.052 | 1.000 |
| bge_base | full | origin | mahalanobis_shrinkage | 0.000 | 0.000 | 3.569 | 0.995 | 0.057 | 1.000 |
| bge_base | full | origin | energy_T1 | 0.000 | 0.000 | 2.605 | 1.000 | 0.078 | 1.000 |
| bge_base | full | origin | msp_T1 | 0.000 | 0.000 | 7.945 | 1.000 | 0.047 | 1.000 |
| bge_base | full | pca | mahalanobis_shrinkage | 0.000 | 0.000 | 10.923 | 1.000 | 0.057 | 1.000 |
| bge_base | full | radial | knn | 0.000 | 0.000 | 15.021 | 1.000 | 0.052 | 1.000 |
| bge_base | full | radial | mahalanobis_shrinkage | 0.000 | 0.000 | 47.544 | 1.000 | 0.057 | 1.000 |
| bge_base | full | radial | energy_T1 | 0.000 | 0.000 | -11.037 | 0.000 | 0.078 | 0.000 |
| bge_base | full | radial | msp_T1 | 0.000 | 0.000 | -0.336 | 0.000 | 0.047 | 0.000 |
| bge_base | full | random | mahalanobis_shrinkage | 0.000 | 0.000 | 143.402 | 1.000 | 0.057 | 1.000 |
| bge_base | full | sphere_pca | knn | 0.000 | 0.000 | 6.133 | 1.000 | 0.052 | 1.000 |
| bge_base | full | sphere_pca | mahalanobis_shrinkage | 0.000 | 0.000 | 13.206 | 1.000 | 0.057 | 1.000 |
| bge_base | full | sphere_pca | energy_T1 | 0.000 | 0.000 | -0.791 | 0.215 | 0.078 | 0.000 |
| bge_base | full | sphere_pca | msp_T1 | 0.000 | 0.000 | 5.989 | 0.996 | 0.047 | 1.000 |
| bge_base | full | sphere_random | knn | 0.000 | 0.000 | 7.287 | 1.000 | 0.052 | 1.000 |
| bge_base | full | sphere_random | mahalanobis_shrinkage | 0.000 | 0.000 | 33.418 | 1.000 | 0.057 | 1.000 |
| bge_base | full | sphere_random | energy_T1 | 0.000 | 0.000 | 1.316 | 0.868 | 0.078 | 0.000 |
| bge_base | full | sphere_random | msp_T1 | 0.000 | 0.000 | 7.224 | 1.000 | 0.047 | 1.000 |
| bge_large | full | origin | knn | 0.000 | 0.000 | 3.081 | 1.000 | 0.065 | 1.000 |
| bge_large | full | origin | mahalanobis_shrinkage | 0.000 | 0.000 | 4.075 | 1.000 | 0.060 | 1.000 |
| bge_large | full | origin | energy_T1 | 0.000 | 0.000 | 2.508 | 0.969 | 0.031 | 1.000 |
| bge_large | full | origin | msp_T1 | 0.000 | 0.000 | 9.462 | 1.000 | 0.062 | 1.000 |
| bge_large | full | pca | mahalanobis_shrinkage | 0.000 | 0.000 | 10.045 | 1.000 | 0.060 | 1.000 |
| bge_large | full | radial | knn | 0.000 | 0.000 | 14.918 | 1.000 | 0.065 | 1.000 |
| bge_large | full | radial | mahalanobis_shrinkage | 0.000 | 0.000 | 47.768 | 1.000 | 0.060 | 1.000 |
| bge_large | full | radial | energy_T1 | 0.000 | 0.000 | -10.796 | 0.000 | 0.031 | 0.000 |
| bge_large | full | radial | msp_T1 | 0.000 | 0.000 | -0.349 | 0.000 | 0.062 | 0.000 |
| bge_large | full | random | mahalanobis_shrinkage | 0.000 | 0.000 | 173.003 | 1.000 | 0.060 | 1.000 |
| bge_large | full | sphere_pca | knn | 0.000 | 0.000 | 6.256 | 1.000 | 0.065 | 1.000 |
| bge_large | full | sphere_pca | mahalanobis_shrinkage | 0.000 | 0.000 | 13.974 | 1.000 | 0.060 | 1.000 |
| bge_large | full | sphere_pca | energy_T1 | 0.000 | 0.000 | -1.076 | 0.163 | 0.031 | 0.000 |
| bge_large | full | sphere_pca | msp_T1 | 0.000 | 0.000 | 7.274 | 0.993 | 0.062 | 0.996 |
| bge_large | full | sphere_random | knn | 0.000 | 0.000 | 7.457 | 1.000 | 0.065 | 1.000 |
| bge_large | full | sphere_random | mahalanobis_shrinkage | 0.000 | 0.000 | 39.527 | 1.000 | 0.060 | 1.000 |
| bge_large | full | sphere_random | energy_T1 | 0.000 | 0.000 | 1.098 | 0.878 | 0.031 | 0.000 |
| bge_large | full | sphere_random | msp_T1 | 0.000 | 0.000 | 8.616 | 1.000 | 0.062 | 1.000 |
| dinov2_s | full | origin | knn | 0.000 | 0.000 | -0.924 | 0.146 | 0.031 | 0.000 |
| dinov2_s | full | origin | mahalanobis_shrinkage | 0.000 | 0.000 | -2.368 | 0.000 | 0.016 | 0.000 |
| dinov2_s | full | origin | energy_T1 | 0.000 | 0.000 | 2.978 | 1.000 | 0.094 | 1.000 |
| dinov2_s | full | origin | msp_T1 | 0.000 | 0.000 | 5.266 | 1.000 | 0.062 | 1.000 |
| dinov2_s | full | pca | mahalanobis_shrinkage | 0.000 | 0.000 | 35.273 | 1.000 | 0.016 | 1.000 |
| dinov2_s | full | radial | knn | 0.000 | 0.000 | 26.645 | 1.000 | 0.031 | 1.000 |
| dinov2_s | full | radial | mahalanobis_shrinkage | 0.000 | 0.000 | 69.327 | 1.000 | 0.016 | 1.000 |
| dinov2_s | full | radial | energy_T1 | 0.000 | 0.000 | -9.505 | 0.000 | 0.094 | 0.000 |
| dinov2_s | full | radial | msp_T1 | 0.000 | 0.000 | -0.153 | 0.000 | 0.062 | 0.016 |
| dinov2_s | full | random | mahalanobis_shrinkage | 0.000 | 0.000 | 80.748 | 1.000 | 0.016 | 1.000 |
| dinov2_s | full | sphere_pca | knn | 0.000 | 0.000 | 4.005 | 1.000 | 0.031 | 1.000 |
| dinov2_s | full | sphere_pca | mahalanobis_shrinkage | 0.000 | 0.000 | 5.027 | 1.000 | 0.016 | 1.000 |
| dinov2_s | full | sphere_pca | energy_T1 | 0.000 | 0.000 | 0.635 | 0.717 | 0.094 | 0.051 |
| dinov2_s | full | sphere_pca | msp_T1 | 0.000 | 0.000 | 1.613 | 0.931 | 0.062 | 0.501 |
| dinov2_s | full | sphere_random | knn | 0.000 | 0.000 | 4.665 | 1.000 | 0.031 | 1.000 |
| dinov2_s | full | sphere_random | mahalanobis_shrinkage | 0.000 | 0.000 | 6.836 | 1.000 | 0.016 | 1.000 |
| dinov2_s | full | sphere_random | energy_T1 | 0.000 | 0.000 | 1.714 | 0.944 | 0.094 | 0.471 |
| dinov2_s | full | sphere_random | msp_T1 | 0.000 | 0.000 | 3.177 | 0.966 | 0.062 | 0.729 |
| minilm | full | origin | knn | 0.000 | 0.000 | 0.721 | 0.719 | 0.078 | 0.000 |
| minilm | full | origin | mahalanobis_shrinkage | 0.000 | 0.000 | -1.163 | 0.141 | 0.060 | 0.000 |
| minilm | full | origin | energy_T1 | 0.000 | 0.000 | 3.233 | 1.000 | 0.062 | 1.000 |
| minilm | full | origin | msp_T1 | 0.000 | 0.000 | 5.999 | 1.000 | 0.062 | 1.000 |
| minilm | full | pca | mahalanobis_shrinkage | 0.000 | 0.000 | 13.883 | 1.000 | 0.060 | 1.000 |
| minilm | full | radial | knn | 0.000 | 0.000 | 14.287 | 1.000 | 0.078 | 1.000 |
| minilm | full | radial | mahalanobis_shrinkage | 0.000 | 0.000 | 39.104 | 1.000 | 0.060 | 1.000 |
| minilm | full | radial | energy_T1 | 0.000 | 0.000 | -16.123 | 0.000 | 0.062 | 0.000 |
| minilm | full | radial | msp_T1 | 0.000 | 0.000 | -0.341 | 0.000 | 0.062 | 0.000 |
| minilm | full | random | mahalanobis_shrinkage | 0.000 | 0.000 | 72.352 | 1.000 | 0.060 | 1.000 |
| minilm | full | sphere_pca | knn | 0.000 | 0.000 | 2.483 | 0.999 | 0.078 | 1.000 |
| minilm | full | sphere_pca | mahalanobis_shrinkage | 0.000 | 0.000 | 2.220 | 0.999 | 0.060 | 0.727 |
| minilm | full | sphere_pca | energy_T1 | 0.000 | 0.000 | 1.969 | 0.953 | 0.062 | 0.661 |
| minilm | full | sphere_pca | msp_T1 | 0.000 | 0.000 | 5.063 | 0.984 | 0.062 | 1.000 |
| minilm | full | sphere_random | knn | 0.000 | 0.000 | 2.632 | 1.000 | 0.078 | 1.000 |
| minilm | full | sphere_random | mahalanobis_shrinkage | 0.000 | 0.000 | 2.763 | 1.000 | 0.060 | 0.968 |
| minilm | full | sphere_random | energy_T1 | 0.000 | 0.000 | 2.097 | 0.953 | 0.062 | 0.786 |
| minilm | full | sphere_random | msp_T1 | 0.000 | 0.000 | 5.073 | 0.979 | 0.062 | 0.996 |
| mpnet | full | origin | knn | 0.000 | 0.000 | 0.837 | 0.734 | 0.055 | 0.000 |
| mpnet | full | origin | mahalanobis_shrinkage | 0.000 | 0.000 | -1.113 | 0.094 | 0.076 | 0.000 |
| mpnet | full | origin | energy_T1 | 0.000 | 0.000 | 3.536 | 1.000 | 0.062 | 1.000 |
| mpnet | full | origin | msp_T1 | 0.000 | 0.000 | 7.758 | 1.000 | 0.031 | 1.000 |
| mpnet | full | pca | mahalanobis_shrinkage | 0.000 | 0.000 | 10.189 | 1.000 | 0.076 | 1.000 |
| mpnet | full | radial | knn | 0.000 | 0.000 | 14.431 | 1.000 | 0.055 | 1.000 |
| mpnet | full | radial | mahalanobis_shrinkage | 0.000 | 0.000 | 34.434 | 1.000 | 0.076 | 1.000 |
| mpnet | full | radial | energy_T1 | 0.000 | 0.000 | -16.885 | 0.000 | 0.062 | 0.000 |
| mpnet | full | radial | msp_T1 | 0.000 | 0.000 | -0.261 | 0.000 | 0.031 | 0.031 |
| mpnet | full | random | mahalanobis_shrinkage | 0.000 | 0.000 | 110.972 | 1.000 | 0.076 | 1.000 |
| mpnet | full | sphere_pca | knn | 0.000 | 0.000 | 2.703 | 1.000 | 0.055 | 1.000 |
| mpnet | full | sphere_pca | mahalanobis_shrinkage | 0.000 | 0.000 | 1.874 | 0.999 | 0.076 | 0.453 |
| mpnet | full | sphere_pca | energy_T1 | 0.000 | 0.000 | 2.332 | 0.969 | 0.062 | 0.982 |
| mpnet | full | sphere_pca | msp_T1 | 0.000 | 0.000 | 6.751 | 1.000 | 0.031 | 1.000 |
| mpnet | full | sphere_random | knn | 0.000 | 0.000 | 2.886 | 1.000 | 0.055 | 1.000 |
| mpnet | full | sphere_random | mahalanobis_shrinkage | 0.000 | 0.000 | 2.929 | 1.000 | 0.076 | 0.984 |
| mpnet | full | sphere_random | energy_T1 | 0.000 | 0.000 | 2.487 | 0.969 | 0.062 | 0.993 |
| mpnet | full | sphere_random | msp_T1 | 0.000 | 0.000 | 6.925 | 1.000 | 0.031 | 1.000 |
| resnet18 | full | origin | knn | 0.000 | 0.000 | -1.153 | 0.188 | 0.044 | 0.000 |
| resnet18 | full | origin | mahalanobis_shrinkage | 0.000 | 0.000 | -1.561 | 0.060 | 0.021 | 0.000 |
| resnet18 | full | origin | energy_T1 | 0.000 | 0.000 | 0.974 | 1.000 | 0.047 | 0.375 |
| resnet18 | full | origin | msp_T1 | 0.000 | 0.000 | 1.030 | 1.000 | 0.031 | 0.203 |
| resnet18 | full | pca | mahalanobis_shrinkage | 0.000 | 0.000 | 4.654 | 1.000 | 0.021 | 0.979 |
| resnet18 | full | radial | knn | 0.000 | 0.000 | 19.454 | 1.000 | 0.044 | 1.000 |
| resnet18 | full | radial | mahalanobis_shrinkage | 0.000 | 0.000 | 29.768 | 1.000 | 0.021 | 1.000 |
| resnet18 | full | radial | energy_T1 | 0.000 | 0.000 | -2.988 | 0.000 | 0.047 | 0.000 |
| resnet18 | full | radial | msp_T1 | 0.000 | 0.000 | -0.455 | 0.000 | 0.031 | 0.016 |
| resnet18 | full | random | mahalanobis_shrinkage | 0.000 | 0.000 | 130.195 | 1.000 | 0.021 | 1.000 |
| resnet18 | full | sphere_pca | knn | 0.000 | 0.000 | 7.872 | 1.000 | 0.044 | 1.000 |
| resnet18 | full | sphere_pca | mahalanobis_shrinkage | 0.000 | 0.000 | 2.401 | 0.843 | 0.021 | 0.492 |
| resnet18 | full | sphere_pca | energy_T1 | 0.000 | 0.000 | -4.531 | 0.000 | 0.047 | 0.000 |
| resnet18 | full | sphere_pca | msp_T1 | 0.000 | 0.000 | 0.615 | 0.732 | 0.031 | 0.098 |
| resnet18 | full | sphere_random | knn | 0.000 | 0.000 | 9.540 | 1.000 | 0.044 | 1.000 |
| resnet18 | full | sphere_random | mahalanobis_shrinkage | 0.000 | 0.000 | 46.524 | 1.000 | 0.021 | 1.000 |
| resnet18 | full | sphere_random | energy_T1 | 0.000 | 0.000 | -1.074 | 0.190 | 0.047 | 0.000 |
| resnet18 | full | sphere_random | msp_T1 | 0.000 | 0.000 | 0.261 | 0.632 | 0.031 | 0.069 |
| resnet50 | full | origin | knn | 0.000 | 0.000 | -1.321 | 0.055 | 0.039 | 0.000 |
| resnet50 | full | origin | mahalanobis_shrinkage | 0.000 | 0.000 | -1.782 | 0.000 | 0.005 | 0.000 |
| resnet50 | full | origin | energy_T1 | 0.000 | 0.000 | 0.921 | 0.844 | 0.031 | 0.000 |
| resnet50 | full | origin | msp_T1 | 0.000 | 0.000 | 2.429 | 1.000 | 0.047 | 0.484 |
| resnet50 | full | pca | mahalanobis_shrinkage | 0.000 | 0.000 | 3.339 | 1.000 | 0.005 | 0.920 |
| resnet50 | full | radial | knn | 0.000 | 0.000 | 16.432 | 1.000 | 0.039 | 1.000 |
| resnet50 | full | radial | mahalanobis_shrinkage | 0.000 | 0.000 | 26.835 | 1.000 | 0.005 | 1.000 |
| resnet50 | full | radial | energy_T1 | 0.000 | 0.000 | -3.303 | 0.109 | 0.031 | 0.078 |
| resnet50 | full | radial | msp_T1 | 0.000 | 0.000 | -0.326 | 0.000 | 0.047 | 0.016 |
| resnet50 | full | random | mahalanobis_shrinkage | 0.000 | 0.000 | 86.075 | 1.000 | 0.005 | 1.000 |
| resnet50 | full | sphere_pca | knn | 0.000 | 0.000 | 3.417 | 1.000 | 0.039 | 1.000 |
| resnet50 | full | sphere_pca | mahalanobis_shrinkage | 0.000 | 0.000 | 0.175 | 0.521 | 0.005 | 0.035 |
| resnet50 | full | sphere_pca | energy_T1 | 0.000 | 0.000 | -3.800 | 0.000 | 0.031 | 0.000 |
| resnet50 | full | sphere_pca | msp_T1 | 0.000 | 0.000 | 1.333 | 0.827 | 0.047 | 0.266 |
| resnet50 | full | sphere_random | knn | 0.000 | 0.000 | 4.251 | 1.000 | 0.039 | 1.000 |
| resnet50 | full | sphere_random | mahalanobis_shrinkage | 0.000 | 0.000 | 10.915 | 1.000 | 0.005 | 0.984 |
| resnet50 | full | sphere_random | energy_T1 | 0.000 | 0.000 | -1.621 | 0.078 | 0.031 | 0.000 |
| resnet50 | full | sphere_random | msp_T1 | 0.000 | 0.000 | 1.704 | 0.845 | 0.047 | 0.324 |
| vit_b16 | full | origin | knn | 0.000 | 0.000 | -1.231 | 0.115 | 0.073 | 0.000 |
| vit_b16 | full | origin | mahalanobis_shrinkage | 0.000 | 0.000 | -1.877 | 0.000 | 0.031 | 0.000 |
| vit_b16 | full | origin | energy_T1 | 0.000 | 0.000 | 2.241 | 1.000 | 0.094 | 0.984 |
| vit_b16 | full | origin | msp_T1 | 0.000 | 0.000 | 2.405 | 1.000 | 0.062 | 0.438 |
| vit_b16 | full | pca | mahalanobis_shrinkage | 0.000 | 0.000 | 8.951 | 1.000 | 0.031 | 1.000 |
| vit_b16 | full | radial | knn | 0.000 | 0.000 | 15.913 | 1.000 | 0.073 | 1.000 |
| vit_b16 | full | radial | mahalanobis_shrinkage | 0.000 | 0.000 | 40.884 | 1.000 | 0.031 | 1.000 |
| vit_b16 | full | radial | energy_T1 | 0.000 | 0.000 | -8.487 | 0.000 | 0.094 | 0.000 |
| vit_b16 | full | radial | msp_T1 | 0.000 | 0.000 | -0.263 | 0.000 | 0.062 | 0.000 |
| vit_b16 | full | random | mahalanobis_shrinkage | 0.000 | 0.000 | 69.584 | 1.000 | 0.031 | 1.000 |
| vit_b16 | full | sphere_pca | knn | 0.000 | 0.000 | 1.697 | 0.954 | 0.073 | 0.662 |
| vit_b16 | full | sphere_pca | mahalanobis_shrinkage | 0.000 | 0.000 | 0.729 | 0.697 | 0.031 | 0.066 |
| vit_b16 | full | sphere_pca | energy_T1 | 0.000 | 0.000 | 0.227 | 0.607 | 0.094 | 0.023 |
| vit_b16 | full | sphere_pca | msp_T1 | 0.000 | 0.000 | 2.352 | 0.926 | 0.062 | 0.448 |
| vit_b16 | full | sphere_random | knn | 0.000 | 0.000 | 2.286 | 1.000 | 0.073 | 0.970 |
| vit_b16 | full | sphere_random | mahalanobis_shrinkage | 0.000 | 0.000 | 5.516 | 1.000 | 0.031 | 0.967 |
| vit_b16 | full | sphere_random | energy_T1 | 0.000 | 0.000 | 1.679 | 0.932 | 0.094 | 0.740 |
| vit_b16 | full | sphere_random | msp_T1 | 0.000 | 0.000 | 3.199 | 0.958 | 0.062 | 0.621 |
