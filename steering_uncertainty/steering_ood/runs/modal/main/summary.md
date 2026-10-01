# Steering / OOD results

## Static held-out OOD detection

| model | rep | detector | id_acc | auroc | fpr95 | aupr | auroc_cifar100 | auroc_svhn |
|---|---|---|---|---|---|---|---|---|
| bge_base | full | knn | 0.961 | 0.939 | 0.260 | 0.782 |  |  |
| bge_base | full | mahalanobis | 0.961 | 0.942 | 0.253 | 0.780 |  |  |
| bge_base | full | energy_T1 | 0.961 | 0.970 | 0.116 | 0.867 |  |  |
| bge_base | full | energy_T1000 | 0.961 | 0.261 | 0.999 | 0.117 |  |  |
| bge_base | full | msp_T1 | 0.961 | 0.960 | 0.170 | 0.858 |  |  |
| bge_base | full | msp_T1000 | 0.961 | 0.977 | 0.092 | 0.910 |  |  |
| bge_base | pca64 | knn | 0.948 | 0.954 | 0.179 | 0.801 |  |  |
| bge_base | pca64 | mahalanobis | 0.948 | 0.956 | 0.157 | 0.809 |  |  |
| bge_base | pca64 | energy_T1 | 0.948 | 0.963 | 0.171 | 0.856 |  |  |
| bge_base | pca64 | energy_T1000 | 0.948 | 0.482 | 0.884 | 0.163 |  |  |
| bge_base | pca64 | msp_T1 | 0.948 | 0.952 | 0.213 | 0.823 |  |  |
| bge_base | pca64 | msp_T1000 | 0.948 | 0.964 | 0.158 | 0.857 |  |  |
| bge_large | full | knn | 0.969 | 0.949 | 0.219 | 0.807 |  |  |
| bge_large | full | mahalanobis | 0.969 | 0.946 | 0.236 | 0.790 |  |  |
| bge_large | full | energy_T1 | 0.969 | 0.975 | 0.101 | 0.891 |  |  |
| bge_large | full | energy_T1000 | 0.969 | 0.290 | 0.995 | 0.122 |  |  |
| bge_large | full | msp_T1 | 0.969 | 0.966 | 0.146 | 0.880 |  |  |
| bge_large | full | msp_T1000 | 0.969 | 0.981 | 0.084 | 0.923 |  |  |
| bge_large | pca64 | knn | 0.960 | 0.961 | 0.152 | 0.826 |  |  |
| bge_large | pca64 | mahalanobis | 0.960 | 0.963 | 0.138 | 0.839 |  |  |
| bge_large | pca64 | energy_T1 | 0.960 | 0.971 | 0.123 | 0.878 |  |  |
| bge_large | pca64 | energy_T1000 | 0.960 | 0.461 | 0.891 | 0.153 |  |  |
| bge_large | pca64 | msp_T1 | 0.960 | 0.961 | 0.167 | 0.860 |  |  |
| bge_large | pca64 | msp_T1000 | 0.960 | 0.972 | 0.125 | 0.888 |  |  |
| dinov2_s | full | knn | 0.964 | 0.890 | 0.399 | 0.932 | 0.923 | 0.857 |
| dinov2_s | full | mahalanobis | 0.964 | 0.958 | 0.256 | 0.979 | 0.928 | 0.987 |
| dinov2_s | full | energy_T1 | 0.964 | 0.977 | 0.117 | 0.987 | 0.957 | 0.996 |
| dinov2_s | full | energy_T1000 | 0.964 | 0.518 | 0.898 | 0.638 | 0.493 | 0.543 |
| dinov2_s | full | msp_T1 | 0.964 | 0.954 | 0.155 | 0.969 | 0.931 | 0.977 |
| dinov2_s | full | msp_T1000 | 0.964 | 0.977 | 0.110 | 0.987 | 0.958 | 0.996 |
| dinov2_s | pca64 | knn | 0.956 | 0.363 | 0.957 | 0.599 | 0.600 | 0.126 |
| dinov2_s | pca64 | mahalanobis | 0.956 | 0.564 | 0.856 | 0.691 | 0.646 | 0.482 |
| dinov2_s | pca64 | energy_T1 | 0.956 | 0.968 | 0.141 | 0.981 | 0.945 | 0.991 |
| dinov2_s | pca64 | energy_T1000 | 0.956 | 0.555 | 0.741 | 0.639 | 0.550 | 0.560 |
| dinov2_s | pca64 | msp_T1 | 0.956 | 0.941 | 0.183 | 0.959 | 0.919 | 0.962 |
| dinov2_s | pca64 | msp_T1000 | 0.956 | 0.972 | 0.123 | 0.984 | 0.951 | 0.992 |
| minilm | full | knn | 0.944 | 0.944 | 0.240 | 0.790 |  |  |
| minilm | full | mahalanobis | 0.944 | 0.936 | 0.203 | 0.692 |  |  |
| minilm | full | energy_T1 | 0.944 | 0.965 | 0.147 | 0.852 |  |  |
| minilm | full | energy_T1000 | 0.944 | 0.213 | 0.996 | 0.111 |  |  |
| minilm | full | msp_T1 | 0.944 | 0.954 | 0.208 | 0.839 |  |  |
| minilm | full | msp_T1000 | 0.944 | 0.971 | 0.116 | 0.889 |  |  |
| minilm | pca64 | knn | 0.933 | 0.925 | 0.257 | 0.714 |  |  |
| minilm | pca64 | mahalanobis | 0.933 | 0.925 | 0.248 | 0.696 |  |  |
| minilm | pca64 | energy_T1 | 0.933 | 0.961 | 0.160 | 0.841 |  |  |
| minilm | pca64 | energy_T1000 | 0.933 | 0.435 | 0.845 | 0.147 |  |  |
| minilm | pca64 | msp_T1 | 0.933 | 0.946 | 0.223 | 0.811 |  |  |
| minilm | pca64 | msp_T1000 | 0.933 | 0.962 | 0.154 | 0.852 |  |  |
| mpnet | full | knn | 0.964 | 0.947 | 0.243 | 0.808 |  |  |
| mpnet | full | mahalanobis | 0.964 | 0.938 | 0.244 | 0.711 |  |  |
| mpnet | full | energy_T1 | 0.964 | 0.976 | 0.089 | 0.894 |  |  |
| mpnet | full | energy_T1000 | 0.964 | 0.228 | 0.995 | 0.113 |  |  |
| mpnet | full | msp_T1 | 0.964 | 0.962 | 0.155 | 0.860 |  |  |
| mpnet | full | msp_T1000 | 0.964 | 0.978 | 0.091 | 0.915 |  |  |
| mpnet | pca64 | knn | 0.942 | 0.906 | 0.309 | 0.623 |  |  |
| mpnet | pca64 | mahalanobis | 0.942 | 0.928 | 0.214 | 0.686 |  |  |
| mpnet | pca64 | energy_T1 | 0.942 | 0.973 | 0.105 | 0.898 |  |  |
| mpnet | pca64 | energy_T1000 | 0.942 | 0.457 | 0.865 | 0.153 |  |  |
| mpnet | pca64 | msp_T1 | 0.942 | 0.956 | 0.184 | 0.842 |  |  |
| mpnet | pca64 | msp_T1000 | 0.942 | 0.971 | 0.111 | 0.885 |  |  |
| resnet18 | full | knn | 0.809 | 0.418 | 0.949 | 0.653 | 0.613 | 0.224 |
| resnet18 | full | mahalanobis | 0.809 | 0.422 | 0.994 | 0.682 | 0.647 | 0.197 |
| resnet18 | full | energy_T1 | 0.809 | 0.851 | 0.494 | 0.885 | 0.796 | 0.905 |
| resnet18 | full | energy_T1000 | 0.809 | 0.234 | 1.000 | 0.529 | 0.436 | 0.032 |
| resnet18 | full | msp_T1 | 0.809 | 0.811 | 0.561 | 0.873 | 0.741 | 0.881 |
| resnet18 | full | msp_T1000 | 0.809 | 0.876 | 0.509 | 0.930 | 0.790 | 0.961 |
| resnet18 | pca64 | knn | 0.807 | 0.471 | 0.886 | 0.649 | 0.598 | 0.344 |
| resnet18 | pca64 | mahalanobis | 0.807 | 0.422 | 0.946 | 0.646 | 0.611 | 0.233 |
| resnet18 | pca64 | energy_T1 | 0.807 | 0.879 | 0.456 | 0.921 | 0.824 | 0.935 |
| resnet18 | pca64 | energy_T1000 | 0.807 | 0.518 | 0.894 | 0.664 | 0.526 | 0.509 |
| resnet18 | pca64 | msp_T1 | 0.807 | 0.812 | 0.511 | 0.868 | 0.775 | 0.849 |
| resnet18 | pca64 | msp_T1000 | 0.807 | 0.876 | 0.454 | 0.913 | 0.818 | 0.935 |
| resnet50 | full | knn | 0.890 | 0.829 | 0.585 | 0.886 | 0.766 | 0.892 |
| resnet50 | full | mahalanobis | 0.890 | 0.570 | 0.977 | 0.747 | 0.480 | 0.660 |
| resnet50 | full | energy_T1 | 0.890 | 0.855 | 0.392 | 0.875 | 0.824 | 0.885 |
| resnet50 | full | energy_T1000 | 0.890 | 0.238 | 0.999 | 0.525 | 0.374 | 0.101 |
| resnet50 | full | msp_T1 | 0.890 | 0.869 | 0.401 | 0.913 | 0.829 | 0.908 |
| resnet50 | full | msp_T1000 | 0.890 | 0.922 | 0.372 | 0.958 | 0.868 | 0.977 |
| resnet50 | pca64 | knn | 0.872 | 0.780 | 0.603 | 0.835 | 0.703 | 0.857 |
| resnet50 | pca64 | mahalanobis | 0.872 | 0.766 | 0.685 | 0.838 | 0.686 | 0.846 |
| resnet50 | pca64 | energy_T1 | 0.872 | 0.920 | 0.322 | 0.951 | 0.889 | 0.952 |
| resnet50 | pca64 | energy_T1000 | 0.872 | 0.523 | 0.847 | 0.656 | 0.540 | 0.507 |
| resnet50 | pca64 | msp_T1 | 0.872 | 0.874 | 0.407 | 0.917 | 0.850 | 0.898 |
| resnet50 | pca64 | msp_T1000 | 0.872 | 0.916 | 0.336 | 0.948 | 0.882 | 0.950 |
| vit_b16 | full | knn | 0.925 | 0.887 | 0.406 | 0.924 | 0.854 | 0.920 |
| vit_b16 | full | mahalanobis | 0.925 | 0.858 | 0.519 | 0.914 | 0.851 | 0.864 |
| vit_b16 | full | energy_T1 | 0.925 | 0.951 | 0.200 | 0.970 | 0.926 | 0.976 |
| vit_b16 | full | energy_T1000 | 0.925 | 0.726 | 0.883 | 0.849 | 0.556 | 0.896 |
| vit_b16 | full | msp_T1 | 0.925 | 0.910 | 0.264 | 0.935 | 0.896 | 0.923 |
| vit_b16 | full | msp_T1000 | 0.925 | 0.948 | 0.204 | 0.967 | 0.928 | 0.969 |
| vit_b16 | pca64 | knn | 0.923 | 0.726 | 0.549 | 0.780 | 0.710 | 0.743 |
| vit_b16 | pca64 | mahalanobis | 0.923 | 0.709 | 0.705 | 0.774 | 0.693 | 0.725 |
| vit_b16 | pca64 | energy_T1 | 0.923 | 0.955 | 0.212 | 0.975 | 0.925 | 0.985 |
| vit_b16 | pca64 | energy_T1000 | 0.923 | 0.700 | 0.797 | 0.804 | 0.550 | 0.850 |
| vit_b16 | pca64 | msp_T1 | 0.923 | 0.916 | 0.275 | 0.944 | 0.900 | 0.931 |
| vit_b16 | pca64 | msp_T1000 | 0.923 | 0.949 | 0.223 | 0.970 | 0.924 | 0.975 |

## Crossed steering (probes pushed along z + alpha*v)

| model | rep | direction | detector | crossing_prob | initial_rejection | median_alpha_star_radii | power_at_horizon | return_prob | var_reference | var_direction | var_interaction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bge_base | full | pca | knn | 1.000 | 0.052 | 0.810 | 1.000 | 0.015 | 0.977 | 0.013 | 0.010 |
| bge_base | full | pca | mahalanobis | 1.000 | 0.060 | 1.524 | 1.000 | 0.000 | 0.993 | 0.004 | 0.003 |
| bge_base | full | pca | energy_T1 | 0.212 | 0.078 | 0.647 | 0.000 | 0.212 | 0.000 | 1.000 | 0.000 |
| bge_base | full | pca | msp_T1 | 0.827 | 0.047 | 1.187 | 0.542 | 0.301 | 0.000 | 1.000 | 0.000 |
| bge_base | full | random | knn | 1.000 | 0.052 | 0.766 | 1.000 | 0.000 | 0.758 | 0.174 | 0.068 |
| bge_base | full | random | mahalanobis | 1.000 | 0.060 | 0.109 | 1.000 | 0.000 | 0.024 | 0.968 | 0.008 |
| bge_base | full | random | energy_T1 | 0.134 | 0.078 | 1.759 | 0.070 | 0.064 | 0.000 | 1.000 | 0.000 |
| bge_base | full | random | msp_T1 | 0.263 | 0.047 | 2.303 | 0.219 | 0.046 | 0.000 | 1.000 | 0.000 |
| bge_base | pca64 | pca | knn | 1.000 | 0.081 | 0.575 | 1.000 | 0.016 | 0.994 | 0.003 | 0.003 |
| bge_base | pca64 | pca | mahalanobis | 1.000 | 0.077 | 0.593 | 1.000 | 0.003 | 0.995 | 0.004 | 0.001 |
| bge_base | pca64 | pca | energy_T1 | 0.136 | 0.070 | 0.654 | 0.000 | 0.136 | 0.000 | 1.000 | 0.000 |
| bge_base | pca64 | pca | msp_T1 | 0.633 | 0.102 | 1.528 | 0.502 | 0.154 | 0.000 | 1.000 | 0.000 |
| bge_base | pca64 | random | knn | 1.000 | 0.081 | 0.575 | 1.000 | 0.008 | 0.495 | 0.386 | 0.119 |
| bge_base | pca64 | random | mahalanobis | 1.000 | 0.077 | 0.512 | 1.000 | 0.004 | 0.317 | 0.621 | 0.062 |
| bge_base | pca64 | random | energy_T1 | 0.089 | 0.070 | 0.461 | 0.000 | 0.089 | 0.000 | 1.000 | 0.000 |
| bge_base | pca64 | random | msp_T1 | 0.817 | 0.102 | 1.383 | 0.159 | 0.686 | 0.000 | 1.000 | 0.000 |
| bge_large | full | pca | knn | 1.000 | 0.065 | 0.833 | 1.000 | 0.007 | 0.975 | 0.017 | 0.008 |
| bge_large | full | pca | mahalanobis | 1.000 | 0.062 | 1.666 | 1.000 | 0.000 | 0.995 | 0.003 | 0.002 |
| bge_large | full | pca | energy_T1 | 0.174 | 0.031 | 0.576 | 0.000 | 0.174 | 0.000 | 1.000 | 0.000 |
| bge_large | full | pca | msp_T1 | 0.823 | 0.062 | 1.179 | 0.572 | 0.290 | 0.000 | 1.000 | 0.000 |
| bge_large | full | random | knn | 1.000 | 0.065 | 0.773 | 1.000 | 0.000 | 0.817 | 0.131 | 0.052 |
| bge_large | full | random | mahalanobis | 1.000 | 0.062 | 0.111 | 1.000 | 0.000 | 0.036 | 0.953 | 0.011 |
| bge_large | full | random | energy_T1 | 0.082 | 0.031 | 1.334 | 0.060 | 0.022 | 0.000 | 1.000 | 0.000 |
| bge_large | full | random | msp_T1 | 0.208 | 0.062 | 2.231 | 0.181 | 0.027 | 0.000 | 1.000 | 0.000 |
| bge_large | pca64 | pca | knn | 1.000 | 0.096 | 0.590 | 1.000 | 0.009 | 0.993 | 0.003 | 0.004 |
| bge_large | pca64 | pca | mahalanobis | 1.000 | 0.073 | 0.597 | 1.000 | 0.003 | 0.996 | 0.003 | 0.000 |
| bge_large | pca64 | pca | energy_T1 | 0.184 | 0.070 | 0.691 | 0.000 | 0.184 | 0.000 | 1.000 | 0.000 |
| bge_large | pca64 | pca | msp_T1 | 0.639 | 0.078 | 1.582 | 0.556 | 0.087 | 0.000 | 1.000 | 0.000 |
| bge_large | pca64 | random | knn | 1.000 | 0.096 | 0.584 | 1.000 | 0.007 | 0.529 | 0.358 | 0.113 |
| bge_large | pca64 | random | mahalanobis | 1.000 | 0.073 | 0.499 | 1.000 | 0.005 | 0.368 | 0.560 | 0.071 |
| bge_large | pca64 | random | energy_T1 | 0.104 | 0.070 | 0.357 | 0.000 | 0.104 | 0.000 | 1.000 | 0.000 |
| bge_large | pca64 | random | msp_T1 | 0.854 | 0.078 | 1.425 | 0.241 | 0.645 | 0.000 | 1.000 | 0.000 |
| dinov2_s | full | pca | knn | 1.000 | 0.031 | 0.639 | 1.000 | 0.012 | 0.140 | 0.775 | 0.085 |
| dinov2_s | full | pca | mahalanobis | 1.000 | 0.023 | 0.816 | 1.000 | 0.139 | 0.139 | 0.790 | 0.071 |
| dinov2_s | full | pca | energy_T1 | 0.193 | 0.094 | 0.499 | 0.000 | 0.193 | 0.000 | 1.000 | 0.000 |
| dinov2_s | full | pca | msp_T1 | 0.858 | 0.062 | 0.749 | 0.073 | 0.790 | 0.000 | 1.000 | 0.000 |
| dinov2_s | full | random | knn | 1.000 | 0.031 | 0.601 | 1.000 | 0.002 | 0.628 | 0.275 | 0.097 |
| dinov2_s | full | random | mahalanobis | 0.352 | 0.023 | 0.001 | 0.333 | 0.018 | 1.000 | 0.000 | 0.000 |
| dinov2_s | full | random | energy_T1 | 0.180 | 0.094 | 1.547 | 0.036 | 0.143 | 0.000 | 1.000 | 0.000 |
| dinov2_s | full | random | msp_T1 | 0.444 | 0.062 | 2.025 | 0.242 | 0.212 | 0.000 | 1.000 | 0.000 |
| dinov2_s | pca64 | pca | knn | 1.000 | 0.050 | 0.559 | 1.000 | 0.010 | 0.720 | 0.194 | 0.086 |
| dinov2_s | pca64 | pca | mahalanobis | 1.000 | 0.039 | 0.420 | 1.000 | 0.045 | 0.512 | 0.355 | 0.132 |
| dinov2_s | pca64 | pca | energy_T1 | 0.111 | 0.039 | 0.340 | 0.000 | 0.111 | 0.000 | 1.000 | 0.000 |
| dinov2_s | pca64 | pca | msp_T1 | 0.781 | 0.078 | 0.839 | 0.146 | 0.678 | 0.000 | 1.000 | 0.000 |
| dinov2_s | pca64 | random | knn | 1.000 | 0.050 | 0.559 | 1.000 | 0.007 | 0.431 | 0.432 | 0.137 |
| dinov2_s | pca64 | random | mahalanobis | 1.000 | 0.039 | 0.531 | 1.000 | 0.005 | 0.274 | 0.647 | 0.079 |
| dinov2_s | pca64 | random | energy_T1 | 0.111 | 0.039 | 0.533 | 0.000 | 0.111 | 0.000 | 1.000 | 0.000 |
| dinov2_s | pca64 | random | msp_T1 | 0.709 | 0.078 | 1.196 | 0.136 | 0.604 | 0.000 | 1.000 | 0.000 |
| minilm | full | pca | knn | 1.000 | 0.078 | 0.794 | 1.000 | 0.018 | 0.282 | 0.621 | 0.097 |
| minilm | full | pca | mahalanobis | 1.000 | 0.049 | 1.247 | 1.000 | 0.000 | 0.522 | 0.441 | 0.037 |
| minilm | full | pca | energy_T1 | 0.159 | 0.062 | 0.730 | 0.000 | 0.159 | 0.000 | 1.000 | 0.000 |
| minilm | full | pca | msp_T1 | 0.802 | 0.062 | 1.226 | 0.438 | 0.391 | 0.000 | 1.000 | 0.000 |
| minilm | full | random | knn | 1.000 | 0.078 | 0.722 | 1.000 | 0.000 | 0.722 | 0.211 | 0.068 |
| minilm | full | random | mahalanobis | 1.000 | 0.049 | 0.008 | 1.000 | 0.000 | 0.043 | 0.949 | 0.008 |
| minilm | full | random | energy_T1 | 0.089 | 0.062 | 1.166 | 0.001 | 0.087 | 0.000 | 1.000 | 0.000 |
| minilm | full | random | msp_T1 | 0.487 | 0.062 | 2.186 | 0.303 | 0.195 | 0.000 | 1.000 | 0.000 |
| minilm | pca64 | pca | knn | 1.000 | 0.066 | 0.587 | 1.000 | 0.014 | 0.979 | 0.007 | 0.014 |
| minilm | pca64 | pca | mahalanobis | 1.000 | 0.053 | 0.630 | 1.000 | 0.008 | 0.975 | 0.013 | 0.012 |
| minilm | pca64 | pca | energy_T1 | 0.113 | 0.062 | 0.350 | 0.000 | 0.113 | 0.000 | 1.000 | 0.000 |
| minilm | pca64 | pca | msp_T1 | 0.648 | 0.086 | 1.570 | 0.373 | 0.294 | 0.000 | 1.000 | 0.000 |
| minilm | pca64 | random | knn | 1.000 | 0.066 | 0.592 | 1.000 | 0.004 | 0.479 | 0.400 | 0.120 |
| minilm | pca64 | random | mahalanobis | 1.000 | 0.053 | 0.545 | 1.000 | 0.005 | 0.284 | 0.652 | 0.064 |
| minilm | pca64 | random | energy_T1 | 0.075 | 0.062 | 0.318 | 0.000 | 0.075 | 0.000 | 1.000 | 0.000 |
| minilm | pca64 | random | msp_T1 | 0.675 | 0.086 | 1.378 | 0.130 | 0.567 | 0.000 | 1.000 | 0.000 |
| mpnet | full | pca | knn | 1.000 | 0.055 | 0.814 | 1.000 | 0.012 | 0.969 | 0.012 | 0.018 |
| mpnet | full | pca | mahalanobis | 1.000 | 0.065 | 1.609 | 1.000 | 0.000 | 0.979 | 0.018 | 0.003 |
| mpnet | full | pca | energy_T1 | 0.164 | 0.062 | 0.717 | 0.000 | 0.164 | 0.000 | 1.000 | 0.000 |
| mpnet | full | pca | msp_T1 | 0.810 | 0.031 | 1.310 | 0.322 | 0.526 | 0.000 | 1.000 | 0.000 |
| mpnet | full | random | knn | 1.000 | 0.055 | 0.772 | 1.000 | 0.000 | 0.793 | 0.150 | 0.057 |
| mpnet | full | random | mahalanobis | 1.000 | 0.065 | 0.013 | 1.000 | 0.000 | 0.032 | 0.964 | 0.004 |
| mpnet | full | random | energy_T1 | 0.096 | 0.062 | 1.609 | 0.038 | 0.059 | 0.000 | 1.000 | 0.000 |
| mpnet | full | random | msp_T1 | 0.290 | 0.031 | 2.240 | 0.210 | 0.083 | 0.000 | 1.000 | 0.000 |
| mpnet | pca64 | pca | knn | 1.000 | 0.056 | 0.600 | 1.000 | 0.012 | 0.985 | 0.006 | 0.009 |
| mpnet | pca64 | pca | mahalanobis | 1.000 | 0.056 | 0.602 | 1.000 | 0.004 | 0.988 | 0.007 | 0.005 |
| mpnet | pca64 | pca | energy_T1 | 0.154 | 0.078 | 0.443 | 0.000 | 0.154 | 0.000 | 1.000 | 0.000 |
| mpnet | pca64 | pca | msp_T1 | 0.653 | 0.094 | 1.643 | 0.441 | 0.226 | 0.000 | 1.000 | 0.000 |
| mpnet | pca64 | random | knn | 1.000 | 0.056 | 0.596 | 1.000 | 0.005 | 0.428 | 0.443 | 0.129 |
| mpnet | pca64 | random | mahalanobis | 1.000 | 0.056 | 0.531 | 1.000 | 0.002 | 0.268 | 0.664 | 0.068 |
| mpnet | pca64 | random | energy_T1 | 0.104 | 0.078 | 0.326 | 0.000 | 0.104 | 0.000 | 1.000 | 0.000 |
| mpnet | pca64 | random | msp_T1 | 0.763 | 0.094 | 1.448 | 0.181 | 0.613 | 0.000 | 1.000 | 0.000 |
| resnet18 | full | pca | knn | 1.000 | 0.044 | 1.003 | 1.000 | 0.018 | 0.848 | 0.101 | 0.051 |
| resnet18 | full | pca | mahalanobis | 0.788 | 0.044 | 2.355 | 0.788 | 0.003 | 0.960 | 0.018 | 0.022 |
| resnet18 | full | pca | energy_T1 | 0.241 | 0.047 | 0.254 | 0.000 | 0.241 | 0.000 | 1.000 | 0.000 |
| resnet18 | full | pca | msp_T1 | 0.702 | 0.031 | 0.587 | 0.036 | 0.688 | 0.000 | 1.000 | 0.000 |
| resnet18 | full | random | knn | 1.000 | 0.044 | 0.675 | 1.000 | 0.001 | 0.672 | 0.256 | 0.073 |
| resnet18 | full | random | mahalanobis | 1.000 | 0.044 | 0.317 | 1.000 | 0.003 | 0.567 | 0.194 | 0.239 |
| resnet18 | full | random | energy_T1 | 0.078 | 0.047 | 0.230 | 0.000 | 0.078 | 0.000 | 1.000 | 0.000 |
| resnet18 | full | random | msp_T1 | 0.697 | 0.031 | 0.985 | 0.022 | 0.681 | 0.000 | 1.000 | 0.000 |
| resnet18 | pca64 | pca | knn | 1.000 | 0.060 | 1.010 | 1.000 | 0.026 | 0.845 | 0.092 | 0.063 |
| resnet18 | pca64 | pca | mahalanobis | 1.000 | 0.032 | 1.386 | 1.000 | 0.006 | 0.885 | 0.081 | 0.034 |
| resnet18 | pca64 | pca | energy_T1 | 0.217 | 0.031 | 0.294 | 0.000 | 0.217 | 0.000 | 1.000 | 0.000 |
| resnet18 | pca64 | pca | msp_T1 | 0.517 | 0.047 | 0.588 | 0.012 | 0.512 | 0.000 | 1.000 | 0.000 |
| resnet18 | pca64 | random | knn | 1.000 | 0.060 | 0.753 | 1.000 | 0.004 | 0.233 | 0.643 | 0.124 |
| resnet18 | pca64 | random | mahalanobis | 1.000 | 0.032 | 0.622 | 1.000 | 0.001 | 0.164 | 0.757 | 0.079 |
| resnet18 | pca64 | random | energy_T1 | 0.072 | 0.031 | 0.200 | 0.000 | 0.072 | 0.000 | 1.000 | 0.000 |
| resnet18 | pca64 | random | msp_T1 | 0.371 | 0.047 | 0.925 | 0.014 | 0.362 | 0.000 | 1.000 | 0.000 |
| resnet50 | full | pca | knn | 1.000 | 0.039 | 0.924 | 1.000 | 0.029 | 0.863 | 0.077 | 0.059 |
| resnet50 | full | pca | mahalanobis | 0.152 | 0.036 | 2.105 | 0.139 | 0.013 | 0.891 | 0.017 | 0.092 |
| resnet50 | full | pca | energy_T1 | 0.195 | 0.031 | 0.347 | 0.000 | 0.195 | 0.000 | 1.000 | 0.000 |
| resnet50 | full | pca | msp_T1 | 0.671 | 0.047 | 0.695 | 0.092 | 0.626 | 0.000 | 1.000 | 0.000 |
| resnet50 | full | random | knn | 1.000 | 0.039 | 0.714 | 1.000 | 0.000 | 0.876 | 0.094 | 0.030 |
| resnet50 | full | random | mahalanobis | 0.603 | 0.036 | 0.467 | 0.570 | 0.040 | 0.072 | 0.299 | 0.629 |
| resnet50 | full | random | energy_T1 | 0.065 | 0.031 | 0.607 | 0.009 | 0.056 | 0.000 | 1.000 | 0.000 |
| resnet50 | full | random | msp_T1 | 0.538 | 0.047 | 1.240 | 0.073 | 0.478 | 0.000 | 1.000 | 0.000 |
| resnet50 | pca64 | pca | knn | 1.000 | 0.045 | 0.764 | 1.000 | 0.051 | 0.757 | 0.101 | 0.142 |
| resnet50 | pca64 | pca | mahalanobis | 1.000 | 0.066 | 1.011 | 1.000 | 0.011 | 0.911 | 0.052 | 0.037 |
| resnet50 | pca64 | pca | energy_T1 | 0.216 | 0.039 | 0.364 | 0.000 | 0.216 | 0.000 | 1.000 | 0.000 |
| resnet50 | pca64 | pca | msp_T1 | 0.597 | 0.055 | 0.769 | 0.123 | 0.529 | 0.000 | 1.000 | 0.000 |
| resnet50 | pca64 | random | knn | 1.000 | 0.045 | 0.647 | 1.000 | 0.003 | 0.255 | 0.595 | 0.149 |
| resnet50 | pca64 | random | mahalanobis | 1.000 | 0.066 | 0.524 | 1.000 | 0.003 | 0.206 | 0.712 | 0.082 |
| resnet50 | pca64 | random | energy_T1 | 0.083 | 0.039 | 0.406 | 0.000 | 0.083 | 0.000 | 1.000 | 0.000 |
| resnet50 | pca64 | random | msp_T1 | 0.679 | 0.055 | 1.098 | 0.041 | 0.653 | 0.000 | 1.000 | 0.000 |
| vit_b16 | full | pca | knn | 1.000 | 0.073 | 0.890 | 1.000 | 0.015 | 0.895 | 0.082 | 0.023 |
| vit_b16 | full | pca | mahalanobis | 0.177 | 0.036 | 1.469 | 0.136 | 0.041 | 0.945 | 0.008 | 0.047 |
| vit_b16 | full | pca | energy_T1 | 0.225 | 0.094 | 0.524 | 0.000 | 0.225 | 0.000 | 1.000 | 0.000 |
| vit_b16 | full | pca | msp_T1 | 0.624 | 0.062 | 0.701 | 0.167 | 0.549 | 0.000 | 1.000 | 0.000 |
| vit_b16 | full | random | knn | 1.000 | 0.073 | 0.741 | 1.000 | 0.001 | 0.686 | 0.234 | 0.080 |
| vit_b16 | full | random | mahalanobis | 0.537 | 0.036 | 0.554 | 0.502 | 0.043 | 0.290 | 0.055 | 0.654 |
| vit_b16 | full | random | energy_T1 | 0.137 | 0.094 | 1.067 | 0.003 | 0.134 | 0.000 | 1.000 | 0.000 |
| vit_b16 | full | random | msp_T1 | 0.440 | 0.062 | 1.766 | 0.096 | 0.354 | 0.000 | 1.000 | 0.000 |
| vit_b16 | pca64 | pca | knn | 1.000 | 0.075 | 0.676 | 1.000 | 0.026 | 0.934 | 0.031 | 0.035 |
| vit_b16 | pca64 | pca | mahalanobis | 1.000 | 0.077 | 0.715 | 1.000 | 0.025 | 0.915 | 0.037 | 0.048 |
| vit_b16 | pca64 | pca | energy_T1 | 0.146 | 0.062 | 0.327 | 0.000 | 0.146 | 0.000 | 1.000 | 0.000 |
| vit_b16 | pca64 | pca | msp_T1 | 0.654 | 0.062 | 0.780 | 0.142 | 0.566 | 0.000 | 1.000 | 0.000 |
| vit_b16 | pca64 | random | knn | 1.000 | 0.075 | 0.625 | 1.000 | 0.007 | 0.350 | 0.503 | 0.148 |
| vit_b16 | pca64 | random | mahalanobis | 1.000 | 0.077 | 0.539 | 1.000 | 0.007 | 0.309 | 0.602 | 0.089 |
| vit_b16 | pca64 | random | energy_T1 | 0.104 | 0.062 | 0.601 | 0.001 | 0.103 | 0.000 | 1.000 | 0.000 |
| vit_b16 | pca64 | random | msp_T1 | 0.664 | 0.062 | 1.371 | 0.084 | 0.604 | 0.000 | 1.000 | 0.000 |

## Steering sensitivity vs static detection

Spearman correlation across model x representation cells, per detector. crossing = fraction of steered ID probes rejected within the horizon; alpha = median crossing distance in ID radii.

| detector | n | rho(crossing_pca,auroc) | rho(crossing_random,auroc) | rho(alpha_radii_pca,auroc) | rho(alpha_radii_random,auroc) |
|---|---|---|---|---|---|
| energy_T1 | 16 | -0.41 (p=0.11) | 0.58 (p=0.019) | 0.63 (p=0.009) | 0.55 (p=0.028) |
| knn | 16 | nan (p=nan) | nan (p=nan) | -0.26 (p=0.33) | 0.09 (p=0.75) |
| mahalanobis | 16 | 0.42 (p=0.1) | -0.06 (p=0.81) | -0.24 (p=0.38) | -0.55 (p=0.026) |
| msp_T1 | 16 | 0.51 (p=0.041) | -0.20 (p=0.46) | 0.75 (p=0.00076) | 0.85 (p=3.5e-05) |

## Steering checks

| model | rep | direction | detector | max_abs_recompute_error | alpha0_matches_unsteered | mean_score_change_over_horizon_in_cal_sd | frac_paths_score_increases | probe_rejection_at_alpha0 | probe_rejection_at_horizon |
|---|---|---|---|---|---|---|---|---|---|
| bge_base | full | pca | knn | 0.000 | 0.000 | 11.491 | 1.000 | 0.052 | 1.000 |
| bge_base | full | pca | mahalanobis | 0.000 | 0.000 | 7.477 | 1.000 | 0.060 | 1.000 |
| bge_base | full | pca | energy_T1 | 0.000 | 0.000 | -5.444 | 0.000 | 0.078 | 0.000 |
| bge_base | full | pca | msp_T1 | 0.000 | 0.000 | 2.577 | 0.897 | 0.047 | 0.542 |
| bge_base | full | random | knn | 0.000 | 0.000 | 13.333 | 1.000 | 0.052 | 1.000 |
| bge_base | full | random | mahalanobis | 0.000 | 0.000 | 2519.555 | 1.000 | 0.060 | 1.000 |
| bge_base | full | random | energy_T1 | 0.000 | 0.000 | -0.211 | 0.445 | 0.078 | 0.070 |
| bge_base | full | random | msp_T1 | 0.000 | 0.000 | 0.954 | 0.766 | 0.047 | 0.219 |
| bge_base | pca64 | pca | knn | 0.000 | 0.000 | 16.970 | 1.000 | 0.081 | 1.000 |
| bge_base | pca64 | pca | mahalanobis | 0.000 | 0.000 | 42.916 | 1.000 | 0.077 | 1.000 |
| bge_base | pca64 | pca | energy_T1 | 0.000 | 0.000 | -2.300 | 0.117 | 0.070 | 0.000 |
| bge_base | pca64 | pca | msp_T1 | 0.000 | 0.000 | 1.788 | 0.792 | 0.102 | 0.502 |
| bge_base | pca64 | random | knn | 0.000 | 0.000 | 18.948 | 1.000 | 0.081 | 1.000 |
| bge_base | pca64 | random | mahalanobis | 0.000 | 0.000 | 68.106 | 1.000 | 0.077 | 1.000 |
| bge_base | pca64 | random | energy_T1 | 0.000 | 0.000 | -4.867 | 0.001 | 0.070 | 0.000 |
| bge_base | pca64 | random | msp_T1 | 0.000 | 0.000 | 0.628 | 0.642 | 0.102 | 0.159 |
| bge_large | full | pca | knn | 0.000 | 0.000 | 11.487 | 1.000 | 0.065 | 1.000 |
| bge_large | full | pca | mahalanobis | 0.000 | 0.000 | 6.195 | 1.000 | 0.062 | 1.000 |
| bge_large | full | pca | energy_T1 | 0.000 | 0.000 | -5.200 | 0.000 | 0.031 | 0.000 |
| bge_large | full | pca | msp_T1 | 0.000 | 0.000 | 2.785 | 0.898 | 0.062 | 0.572 |
| bge_large | full | random | knn | 0.000 | 0.000 | 13.332 | 1.000 | 0.065 | 1.000 |
| bge_large | full | random | mahalanobis | 0.000 | 0.000 | 2478.719 | 1.000 | 0.062 | 1.000 |
| bge_large | full | random | energy_T1 | 0.000 | 0.000 | -0.129 | 0.456 | 0.031 | 0.060 |
| bge_large | full | random | msp_T1 | 0.000 | 0.000 | 0.812 | 0.758 | 0.062 | 0.181 |
| bge_large | pca64 | pca | knn | 0.000 | 0.000 | 17.064 | 1.000 | 0.096 | 1.000 |
| bge_large | pca64 | pca | mahalanobis | 0.000 | 0.000 | 44.525 | 1.000 | 0.073 | 1.000 |
| bge_large | pca64 | pca | energy_T1 | 0.000 | 0.000 | -2.281 | 0.128 | 0.070 | 0.000 |
| bge_large | pca64 | pca | msp_T1 | 0.000 | 0.000 | 2.165 | 0.796 | 0.078 | 0.556 |
| bge_large | pca64 | random | knn | 0.000 | 0.000 | 19.114 | 1.000 | 0.096 | 1.000 |
| bge_large | pca64 | random | mahalanobis | 0.000 | 0.000 | 73.611 | 1.000 | 0.073 | 1.000 |
| bge_large | pca64 | random | energy_T1 | 0.000 | 0.000 | -4.805 | 0.000 | 0.070 | 0.000 |
| bge_large | pca64 | random | msp_T1 | 0.000 | 0.000 | 1.060 | 0.726 | 0.078 | 0.241 |
| dinov2_s | full | pca | knn | 0.000 | 0.000 | 18.734 | 1.000 | 0.031 | 1.000 |
| dinov2_s | full | pca | mahalanobis | 0.000 | 0.000 | 30.064 | 1.000 | 0.023 | 1.000 |
| dinov2_s | full | pca | energy_T1 | 0.000 | 0.000 | -8.693 | 0.000 | 0.094 | 0.000 |
| dinov2_s | full | pca | msp_T1 | 0.000 | 0.000 | -0.016 | 0.236 | 0.062 | 0.073 |
| dinov2_s | full | random | knn | 0.000 | 0.000 | 21.573 | 1.000 | 0.031 | 1.000 |
| dinov2_s | full | random | mahalanobis | 0.000 | 0.000 | 961876.056 | 0.333 | 0.023 | 0.333 |
| dinov2_s | full | random | energy_T1 | 0.000 | 0.000 | -0.546 | 0.368 | 0.094 | 0.036 |
| dinov2_s | full | random | msp_T1 | 0.000 | 0.000 | 0.629 | 0.595 | 0.062 | 0.242 |
| dinov2_s | pca64 | pca | knn | 0.000 | 0.000 | 20.362 | 1.000 | 0.050 | 1.000 |
| dinov2_s | pca64 | pca | mahalanobis | 0.000 | 0.000 | 80.922 | 1.000 | 0.039 | 1.000 |
| dinov2_s | pca64 | pca | energy_T1 | 0.000 | 0.000 | -4.768 | 0.002 | 0.039 | 0.000 |
| dinov2_s | pca64 | pca | msp_T1 | 0.000 | 0.000 | 0.262 | 0.524 | 0.078 | 0.146 |
| dinov2_s | pca64 | random | knn | 0.000 | 0.000 | 22.902 | 1.000 | 0.050 | 1.000 |
| dinov2_s | pca64 | random | mahalanobis | 0.000 | 0.000 | 61.825 | 1.000 | 0.039 | 1.000 |
| dinov2_s | pca64 | random | energy_T1 | 0.000 | 0.000 | -2.591 | 0.064 | 0.039 | 0.000 |
| dinov2_s | pca64 | random | msp_T1 | 0.000 | 0.000 | 0.198 | 0.523 | 0.078 | 0.136 |
| minilm | full | pca | knn | 0.000 | 0.000 | 10.532 | 1.000 | 0.078 | 1.000 |
| minilm | full | pca | mahalanobis | 0.000 | 0.000 | 10.794 | 1.000 | 0.049 | 1.000 |
| minilm | full | pca | energy_T1 | 0.000 | 0.000 | -4.909 | 0.000 | 0.062 | 0.000 |
| minilm | full | pca | msp_T1 | 0.000 | 0.000 | 2.037 | 0.871 | 0.062 | 0.438 |
| minilm | full | random | knn | 0.000 | 0.000 | 12.095 | 1.000 | 0.078 | 1.000 |
| minilm | full | random | mahalanobis | 0.000 | 0.000 | 581508.662 | 1.000 | 0.049 | 1.000 |
| minilm | full | random | energy_T1 | 0.000 | 0.000 | -0.770 | 0.305 | 0.062 | 0.001 |
| minilm | full | random | msp_T1 | 0.000 | 0.000 | 1.407 | 0.788 | 0.062 | 0.303 |
| minilm | pca64 | pca | knn | 0.000 | 0.000 | 15.733 | 1.000 | 0.066 | 1.000 |
| minilm | pca64 | pca | mahalanobis | 0.000 | 0.000 | 39.602 | 1.000 | 0.053 | 1.000 |
| minilm | pca64 | pca | energy_T1 | 0.000 | 0.000 | -2.574 | 0.053 | 0.062 | 0.000 |
| minilm | pca64 | pca | msp_T1 | 0.000 | 0.000 | 1.429 | 0.794 | 0.086 | 0.373 |
| minilm | pca64 | random | knn | 0.000 | 0.000 | 17.562 | 1.000 | 0.066 | 1.000 |
| minilm | pca64 | random | mahalanobis | 0.000 | 0.000 | 60.727 | 1.000 | 0.053 | 1.000 |
| minilm | pca64 | random | energy_T1 | 0.000 | 0.000 | -4.381 | 0.000 | 0.062 | 0.000 |
| minilm | pca64 | random | msp_T1 | 0.000 | 0.000 | 0.613 | 0.642 | 0.086 | 0.130 |
| mpnet | full | pca | knn | 0.000 | 0.000 | 10.830 | 1.000 | 0.055 | 1.000 |
| mpnet | full | pca | mahalanobis | 0.000 | 0.000 | 6.917 | 1.000 | 0.065 | 1.000 |
| mpnet | full | pca | energy_T1 | 0.000 | 0.000 | -4.565 | 0.005 | 0.062 | 0.000 |
| mpnet | full | pca | msp_T1 | 0.000 | 0.000 | 1.662 | 0.870 | 0.031 | 0.322 |
| mpnet | full | random | knn | 0.000 | 0.000 | 12.371 | 1.000 | 0.055 | 1.000 |
| mpnet | full | random | mahalanobis | 0.000 | 0.000 | 131139.739 | 1.000 | 0.065 | 1.000 |
| mpnet | full | random | energy_T1 | 0.000 | 0.000 | -0.318 | 0.414 | 0.062 | 0.038 |
| mpnet | full | random | msp_T1 | 0.000 | 0.000 | 0.979 | 0.728 | 0.031 | 0.210 |
| mpnet | pca64 | pca | knn | 0.000 | 0.000 | 16.348 | 1.000 | 0.056 | 1.000 |
| mpnet | pca64 | pca | mahalanobis | 0.000 | 0.000 | 44.567 | 1.000 | 0.056 | 1.000 |
| mpnet | pca64 | pca | energy_T1 | 0.000 | 0.000 | -2.295 | 0.089 | 0.078 | 0.000 |
| mpnet | pca64 | pca | msp_T1 | 0.000 | 0.000 | 1.746 | 0.817 | 0.094 | 0.441 |
| mpnet | pca64 | random | knn | 0.000 | 0.000 | 18.076 | 1.000 | 0.056 | 1.000 |
| mpnet | pca64 | random | mahalanobis | 0.000 | 0.000 | 65.054 | 1.000 | 0.056 | 1.000 |
| mpnet | pca64 | random | energy_T1 | 0.000 | 0.000 | -4.530 | 0.002 | 0.078 | 0.000 |
| mpnet | pca64 | random | msp_T1 | 0.000 | 0.000 | 0.893 | 0.696 | 0.094 | 0.181 |
| resnet18 | full | pca | knn | 0.000 | 0.000 | 13.374 | 1.000 | 0.044 | 1.000 |
| resnet18 | full | pca | mahalanobis | 0.000 | 0.000 | 2.764 | 1.000 | 0.044 | 0.788 |
| resnet18 | full | pca | energy_T1 | 0.000 | 0.000 | -7.933 | 0.000 | 0.047 | 0.000 |
| resnet18 | full | pca | msp_T1 | 0.000 | 0.000 | -0.036 | 0.466 | 0.031 | 0.036 |
| resnet18 | full | random | knn | 0.000 | 0.000 | 17.451 | 1.000 | 0.044 | 1.000 |
| resnet18 | full | random | mahalanobis | 0.000 | 0.000 | 171.158 | 1.000 | 0.044 | 1.000 |
| resnet18 | full | random | energy_T1 | 0.000 | 0.000 | -2.705 | 0.052 | 0.047 | 0.000 |
| resnet18 | full | random | msp_T1 | 0.000 | 0.000 | -0.285 | 0.384 | 0.031 | 0.022 |
| resnet18 | pca64 | pca | knn | 0.000 | 0.000 | 13.250 | 1.000 | 0.060 | 1.000 |
| resnet18 | pca64 | pca | mahalanobis | 0.000 | 0.000 | 10.849 | 1.000 | 0.032 | 1.000 |
| resnet18 | pca64 | pca | energy_T1 | 0.000 | 0.000 | -6.768 | 0.000 | 0.031 | 0.000 |
| resnet18 | pca64 | pca | msp_T1 | 0.000 | 0.000 | -0.282 | 0.390 | 0.047 | 0.012 |
| resnet18 | pca64 | random | knn | 0.000 | 0.000 | 16.989 | 1.000 | 0.060 | 1.000 |
| resnet18 | pca64 | random | mahalanobis | 0.000 | 0.000 | 48.737 | 1.000 | 0.032 | 1.000 |
| resnet18 | pca64 | random | energy_T1 | 0.000 | 0.000 | -2.321 | 0.065 | 0.031 | 0.000 |
| resnet18 | pca64 | random | msp_T1 | 0.000 | 0.000 | -0.338 | 0.369 | 0.047 | 0.014 |
| resnet50 | full | pca | knn | 0.000 | 0.000 | 11.162 | 1.000 | 0.039 | 1.000 |
| resnet50 | full | pca | mahalanobis | 0.000 | 0.000 | 0.064 | 0.532 | 0.036 | 0.139 |
| resnet50 | full | pca | energy_T1 | 0.000 | 0.000 | -7.195 | 0.000 | 0.031 | 0.000 |
| resnet50 | full | pca | msp_T1 | 0.000 | 0.000 | 0.421 | 0.634 | 0.047 | 0.092 |
| resnet50 | full | random | knn | 0.000 | 0.000 | 14.045 | 1.000 | 0.039 | 1.000 |
| resnet50 | full | random | mahalanobis | 0.000 | 0.000 | -3.112 | 0.579 | 0.036 | 0.570 |
| resnet50 | full | random | energy_T1 | 0.000 | 0.000 | -1.489 | 0.195 | 0.031 | 0.009 |
| resnet50 | full | random | msp_T1 | 0.000 | 0.000 | 0.159 | 0.507 | 0.047 | 0.073 |
| resnet50 | pca64 | pca | knn | 0.000 | 0.000 | 11.814 | 1.000 | 0.045 | 1.000 |
| resnet50 | pca64 | pca | mahalanobis | 0.000 | 0.000 | 14.850 | 1.000 | 0.066 | 1.000 |
| resnet50 | pca64 | pca | energy_T1 | 0.000 | 0.000 | -4.925 | 0.007 | 0.039 | 0.000 |
| resnet50 | pca64 | pca | msp_T1 | 0.000 | 0.000 | 0.643 | 0.648 | 0.055 | 0.123 |
| resnet50 | pca64 | random | knn | 0.000 | 0.000 | 14.757 | 1.000 | 0.045 | 1.000 |
| resnet50 | pca64 | random | mahalanobis | 0.000 | 0.000 | 57.939 | 1.000 | 0.066 | 1.000 |
| resnet50 | pca64 | random | energy_T1 | 0.000 | 0.000 | -3.267 | 0.063 | 0.039 | 0.000 |
| resnet50 | pca64 | random | msp_T1 | 0.000 | 0.000 | -0.067 | 0.427 | 0.055 | 0.041 |
| vit_b16 | full | pca | knn | 0.000 | 0.000 | 10.978 | 1.000 | 0.073 | 1.000 |
| vit_b16 | full | pca | mahalanobis | 0.000 | 0.000 | -22.589 | 0.313 | 0.036 | 0.136 |
| vit_b16 | full | pca | energy_T1 | 0.000 | 0.000 | -5.511 | 0.000 | 0.094 | 0.000 |
| vit_b16 | full | pca | msp_T1 | 0.000 | 0.000 | 0.650 | 0.660 | 0.062 | 0.167 |
| vit_b16 | full | random | knn | 0.000 | 0.000 | 13.240 | 1.000 | 0.073 | 1.000 |
| vit_b16 | full | random | mahalanobis | 0.000 | 0.000 | 363.115 | 0.541 | 0.036 | 0.502 |
| vit_b16 | full | random | energy_T1 | 0.000 | 0.000 | -0.669 | 0.302 | 0.094 | 0.003 |
| vit_b16 | full | random | msp_T1 | 0.000 | 0.000 | 0.307 | 0.625 | 0.062 | 0.096 |
| vit_b16 | pca64 | pca | knn | 0.000 | 0.000 | 12.243 | 1.000 | 0.075 | 1.000 |
| vit_b16 | pca64 | pca | mahalanobis | 0.000 | 0.000 | 27.481 | 1.000 | 0.077 | 1.000 |
| vit_b16 | pca64 | pca | energy_T1 | 0.000 | 0.000 | -4.364 | 0.000 | 0.062 | 0.000 |
| vit_b16 | pca64 | pca | msp_T1 | 0.000 | 0.000 | 0.616 | 0.669 | 0.062 | 0.142 |
| vit_b16 | pca64 | random | knn | 0.000 | 0.000 | 14.614 | 1.000 | 0.075 | 1.000 |
| vit_b16 | pca64 | random | mahalanobis | 0.000 | 0.000 | 56.514 | 1.000 | 0.077 | 1.000 |
| vit_b16 | pca64 | random | energy_T1 | 0.000 | 0.000 | -1.969 | 0.100 | 0.062 | 0.001 |
| vit_b16 | pca64 | random | msp_T1 | 0.000 | 0.000 | 0.231 | 0.556 | 0.062 | 0.084 |
