# Project: Estimate Normals from a Point Cloud using CUDA

## Overview

This project presents a CUDA-optimized algorithm to estimate surface normals from a point cloud derived from 2D images. The method is tailored to exploit parallel computing, achieving substantial performance gains compared to traditional implementations. The main contributions include transforming images into point clouds, efficient neighbor searches with KD trees, and CUDA-based computations for surface normal estimation.

---

## Key Features

1. **Point Cloud Generation**
   - Utilizes OpenCV for image handling and transformation.
   - Converts images to point clouds with CUDA kernels that map 2D image coordinates and depth values to 3D space.

2. **KD Tree for Efficient k-Nearest Neighbors Search**
   - Constructs a KD tree to optimize the search process for neighboring points.
   - Implements recursive splitting for efficient spatial organization.

3. **Covariance Matrix Calculation**
   - Computes the local geometric structure for each point using its neighbors.
   - Utilizes CUDA to perform parallel calculations of the covariance matrix.

4. **Normal Estimation**
   - Extracts surface normals by solving eigenvalue and eigenvector problems on the covariance matrices.
   - Chooses the eigenvector associated with the smallest eigenvalue to determine the normal.

---

## Implementation Details

- **Programming Languages**: C++, CUDA
- **Libraries Used**: OpenCV, Point Cloud Library (PCL), NVIDIA CUDA Toolkit
- **GPU Utilization**: The project leverages an NVIDIA GeForce RTX 4060 GPU with CUDA architecture, exploiting its cores for substantial speedup.

### GPU Specifications

| **Field**                     | **Value**                      |
|-------------------------------|--------------------------------|
| Device Name                   | NVIDIA GeForce RTX 4060 Laptop GPU |
| CUDA Cores                    | 3072                           |
| Global Memory                 | 8188 MB                        |
| CUDA Capability Version       | 8.9                            |

---

## Performance

- **Comparative Execution Times**: The CUDA-based implementation shows significant performance improvements over raw C++ and PCL-based methods.
- **Speedup Achieved**: Up to 10.59x over raw C++ and 14.01x over PCL for large point clouds.

| **Point Cloud Size** | **CUDA** | **Raw C++** | **PCL Library** |
|----------------------|----------|-------------|------------------|
| 100k Points          | 1020.34 ms | 6704.6 ms   | 1879.78 ms       |
| 250k Points          | 1715.35 ms | 16398.7 ms  | 6323.51 ms       |
| 1M Points            | 4261.12 ms | 45152.9 ms  | 59732.6 ms       |

---

## Improvements & Future Work

1. **Parallel KD Tree Construction**: The current KD tree construction is sequential. Future work can leverage GPU-based KD tree building as seen in NVIDIA's CUDA KD Tree project.
2. **Optimized k-Nearest Neighbors Search**: The recursive nearest neighbor search could be made more efficient by adopting other CUDA-based implementations.

### References

- [CUDA KD Tree Implementation by NVIDIA](https://github.com/ingowald/cudaKDTree)
- [kNN CUDA Project](https://vincentfpgarcia.github.io/kNN-CUDA/)
- [Point Cloud Library - Normal Estimation](https://pcl.readthedocs.io/projects/tutorials/en/latest/normal_estimation.html)
