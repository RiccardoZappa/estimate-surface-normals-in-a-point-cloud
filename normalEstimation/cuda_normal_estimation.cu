#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <queue>
#include <cmath>
#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <cusolverDn.h>
#include <iostream>


#define eChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define CUSOLVER_CHECK(call) { cusolverAssert((call), __FILE__, __LINE__); }                                             
inline void cusolverAssert(cusolverStatus_t status, const char *file, int line, bool abort=true) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "CUSOLVER error in file '%s' in line %d: %d\n", file, line, status);
        if (abort) exit(EXIT_FAILURE);
    }
}

const char* imagePath = "/home/riccardozappa/estimate-surface-normals-in-a-point-cloud/normalEstimation/images/depth_map_1m.png";
const int MAX_DIM = 3;   // Maximum dimensions of points (can be changed)
const int BLOCK_SIZE = 512;
const int K_NEIGHBORS = 10, INF = 1e9;

struct Point {
    float coords[MAX_DIM];  // Coordinates in MAX_DIM-dimensional space
};

struct KDNode {
    Point point;
    int left;
    int right;
    int axis;  // Splitting axis (0 to MAX_DIM-1)
};

// Function Prototypes
__host__ void printPoints(Point *points, int n);
__host__ void generatePoints(Point *points, int n);
__host__ void buildKDTree(Point *points, KDNode *tree, int n, int m);
__global__ void kNearestNeighborsGPU(KDNode *tree, int treeSize, Point *queries, Point *results, int nQueries, int k);
__host__ void printResults(Point *queries, Point *results, int start, int end);


// Camera intrinsics (you will need to fill these in based on your camera)
const float fx = 500.0f;  // Focal length in x direction
const float fy = 500.0f;  // Focal length in y direction
const float cx = 512.0f;  // Principal point (center of the image in x)
const float cy = 512.0f;  // Principal point (center of the image in y)

// Convert depth map to point cloud using PCL
pcl::PointCloud<pcl::PointXYZ>::Ptr depthMapToPointCloud(const cv::Mat& depthMap) {
    // Create a new point cloud object
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Loop through each pixel in the depth map
    for (int v = 0; v < depthMap.rows; ++v) {
        for (int u = 0; u < depthMap.cols; ++u) {
            float depthValue = depthMap.at<uchar>(v, u);  // Grayscale depth value
            float Z = depthValue;  // Depth value (this can be scaled as per your needs)

            if (Z > 0) {  // Ignore points with zero or invalid depth
                float X = (u - cx) * Z / fx;
                float Y = (v - cy) * Z / fy;

                // Add the 3D point to the point cloud
                pcl::PointXYZ point;
                point.x = X;
                point.y = Y;
                point.z = Z;
                cloud->points.push_back(point);
            }
        }
    }

    // Set the width and height of the point cloud
    cloud->width = cloud->points.size();
    cloud->height = 1;  // Unordered point cloud (1 row)
    cloud->is_dense = false;

    return cloud;
}


// CUDA kernel to convert pcl::PointXYZ to Point
__global__ void convertPointCloudKernel(const pcl::PointXYZ* inputPoints, Point* outputPoints, Point* outputQueries, int numPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        outputPoints[idx].coords[0] = inputPoints[idx].x;  // x coordinate
        outputPoints[idx].coords[1] = inputPoints[idx].y;  // y coordinate
        outputPoints[idx].coords[2] = inputPoints[idx].z;  // z coordinate
        //queries
        outputQueries[idx].coords[0] = inputPoints[idx].x;  // x coordinate
        outputQueries[idx].coords[1] = inputPoints[idx].y;  // y coordinate
        outputQueries[idx].coords[2] = inputPoints[idx].z;  // z coordinate
    }
}

void savePointCloudToStructCUDA(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Point* points, Point* queries) {
    int numPoints = cloud->size();

    // Allocate host memory for the output points

    // Allocate device memory
    pcl::PointXYZ* deviceInputPoints;
    Point* deviceOutputPoints;
    Point* deviceOutputQueries;

    cudaMalloc((void**)&deviceInputPoints, numPoints * sizeof(pcl::PointXYZ));
    cudaMalloc((void**)&deviceOutputPoints, numPoints * sizeof(Point));
    cudaMalloc((void**)&deviceOutputQueries, numPoints * sizeof(Point));

    // Copy input point cloud data to the device
    cudaMemcpy(deviceInputPoints, cloud->points.data(), numPoints * sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice);

    // Define block size and grid size
    int gridSize = (numPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch the CUDA kernel
    convertPointCloudKernel<<<gridSize, BLOCK_SIZE>>>(deviceInputPoints, deviceOutputPoints, deviceOutputQueries, numPoints);

    // Copy the results back to the host
    cudaMemcpy(points, deviceOutputPoints, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(queries, deviceOutputQueries, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(deviceInputPoints);
    cudaFree(deviceOutputPoints);
    cudaFree(deviceOutputQueries);
}

__global__ void computeNormalsKernel(float* d_covarianceMatrices, float* d_eigenValues, float* d_eigenVectors, Point* d_normals, int nPoints, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nPoints) {
        // Find the eigenvector corresponding to the smallest eigenvalue
        int minIndex = 0;
        for (int j = 1; j < dim; j++) {
            if (d_eigenValues[idx * dim + j] < d_eigenValues[idx * dim + minIndex]) {
                minIndex = j;
            }
        }
        // Copy the normal vector (corresponding to the smallest eigenvalue)
        for (int d = 0; d < dim; d++) {
            d_normals[idx].coords[d] = d_eigenVectors[idx * dim * dim + d + minIndex * dim];
        }
    }
}

__host__ void computeNormalsWithCuSolver(float* covarianceMatrices, Point* normals, int nPoints, int dim) {
    // cuSolver handle
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);

    // Allocate device memory
    float *d_covarianceMatrices, *d_eigenValues, *d_eigenVectors;
    Point *d_normals;
    int *d_info;
    float *d_work;

    eChk(cudaMalloc((void**)&d_covarianceMatrices, nPoints * dim * dim * sizeof(float)));
    eChk(cudaMalloc((void**)&d_eigenValues, nPoints * dim * sizeof(float)));
    eChk(cudaMalloc((void**)&d_eigenVectors, nPoints * dim * dim * sizeof(float)));
    eChk(cudaMalloc((void**)&d_normals, nPoints * sizeof(Point)));
    eChk(cudaMalloc((void**)&d_info, sizeof(int)));

    // Copy data to device
    eChk(cudaMemcpy(d_covarianceMatrices, covarianceMatrices, nPoints * dim * dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Workspace size
    int lda = dim;
    int workSize = 0;
    CUSOLVER_CHECK(cusolverDnSsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 
                                dim, d_covarianceMatrices, lda, d_eigenValues, &workSize));

    cudaMalloc((void**)&d_work, workSize * sizeof(float));

    // Compute eigenvalues and eigenvectors for all matrices:the eigenvectors will be computed in place in the d_covarianceMatrices variable
    CUSOLVER_CHECK(cusolverDnSsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                     dim, d_covarianceMatrices, lda, d_eigenValues, d_work, workSize, d_info));

    // Check if the operation was successful
    int info;
    eChk(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Launch kernel to compute normals
    int blocks = (nPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;

    computeNormalsKernel<<<blocks, BLOCK_SIZE>>>(d_covarianceMatrices, d_eigenValues, d_covarianceMatrices, d_normals, nPoints, dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    // Copy results back to host
    cudaMemcpy(normals, d_normals, nPoints * sizeof(Point), cudaMemcpyDeviceToHost);

    // Cleanup
    cusolverDnDestroy(cusolverH);
    cudaFree(d_covarianceMatrices);
    cudaFree(d_eigenValues);
    cudaFree(d_eigenVectors);
    cudaFree(d_normals);
    cudaFree(d_info);
    cudaFree(d_work);
}

__global__ void computeCovarianceMatrix(Point* points, Point* neighbors, float* covarianceMatrices, int numPoints, int kN) {
    int pointIdx = blockIdx.x;  // Each block processes one point
    int tid = threadIdx.x;      // Thread ID within the warp

    if (pointIdx >= numPoints || tid >= kN) return;  // Out-of-bounds protection

    // Step 1: Calculate mean (parallelized by threads within the warp)
    __shared__ float mean[3];
    if (tid < 3) mean[tid] = 0.0f;
    __syncthreads();

    // Each thread handles one neighbor
    Point neighbor = neighbors[pointIdx * kN + tid];

    // Accumulate each neighbor's coordinates to the mean
    atomicAdd(&mean[0], neighbor.coords[0]);
    atomicAdd(&mean[1], neighbor.coords[1]);
    atomicAdd(&mean[2], neighbor.coords[2]);

    __syncthreads();

    // Compute the mean for this point (only the first 3 threads handle this)
    if (tid < 3) {
        mean[tid] /= kN;
    }
    __syncthreads();

    // Step 2: Calculate the covariance matrix (each thread handles one neighbor)
    __shared__ float cov[3][3];
    if (tid < 9) {  // Initialize the shared covariance matrix
        cov[tid / 3][tid % 3] = 0.0f;
    }
    __syncthreads();

    // Difference from the mean
    float diff[3] = {
        neighbor.coords[0] - mean[0],
        neighbor.coords[1] - mean[1],
        neighbor.coords[2] - mean[2]
    };

    // Accumulate to the covariance matrix in parallel
    atomicAdd(&cov[0][0], diff[0] * diff[0]);
    atomicAdd(&cov[0][1], diff[0] * diff[1]);
    atomicAdd(&cov[0][2], diff[0] * diff[2]);
    atomicAdd(&cov[1][0], diff[1] * diff[0]);
    atomicAdd(&cov[1][1], diff[1] * diff[1]);
    atomicAdd(&cov[1][2], diff[1] * diff[2]);
    atomicAdd(&cov[2][0], diff[2] * diff[0]);
    atomicAdd(&cov[2][1], diff[2] * diff[1]);
    atomicAdd(&cov[2][2], diff[2] * diff[2]);

    __syncthreads();

    // Step 3: Normalize and store the covariance matrix
    if (tid < 9) {  // Each thread stores one element of the covariance matrix
        int i = tid / 3;
        int j = tid % 3;
        covarianceMatrices[pointIdx * 9 + i * 3 + j] = cov[i][j] / (kN - 1);

        // Add small value to diagonal for numerical stability
        if (i == j) covarianceMatrices[pointIdx * 9 + i * 3 + j] += 1e-6f;
    }
}

int main() {

        // Load the depth map image using OpenCV
    cv::Mat depthMap = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    if (depthMap.empty()) {
        std::cerr << "Could not load depth map image." << std::endl;
        return -1;
    }
    auto start = std::chrono::high_resolution_clock::now();

    // Convert the depth map to a point cloud using PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud = depthMapToPointCloud(depthMap);
    auto numPoints = pointCloud->size();
    Point* h_points = (Point*)malloc(numPoints * sizeof(Point));
    Point* queries;

    eChk(cudaMallocManaged(&queries, numPoints * sizeof(Point)));

    savePointCloudToStructCUDA(pointCloud, h_points, queries);

    int TREE_SIZE = 1;
    while (TREE_SIZE < numPoints) TREE_SIZE <<= 1;
    
    KDNode *tree;
    
    eChk(cudaMallocManaged(&tree, TREE_SIZE * sizeof(KDNode)));

    buildKDTree(h_points, tree, numPoints, TREE_SIZE);
    int numBlocks = (numPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    size_t newStackSize = 16 * 1024;  // 16 KB to start
    cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);

    Point *results;
    eChk(cudaMallocManaged(&results, numPoints * K_NEIGHBORS * sizeof(Point)));

    std::cout << "serarching for k nearest neighbours"<< std::endl;
    kNearestNeighborsGPU<<<numBlocks, BLOCK_SIZE>>>(tree, TREE_SIZE, queries, results, numPoints, K_NEIGHBORS);
    eChk(cudaDeviceSynchronize());

    float* covarianceMatrices;
    Point* normals;

    // Allocate memory
    eChk(cudaMallocManaged(&covarianceMatrices, numPoints * MAX_DIM * MAX_DIM * sizeof(float)));
    eChk(cudaMallocManaged(&normals, numPoints * sizeof(Point)));

    // Step 1: Compute covariance matrices
    computeCovarianceMatrix<<<numPoints , 32>>>(queries, results, covarianceMatrices, numPoints, K_NEIGHBORS);
    eChk(cudaDeviceSynchronize());

    computeNormalsWithCuSolver(covarianceMatrices, normals, numPoints, MAX_DIM);

    auto end = std::chrono::system_clock::now();

    float duration = 1000.0 * std::chrono::duration<float>(end - start).count();

    std::cout << "Elapsed time in milliseconds full: " << duration << "ms\n\n";

    printPoints(normals, 1);

    eChk(cudaFree(results));
    eChk(cudaFree(tree));
    eChk(cudaFree(queries));
    eChk(cudaFree(covarianceMatrices));
    eChk(cudaFree(normals));
    free(h_points);
    
    return 0;
}

// Comparator for sorting points based on the current axis
struct PointComparator {
    int axis;
    PointComparator(int ax) : axis(ax) {}

    bool operator()(const Point &p1, const Point &p2) {
        return p1.coords[axis] < p2.coords[axis];
    }
};

// Recursive function to build KDTree
__host__ void buildSubTree(Point *points, KDNode *tree, int start, int end, int depth, int node) {
    if (start >= end) return;

    int axis = depth % MAX_DIM;
    std::sort(points + start, points + end, PointComparator(axis));

    int split = (start + end - 1) / 2;
    tree[node].point = points[split];
    tree[node].axis = axis;

    buildSubTree(points, tree, start, split, depth + 1, node * 2);
    buildSubTree(points, tree, split + 1, end, depth + 1, node * 2 + 1);
}

// Function to initialize the KDTree
__host__ void buildKDTree(Point *points, KDNode *tree, int n, int treeSize) {
    for (int i = 0; i < treeSize; i++) {
        tree[i].left = tree[i].right = -1;  // Default values
    }
    buildSubTree(points, tree, 0, n, 0, 1);
}

__device__ float distance(const Point &p1, const Point &p2) {
    float dist = 0.0f;
    for (int i = 0; i < MAX_DIM; i++) {
        dist += powf(p1.coords[i] - p2.coords[i], 2);
    }
    return sqrtf(dist);
}

// Recursive device function for finding K nearest neighbors
 __device__ void findKNearestNeighbors(KDNode *tree, int treeSize, int treeNode, int depth, Point query, Point *neighbors, int k) {
    // Base case
    if (treeNode >= treeSize) return;
    KDNode node = tree[treeNode];
    if (node.axis == -1) return;

    bool near = false;
    int max = 0;
    float dist = 0;
    int index = 0;
    // Push the current node point into neighbors array
    for (int i = 0; i < k; i++) {
        if ( distance(node.point, query) < distance(neighbors[i], query) ) {
            near = true;
            break;
        }
    }
    if (near)
    {
        for (int j=0; j < k; j++)
        {  
            dist = distance(neighbors[j], query);
            if (max < dist)
            {
                max = dist;
                index = j;
            }
        }
        neighbors[index] = node.point;
    }
     // Find the next subtree to search
    int nextAxis = (depth + 1) % MAX_DIM;
    if (query.coords[node.axis] < node.point.coords[node.axis]) {
         findKNearestNeighbors(tree, treeSize, treeNode * 2, nextAxis, query, neighbors, k);
    } else {
         findKNearestNeighbors(tree, treeSize, treeNode * 2 + 1, nextAxis, query, neighbors, k);
    }
 }



// Kernel to perform K nearest neighbor search for all queries
__global__ void kNearestNeighborsGPU(KDNode *tree, int treeSize, Point *queries, Point *results, int nQueries, int k) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < nQueries) {
        Point neighbors[K_NEIGHBORS];  // Local array to store the K nearest neighbors

        for (int i = 0; i < k; i++) {
            neighbors[i].coords[0] = INF;
            neighbors[i].coords[1] = INF;
            neighbors[i].coords[2] = INF;  // Assuming Point() initializes it to an invalid point
        }

        findKNearestNeighbors(tree, treeSize, 1, 0, queries[index], neighbors, k);
        // Copy the neighbors to the results array
        for (int i = 0; i < k; i++) {
            results[index * k + i] = neighbors[i];
            
        }
    }
} 
// Print a list of points
void printPoints(Point *points, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << "[";
        for (int d = 0; d < MAX_DIM; d++) {
            std::cout << points[i].coords[d];
            if (d < MAX_DIM - 1) std::cout << ", ";
        }
        std::cout << "] ";
    }
    std::cout << std::endl;
}

// Print query results
__host__ void printResults(Point *queries, Point *results, int start, int end) {
    for (int i = start; i < end; i++) {
        std::cout << "Query: [";
        for (int d = 0; d < MAX_DIM; d++) {
            std::cout << queries[i].coords[d];
            if (d < MAX_DIM - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        for (int j = 0; j < K_NEIGHBORS; j++) {
            std::cout << "\tNeighbor " << j+1 << ": [";
            for (int d = 0; d < MAX_DIM; d++) {
                std::cout << results[i * K_NEIGHBORS + j].coords[d];
                if (d < MAX_DIM - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
        std::cout << std::endl;
    }
}
