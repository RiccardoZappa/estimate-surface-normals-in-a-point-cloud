#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <vector>

#define MAX_DIM 3
#define MAX_POINTS 1000000
#define BLOCK_SIZE 256
#define MAX_K 100
#define MAX_NEIGHBOURS 100 // Maximum neighbours to calculate with radius search

// Structure to represent a point
struct Point {
    float coords[MAX_DIM];
};

// Structure to represent a KD-tree node
struct KDNode {
    Point point;
    int left;
    int right;
    int axis;
};   

// Structure to represent a neighbor
struct Neighbor {
    int idx;
    float dist;
};

// Global variables
__device__ KDNode d_tree[MAX_POINTS];
__device__ Point d_points[MAX_POINTS];
__device__ Neighbor d_neighbors[MAX_K];
__device__ int d_neighbor_count;

// Function to convert depth map to PCL point cloud
pcl::PointCloud<pcl::PointXYZ>::Ptr depthMapToPointCloud(const cv::Mat& depthMap, const cv::Mat& cameraMatrix) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

    float fx = cameraMatrix.at<float>(0, 0); // focal length x
    float fy = cameraMatrix.at<float>(1, 1); // focal length y
    float cx = cameraMatrix.at<float>(0, 2); // principal point x
    float cy = cameraMatrix.at<float>(1, 2); // principal point y

    for (int y = 0; y < depthMap.rows; y++) {
        for (int x = 0; x < depthMap.cols; x++) {
            float depth = depthMap.at<float>(y, x);

            if (depth > 0) {
                pcl::PointXYZ point;
                point.x = (x - cx) * depth / fx;
                point.y = (y - cy) * depth / fy;
                point.z = depth;
                cloud->points.push_back(point);
            }
        }
    }
    cloud->width = (int)cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

// CUDA kernel for building the KD-tree
__global__ void buildKDTreeKernel(Point* points, int n, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int axis = depth % MAX_DIM;
    int nodeIdx = idx + n - 1;

    d_tree[nodeIdx].point = points[idx];
    d_tree[nodeIdx].axis = axis;
    d_tree[nodeIdx].left = -1;
    d_tree[nodeIdx].right = -1;

    if (idx > 0) {
        int parentIdx = (idx - 1) / 2 + n - 1;
        if (idx % 2 == 1) {
            d_tree[parentIdx].left = nodeIdx;
        } else {
            d_tree[parentIdx].right = nodeIdx;
        }
    }
}

// Host function to build the KD-tree
void buildKDTree(Point* h_points, int numPoints) {
    Point* d_points;
    cudaMalloc(&d_points, numPoints * sizeof(Point));
    cudaMemcpy(d_points, h_points, numPoints * sizeof(Point), cudaMemcpyHostToDevice);

    int numBlocks = (numPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
    buildKDTreeKernel<<<numBlocks, BLOCK_SIZE>>>(d_points, numPoints, 0);

    cudaFree(d_points);
}

// CUDA kernel for radius search
__global__ void radiusSearchKernel(Point query, float radius, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float dist_squared = 0;
    for (int i = 0; i < MAX_DIM; i++) {
        float diff = d_tree[idx].point.coords[i] - query.coords[i];
        dist_squared += diff * diff;
    }

    if (dist_squared <= radius * radius) {
        int insert_pos = atomicAdd(&d_neighbor_count, 1);
        if (insert_pos < MAX_NEIGHBOURS) {
            d_neighbors[insert_pos].idx = idx;
            d_neighbors[insert_pos].dist = sqrtf(dist_squared);
        }
    }
}

// Host function to perform radius search
int radiusSearch(Point query, float radius, int n, Neighbor* results) {
    Point* d_query;
    cudaMalloc(&d_query, sizeof(Point));
    cudaMemcpy(d_query, &query, sizeof(Point), cudaMemcpyHostToDevice);

    // Initialize d_neighbor_count to 0
    cudaMemset(&d_neighbor_count, 0, sizeof(int));

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    radiusSearchKernel<<<numBlocks, BLOCK_SIZE>>>(*d_query, radius, n);

    // Copy neighbor count back to host
    int h_neighbor_count;
    cudaMemcpy(&h_neighbor_count, &d_neighbor_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Clamp neighbor count to MAX_NEIGHBORS
    int result_count = min(h_neighbor_count, MAX_NEIGHBOURS);

    // Copy results back to host
    cudaMemcpy(results, d_neighbors, result_count * sizeof(Neighbor), cudaMemcpyDeviceToHost);

    // Sort the neighbors by distance
    thrust::sort(thrust::host, results, results + result_count, 
                 [](const Neighbor& a, const Neighbor& b) { return a.dist < b.dist; });

    cudaFree(d_query);
    return result_count;
}

// CUDA kernel to convert pcl::PointXYZ to Point
__global__ void convertPointCloudKernel(const pcl::PointXYZ* inputPoints, Point* outputPoints, int numPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        outputPoints[idx].coords[0] = inputPoints[idx].x;  // x coordinate
        outputPoints[idx].coords[1] = inputPoints[idx].y;  // y coordinate
        outputPoints[idx].coords[2] = inputPoints[idx].z;  // z coordinate
    }
}

void savePointCloudToStructCUDA(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Point* points) {
    int numPoints = cloud->size();

    // Allocate host memory for the output points

    // Allocate device memory
    pcl::PointXYZ* deviceInputPoints;
    Point* deviceOutputPoints;

    cudaMalloc((void**)&deviceInputPoints, numPoints * sizeof(pcl::PointXYZ));
    cudaMalloc((void**)&deviceOutputPoints, numPoints * sizeof(Point));

    // Copy input point cloud data to the device
    cudaMemcpy(deviceInputPoints, cloud->points.data(), numPoints * sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice);

    // Define block size and grid size
    int blockSize = 256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    convertPointCloudKernel<<<gridSize, blockSize>>>(deviceInputPoints, deviceOutputPoints, numPoints);

    // Copy the results back to the host
    cudaMemcpy(points, deviceOutputPoints, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(deviceInputPoints);
    cudaFree(deviceOutputPoints);
}

__device__ void radiusSearchKDTree(const Point& query, float radius, KDNode* kdTree, int root_idx, int depth, Neighbor* neighbors, int* neighbor_count) {
    if (root_idx == -1) return;  // Base case for recursion (no node)
    
    // Get the current node
    KDNode node = kdTree[root_idx];
    float dist_squared = 0;
    for (int i = 0; i < MAX_DIM; i++) {
        float diff = node.point.coords[i] - query.coords[i];
        dist_squared += diff * diff;
    }

    // If the current node is within the radius, store it
    if (dist_squared <= radius * radius) {
        int insert_pos = atomicAdd(neighbor_count, 1);
        if (insert_pos < MAX_NEIGHBOURS) {
            neighbors[insert_pos].idx = root_idx;
            neighbors[insert_pos].dist = sqrtf(dist_squared);
        }
    }

    // Recursively traverse the KD-tree
    int axis = node.axis;  // Axis along which this node was split
    float diff_axis = query.coords[axis] - node.point.coords[axis];

    // First, explore the side of the tree that contains the query point
    if (diff_axis <= 0) {
        radiusSearchKDTree(query, radius, kdTree, node.left, depth + 1, neighbors, neighbor_count);
    } else {
        radiusSearchKDTree(query, radius, kdTree, node.right, depth + 1, neighbors, neighbor_count);
    }

    // Check if we need to search the other side of the tree
    if (diff_axis * diff_axis <= radius * radius) {
        if (diff_axis <= 0) {
            radiusSearchKDTree(query, radius, kdTree, node.right, depth + 1, neighbors, neighbor_count);
        } else {
            radiusSearchKDTree(query, radius, kdTree, node.left, depth + 1, neighbors, neighbor_count);
        }
    }
}

__global__ void radiusSearchKDTreeKernel(Point* queryPoints, float radius, int n, KDNode* kdTree, Neighbor* neighbors, int* neighbor_counts) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= n) return;

    Point query = queryPoints[query_idx];
    int neighbor_count = 0;

    // Perform the radius search for this query point using the KD-tree
    radiusSearchKDTree(query, radius, kdTree, 0, 0, neighbors + query_idx * MAX_NEIGHBOURS, &neighbor_count);

    // Store the number of neighbors found for this point
    neighbor_counts[query_idx] = neighbor_count;
}

__device__ void computeCovarianceMatrix(Point* points, Neighbor* neighbors, int neighbor_count, float covariance[3][3]) {
    // Compute centroid
    Point centroid = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < neighbor_count; i++) {
        int neighbor_idx = neighbors[i].idx;
        centroid.coords[0] += points[neighbor_idx].coords[0];
        centroid.coords[1] += points[neighbor_idx].coords[1];
        centroid.coords[2] += points[neighbor_idx].coords[2];
    }
    centroid.coords[0] /= neighbor_count;
    centroid.coords[1] /= neighbor_count;
    centroid.coords[2] /= neighbor_count;

    // Initialize the covariance matrix to zero
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            covariance[i][j] = 0.0f;
        }
    }

    // Compute covariance matrix elements
    for (int i = 0; i < neighbor_count; i++) {
        int neighbor_idx = neighbors[i].idx;
        float dx = points[neighbor_idx].coords[0] - centroid.coords[0];
        float dy = points[neighbor_idx].coords[1] - centroid.coords[1];
        float dz = points[neighbor_idx].coords[2] - centroid.coords[2];

        covariance[0][0] += dx * dx;
        covariance[0][1] += dx * dy;
        covariance[0][2] += dx * dz;

        covariance[1][0] += dy * dx;
        covariance[1][1] += dy * dy;
        covariance[1][2] += dy * dz;

        covariance[2][0] += dz * dx;
        covariance[2][1] += dz * dy;
        covariance[2][2] += dz * dz;
    }

    // Normalize the covariance matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            covariance[i][j] /= neighbor_count;
        }
    }
}

__global__ void computeNormalsKernel(Point* points, Neighbor* neighbors, int* neighbor_counts, int n, float* normals, cusolverDnHandle_t cusolverH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int neighbor_count = neighbor_counts[idx];
    if (neighbor_count < 3) {
        // Not enough neighbors to compute covariance matrix (minimum 3 neighbors required)
        normals[3 * idx + 0] = 0.0f;
        normals[3 * idx + 1] = 0.0f;
        normals[3 * idx + 2] = 0.0f;
        return;
    }

    // Compute the covariance matrix
    float covariance[3][3];
    computeCovarianceMatrix(points, &neighbors[idx * MAX_NEIGHBOURS], neighbor_count, covariance);

    // Flatten the covariance matrix to pass to cuSolver
    float A[9] = {
        covariance[0][0], covariance[0][1], covariance[0][2],
        covariance[1][0], covariance[1][1], covariance[1][2],
        covariance[2][0], covariance[2][1], covariance[2][2]
    };

    // Allocate space for eigenvalues and eigenvectors
    float eigenvalues[3];
    float eigenvectors[9]; // 3x3 matrix

    // Call cuSolver to compute the eigenvalues and eigenvectors
    int lwork = 0;
    float* d_work = nullptr;
    int* devInfo = nullptr;
    
    // Allocate memory for device info
    cudaMalloc(&devInfo, sizeof(int));

    // Query workspace size
    cusolverDnSsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, A, 3, eigenvalues, &lwork);

    // Allocate workspace
    cudaMalloc(&d_work, lwork * sizeof(float));

    // Perform the eigen decomposition
    cusolverDnSsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, A, 3, eigenvalues, d_work, lwork, devInfo);

    // Check if the operation was successful
    int h_devInfo;
    cudaMemcpy(&h_devInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_devInfo != 0) {
        printf("Error: Eigenvalue computation failed with error code %d\n", h_devInfo);
        return;
    }

    // The normal is the eigenvector corresponding to the smallest eigenvalue
    int min_idx = 0;
    if (eigenvalues[1] < eigenvalues[0]) min_idx = 1;
    if (eigenvalues[2] < eigenvalues[min_idx]) min_idx = 2;

    normals[3 * idx + 0] = eigenvectors[0 + min_idx * 3];
    normals[3 * idx + 1] = eigenvectors[1 + min_idx * 3];
    normals[3 * idx + 2] = eigenvectors[2 + min_idx * 3];

    // Free workspace memory
    cudaFree(d_work);
    cudaFree(devInfo);
}

void computeNormals(Point* h_points, Neighbor* h_neighbors, int* h_neighbor_counts, int numPoints) {
    Point* d_points;
    Neighbor* d_neighbors;
    int* d_neighbor_counts;
    float* d_normals;

    // Allocate device memory for points, neighbors, neighbor counts, and normals
    cudaMalloc(&d_points, numPoints * sizeof(Point));
    cudaMalloc(&d_neighbors, numPoints * MAX_NEIGHBOURS * sizeof(Neighbor));
    cudaMalloc(&d_neighbor_counts, numPoints * sizeof(int));
    cudaMalloc(&d_normals, 3 * numPoints * sizeof(float));  // 3 floats per normal (x, y, z)

    // Copy data to device
    cudaMemcpy(d_points, h_points, numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors, h_neighbors, numPoints * MAX_NEIGHBOURS * sizeof(Neighbor), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbor_counts, h_neighbor_counts, numPoints * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize cuSolver
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    // Launch kernel to compute normals
    int numBlocks = (numPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
    computeNormalsKernel<<<numBlocks, BLOCK_SIZE>>>(d_points, d_neighbors, d_neighbor_counts, numPoints, d_normals, cusolverH);

    // Copy normals back to host
    float* h_normals = (float*)malloc(3 * numPoints * sizeof(float));
    cudaMemcpy(h_normals, d_normals, 3 * numPoints * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory and destroy cuSolver handle
    cudaFree(d_points);
    cudaFree(d_neighbors);
    cudaFree(d_neighbor_counts);
    cudaFree(d_normals);
    cusolverDnDestroy(cusolverH);

    // Now h_normals contains the computed normals
}

int main() {
    // Example usage
    const int n = 1000000;
    const int k = 5;  // Number of nearest neighbors to find

    //read the depth map
    cv::Mat depthMap = cv::imread("/home/riccardozappa/esitmate-surface-normals-in-a-point-cloud/normalEstimation/result.png", cv::IMREAD_GRAYSCALE);
    float focalLength = 525.0f;
    int width = depthMap.cols;
    int height = depthMap.rows;
    cv::Mat customCameraMatrix = (cv::Mat_<float>(3, 3) << 
    focalLength, 0, width / 2.0f, 
    0, focalLength, height / 2.0f, 
    0, 0, 1);

    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud;
    //obtain the point cloud 
    pointCloud = depthMapToPointCloud(depthMap, customCameraMatrix);
    int numPoints = pointCloud->size();
    std::cout << "the point cloud has " << numPoints << " Points" << std::endl;
    Point* h_points = (Point*)malloc(numPoints * sizeof(Point));

    savePointCloudToStructCUDA(pointCloud, h_points);

    // Build KD-tree
    buildKDTree(h_points, numPoints);


    Point* d_queryPoints;
    Neighbor* d_neighbors;
    KDNode* d_kdTree;
    int* d_neighbor_counts;

    cudaMalloc(&d_queryPoints, numPoints * sizeof(Point));
    cudaMalloc(&d_neighbors, numPoints * MAX_NEIGHBOURS * sizeof(Neighbor));
    cudaMalloc(&d_kdTree, numPoints * sizeof(KDNode));  // Assuming the KD-tree is stored as an array of nodes
    cudaMalloc(&d_neighbor_counts, numPoints * sizeof(int));

    cudaMemcpy(d_queryPoints, h_points, numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kdTree, kdTree, numPoints * sizeof(KDNode), cudaMemcpyHostToDevice);
    cudaMemset(d_neighbor_counts, 0, numPoints * sizeof(int));

    int numBlocks = (numPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
    radiusSearchKDTreeKernel<<<numBlocks, BLOCK_SIZE>>>(d_queryPoints, radius, numPoints, d_kdTree, d_neighbors, d_neighbor_counts);
    
    cudaMemcpy(h_neighbors, d_neighbors, numPoints * MAX_NEIGHBOURS * sizeof(Neighbor), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_neighbor_counts, d_neighbor_counts, numPoints * sizeof(int), cudaMemcpyDeviceToHost);

    

    cudaFree(d_queryPoints);
    cudaFree(d_neighbors);
    cudaFree(d_kdTree);
    cudaFree(d_neighbor_counts);
    // free(h_points);
    // free(results);
    return 0;
}




