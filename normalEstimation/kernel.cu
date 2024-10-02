#include <cuda_runtime.h>
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
void buildKDTree(Point* h_points, int n) {
    Point* d_points;
    cudaMalloc(&d_points, n * sizeof(Point));
    cudaMemcpy(d_points, h_points, n * sizeof(Point), cudaMemcpyHostToDevice);

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    buildKDTreeKernel<<<numBlocks, BLOCK_SIZE>>>(d_points, n, 0);

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


int main() {
    // Example usage
    const int n = 1000000;
    const int k = 5;  // Number of nearest neighbors to find

    //read the depth map
    cv::Mat depthMap = cv::imread("result.png", cv::IMREAD_GRAYSCALE);
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

    // Initialize points (you should replace this with your actual data)
    for (int i =  0; i < n; i++) {
        for (int j = 0; j < MAX_DIM; j++) {
            h_points[i].coords[j] = (float)rand() / RAND_MAX;
        }
    }

    // Build KD-tree
    buildKDTree(h_points, n);

    // Perform k-nearest neighbors search
    Point query = {{0.5, 0.5, 0.5}};
    Neighbor* results = (Neighbor*)malloc(k * sizeof(Neighbor));

    printf("%d nearest neighbors to (%.2f, %.2f, %.2f):\n", k,
           query.coords[0], query.coords[1], query.coords[2]);
    for (int i = 0; i < k; i++) {
        Point p = h_points[results[i].idx];
        printf("%d. (%.2f, %.2f, %.2f) - distance: %.4f\n", i+1,
               p.coords[0], p.coords[1], p.coords[2], sqrt(results[i].dist));
    }

    free(h_points);
    free(results);
    return 0;
}