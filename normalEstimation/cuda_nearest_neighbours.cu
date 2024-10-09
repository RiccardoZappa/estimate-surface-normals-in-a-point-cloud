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

#define eChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const int MAX_DIM = 3;   // Maximum dimensions of points (can be changed)
const int N_POINTS = 1e4, N_QUERIES = 1e6, K_NEIGHBORS = 5, INF = 1e9, RANGE_MAX = 100, N_PRINT = 10;

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
    srand(16);

    int TREE_SIZE = 1;
    while (TREE_SIZE < N_POINTS) TREE_SIZE <<= 1;

    Point *points;
    KDNode *tree;
    Point *queries;

    eChk(cudaMallocManaged(&points, N_POINTS * sizeof(Point)));
    eChk(cudaMallocManaged(&tree, TREE_SIZE * sizeof(KDNode)));
    eChk(cudaMallocManaged(&queries, N_QUERIES * sizeof(Point)));

    generatePoints(points, N_POINTS);
    buildKDTree(points, tree, N_POINTS, TREE_SIZE);
    generatePoints(queries, N_QUERIES);

    auto start = std::chrono::system_clock::now();

    Point *results;
    eChk(cudaMallocManaged(&results, N_QUERIES * K_NEIGHBORS * sizeof(Point)));

    kNearestNeighborsGPU<<<32768, 32>>>(tree, TREE_SIZE, queries, results, N_QUERIES, K_NEIGHBORS);
    eChk(cudaDeviceSynchronize());
    
    auto end = std::chrono::system_clock::now();
    float duration = 1000.0 * std::chrono::duration<float>(end - start).count();

    printResults(queries, results, N_QUERIES - N_PRINT - 1, N_QUERIES);

    std::cout << "Elapsed time in milliseconds : " << duration << "ms\n\n";

    eChk(cudaFree(results));
    eChk(cudaFree(points));
    eChk(cudaFree(tree));
    eChk(cudaFree(queries));
}

// Helper function to generate random points in MAX_DIM dimensions
__host__ void generatePoints(Point *points, int n) {
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < MAX_DIM; d++) {
            points[i].coords[d] = static_cast<float>(rand() % RANGE_MAX + 1);
        }
    }
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

// Device function to compute Euclidean distance between two points
__device__ float distance(const Point &p1, const Point &p2) {
    float dist = 0.0f;
    for (int i = 0; i < MAX_DIM; i++) {
        dist += powf(p1.coords[i] - p2.coords[i], 2);
    }
    return sqrtf(dist);
}

// Device function to compare two points for K-nearest neighbor search
struct KNNComparator {
    __device__ bool operator()(const Point &p1, const Point &p2, const Point &query) {
        return distance(p1, query) < distance(p2, query);
    }
};

// Recursive device function for finding K nearest neighbors
__device__ void findKNearestNeighbors(KDNode *tree, int treeSize, int treeNode, int depth, Point query, Point *neighbors, int k) {
    // Base case
    if (treeNode >= treeSize) return;

    KDNode node = tree[treeNode];
    if (node.axis == -1) return;

    // Push the current node point into neighbors array
    for (int i = 0; i < k; i++) {
        if (i == k - 1 || distance(node.point, query) < distance(neighbors[i], query)) {
            neighbors[i] = node.point;
            break;
        }
    }

    // Find the next subtree to search
    int nextAxis = (depth + 1) % MAX_DIM;
    if (query.coords[node.axis] < node.point.coords[node.axis]) {
        findKNearestNeighbors(tree, treeSize, treeNode * 2, depth + 1, query, neighbors, k);
    } else {
        findKNearestNeighbors(tree, treeSize, treeNode * 2 + 1, depth + 1, query, neighbors, k);
    }
}

// Kernel to perform K nearest neighbor search for all queries
__global__ void kNearestNeighborsGPU(KDNode *tree, int treeSize, Point *queries, Point *results, int nQueries, int k) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < nQueries) {
        Point neighbors[K_NEIGHBORS];  // Local array to store the K nearest neighbors
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
