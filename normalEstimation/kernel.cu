#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_DIM 3
#define MAX_POINTS 1000000
#define BLOCK_SIZE 256
#define MAX_K 100  // Maximum number of nearest neighbors to find
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

// CUDA kernel for k-nearest neighbors search
__global__ void kNearestNeighborsKernel(Point query, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float dist = 0;
    for (int i = 0; i < MAX_DIM; i++) {
        float diff = d_tree[idx].point.coords[i] - query.coords[i];
        dist += diff * diff;
    }

    // Use atomics to safely update the neighbor list
    int insert_pos = atomicAdd(&d_neighbor_count, 1);
    if (insert_pos < k) {
        d_neighbors[insert_pos].idx = idx;
        d_neighbors[insert_pos].dist = dist;
    } else {
        // Find the maximum distance in the current neighbor list
        float max_dist = d_neighbors[0].dist;
        int max_idx = 0;
        for (int i = 1; i < k; i++) {
            if (d_neighbors[i].dist > max_dist) {
                max_dist = d_neighbors[i].dist;
                max_idx = i;
            }
        }

        // Replace the maximum if the current point is closer
        if (dist < max_dist) {
            atomicExch(&d_neighbors[max_idx].idx, idx);
            atomicExch(&d_neighbors[max_idx].dist, dist);
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

// Host function to perform k-nearest neighbors search
void kNearestNeighbors(Point query, int n, int k, Neighbor* results) {
    Point* d_query;
    cudaMalloc(&d_query, sizeof(Point));
    cudaMemcpy(d_query, &query, sizeof(Point), cudaMemcpyHostToDevice);

    // Initialize d_neighbor_count to 0
    cudaMemset(&d_neighbor_count, 0, sizeof(int));

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kNearestNeighborsKernel<<<numBlocks, BLOCK_SIZE>>>(*d_query, n, k);

    // Copy results back to host
    Neighbor* h_neighbors = (Neighbor*)malloc(k * sizeof(Neighbor));
    cudaMemcpy(h_neighbors, d_neighbors, k * sizeof(Neighbor), cudaMemcpyDeviceToHost);

    // Sort the neighbors by distance
    thrust::sort(thrust::host, h_neighbors, h_neighbors + k, 
                 [](const Neighbor& a, const Neighbor& b) { return a.dist < b.dist; });

    // Copy sorted results
    memcpy(results, h_neighbors, k * sizeof(Neighbor));

    cudaFree(d_query);
    free(h_neighbors);
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


int main() {
    // Example usage
    const int n = 1000000;
    const int k = 5;  // Number of nearest neighbors to find
    Point* h_points = (Point*)malloc(n * sizeof(Point));

    // Initialize points (you should replace this with your actual data)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < MAX_DIM; j++) {
            h_points[i].coords[j] = (float)rand() / RAND_MAX;
        }
    }

    // Build KD-tree
    buildKDTree(h_points, n);

    // Perform k-nearest neighbors search
    Point query = {{0.5, 0.5, 0.5}};
    Neighbor* results = (Neighbor*)malloc(k * sizeof(Neighbor));
    kNearestNeighbors(query, n, k, results);

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