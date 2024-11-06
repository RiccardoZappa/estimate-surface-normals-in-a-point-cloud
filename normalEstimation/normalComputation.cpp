#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <vector>
#include <chrono>

// Point structure definition
struct Point {
    float x, y, z;
};

// Camera intrinsics (you will need to fill these in based on your camera)
const float fx = 500.0f;  // Focal length in x direction
const float fy = 500.0f;  // Focal length in y direction
const float cx = 512.0f;  // Principal point (center of the image in x)
const float cy = 512.0f;  // Principal point (center of the image in y)

// Function to convert depth map to a point cloud
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

// Function to save point cloud into a custom Point structure
void savePointCloudToStruct(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<Point>& points) {
    points.resize(cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i) {
        points[i] = {cloud->points[i].x, cloud->points[i].y, cloud->points[i].z};
    }
}

// Function to compute covariance matrix of K neighbors
Eigen::Matrix3f computeCovarianceMatrix(const std::vector<Point>& neighbors) {
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    Eigen::Vector3f centroid(0, 0, 0);

    // Compute centroid
    for (const auto& point : neighbors) {
        centroid += Eigen::Vector3f(point.x, point.y, point.z);
    }
    centroid /= neighbors.size();

    // Compute covariance matrix
    for (const auto& point : neighbors) {
        Eigen::Vector3f p(point.x, point.y, point.z);
        Eigen::Vector3f centered = p - centroid;
        covariance += centered * centered.transpose();
    }
    covariance /= (neighbors.size() - 1);
      // Print covariance matrix
    covariance += Eigen::Matrix3f::Identity() * 1e-6;
    //std::cout << "Covariance Matrix:\n" << covariance << std::endl;
    return covariance;
}

// Function to compute normals from covariance matrices
Eigen::Vector3f computeNormal(const Eigen::Matrix3f& covarianceMatrix) {
    // Perform Eigen decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covarianceMatrix);
    auto eigenvalues = solver.eigenvalues();
    auto eigenvectors = solver.eigenvectors();

    // Find the index of the smallest eigenvalue
    int minIndex;
    eigenvalues.minCoeff(&minIndex);

    // Return the corresponding eigenvector (normal vector)
    //std::cout << eigenvectors.col(minIndex) << std::endl;
    return eigenvectors.col(minIndex);
}

// Main function
int main() {
    // Load the depth map image using OpenCV
    cv::Mat depthMap = cv::imread("/home/riccardozappa/estimate-surface-normals-in-a-point-cloud/normalEstimation/images/depth_map_1m.png", cv::IMREAD_GRAYSCALE);

    if (depthMap.empty()) {
        std::cerr << "Could not load depth map image." << std::endl;
        return -1;
    }
    auto start = std::chrono::high_resolution_clock::now();

    // Convert the depth map to a point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud = depthMapToPointCloud(depthMap);
    int numPoints = pointCloud->size();
    std::cout << "The point cloud has " << numPoints << " points." << std::endl;

    // Save point cloud into Point structure
    std::vector<Point> points, queries;
    savePointCloudToStruct(pointCloud, points);
    queries = points;  // For simplicity, using the same points as queries

    // Use a KD-Tree for nearest neighbor search
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(pointCloud);

    // Parameters
    int K_NEIGHBORS = 10;  // Number of nearest neighbors to consider
    std::vector<Point> normals;

    // Timer start
    std::vector<Point> neighbors_2;
    // Compute normals
    for (int i = 0; i < numPoints; ++i) {
        pcl::PointXYZ searchPoint = pointCloud->points[i];
        std::vector<int> pointIdxNKNSearch(K_NEIGHBORS);
        std::vector<float> pointNKNSquaredDistance(K_NEIGHBORS);

        // Search for nearest neighbors
        int neig = kdtree.nearestKSearch(searchPoint, K_NEIGHBORS, pointIdxNKNSearch, pointNKNSquaredDistance);
        if (neig > 0) {
            std::vector<Point> neighbors;
            //std::cout << "point " << i << "neighbours " << neig << std::endl;
            for (size_t j = 0; j < pointIdxNKNSearch.size(); ++j) {
                const pcl::PointXYZ& pt = pointCloud->points[pointIdxNKNSearch[j]];
                neighbors.push_back({pt.x, pt.y, pt.z});
            }
            // Compute covariance matrix and normal for current point
            Eigen::Matrix3f covarianceMatrix = computeCovarianceMatrix(neighbors);
            Eigen::Vector3f normal = computeNormal(covarianceMatrix);
            neighbors_2 = neighbors;
            //std::cout << "Normal " << i << ": [" << normal(0) << ", " << normal(1) << ", " << normal(1) << "]" << std::endl;
            normals.push_back({normal(0), normal(1), normal(2)});
        }
    }

    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration<float, std::milli>(end - start).count();
    std::cout << "Elapsed time in milliseconds: " << duration << " ms" << std::endl;

    // Print first N normals
    int N_PRINT = 10;
    for (int i = 0; i < std::min(N_PRINT, (int)normals.size()); ++i) {
        std::cout << "Normal " << i << ": [" << normals[i].x << ", " << normals[i].y << ", " << normals[i].z << "]" << std::endl;
    }

    return 0;
}
