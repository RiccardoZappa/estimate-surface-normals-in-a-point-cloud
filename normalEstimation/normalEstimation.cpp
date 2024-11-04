#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>
#include <thread>

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

int main()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        //read the depth map
    cv::Mat depthMap = cv::imread("/home/riccardozappa/estimate-surface-normals-in-a-point-cloud/normalEstimation/depth_map_250k.png", cv::IMREAD_GRAYSCALE);

    if (depthMap.empty()) {
        std::cerr << "Could not load depth map image." << std::endl;
        return -1;
    }
    auto start = std::chrono::high_resolution_clock::now();

    cloud = depthMapToPointCloud(depthMap);

    int numPoints = cloud->size();
    std::cout << "the point cloud has " << numPoints << " Points" << std::endl;
   

    // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne.setSearchMethod (tree);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 3cm
    ne.setRadiusSearch (7);
    
    // Compute the features
    ne.compute (*cloud_normals);

    auto end = std::chrono::high_resolution_clock::now();
    float duration = 1000.0 * std::chrono::duration<float>(end - start).count();
    std::cout << "Elapsed time: " << duration << " milliseconds\n";

    std::cout << "the size of the normals cloud is: " << cloud_normals->size() << std::endl; //should have the same size as the input cloud->size ()*
            // Find and display neighbor count for each of the first 10 points
    for (size_t i = 0; i < std::min(cloud_normals->size(), static_cast<size_t>(10)); ++i) {
        std::vector<int> neighbor_indices;
        std::vector<float> neighbor_distances;
        int neighbors_found = tree->radiusSearch(cloud->points[i], 7, neighbor_indices, neighbor_distances);

        std::cout << "Normal " << i << ": [" 
                  << cloud_normals->points[i].normal_x << ", "
                  << cloud_normals->points[i].normal_y << ", "
                  << cloud_normals->points[i].normal_z << "]"
                  << " | Neighbors found: " << neighbors_found << std::endl;
        // Print the neighbors' coordinates
        std::cout << "Neighbors for point " << i << " " << cloud->points[i] << ":" << std::endl;
        for (size_t j = 0; j < neighbor_indices.size(); ++j) {
            const auto& neighbor_point = cloud->points[neighbor_indices[j]];
            std::cout << "    Neighbor " << j << ": ["
                  << neighbor_point.x << ", "
                  << neighbor_point.y << ", "
                  << neighbor_point.z << "] | Distance: " << neighbor_distances[j] << std::endl;
        }
    }
    std::cout << "First 10 normals:" << std::endl;
    for (size_t i = 0; i < std::min(cloud_normals->size(), static_cast<size_t>(10)); ++i) {
        std::cout << "Normal " << i << ": [" 
                  << cloud_normals->points[i].normal_x << ", "
                  << cloud_normals->points[i].normal_y << ", "
                  << cloud_normals->points[i].normal_z << "]" << std::endl;
    }
    return 0;
}
