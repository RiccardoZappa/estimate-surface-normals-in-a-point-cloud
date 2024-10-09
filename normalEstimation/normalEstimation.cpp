#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>
#include <thread>

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

int main()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        //read the depth map
    cv::Mat depthMap = cv::imread("/home/riccardozappa/esitmate-surface-normals-in-a-point-cloud/normalEstimation/result.png", cv::IMREAD_GRAYSCALE);
    float focalLength = 525.0f;
    int width = depthMap.cols;
    int height = depthMap.rows;
    cv::Mat customCameraMatrix = (cv::Mat_<float>(3, 3) << 
    focalLength, 0, width / 2.0f, 
    0, focalLength, height / 2.0f, 
    0, 0, 1);

    cloud = depthMapToPointCloud(depthMap, customCameraMatrix);

    int numPoints = cloud->size();
    std::cout << "the point cloud has " << numPoints << " Points" << std::endl;


    auto start = std::chrono::high_resolution_clock::now();
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
    ne.setRadiusSearch (0.03);

    // Compute the features
    ne.compute (*cloud_normals);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    std::cout << "the size of the normals cloud is: " << cloud_normals->size() << std::endl; //should have the same size as the input cloud->size ()*

    return 0;
}
