/**
 * @file utils.cpp
 * @author Bence Magyar
 * @date June 2012
 * @version 0.1
 * @brief ROS2ArUco utilities.
 */

#include <ros/console.h>
#include <iostream>
#include <tf/transform_datatypes.h>
#include <opencv2/calib3d/calib3d.hpp>

#include <ar_sys/utils.h>

void ar_sys::getCamParams(const sensor_msgs::CameraInfo& cam_info,
        bool useRectifiedParameters) {
    cameraMatrix = cv::Mat::zeros(3, 3, CV_32FC1);
    distortionCoeffs = cv::Mat::zeros(5, 1, CV_32FC1);
    image_size = cv::Size(cam_info.height, cam_info.width);
    //std::cout << "Setting camera_info " << cam_info << std::endl;
    if (useRectifiedParameters) {
        cameraMatrix.at<float>(0, 0) = cam_info.P[0];
        cameraMatrix.at<float>(0, 1) = cam_info.P[1];
        cameraMatrix.at<float>(0, 2) = cam_info.P[2];
        cameraMatrix.at<float>(1, 0) = cam_info.P[4];
        cameraMatrix.at<float>(1, 1) = cam_info.P[5];
        cameraMatrix.at<float>(1, 2) = cam_info.P[6];
        cameraMatrix.at<float>(2, 0) = cam_info.P[8];
        cameraMatrix.at<float>(2, 1) = cam_info.P[9];
        cameraMatrix.at<float>(2, 2) = cam_info.P[10];

        for (int i = 0; i < 5; ++i)
            distortionCoeffs.at<float>(i, 0) = 0;
    } else {
        for (int i = 0; i < 9; ++i) {
            //std::cout << cam_info.K[i] <<" " << std::endl;
            cameraMatrix.at<float>(i / 3, i % 3) = cam_info.K[i];
        }
        for (int i = 0; i < 5; ++i) {
            //std::cout << cam_info.D[i] <<" " << std::endl;
            distortionCoeffs.at<float>(i, 0) = cam_info.D[i];
        }
    }
}

tf::Transform ar_sys::getTf(const cv::Vec3d &rvec, const cv::Vec3d &tvec) {
    //cv::Mat rot(3, 3, CV_32FC1);
    cv::Mat rot;
    cv::Rodrigues(rvec, rot);
    //std::cout << "rtype " << rot.type() << " rsys " << rotate_to_sys.type() << std::endl;
    std::cout << "rvec " << rvec << " tvec " << tvec << std::endl; //<< " rot " << rot <<std::endl;
    rot = rot * rotate_to_sys;
    //rot = rot * rotate_to_sys.t();
    tf::Matrix3x3 tf_rot(rot.at<float>(0, 0), rot.at<float>(0, 1), rot.at<float>(0, 2),
            rot.at<float>(1, 0), rot.at<float>(1, 1), rot.at<float>(1, 2),
            rot.at<float>(2, 0), rot.at<float>(2, 1), rot.at<float>(2, 2));
    tf::Quaternion quat;
    tf_rot.getRotation(quat);
    quat.normalize();
    tf_rot.setRotation(quat);
    tf::Vector3 tf_orig(tvec[0], tvec[1], tvec[2]);

    return tf::Transform(tf_rot, tf_orig);
}
