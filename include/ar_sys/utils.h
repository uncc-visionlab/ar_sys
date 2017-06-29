#ifndef UTILS_H
#define UTILS_H

#include <opencv2/aruco.hpp>

//#include "aruco/aruco.h"
#include <sensor_msgs/CameraInfo.h>
#include <tf/transform_datatypes.h>

class ar_sys {
public:

    ar_sys() {
        rotate_to_sys = cv::Mat::zeros(3, 3, CV_64FC1);
        /* Fixed the rotation to meet the ROS system
        /* Doing a basic rotation around X with theta=PI
        /* By Sahloul
        /* See http://en.wikipedia.org/wiki/Rotation_matrix for details
         */
        //	1	0	0
        //	0	-1	0
        //	0	0	-1
        rotate_to_sys.at<float>(0, 0) = 1.0;
        rotate_to_sys.at<float>(0, 1) = 0.0;
        rotate_to_sys.at<float>(0, 2) = 0.0;
        rotate_to_sys.at<float>(1, 0) = 0.0;
        rotate_to_sys.at<float>(1, 1) = -1.0;
        rotate_to_sys.at<float>(1, 2) = 0.0;
        rotate_to_sys.at<float>(2, 0) = 0.0;
        rotate_to_sys.at<float>(2, 1) = 0.0;
        rotate_to_sys.at<float>(2, 2) = -1.0;
    }
    /**
     * @brief getCamParams gets the camera intrinsics from a CameraInfo message and copies them
     *                                     to ar_sys own data structure
     * @param cam_info
     * @param useRectifiedParameters if true, the intrinsics are taken from cam_info.P and the distortion parameters
     *                               are set to 0. Otherwise, cam_info.K and cam_info.D are taken.
     * @return
     */
    void getCamParams(const sensor_msgs::CameraInfo& cam_info, bool useRectifiedParameters);
    /**
     * @brief getTf converts OpenCV coordinates to ROS Transform
     * @param Rvec
     * @param Tvec
     * @return tf::Transform
     */
    tf::Transform getTf(const cv::Vec3d &Rvec, const cv::Vec3d &Tvec);
protected:
    cv::Size image_size;
    cv::Mat cameraMatrix, distortionCoeffs;
    cv::Mat rotate_to_sys;
};
#endif // UTILS_H
