/**
 * @file single_board.cpp
 * @author Hamdi Sahloul
 * @date September 2014
 * @version 0.1
 * @brief ROS version of the example named "simple_board" in the Aruco software package.
 */

#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#include <opencv2/aruco.hpp>

#include <ar_sys/utils.h>

class ArSysSingleBoard : public ar_sys {
private:
    cv::Mat inImage, resultImg;
    //aruco::CameraParameters camParam;
    bool useRectifiedImages;
    bool draw_markers;
    bool draw_markers_cube;
    bool draw_markers_axis;
    bool publish_tf;
    ros::Subscriber cam_info_sub;
    bool cam_info_received;
    image_transport::Publisher image_pub;
    image_transport::Publisher debug_pub;
    ros::Publisher pose_pub;
    ros::Publisher transform_pub;
    ros::Publisher position_pub;
    std::string board_frame;
    int nMarkers, nMarkerDetectThreshold;
    double marker_size_m;
    std::string board_config;
    double board_scale;

    ros::NodeHandle nh;
    image_transport::ImageTransport it;
    image_transport::Subscriber image_sub;

    tf::TransformListener _tfListener;

    cv::Ptr<cv::aruco::Dictionary> dictionary;
    cv::Ptr<cv::aruco::DetectorParameters> detectorParams;
    cv::Ptr<cv::aruco::Board> board;

public:

    ArSysSingleBoard() : cam_info_received(false),
    nh("~"),
    it(nh), nMarkers(0), nMarkerDetectThreshold(0) {
        image_sub = it.subscribe("/image", 1, &ArSysSingleBoard::image_callback, this);
        cam_info_sub = nh.subscribe("/camera_info", 1, &ArSysSingleBoard::cam_info_callback, this);

        image_pub = it.advertise("result", 1);
        debug_pub = it.advertise("debug", 1);
        pose_pub = nh.advertise<geometry_msgs::PoseStamped>("pose", 100);
        transform_pub = nh.advertise<geometry_msgs::TransformStamped>("transform", 100);
        position_pub = nh.advertise<geometry_msgs::Vector3Stamped>("position", 100);

        nh.param<double>("marker_size", marker_size_m, 0.05);
        nh.param<std::string>("board_config", board_config, "boardConfiguration.yml");
        nh.param<std::string>("board_frame", board_frame, "");
        nh.param<bool>("image_is_rectified", useRectifiedImages, true);
        nh.param<bool>("draw_markers", draw_markers, false);
        nh.param<bool>("draw_markers_cube", draw_markers_cube, false);
        nh.param<bool>("draw_markers_axis", draw_markers_axis, false);
        nh.param<bool>("publish_tf", publish_tf, false);

        cv::aruco::PREDEFINED_DICTIONARY_NAME dictionaryId = cv::aruco::DICT_ARUCO_ORIGINAL;
        dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
        detectorParams = cv::aruco::DetectorParameters::create();
        detectorParams->doCornerRefinement = true; // do corner refinement in markers
        detectorParams->cornerRefinementWinSize = 10; // do corner refinement in markers
        detectorParams->cornerRefinementMaxIterations = 30; // do corner refinement in markers

        //the_board_config.readFromFile(board_config.c_str());
        readBoard();
        int minMarkers = std::min(4,nMarkers);
        nMarkerDetectThreshold = std::max(minMarkers,nMarkers/2);
        ROS_INFO("ArSys node started with marker size of %f m and board configuration: %s",
                marker_size_m, board_config.c_str());
    }

    void readBoard() {
        cv::FileStorage fs(board_config, cv::FileStorage::READ);
        float mInfoType;
        cv::Mat markers;
        cv::FileNode nmarkers = fs["aruco_bc_nmarkers"];
        cv::read(nmarkers, nMarkers, 0);
        //std::cout << "nmarkers " << nMarkers << std::endl;
        cv::FileNode minfotype = fs["aruco_bc_mInfoType"];
        cv::read(minfotype, mInfoType, 0);
        cv::FileNode markersnode = fs["aruco_bc_markers"];
        cv::FileNodeIterator it = markersnode.begin(), it_end = markersnode.end();
        int idx = 0, idx2;
        float id, lenX, lenY, lenZ, markerSideLength_pixels;
        std::vector< float> ids;
        std::vector< std::vector<float> > cornervals;
        std::vector< cv::Mat > idcorners;
        // iterate through a sequence using FileNodeIterator
        for (; it != it_end; ++it, idx++) {
            (*it)["id"] >> id;
            ids.push_back(id);
            //std::cout << "id " << id << std::endl;
            (*it)["corners"] >> cornervals;
            cv::Mat markercorners = cv::Mat::zeros(1, 4, CV_32FC3);
            for (int i = 0; i < (int) cornervals.size(); i++) {
                markercorners.at<cv::Vec3f>(0, i)[0] = cornervals[i][0];
                markercorners.at<cv::Vec3f>(0, i)[1] = cornervals[i][1];
                markercorners.at<cv::Vec3f>(0, i)[2] = cornervals[i][2];
                if (i == 1) {
                    lenX = fabs(markercorners.at<cv::Vec3f>(0, 0)[0] - markercorners.at<cv::Vec3f>(0, 1)[0]);
                    lenY = fabs(markercorners.at<cv::Vec3f>(0, 0)[1] - markercorners.at<cv::Vec3f>(0, 1)[1]);
                    lenZ = fabs(markercorners.at<cv::Vec3f>(0, 0)[2] - markercorners.at<cv::Vec3f>(0, 1)[2]);
                    markerSideLength_pixels = std::max(lenZ, std::max(lenX, lenY));
                    //std::cout << "markerSideLength = " << markerSideLength_pixels << std::endl;
                }
            }
            //std::cout << "id " << idx << " corners " << markercorners << std::endl;
            markercorners *= marker_size_m / markerSideLength_pixels; // convert to m
            idcorners.push_back(markercorners);
        }
        board_scale = sqrt(nMarkers / 2);
        //for (cv::Mat cornerVals : idcorners) 
        //        std::cout << "id corners " << cornerVals << std::endl;
        board = cv::aruco::Board::create(idcorners, dictionary, ids);
    }

    void image_callback(const sensor_msgs::ImageConstPtr& msg) {
        static tf::TransformBroadcaster br;

        if (!cam_info_received) return;

        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
            inImage = cv_ptr->image;

            bool refindStrategy = false;
            std::vector< int > ids;
            std::vector< std::vector< cv::Point2f > > corners, rejected;
            cv::Vec3d rvec, tvec;

            // detect markers
            cv::aruco::detectMarkers(inImage, dictionary, corners, ids, detectorParams, rejected);

            // refind strategy to detect more markers
            if (refindStrategy) {
                cv::aruco::refineDetectedMarkers(inImage, board, corners, ids, rejected, cameraMatrix,
                        distortionCoeffs);
            }
            // estimate board pose
            int markersOfBoardDetected = 0;
            if (ids.size() > nMarkerDetectThreshold) {
                markersOfBoardDetected =
                        cv::aruco::estimatePoseBoard(corners, ids, board, cameraMatrix, distortionCoeffs, rvec, tvec);

                tf::Transform transform = getTf(rvec, tvec);

                tf::StampedTransform stampedTransform(transform, msg->header.stamp, msg->header.frame_id, board_frame);

                if (publish_tf)
                    br.sendTransform(stampedTransform);

                geometry_msgs::PoseStamped poseMsg;
                tf::poseTFToMsg(transform, poseMsg.pose);
                poseMsg.header.frame_id = msg->header.frame_id;
                poseMsg.header.stamp = msg->header.stamp;
                pose_pub.publish(poseMsg);

                geometry_msgs::TransformStamped transformMsg;
                tf::transformStampedTFToMsg(stampedTransform, transformMsg);
                transform_pub.publish(transformMsg);

                geometry_msgs::Vector3Stamped positionMsg;
                positionMsg.header = transformMsg.header;
                positionMsg.vector = transformMsg.transform.translation;
                position_pub.publish(positionMsg);
            }
            //for each marker, draw info and its boundaries in the image
            //image.copyTo(imageCopy);
            resultImg = cv_ptr->image.clone();
            if (ids.size() > 0) {
                cv::aruco::drawDetectedMarkers(resultImg, corners, ids);
            }
            if (ids.size() > 0) {
                //std::cout << "board scale " << board_scale << std::endl;
                if (draw_markers_axis) {
                    //cv::aruco::drawAxis(resultImg, cameraMatrix, distortionCoeffs,
                    //        rvec, tvec, 2 * marker_size_m);
                }
                //cv::Size outSize(marker_size_m*scale, marker_size_m * scale);
                //cv::aruco::drawPlanarBoard(board, outSize, resultImg);
            }

            if (image_pub.getNumSubscribers() > 0) {
                //show input with augmented information
                cv_bridge::CvImage out_msg;
                out_msg.header.frame_id = msg->header.frame_id;
                out_msg.header.stamp = msg->header.stamp;
                out_msg.encoding = sensor_msgs::image_encodings::RGB8;
                out_msg.image = resultImg;
                image_pub.publish(out_msg.toImageMsg());
            }
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }

    // wait for one camerainfo, then shut down that subscriber

    void cam_info_callback(const sensor_msgs::CameraInfo &msg) {
        if (msg.K[0] == 0) {
            std::cout << msg << std::endl;
            ROS_ERROR("Camera Info message is zero --> Cannot use an uncalibrated camera!");
            return;
        }
        getCamParams(msg, useRectifiedImages);
        cam_info_received = true;
        cam_info_sub.shutdown();
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "ar_single_board");

    ArSysSingleBoard node;

    ros::spin();
}
