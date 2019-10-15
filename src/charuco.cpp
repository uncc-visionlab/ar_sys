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

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>

#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#include <ar_sys/utils.h>
#include <ar_sys/ArucoCornerMsg.h>
#include <ar_sys/OneMarker.h>

namespace cv {
    namespace aruco {

        void getBoardObjectAndImagePointsCustom(const Ptr<Board> &board, InputArrayOfArrays detectedCorners,
                InputArray detectedIds, OutputArray objPoints, OutputArray imgPoints) {

            CV_Assert(board->ids.size() == board->objPoints.size());
            CV_Assert(detectedIds.total() == detectedCorners.total());

            size_t nDetectedMarkers = detectedIds.total();

            std::vector< Point3f > objPnts;
            objPnts.reserve(nDetectedMarkers);

            std::vector< Point2f > imgPnts;
            imgPnts.reserve(nDetectedMarkers);

            // look for detected markers that belong to the board and get their information
            for (unsigned int i = 0; i < nDetectedMarkers; i++) {
                int currentId = detectedIds.getMat().ptr< int >(0)[i];
                for (unsigned int j = 0; j < board->ids.size(); j++) {
                    if (currentId == board->ids[j]) {
                        for (int p = 0; p < 4; p++) {
                            objPnts.push_back(board->objPoints[j][p]);
                            imgPnts.push_back(detectedCorners.getMat(i).ptr< Point2f >(0)[p]);
                        }
                    }
                }
            }

            // create output
            Mat(objPnts).copyTo(objPoints);
            Mat(imgPnts).copyTo(imgPoints);
        }

        /**
         */
        int estimatePoseBoardCustom(InputArrayOfArrays _corners, InputArray _ids, const Ptr<Board> &board,
                InputArray _cameraMatrix, InputArray _distCoeffs, OutputArray _rvec,
                OutputArray _tvec, bool useExtrinsicGuess) {
            //std::cout << "Calling my estimatePoseBoard!" << std::endl;
            CV_Assert(_corners.total() == _ids.total());

            // get object and image points for the solvePnP function
            Mat objPoints, imgPoints;
            getBoardObjectAndImagePointsCustom(board, _corners, _ids, objPoints, imgPoints);

            CV_Assert(imgPoints.total() == objPoints.total());

            if (objPoints.total() == 0) // 0 of the detected markers in board
                return 0;

            solvePnP(objPoints, imgPoints, _cameraMatrix, _distCoeffs, _rvec, _tvec, useExtrinsicGuess);

            // divide by four since all the four corners are concatenated in the array for each marker
            return (int) objPoints.total() / 4;
        }

    }
}

class ArSysSingleBoard : public ARSYS {
private:
    cv::Mat inImage, resultImg;
    //aruco::CameraParameters camParam;
    bool useRectifiedImages;
    bool draw_markers;
    bool draw_markers_axis;
    bool publish_tf;
    bool publish_corners;
    ros::Subscriber cam_info_sub;
    bool cam_info_received;
    image_transport::Publisher image_pub;
    image_transport::Publisher debug_pub;
    ros::Publisher pose_pub;
    ros::Publisher transform_pub;
    ros::Publisher position_pub;
    ros::Publisher corner_pub;
    std::string board_frame;
    int nMarkers, nMarkerDetectThreshold;
    double marker_size_m;
    std::string board_config;
    double board_scale;
    std::vector< cv::Mat > idcorners;
    std::vector< cv::Mat > idcornerspx;
    std::vector< float> board_ids;

    ros::NodeHandle nh;
    image_transport::ImageTransport it;
    image_transport::Subscriber image_sub;

    tf::TransformListener _tfListener;

    cv::Ptr<cv::aruco::Dictionary> dictionary;
    cv::Ptr<cv::aruco::DetectorParameters> detectorParams;
    cv::Ptr<cv::aruco::CharucoBoard> board;

public:

    ArSysSingleBoard() : ARSYS(), cam_info_received(false),
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
        nh.param<bool>("draw_markers_axis", draw_markers_axis, false);
        nh.param<bool>("publish_tf", publish_tf, false);
        nh.param<bool>("publish_corners", publish_corners, false);

        if(publish_corners)
            corner_pub = nh.advertise<ar_sys::ArucoCornerMsg>("corner",100);

        cv::aruco::PREDEFINED_DICTIONARY_NAME dictionaryId = cv::aruco::DICT_4X4_50;

        dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
        detectorParams = cv::aruco::DetectorParameters::create();

#ifdef OPENCV_3_2
        detectorParams->doCornerRefinement = true; // do corner refinement in markers
#endif
        
#ifdef OPENCV_3_3
        detectorParams->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
#endif
        
        detectorParams->cornerRefinementWinSize = 10; // do corner refinement in markers
        detectorParams->cornerRefinementMaxIterations = 30; // do corner refinement in markers

        //readBoard();
        board = cv::aruco::CharucoBoard::create(6,4,.05,.025,dictionary);
        nMarkers = 24;
        int minMarkers = std::min(4, nMarkers);
        nMarkerDetectThreshold = std::max(minMarkers, nMarkers / 2);
        /*ROS_INFO("ArSys node started with marker size of %f m and board configuration: %s",
                marker_size_m, board_config.c_str());*/
    }

    void readBoard() {
        /*
        cv::FileStorage fs(board_config, cv::FileStorage::READ);
        float mInfoType;
        cv::Mat markers;
        cv::FileNode nmarkers = fs["aruco_bc_nmarkers"];
        cv::read(nmarkers, nMarkers, 0);
        std::cout << "nmarkers " << nMarkers << std::endl;
        cv::FileNode minfotype = fs["aruco_bc_mInfoType"];
        cv::read(minfotype, mInfoType, 0);
        cv::FileNode markersnode = fs["aruco_bc_markers"];
        cv::FileNodeIterator it = markersnode.begin(), it_end = markersnode.end();
        int idx = 0, idx2;
        float id, lenX, lenY, lenZ, markerSideLength_pixels;

        std::vector< std::vector<float> > cornervals;

        // iterate through a sequence using FileNodeIterator
        for (; it != it_end; ++it, idx++) {
            (*it)["id"] >> id;
            board_ids.push_back(id);
            std::cout << "id " << id << std::endl;
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
                }
            }
            idcornerspx.push_back(markercorners);
            markercorners *= marker_size_m / markerSideLength_pixels; // convert to m
            idcorners.push_back(markercorners);
        }
        board_scale = sqrt(nMarkers / 2);
        for (cv::Mat cornerVals : idcorners)
                std::cout << "id corners " << cornerVals << std::endl;
        board = cv::aruco::Board::create(idcorners, dictionary, board_ids);
        */
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
            cv::Vec3d tvec(0, 0, 1);
            cv::Vec3d rvec(0, 0, 0);
            cv::Mat guessRotMat = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
            cv::Rodrigues(guessRotMat, rvec);
            // detect markers
            cv::aruco::detectMarkers(inImage, dictionary, corners, ids, detectorParams, rejected);

            // refind strategy to detect more markers
            if (refindStrategy) {
                cv::aruco::refineDetectedMarkers(inImage, board, corners, ids, rejected, cameraMatrix,
                        distortionCoeffs);
            }
            // estimate board pose
            int markersOfBoardDetected = 0;
            if (ids.size() <= nMarkerDetectThreshold) {
                return;
            }
            /*markersOfBoardDetected =
                    cv::aruco::estimatePoseBoardCustom(corners, ids, board, cameraMatrix, distortionCoeffs, rvec, tvec, true);*/
            std::vector<cv::Point2f> charucoCorners;
            std::vector<int> charucoIds;
            cv::aruco::interpolateCornersCharuco(corners, ids, inImage, board, charucoCorners, charucoIds, cameraMatrix, distortionCoeffs);
            cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distortionCoeffs, rvec, tvec);
            cv::Mat rotMat;
            cv::Rodrigues(rvec, rotMat);
            cv::Mat eZ = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
            cv::Mat eZ_prime = rotMat*eZ;
            if (tvec[2] < 0) {
                std::cout << "cv::solvePnP converged to invalid transform translation z = " << tvec[2] <<
                        " when, in reality we must assert, z > 0." << std::endl;
                return;
            }

            if (publish_corners)
            {
                ar_sys::ArucoCornerMsg cornerMsg;
                ar_sys::OneMarker PixelMsg;
                ar_sys::OneMarker MetricMsg;
                cornerMsg.header = msg->header;

                for (int i = 0; i < (int) ids.size(); i++)
                {
                    int index = std::distance(board_ids.begin(), std::find (board_ids.begin(), board_ids.end(), ids[i]));
                    if ( index < 0 || index > board_ids.size()-1)
                        continue;
                    PixelMsg.id = ids[i];
                    PixelMsg.top_left.x = corners[i][0].x;
                    PixelMsg.top_left.y = corners[i][0].y;
                    PixelMsg.top_right.x = corners[i][1].x;
                    PixelMsg.top_right.y = corners[i][1].y;
                    PixelMsg.bottom_right.x = corners[i][2].x;
                    PixelMsg.bottom_right.y = corners[i][2].y;
                    PixelMsg.bottom_left.x = corners[i][3].x;
                    PixelMsg.bottom_left.y = corners[i][3].y;

                    cornerMsg.pixel_corners.push_back(PixelMsg);

                    MetricMsg.id = ids[i];
                    MetricMsg.top_left.x = idcornerspx.at(index).at<cv::Vec3f>(0, 0)[0];
                    MetricMsg.top_left.y = idcornerspx.at(index).at<cv::Vec3f>(0, 0)[1];
                    MetricMsg.top_left.z = idcornerspx.at(index).at<cv::Vec3f>(0, 0)[2];
                    MetricMsg.top_right.x = idcornerspx.at(index).at<cv::Vec3f>(0, 1)[0];
                    MetricMsg.top_right.y = idcornerspx.at(index).at<cv::Vec3f>(0, 1)[1];
                    MetricMsg.top_right.z = idcornerspx.at(index).at<cv::Vec3f>(0, 1)[2];
                    MetricMsg.bottom_right.x = idcornerspx.at(index).at<cv::Vec3f>(0, 2)[0];
                    MetricMsg.bottom_right.y = idcornerspx.at(index).at<cv::Vec3f>(0, 2)[1];
                    MetricMsg.bottom_right.z = idcornerspx.at(index).at<cv::Vec3f>(0, 2)[2];
                    MetricMsg.bottom_left.x = idcornerspx.at(index).at<cv::Vec3f>(0, 3)[0];
                    MetricMsg.bottom_left.y = idcornerspx.at(index).at<cv::Vec3f>(0, 3)[1];
                    MetricMsg.bottom_left.z = idcornerspx.at(index).at<cv::Vec3f>(0, 3)[2];

                    cornerMsg.metric_corners.push_back(MetricMsg);
                }
                corner_pub.publish(cornerMsg);
            }

 /*           if (eZ_prime.at<double>(2,0) > 0) {
                // flip y and z
                rotMat.at<double>(0, 1) *= -1.0;
                rotMat.at<double>(1, 1) *= -1.0;
                rotMat.at<double>(2, 1) *= -1.0;
                rotMat.at<double>(0, 2) *= -1.0;
                rotMat.at<double>(1, 2) *= -1.0;
                rotMat.at<double>(2, 2) *= -1.0;
                eZ = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
                eZ_prime = rotMat*eZ;
                if (eZ_prime.at<double>(2,0) > 0) {
                    // flip y again 
                    rotMat.at<double>(0, 1) *= -1.0;
                    rotMat.at<double>(1, 1) *= -1.0;
                    rotMat.at<double>(2, 1) *= -1.0;
                    // flip x and z (z is already flipped from above)
                    rotMat.at<double>(0, 0) *= -1.0;
                    rotMat.at<double>(1, 0) *= -1.0;
                    rotMat.at<double>(2, 0) *= -1.0;
                    //std::cout << "Different fix via XZ-flip applied." << std::endl;
                } else {
                    //std::cout << "fix via YZ-flip applied." << std::endl;
                }
                eZ = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
                eZ_prime = rotMat*eZ;
                cv::Rodrigues(rotMat, rvec);
            }
*/

            tf::Transform transform;
            transform.setIdentity();
            bool validTransform = getTf(rvec, tvec, transform);
            if (!validTransform) {
                return;
            }

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

            //for each marker, draw info and its boundaries in the image
            //image.copyTo(imageCopy);
            resultImg = cv_ptr->image.clone();
            if (draw_markers) {
                cv::aruco::drawDetectedMarkers(resultImg, corners, ids);
            }
            if (draw_markers_axis) {
                //std::cout << "board scale " << board_scale << std::endl;
                cv::aruco::drawAxis(resultImg, cameraMatrix, distortionCoeffs,
                        rvec, tvec, board_scale * marker_size_m);
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
