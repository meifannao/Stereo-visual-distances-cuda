#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include "params.h"

#include <iostream>
#include <string>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;


/*****描述：鼠标操作回调*****/
static void onMouse(int event, int x, int y, int, void*)
{
    if (selectObject)
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
    }

    switch (event)
    {
    case EVENT_LBUTTONDOWN: // 鼠标左按钮按下的事件
        origin = Point(x, y);
        selection = Rect(x, y, 0, 0);
        selectObject = true;
        // cout << origin << "in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
        point3 = xyz.at<Vec3f>(origin);
        point3[0];
        // cout << "point3[0]:" << point3[0] << "point3[1]:" << point3[1] << "point3[2]:" << point3[2]<<endl;
        cout << "世界坐标" << endl;
        cout << "x: " << point3[0] << "  y: " << point3[1] << "  z: " << point3[2] << endl;
        d = point3[0] * point3[0] + point3[1] * point3[1] + point3[2] * point3[2];
        d = sqrt(d); // mm
        // cout << "距离是:" << d << "mm" << endl;

        d = d / 1000.0;   //m
        cout << "距离是:" << d << "m" << endl;

        break;
    case EVENT_LBUTTONUP: // 鼠标左按钮释放的事件
        selectObject = false;
        if (selection.width > 0 && selection.height > 0)
            break;
    }
}

const String keys =
"{help h usage ? |                  | print this message                                                }"
"{@left          |./aloeL.jpg | left view of the stereopair                                       }"
"{@right         |./aloeR.jpg | right view of the stereopair                                      }"
"{GT             |./aloeGT.png| optional ground-truth disparity (MPI-Sintel or Middlebury format) }"
"{dst_path       |None              | optional path to save the resulting filtered disparity map        }"
"{dst_raw_path   |None              | optional path to save raw disparity map before filtering          }"
"{video_path     |./car.avi         | loading video path                                                }"
"{algorithm      |sgbm                | stereo matching method (bm or sgbm)                               }"
"{filter         |wls_conf          | used post-filtering (wls_conf or wls_no_conf or fbs_conf)         }"
"{no-display     |                  | don't display results                                             }"
"{no-downscale   |                  | force stereo matching on full-sized views to improve quality      }"
"{dst_conf_path  |None              | optional path to save the confidence map used in filtering        }"
"{vis_mult       |2.0               | coefficient used to scale disparity map visualizations            }"
"{max_disparity  |128               | parameter of stereo matching                                      }"
"{window_size    |-1                | parameter of stereo matching                                      }"
"{wls_lambda     |8000.0            | parameter of wls post-filtering                                   }"
"{wls_sigma      |1.5               | parameter of wls post-filtering                                   }"
;

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("Disparity Filtering Demo");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String left_im = parser.get<String>(0);
    String right_im = parser.get<String>(1);
    String GT_path = parser.get<String>("GT");
    String video_path = parser.get<String>("video_path");
    String dst_path = parser.get<String>("dst_path");
    String dst_raw_path = parser.get<String>("dst_raw_path");
    String dst_conf_path = parser.get<String>("dst_conf_path");
    String algo = parser.get<String>("algorithm");
    String filter = parser.get<String>("filter");
    bool no_display = parser.has("no-display");
    bool no_downscale = parser.has("no-downscale");
    int max_disp = parser.get<int>("max_disparity");
    double lambda = parser.get<double>("wls_lambda");
    double sigma = parser.get<double>("wls_sigma");
    double vis_mult = parser.get<double>("vis_mult");
    int wsize;
    int preFilterCap = 1;       // 范围(1 - 30)
    int UniquenessRatio = 10;   // 范围(5 - 15)
    int SpeckleRange = 100;     // 范围(50 - 150)
    int SpeckleWindowSize = 100; // 范围(10 - 100)

    if (parser.get<int>("window_size") >= 0) //user provided window_size value
        wsize = parser.get<int>("window_size");
    else
    {
        if (algo == "sgbm")
            wsize = 3; //default window size for SGBM
        else if (!no_downscale && algo == "bm" && filter == "wls_conf")
            wsize = 7; //default window size for BM on downscaled views (downscaling is performed only for wls_conf)
        else
            wsize = 15; //default window size for BM on full-sized views
    }

    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    Rodrigues(rec, R); // Rodrigues变换
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
        0, imageSize, &validROIL, &validROIR);
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

    VideoCapture capture = VideoCapture(video_path);
    cv::Mat frame;
    bool ret = capture.read(frame);

    if (!ret)
    {
        cout << "Cannot read video file ";
        return -1;
    }
    int cnt = 0;
    while (ret)
    {
        cnt++;
        double t1 = (double)cv::getTickCount();
        bool ret = capture.read(frame);
        //! [load_views]
        Mat rgbImageL = frame(cv::Range(0, 480), cv::Range(0, 640));
        Mat rgbImageR = frame(cv::Range(0, 480), cv::Range(640, 1280));

        // 将BGR格式转换成灰度图片，用于畸变矫正
        Mat grayImageL, grayImageR;
        cvtColor(rgbImageL, grayImageL, cv::COLOR_BGR2GRAY);
        cvtColor(rgbImageR, grayImageR, cv::COLOR_BGR2GRAY);

        /*
        经过remap之后，左右相机的图像已经共面并且行对准了
        */
        Mat rectifyImageL, rectifyImageR;
        remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
        remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

        cvtColor(rectifyImageL, rgbImageL, cv::COLOR_GRAY2BGR);
        cvtColor(rectifyImageR, rgbImageR, cv::COLOR_GRAY2BGR);


        Mat left_for_matcher, right_for_matcher;
        Mat left_disp, right_disp;
        Mat filtered_disp;
        Mat conf_map = Mat(rgbImageL.rows, rgbImageR.cols, CV_8U);
        conf_map = Scalar(255);
        Rect ROI;
        Ptr<DisparityWLSFilter> wls_filter;
        double matching_time, filtering_time;
        if (max_disp <= 0 || max_disp % 16 != 0)
        {
            cout << "Incorrect max_disparity value: it should be positive and divisible by 16";
            return -1;
        }
        if (wsize <= 0 || wsize % 2 != 1)
        {
            cout << "Incorrect window_size value: it should be positive and odd";
            return -1;
        }

        if (!no_downscale)
        {
            // downscale the views to speed-up the matching stage, as we will need to compute both left
            // and right disparity maps for confidence map computation
            //! [downscale]
            max_disp /= 2;
            if (max_disp % 16 != 0)
                max_disp += 16 - (max_disp % 16);
            resize(rectifyImageL, left_for_matcher, Size(), 0.5, 0.5, INTER_LINEAR_EXACT);
            resize(rectifyImageR, right_for_matcher, Size(), 0.5, 0.5, INTER_LINEAR_EXACT);
            //! [downscale]
        }
        else
        {
            left_for_matcher = rectifyImageL.clone();
            right_for_matcher = rectifyImageR.clone();
        }

        if (algo == "bm")
        {
            //! [matching]
            Ptr<StereoBM> left_matcher = StereoBM::create(max_disp, wsize);
            wls_filter = createDisparityWLSFilter(left_matcher);
            Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

            // cvtColor(left_for_matcher, left_for_matcher, COLOR_BGR2GRAY);
            // cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);

            matching_time = (double)getTickCount();
            left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
            right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
            matching_time = ((double)getTickCount() - matching_time) / getTickFrequency();
        }
        else if (algo == "sgbm")
        {
            Ptr<StereoSGBM> left_matcher = StereoSGBM::create(1, max_disp, wsize);
            left_matcher->setP1(24 * wsize * wsize);
            left_matcher->setP2(96 * wsize * wsize);

            // left_matcher->setPreFilterCap(preFilterCap);
            // left_matcher->setUniquenessRatio(UniquenessRatio);
            // left_matcher->setSpeckleRange(SpeckleRange);
            // left_matcher->setSpeckleWindowSize(SpeckleWindowSize);

            wls_filter = createDisparityWLSFilter(left_matcher);
            Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

            // cvtColor(left_for_matcher, left_for_matcher, COLOR_GRAY2BGR);
            // cvtColor(right_for_matcher, right_for_matcher, COLOR_GRAY2BGR);

            // cvtColor(left_for_matcher, left_for_matcher, COLOR_BGR2GRAY);
            // cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);


            matching_time = (double)getTickCount();
            left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
            right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
            matching_time = ((double)getTickCount() - matching_time) / getTickFrequency();
        }
        else
        {
            cout << "Unsupported algorithm";
            return -1;
        }

        //! [filtering]
        wls_filter->setLambda(lambda);
        wls_filter->setSigmaColor(sigma);
        filtering_time = (double)getTickCount();
        wls_filter->filter(left_disp, rgbImageL, filtered_disp, right_disp);
        filtering_time = ((double)getTickCount() - filtering_time) / getTickFrequency();
        //! [filtering]
        conf_map = wls_filter->getConfidenceMap();

        // Get the ROI that was used in the last filter call:
        ROI = wls_filter->getROI();
        if (!no_downscale)
        {
            // upscale raw disparity and ROI back for a proper comparison:
            resize(left_disp, left_disp, Size(), 2.0, 2.0, INTER_LINEAR_EXACT);
            left_disp = left_disp * 2.0;
            ROI = Rect(ROI.x * 2, ROI.y * 2, ROI.width * 2, ROI.height * 2);
        }

        //collect and print all the stats:
        // 禁用科学计数法
        std::cout << std::fixed;
        cout.precision(2);
        if (cnt % 50 == 0)
        {
            double t2 = (double)cv::getTickCount();
            double frameTime = (t2 - t1) / cv::getTickFrequency();
            cnt = 0;
            cout << "Matching time:  " << matching_time << "s" << endl;
            cout << "Filtering time: " << filtering_time << "s" << endl;
            // cout << "Solving time: " << solving_time << "s" << endl;
            cout << "Matach + Filter FPS: " << 1.0 / (matching_time + filtering_time) << endl;
            cout << "FPS ALL:" << 1.0 / frameTime << endl;
            cout << endl;
        }

        if (dst_path != "None")
        {
            Mat filtered_disp_vis;
            getDisparityVis(filtered_disp, filtered_disp_vis, vis_mult);
            // normalize(filtered_disp, filtered_disp_vis, 0, 255, NORM_MINMAX, CV_8UC1);
            imwrite(dst_path, filtered_disp_vis);
        }
        if (dst_raw_path != "None")
        {
            Mat raw_disp_vis;
            getDisparityVis(left_disp, raw_disp_vis, vis_mult);
            imwrite(dst_raw_path, raw_disp_vis);
        }
        if (dst_conf_path != "None")
        {
            imwrite(dst_conf_path, conf_map);
        }

        if (!no_display)
        {
            namedWindow("left", WINDOW_AUTOSIZE);
            imshow("left", rgbImageL);
            namedWindow("right", WINDOW_AUTOSIZE);
            imshow("right", rgbImageR);

            //! [visualization]
            Mat raw_disp_vis;
            getDisparityVis(left_disp, raw_disp_vis, vis_mult);
            namedWindow("raw disparity", WINDOW_AUTOSIZE);
            imshow("raw disparity", raw_disp_vis);
            Mat filtered_disp_vis;
            getDisparityVis(filtered_disp, filtered_disp_vis, vis_mult);
            namedWindow("filtered disparity", WINDOW_AUTOSIZE);
            reprojectImageTo3D(filtered_disp, xyz, Q, true); // 在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
            xyz = xyz * 16;
            setMouseCallback("filtered disparity", onMouse, 0);
            imshow("filtered disparity", filtered_disp_vis);



            if (cv::waitKey(30) == 'q')
            {
                break;
            }
            /*
            while (1)
            {
                char key = (char)waitKey();
                if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
                    break;
            }*/
            //! [visualization]
        }
    }

    return 0;
}

