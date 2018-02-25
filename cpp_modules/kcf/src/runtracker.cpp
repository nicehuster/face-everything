#include <iostream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "kcftracker.hpp"

#define SAVE;
int main(int argc, char* argv[]){

	if (argc < 2) {
        printf("Please specify a video directory\n");
        return -1;
    }

    std::string fileName = argv[1];
    std::string windowName = "tracking out";
	// Create KCFTracker object
	KCFTracker tracker(true, false, true, true);

	// Frame readed
	cv::Mat frame;
    cv::Mat displayImage;
	// Tracker results
	cv::Rect result;

    cv::VideoCapture video(fileName);
#ifdef SAVE
    cv::VideoWriter saveVideo;
    saveVideo.open("/home/ltt/ClionProjects/KCFcpp/video/result1.avi",CV_FOURCC('M','J','P','G'),50,
                   cv::Size(frame.cols,frame.rows));
#endif

    while (video.read(frame))
    {
        frame.copyTo(displayImage);

        tracker.faceResult(frame,result);
        rectangle(displayImage,result,cv::Scalar(0,0,255),2,8);
#ifdef SAVE
        saveVideo << displayImage;
#endif
        cv::imshow(windowName,displayImage);
        cv::waitKey(1);
    }

    return 0;
}
