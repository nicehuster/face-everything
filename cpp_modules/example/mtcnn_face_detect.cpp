//
// Created by root on 17-8-18.
//

#include "cl_common.h"
#include "mtcnn.h"
using namespace cv;
using namespace std;

using namespace cl::fd;
using namespace cl::fa;
using namespace mtcnn::fd;
#define SAVE


int main(int argc, char** argv) {

    /*must change these two lines to yours**/
    const string detection_dir = "/home/nice/data/face-everthing/trained_models/detection"; // mtcnn model dir

    const vector<string> argvs = {
            detection_dir + "/det1.prototxt",
            detection_dir + "/det2.prototxt",
            detection_dir + "/det3.prototxt",
            detection_dir + "/det1.caffemodel",
            detection_dir + "/det2.caffemodel",
            detection_dir + "/det3.caffemodel",
    };
    /**mtcnn face detector*/
    std::shared_ptr<Detector> face_detector = std::make_shared<mtcnn::fd::FaceDetector>();
    if(!face_detector->load_model({argvs[0], argvs[1], argvs[2]},{argvs[3], argvs[4], argvs[5]}))
    {
        cout <<"Can't load face detection model!" << endl;
        return  -1;
    }

    VideoCapture cap(argv[1]);
    if(!cap.isOpened())
    {
        cout <<"Can't open video!" << endl;
        return  -3;
    }
    Mat img;
    cap.read(img);
    namedWindow("detection");
    double t=0,fps=0;
#ifdef SAVE
	cv::VideoWriter saveVideo;
	saveVideo.open("mtcnn_detect_result.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25, cv::Size(img.cols, img.rows));
#endif
    for(;;)
    {
        cap.read(img);
        if (img.empty()) continue;

        Mat display_img = img.clone();
        vector<cl::FaceBox> windows;
        vector<cl::FaceLandmark> landmarks;

        /**detect */
        t = (double)cv::getTickCount();
        if(face_detector->detect(img, windows)) {
            for(const auto& e: windows){
                cv::rectangle(display_img, e.bbox_, cv::Scalar(0, 255, 0), 1);
            }
        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        fps = 1.0 / t;
        char string[20];
        sprintf(string, "%.2f", fps);     
        std::string fpsString("FPS:");
        fpsString += string;                   
        putText(display_img, fpsString, cv::Point(5, 20),
         cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0, 0, 0));
        cv::imshow("detection", display_img);
#ifdef SAVE
		saveVideo << display_img;
#endif
        char key = waitKey(1);
        if(key == 27){
            break;
        }
    }
}

