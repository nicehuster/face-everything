//
// Created by nice on 18-01-16.
//
#include "cl_common.h"
#include "mtcnn.h"

#include <fstream>

using namespace cv;
using namespace std;
using namespace cl::fd;
using namespace cl::fa;
#define SAVE
int main(int argc, char** argv) {

    /*must change these there lines to yours**/
    const string model_dir = "/home/nice/data/face-everthing/trained_models/detection"; // mtcnn model dir
  
    const int AlignWidth = 128;
    const int AlignHeight = 128;

    bool verbose = false;//turn on to visulize align result
    const vector<string> argvs = {
            model_dir + "/det1.prototxt",
            model_dir + "/det2.prototxt",
            model_dir + "/det3.prototxt",
            model_dir + "/det1.caffemodel",
            model_dir + "/det2.caffemodel",
            model_dir + "/det3.caffemodel",
    };

    std::shared_ptr<Alignment> face_alignment = std::make_shared<mtcnn::fd::FaceDetector>();
    if(!face_alignment->load_model({argvs[0], argvs[1], argvs[2]}, {argvs[3], argvs[4], argvs[5]}))
    {
        cout <<"Can't load face detection model!" << endl;
        return  -1;
    }

    TickMeter tm;

    VideoCapture cap(argv[1]);
    if(!cap.isOpened())
    {
        cout <<"Can't open video!" << endl;
        return  -3;
    }
    Mat src;
    cap.read(src);
    double t=0,fps=0;
#ifdef SAVE
	cv::VideoWriter saveVideo;
	saveVideo.open("result1.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25, cv::Size(src.cols, src.rows));
#endif
    
     for(;;)
    {
        cap.read(src);
        if (src.empty()) continue;
        Mat src_gray;
        Mat aligned_face;
        Mat src_display = src.clone();
        if(src.channels() == 3)
        cv::cvtColor(src, src_gray, CV_BGR2GRAY);
        
        vector<cl::FaceBox> face_boxes;
        vector<cl::FaceLandmark> face_landmarks;
        tm.reset();tm.start();
        t = (double)cv::getTickCount();
        face_alignment->detect(src, face_boxes, face_landmarks);
        if(face_landmarks.size() > 0){
            auto biggest_id = cl::fa::get_biggest_id(face_landmarks, src.cols, src.rows);
            auto aligned_face = face_alignment->align_face(src, face_landmarks[biggest_id], AlignWidth, AlignHeight);
            cl::fa::draw(face_boxes[biggest_id], face_landmarks[biggest_id], src_display);
        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        fps = 1.0 / t;
        char string[20];
        sprintf(string, "%.2f", fps);     
        std::string fpsString("FPS:");
        fpsString += string;                   
        putText(src_display, fpsString, cv::Point(5, 20),
         cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0, 0, 0));
        //imshow("aligned", src_display);
#ifdef SAVE
		saveVideo << src_display;
#endif
    tm.stop();
    LOG(INFO) << "Align all dataset, time: " << tm.getTimeSec() << " seconds";
    char key = waitKey(1);
        if(key == 27){
            break;
        }
    }
    
}
