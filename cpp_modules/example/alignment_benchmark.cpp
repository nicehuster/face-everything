//
// Created by root on 17-8-18.
//

#include "cl_common.h"
#include "seeta_detector.h"
#include "mtcnn.h"
#include "cln_alignment.h"
#include "es_alignment.h"
#include "lbf_alignment.h"
#include "openpose/face/op_face.hpp"

using namespace cv;
using namespace std;

using namespace cl::fd;
using namespace cl::fa;
using namespace seeta::fd;
using namespace mtcnn::fd;
using namespace cln::fa;
using namespace es::fa;
using namespace lbf::fa;
using namespace op::fa;

#define CLN 0
#define ES 0
#define LBF 0
#define OPENPOSE 1
#define SAVE
int main(int argc, char** argv) {

    /*must change these two lines to yours**/
    const string detection_dir = "/home/nice/data/face-everthing/trained_models/detection"; // mtcnn model dir
    const string alignment_dir = "/home/nice/data/face-everthing/trained_models/alignment"; //dataset need to be align

    const vector<string> argvs = {
            detection_dir + "/det1.prototxt",
            detection_dir + "/det2.prototxt",
            detection_dir + "/det3.prototxt",
            detection_dir + "/det1.caffemodel",
            detection_dir + "/det2.caffemodel",
            detection_dir + "/det3.caffemodel",

            alignment_dir + "/cln/main_clnf_general.txt",
            alignment_dir  + "/es/model.txt",
            alignment_dir + "/lbf/LBF.model",
            alignment_dir + "/lbf/Regressor.model",
            alignment_dir + "/op_face/pose_deploy.prototxt",
            alignment_dir + "/op_face/pose_iter_116000.caffemodel",
    };


    /**mtcnn face detector*/
    std::shared_ptr<Detector> face_detector = std::make_shared<mtcnn::fd::FaceDetector>();
    if(!face_detector->load_model({argvs[0], argvs[1], argvs[2]},{argvs[3], argvs[4], argvs[5]}))
    {
        cout <<"Can't load face detection model!" << endl;
        return  -1;
    }


#if CLN
    std::shared_ptr<Alignment> face_alignment = std::make_shared<CLNAlignment>();
    if(!face_alignment->load_model({}, {argvs[6]}))
    {
        cout <<"Can't load face alignment model!" << endl;
        return -2;
    }
#elif ES
    std::shared_ptr<Alignment> face_alignment = std::make_shared<ExplicitShape>();
    if(!face_alignment->load_model({}, {argvs[7]}))
    {
        cout <<"Can't load face alignment model!" << endl;
        return -2;
    }
#elif LBF
    std::shared_ptr<Alignment> face_alignment = std::make_shared<LBFAlignment>();
    if(!face_alignment->load_model({argvs[8]}, {argvs[9]}))
    {
        cout <<"Can't load face alignment model!" << endl;
        return -2;
    }
#elif OPENPOSE
    std::shared_ptr<Alignment> face_alignment = std::make_shared<OPFace>();
    if(!face_alignment->load_model({argvs[10]}, {argvs[11]}))
    {
        cout <<"Load alignment model failed!" << endl;
        return -2;
    }
#endif


    VideoCapture cap(argv[1]);
    if(!cap.isOpened())
    {
        cout <<"Can't open video!" << endl;
        return  -3;
    }
   double t=0,fps=0;
   Mat img;
   cap.read(img);
#ifdef SAVE
	cv::VideoWriter saveVideo;
	saveVideo.open("result.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25, cv::Size(img.cols, img.rows));
#endif
    for(;;)
    {
        cap.read(img);
        if (img.empty()) continue;

        Mat display_img = img.clone();
        Mat blankimg(img.rows,img.cols,CV_8UC3,cv::Scalar(255,255,255));
        vector<cl::FaceBox> windows;
        vector<cl::FaceLandmark> landmarks;

        /**detect */
        t = (double)cv::getTickCount();
        if(face_detector->detect(img, windows)) {
            for(const auto& e: windows){
                cv::rectangle(display_img, e.bbox_, cv::Scalar(0, 255, 0), 1);
            }
        }
        /**alignment*/
        if(face_alignment->detect(img, windows, landmarks)){
            for(const auto& face: landmarks){
                auto bounding_box = cv::boundingRect(face.points_);

                double scale = 0.2;
                double x = std::max<double>(0, bounding_box.x - scale*bounding_box.width/2);
                double y = std::max<double>(0, bounding_box.y - scale*bounding_box.width);
                double width = (1 + scale)*bounding_box.width;
                double height = (1 + scale)*bounding_box.width;

                bounding_box = {
                        (int)x,
                        (int)y,
                        (int)width,
                        (int)height,
                };

                cv::rectangle(display_img,bounding_box, cv::Scalar(255, 0, 0), 2);

                for(const auto& e: face.points_){
                    cv::circle(display_img, Point(e.x, e.y), 1, cv::Scalar(0, 0, 255), 2);
                    cv::circle(blankimg, Point(e.x, e.y), 1, cv::Scalar(0, 0, 255), 2);
                }
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
        cv::imshow("blankimg", blankimg);
        cv::imshow("detection", display_img);
#ifdef SAVE
		saveVideo << blankimg;
#endif
        char key = waitKey(1);
        if(key == 27){
            break;
        }
    
    }
}

