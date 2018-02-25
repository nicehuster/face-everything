
#include "facedetect.h"
#include <string>

bool facedetect(cv::Mat frame,cv::Rect &Box)
{
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
     vector<cl::FaceBox> windows;
     int biggstFaceArea = 0;
     if(face_detector->detect(frame, windows)) {
         for(const auto& e: windows){
             if(e.bbox_.width * e.bbox_.height > biggstFaceArea)
             {
                 Box = e.bbox_;
                 biggstFaceArea = e.bbox_.width * e.bbox_.height;
            }
                //cv::rectangle(display_img, e.bbox_, cv::Scalar(0, 255, 0), 1);
        }
    }
    return bool(windows.size() != 0);
}