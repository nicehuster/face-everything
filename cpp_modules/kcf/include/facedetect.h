#ifndef FACEDETECT__H__
#define FACEDETECT__H__
#include "cl_common.h"
#include "mtcnn.h"
using namespace cv;
using namespace std;

using namespace cl::fd;
using namespace cl::fa;
using namespace mtcnn::fd;
bool facedetect(cv::Mat frame,cv::Rect &Box);
#endif