//
// Created by ise on 16-10-28.
//

#ifndef TEST_GETFIRSTBOX_H
#define TEST_GETFIRSTBOX_H


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <string>

void getFirstBox(const cv::Mat& firstFrame, cv::Rect& firstBox,  std::string windowName );
void mouseHandler(int event, int x, int y, int flag, void* userdata);

#endif //TEST_GETFIRSTBOX_H
