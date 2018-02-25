//
// Created by ise on 16-10-28.
//


#include "getFirstBox.h"
#include <iostream>
bool getBox = false;

void getFirstBox(const cv::Mat& firstFrame, cv::Rect& firstBox,  std::string windowName ){

    cv::Mat frameCopy;
    cv::Rect boxTemp;
    cv::namedWindow(windowName);
    cv::setMouseCallback(windowName,mouseHandler,&boxTemp);

    while(!getBox){
        firstFrame.copyTo(frameCopy);
        cv::rectangle(frameCopy,boxTemp,CV_RGB(255,0,0),2);
        cv::imshow(windowName,frameCopy);
        cv::waitKey(1);
    }
    cv::waitKey(0);
    firstBox = boxTemp;

    std::cout<<firstBox<<std::endl;

    return;
}


void mouseHandler(int event, int x, int y, int flag, void* userdata){

    static bool drawingBox = false;
    cv::Rect* firstBox = (cv::Rect*)userdata;

    switch (event){
        case CV_EVENT_MOUSEMOVE:
            if(drawingBox){
                firstBox->width = x - firstBox->x;
                firstBox->height = y - firstBox->y;
            }
            break;
        case CV_EVENT_LBUTTONDOWN:
            drawingBox = true;
            firstBox->x = x;
            firstBox->y = y;
            break;
        case CV_EVENT_LBUTTONUP:
            drawingBox = false;
            if (firstBox->width < 0)
            {
                firstBox->x += firstBox->width;
                firstBox->width *= -1;
            }
            if (firstBox->height < 0)
            {
                firstBox->y += firstBox->height;
                firstBox->height *= -1;
            }
            getBox = true;
            break;
        default:
            break;
    }

}

