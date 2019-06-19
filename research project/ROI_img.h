#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
//#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video/tracking.hpp>


typedef struct {
	cv::Point matchLoc;
	cv::Mat re_temp;
	double Max_score;
	int index;
}RE_Matching;

RE_Matching ROI_Temp_img(cv::Mat img, cv::Mat templ);
void Image_Processing(cv::Mat & temp1_T, float gamma);