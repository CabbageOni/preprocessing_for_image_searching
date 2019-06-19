#include "ROI_img.h"
#include "ppfis.h"
using namespace std;

RE_Matching func(cv::Point matchLoc, cv::Mat temp, double Max_Score, int index) {
	RE_Matching Matching;
	Matching.matchLoc = matchLoc;
	Matching.re_temp = temp;
	Matching.Max_score = Max_Score;
	Matching.index = index;
	return Matching;
}

// Original template Matching
RE_Matching ROI_Temp_img(cv::Mat img, cv::Mat templ)
{
	cv::Mat img_1, img_2, img_3, img_4, img_5;
	cv::Mat templ_1, templ_2, templ_3, templ_4, templ_5;
	cv::Mat Temp_T = templ.clone();
	cv::Mat result_1, result_2, result_3, result_4, result_5;
	RE_Matching Matching;
	double minVal; double maxVal = 0;
	cv::Point minLoc(-1, -1); cv::Point maxLoc(-1, -1);

	// ROI 105%
	resize(img, img_1, cv::Size(), 1.05, 1.05);
	matchTemplate(img_1, Temp_T, result_1, cv::TM_CCORR_NORMED);
	minMaxLoc(result_1, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
	cv::Point matchLoc_1;
	matchLoc_1 = maxLoc;
	matchLoc_1.x = matchLoc_1.x * 0.95;
	matchLoc_1.y = matchLoc_1.y * 0.95;
	double max_scores_1 = 100 * maxVal;
	double min_scores_1 = 100 * (1 - minVal);
	
	// ROI 100%
	resize(img, img_2, cv::Size(), 1.00, 1.00);
	matchTemplate(img_2, Temp_T, result_2, cv::TM_CCORR_NORMED);
	minMaxLoc(result_2, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat()); 
	cv::Point matchLoc_2;
	matchLoc_2 = maxLoc;
	matchLoc_2.x = matchLoc_2.x* 1.00;
	matchLoc_2.y = matchLoc_2.y* 1.00;
	double max_scores_2 = 100 * maxVal;
	double min_scores_2 = 100 * (1 - minVal);

	// ROI 95%
	resize(img, img_3, cv::Size(), 0.95, 0.95);
	matchTemplate(img_3, Temp_T, result_3, cv::TM_CCORR_NORMED);
	minMaxLoc(result_3, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
	cv::Point matchLoc_3;
	matchLoc_3 = maxLoc;
	matchLoc_3.x = matchLoc_3.x* 1.05;
	matchLoc_3.y = matchLoc_3.y* 1.05;
	double max_scores_3 = 100 * maxVal;
	double min_scores_3 = 100 * (1 - minVal);

	// ROI 90%
	resize(img, img_4, cv::Size(), 0.90, 0.90);
	matchTemplate(img_4, Temp_T, result_4, cv::TM_CCORR_NORMED);
	minMaxLoc(result_4, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
	cv::Point matchLoc_4;
	matchLoc_4 = maxLoc;
	matchLoc_4.x = matchLoc_4.x * 1.10;
	matchLoc_4.y = matchLoc_4.y * 1.10;
	double max_scores_4 = 100 * maxVal;
	double min_scores_4 = 100 * (1 - minVal);

	// ROI 85%
	resize(img, img_5, cv::Size(), 0.85, 0.85);
	matchTemplate(img_5, Temp_T, result_5, cv::TM_CCORR_NORMED);
	minMaxLoc(result_5, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
	cv::Point matchLoc_5;
	matchLoc_5 = maxLoc;
	matchLoc_5.x = matchLoc_5.x * 1.15;
	matchLoc_5.y = matchLoc_5.y * 1.15;
	double max_scores_5 = 100 * maxVal;
	double min_scores_5 = 100 * (1 - minVal);

	resize(Temp_T, templ_1, cv::Size(), 0.95, 0.95);
	resize(Temp_T, templ_2, cv::Size(), 1.00, 1.00);
	resize(Temp_T, templ_3, cv::Size(), 1.05, 1.05);
	resize(Temp_T, templ_4, cv::Size(), 1.10, 1.10);
	resize(Temp_T, templ_5, cv::Size(), 1.15, 1.15);

	// only output MAX values
	std::vector<double> vi{ max_scores_1, max_scores_2, max_scores_3, max_scores_4, max_scores_5 };
	double Max_Temp = *max_element(vi.begin(), vi.end());
	double Min_Temp = *min_element(vi.begin(), vi.end());

	if (Max_Temp == max_scores_1)
	{
		RE_Matching Matching = func(matchLoc_1, templ_1, max_scores_1, 1);
		return Matching;
	}
	else if (Max_Temp == max_scores_2) 
	{
		RE_Matching Matching = func(matchLoc_2, templ_2, max_scores_2,2);
		return Matching;
	}
	else if (Max_Temp == max_scores_3) 
	{
		RE_Matching Matching = func(matchLoc_3, templ_3, max_scores_3,3);
		return Matching;
	}
	else if (Max_Temp == max_scores_4)
	{
		RE_Matching Matching = func(matchLoc_4, templ_4, max_scores_4,4);
		return Matching;
	}
	else if (Max_Temp == max_scores_5)
	{
		RE_Matching Matching = func(matchLoc_5, templ_5, max_scores_5,5);
		return Matching;
	}

}

// Image_Processing (Gray + LUT(Brightness) + OTSU_Threshold + Opening_Filter)
void Image_Processing(cv::Mat & temp1_T, float gamma)
{
	using namespace ppfis;

	mask m(&temp1_T.data, temp1_T.rows, temp1_T.cols);
	m.set_thread_count(0); //run on no thread

	// Gray Image
	grayscale(m);

	// simple brightness, light and shade adjustment
	void (*brightness_func)(pixel& p) = [](pixel& p)
	{
		constexpr uchar brightness = 6;

		p.r = std::min(p.r + brightness, 255);
		p.g = std::min(p.g + brightness, 255);
		p.b = std::min(p.b + brightness, 255);
	};
	m.operate(brightness_func);

	cv::Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0); // generting this lookup table every time seems redundant!
	LUT(temp1_T, lookUpTable, temp1_T);
	
	// OTSU_Threshold 
	otsu_threshold(m);

	// Opening_Filtering
	opening(m);
}
