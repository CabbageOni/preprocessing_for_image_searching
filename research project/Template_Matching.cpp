#include "ROI_img.h"
#include "ppfis.h"

// compile with:
// g++ -pthread main.cpp -o run.exe $(pkg-config opencv --cflags --libs) -std=c++17

#define gamma 3.0

using namespace std;
using namespace cv;
//using namespace cv::xfeatures2d;
using namespace ppfis;

// successful run count 
int index = 0;

// Image
Mat img;
Mat img_display;

// Templete_Image
Mat templ, Proc_temp1_T;

// Video Output
VideoWriter Video_output;

// Image Windows Name
const char* image_window = "Source Image";
const char* result_window = "Result window";

//void MatchingMethod(int, void*);
void built_in_function_examples(mask& m);
void simple_thread_exmaples();

void built_in_function_examples(mask& m)
{
	m.set_relative_border(0, 0, 100, 100);
	m.set_relative_border(50, 50, 50, 50);
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		std::cout << "usage: " << argv[0] << " <Image_Path>" << endl;
		return -1;
	}

	Mat image;
	image = imread(argv[1]);

	if (image.empty())
	{
		std::cout << "image file " << argv[1] << " could not be opened." << endl;
		return -1;
	}

	int width = image.rows, height = image.cols;

	mask m(&image.data, width, height);

	built_in_function_examples(m);
	cv::imwrite("output.jpg", image);


	// Read video and templates
	VideoCapture cap1("../../Data/Video.avi");
	if (!cap1.isOpened())
	{
		printf("Could not successfully read video. \n");
		return -1;
	}

	// Read template image
	templ = imread("../../Data/Temp_img.jpg", IMREAD_COLOR);
	if (templ.empty())
	{
		std::cout << "Can't read one of the images" << endl;
		return -1;
	}

	// Video saving properties setup
	Video_output.open("../../Data/Video_output.avi", VideoWriter::fourcc('D', 'I', 'V', 'X'), 10, Size(cap1.get(CAP_PROP_FRAME_WIDTH), cap1.get(CAP_PROP_FRAME_HEIGHT)));

	///==========================================================================================
	// Image_Processing( templete, output_templete, gamma) 
	Image_Processing(templ, Proc_temp1_T, gamma);
	// Template processing part.
	// Template color image with 4 steps (grayscaale -> brightness ->  OTSU_Threshold -> Opening_Filtering)
	///==========================================================================================

	// Window_Size
	namedWindow(image_window, WINDOW_AUTOSIZE);

	///==========================================================================================
	// cature the video and match.
	clock_t t1; // duraction check
	while (char(waitKey(1)) != 'q') {
		cap1 >> img;
		t1 = clock();
		//Check if the video is over
		if (img.empty())
		{
			std::cout << "Video over" << endl;
			break;
		}
		else
		{
			//Mat img_display;
			img.copyTo(img_display);
			simple_thread<4, int, int, int, int> t;

			void(*MatchingMethod)(int, int, int, int) = [](int ROI_LEFT_X, int ROI_LEFT_Y, int ROI_RIGHT_X, int ROI_RIGHT_Y)
			{
				Mat roiImg, Proc_roiImg, Origin_img;;
				// copy image
				//Mat img_display, 
				Mat img_roi;
				img.copyTo(img_display);
				img.copyTo(img_roi);

				int result_cols = img.cols - templ.cols + 1;
				int result_rows = img.rows - templ.rows + 1;
				// Roi Size
				Rect roi, re_roi;

				// Roi Point
				Point Roi_point;
				RE_Matching Temp_Loc_Max;
				///==========================================================================================
				// imge ROI settings: 700, 500  1520, 880 [resolution: 1920 x 1080] 
				// to divide ROI into small parts, do it here, and setup afterwards. 
				roi = Rect(Point(ROI_LEFT_X, ROI_LEFT_Y), Point(ROI_RIGHT_X, ROI_RIGHT_Y));
				// ROI_Image(820 x 380)
				roiImg = img_roi(roi);     // matching through this part
				Origin_img = img_roi(roi); // to save COlOR searching part
				///==========================================================================================
				
				///==========================================================================================
				// Image_Processing(ROI_Image, output_ROI_Image, gamma) 
				// only operate on ROI (for each video)
				Image_Processing(roiImg, Proc_roiImg, gamma);
				// 4 steps of ROI color image (grayscale -> brightness ->  OTSU_Threshold -> Opening_Filtering)
				///==========================================================================================

				///==========================================================================================
				// Original template Matching (image-processed ROI image, image-processed template image) 
				Temp_Loc_Max = ROI_Temp_img(Proc_roiImg, Proc_temp1_T);
				//std::cout << "score : " << Temp_Loc_Max.Max_score << endl;
				// calculate score and template point(x,y) output
				///==========================================================================================
				// consider only above 90 matching score
				if (Temp_Loc_Max.Max_score > 90)
				{
					// if completely elsewhere, ignore
					if (Roi_point.x - Temp_Loc_Max.matchLoc.x < templ.cols && Roi_point.y - Temp_Loc_Max.matchLoc.y < templ.rows)
					{
						// adjust Point 
						Roi_point = Temp_Loc_Max.matchLoc;
						///==========================================================================================
						// draw a box on good Matching
						// draw a box on image (image, point, other side point, color, thickness, type, shift) 	
						rectangle(img_display, Point(Temp_Loc_Max.matchLoc.x + ROI_LEFT_X, Temp_Loc_Max.matchLoc.y + ROI_LEFT_Y), Point(Temp_Loc_Max.matchLoc.x + Temp_Loc_Max.re_temp.cols + ROI_LEFT_X, Temp_Loc_Max.matchLoc.y + Temp_Loc_Max.re_temp.rows + ROI_LEFT_Y), Scalar(0, 0, 255), 2, 8, 0);
						//rectangle(img_display, Point(matchLoc.x, matchLoc.y), Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 0, 255), 2, 8, 0); draw a box on image (image, point, other side point, color, thickness, type, shift) 	
						///==========================================================================================

						///==========================================================================================
						// from ROI image, RE_ROI on template part (image that wasn't image-processed)
						re_roi = Rect(Point(Temp_Loc_Max.matchLoc.x, Temp_Loc_Max.matchLoc.y), Point(Temp_Loc_Max.matchLoc.x + Temp_Loc_Max.re_temp.cols, Temp_Loc_Max.matchLoc.y + Temp_Loc_Max.re_temp.rows));
						///==========================================================================================

					}
					// matching count computation
					index++;
				}
				// save video
				//Video_output.write(img_display);

				// show result image
				//cv::imshow(image_window, img_display);
				//return true;
			};
			int x1 = 0, y1 = 0;
			int x2 = 850, y2 = 360;
			int x3 = 850, y3 = 0;
			int x4 = 0, y4 = 360;

			// image ROI setup: [850 x 450 resolution per ROI] [original resolution: 1920 x 1080] 
			t.run(MatchingMethod, 300, 300, 1150, 750); 
			t.run(MatchingMethod, 950, 550, 1800,1000);
			t.run(MatchingMethod, 950, 300, 1800, 750); 
			t.run(MatchingMethod, 300, 550, 1150, 1000); 
			t.wait();

			t1 = clock() - t1; 
			std::cout << "it takes " << (((float)t1) / CLOCKS_PER_SEC) * 1000 << " ms to capture a frame. The capture rate can reach " << 1 / (((float)t1) / CLOCKS_PER_SEC) \
				<< " FPS" << std::endl;

			// save video
			Video_output.write(img_display);

			// show result image
			cv::imshow(image_window, img_display);
		}
	}
	cv::waitKey((((float)t1) / CLOCKS_PER_SEC) * 1000); // time check
	cap1.release();
	///==========================================================================================

	return 0;
}
