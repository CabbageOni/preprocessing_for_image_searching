#include <opencv2/opencv.hpp>
#include <iostream>

#include "ppfis.h"

using namespace std;
using namespace cv;
using namespace ppfis;

// compile with:
// g++ main.cpp -o run.exe $(pkg-config opencv --cflags --libs) -std=c++11

int main(int argc, char** argv)
{
    // somehow, my laptop doesn't support video.
    // please check if this works!
    /*
    if (argc != 2)
    {
        cout << "usage: " << argv[0] << " <Video_Path>" << endl;
        return -1;
    }

    VideoCapture cap(argv[1]);

    if (!cap.isOpened())
    {
        cout << "video file " << argv[1] << " could not be opened." << endl;
        return -1;
    }*/

    if (argc != 2)
    {
        cout << "usage: " << argv[0] << " <Image_Path>" << endl;
        return -1;
    }

    Mat image;
    image = imread(argv[1]);

    if (image.empty())
    {
        cout << "image file " << argv[1] << " could not be opened." << endl;
        return -1;
    }

    uchar* image_data = image.data;
    int width = image.rows, height = image.cols;

    grayscale(image_data, width, height);

    imwrite("output.jpg", image);

    return 0;
}
