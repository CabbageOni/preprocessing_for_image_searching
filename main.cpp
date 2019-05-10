#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
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
    }

    return 0;
}

// g++ main.cpp -o run.exe $(pkg-config opencv --cflags --libs) -std=c++11