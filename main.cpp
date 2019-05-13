#include <opencv2/opencv.hpp>
#include <iostream>

#include "ppfis.h"

using namespace std;
using namespace cv;
using namespace ppfis;

// compile with:
// g++ -pthread main.cpp -o run.exe $(pkg-config opencv --cflags --libs) -std=c++11

int main(int argc, char** argv)
{
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

    int width = image.rows, height = image.cols;

    // example of using mask to do grayscale
    mask m(50, 50, width - 50, height - 50);
    m.operate(image.data, width, height, [](pixel& p)
    {
        uchar gray = uchar((int(p.r) + int(p.g) + int(p.b))/3);
        p = gray;
    });

    // example of using mask to only show red
    mask m2(100, 0, width, height - 100);
    m2.operate(image.data, width, height, [](pixel& p)
    {
        p = {0,0,p.r};
    });

    imwrite("output.jpg", image);

    return 0;
}
