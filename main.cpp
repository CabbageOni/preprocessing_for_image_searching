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
    mask m(&image.data, width, height);
    m.set_relative_border(50, 50, 50, 50);
    m.operate([](pixel& p)
    {
        uchar gray = uchar((int(p.r) + int(p.g) + int(p.b))/3);
        p = gray;
    });

    // example of using mask to only show red
    m.set_relative_border(100, 0, 0, 100);
    m.operate([](pixel& p)
    {
        p = {p.b,0,p.r};
    });

    // example of using mask to compute blur
    m.set_relative_border(0,150,150,0);
    m.operate([](pixels& op, pixel& np)
    {
        int r = 0, g = 0, b = 0;
        for (int row = -3; row < 4; ++row)
            for (int col = -3; col < 4; ++col)
            {
                const pixel& p = op[row][col];
                // below is same as above, more efficient. consider using this
                // pixel& p = op.at(row, col);
                r += p.r;
                g += p.g;
                b += p.b;
            }
        
        np = pixel(b/49, g/49, r/49);
    });

    // example of using simple thread
    simple_thread<int> t;

    t.run([](int num){ cout << "test " << num << endl; }, 1);
    t.run([](int num){ cout << "test " << num << endl; }, 2);
    t.run([](int num){ cout << "test " << num << endl; }, 3);
    t.wait();

    imwrite("output.jpg", image);

    return 0;
}
