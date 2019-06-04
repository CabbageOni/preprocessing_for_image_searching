#include <opencv2/opencv.hpp>
#include <iostream>

#include "ppfis.h"

using namespace std;
using namespace cv;
using namespace ppfis;

// compile with:
// g++ -pthread main.cpp -o run.exe $(pkg-config opencv --cflags --libs) -std=c++17

void simple_thread_exmaples();

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

    imwrite("output.jpg", image);

    simple_thread_exmaples();

    return 0;
}

void simple_thread_exmaples()
{
    cout << "simple thread test 1 ===" << endl;
    simple_thread<3, int> t;

    t.run([](int num){ cout << "test " << num << endl; }, 10);
    t.run([](int num){ cout << "test " << num << endl; }, 20);
    t.run([](int num){ cout << "test " << num << endl; }, 30);
    t.wait();

    cout << "simple thread test 2 ===" << endl;
    simple_thread<5, int, float> t2;

    void (*t2_func)(int, float) = [](int num, float num2)
    {
        float sum = 0;
        for (int i = 0; i < num; ++i)
            sum += num2;
        cout << "test2 " << num << " " << sum << endl;
    };

    t2.run(t2_func, 1, 0.5f);
    t2.run(t2_func, 2, 1.5f);
    t2.run(t2_func, 3, 2.5f);
    t2.run(t2_func, 4, 3.5f);
    t2.run(t2_func, 5, 4.5f);
    t2.wait();

    cout << "simple thread test 3 ===" << endl;
    constexpr size_t t3_thread_count = 10;
    simple_thread<t3_thread_count, int*, int> t3;

    void (*t3_func)(int*, int) = [](int* result, int n)
    {
        *result = 0;
        for (int i = 0; i < n; ++i)
            *result += i;
    };

    int t3_results[t3_thread_count] = { 0 };

    for (int i = 0; i < t3_thread_count; ++i)
        t3.run(t3_func, &t3_results[i], 5 * i);
    t3.wait();

    int t3_result = 0;
    for (int& r : t3_results)
        t3_result += r;
    cout << "test3 " << t3_result << endl;
} 