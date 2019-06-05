#include <opencv2/opencv.hpp>
#include <iostream>

#include "ppfis.h"

using namespace std;
using namespace cv;
using namespace ppfis;

// compile with:
// g++ -pthread main.cpp -o run.exe $(pkg-config opencv --cflags --libs) -std=c++17

void built_in_function_examples(mask& m);
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

    mask m(&image.data, width, height);

    built_in_function_examples(m);
    imwrite("output.jpg", image);

    simple_thread_exmaples();

    return 0;
}

void built_in_function_examples(mask& m)
{
    m.set_relative_border(50,50,50,50);
    mean_filter(m, 15);
}

void simple_thread_exmaples()
{
    cout << "simple thread test 1 ===" << endl;
    simple_thread<3, int> t([](int num){ cout << "test " << num << endl; });

    t.run(10);
    t.run(20);
    t.run(30);
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
    simple_thread<t3_thread_count, int*, int> t3([](int* result, int n)
    {
        *result = 0;
        for (int i = 0; i < n; ++i)
            *result += i;
    });

    int t3_results[t3_thread_count] = { 0 };

    for (int i = 0; i < t3_thread_count; ++i)
        t3.run(&t3_results[i], 5 * i);
    t3.wait();

    int t3_result = 0;
    for (int& r : t3_results)
        t3_result += r;
    cout << "test3 " << t3_result << endl;
} 