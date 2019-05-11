#include <algorithm>

typedef unsigned char uchar;

namespace ppfis
{
    // pixel is based on CV_8UC3
    // support diverse channel and size later!
    union pixel
    {
        uchar data[3] = {0};
        struct
        {
            uchar x, y, z;
        };

        struct 
        {
            uchar b, g, r;
        };
        
        inline pixel() {}
        inline pixel(uchar uniform) : x(uniform), y(uniform), z(uniform) {}
        inline pixel(uchar x, uchar y, uchar z) : x(x), y(y), z(z) {}

        inline uchar& operator[](const int index) { return data[index]; }
        inline bool operator==(const pixel& rhs) const { return x == rhs.x && y == rhs.y && z == rhs.z; }
    };

    class mask
    {
    public:
        int border_left = 0;
        int border_top = 0;
        int border_right = 0;
        int border_bottom = 0;

        inline mask(int width, int height) : border_right(width), border_bottom(height) {}
        inline mask(int border_left, int border_top, int border_right, int border_bottom) :
        border_left(border_left), border_top(border_top),
        border_right(border_right), border_bottom(border_bottom)
        {}
        void operate(uchar* image, int width, int height, void (*per_pixel_func)(pixel&)) const;
    };

    inline void mask::operate(uchar* image, int width, int height, void (*per_pixel_func)(pixel&)) const
    {
        // do nothing if there is no function
        if (!per_pixel_func) return;

        // map ranges due to borders
        int right = std::min(std::max(border_left, border_right), width);
        int left = std::max(std::min(border_left, right), 0);
        int bottom = std::min(std::max(border_top, border_bottom), height);
        int top = std::max(std::min(border_top, bottom), 0);

        for (int r = left; r < right; ++r)
            for (int c = top; c < bottom; ++c)
                per_pixel_func(*reinterpret_cast<pixel*>(&image[r * height * 3 + c * 3]));
    }

    // example implementation of raw image processing
    // example of masking is inside main.cpp!
    inline void raw_grayscale(uchar* image, int width, int height)
    {
        for (int r = 0; r < width; ++r)
        {
            for (int c = 0; c < height; ++c)
            {
                pixel& curr_pixel = *reinterpret_cast<pixel*>(&image[r * height * 3 + c * 3]);
                uchar gray = uchar((int(curr_pixel.r) + int(curr_pixel.g) + int(curr_pixel.b))/3);
                curr_pixel = gray;
            }            
        }
    }
}