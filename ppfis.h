#include <algorithm>
#include <pthread.h>

typedef unsigned char uchar;

#define PPFIS_RUN_THREAD(thread, func, parameter) \
if (err = pthread_create(&thread, nullptr, reinterpret_cast<void*(*)(void*)>(func), reinterpret_cast<void*>(parameter))) \
    return false

#define PPFIS_WAIT_THREAD(thread, return_target) \
pthread_join(thread, (void**)return_target)

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

    // change this later to masked_image, and contain image pointer too
    // that way will be more flexible
    class mask
    {
    private:
        struct operate_thread_param
        {
            uchar* image;
            int image_height;
            int left, right, top, bottom;
            void (*per_pixel_func)(pixel&);
        };

        static void operate_thread(operate_thread_param* param);

    public:
        int border_left = 0;
        int border_top = 0;
        int border_right = 0;
        int border_bottom = 0;

        inline mask(int width, int height) : border_right(width), border_bottom(height) {}
        inline mask(int border_left, int border_top, int border_right, int border_bottom) :
        border_left(border_left), border_top(border_top), border_right(border_right), border_bottom(border_bottom) {}
        bool operate(uchar* image, int image_width, int image_height, void (*per_pixel_func)(pixel& current_pixel)) const;
    };

    inline void mask::operate_thread(mask::operate_thread_param* param)
    {
        uchar*& image = param->image;
        int& image_height = param->image_height;
        int& left = param->left, &right = param->right, &top = param->top, &bottom = param->bottom;
        void (*per_pixel_func)(pixel& current_pixel) = param->per_pixel_func;

        for (int r = left; r < right; ++r)
            for (int c = top; c < bottom; ++c)
                per_pixel_func(*reinterpret_cast<pixel*>(&image[r * image_height * 3 + c * 3]));
    }

    inline bool mask::operate(uchar* image, int image_width, int image_height, void (*per_pixel_func)(pixel& current_pixel)) const
    {
        // do nothing if there is no function
        if (!per_pixel_func) return false;

        int err = 0;
        pthread_t thread_1, thread_2;

        // map ranges due to borders
        int right = std::min(std::max(border_left, border_right), image_width);
        int left = std::max(std::min(border_left, right), 0);
        int bottom = std::min(std::max(border_top, border_bottom), image_height);
        int top = std::max(std::min(border_top, bottom), 0);

        // temporary division just for two threads, will change later
        int mid = (left + right) / 2;

        operate_thread_param thread_param_1 = {image,image_height,left,mid,top,bottom,per_pixel_func};
        operate_thread_param thread_param_2 = {image,image_height,mid,right,top,bottom,per_pixel_func};

        PPFIS_RUN_THREAD(thread_1, &mask::operate_thread, &thread_param_1);
        PPFIS_RUN_THREAD(thread_2, &mask::operate_thread, &thread_param_2);

        PPFIS_WAIT_THREAD(thread_1, nullptr);
        PPFIS_WAIT_THREAD(thread_2, nullptr);

        return true;
    }
}