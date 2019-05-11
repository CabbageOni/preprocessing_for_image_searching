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

    // example implementation of image processing
    inline void grayscale(uchar* image, int width, int height)
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
