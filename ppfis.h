#include <algorithm>
#include <pthread.h>
#include <tuple>
#include <cmath>

typedef unsigned char uchar;

#define PPFIS_RUN_THREAD(thread, func, parameter) \
if (err = pthread_create(&thread, nullptr, reinterpret_cast<void*(*)(void*)>(func), reinterpret_cast<void*>(parameter))) \
    return false

#define PPFIS_WAIT_THREAD(thread, return_target) \
pthread_join(thread, (void**)return_target)

namespace ppfis
{
    // pixel is based on CV_8UC3
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

    class pixels;
    class row_pixels;

    class mask
    {
    private:
        struct operate_per_pixel_thread_param
        {
            uchar* image;
            int image_width;
            int left, right, top, bottom;
            void (*per_pixel_func)(pixel&);
        };

        uchar** m_image_ptr = nullptr;
        int m_image_width = -1, m_image_height = -1;

        static void operate_per_pixel_thread(operate_per_pixel_thread_param* param);

        friend pixels;
        friend row_pixels;
    public:
        int border_left = 0;
        int border_top = 0;
        int border_right = 0;
        int border_bottom = 0;

        mask(uchar** image_ptr, int image_width, int image_height);
        void set_border(int left, int top, int right, int bottom);
        void set_relative_border(int d_left, int d_top, int d_right, int d_bottom);

        bool operate(void (*per_pixel_func)(pixel& current_pixel)) const; 
        bool operate(void (*per_pixel_func)(pixels& original_pixel, pixel& output_pixel));
    };

    class row_pixels
    {
    private:
        const mask* m_mask_ptr;
        int m_current_row;
        int m_current_column;
    
        inline row_pixels(int current_row, int current_column, const mask* mask_ptr) : m_current_row(current_row), m_current_column(current_column), m_mask_ptr(mask_ptr) {}
        
        friend pixels;

    public:
        inline const pixel& operator[](int relative_column)
        {
            int row = std::max(0,std::min(m_mask_ptr->m_image_width, m_current_row));
            int column = std::max(0,std::min(m_mask_ptr->m_image_height, m_current_column + relative_column));
            return reinterpret_cast<pixel*>(*m_mask_ptr->m_image_ptr)[column * m_mask_ptr->m_image_width + row];
        }
    };

    class pixels
    {
    private:
        const mask* m_mask_ptr;
        int m_current_row;
        int m_current_column;

        inline pixels() {}

        friend mask;
    public:
        inline const pixel& at(int relative_row, int relative_column)
        {
            int row = std::max(0,std::min(m_mask_ptr->m_image_width, m_current_row + relative_row));
            int column = std::max(0,std::min(m_mask_ptr->m_image_height, m_current_column + relative_column));
            return reinterpret_cast<pixel*>(*m_mask_ptr->m_image_ptr)[column * m_mask_ptr->m_image_width + row];
        }

        inline row_pixels operator[](int relative_row) const { return row_pixels(m_current_row + relative_row, m_current_column, m_mask_ptr); }
    };

    inline void mask::operate_per_pixel_thread(operate_per_pixel_thread_param* param)
    {
        uchar*& image = param->image;
        int& image_width = param->image_width;
        int& left = param->left, &right = param->right, &top = param->top, &bottom = param->bottom;
        void (*per_pixel_func)(pixel& current_pixel) = param->per_pixel_func;

        for (int c = top; c < bottom; ++c)
            for (int r = left; r < right; ++r)
                per_pixel_func(reinterpret_cast<pixel*>(image)[c * image_width + r]);
    }

    inline mask::mask(uchar** image_ptr, int image_width, int image_height) : m_image_ptr(image_ptr), m_image_width(image_height), m_image_height(image_width)
    {
        border_left = 0;
        border_top = 0;
        border_right = image_height;
        border_bottom = image_width;
    }

    inline void mask::set_border(int left, int top, int right, int bottom)
    {
        border_left = left;
        border_top = top;
        border_right = right;
        border_bottom = bottom;
    }

    inline void mask::set_relative_border(int d_left, int d_top, int d_right, int d_bottom)
    {
        border_left = 0 + d_left;
        border_top = 0 + d_top;
        border_right = m_image_width - d_right;
        border_bottom = m_image_height - d_bottom;
    }

    inline bool mask::operate(void (*per_pixel_func)(pixel& current_pixel)) const
    {
        if (!m_image_ptr || !per_pixel_func) return false;

        int err = 0;
        pthread_t thread_1, thread_2;

        // map ranges due to borders
        int right = std::min(std::max(border_left, border_right), m_image_width);
        int left = std::max(std::min(border_left, right), 0);
        int bottom = std::min(std::max(border_top, border_bottom), m_image_height);
        int top = std::max(std::min(border_top, bottom), 0);

        // temporary division just for two threads, will change later
        // also, make main core run too
        int mid = (bottom + top) / 2;

        operate_per_pixel_thread_param thread_param_1 = {*m_image_ptr,m_image_width,left,right,top,mid,per_pixel_func};
        operate_per_pixel_thread_param thread_param_2 = {*m_image_ptr,m_image_width,left,right,mid,bottom,per_pixel_func};

        PPFIS_RUN_THREAD(thread_1, &mask::operate_per_pixel_thread, &thread_param_1);
        PPFIS_RUN_THREAD(thread_2, &mask::operate_per_pixel_thread, &thread_param_2);

        PPFIS_WAIT_THREAD(thread_1, nullptr);
        PPFIS_WAIT_THREAD(thread_2, nullptr);

        return true;
    }

    inline bool mask::operate(void (*per_pixel_func)(pixels& original_pixel, pixel& output_pixel))
    {
        if (!m_image_ptr || !per_pixel_func) return false;

        // create new image
        pixel* new_image = new pixel[m_image_width * m_image_height];
        memcpy(new_image, *m_image_ptr, m_image_width * m_image_height * 3);

        // map ranges due to borders
        int right = std::min(std::max(border_left, border_right), m_image_width);
        int left = std::max(std::min(border_left, right), 0);
        int bottom = std::min(std::max(border_top, border_bottom), m_image_height);
        int top = std::max(std::min(border_top, bottom), 0);

        pixels op;
        op.m_mask_ptr = this;
        for (int c = top; c < bottom; ++c)
            for (int r = left; r < right; ++r)
            {
                op.m_current_column = c;
                op.m_current_row = r;
                per_pixel_func(op, new_image[c * m_image_width + r]);
            }

        // copy result
        memcpy(*m_image_ptr, new_image, m_image_width * m_image_height * 3);
        delete[] new_image;
        return true;
    }

    //example of built-in grayscale function
    inline void grayscale(mask& m)
    {
        m.operate([](pixel& p)
        {
            uchar gray = uchar((int(p.r) + int(p.g) + int(p.b))/3);
            p = gray;
        });
    }
    
    inline void sobel_operator(mask& m)
    {
        m.operate([](pixels& op, pixel& np)
        {
            int f[] = {1, 2, 1};

            int x_r = 0, x_g = 0, x_b = 0;
            for (int row = -1; row < 2; row++)
                for (int col = -1; col < 2; col++)
                {
                    const pixel& p = op.at(row, col);
                    x_r = f[row+1] * col * p.r;
                    x_g = f[row+1] * col * p.g;
                    x_b = f[row+1] * col * p.b;
                }

            int y_r = 0, y_g = 0, y_b = 0;
            for (int row = -1; row < 2; row++)
                for (int col = -1; col < 2; col++)
                {
                    const pixel& p = op.at(row, col);
                    y_r = f[col+1] * row * p.r;
                    y_g = f[col+1] * row * p.g;
                    y_b = f[col+1] * row * p.b;
                }
            
            int mag_r = 0, mag_g = 0, mag_b = 0;
            mag_r = sqrt(pow(x_r, 2) + pow(y_r, 2));
            mag_g = sqrt(pow(x_g, 2) + pow(y_g, 2));
            mag_b = sqrt(pow(x_b, 2) + pow(y_b, 2));

            np = pixel(mag_b, mag_g, mag_r);
        });
    }

    inline void mean_filter(mask& m, int k)
    {
        // k cannot be inside the lambda function
        // needs better solution
        /*
        m.operate([](pixels& op, pixel& np)
        {
            int r = 0, g = 0, b = 0;
            int size = (k-1)/2;
            for (int row = -1 * size; row < size + 1; row++)
                for (int col = -1 * size; col < size + 1; col++)
                {
                    const pixel& p = op.at(row, col);
                    r += p.r;
                    g += p.g;
                    b += p.b;
                }
            np = pixel(b/pow(k, 2), g/pow(k, 2), r/pow(k, 2));
        });*/
    }

    inline void median_filter(mask& m, int k)
    {
        // k cannot be inside the lambda function
        // needs better solution
        /*
        m.operate([](pixels& op, pixel& np)
        {
            int r[pow(k, 2)] = {0}, g[pow(k, 2)] = {0}, b[pow(k, 2)] = {0};
            int size = (k-1)/2;
            int i = 0;

            for (int row = -1 * size; row < size + 1; row++)
                for (int col = -1 * size; col < size + 1; col++)
                {
                    const pixel& p = op.at(row, col);
                    r[i] = p.r;
                    g[i] = p.g;
                    b[i] = p.b;
                    i++;
                 }
            sort(r, r + pow(k, 2)); 
            sort(g, b + pow(k, 2)); 
            sort(g, b + pow(k, 2)); 

            np = pixel(b[size+1], g[size+1], r[size+1]);
        });*/
    }

    // Simple thread managing class
    template <int max_thread_count = 5, typename ... parameters>
    class simple_thread
    {
    private:
        pthread_t m_threads[max_thread_count];
        struct thread_parameter
        {
            void (*func)(parameters...);
            std::tuple<parameters...> params;
        } m_thread_parameters[max_thread_count];
        size_t m_current_thread = 0;

        static inline void run_thread(void* param)
        {
            thread_parameter* p = reinterpret_cast<thread_parameter*>(param);
            std::apply(p->func, p->params);
        }

    public:
        // Lock is not garunteed, proceed with caution with shared variables!
        inline bool run(void (*func)(parameters...), parameters ... params)
        {
            if (m_current_thread == max_thread_count)
                return false;

            m_thread_parameters[m_current_thread] = { func, std::forward_as_tuple(params...) };

            if(pthread_create(&m_threads[m_current_thread], nullptr, reinterpret_cast<void*(*)(void*)>(run_thread), &m_thread_parameters[m_current_thread]))
                return false;

            ++m_current_thread;
            return true;   
        }

        // Note that there is no return variable.
        // If needed, utilized parameters instead.
        inline void wait()
        {
            for (size_t i = 0; i < m_current_thread; ++i)
                pthread_join(m_threads[i], nullptr);

            m_current_thread = 0;
        }
    };
}
