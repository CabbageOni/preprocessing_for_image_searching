#include <algorithm>
#include <pthread.h>
#include <tuple>
#include <cmath>
#include <vector>

typedef unsigned char uchar;

namespace ppfis
{
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
        void (*default_func)(parameters...) = nullptr;

        inline simple_thread(void (*func)(parameters...) = nullptr) : default_func(func) { }

        // Lock is not garunteed, proceed with caution with shared variables!
        inline bool run(void (*func)(parameters...), parameters ... params)
        {
            if (!func)
                return false;

            if (m_current_thread == max_thread_count)
                return false;

            m_thread_parameters[m_current_thread] = { func, std::forward_as_tuple(params...) };

            if(pthread_create(&m_threads[m_current_thread], nullptr, reinterpret_cast<void*(*)(void*)>(run_thread), &m_thread_parameters[m_current_thread]))
                return false;

            ++m_current_thread;
            return true;   
        }

        inline bool run(parameters ... params)
        {
            return run(default_func, params...);
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
        constexpr static int m_mask_maximum_thread = 8;

        uchar** m_image_ptr = nullptr;
        int m_image_width = -1, m_image_height = -1;

        static void operate_per_pixel_thread(pixel* image, int image_width, void (*per_pixel_func)(pixel&), int left, int right, int top, int bottom);
        static void operate_per_pixel_thread(mask* m, pixel* new_image, void (*per_pixel_func)(pixels&, pixel&), int left, int right, int top, int bottom);
        template <typename ... parameters>
        static void operate_per_pixel_thread(pixel* image, int image_width, void (*per_pixel_func)(pixel&, parameters ... params), int left, int right, int top, int bottom, parameters ... params);
        template <typename ... parameters>
        static void operate_per_pixel_thread(mask* m, pixel* new_image, void (*per_pixel_func)(pixels&, pixel&, parameters ... params), int left, int right, int top, int bottom, parameters ... params);

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

        bool operate(void (*per_pixel_func)(pixel& current_pixel)); 
        bool operate(void (*per_pixel_func)(pixels& original_pixel, pixel& output_pixel));
        template <typename ... parameters>
        bool operate(void (*per_pixel_func)(pixel& current_pixel, parameters ... params), parameters ... params);
        template <typename ... parameters>
        bool operate(void (*per_pixel_func)(pixels& original_pixel, pixel& output_pixel, parameters ... params), parameters ... params);
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

    inline void mask::operate_per_pixel_thread(pixel* image, int image_width, void (*per_pixel_func)(pixel&), int left, int right, int top, int bottom)
    {
        for (int c = top; c < bottom; ++c)
            for (int r = left; r < right; ++r)
                per_pixel_func(image[c * image_width + r]);
    }

    inline void mask::operate_per_pixel_thread(mask* m, pixel* new_image, void (*per_pixel_func)(pixels&, pixel&), int left, int right, int top, int bottom)
    {
        pixels op;
        op.m_mask_ptr = m;
        for (int c = top; c < bottom; ++c)
            for (int r = left; r < right; ++r)
            {
                op.m_current_column = c;
                op.m_current_row = r;
                per_pixel_func(op, new_image[c * m->m_image_width + r]);
            }
    }

    template <typename ... parameters>
    inline void mask::operate_per_pixel_thread(pixel* image, int image_width, void (*per_pixel_func)(pixel&, parameters ...), int left, int right, int top, int bottom, parameters ... params)
    {
        for (int c = top; c < bottom; ++c)
            for (int r = left; r < right; ++r)
                per_pixel_func(image[c * image_width + r], params...);
    }

    template <typename ... parameters>
    inline void mask::operate_per_pixel_thread(mask* m, pixel* new_image, void (*per_pixel_func)(pixels&, pixel&, parameters ...), int left, int right, int top, int bottom, parameters ... params)
    {
        pixels op;
        op.m_mask_ptr = m;
        for (int c = top; c < bottom; ++c)
            for (int r = left; r < right; ++r)
            {
                op.m_current_column = c;
                op.m_current_row = r;
                per_pixel_func(op, new_image[c * m->m_image_width + r], params...);
            }
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

    inline bool mask::operate(void (*per_pixel_func)(pixel& current_pixel))
    {
        if (!m_image_ptr || !per_pixel_func) return false;

        // map ranges due to borders
        int right = std::min(std::max(border_left, border_right), m_image_width);
        int left = std::max(std::min(border_left, right), 0);
        int bottom = std::min(std::max(border_top, border_bottom), m_image_height);
        int top = std::max(std::min(border_top, bottom), 0);

        simple_thread<m_mask_maximum_thread, pixel*, int, void (*)(pixel&), int, int, int, int> t(mask::operate_per_pixel_thread);

        pixel* image = reinterpret_cast<pixel*>(*m_image_ptr);

        // estimate thread count
        int concurrent_operation_count = bottom - top > m_mask_maximum_thread + 1 ? m_mask_maximum_thread + 1 : 1;
        int height_per_thread = (bottom - top) / concurrent_operation_count;

        // run threads
        for (int i = 0; i < concurrent_operation_count - 1; ++i)
            t.run(image, m_image_width, per_pixel_func, left, right, top + height_per_thread * i, top + height_per_thread * (i + 1));
        operate_per_pixel_thread(image, m_image_width, per_pixel_func, left, right, top + height_per_thread * (concurrent_operation_count - 1), bottom);

        t.wait();

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

        simple_thread<m_mask_maximum_thread, mask*, pixel*, void (*)(pixels&, pixel&), int, int, int, int> t(mask::operate_per_pixel_thread);

        // estimate thread count
        int concurrent_operation_count = bottom - top > m_mask_maximum_thread + 1 ? m_mask_maximum_thread + 1 : 1;
        int height_per_thread = (bottom - top) / concurrent_operation_count;

        // run threads
        for (int i = 0; i < concurrent_operation_count - 1; ++i)
            t.run(this, new_image, per_pixel_func, left, right, top + height_per_thread * i, top + height_per_thread * (i + 1));
        operate_per_pixel_thread(this, new_image, per_pixel_func, left, right, top + height_per_thread * (concurrent_operation_count - 1), bottom);
        t.wait();

        // copy result
        memcpy(*m_image_ptr, new_image, m_image_width * m_image_height * 3);
        delete[] new_image;
        return true;
    }

    template <typename ... parameters>
    inline bool mask::operate(void (*per_pixel_func)(pixel& current_pixel, parameters ... params), parameters ... params)
    {
        if (!m_image_ptr || !per_pixel_func) return false;

        // map ranges due to borders
        int right = std::min(std::max(border_left, border_right), m_image_width);
        int left = std::max(std::min(border_left, right), 0);
        int bottom = std::min(std::max(border_top, border_bottom), m_image_height);
        int top = std::max(std::min(border_top, bottom), 0);

        simple_thread<m_mask_maximum_thread, pixel*, int, void (*)(pixel&, parameters...), int, int, int, int, parameters ... > t(mask::operate_per_pixel_thread);

        pixel* image = reinterpret_cast<pixel*>(*m_image_ptr);

        // estimate thread count
        int concurrent_operation_count = bottom - top > m_mask_maximum_thread + 1 ? m_mask_maximum_thread + 1 : 1;
        int height_per_thread = (bottom - top) / concurrent_operation_count;

        // run threads
        for (int i = 0; i < concurrent_operation_count - 1; ++i)
            t.run(image, m_image_width, per_pixel_func, left, right, top + height_per_thread * i, top + height_per_thread * (i + 1), params...);
        operate_per_pixel_thread(image, m_image_width, per_pixel_func, left, right, top + height_per_thread * (concurrent_operation_count - 1), bottom, params...);

        t.wait();
        return true;
    }

    template <typename ... parameters>
    inline bool mask::operate(void (*per_pixel_func)(pixels& original_pixel, pixel& output_pixel, parameters ... params), parameters ... params)
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

        simple_thread<m_mask_maximum_thread, mask*, pixel*, void (*)(pixels&, pixel&, parameters...), int, int, int, int, parameters ... > t(mask::operate_per_pixel_thread);

        // estimate thread count
        int concurrent_operation_count = bottom - top > m_mask_maximum_thread + 1 ? m_mask_maximum_thread + 1 : 1;
        int height_per_thread = (bottom - top) / concurrent_operation_count;

        // run threads
        for (int i = 0; i < concurrent_operation_count - 1; ++i)
            t.run(this, new_image, per_pixel_func, left, right, top + height_per_thread * i, top + height_per_thread * (i + 1), params...);
        operate_per_pixel_thread(this, new_image, per_pixel_func, left, right, top + height_per_thread * (concurrent_operation_count - 1), bottom, params...);

        t.wait();

        // copy result
        memcpy(*m_image_ptr, new_image, m_image_width * m_image_height * 3);
        delete[] new_image;
        return true;
    }

    inline void grayscale(mask& m)
    {
        m.operate([](pixel& p)
        {
            uchar gray = uchar((int(p.r) + int(p.g) + int(p.b))/3);
            p = gray;
        });
    }

    // threshold
    inline void threshold(mask& m, int threshold)
    {
        grayscale(m);

        constexpr void (*func)(pixel&, int) = [](pixel& p, int threshold)
        {
            if (p.r < threshold)
                p = 0;
            else
                p = 255; 
        };

        m.operate(func, threshold);
    }
    
    inline void compute_hist(mask& m, unsigned* hist)
    {
        // make it publicly available?

        constexpr void (*func)(pixel&, unsigned*) = [](pixel& p, unsigned* hist)
        {
            // atomicity warning!
            hist[p.r]++; 
        };

        m.operate(func, hist);
    }

    inline int compute_otsu(mask& m, unsigned *hist)
    {
        // Need to get the size of mask
        long int N = (m.border_right - m.border_left) * (m.border_bottom - m.border_top);
        int threshold = 0;

        float sum = 0;
        float sumB = 0;
        int q1 = 0;
        int q2 = 0;
        float varMax = 0;

        for (int i = 0; i <= 255; i++)
            sum += i * ((int)hist[i]);

        for (int i = 0; i <= 255; i++)
        {
            q1 += hist[i];
            if (q1 == 0)
                continue;

            q2 = N - q1;
            if (q2 == 0)
                break;

            sumB += (float) (i * ((int)hist[i]));
            float m1 = sumB / q1;
            float m2 = (sum - sumB) / q2;

            float varBetween = (float) q1 * (float) q2 * (m1 - m2) * (m1 - m2);

            if (varBetween > varMax) 
            {
                varMax = varBetween;
                threshold = i;
            }
        }

        return threshold;
    }

    inline void otsu_threshold(mask& m)
    {
        unsigned hist[256] = {0};
        
        compute_hist(m, hist);

        constexpr void (*func)(pixel&, int) = [](pixel& p, int threshold)
        {
            if (p.r < threshold)
                p = 0;
            else
                p = 255;  
        };

        m.operate(func, compute_otsu(m, hist));
    }

    // edge detection 
    inline void sobel_operator(mask& m)
    {
        m.operate([](pixels& op, pixel& np)
        {
            constexpr int filter[] = {1, 2, 1};

            int x = 0;
            for (int row = -1; row < 2; row++)
                for (int col = -1; col < 2; col++)
                    x += filter[row+1] * col * op.at(row, col).r;

            int y = 0;
            for (int row = -1; row < 2; row++)
                for (int col = -1; col < 2; col++)
                    y += filter[col+1] * row * op.at(row, col).r;
            
            np = sqrt(x * x + y * y);
        });
    }

    inline void laplacian(mask& m)
    {
        m.operate([](pixels& op, pixel& np)
        {
            constexpr int filter[3][3] = {{  0,  1,  0},
                                          {  1, -4,  1},
                                          {  0,  1,  0}};
            int r = 0, g = 0, b = 0;
            for (int row = -1; row < 2; row++)
                for (int col = -1; col < 2; col++)
                {
                    const pixel& p = op.at(row, col);
                    r += filter[row+1][col+1] * p.r;
                    g += filter[row+1][col+1] * p.g;
                    b += filter[row+1][col+1] * p.b;
                }
            
            np = pixel(b, g, r);
        });
    }

    // filtering
    inline void sharpen_filter(mask& m)
    {
        m.operate([](pixels& op, pixel& np)
        {
            constexpr int filter[3][3] = {{  0, -1,  0},
                                          { -1,  5, -1},
                                          {  0, -1,  0}};
            int r = 0, g = 0, b = 0;
            for (int row = -1; row < 2; row++)
                for (int col = -1; col < 2; col++)
                {
                    const pixel& p = op.at(row, col);
                    r += filter[row+1][col+1] * p.r;
                    g += filter[row+1][col+1] * p.g;
                    b += filter[row+1][col+1] * p.b;
                }
            
            np = pixel(b, g, r);
        });
    }

    inline void mean_filter(mask& m, int k)
    {
        void (*mean_func)(pixels&, pixel&, int) = [](pixels& op, pixel& np, int k)
        {
            int power = k * k;
            int size = (k-1)/2;

            int r = 0, g = 0, b = 0;

            for (int row = -1 * size; row < size + 1; row++)
                for (int col = -1 * size; col < size + 1; col++)
                {
                    const pixel& p = op.at(row, col);
                    r += p.r;
                    g += p.g;
                    b += p.b;
                }
            np = pixel(b/pow(k, 2), g/pow(k, 2), r/pow(k, 2));
        };

        m.operate(mean_func, k);
    }

    inline void median_filter(mask& m, int k)
    {
        void (*median_func)(pixels&, pixel&, int) = [](pixels& op, pixel& np, int k) {
            int power = k * k;
            int size = (k-1)/2;

            std::vector<int> r(power, 0);
            std::vector<int> g(power, 0);
            std::vector<int> b(power, 0);

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

            std::sort(r.begin(), r.end()); 
            std::sort(g.begin(), g.end()); 
            std::sort(b.begin(), b.end());

            np = pixel(b[size+1], g[size+1], r[size+1]); // why size + 1?

            //int rr = *std::max_element(r.begin(), r.end()); 
            //int rg = *std::max_element(g.begin(), g.end()); 
            //int rb = *std::max_element(b.begin(), b.end());

            //np = pixel(rb, rg, rr);
        };

        m.operate(median_func, k);
    }

    // morphological
    inline void erosion(mask& m)
    {
        m.operate([](pixels& op, pixel& np)
        {
            constexpr int filter[3][3] = {{  0,  1,  0},
                                          {  1,  1,  1},
                                          {  0,  1,  0}};
            int value = 0;
            for (int row = -1; row < 2; row++)
                for (int col = -1; col < 2; col++)
                {
                    // not implemented yet
                    const pixel& p = op.at(row, col);
                }
            
            np = value;
        });
    }

    inline void dilation(mask& m)
    {
        m.operate([](pixels& op, pixel& np)
        {
            constexpr int filter[3][3] = {{  0,  1,  0},
                                          {  1,  1,  1},
                                          {  0,  1,  0}};
            int value = 0;
            for (int row = -1; row < 2; row++)
                for (int col = -1; col < 2; col++)
                {
                    // not implemented yet
                    const pixel& p = op.at(row, col);
                }
            
            np = value;
        });
    }

    inline void opening(mask& m)
    {
        erosion(m);
        dilation(m);
    }

    inline void closing(mask& m)
    {
        dilation(m);
        erosion(m);
    }
}
