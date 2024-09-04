#include <immintrin.h>
#include <cstdint>
#include <x86intrin.h>
#include <cassert>
#ifndef DISTANCE_H
#define DISTANCE_H
enum Metric
{
    L2 = 0,
    INNER_PRODUCT = 1,
    COSINE = 2,
};

template <typename T> class Distance
{
  public:
    Distance(Metric dist_metric) : _distance_metric(dist_metric)
    {
    }

    virtual float compare(const T *pVec1, const T *pVec2, uint32_t dim) const = 0;

    ~Distance(){

    }

  protected:
    Metric _distance_metric;
    size_t _alignment_factor = 8;
};

#define ALIGNED(x) __attribute__((aligned(x)))
#define _MM_SHFFLE(fp3, fp2, fp1, fp0) (((fp3) << 6) | ((fp2) << 4) | ((fp1) << 2) | (fp0))

static float HsumFloat128(__m128 x) {
   __m128 h64 = _mm_shuffle_ps(x, x, _MM_SHFFLE(1, 0, 3, 2));
   __m128 sum64 = _mm_add_ps(h64, x);
   __m128 h32 = _mm_shuffle_ps(sum64, sum64, _MM_SHFFLE(0, 1, 2, 3));
   __m128 sum32 = _mm_add_ps(sum64, h32);
   return _mm_cvtss_f32(sum32);
}

static inline __m128 MaskedReadFloat(const std::size_t dim, const float* data) {
   //assert(0<=dim && dim < 4 );
   ALIGNED(16) float buf[4] = {0, 0, 0, 0};
   switch (dim) {
       case 3:
           buf[2] = data[2];
       case 2:
           buf[1] = data[1];
       case 1:
           buf[0] = data[0];
   }
   return _mm_load_ps(buf);
}


class AVXDistanceL2Float : public Distance<float>
{
 public:
   AVXDistanceL2Float() : Distance<float>(Metric::L2)
   {
   }
   float compare(const float *pVec1, const float *pVec2, uint32_t dim) const{
       __m512 mx512, my512, diff512;
       __m512 sum512 = _mm512_setzero_ps();

       while (dim >= 16) {
           mx512 = _mm512_loadu_ps(pVec1); pVec1 += 16;
           my512 = _mm512_loadu_ps(pVec2); pVec2 += 16;
           diff512 = _mm512_sub_ps(mx512, my512);
           sum512 = _mm512_fmadd_ps(diff512, diff512, sum512);
           dim -= 16;
       }
       __m256 sum256 = _mm256_add_ps(_mm512_castps512_ps256(sum512), _mm512_extractf32x8_ps(sum512, 1));

       if (dim >= 8) {
           __m256 mx256 = _mm256_loadu_ps(pVec1); pVec1 += 8;
           __m256 my256 = _mm256_loadu_ps(pVec2); pVec2 += 8;
           __m256 diff256 = _mm256_sub_ps(mx256, my256);
           sum256 = _mm256_fmadd_ps(diff256, diff256, sum256);
           dim -= 8;
       }
       __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
       __m128 mx128, my128, diff128;

       if (dim >= 4) {
           mx128 = _mm_loadu_ps(pVec1); pVec1 += 4;
           my128 = _mm_loadu_ps(pVec2); pVec2 += 4;
           diff128 = _mm_sub_ps(mx128, my128);
           sum128  = _mm_fmadd_ps(diff128, diff128, sum128);
           dim -= 4;
       }

       if (dim > 0) {
           mx128 = MaskedReadFloat(dim, pVec1);
           my128 = MaskedReadFloat(dim, pVec2);
           diff128 = _mm_sub_ps(mx128, my128);
           sum128  = _mm_fmadd_ps(diff128, diff128, sum128);
       }
       return HsumFloat128(sum128);
   }
};

class DistanceL2UInt8 : public Distance<uint8_t>
{
  public:
    DistanceL2UInt8() : Distance<uint8_t>(Metric::L2)
    {
    }
    float compare(const uint8_t *a, const uint8_t *b, uint32_t size) const{
        uint32_t result = 0;
        #pragma omp simd reduction(+ : result) aligned(a, b : 8)
        for (int32_t i = 0; i < (int32_t)size; i++)
        {
            result += ((int32_t)((int16_t)a[i] - (int16_t)b[i])) * ((int32_t)((int16_t)a[i] - (int16_t)b[i]));
        }
        return (float)result;
    }
};

class DistanceL2Float : public Distance<float>
{
public:
    DistanceL2Float() : Distance<float>(Metric::L2)
    {
    }
    float compare(const float *a, const float *b, uint32_t size) const{
        float result = 0;

        for (int32_t i = 0; i < (int32_t)size; i++)
        {
            result += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return result;
    }
};

#endif
