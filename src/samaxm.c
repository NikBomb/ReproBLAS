#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "config.h"
#include "Common/Common.h"
#include <immintrin.h>
#include <emmintrin.h>


#if defined( __AVX__ )
  float samaxm(int n, float* v, int incv, float* y, int incy){
    __m256 mask_ABS; AVX_ABS_MASKS(mask_ABS);
    float tmp_max[8] __attribute__((aligned(32)));
    int i;
    float max;

    __m256 v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7;
    __m256 y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7;
    __m256 m_0;
    m_0 = _mm256_setzero_ps();

    if(incv == 1 && incy == 1){

      for(i = 0; i + 64 <= n; i += 64, v += 64, y += 64){
        v_0 = _mm256_loadu_ps(v);
        v_1 = _mm256_loadu_ps(v + 8);
        v_2 = _mm256_loadu_ps(v + 16);
        v_3 = _mm256_loadu_ps(v + 24);
        v_4 = _mm256_loadu_ps(v + 32);
        v_5 = _mm256_loadu_ps(v + 40);
        v_6 = _mm256_loadu_ps(v + 48);
        v_7 = _mm256_loadu_ps(v + 56);
        y_0 = _mm256_loadu_ps(y);
        y_1 = _mm256_loadu_ps(y + 8);
        y_2 = _mm256_loadu_ps(y + 16);
        y_3 = _mm256_loadu_ps(y + 24);
        y_4 = _mm256_loadu_ps(y + 32);
        y_5 = _mm256_loadu_ps(y + 40);
        y_6 = _mm256_loadu_ps(y + 48);
        y_7 = _mm256_loadu_ps(y + 56);
        v_0 = _mm256_and_ps(_mm256_mul_ps(v_0, y_0), mask_ABS);
        v_1 = _mm256_and_ps(_mm256_mul_ps(v_1, y_1), mask_ABS);
        v_2 = _mm256_and_ps(_mm256_mul_ps(v_2, y_2), mask_ABS);
        v_3 = _mm256_and_ps(_mm256_mul_ps(v_3, y_3), mask_ABS);
        v_4 = _mm256_and_ps(_mm256_mul_ps(v_4, y_4), mask_ABS);
        v_5 = _mm256_and_ps(_mm256_mul_ps(v_5, y_5), mask_ABS);
        v_6 = _mm256_and_ps(_mm256_mul_ps(v_6, y_6), mask_ABS);
        v_7 = _mm256_and_ps(_mm256_mul_ps(v_7, y_7), mask_ABS);
        m_0 = _mm256_max_ps(m_0, v_0);
        m_0 = _mm256_max_ps(m_0, v_1);
        m_0 = _mm256_max_ps(m_0, v_2);
        m_0 = _mm256_max_ps(m_0, v_3);
        m_0 = _mm256_max_ps(m_0, v_4);
        m_0 = _mm256_max_ps(m_0, v_5);
        m_0 = _mm256_max_ps(m_0, v_6);
        m_0 = _mm256_max_ps(m_0, v_7);
      }
      if(i + 32 <= n){
        v_0 = _mm256_loadu_ps(v);
        v_1 = _mm256_loadu_ps(v + 8);
        v_2 = _mm256_loadu_ps(v + 16);
        v_3 = _mm256_loadu_ps(v + 24);
        y_0 = _mm256_loadu_ps(y);
        y_1 = _mm256_loadu_ps(y + 8);
        y_2 = _mm256_loadu_ps(y + 16);
        y_3 = _mm256_loadu_ps(y + 24);
        v_0 = _mm256_and_ps(_mm256_mul_ps(v_0, y_0), mask_ABS);
        v_1 = _mm256_and_ps(_mm256_mul_ps(v_1, y_1), mask_ABS);
        v_2 = _mm256_and_ps(_mm256_mul_ps(v_2, y_2), mask_ABS);
        v_3 = _mm256_and_ps(_mm256_mul_ps(v_3, y_3), mask_ABS);
        m_0 = _mm256_max_ps(m_0, v_0);
        m_0 = _mm256_max_ps(m_0, v_1);
        m_0 = _mm256_max_ps(m_0, v_2);
        m_0 = _mm256_max_ps(m_0, v_3);
        i += 32, v += 32, y += 32;
      }
      if(i + 16 <= n){
        v_0 = _mm256_loadu_ps(v);
        v_1 = _mm256_loadu_ps(v + 8);
        y_0 = _mm256_loadu_ps(y);
        y_1 = _mm256_loadu_ps(y + 8);
        v_0 = _mm256_and_ps(_mm256_mul_ps(v_0, y_0), mask_ABS);
        v_1 = _mm256_and_ps(_mm256_mul_ps(v_1, y_1), mask_ABS);
        m_0 = _mm256_max_ps(m_0, v_0);
        m_0 = _mm256_max_ps(m_0, v_1);
        i += 16, v += 16, y += 16;
      }
      if(i + 8 <= n){
        v_0 = _mm256_loadu_ps(v);
        y_0 = _mm256_loadu_ps(y);
        v_0 = _mm256_and_ps(_mm256_mul_ps(v_0, y_0), mask_ABS);
        m_0 = _mm256_max_ps(m_0, v_0);
        i += 8, v += 8, y += 8;
      }
      if(i < n){
        v_0 = _mm256_set_ps(0, (n - i)>6?v[6]:0, (n - i)>5?v[5]:0, (n - i)>4?v[4]:0, (n - i)>3?v[3]:0, (n - i)>2?v[2]:0, (n - i)>1?v[1]:0, v[0]);
        y_0 = _mm256_set_ps(0, (n - i)>6?y[6]:0, (n - i)>5?y[5]:0, (n - i)>4?y[4]:0, (n - i)>3?y[3]:0, (n - i)>2?y[2]:0, (n - i)>1?y[1]:0, y[0]);
        v_0 = _mm256_and_ps(_mm256_mul_ps(v_0, y_0), mask_ABS);
        m_0 = _mm256_max_ps(m_0, v_0);
      }
    }else{

      for(i = 0; i + 64 <= n; i += 64, v += (incv * 64), y += (incy * 64)){
        v_0 = _mm256_set_ps(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)], v[(incv * 3)], v[(incv * 2)], v[incv], v[0]);
        v_1 = _mm256_set_ps(v[(incv * 15)], v[(incv * 14)], v[(incv * 13)], v[(incv * 12)], v[(incv * 11)], v[(incv * 10)], v[(incv * 9)], v[(incv * 8)]);
        v_2 = _mm256_set_ps(v[(incv * 23)], v[(incv * 22)], v[(incv * 21)], v[(incv * 20)], v[(incv * 19)], v[(incv * 18)], v[(incv * 17)], v[(incv * 16)]);
        v_3 = _mm256_set_ps(v[(incv * 31)], v[(incv * 30)], v[(incv * 29)], v[(incv * 28)], v[(incv * 27)], v[(incv * 26)], v[(incv * 25)], v[(incv * 24)]);
        v_4 = _mm256_set_ps(v[(incv * 39)], v[(incv * 38)], v[(incv * 37)], v[(incv * 36)], v[(incv * 35)], v[(incv * 34)], v[(incv * 33)], v[(incv * 32)]);
        v_5 = _mm256_set_ps(v[(incv * 47)], v[(incv * 46)], v[(incv * 45)], v[(incv * 44)], v[(incv * 43)], v[(incv * 42)], v[(incv * 41)], v[(incv * 40)]);
        v_6 = _mm256_set_ps(v[(incv * 55)], v[(incv * 54)], v[(incv * 53)], v[(incv * 52)], v[(incv * 51)], v[(incv * 50)], v[(incv * 49)], v[(incv * 48)]);
        v_7 = _mm256_set_ps(v[(incv * 63)], v[(incv * 62)], v[(incv * 61)], v[(incv * 60)], v[(incv * 59)], v[(incv * 58)], v[(incv * 57)], v[(incv * 56)]);
        y_0 = _mm256_set_ps(y[(incy * 7)], y[(incy * 6)], y[(incy * 5)], y[(incy * 4)], y[(incy * 3)], y[(incy * 2)], y[incy], y[0]);
        y_1 = _mm256_set_ps(y[(incy * 15)], y[(incy * 14)], y[(incy * 13)], y[(incy * 12)], y[(incy * 11)], y[(incy * 10)], y[(incy * 9)], y[(incy * 8)]);
        y_2 = _mm256_set_ps(y[(incy * 23)], y[(incy * 22)], y[(incy * 21)], y[(incy * 20)], y[(incy * 19)], y[(incy * 18)], y[(incy * 17)], y[(incy * 16)]);
        y_3 = _mm256_set_ps(y[(incy * 31)], y[(incy * 30)], y[(incy * 29)], y[(incy * 28)], y[(incy * 27)], y[(incy * 26)], y[(incy * 25)], y[(incy * 24)]);
        y_4 = _mm256_set_ps(y[(incy * 39)], y[(incy * 38)], y[(incy * 37)], y[(incy * 36)], y[(incy * 35)], y[(incy * 34)], y[(incy * 33)], y[(incy * 32)]);
        y_5 = _mm256_set_ps(y[(incy * 47)], y[(incy * 46)], y[(incy * 45)], y[(incy * 44)], y[(incy * 43)], y[(incy * 42)], y[(incy * 41)], y[(incy * 40)]);
        y_6 = _mm256_set_ps(y[(incy * 55)], y[(incy * 54)], y[(incy * 53)], y[(incy * 52)], y[(incy * 51)], y[(incy * 50)], y[(incy * 49)], y[(incy * 48)]);
        y_7 = _mm256_set_ps(y[(incy * 63)], y[(incy * 62)], y[(incy * 61)], y[(incy * 60)], y[(incy * 59)], y[(incy * 58)], y[(incy * 57)], y[(incy * 56)]);
        v_0 = _mm256_and_ps(_mm256_mul_ps(v_0, y_0), mask_ABS);
        v_1 = _mm256_and_ps(_mm256_mul_ps(v_1, y_1), mask_ABS);
        v_2 = _mm256_and_ps(_mm256_mul_ps(v_2, y_2), mask_ABS);
        v_3 = _mm256_and_ps(_mm256_mul_ps(v_3, y_3), mask_ABS);
        v_4 = _mm256_and_ps(_mm256_mul_ps(v_4, y_4), mask_ABS);
        v_5 = _mm256_and_ps(_mm256_mul_ps(v_5, y_5), mask_ABS);
        v_6 = _mm256_and_ps(_mm256_mul_ps(v_6, y_6), mask_ABS);
        v_7 = _mm256_and_ps(_mm256_mul_ps(v_7, y_7), mask_ABS);
        m_0 = _mm256_max_ps(m_0, v_0);
        m_0 = _mm256_max_ps(m_0, v_1);
        m_0 = _mm256_max_ps(m_0, v_2);
        m_0 = _mm256_max_ps(m_0, v_3);
        m_0 = _mm256_max_ps(m_0, v_4);
        m_0 = _mm256_max_ps(m_0, v_5);
        m_0 = _mm256_max_ps(m_0, v_6);
        m_0 = _mm256_max_ps(m_0, v_7);
      }
      if(i + 32 <= n){
        v_0 = _mm256_set_ps(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)], v[(incv * 3)], v[(incv * 2)], v[incv], v[0]);
        v_1 = _mm256_set_ps(v[(incv * 15)], v[(incv * 14)], v[(incv * 13)], v[(incv * 12)], v[(incv * 11)], v[(incv * 10)], v[(incv * 9)], v[(incv * 8)]);
        v_2 = _mm256_set_ps(v[(incv * 23)], v[(incv * 22)], v[(incv * 21)], v[(incv * 20)], v[(incv * 19)], v[(incv * 18)], v[(incv * 17)], v[(incv * 16)]);
        v_3 = _mm256_set_ps(v[(incv * 31)], v[(incv * 30)], v[(incv * 29)], v[(incv * 28)], v[(incv * 27)], v[(incv * 26)], v[(incv * 25)], v[(incv * 24)]);
        y_0 = _mm256_set_ps(y[(incy * 7)], y[(incy * 6)], y[(incy * 5)], y[(incy * 4)], y[(incy * 3)], y[(incy * 2)], y[incy], y[0]);
        y_1 = _mm256_set_ps(y[(incy * 15)], y[(incy * 14)], y[(incy * 13)], y[(incy * 12)], y[(incy * 11)], y[(incy * 10)], y[(incy * 9)], y[(incy * 8)]);
        y_2 = _mm256_set_ps(y[(incy * 23)], y[(incy * 22)], y[(incy * 21)], y[(incy * 20)], y[(incy * 19)], y[(incy * 18)], y[(incy * 17)], y[(incy * 16)]);
        y_3 = _mm256_set_ps(y[(incy * 31)], y[(incy * 30)], y[(incy * 29)], y[(incy * 28)], y[(incy * 27)], y[(incy * 26)], y[(incy * 25)], y[(incy * 24)]);
        v_0 = _mm256_and_ps(_mm256_mul_ps(v_0, y_0), mask_ABS);
        v_1 = _mm256_and_ps(_mm256_mul_ps(v_1, y_1), mask_ABS);
        v_2 = _mm256_and_ps(_mm256_mul_ps(v_2, y_2), mask_ABS);
        v_3 = _mm256_and_ps(_mm256_mul_ps(v_3, y_3), mask_ABS);
        m_0 = _mm256_max_ps(m_0, v_0);
        m_0 = _mm256_max_ps(m_0, v_1);
        m_0 = _mm256_max_ps(m_0, v_2);
        m_0 = _mm256_max_ps(m_0, v_3);
        i += 32, v += (incv * 32), y += (incy * 32);
      }
      if(i + 16 <= n){
        v_0 = _mm256_set_ps(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)], v[(incv * 3)], v[(incv * 2)], v[incv], v[0]);
        v_1 = _mm256_set_ps(v[(incv * 15)], v[(incv * 14)], v[(incv * 13)], v[(incv * 12)], v[(incv * 11)], v[(incv * 10)], v[(incv * 9)], v[(incv * 8)]);
        y_0 = _mm256_set_ps(y[(incy * 7)], y[(incy * 6)], y[(incy * 5)], y[(incy * 4)], y[(incy * 3)], y[(incy * 2)], y[incy], y[0]);
        y_1 = _mm256_set_ps(y[(incy * 15)], y[(incy * 14)], y[(incy * 13)], y[(incy * 12)], y[(incy * 11)], y[(incy * 10)], y[(incy * 9)], y[(incy * 8)]);
        v_0 = _mm256_and_ps(_mm256_mul_ps(v_0, y_0), mask_ABS);
        v_1 = _mm256_and_ps(_mm256_mul_ps(v_1, y_1), mask_ABS);
        m_0 = _mm256_max_ps(m_0, v_0);
        m_0 = _mm256_max_ps(m_0, v_1);
        i += 16, v += (incv * 16), y += (incy * 16);
      }
      if(i + 8 <= n){
        v_0 = _mm256_set_ps(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)], v[(incv * 3)], v[(incv * 2)], v[incv], v[0]);
        y_0 = _mm256_set_ps(y[(incy * 7)], y[(incy * 6)], y[(incy * 5)], y[(incy * 4)], y[(incy * 3)], y[(incy * 2)], y[incy], y[0]);
        v_0 = _mm256_and_ps(_mm256_mul_ps(v_0, y_0), mask_ABS);
        m_0 = _mm256_max_ps(m_0, v_0);
        i += 8, v += (incv * 8), y += (incy * 8);
      }
      if(i < n){
        v_0 = _mm256_set_ps(0, (n - i)>6?v[(incv * 6)]:0, (n - i)>5?v[(incv * 5)]:0, (n - i)>4?v[(incv * 4)]:0, (n - i)>3?v[(incv * 3)]:0, (n - i)>2?v[(incv * 2)]:0, (n - i)>1?v[incv]:0, v[0]);
        y_0 = _mm256_set_ps(0, (n - i)>6?y[(incy * 6)]:0, (n - i)>5?y[(incy * 5)]:0, (n - i)>4?y[(incy * 4)]:0, (n - i)>3?y[(incy * 3)]:0, (n - i)>2?y[(incy * 2)]:0, (n - i)>1?y[incy]:0, y[0]);
        v_0 = _mm256_and_ps(_mm256_mul_ps(v_0, y_0), mask_ABS);
        m_0 = _mm256_max_ps(m_0, v_0);
      }
    }
    _mm256_store_ps(tmp_max, m_0);
    tmp_max[0] = (tmp_max[0] > tmp_max[1] ? tmp_max[0]: tmp_max[1]);
    tmp_max[0] = (tmp_max[0] > tmp_max[2] ? tmp_max[0]: tmp_max[2]);
    tmp_max[0] = (tmp_max[0] > tmp_max[3] ? tmp_max[0]: tmp_max[3]);
    tmp_max[0] = (tmp_max[0] > tmp_max[4] ? tmp_max[0]: tmp_max[4]);
    tmp_max[0] = (tmp_max[0] > tmp_max[5] ? tmp_max[0]: tmp_max[5]);
    tmp_max[0] = (tmp_max[0] > tmp_max[6] ? tmp_max[0]: tmp_max[6]);
    tmp_max[0] = (tmp_max[0] > tmp_max[7] ? tmp_max[0]: tmp_max[7]);
    (&max)[0] = ((float*)tmp_max)[0];
    return max;
  }
#elif defined( __SSE2__ )
  float samaxm(int n, float* v, int incv, float* y, int incy){
    __m128 mask_ABS; SSE_ABS_MASKS(mask_ABS);
    float tmp_max[4] __attribute__((aligned(16)));
    int i;
    float max;

    __m128 v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7;
    __m128 y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7;
    __m128 m_0;
    m_0 = _mm_setzero_ps();

    if(incv == 1 && incy == 1){

      for(i = 0; i + 32 <= n; i += 32, v += 32, y += 32){
        v_0 = _mm_loadu_ps(v);
        v_1 = _mm_loadu_ps(v + 4);
        v_2 = _mm_loadu_ps(v + 8);
        v_3 = _mm_loadu_ps(v + 12);
        v_4 = _mm_loadu_ps(v + 16);
        v_5 = _mm_loadu_ps(v + 20);
        v_6 = _mm_loadu_ps(v + 24);
        v_7 = _mm_loadu_ps(v + 28);
        y_0 = _mm_loadu_ps(y);
        y_1 = _mm_loadu_ps(y + 4);
        y_2 = _mm_loadu_ps(y + 8);
        y_3 = _mm_loadu_ps(y + 12);
        y_4 = _mm_loadu_ps(y + 16);
        y_5 = _mm_loadu_ps(y + 20);
        y_6 = _mm_loadu_ps(y + 24);
        y_7 = _mm_loadu_ps(y + 28);
        v_0 = _mm_and_ps(_mm_mul_ps(v_0, y_0), mask_ABS);
        v_1 = _mm_and_ps(_mm_mul_ps(v_1, y_1), mask_ABS);
        v_2 = _mm_and_ps(_mm_mul_ps(v_2, y_2), mask_ABS);
        v_3 = _mm_and_ps(_mm_mul_ps(v_3, y_3), mask_ABS);
        v_4 = _mm_and_ps(_mm_mul_ps(v_4, y_4), mask_ABS);
        v_5 = _mm_and_ps(_mm_mul_ps(v_5, y_5), mask_ABS);
        v_6 = _mm_and_ps(_mm_mul_ps(v_6, y_6), mask_ABS);
        v_7 = _mm_and_ps(_mm_mul_ps(v_7, y_7), mask_ABS);
        m_0 = _mm_max_ps(m_0, v_0);
        m_0 = _mm_max_ps(m_0, v_1);
        m_0 = _mm_max_ps(m_0, v_2);
        m_0 = _mm_max_ps(m_0, v_3);
        m_0 = _mm_max_ps(m_0, v_4);
        m_0 = _mm_max_ps(m_0, v_5);
        m_0 = _mm_max_ps(m_0, v_6);
        m_0 = _mm_max_ps(m_0, v_7);
      }
      if(i + 16 <= n){
        v_0 = _mm_loadu_ps(v);
        v_1 = _mm_loadu_ps(v + 4);
        v_2 = _mm_loadu_ps(v + 8);
        v_3 = _mm_loadu_ps(v + 12);
        y_0 = _mm_loadu_ps(y);
        y_1 = _mm_loadu_ps(y + 4);
        y_2 = _mm_loadu_ps(y + 8);
        y_3 = _mm_loadu_ps(y + 12);
        v_0 = _mm_and_ps(_mm_mul_ps(v_0, y_0), mask_ABS);
        v_1 = _mm_and_ps(_mm_mul_ps(v_1, y_1), mask_ABS);
        v_2 = _mm_and_ps(_mm_mul_ps(v_2, y_2), mask_ABS);
        v_3 = _mm_and_ps(_mm_mul_ps(v_3, y_3), mask_ABS);
        m_0 = _mm_max_ps(m_0, v_0);
        m_0 = _mm_max_ps(m_0, v_1);
        m_0 = _mm_max_ps(m_0, v_2);
        m_0 = _mm_max_ps(m_0, v_3);
        i += 16, v += 16, y += 16;
      }
      if(i + 8 <= n){
        v_0 = _mm_loadu_ps(v);
        v_1 = _mm_loadu_ps(v + 4);
        y_0 = _mm_loadu_ps(y);
        y_1 = _mm_loadu_ps(y + 4);
        v_0 = _mm_and_ps(_mm_mul_ps(v_0, y_0), mask_ABS);
        v_1 = _mm_and_ps(_mm_mul_ps(v_1, y_1), mask_ABS);
        m_0 = _mm_max_ps(m_0, v_0);
        m_0 = _mm_max_ps(m_0, v_1);
        i += 8, v += 8, y += 8;
      }
      if(i + 4 <= n){
        v_0 = _mm_loadu_ps(v);
        y_0 = _mm_loadu_ps(y);
        v_0 = _mm_and_ps(_mm_mul_ps(v_0, y_0), mask_ABS);
        m_0 = _mm_max_ps(m_0, v_0);
        i += 4, v += 4, y += 4;
      }
      if(i < n){
        v_0 = _mm_set_ps(0, (n - i)>2?v[2]:0, (n - i)>1?v[1]:0, v[0]);
        y_0 = _mm_set_ps(0, (n - i)>2?y[2]:0, (n - i)>1?y[1]:0, y[0]);
        v_0 = _mm_and_ps(_mm_mul_ps(v_0, y_0), mask_ABS);
        m_0 = _mm_max_ps(m_0, v_0);
      }
    }else{

      for(i = 0; i + 32 <= n; i += 32, v += (incv * 32), y += (incy * 32)){
        v_0 = _mm_set_ps(v[(incv * 3)], v[(incv * 2)], v[incv], v[0]);
        v_1 = _mm_set_ps(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)]);
        v_2 = _mm_set_ps(v[(incv * 11)], v[(incv * 10)], v[(incv * 9)], v[(incv * 8)]);
        v_3 = _mm_set_ps(v[(incv * 15)], v[(incv * 14)], v[(incv * 13)], v[(incv * 12)]);
        v_4 = _mm_set_ps(v[(incv * 19)], v[(incv * 18)], v[(incv * 17)], v[(incv * 16)]);
        v_5 = _mm_set_ps(v[(incv * 23)], v[(incv * 22)], v[(incv * 21)], v[(incv * 20)]);
        v_6 = _mm_set_ps(v[(incv * 27)], v[(incv * 26)], v[(incv * 25)], v[(incv * 24)]);
        v_7 = _mm_set_ps(v[(incv * 31)], v[(incv * 30)], v[(incv * 29)], v[(incv * 28)]);
        y_0 = _mm_set_ps(y[(incy * 3)], y[(incy * 2)], y[incy], y[0]);
        y_1 = _mm_set_ps(y[(incy * 7)], y[(incy * 6)], y[(incy * 5)], y[(incy * 4)]);
        y_2 = _mm_set_ps(y[(incy * 11)], y[(incy * 10)], y[(incy * 9)], y[(incy * 8)]);
        y_3 = _mm_set_ps(y[(incy * 15)], y[(incy * 14)], y[(incy * 13)], y[(incy * 12)]);
        y_4 = _mm_set_ps(y[(incy * 19)], y[(incy * 18)], y[(incy * 17)], y[(incy * 16)]);
        y_5 = _mm_set_ps(y[(incy * 23)], y[(incy * 22)], y[(incy * 21)], y[(incy * 20)]);
        y_6 = _mm_set_ps(y[(incy * 27)], y[(incy * 26)], y[(incy * 25)], y[(incy * 24)]);
        y_7 = _mm_set_ps(y[(incy * 31)], y[(incy * 30)], y[(incy * 29)], y[(incy * 28)]);
        v_0 = _mm_and_ps(_mm_mul_ps(v_0, y_0), mask_ABS);
        v_1 = _mm_and_ps(_mm_mul_ps(v_1, y_1), mask_ABS);
        v_2 = _mm_and_ps(_mm_mul_ps(v_2, y_2), mask_ABS);
        v_3 = _mm_and_ps(_mm_mul_ps(v_3, y_3), mask_ABS);
        v_4 = _mm_and_ps(_mm_mul_ps(v_4, y_4), mask_ABS);
        v_5 = _mm_and_ps(_mm_mul_ps(v_5, y_5), mask_ABS);
        v_6 = _mm_and_ps(_mm_mul_ps(v_6, y_6), mask_ABS);
        v_7 = _mm_and_ps(_mm_mul_ps(v_7, y_7), mask_ABS);
        m_0 = _mm_max_ps(m_0, v_0);
        m_0 = _mm_max_ps(m_0, v_1);
        m_0 = _mm_max_ps(m_0, v_2);
        m_0 = _mm_max_ps(m_0, v_3);
        m_0 = _mm_max_ps(m_0, v_4);
        m_0 = _mm_max_ps(m_0, v_5);
        m_0 = _mm_max_ps(m_0, v_6);
        m_0 = _mm_max_ps(m_0, v_7);
      }
      if(i + 16 <= n){
        v_0 = _mm_set_ps(v[(incv * 3)], v[(incv * 2)], v[incv], v[0]);
        v_1 = _mm_set_ps(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)]);
        v_2 = _mm_set_ps(v[(incv * 11)], v[(incv * 10)], v[(incv * 9)], v[(incv * 8)]);
        v_3 = _mm_set_ps(v[(incv * 15)], v[(incv * 14)], v[(incv * 13)], v[(incv * 12)]);
        y_0 = _mm_set_ps(y[(incy * 3)], y[(incy * 2)], y[incy], y[0]);
        y_1 = _mm_set_ps(y[(incy * 7)], y[(incy * 6)], y[(incy * 5)], y[(incy * 4)]);
        y_2 = _mm_set_ps(y[(incy * 11)], y[(incy * 10)], y[(incy * 9)], y[(incy * 8)]);
        y_3 = _mm_set_ps(y[(incy * 15)], y[(incy * 14)], y[(incy * 13)], y[(incy * 12)]);
        v_0 = _mm_and_ps(_mm_mul_ps(v_0, y_0), mask_ABS);
        v_1 = _mm_and_ps(_mm_mul_ps(v_1, y_1), mask_ABS);
        v_2 = _mm_and_ps(_mm_mul_ps(v_2, y_2), mask_ABS);
        v_3 = _mm_and_ps(_mm_mul_ps(v_3, y_3), mask_ABS);
        m_0 = _mm_max_ps(m_0, v_0);
        m_0 = _mm_max_ps(m_0, v_1);
        m_0 = _mm_max_ps(m_0, v_2);
        m_0 = _mm_max_ps(m_0, v_3);
        i += 16, v += (incv * 16), y += (incy * 16);
      }
      if(i + 8 <= n){
        v_0 = _mm_set_ps(v[(incv * 3)], v[(incv * 2)], v[incv], v[0]);
        v_1 = _mm_set_ps(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)]);
        y_0 = _mm_set_ps(y[(incy * 3)], y[(incy * 2)], y[incy], y[0]);
        y_1 = _mm_set_ps(y[(incy * 7)], y[(incy * 6)], y[(incy * 5)], y[(incy * 4)]);
        v_0 = _mm_and_ps(_mm_mul_ps(v_0, y_0), mask_ABS);
        v_1 = _mm_and_ps(_mm_mul_ps(v_1, y_1), mask_ABS);
        m_0 = _mm_max_ps(m_0, v_0);
        m_0 = _mm_max_ps(m_0, v_1);
        i += 8, v += (incv * 8), y += (incy * 8);
      }
      if(i + 4 <= n){
        v_0 = _mm_set_ps(v[(incv * 3)], v[(incv * 2)], v[incv], v[0]);
        y_0 = _mm_set_ps(y[(incy * 3)], y[(incy * 2)], y[incy], y[0]);
        v_0 = _mm_and_ps(_mm_mul_ps(v_0, y_0), mask_ABS);
        m_0 = _mm_max_ps(m_0, v_0);
        i += 4, v += (incv * 4), y += (incy * 4);
      }
      if(i < n){
        v_0 = _mm_set_ps(0, (n - i)>2?v[(incv * 2)]:0, (n - i)>1?v[incv]:0, v[0]);
        y_0 = _mm_set_ps(0, (n - i)>2?y[(incy * 2)]:0, (n - i)>1?y[incy]:0, y[0]);
        v_0 = _mm_and_ps(_mm_mul_ps(v_0, y_0), mask_ABS);
        m_0 = _mm_max_ps(m_0, v_0);
      }
    }
    _mm_store_ps(tmp_max, m_0);
    tmp_max[0] = (tmp_max[0] > tmp_max[1] ? tmp_max[0]: tmp_max[1]);
    tmp_max[0] = (tmp_max[0] > tmp_max[2] ? tmp_max[0]: tmp_max[2]);
    tmp_max[0] = (tmp_max[0] > tmp_max[3] ? tmp_max[0]: tmp_max[3]);
    (&max)[0] = ((float*)tmp_max)[0];
    return max;
  }
#else
  float samaxm(int n, float* v, int incv, float* y, int incy){
    int i;
    float max;

    float v_0, v_1;
    float y_0, y_1;
    float m_0;
    m_0 = 0;

    if(incv == 1 && incy == 1){

      for(i = 0; i + 2 <= n; i += 2, v += 2, y += 2){
        v_0 = v[0];
        v_1 = v[1];
        y_0 = y[0];
        y_1 = y[1];
        v_0 = fabs(v_0 * y_0);
        v_1 = fabs(v_1 * y_1);
        m_0 = (m_0 > v_0? m_0: v_0);
        m_0 = (m_0 > v_1? m_0: v_1);
      }
      if(i + 1 <= n){
        v_0 = v[0];
        y_0 = y[0];
        v_0 = fabs(v_0 * y_0);
        m_0 = (m_0 > v_0? m_0: v_0);
        i += 1, v += 1, y += 1;
      }
    }else{

      for(i = 0; i + 2 <= n; i += 2, v += (incv * 2), y += (incy * 2)){
        v_0 = v[0];
        v_1 = v[incv];
        y_0 = y[0];
        y_1 = y[incy];
        v_0 = fabs(v_0 * y_0);
        v_1 = fabs(v_1 * y_1);
        m_0 = (m_0 > v_0? m_0: v_0);
        m_0 = (m_0 > v_1? m_0: v_1);
      }
      if(i + 1 <= n){
        v_0 = v[0];
        y_0 = y[0];
        v_0 = fabs(v_0 * y_0);
        m_0 = (m_0 > v_0? m_0: v_0);
        i += 1, v += incv, y += incy;
      }
    }
    (&max)[0] = m_0;
    return max;
  }
#endif