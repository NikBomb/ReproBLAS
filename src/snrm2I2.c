#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "config.h"
#include "Common/Common.h"
#include <immintrin.h>
#include <emmintrin.h>


#if defined( __AVX__ )
  void snrm2I2(int n, float* v, int incv, float scale, int fold, float* sum){
    __m256 scale_mask = _mm256_set1_ps(scale);
    __m256 mask_BLP; AVX_BLP_MASKS(mask_BLP);
    float tmp_cons[8] __attribute__((aligned(32)));
    SET_DAZ_FLAG;
    switch(fold){
      case 3:{
        int i;

        __m256 v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7;
        __m256 q_0, q_1;
        __m256 s_0_0, s_0_1;
        __m256 s_1_0, s_1_1;
        __m256 s_2_0, s_2_1;

        s_0_0 = s_0_1 = _mm256_broadcast_ss(sum);
        s_1_0 = s_1_1 = _mm256_broadcast_ss(sum + 1);
        s_2_0 = s_2_1 = _mm256_broadcast_ss(sum + 2);
        if(incv == 1){

          for(i = 0; i + 64 <= n; i += 64, v += 64){
            v_0 = _mm256_mul_ps(_mm256_loadu_ps(v), scale_mask);
            v_1 = _mm256_mul_ps(_mm256_loadu_ps(v + 8), scale_mask);
            v_2 = _mm256_mul_ps(_mm256_loadu_ps(v + 16), scale_mask);
            v_3 = _mm256_mul_ps(_mm256_loadu_ps(v + 24), scale_mask);
            v_4 = _mm256_mul_ps(_mm256_loadu_ps(v + 32), scale_mask);
            v_5 = _mm256_mul_ps(_mm256_loadu_ps(v + 40), scale_mask);
            v_6 = _mm256_mul_ps(_mm256_loadu_ps(v + 48), scale_mask);
            v_7 = _mm256_mul_ps(_mm256_loadu_ps(v + 56), scale_mask);
            v_0 = _mm256_mul_ps(v_0, v_0);
            v_1 = _mm256_mul_ps(v_1, v_1);
            v_2 = _mm256_mul_ps(v_2, v_2);
            v_3 = _mm256_mul_ps(v_3, v_3);
            v_4 = _mm256_mul_ps(v_4, v_4);
            v_5 = _mm256_mul_ps(v_5, v_5);
            v_6 = _mm256_mul_ps(v_6, v_6);
            v_7 = _mm256_mul_ps(v_7, v_7);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_0, mask_BLP));
            s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(v_1, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            q_1 = _mm256_sub_ps(q_1, s_0_1);
            v_0 = _mm256_add_ps(v_0, q_0);
            v_1 = _mm256_add_ps(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_0, mask_BLP));
            s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(v_1, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            q_1 = _mm256_sub_ps(q_1, s_1_1);
            v_0 = _mm256_add_ps(v_0, q_0);
            v_1 = _mm256_add_ps(v_1, q_1);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_0, mask_BLP));
            s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(v_1, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_2, mask_BLP));
            s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(v_3, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            q_1 = _mm256_sub_ps(q_1, s_0_1);
            v_2 = _mm256_add_ps(v_2, q_0);
            v_3 = _mm256_add_ps(v_3, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_2, mask_BLP));
            s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(v_3, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            q_1 = _mm256_sub_ps(q_1, s_1_1);
            v_2 = _mm256_add_ps(v_2, q_0);
            v_3 = _mm256_add_ps(v_3, q_1);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_2, mask_BLP));
            s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(v_3, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_4, mask_BLP));
            s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(v_5, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            q_1 = _mm256_sub_ps(q_1, s_0_1);
            v_4 = _mm256_add_ps(v_4, q_0);
            v_5 = _mm256_add_ps(v_5, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_4, mask_BLP));
            s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(v_5, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            q_1 = _mm256_sub_ps(q_1, s_1_1);
            v_4 = _mm256_add_ps(v_4, q_0);
            v_5 = _mm256_add_ps(v_5, q_1);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_4, mask_BLP));
            s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(v_5, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_6, mask_BLP));
            s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(v_7, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            q_1 = _mm256_sub_ps(q_1, s_0_1);
            v_6 = _mm256_add_ps(v_6, q_0);
            v_7 = _mm256_add_ps(v_7, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_6, mask_BLP));
            s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(v_7, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            q_1 = _mm256_sub_ps(q_1, s_1_1);
            v_6 = _mm256_add_ps(v_6, q_0);
            v_7 = _mm256_add_ps(v_7, q_1);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_6, mask_BLP));
            s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(v_7, mask_BLP));
          }
          if(i + 32 <= n){
            v_0 = _mm256_mul_ps(_mm256_loadu_ps(v), scale_mask);
            v_1 = _mm256_mul_ps(_mm256_loadu_ps(v + 8), scale_mask);
            v_2 = _mm256_mul_ps(_mm256_loadu_ps(v + 16), scale_mask);
            v_3 = _mm256_mul_ps(_mm256_loadu_ps(v + 24), scale_mask);
            v_0 = _mm256_mul_ps(v_0, v_0);
            v_1 = _mm256_mul_ps(v_1, v_1);
            v_2 = _mm256_mul_ps(v_2, v_2);
            v_3 = _mm256_mul_ps(v_3, v_3);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_0, mask_BLP));
            s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(v_1, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            q_1 = _mm256_sub_ps(q_1, s_0_1);
            v_0 = _mm256_add_ps(v_0, q_0);
            v_1 = _mm256_add_ps(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_0, mask_BLP));
            s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(v_1, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            q_1 = _mm256_sub_ps(q_1, s_1_1);
            v_0 = _mm256_add_ps(v_0, q_0);
            v_1 = _mm256_add_ps(v_1, q_1);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_0, mask_BLP));
            s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(v_1, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_2, mask_BLP));
            s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(v_3, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            q_1 = _mm256_sub_ps(q_1, s_0_1);
            v_2 = _mm256_add_ps(v_2, q_0);
            v_3 = _mm256_add_ps(v_3, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_2, mask_BLP));
            s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(v_3, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            q_1 = _mm256_sub_ps(q_1, s_1_1);
            v_2 = _mm256_add_ps(v_2, q_0);
            v_3 = _mm256_add_ps(v_3, q_1);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_2, mask_BLP));
            s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(v_3, mask_BLP));
            i += 32, v += 32;
          }
          if(i + 16 <= n){
            v_0 = _mm256_mul_ps(_mm256_loadu_ps(v), scale_mask);
            v_1 = _mm256_mul_ps(_mm256_loadu_ps(v + 8), scale_mask);
            v_0 = _mm256_mul_ps(v_0, v_0);
            v_1 = _mm256_mul_ps(v_1, v_1);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_0, mask_BLP));
            s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(v_1, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            q_1 = _mm256_sub_ps(q_1, s_0_1);
            v_0 = _mm256_add_ps(v_0, q_0);
            v_1 = _mm256_add_ps(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_0, mask_BLP));
            s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(v_1, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            q_1 = _mm256_sub_ps(q_1, s_1_1);
            v_0 = _mm256_add_ps(v_0, q_0);
            v_1 = _mm256_add_ps(v_1, q_1);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_0, mask_BLP));
            s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(v_1, mask_BLP));
            i += 16, v += 16;
          }
          if(i + 8 <= n){
            v_0 = _mm256_mul_ps(_mm256_loadu_ps(v), scale_mask);
            v_0 = _mm256_mul_ps(v_0, v_0);
            q_0 = s_0_0;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_0, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            v_0 = _mm256_add_ps(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_0, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            v_0 = _mm256_add_ps(v_0, q_0);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_0, mask_BLP));
            i += 8, v += 8;
          }
          if(i < n){
            v_0 = _mm256_mul_ps(_mm256_set_ps(0, (n - i)>6?v[6]:0, (n - i)>5?v[5]:0, (n - i)>4?v[4]:0, (n - i)>3?v[3]:0, (n - i)>2?v[2]:0, (n - i)>1?v[1]:0, v[0]), scale_mask);
            v_0 = _mm256_mul_ps(v_0, v_0);
            q_0 = s_0_0;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_0, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            v_0 = _mm256_add_ps(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_0, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            v_0 = _mm256_add_ps(v_0, q_0);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_0, mask_BLP));
          }
        }else{

          for(i = 0; i + 64 <= n; i += 64, v += (incv * 64)){
            v_0 = _mm256_mul_ps(_mm256_set_ps(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)], v[(incv * 3)], v[(incv * 2)], v[incv], v[0]), scale_mask);
            v_1 = _mm256_mul_ps(_mm256_set_ps(v[(incv * 15)], v[(incv * 14)], v[(incv * 13)], v[(incv * 12)], v[(incv * 11)], v[(incv * 10)], v[(incv * 9)], v[(incv * 8)]), scale_mask);
            v_2 = _mm256_mul_ps(_mm256_set_ps(v[(incv * 23)], v[(incv * 22)], v[(incv * 21)], v[(incv * 20)], v[(incv * 19)], v[(incv * 18)], v[(incv * 17)], v[(incv * 16)]), scale_mask);
            v_3 = _mm256_mul_ps(_mm256_set_ps(v[(incv * 31)], v[(incv * 30)], v[(incv * 29)], v[(incv * 28)], v[(incv * 27)], v[(incv * 26)], v[(incv * 25)], v[(incv * 24)]), scale_mask);
            v_4 = _mm256_mul_ps(_mm256_set_ps(v[(incv * 39)], v[(incv * 38)], v[(incv * 37)], v[(incv * 36)], v[(incv * 35)], v[(incv * 34)], v[(incv * 33)], v[(incv * 32)]), scale_mask);
            v_5 = _mm256_mul_ps(_mm256_set_ps(v[(incv * 47)], v[(incv * 46)], v[(incv * 45)], v[(incv * 44)], v[(incv * 43)], v[(incv * 42)], v[(incv * 41)], v[(incv * 40)]), scale_mask);
            v_6 = _mm256_mul_ps(_mm256_set_ps(v[(incv * 55)], v[(incv * 54)], v[(incv * 53)], v[(incv * 52)], v[(incv * 51)], v[(incv * 50)], v[(incv * 49)], v[(incv * 48)]), scale_mask);
            v_7 = _mm256_mul_ps(_mm256_set_ps(v[(incv * 63)], v[(incv * 62)], v[(incv * 61)], v[(incv * 60)], v[(incv * 59)], v[(incv * 58)], v[(incv * 57)], v[(incv * 56)]), scale_mask);
            v_0 = _mm256_mul_ps(v_0, v_0);
            v_1 = _mm256_mul_ps(v_1, v_1);
            v_2 = _mm256_mul_ps(v_2, v_2);
            v_3 = _mm256_mul_ps(v_3, v_3);
            v_4 = _mm256_mul_ps(v_4, v_4);
            v_5 = _mm256_mul_ps(v_5, v_5);
            v_6 = _mm256_mul_ps(v_6, v_6);
            v_7 = _mm256_mul_ps(v_7, v_7);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_0, mask_BLP));
            s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(v_1, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            q_1 = _mm256_sub_ps(q_1, s_0_1);
            v_0 = _mm256_add_ps(v_0, q_0);
            v_1 = _mm256_add_ps(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_0, mask_BLP));
            s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(v_1, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            q_1 = _mm256_sub_ps(q_1, s_1_1);
            v_0 = _mm256_add_ps(v_0, q_0);
            v_1 = _mm256_add_ps(v_1, q_1);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_0, mask_BLP));
            s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(v_1, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_2, mask_BLP));
            s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(v_3, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            q_1 = _mm256_sub_ps(q_1, s_0_1);
            v_2 = _mm256_add_ps(v_2, q_0);
            v_3 = _mm256_add_ps(v_3, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_2, mask_BLP));
            s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(v_3, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            q_1 = _mm256_sub_ps(q_1, s_1_1);
            v_2 = _mm256_add_ps(v_2, q_0);
            v_3 = _mm256_add_ps(v_3, q_1);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_2, mask_BLP));
            s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(v_3, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_4, mask_BLP));
            s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(v_5, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            q_1 = _mm256_sub_ps(q_1, s_0_1);
            v_4 = _mm256_add_ps(v_4, q_0);
            v_5 = _mm256_add_ps(v_5, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_4, mask_BLP));
            s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(v_5, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            q_1 = _mm256_sub_ps(q_1, s_1_1);
            v_4 = _mm256_add_ps(v_4, q_0);
            v_5 = _mm256_add_ps(v_5, q_1);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_4, mask_BLP));
            s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(v_5, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_6, mask_BLP));
            s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(v_7, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            q_1 = _mm256_sub_ps(q_1, s_0_1);
            v_6 = _mm256_add_ps(v_6, q_0);
            v_7 = _mm256_add_ps(v_7, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_6, mask_BLP));
            s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(v_7, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            q_1 = _mm256_sub_ps(q_1, s_1_1);
            v_6 = _mm256_add_ps(v_6, q_0);
            v_7 = _mm256_add_ps(v_7, q_1);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_6, mask_BLP));
            s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(v_7, mask_BLP));
          }
          if(i + 32 <= n){
            v_0 = _mm256_mul_ps(_mm256_set_ps(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)], v[(incv * 3)], v[(incv * 2)], v[incv], v[0]), scale_mask);
            v_1 = _mm256_mul_ps(_mm256_set_ps(v[(incv * 15)], v[(incv * 14)], v[(incv * 13)], v[(incv * 12)], v[(incv * 11)], v[(incv * 10)], v[(incv * 9)], v[(incv * 8)]), scale_mask);
            v_2 = _mm256_mul_ps(_mm256_set_ps(v[(incv * 23)], v[(incv * 22)], v[(incv * 21)], v[(incv * 20)], v[(incv * 19)], v[(incv * 18)], v[(incv * 17)], v[(incv * 16)]), scale_mask);
            v_3 = _mm256_mul_ps(_mm256_set_ps(v[(incv * 31)], v[(incv * 30)], v[(incv * 29)], v[(incv * 28)], v[(incv * 27)], v[(incv * 26)], v[(incv * 25)], v[(incv * 24)]), scale_mask);
            v_0 = _mm256_mul_ps(v_0, v_0);
            v_1 = _mm256_mul_ps(v_1, v_1);
            v_2 = _mm256_mul_ps(v_2, v_2);
            v_3 = _mm256_mul_ps(v_3, v_3);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_0, mask_BLP));
            s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(v_1, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            q_1 = _mm256_sub_ps(q_1, s_0_1);
            v_0 = _mm256_add_ps(v_0, q_0);
            v_1 = _mm256_add_ps(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_0, mask_BLP));
            s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(v_1, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            q_1 = _mm256_sub_ps(q_1, s_1_1);
            v_0 = _mm256_add_ps(v_0, q_0);
            v_1 = _mm256_add_ps(v_1, q_1);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_0, mask_BLP));
            s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(v_1, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_2, mask_BLP));
            s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(v_3, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            q_1 = _mm256_sub_ps(q_1, s_0_1);
            v_2 = _mm256_add_ps(v_2, q_0);
            v_3 = _mm256_add_ps(v_3, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_2, mask_BLP));
            s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(v_3, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            q_1 = _mm256_sub_ps(q_1, s_1_1);
            v_2 = _mm256_add_ps(v_2, q_0);
            v_3 = _mm256_add_ps(v_3, q_1);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_2, mask_BLP));
            s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(v_3, mask_BLP));
            i += 32, v += (incv * 32);
          }
          if(i + 16 <= n){
            v_0 = _mm256_mul_ps(_mm256_set_ps(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)], v[(incv * 3)], v[(incv * 2)], v[incv], v[0]), scale_mask);
            v_1 = _mm256_mul_ps(_mm256_set_ps(v[(incv * 15)], v[(incv * 14)], v[(incv * 13)], v[(incv * 12)], v[(incv * 11)], v[(incv * 10)], v[(incv * 9)], v[(incv * 8)]), scale_mask);
            v_0 = _mm256_mul_ps(v_0, v_0);
            v_1 = _mm256_mul_ps(v_1, v_1);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_0, mask_BLP));
            s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(v_1, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            q_1 = _mm256_sub_ps(q_1, s_0_1);
            v_0 = _mm256_add_ps(v_0, q_0);
            v_1 = _mm256_add_ps(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_0, mask_BLP));
            s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(v_1, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            q_1 = _mm256_sub_ps(q_1, s_1_1);
            v_0 = _mm256_add_ps(v_0, q_0);
            v_1 = _mm256_add_ps(v_1, q_1);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_0, mask_BLP));
            s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(v_1, mask_BLP));
            i += 16, v += (incv * 16);
          }
          if(i + 8 <= n){
            v_0 = _mm256_mul_ps(_mm256_set_ps(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)], v[(incv * 3)], v[(incv * 2)], v[incv], v[0]), scale_mask);
            v_0 = _mm256_mul_ps(v_0, v_0);
            q_0 = s_0_0;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_0, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            v_0 = _mm256_add_ps(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_0, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            v_0 = _mm256_add_ps(v_0, q_0);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_0, mask_BLP));
            i += 8, v += (incv * 8);
          }
          if(i < n){
            v_0 = _mm256_mul_ps(_mm256_set_ps(0, (n - i)>6?v[(incv * 6)]:0, (n - i)>5?v[(incv * 5)]:0, (n - i)>4?v[(incv * 4)]:0, (n - i)>3?v[(incv * 3)]:0, (n - i)>2?v[(incv * 2)]:0, (n - i)>1?v[incv]:0, v[0]), scale_mask);
            v_0 = _mm256_mul_ps(v_0, v_0);
            q_0 = s_0_0;
            s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(v_0, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_0_0);
            v_0 = _mm256_add_ps(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(v_0, mask_BLP));
            q_0 = _mm256_sub_ps(q_0, s_1_0);
            v_0 = _mm256_add_ps(v_0, q_0);
            s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(v_0, mask_BLP));
          }
        }
        s_0_0 = _mm256_sub_ps(s_0_0, _mm256_set_ps(sum[0], sum[0], sum[0], sum[0], sum[0], sum[0], sum[0], 0));
        q_0 = _mm256_broadcast_ss(sum);
        s_0_0 = _mm256_add_ps(s_0_0, _mm256_sub_ps(s_0_1, q_0));
        _mm256_store_ps(tmp_cons, s_0_0);
        sum[0] = tmp_cons[0] + tmp_cons[1] + tmp_cons[2] + tmp_cons[3] + tmp_cons[4] + tmp_cons[5] + tmp_cons[6] + tmp_cons[7];
        s_1_0 = _mm256_sub_ps(s_1_0, _mm256_set_ps(sum[1], sum[1], sum[1], sum[1], sum[1], sum[1], sum[1], 0));
        q_0 = _mm256_broadcast_ss(sum + 1);
        s_1_0 = _mm256_add_ps(s_1_0, _mm256_sub_ps(s_1_1, q_0));
        _mm256_store_ps(tmp_cons, s_1_0);
        sum[1] = tmp_cons[0] + tmp_cons[1] + tmp_cons[2] + tmp_cons[3] + tmp_cons[4] + tmp_cons[5] + tmp_cons[6] + tmp_cons[7];
        s_2_0 = _mm256_sub_ps(s_2_0, _mm256_set_ps(sum[2], sum[2], sum[2], sum[2], sum[2], sum[2], sum[2], 0));
        q_0 = _mm256_broadcast_ss(sum + 2);
        s_2_0 = _mm256_add_ps(s_2_0, _mm256_sub_ps(s_2_1, q_0));
        _mm256_store_ps(tmp_cons, s_2_0);
        sum[2] = tmp_cons[0] + tmp_cons[1] + tmp_cons[2] + tmp_cons[3] + tmp_cons[4] + tmp_cons[5] + tmp_cons[6] + tmp_cons[7];
        RESET_DAZ_FLAG
        return;
      }
      default:{
        int i, j;

        __m256 v_0;
        __m256 q_0;
        __m256 s_0;
        __m256 s_buffer[MAX_FOLD];

        for(j = 0; j < fold; j += 1){
          s_buffer[j] = _mm256_broadcast_ss(sum + j);
        }
        if(incv == 1){

          for(i = 0; i + 8 <= n; i += 8, v += 8){
            v_0 = _mm256_mul_ps(_mm256_loadu_ps(v), scale_mask);
            v_0 = _mm256_mul_ps(v_0, v_0);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[j];
              q_0 = _mm256_add_ps(s_0, _mm256_or_ps(v_0, mask_BLP));
              s_buffer[j] = q_0;
              q_0 = _mm256_sub_ps(s_0, q_0);
              v_0 = _mm256_add_ps(v_0, q_0);
            }
            s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(v_0, mask_BLP));
          }
          if(i < n){
            v_0 = _mm256_mul_ps(_mm256_set_ps(0, (n - i)>6?v[6]:0, (n - i)>5?v[5]:0, (n - i)>4?v[4]:0, (n - i)>3?v[3]:0, (n - i)>2?v[2]:0, (n - i)>1?v[1]:0, v[0]), scale_mask);
            v_0 = _mm256_mul_ps(v_0, v_0);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[j];
              q_0 = _mm256_add_ps(s_0, _mm256_or_ps(v_0, mask_BLP));
              s_buffer[j] = q_0;
              q_0 = _mm256_sub_ps(s_0, q_0);
              v_0 = _mm256_add_ps(v_0, q_0);
            }
            s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(v_0, mask_BLP));
          }
        }else{

          for(i = 0; i + 8 <= n; i += 8, v += (incv * 8)){
            v_0 = _mm256_mul_ps(_mm256_set_ps(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)], v[(incv * 3)], v[(incv * 2)], v[incv], v[0]), scale_mask);
            v_0 = _mm256_mul_ps(v_0, v_0);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[j];
              q_0 = _mm256_add_ps(s_0, _mm256_or_ps(v_0, mask_BLP));
              s_buffer[j] = q_0;
              q_0 = _mm256_sub_ps(s_0, q_0);
              v_0 = _mm256_add_ps(v_0, q_0);
            }
            s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(v_0, mask_BLP));
          }
          if(i < n){
            v_0 = _mm256_mul_ps(_mm256_set_ps(0, (n - i)>6?v[(incv * 6)]:0, (n - i)>5?v[(incv * 5)]:0, (n - i)>4?v[(incv * 4)]:0, (n - i)>3?v[(incv * 3)]:0, (n - i)>2?v[(incv * 2)]:0, (n - i)>1?v[incv]:0, v[0]), scale_mask);
            v_0 = _mm256_mul_ps(v_0, v_0);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[j];
              q_0 = _mm256_add_ps(s_0, _mm256_or_ps(v_0, mask_BLP));
              s_buffer[j] = q_0;
              q_0 = _mm256_sub_ps(s_0, q_0);
              v_0 = _mm256_add_ps(v_0, q_0);
            }
            s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(v_0, mask_BLP));
          }
        }
        for(j = 0; j < fold; j += 1){
          s_buffer[j] = _mm256_sub_ps(s_buffer[j], _mm256_set_ps(sum[j], sum[j], sum[j], sum[j], sum[j], sum[j], sum[j], 0));
          _mm256_store_ps(tmp_cons, s_buffer[j]);
          sum[j] = tmp_cons[0] + tmp_cons[1] + tmp_cons[2] + tmp_cons[3] + tmp_cons[4] + tmp_cons[5] + tmp_cons[6] + tmp_cons[7];
        }
        RESET_DAZ_FLAG
        return;
      }
    }
  }
#elif defined( __SSE2__ )
  void snrm2I2(int n, float* v, int incv, float scale, int fold, float* sum){
    __m128 scale_mask = _mm_set1_ps(scale);
    __m128 mask_BLP; SSE_BLP_MASKS(mask_BLP);
    float tmp_cons[4] __attribute__((aligned(16)));
    SET_DAZ_FLAG;
    switch(fold){
      case 3:{
        int i;

        __m128 v_0, v_1, v_2, v_3;
        __m128 q_0, q_1, q_2, q_3;
        __m128 s_0_0, s_0_1, s_0_2, s_0_3;
        __m128 s_1_0, s_1_1, s_1_2, s_1_3;
        __m128 s_2_0, s_2_1, s_2_2, s_2_3;

        s_0_0 = s_0_1 = s_0_2 = s_0_3 = _mm_load1_ps(sum);
        s_1_0 = s_1_1 = s_1_2 = s_1_3 = _mm_load1_ps(sum + 1);
        s_2_0 = s_2_1 = s_2_2 = s_2_3 = _mm_load1_ps(sum + 2);
        if(incv == 1){

          for(i = 0; i + 16 <= n; i += 16, v += 16){
            v_0 = _mm_mul_ps(_mm_loadu_ps(v), scale_mask);
            v_1 = _mm_mul_ps(_mm_loadu_ps(v + 4), scale_mask);
            v_2 = _mm_mul_ps(_mm_loadu_ps(v + 8), scale_mask);
            v_3 = _mm_mul_ps(_mm_loadu_ps(v + 12), scale_mask);
            v_0 = _mm_mul_ps(v_0, v_0);
            v_1 = _mm_mul_ps(v_1, v_1);
            v_2 = _mm_mul_ps(v_2, v_2);
            v_3 = _mm_mul_ps(v_3, v_3);
            q_0 = s_0_0;
            q_1 = s_0_1;
            q_2 = s_0_2;
            q_3 = s_0_3;
            s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(v_0, mask_BLP));
            s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(v_1, mask_BLP));
            s_0_2 = _mm_add_ps(s_0_2, _mm_or_ps(v_2, mask_BLP));
            s_0_3 = _mm_add_ps(s_0_3, _mm_or_ps(v_3, mask_BLP));
            q_0 = _mm_sub_ps(q_0, s_0_0);
            q_1 = _mm_sub_ps(q_1, s_0_1);
            q_2 = _mm_sub_ps(q_2, s_0_2);
            q_3 = _mm_sub_ps(q_3, s_0_3);
            v_0 = _mm_add_ps(v_0, q_0);
            v_1 = _mm_add_ps(v_1, q_1);
            v_2 = _mm_add_ps(v_2, q_2);
            v_3 = _mm_add_ps(v_3, q_3);
            q_0 = s_1_0;
            q_1 = s_1_1;
            q_2 = s_1_2;
            q_3 = s_1_3;
            s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(v_0, mask_BLP));
            s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(v_1, mask_BLP));
            s_1_2 = _mm_add_ps(s_1_2, _mm_or_ps(v_2, mask_BLP));
            s_1_3 = _mm_add_ps(s_1_3, _mm_or_ps(v_3, mask_BLP));
            q_0 = _mm_sub_ps(q_0, s_1_0);
            q_1 = _mm_sub_ps(q_1, s_1_1);
            q_2 = _mm_sub_ps(q_2, s_1_2);
            q_3 = _mm_sub_ps(q_3, s_1_3);
            v_0 = _mm_add_ps(v_0, q_0);
            v_1 = _mm_add_ps(v_1, q_1);
            v_2 = _mm_add_ps(v_2, q_2);
            v_3 = _mm_add_ps(v_3, q_3);
            s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(v_0, mask_BLP));
            s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(v_1, mask_BLP));
            s_2_2 = _mm_add_ps(s_2_2, _mm_or_ps(v_2, mask_BLP));
            s_2_3 = _mm_add_ps(s_2_3, _mm_or_ps(v_3, mask_BLP));
          }
          if(i + 8 <= n){
            v_0 = _mm_mul_ps(_mm_loadu_ps(v), scale_mask);
            v_1 = _mm_mul_ps(_mm_loadu_ps(v + 4), scale_mask);
            v_0 = _mm_mul_ps(v_0, v_0);
            v_1 = _mm_mul_ps(v_1, v_1);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(v_0, mask_BLP));
            s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(v_1, mask_BLP));
            q_0 = _mm_sub_ps(q_0, s_0_0);
            q_1 = _mm_sub_ps(q_1, s_0_1);
            v_0 = _mm_add_ps(v_0, q_0);
            v_1 = _mm_add_ps(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(v_0, mask_BLP));
            s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(v_1, mask_BLP));
            q_0 = _mm_sub_ps(q_0, s_1_0);
            q_1 = _mm_sub_ps(q_1, s_1_1);
            v_0 = _mm_add_ps(v_0, q_0);
            v_1 = _mm_add_ps(v_1, q_1);
            s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(v_0, mask_BLP));
            s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(v_1, mask_BLP));
            i += 8, v += 8;
          }
          if(i + 4 <= n){
            v_0 = _mm_mul_ps(_mm_loadu_ps(v), scale_mask);
            v_0 = _mm_mul_ps(v_0, v_0);
            q_0 = s_0_0;
            s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(v_0, mask_BLP));
            q_0 = _mm_sub_ps(q_0, s_0_0);
            v_0 = _mm_add_ps(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(v_0, mask_BLP));
            q_0 = _mm_sub_ps(q_0, s_1_0);
            v_0 = _mm_add_ps(v_0, q_0);
            s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(v_0, mask_BLP));
            i += 4, v += 4;
          }
          if(i < n){
            v_0 = _mm_mul_ps(_mm_set_ps(0, (n - i)>2?v[2]:0, (n - i)>1?v[1]:0, v[0]), scale_mask);
            v_0 = _mm_mul_ps(v_0, v_0);
            q_0 = s_0_0;
            s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(v_0, mask_BLP));
            q_0 = _mm_sub_ps(q_0, s_0_0);
            v_0 = _mm_add_ps(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(v_0, mask_BLP));
            q_0 = _mm_sub_ps(q_0, s_1_0);
            v_0 = _mm_add_ps(v_0, q_0);
            s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(v_0, mask_BLP));
          }
        }else{

          for(i = 0; i + 16 <= n; i += 16, v += (incv * 16)){
            v_0 = _mm_mul_ps(_mm_set_ps(v[(incv * 3)], v[(incv * 2)], v[incv], v[0]), scale_mask);
            v_1 = _mm_mul_ps(_mm_set_ps(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)]), scale_mask);
            v_2 = _mm_mul_ps(_mm_set_ps(v[(incv * 11)], v[(incv * 10)], v[(incv * 9)], v[(incv * 8)]), scale_mask);
            v_3 = _mm_mul_ps(_mm_set_ps(v[(incv * 15)], v[(incv * 14)], v[(incv * 13)], v[(incv * 12)]), scale_mask);
            v_0 = _mm_mul_ps(v_0, v_0);
            v_1 = _mm_mul_ps(v_1, v_1);
            v_2 = _mm_mul_ps(v_2, v_2);
            v_3 = _mm_mul_ps(v_3, v_3);
            q_0 = s_0_0;
            q_1 = s_0_1;
            q_2 = s_0_2;
            q_3 = s_0_3;
            s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(v_0, mask_BLP));
            s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(v_1, mask_BLP));
            s_0_2 = _mm_add_ps(s_0_2, _mm_or_ps(v_2, mask_BLP));
            s_0_3 = _mm_add_ps(s_0_3, _mm_or_ps(v_3, mask_BLP));
            q_0 = _mm_sub_ps(q_0, s_0_0);
            q_1 = _mm_sub_ps(q_1, s_0_1);
            q_2 = _mm_sub_ps(q_2, s_0_2);
            q_3 = _mm_sub_ps(q_3, s_0_3);
            v_0 = _mm_add_ps(v_0, q_0);
            v_1 = _mm_add_ps(v_1, q_1);
            v_2 = _mm_add_ps(v_2, q_2);
            v_3 = _mm_add_ps(v_3, q_3);
            q_0 = s_1_0;
            q_1 = s_1_1;
            q_2 = s_1_2;
            q_3 = s_1_3;
            s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(v_0, mask_BLP));
            s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(v_1, mask_BLP));
            s_1_2 = _mm_add_ps(s_1_2, _mm_or_ps(v_2, mask_BLP));
            s_1_3 = _mm_add_ps(s_1_3, _mm_or_ps(v_3, mask_BLP));
            q_0 = _mm_sub_ps(q_0, s_1_0);
            q_1 = _mm_sub_ps(q_1, s_1_1);
            q_2 = _mm_sub_ps(q_2, s_1_2);
            q_3 = _mm_sub_ps(q_3, s_1_3);
            v_0 = _mm_add_ps(v_0, q_0);
            v_1 = _mm_add_ps(v_1, q_1);
            v_2 = _mm_add_ps(v_2, q_2);
            v_3 = _mm_add_ps(v_3, q_3);
            s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(v_0, mask_BLP));
            s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(v_1, mask_BLP));
            s_2_2 = _mm_add_ps(s_2_2, _mm_or_ps(v_2, mask_BLP));
            s_2_3 = _mm_add_ps(s_2_3, _mm_or_ps(v_3, mask_BLP));
          }
          if(i + 8 <= n){
            v_0 = _mm_mul_ps(_mm_set_ps(v[(incv * 3)], v[(incv * 2)], v[incv], v[0]), scale_mask);
            v_1 = _mm_mul_ps(_mm_set_ps(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)]), scale_mask);
            v_0 = _mm_mul_ps(v_0, v_0);
            v_1 = _mm_mul_ps(v_1, v_1);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(v_0, mask_BLP));
            s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(v_1, mask_BLP));
            q_0 = _mm_sub_ps(q_0, s_0_0);
            q_1 = _mm_sub_ps(q_1, s_0_1);
            v_0 = _mm_add_ps(v_0, q_0);
            v_1 = _mm_add_ps(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(v_0, mask_BLP));
            s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(v_1, mask_BLP));
            q_0 = _mm_sub_ps(q_0, s_1_0);
            q_1 = _mm_sub_ps(q_1, s_1_1);
            v_0 = _mm_add_ps(v_0, q_0);
            v_1 = _mm_add_ps(v_1, q_1);
            s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(v_0, mask_BLP));
            s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(v_1, mask_BLP));
            i += 8, v += (incv * 8);
          }
          if(i + 4 <= n){
            v_0 = _mm_mul_ps(_mm_set_ps(v[(incv * 3)], v[(incv * 2)], v[incv], v[0]), scale_mask);
            v_0 = _mm_mul_ps(v_0, v_0);
            q_0 = s_0_0;
            s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(v_0, mask_BLP));
            q_0 = _mm_sub_ps(q_0, s_0_0);
            v_0 = _mm_add_ps(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(v_0, mask_BLP));
            q_0 = _mm_sub_ps(q_0, s_1_0);
            v_0 = _mm_add_ps(v_0, q_0);
            s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(v_0, mask_BLP));
            i += 4, v += (incv * 4);
          }
          if(i < n){
            v_0 = _mm_mul_ps(_mm_set_ps(0, (n - i)>2?v[(incv * 2)]:0, (n - i)>1?v[incv]:0, v[0]), scale_mask);
            v_0 = _mm_mul_ps(v_0, v_0);
            q_0 = s_0_0;
            s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(v_0, mask_BLP));
            q_0 = _mm_sub_ps(q_0, s_0_0);
            v_0 = _mm_add_ps(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(v_0, mask_BLP));
            q_0 = _mm_sub_ps(q_0, s_1_0);
            v_0 = _mm_add_ps(v_0, q_0);
            s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(v_0, mask_BLP));
          }
        }
        s_0_0 = _mm_sub_ps(s_0_0, _mm_set_ps(sum[0], sum[0], sum[0], 0));
        q_0 = _mm_load1_ps(sum);
        s_0_0 = _mm_add_ps(s_0_0, _mm_sub_ps(s_0_1, q_0));
        s_0_0 = _mm_add_ps(s_0_0, _mm_sub_ps(s_0_2, q_0));
        s_0_0 = _mm_add_ps(s_0_0, _mm_sub_ps(s_0_3, q_0));
        _mm_store_ps(tmp_cons, s_0_0);
        sum[0] = tmp_cons[0] + tmp_cons[1] + tmp_cons[2] + tmp_cons[3];
        s_1_0 = _mm_sub_ps(s_1_0, _mm_set_ps(sum[1], sum[1], sum[1], 0));
        q_0 = _mm_load1_ps(sum + 1);
        s_1_0 = _mm_add_ps(s_1_0, _mm_sub_ps(s_1_1, q_0));
        s_1_0 = _mm_add_ps(s_1_0, _mm_sub_ps(s_1_2, q_0));
        s_1_0 = _mm_add_ps(s_1_0, _mm_sub_ps(s_1_3, q_0));
        _mm_store_ps(tmp_cons, s_1_0);
        sum[1] = tmp_cons[0] + tmp_cons[1] + tmp_cons[2] + tmp_cons[3];
        s_2_0 = _mm_sub_ps(s_2_0, _mm_set_ps(sum[2], sum[2], sum[2], 0));
        q_0 = _mm_load1_ps(sum + 2);
        s_2_0 = _mm_add_ps(s_2_0, _mm_sub_ps(s_2_1, q_0));
        s_2_0 = _mm_add_ps(s_2_0, _mm_sub_ps(s_2_2, q_0));
        s_2_0 = _mm_add_ps(s_2_0, _mm_sub_ps(s_2_3, q_0));
        _mm_store_ps(tmp_cons, s_2_0);
        sum[2] = tmp_cons[0] + tmp_cons[1] + tmp_cons[2] + tmp_cons[3];
        RESET_DAZ_FLAG
        return;
      }
      default:{
        int i, j;

        __m128 v_0, v_1;
        __m128 q_0, q_1;
        __m128 s_0, s_1;
        __m128 s_buffer[(MAX_FOLD * 2)];

        for(j = 0; j < fold; j += 1){
          s_buffer[(j * 2)] = s_buffer[((j * 2) + 1)] = _mm_load1_ps(sum + j);
        }
        if(incv == 1){

          for(i = 0; i + 8 <= n; i += 8, v += 8){
            v_0 = _mm_mul_ps(_mm_loadu_ps(v), scale_mask);
            v_1 = _mm_mul_ps(_mm_loadu_ps(v + 4), scale_mask);
            v_0 = _mm_mul_ps(v_0, v_0);
            v_1 = _mm_mul_ps(v_1, v_1);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              q_0 = _mm_add_ps(s_0, _mm_or_ps(v_0, mask_BLP));
              q_1 = _mm_add_ps(s_1, _mm_or_ps(v_1, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = _mm_sub_ps(s_0, q_0);
              q_1 = _mm_sub_ps(s_1, q_1);
              v_0 = _mm_add_ps(v_0, q_0);
              v_1 = _mm_add_ps(v_1, q_1);
            }
            s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(v_0, mask_BLP));
            s_buffer[((j * 2) + 1)] = _mm_add_ps(s_buffer[((j * 2) + 1)], _mm_or_ps(v_1, mask_BLP));
          }
          if(i + 4 <= n){
            v_0 = _mm_mul_ps(_mm_loadu_ps(v), scale_mask);
            v_0 = _mm_mul_ps(v_0, v_0);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              q_0 = _mm_add_ps(s_0, _mm_or_ps(v_0, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              q_0 = _mm_sub_ps(s_0, q_0);
              v_0 = _mm_add_ps(v_0, q_0);
            }
            s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(v_0, mask_BLP));
            i += 4, v += 4;
          }
          if(i < n){
            v_0 = _mm_mul_ps(_mm_set_ps(0, (n - i)>2?v[2]:0, (n - i)>1?v[1]:0, v[0]), scale_mask);
            v_0 = _mm_mul_ps(v_0, v_0);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              q_0 = _mm_add_ps(s_0, _mm_or_ps(v_0, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              q_0 = _mm_sub_ps(s_0, q_0);
              v_0 = _mm_add_ps(v_0, q_0);
            }
            s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(v_0, mask_BLP));
          }
        }else{

          for(i = 0; i + 8 <= n; i += 8, v += (incv * 8)){
            v_0 = _mm_mul_ps(_mm_set_ps(v[(incv * 3)], v[(incv * 2)], v[incv], v[0]), scale_mask);
            v_1 = _mm_mul_ps(_mm_set_ps(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)]), scale_mask);
            v_0 = _mm_mul_ps(v_0, v_0);
            v_1 = _mm_mul_ps(v_1, v_1);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              q_0 = _mm_add_ps(s_0, _mm_or_ps(v_0, mask_BLP));
              q_1 = _mm_add_ps(s_1, _mm_or_ps(v_1, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = _mm_sub_ps(s_0, q_0);
              q_1 = _mm_sub_ps(s_1, q_1);
              v_0 = _mm_add_ps(v_0, q_0);
              v_1 = _mm_add_ps(v_1, q_1);
            }
            s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(v_0, mask_BLP));
            s_buffer[((j * 2) + 1)] = _mm_add_ps(s_buffer[((j * 2) + 1)], _mm_or_ps(v_1, mask_BLP));
          }
          if(i + 4 <= n){
            v_0 = _mm_mul_ps(_mm_set_ps(v[(incv * 3)], v[(incv * 2)], v[incv], v[0]), scale_mask);
            v_0 = _mm_mul_ps(v_0, v_0);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              q_0 = _mm_add_ps(s_0, _mm_or_ps(v_0, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              q_0 = _mm_sub_ps(s_0, q_0);
              v_0 = _mm_add_ps(v_0, q_0);
            }
            s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(v_0, mask_BLP));
            i += 4, v += (incv * 4);
          }
          if(i < n){
            v_0 = _mm_mul_ps(_mm_set_ps(0, (n - i)>2?v[(incv * 2)]:0, (n - i)>1?v[incv]:0, v[0]), scale_mask);
            v_0 = _mm_mul_ps(v_0, v_0);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              q_0 = _mm_add_ps(s_0, _mm_or_ps(v_0, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              q_0 = _mm_sub_ps(s_0, q_0);
              v_0 = _mm_add_ps(v_0, q_0);
            }
            s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(v_0, mask_BLP));
          }
        }
        for(j = 0; j < fold; j += 1){
          s_buffer[(j * 2)] = _mm_sub_ps(s_buffer[(j * 2)], _mm_set_ps(sum[j], sum[j], sum[j], 0));
          q_0 = _mm_load1_ps(sum + j);
          s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_sub_ps(s_buffer[((j * 2) + 1)], q_0));
          _mm_store_ps(tmp_cons, s_buffer[(j * 2)]);
          sum[j] = tmp_cons[0] + tmp_cons[1] + tmp_cons[2] + tmp_cons[3];
        }
        RESET_DAZ_FLAG
        return;
      }
    }
  }
#else
  void snrm2I2(int n, float* v, int incv, float scale, int fold, float* sum){
    float scale_mask = scale;
    int_float tmp_BLP;
    SET_DAZ_FLAG;
    switch(fold){
      case 3:{
        int i;

        float v_0, v_1;
        float q_0, q_1;
        float s_0_0, s_0_1;
        float s_1_0, s_1_1;
        float s_2_0, s_2_1;

        s_0_0 = s_0_1 = sum[0];
        s_1_0 = s_1_1 = sum[1];
        s_2_0 = s_2_1 = sum[2];
        if(incv == 1){

          for(i = 0; i + 2 <= n; i += 2, v += 2){
            v_0 = v[0] * scale_mask;
            v_1 = v[1] * scale_mask;
            v_0 = v_0 * v_0;
            v_1 = v_1 * v_1;
            q_0 = s_0_0;
            q_1 = s_0_1;
            tmp_BLP.f = v_0;
            tmp_BLP.i |= 1;
            s_0_0 = s_0_0 + tmp_BLP.f;
            tmp_BLP.f = v_1;
            tmp_BLP.i |= 1;
            s_0_1 = s_0_1 + tmp_BLP.f;
            q_0 = q_0 - s_0_0;
            q_1 = q_1 - s_0_1;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            q_0 = s_1_0;
            q_1 = s_1_1;
            tmp_BLP.f = v_0;
            tmp_BLP.i |= 1;
            s_1_0 = s_1_0 + tmp_BLP.f;
            tmp_BLP.f = v_1;
            tmp_BLP.i |= 1;
            s_1_1 = s_1_1 + tmp_BLP.f;
            q_0 = q_0 - s_1_0;
            q_1 = q_1 - s_1_1;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            tmp_BLP.f = v_0;
            tmp_BLP.i |= 1;
            s_2_0 = s_2_0 + tmp_BLP.f;
            tmp_BLP.f = v_1;
            tmp_BLP.i |= 1;
            s_2_1 = s_2_1 + tmp_BLP.f;
          }
          if(i + 1 <= n){
            v_0 = v[0] * scale_mask;
            v_0 = v_0 * v_0;
            q_0 = s_0_0;
            tmp_BLP.f = v_0;
            tmp_BLP.i |= 1;
            s_0_0 = s_0_0 + tmp_BLP.f;
            q_0 = q_0 - s_0_0;
            v_0 = v_0 + q_0;
            q_0 = s_1_0;
            tmp_BLP.f = v_0;
            tmp_BLP.i |= 1;
            s_1_0 = s_1_0 + tmp_BLP.f;
            q_0 = q_0 - s_1_0;
            v_0 = v_0 + q_0;
            tmp_BLP.f = v_0;
            tmp_BLP.i |= 1;
            s_2_0 = s_2_0 + tmp_BLP.f;
            i += 1, v += 1;
          }
        }else{

          for(i = 0; i + 2 <= n; i += 2, v += (incv * 2)){
            v_0 = v[0] * scale_mask;
            v_1 = v[incv] * scale_mask;
            v_0 = v_0 * v_0;
            v_1 = v_1 * v_1;
            q_0 = s_0_0;
            q_1 = s_0_1;
            tmp_BLP.f = v_0;
            tmp_BLP.i |= 1;
            s_0_0 = s_0_0 + tmp_BLP.f;
            tmp_BLP.f = v_1;
            tmp_BLP.i |= 1;
            s_0_1 = s_0_1 + tmp_BLP.f;
            q_0 = q_0 - s_0_0;
            q_1 = q_1 - s_0_1;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            q_0 = s_1_0;
            q_1 = s_1_1;
            tmp_BLP.f = v_0;
            tmp_BLP.i |= 1;
            s_1_0 = s_1_0 + tmp_BLP.f;
            tmp_BLP.f = v_1;
            tmp_BLP.i |= 1;
            s_1_1 = s_1_1 + tmp_BLP.f;
            q_0 = q_0 - s_1_0;
            q_1 = q_1 - s_1_1;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            tmp_BLP.f = v_0;
            tmp_BLP.i |= 1;
            s_2_0 = s_2_0 + tmp_BLP.f;
            tmp_BLP.f = v_1;
            tmp_BLP.i |= 1;
            s_2_1 = s_2_1 + tmp_BLP.f;
          }
          if(i + 1 <= n){
            v_0 = v[0] * scale_mask;
            v_0 = v_0 * v_0;
            q_0 = s_0_0;
            tmp_BLP.f = v_0;
            tmp_BLP.i |= 1;
            s_0_0 = s_0_0 + tmp_BLP.f;
            q_0 = q_0 - s_0_0;
            v_0 = v_0 + q_0;
            q_0 = s_1_0;
            tmp_BLP.f = v_0;
            tmp_BLP.i |= 1;
            s_1_0 = s_1_0 + tmp_BLP.f;
            q_0 = q_0 - s_1_0;
            v_0 = v_0 + q_0;
            tmp_BLP.f = v_0;
            tmp_BLP.i |= 1;
            s_2_0 = s_2_0 + tmp_BLP.f;
            i += 1, v += incv;
          }
        }
        q_0 = sum[0];
        s_0_0 = s_0_0 + (s_0_1 - q_0);
        sum[0] = s_0_0;
        q_0 = sum[1];
        s_1_0 = s_1_0 + (s_1_1 - q_0);
        sum[1] = s_1_0;
        q_0 = sum[2];
        s_2_0 = s_2_0 + (s_2_1 - q_0);
        sum[2] = s_2_0;
        RESET_DAZ_FLAG
        return;
      }
      default:{
        int i, j;

        float v_0, v_1;
        float q_0, q_1;
        float s_0, s_1;
        float s_buffer[(MAX_FOLD * 2)];

        for(j = 0; j < fold; j += 1){
          s_buffer[(j * 2)] = s_buffer[((j * 2) + 1)] = sum[j];
        }
        if(incv == 1){

          for(i = 0; i + 2 <= n; i += 2, v += 2){
            v_0 = v[0] * scale_mask;
            v_1 = v[1] * scale_mask;
            v_0 = v_0 * v_0;
            v_1 = v_1 * v_1;
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              tmp_BLP.f = v_0;
              tmp_BLP.i |= 1;
              q_0 = s_0 + tmp_BLP.f;
              tmp_BLP.f = v_1;
              tmp_BLP.i |= 1;
              q_1 = s_1 + tmp_BLP.f;
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = s_0 - q_0;
              q_1 = s_1 - q_1;
              v_0 = v_0 + q_0;
              v_1 = v_1 + q_1;
            }
            tmp_BLP.f = v_0;
            tmp_BLP.i |= 1;
            s_buffer[(j * 2)] = s_buffer[(j * 2)] + tmp_BLP.f;
            tmp_BLP.f = v_1;
            tmp_BLP.i |= 1;
            s_buffer[((j * 2) + 1)] = s_buffer[((j * 2) + 1)] + tmp_BLP.f;
          }
          if(i + 1 <= n){
            v_0 = v[0] * scale_mask;
            v_0 = v_0 * v_0;
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              tmp_BLP.f = v_0;
              tmp_BLP.i |= 1;
              q_0 = s_0 + tmp_BLP.f;
              s_buffer[(j * 2)] = q_0;
              q_0 = s_0 - q_0;
              v_0 = v_0 + q_0;
            }
            tmp_BLP.f = v_0;
            tmp_BLP.i |= 1;
            s_buffer[(j * 2)] = s_buffer[(j * 2)] + tmp_BLP.f;
            i += 1, v += 1;
          }
        }else{

          for(i = 0; i + 2 <= n; i += 2, v += (incv * 2)){
            v_0 = v[0] * scale_mask;
            v_1 = v[incv] * scale_mask;
            v_0 = v_0 * v_0;
            v_1 = v_1 * v_1;
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              tmp_BLP.f = v_0;
              tmp_BLP.i |= 1;
              q_0 = s_0 + tmp_BLP.f;
              tmp_BLP.f = v_1;
              tmp_BLP.i |= 1;
              q_1 = s_1 + tmp_BLP.f;
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = s_0 - q_0;
              q_1 = s_1 - q_1;
              v_0 = v_0 + q_0;
              v_1 = v_1 + q_1;
            }
            tmp_BLP.f = v_0;
            tmp_BLP.i |= 1;
            s_buffer[(j * 2)] = s_buffer[(j * 2)] + tmp_BLP.f;
            tmp_BLP.f = v_1;
            tmp_BLP.i |= 1;
            s_buffer[((j * 2) + 1)] = s_buffer[((j * 2) + 1)] + tmp_BLP.f;
          }
          if(i + 1 <= n){
            v_0 = v[0] * scale_mask;
            v_0 = v_0 * v_0;
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              tmp_BLP.f = v_0;
              tmp_BLP.i |= 1;
              q_0 = s_0 + tmp_BLP.f;
              s_buffer[(j * 2)] = q_0;
              q_0 = s_0 - q_0;
              v_0 = v_0 + q_0;
            }
            tmp_BLP.f = v_0;
            tmp_BLP.i |= 1;
            s_buffer[(j * 2)] = s_buffer[(j * 2)] + tmp_BLP.f;
            i += 1, v += incv;
          }
        }
        for(j = 0; j < fold; j += 1){
          q_0 = sum[j];
          s_buffer[(j * 2)] = s_buffer[(j * 2)] + (s_buffer[((j * 2) + 1)] - q_0);
          sum[j] = s_buffer[(j * 2)];
        }
        RESET_DAZ_FLAG
        return;
      }
    }
  }
#endif