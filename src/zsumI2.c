#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "config.h"
#include "Common/Common.h"
#include <immintrin.h>
#include <emmintrin.h>


#if defined( __AVX__ )
  void zsumI2(int n, double complex* v, int incv, int fold, double complex* sum){
    __m256d mask_BLP; AVX_BLP_MASKD(mask_BLP);
    double complex tmp_cons[2] __attribute__((aligned(32)));
    SET_DAZ_FLAG;
    switch(fold){
      case 3:{
        int i;

        double* sum_base = (double*) sum;
        double* v_base = (double*) v;
        __m256d v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7;
        __m256d q_0, q_1;
        __m256d s_0_0, s_0_1;
        __m256d s_1_0, s_1_1;
        __m256d s_2_0, s_2_1;

        s_0_0 = s_0_1 = _mm256_broadcast_pd((__m128d *)(sum_base));
        s_1_0 = s_1_1 = _mm256_broadcast_pd((__m128d *)(sum_base + 2));
        s_2_0 = s_2_1 = _mm256_broadcast_pd((__m128d *)(sum_base + 4));
        if(incv == 1){

          for(i = 0; i + 16 <= n; i += 16, v_base += 32){
            v_0 = _mm256_loadu_pd(v_base);
            v_1 = _mm256_loadu_pd(v_base + 4);
            v_2 = _mm256_loadu_pd(v_base + 8);
            v_3 = _mm256_loadu_pd(v_base + 12);
            v_4 = _mm256_loadu_pd(v_base + 16);
            v_5 = _mm256_loadu_pd(v_base + 20);
            v_6 = _mm256_loadu_pd(v_base + 24);
            v_7 = _mm256_loadu_pd(v_base + 28);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_2, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_3, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_2 = _mm256_add_pd(v_2, q_0);
            v_3 = _mm256_add_pd(v_3, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_2, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_3, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_2 = _mm256_add_pd(v_2, q_0);
            v_3 = _mm256_add_pd(v_3, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_2, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_3, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_4, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_5, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_4 = _mm256_add_pd(v_4, q_0);
            v_5 = _mm256_add_pd(v_5, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_4, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_5, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_4 = _mm256_add_pd(v_4, q_0);
            v_5 = _mm256_add_pd(v_5, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_4, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_5, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_6, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_7, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_6 = _mm256_add_pd(v_6, q_0);
            v_7 = _mm256_add_pd(v_7, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_6, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_7, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_6 = _mm256_add_pd(v_6, q_0);
            v_7 = _mm256_add_pd(v_7, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_6, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_7, mask_BLP));
          }
          if(i + 8 <= n){
            v_0 = _mm256_loadu_pd(v_base);
            v_1 = _mm256_loadu_pd(v_base + 4);
            v_2 = _mm256_loadu_pd(v_base + 8);
            v_3 = _mm256_loadu_pd(v_base + 12);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_2, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_3, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_2 = _mm256_add_pd(v_2, q_0);
            v_3 = _mm256_add_pd(v_3, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_2, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_3, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_2 = _mm256_add_pd(v_2, q_0);
            v_3 = _mm256_add_pd(v_3, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_2, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_3, mask_BLP));
            i += 8, v_base += 16;
          }
          if(i + 4 <= n){
            v_0 = _mm256_loadu_pd(v_base);
            v_1 = _mm256_loadu_pd(v_base + 4);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_1, mask_BLP));
            i += 4, v_base += 8;
          }
          if(i + 2 <= n){
            v_0 = _mm256_loadu_pd(v_base);
            q_0 = s_0_0;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            v_0 = _mm256_add_pd(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            v_0 = _mm256_add_pd(v_0, q_0);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
            i += 2, v_base += 4;
          }
          if(i < n){
            v_0 = _mm256_set_pd(0, 0, v_base[1], v_base[0]);
            q_0 = s_0_0;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            v_0 = _mm256_add_pd(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            v_0 = _mm256_add_pd(v_0, q_0);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
          }
        }else{

          for(i = 0; i + 16 <= n; i += 16, v_base += (incv * 32)){
            v_0 = _mm256_set_pd(v_base[((incv * 2) + 1)], v_base[(incv * 2)], v_base[1], v_base[0]);
            v_1 = _mm256_set_pd(v_base[((incv * 6) + 1)], v_base[(incv * 6)], v_base[((incv * 4) + 1)], v_base[(incv * 4)]);
            v_2 = _mm256_set_pd(v_base[((incv * 10) + 1)], v_base[(incv * 10)], v_base[((incv * 8) + 1)], v_base[(incv * 8)]);
            v_3 = _mm256_set_pd(v_base[((incv * 14) + 1)], v_base[(incv * 14)], v_base[((incv * 12) + 1)], v_base[(incv * 12)]);
            v_4 = _mm256_set_pd(v_base[((incv * 18) + 1)], v_base[(incv * 18)], v_base[((incv * 16) + 1)], v_base[(incv * 16)]);
            v_5 = _mm256_set_pd(v_base[((incv * 22) + 1)], v_base[(incv * 22)], v_base[((incv * 20) + 1)], v_base[(incv * 20)]);
            v_6 = _mm256_set_pd(v_base[((incv * 26) + 1)], v_base[(incv * 26)], v_base[((incv * 24) + 1)], v_base[(incv * 24)]);
            v_7 = _mm256_set_pd(v_base[((incv * 30) + 1)], v_base[(incv * 30)], v_base[((incv * 28) + 1)], v_base[(incv * 28)]);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_2, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_3, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_2 = _mm256_add_pd(v_2, q_0);
            v_3 = _mm256_add_pd(v_3, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_2, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_3, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_2 = _mm256_add_pd(v_2, q_0);
            v_3 = _mm256_add_pd(v_3, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_2, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_3, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_4, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_5, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_4 = _mm256_add_pd(v_4, q_0);
            v_5 = _mm256_add_pd(v_5, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_4, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_5, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_4 = _mm256_add_pd(v_4, q_0);
            v_5 = _mm256_add_pd(v_5, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_4, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_5, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_6, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_7, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_6 = _mm256_add_pd(v_6, q_0);
            v_7 = _mm256_add_pd(v_7, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_6, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_7, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_6 = _mm256_add_pd(v_6, q_0);
            v_7 = _mm256_add_pd(v_7, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_6, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_7, mask_BLP));
          }
          if(i + 8 <= n){
            v_0 = _mm256_set_pd(v_base[((incv * 2) + 1)], v_base[(incv * 2)], v_base[1], v_base[0]);
            v_1 = _mm256_set_pd(v_base[((incv * 6) + 1)], v_base[(incv * 6)], v_base[((incv * 4) + 1)], v_base[(incv * 4)]);
            v_2 = _mm256_set_pd(v_base[((incv * 10) + 1)], v_base[(incv * 10)], v_base[((incv * 8) + 1)], v_base[(incv * 8)]);
            v_3 = _mm256_set_pd(v_base[((incv * 14) + 1)], v_base[(incv * 14)], v_base[((incv * 12) + 1)], v_base[(incv * 12)]);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_2, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_3, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_2 = _mm256_add_pd(v_2, q_0);
            v_3 = _mm256_add_pd(v_3, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_2, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_3, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_2 = _mm256_add_pd(v_2, q_0);
            v_3 = _mm256_add_pd(v_3, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_2, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_3, mask_BLP));
            i += 8, v_base += (incv * 16);
          }
          if(i + 4 <= n){
            v_0 = _mm256_set_pd(v_base[((incv * 2) + 1)], v_base[(incv * 2)], v_base[1], v_base[0]);
            v_1 = _mm256_set_pd(v_base[((incv * 6) + 1)], v_base[(incv * 6)], v_base[((incv * 4) + 1)], v_base[(incv * 4)]);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_1, mask_BLP));
            i += 4, v_base += (incv * 8);
          }
          if(i + 2 <= n){
            v_0 = _mm256_set_pd(v_base[((incv * 2) + 1)], v_base[(incv * 2)], v_base[1], v_base[0]);
            q_0 = s_0_0;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            v_0 = _mm256_add_pd(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            v_0 = _mm256_add_pd(v_0, q_0);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
            i += 2, v_base += (incv * 4);
          }
          if(i < n){
            v_0 = _mm256_set_pd(0, 0, v_base[1], v_base[0]);
            q_0 = s_0_0;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            v_0 = _mm256_add_pd(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            v_0 = _mm256_add_pd(v_0, q_0);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
          }
        }
        s_0_0 = _mm256_sub_pd(s_0_0, _mm256_set_pd(sum_base[1], sum_base[0], 0, 0));
        q_0 = _mm256_broadcast_pd((__m128d *)(sum_base));
        s_0_0 = _mm256_add_pd(s_0_0, _mm256_sub_pd(s_0_1, q_0));
        _mm256_store_pd((double*)tmp_cons, s_0_0);
        sum[0] = tmp_cons[0] + tmp_cons[1];
        s_1_0 = _mm256_sub_pd(s_1_0, _mm256_set_pd(sum_base[3], sum_base[2], 0, 0));
        q_0 = _mm256_broadcast_pd((__m128d *)(sum_base + 2));
        s_1_0 = _mm256_add_pd(s_1_0, _mm256_sub_pd(s_1_1, q_0));
        _mm256_store_pd((double*)tmp_cons, s_1_0);
        sum[1] = tmp_cons[0] + tmp_cons[1];
        s_2_0 = _mm256_sub_pd(s_2_0, _mm256_set_pd(sum_base[5], sum_base[4], 0, 0));
        q_0 = _mm256_broadcast_pd((__m128d *)(sum_base + 4));
        s_2_0 = _mm256_add_pd(s_2_0, _mm256_sub_pd(s_2_1, q_0));
        _mm256_store_pd((double*)tmp_cons, s_2_0);
        sum[2] = tmp_cons[0] + tmp_cons[1];
        RESET_DAZ_FLAG
        return;
      }
      default:{
        int i, j;

        double* sum_base = (double*) sum;
        double* v_base = (double*) v;
        __m256d v_0, v_1, v_2, v_3;
        __m256d q_0, q_1, q_2, q_3;
        __m256d s_0, s_1, s_2, s_3;
        __m256d s_buffer[(MAX_FOLD * 4)];

        for(j = 0; j < fold; j += 1){
          s_buffer[(j * 4)] = s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 3)] = _mm256_broadcast_pd((__m128d *)(sum_base + (j * 2)));
        }
        if(incv == 1){

          for(i = 0; i + 8 <= n; i += 8, v_base += 16){
            v_0 = _mm256_loadu_pd(v_base);
            v_1 = _mm256_loadu_pd(v_base + 4);
            v_2 = _mm256_loadu_pd(v_base + 8);
            v_3 = _mm256_loadu_pd(v_base + 12);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 4)];
              s_1 = s_buffer[((j * 4) + 1)];
              s_2 = s_buffer[((j * 4) + 2)];
              s_3 = s_buffer[((j * 4) + 3)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_0, mask_BLP));
              q_1 = _mm256_add_pd(s_1, _mm256_or_pd(v_1, mask_BLP));
              q_2 = _mm256_add_pd(s_2, _mm256_or_pd(v_2, mask_BLP));
              q_3 = _mm256_add_pd(s_3, _mm256_or_pd(v_3, mask_BLP));
              s_buffer[(j * 4)] = q_0;
              s_buffer[((j * 4) + 1)] = q_1;
              s_buffer[((j * 4) + 2)] = q_2;
              s_buffer[((j * 4) + 3)] = q_3;
              q_0 = _mm256_sub_pd(s_0, q_0);
              q_1 = _mm256_sub_pd(s_1, q_1);
              q_2 = _mm256_sub_pd(s_2, q_2);
              q_3 = _mm256_sub_pd(s_3, q_3);
              v_0 = _mm256_add_pd(v_0, q_0);
              v_1 = _mm256_add_pd(v_1, q_1);
              v_2 = _mm256_add_pd(v_2, q_2);
              v_3 = _mm256_add_pd(v_3, q_3);
            }
            s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(v_0, mask_BLP));
            s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(v_1, mask_BLP));
            s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(v_2, mask_BLP));
            s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(v_3, mask_BLP));
          }
          if(i + 4 <= n){
            v_0 = _mm256_loadu_pd(v_base);
            v_1 = _mm256_loadu_pd(v_base + 4);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 4)];
              s_1 = s_buffer[((j * 4) + 1)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_0, mask_BLP));
              q_1 = _mm256_add_pd(s_1, _mm256_or_pd(v_1, mask_BLP));
              s_buffer[(j * 4)] = q_0;
              s_buffer[((j * 4) + 1)] = q_1;
              q_0 = _mm256_sub_pd(s_0, q_0);
              q_1 = _mm256_sub_pd(s_1, q_1);
              v_0 = _mm256_add_pd(v_0, q_0);
              v_1 = _mm256_add_pd(v_1, q_1);
            }
            s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(v_0, mask_BLP));
            s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(v_1, mask_BLP));
            i += 4, v_base += 8;
          }
          if(i + 2 <= n){
            v_0 = _mm256_loadu_pd(v_base);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 4)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_0, mask_BLP));
              s_buffer[(j * 4)] = q_0;
              q_0 = _mm256_sub_pd(s_0, q_0);
              v_0 = _mm256_add_pd(v_0, q_0);
            }
            s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(v_0, mask_BLP));
            i += 2, v_base += 4;
          }
          if(i < n){
            v_0 = _mm256_set_pd(0, 0, v_base[1], v_base[0]);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 4)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_0, mask_BLP));
              s_buffer[(j * 4)] = q_0;
              q_0 = _mm256_sub_pd(s_0, q_0);
              v_0 = _mm256_add_pd(v_0, q_0);
            }
            s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(v_0, mask_BLP));
          }
        }else{

          for(i = 0; i + 8 <= n; i += 8, v_base += (incv * 16)){
            v_0 = _mm256_set_pd(v_base[((incv * 2) + 1)], v_base[(incv * 2)], v_base[1], v_base[0]);
            v_1 = _mm256_set_pd(v_base[((incv * 6) + 1)], v_base[(incv * 6)], v_base[((incv * 4) + 1)], v_base[(incv * 4)]);
            v_2 = _mm256_set_pd(v_base[((incv * 10) + 1)], v_base[(incv * 10)], v_base[((incv * 8) + 1)], v_base[(incv * 8)]);
            v_3 = _mm256_set_pd(v_base[((incv * 14) + 1)], v_base[(incv * 14)], v_base[((incv * 12) + 1)], v_base[(incv * 12)]);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 4)];
              s_1 = s_buffer[((j * 4) + 1)];
              s_2 = s_buffer[((j * 4) + 2)];
              s_3 = s_buffer[((j * 4) + 3)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_0, mask_BLP));
              q_1 = _mm256_add_pd(s_1, _mm256_or_pd(v_1, mask_BLP));
              q_2 = _mm256_add_pd(s_2, _mm256_or_pd(v_2, mask_BLP));
              q_3 = _mm256_add_pd(s_3, _mm256_or_pd(v_3, mask_BLP));
              s_buffer[(j * 4)] = q_0;
              s_buffer[((j * 4) + 1)] = q_1;
              s_buffer[((j * 4) + 2)] = q_2;
              s_buffer[((j * 4) + 3)] = q_3;
              q_0 = _mm256_sub_pd(s_0, q_0);
              q_1 = _mm256_sub_pd(s_1, q_1);
              q_2 = _mm256_sub_pd(s_2, q_2);
              q_3 = _mm256_sub_pd(s_3, q_3);
              v_0 = _mm256_add_pd(v_0, q_0);
              v_1 = _mm256_add_pd(v_1, q_1);
              v_2 = _mm256_add_pd(v_2, q_2);
              v_3 = _mm256_add_pd(v_3, q_3);
            }
            s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(v_0, mask_BLP));
            s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(v_1, mask_BLP));
            s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(v_2, mask_BLP));
            s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(v_3, mask_BLP));
          }
          if(i + 4 <= n){
            v_0 = _mm256_set_pd(v_base[((incv * 2) + 1)], v_base[(incv * 2)], v_base[1], v_base[0]);
            v_1 = _mm256_set_pd(v_base[((incv * 6) + 1)], v_base[(incv * 6)], v_base[((incv * 4) + 1)], v_base[(incv * 4)]);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 4)];
              s_1 = s_buffer[((j * 4) + 1)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_0, mask_BLP));
              q_1 = _mm256_add_pd(s_1, _mm256_or_pd(v_1, mask_BLP));
              s_buffer[(j * 4)] = q_0;
              s_buffer[((j * 4) + 1)] = q_1;
              q_0 = _mm256_sub_pd(s_0, q_0);
              q_1 = _mm256_sub_pd(s_1, q_1);
              v_0 = _mm256_add_pd(v_0, q_0);
              v_1 = _mm256_add_pd(v_1, q_1);
            }
            s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(v_0, mask_BLP));
            s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(v_1, mask_BLP));
            i += 4, v_base += (incv * 8);
          }
          if(i + 2 <= n){
            v_0 = _mm256_set_pd(v_base[((incv * 2) + 1)], v_base[(incv * 2)], v_base[1], v_base[0]);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 4)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_0, mask_BLP));
              s_buffer[(j * 4)] = q_0;
              q_0 = _mm256_sub_pd(s_0, q_0);
              v_0 = _mm256_add_pd(v_0, q_0);
            }
            s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(v_0, mask_BLP));
            i += 2, v_base += (incv * 4);
          }
          if(i < n){
            v_0 = _mm256_set_pd(0, 0, v_base[1], v_base[0]);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 4)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_0, mask_BLP));
              s_buffer[(j * 4)] = q_0;
              q_0 = _mm256_sub_pd(s_0, q_0);
              v_0 = _mm256_add_pd(v_0, q_0);
            }
            s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(v_0, mask_BLP));
          }
        }
        for(j = 0; j < fold; j += 1){
          s_buffer[(j * 4)] = _mm256_sub_pd(s_buffer[(j * 4)], _mm256_set_pd(sum_base[((j * 2) + 1)], sum_base[(j * 2)], 0, 0));
          q_0 = _mm256_broadcast_pd((__m128d *)(sum_base + (j * 2)));
          s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_sub_pd(s_buffer[((j * 4) + 1)], q_0));
          s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_sub_pd(s_buffer[((j * 4) + 2)], q_0));
          s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_sub_pd(s_buffer[((j * 4) + 3)], q_0));
          _mm256_store_pd((double*)tmp_cons, s_buffer[(j * 4)]);
          sum[j] = tmp_cons[0] + tmp_cons[1];
        }
        RESET_DAZ_FLAG
        return;
      }
    }
  }
#elif defined( __SSE2__ )
  void zsumI2(int n, double complex* v, int incv, int fold, double complex* sum){
    __m128d mask_BLP; SSE_BLP_MASKD(mask_BLP);
    double complex tmp_cons[1] __attribute__((aligned(16)));
    SET_DAZ_FLAG;
    switch(fold){
      case 3:{
        int i;

        double* sum_base = (double*) sum;
        double* v_base = (double*) v;
        __m128d v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7;
        __m128d q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7;
        __m128d s_0_0, s_0_1, s_0_2, s_0_3, s_0_4, s_0_5, s_0_6, s_0_7;
        __m128d s_1_0, s_1_1, s_1_2, s_1_3, s_1_4, s_1_5, s_1_6, s_1_7;
        __m128d s_2_0, s_2_1, s_2_2, s_2_3, s_2_4, s_2_5, s_2_6, s_2_7;

        s_0_0 = s_0_1 = s_0_2 = s_0_3 = s_0_4 = s_0_5 = s_0_6 = s_0_7 = _mm_loadu_pd(sum_base);
        s_1_0 = s_1_1 = s_1_2 = s_1_3 = s_1_4 = s_1_5 = s_1_6 = s_1_7 = _mm_loadu_pd(sum_base + 2);
        s_2_0 = s_2_1 = s_2_2 = s_2_3 = s_2_4 = s_2_5 = s_2_6 = s_2_7 = _mm_loadu_pd(sum_base + 4);
        if(incv == 1){

          for(i = 0; i + 8 <= n; i += 8, v_base += 16){
            v_0 = _mm_loadu_pd(v_base);
            v_1 = _mm_loadu_pd(v_base + 2);
            v_2 = _mm_loadu_pd(v_base + 4);
            v_3 = _mm_loadu_pd(v_base + 6);
            v_4 = _mm_loadu_pd(v_base + 8);
            v_5 = _mm_loadu_pd(v_base + 10);
            v_6 = _mm_loadu_pd(v_base + 12);
            v_7 = _mm_loadu_pd(v_base + 14);
            q_0 = s_0_0;
            q_1 = s_0_1;
            q_2 = s_0_2;
            q_3 = s_0_3;
            q_4 = s_0_4;
            q_5 = s_0_5;
            q_6 = s_0_6;
            q_7 = s_0_7;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_0, mask_BLP));
            s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(v_1, mask_BLP));
            s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(v_2, mask_BLP));
            s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(v_3, mask_BLP));
            s_0_4 = _mm_add_pd(s_0_4, _mm_or_pd(v_4, mask_BLP));
            s_0_5 = _mm_add_pd(s_0_5, _mm_or_pd(v_5, mask_BLP));
            s_0_6 = _mm_add_pd(s_0_6, _mm_or_pd(v_6, mask_BLP));
            s_0_7 = _mm_add_pd(s_0_7, _mm_or_pd(v_7, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            q_1 = _mm_sub_pd(q_1, s_0_1);
            q_2 = _mm_sub_pd(q_2, s_0_2);
            q_3 = _mm_sub_pd(q_3, s_0_3);
            q_4 = _mm_sub_pd(q_4, s_0_4);
            q_5 = _mm_sub_pd(q_5, s_0_5);
            q_6 = _mm_sub_pd(q_6, s_0_6);
            q_7 = _mm_sub_pd(q_7, s_0_7);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            v_2 = _mm_add_pd(v_2, q_2);
            v_3 = _mm_add_pd(v_3, q_3);
            v_4 = _mm_add_pd(v_4, q_4);
            v_5 = _mm_add_pd(v_5, q_5);
            v_6 = _mm_add_pd(v_6, q_6);
            v_7 = _mm_add_pd(v_7, q_7);
            q_0 = s_1_0;
            q_1 = s_1_1;
            q_2 = s_1_2;
            q_3 = s_1_3;
            q_4 = s_1_4;
            q_5 = s_1_5;
            q_6 = s_1_6;
            q_7 = s_1_7;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_0, mask_BLP));
            s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(v_1, mask_BLP));
            s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(v_2, mask_BLP));
            s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(v_3, mask_BLP));
            s_1_4 = _mm_add_pd(s_1_4, _mm_or_pd(v_4, mask_BLP));
            s_1_5 = _mm_add_pd(s_1_5, _mm_or_pd(v_5, mask_BLP));
            s_1_6 = _mm_add_pd(s_1_6, _mm_or_pd(v_6, mask_BLP));
            s_1_7 = _mm_add_pd(s_1_7, _mm_or_pd(v_7, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            q_1 = _mm_sub_pd(q_1, s_1_1);
            q_2 = _mm_sub_pd(q_2, s_1_2);
            q_3 = _mm_sub_pd(q_3, s_1_3);
            q_4 = _mm_sub_pd(q_4, s_1_4);
            q_5 = _mm_sub_pd(q_5, s_1_5);
            q_6 = _mm_sub_pd(q_6, s_1_6);
            q_7 = _mm_sub_pd(q_7, s_1_7);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            v_2 = _mm_add_pd(v_2, q_2);
            v_3 = _mm_add_pd(v_3, q_3);
            v_4 = _mm_add_pd(v_4, q_4);
            v_5 = _mm_add_pd(v_5, q_5);
            v_6 = _mm_add_pd(v_6, q_6);
            v_7 = _mm_add_pd(v_7, q_7);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_0, mask_BLP));
            s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(v_1, mask_BLP));
            s_2_2 = _mm_add_pd(s_2_2, _mm_or_pd(v_2, mask_BLP));
            s_2_3 = _mm_add_pd(s_2_3, _mm_or_pd(v_3, mask_BLP));
            s_2_4 = _mm_add_pd(s_2_4, _mm_or_pd(v_4, mask_BLP));
            s_2_5 = _mm_add_pd(s_2_5, _mm_or_pd(v_5, mask_BLP));
            s_2_6 = _mm_add_pd(s_2_6, _mm_or_pd(v_6, mask_BLP));
            s_2_7 = _mm_add_pd(s_2_7, _mm_or_pd(v_7, mask_BLP));
          }
          if(i + 4 <= n){
            v_0 = _mm_loadu_pd(v_base);
            v_1 = _mm_loadu_pd(v_base + 2);
            v_2 = _mm_loadu_pd(v_base + 4);
            v_3 = _mm_loadu_pd(v_base + 6);
            q_0 = s_0_0;
            q_1 = s_0_1;
            q_2 = s_0_2;
            q_3 = s_0_3;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_0, mask_BLP));
            s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(v_1, mask_BLP));
            s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(v_2, mask_BLP));
            s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(v_3, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            q_1 = _mm_sub_pd(q_1, s_0_1);
            q_2 = _mm_sub_pd(q_2, s_0_2);
            q_3 = _mm_sub_pd(q_3, s_0_3);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            v_2 = _mm_add_pd(v_2, q_2);
            v_3 = _mm_add_pd(v_3, q_3);
            q_0 = s_1_0;
            q_1 = s_1_1;
            q_2 = s_1_2;
            q_3 = s_1_3;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_0, mask_BLP));
            s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(v_1, mask_BLP));
            s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(v_2, mask_BLP));
            s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(v_3, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            q_1 = _mm_sub_pd(q_1, s_1_1);
            q_2 = _mm_sub_pd(q_2, s_1_2);
            q_3 = _mm_sub_pd(q_3, s_1_3);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            v_2 = _mm_add_pd(v_2, q_2);
            v_3 = _mm_add_pd(v_3, q_3);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_0, mask_BLP));
            s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(v_1, mask_BLP));
            s_2_2 = _mm_add_pd(s_2_2, _mm_or_pd(v_2, mask_BLP));
            s_2_3 = _mm_add_pd(s_2_3, _mm_or_pd(v_3, mask_BLP));
            i += 4, v_base += 8;
          }
          if(i + 2 <= n){
            v_0 = _mm_loadu_pd(v_base);
            v_1 = _mm_loadu_pd(v_base + 2);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_0, mask_BLP));
            s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(v_1, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            q_1 = _mm_sub_pd(q_1, s_0_1);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_0, mask_BLP));
            s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(v_1, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            q_1 = _mm_sub_pd(q_1, s_1_1);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_0, mask_BLP));
            s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(v_1, mask_BLP));
            i += 2, v_base += 4;
          }
          if(i + 1 <= n){
            v_0 = _mm_loadu_pd(v_base);
            q_0 = s_0_0;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_0, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            v_0 = _mm_add_pd(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_0, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            v_0 = _mm_add_pd(v_0, q_0);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_0, mask_BLP));
            i += 1, v_base += 2;
          }
        }else{

          for(i = 0; i + 8 <= n; i += 8, v_base += (incv * 16)){
            v_0 = _mm_loadu_pd(v_base);
            v_1 = _mm_loadu_pd(v_base + (incv * 2));
            v_2 = _mm_loadu_pd(v_base + (incv * 4));
            v_3 = _mm_loadu_pd(v_base + (incv * 6));
            v_4 = _mm_loadu_pd(v_base + (incv * 8));
            v_5 = _mm_loadu_pd(v_base + (incv * 10));
            v_6 = _mm_loadu_pd(v_base + (incv * 12));
            v_7 = _mm_loadu_pd(v_base + (incv * 14));
            q_0 = s_0_0;
            q_1 = s_0_1;
            q_2 = s_0_2;
            q_3 = s_0_3;
            q_4 = s_0_4;
            q_5 = s_0_5;
            q_6 = s_0_6;
            q_7 = s_0_7;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_0, mask_BLP));
            s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(v_1, mask_BLP));
            s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(v_2, mask_BLP));
            s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(v_3, mask_BLP));
            s_0_4 = _mm_add_pd(s_0_4, _mm_or_pd(v_4, mask_BLP));
            s_0_5 = _mm_add_pd(s_0_5, _mm_or_pd(v_5, mask_BLP));
            s_0_6 = _mm_add_pd(s_0_6, _mm_or_pd(v_6, mask_BLP));
            s_0_7 = _mm_add_pd(s_0_7, _mm_or_pd(v_7, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            q_1 = _mm_sub_pd(q_1, s_0_1);
            q_2 = _mm_sub_pd(q_2, s_0_2);
            q_3 = _mm_sub_pd(q_3, s_0_3);
            q_4 = _mm_sub_pd(q_4, s_0_4);
            q_5 = _mm_sub_pd(q_5, s_0_5);
            q_6 = _mm_sub_pd(q_6, s_0_6);
            q_7 = _mm_sub_pd(q_7, s_0_7);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            v_2 = _mm_add_pd(v_2, q_2);
            v_3 = _mm_add_pd(v_3, q_3);
            v_4 = _mm_add_pd(v_4, q_4);
            v_5 = _mm_add_pd(v_5, q_5);
            v_6 = _mm_add_pd(v_6, q_6);
            v_7 = _mm_add_pd(v_7, q_7);
            q_0 = s_1_0;
            q_1 = s_1_1;
            q_2 = s_1_2;
            q_3 = s_1_3;
            q_4 = s_1_4;
            q_5 = s_1_5;
            q_6 = s_1_6;
            q_7 = s_1_7;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_0, mask_BLP));
            s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(v_1, mask_BLP));
            s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(v_2, mask_BLP));
            s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(v_3, mask_BLP));
            s_1_4 = _mm_add_pd(s_1_4, _mm_or_pd(v_4, mask_BLP));
            s_1_5 = _mm_add_pd(s_1_5, _mm_or_pd(v_5, mask_BLP));
            s_1_6 = _mm_add_pd(s_1_6, _mm_or_pd(v_6, mask_BLP));
            s_1_7 = _mm_add_pd(s_1_7, _mm_or_pd(v_7, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            q_1 = _mm_sub_pd(q_1, s_1_1);
            q_2 = _mm_sub_pd(q_2, s_1_2);
            q_3 = _mm_sub_pd(q_3, s_1_3);
            q_4 = _mm_sub_pd(q_4, s_1_4);
            q_5 = _mm_sub_pd(q_5, s_1_5);
            q_6 = _mm_sub_pd(q_6, s_1_6);
            q_7 = _mm_sub_pd(q_7, s_1_7);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            v_2 = _mm_add_pd(v_2, q_2);
            v_3 = _mm_add_pd(v_3, q_3);
            v_4 = _mm_add_pd(v_4, q_4);
            v_5 = _mm_add_pd(v_5, q_5);
            v_6 = _mm_add_pd(v_6, q_6);
            v_7 = _mm_add_pd(v_7, q_7);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_0, mask_BLP));
            s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(v_1, mask_BLP));
            s_2_2 = _mm_add_pd(s_2_2, _mm_or_pd(v_2, mask_BLP));
            s_2_3 = _mm_add_pd(s_2_3, _mm_or_pd(v_3, mask_BLP));
            s_2_4 = _mm_add_pd(s_2_4, _mm_or_pd(v_4, mask_BLP));
            s_2_5 = _mm_add_pd(s_2_5, _mm_or_pd(v_5, mask_BLP));
            s_2_6 = _mm_add_pd(s_2_6, _mm_or_pd(v_6, mask_BLP));
            s_2_7 = _mm_add_pd(s_2_7, _mm_or_pd(v_7, mask_BLP));
          }
          if(i + 4 <= n){
            v_0 = _mm_loadu_pd(v_base);
            v_1 = _mm_loadu_pd(v_base + (incv * 2));
            v_2 = _mm_loadu_pd(v_base + (incv * 4));
            v_3 = _mm_loadu_pd(v_base + (incv * 6));
            q_0 = s_0_0;
            q_1 = s_0_1;
            q_2 = s_0_2;
            q_3 = s_0_3;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_0, mask_BLP));
            s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(v_1, mask_BLP));
            s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(v_2, mask_BLP));
            s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(v_3, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            q_1 = _mm_sub_pd(q_1, s_0_1);
            q_2 = _mm_sub_pd(q_2, s_0_2);
            q_3 = _mm_sub_pd(q_3, s_0_3);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            v_2 = _mm_add_pd(v_2, q_2);
            v_3 = _mm_add_pd(v_3, q_3);
            q_0 = s_1_0;
            q_1 = s_1_1;
            q_2 = s_1_2;
            q_3 = s_1_3;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_0, mask_BLP));
            s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(v_1, mask_BLP));
            s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(v_2, mask_BLP));
            s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(v_3, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            q_1 = _mm_sub_pd(q_1, s_1_1);
            q_2 = _mm_sub_pd(q_2, s_1_2);
            q_3 = _mm_sub_pd(q_3, s_1_3);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            v_2 = _mm_add_pd(v_2, q_2);
            v_3 = _mm_add_pd(v_3, q_3);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_0, mask_BLP));
            s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(v_1, mask_BLP));
            s_2_2 = _mm_add_pd(s_2_2, _mm_or_pd(v_2, mask_BLP));
            s_2_3 = _mm_add_pd(s_2_3, _mm_or_pd(v_3, mask_BLP));
            i += 4, v_base += (incv * 8);
          }
          if(i + 2 <= n){
            v_0 = _mm_loadu_pd(v_base);
            v_1 = _mm_loadu_pd(v_base + (incv * 2));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_0, mask_BLP));
            s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(v_1, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            q_1 = _mm_sub_pd(q_1, s_0_1);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_0, mask_BLP));
            s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(v_1, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            q_1 = _mm_sub_pd(q_1, s_1_1);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_0, mask_BLP));
            s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(v_1, mask_BLP));
            i += 2, v_base += (incv * 4);
          }
          if(i + 1 <= n){
            v_0 = _mm_loadu_pd(v_base);
            q_0 = s_0_0;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_0, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            v_0 = _mm_add_pd(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_0, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            v_0 = _mm_add_pd(v_0, q_0);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_0, mask_BLP));
            i += 1, v_base += (incv * 2);
          }
        }
        q_0 = _mm_loadu_pd(sum_base);
        s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_1, q_0));
        s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_2, q_0));
        s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_3, q_0));
        s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_4, q_0));
        s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_5, q_0));
        s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_6, q_0));
        s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_7, q_0));
        _mm_store_pd((double*)sum, s_0_0);
        q_0 = _mm_loadu_pd(sum_base + 2);
        s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_1, q_0));
        s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_2, q_0));
        s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_3, q_0));
        s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_4, q_0));
        s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_5, q_0));
        s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_6, q_0));
        s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_7, q_0));
        _mm_store_pd((double*)sum + 2, s_1_0);
        q_0 = _mm_loadu_pd(sum_base + 4);
        s_2_0 = _mm_add_pd(s_2_0, _mm_sub_pd(s_2_1, q_0));
        s_2_0 = _mm_add_pd(s_2_0, _mm_sub_pd(s_2_2, q_0));
        s_2_0 = _mm_add_pd(s_2_0, _mm_sub_pd(s_2_3, q_0));
        s_2_0 = _mm_add_pd(s_2_0, _mm_sub_pd(s_2_4, q_0));
        s_2_0 = _mm_add_pd(s_2_0, _mm_sub_pd(s_2_5, q_0));
        s_2_0 = _mm_add_pd(s_2_0, _mm_sub_pd(s_2_6, q_0));
        s_2_0 = _mm_add_pd(s_2_0, _mm_sub_pd(s_2_7, q_0));
        _mm_store_pd((double*)sum + 4, s_2_0);
        RESET_DAZ_FLAG
        return;
      }
      default:{
        int i, j;

        double* sum_base = (double*) sum;
        double* v_base = (double*) v;
        __m128d v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7;
        __m128d q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7;
        __m128d s_0, s_1, s_2, s_3, s_4, s_5, s_6, s_7;
        __m128d s_buffer[(MAX_FOLD * 8)];

        for(j = 0; j < fold; j += 1){
          s_buffer[(j * 8)] = s_buffer[((j * 8) + 1)] = s_buffer[((j * 8) + 2)] = s_buffer[((j * 8) + 3)] = s_buffer[((j * 8) + 4)] = s_buffer[((j * 8) + 5)] = s_buffer[((j * 8) + 6)] = s_buffer[((j * 8) + 7)] = _mm_loadu_pd(sum_base + (j * 2));
        }
        if(incv == 1){

          for(i = 0; i + 8 <= n; i += 8, v_base += 16){
            v_0 = _mm_loadu_pd(v_base);
            v_1 = _mm_loadu_pd(v_base + 2);
            v_2 = _mm_loadu_pd(v_base + 4);
            v_3 = _mm_loadu_pd(v_base + 6);
            v_4 = _mm_loadu_pd(v_base + 8);
            v_5 = _mm_loadu_pd(v_base + 10);
            v_6 = _mm_loadu_pd(v_base + 12);
            v_7 = _mm_loadu_pd(v_base + 14);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 8)];
              s_1 = s_buffer[((j * 8) + 1)];
              s_2 = s_buffer[((j * 8) + 2)];
              s_3 = s_buffer[((j * 8) + 3)];
              s_4 = s_buffer[((j * 8) + 4)];
              s_5 = s_buffer[((j * 8) + 5)];
              s_6 = s_buffer[((j * 8) + 6)];
              s_7 = s_buffer[((j * 8) + 7)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_0, mask_BLP));
              q_1 = _mm_add_pd(s_1, _mm_or_pd(v_1, mask_BLP));
              q_2 = _mm_add_pd(s_2, _mm_or_pd(v_2, mask_BLP));
              q_3 = _mm_add_pd(s_3, _mm_or_pd(v_3, mask_BLP));
              q_4 = _mm_add_pd(s_4, _mm_or_pd(v_4, mask_BLP));
              q_5 = _mm_add_pd(s_5, _mm_or_pd(v_5, mask_BLP));
              q_6 = _mm_add_pd(s_6, _mm_or_pd(v_6, mask_BLP));
              q_7 = _mm_add_pd(s_7, _mm_or_pd(v_7, mask_BLP));
              s_buffer[(j * 8)] = q_0;
              s_buffer[((j * 8) + 1)] = q_1;
              s_buffer[((j * 8) + 2)] = q_2;
              s_buffer[((j * 8) + 3)] = q_3;
              s_buffer[((j * 8) + 4)] = q_4;
              s_buffer[((j * 8) + 5)] = q_5;
              s_buffer[((j * 8) + 6)] = q_6;
              s_buffer[((j * 8) + 7)] = q_7;
              q_0 = _mm_sub_pd(s_0, q_0);
              q_1 = _mm_sub_pd(s_1, q_1);
              q_2 = _mm_sub_pd(s_2, q_2);
              q_3 = _mm_sub_pd(s_3, q_3);
              q_4 = _mm_sub_pd(s_4, q_4);
              q_5 = _mm_sub_pd(s_5, q_5);
              q_6 = _mm_sub_pd(s_6, q_6);
              q_7 = _mm_sub_pd(s_7, q_7);
              v_0 = _mm_add_pd(v_0, q_0);
              v_1 = _mm_add_pd(v_1, q_1);
              v_2 = _mm_add_pd(v_2, q_2);
              v_3 = _mm_add_pd(v_3, q_3);
              v_4 = _mm_add_pd(v_4, q_4);
              v_5 = _mm_add_pd(v_5, q_5);
              v_6 = _mm_add_pd(v_6, q_6);
              v_7 = _mm_add_pd(v_7, q_7);
            }
            s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(v_0, mask_BLP));
            s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(v_1, mask_BLP));
            s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(v_2, mask_BLP));
            s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(v_3, mask_BLP));
            s_buffer[((j * 8) + 4)] = _mm_add_pd(s_buffer[((j * 8) + 4)], _mm_or_pd(v_4, mask_BLP));
            s_buffer[((j * 8) + 5)] = _mm_add_pd(s_buffer[((j * 8) + 5)], _mm_or_pd(v_5, mask_BLP));
            s_buffer[((j * 8) + 6)] = _mm_add_pd(s_buffer[((j * 8) + 6)], _mm_or_pd(v_6, mask_BLP));
            s_buffer[((j * 8) + 7)] = _mm_add_pd(s_buffer[((j * 8) + 7)], _mm_or_pd(v_7, mask_BLP));
          }
          if(i + 4 <= n){
            v_0 = _mm_loadu_pd(v_base);
            v_1 = _mm_loadu_pd(v_base + 2);
            v_2 = _mm_loadu_pd(v_base + 4);
            v_3 = _mm_loadu_pd(v_base + 6);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 8)];
              s_1 = s_buffer[((j * 8) + 1)];
              s_2 = s_buffer[((j * 8) + 2)];
              s_3 = s_buffer[((j * 8) + 3)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_0, mask_BLP));
              q_1 = _mm_add_pd(s_1, _mm_or_pd(v_1, mask_BLP));
              q_2 = _mm_add_pd(s_2, _mm_or_pd(v_2, mask_BLP));
              q_3 = _mm_add_pd(s_3, _mm_or_pd(v_3, mask_BLP));
              s_buffer[(j * 8)] = q_0;
              s_buffer[((j * 8) + 1)] = q_1;
              s_buffer[((j * 8) + 2)] = q_2;
              s_buffer[((j * 8) + 3)] = q_3;
              q_0 = _mm_sub_pd(s_0, q_0);
              q_1 = _mm_sub_pd(s_1, q_1);
              q_2 = _mm_sub_pd(s_2, q_2);
              q_3 = _mm_sub_pd(s_3, q_3);
              v_0 = _mm_add_pd(v_0, q_0);
              v_1 = _mm_add_pd(v_1, q_1);
              v_2 = _mm_add_pd(v_2, q_2);
              v_3 = _mm_add_pd(v_3, q_3);
            }
            s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(v_0, mask_BLP));
            s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(v_1, mask_BLP));
            s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(v_2, mask_BLP));
            s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(v_3, mask_BLP));
            i += 4, v_base += 8;
          }
          if(i + 2 <= n){
            v_0 = _mm_loadu_pd(v_base);
            v_1 = _mm_loadu_pd(v_base + 2);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 8)];
              s_1 = s_buffer[((j * 8) + 1)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_0, mask_BLP));
              q_1 = _mm_add_pd(s_1, _mm_or_pd(v_1, mask_BLP));
              s_buffer[(j * 8)] = q_0;
              s_buffer[((j * 8) + 1)] = q_1;
              q_0 = _mm_sub_pd(s_0, q_0);
              q_1 = _mm_sub_pd(s_1, q_1);
              v_0 = _mm_add_pd(v_0, q_0);
              v_1 = _mm_add_pd(v_1, q_1);
            }
            s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(v_0, mask_BLP));
            s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(v_1, mask_BLP));
            i += 2, v_base += 4;
          }
          if(i + 1 <= n){
            v_0 = _mm_loadu_pd(v_base);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 8)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_0, mask_BLP));
              s_buffer[(j * 8)] = q_0;
              q_0 = _mm_sub_pd(s_0, q_0);
              v_0 = _mm_add_pd(v_0, q_0);
            }
            s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(v_0, mask_BLP));
            i += 1, v_base += 2;
          }
        }else{

          for(i = 0; i + 8 <= n; i += 8, v_base += (incv * 16)){
            v_0 = _mm_loadu_pd(v_base);
            v_1 = _mm_loadu_pd(v_base + (incv * 2));
            v_2 = _mm_loadu_pd(v_base + (incv * 4));
            v_3 = _mm_loadu_pd(v_base + (incv * 6));
            v_4 = _mm_loadu_pd(v_base + (incv * 8));
            v_5 = _mm_loadu_pd(v_base + (incv * 10));
            v_6 = _mm_loadu_pd(v_base + (incv * 12));
            v_7 = _mm_loadu_pd(v_base + (incv * 14));
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 8)];
              s_1 = s_buffer[((j * 8) + 1)];
              s_2 = s_buffer[((j * 8) + 2)];
              s_3 = s_buffer[((j * 8) + 3)];
              s_4 = s_buffer[((j * 8) + 4)];
              s_5 = s_buffer[((j * 8) + 5)];
              s_6 = s_buffer[((j * 8) + 6)];
              s_7 = s_buffer[((j * 8) + 7)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_0, mask_BLP));
              q_1 = _mm_add_pd(s_1, _mm_or_pd(v_1, mask_BLP));
              q_2 = _mm_add_pd(s_2, _mm_or_pd(v_2, mask_BLP));
              q_3 = _mm_add_pd(s_3, _mm_or_pd(v_3, mask_BLP));
              q_4 = _mm_add_pd(s_4, _mm_or_pd(v_4, mask_BLP));
              q_5 = _mm_add_pd(s_5, _mm_or_pd(v_5, mask_BLP));
              q_6 = _mm_add_pd(s_6, _mm_or_pd(v_6, mask_BLP));
              q_7 = _mm_add_pd(s_7, _mm_or_pd(v_7, mask_BLP));
              s_buffer[(j * 8)] = q_0;
              s_buffer[((j * 8) + 1)] = q_1;
              s_buffer[((j * 8) + 2)] = q_2;
              s_buffer[((j * 8) + 3)] = q_3;
              s_buffer[((j * 8) + 4)] = q_4;
              s_buffer[((j * 8) + 5)] = q_5;
              s_buffer[((j * 8) + 6)] = q_6;
              s_buffer[((j * 8) + 7)] = q_7;
              q_0 = _mm_sub_pd(s_0, q_0);
              q_1 = _mm_sub_pd(s_1, q_1);
              q_2 = _mm_sub_pd(s_2, q_2);
              q_3 = _mm_sub_pd(s_3, q_3);
              q_4 = _mm_sub_pd(s_4, q_4);
              q_5 = _mm_sub_pd(s_5, q_5);
              q_6 = _mm_sub_pd(s_6, q_6);
              q_7 = _mm_sub_pd(s_7, q_7);
              v_0 = _mm_add_pd(v_0, q_0);
              v_1 = _mm_add_pd(v_1, q_1);
              v_2 = _mm_add_pd(v_2, q_2);
              v_3 = _mm_add_pd(v_3, q_3);
              v_4 = _mm_add_pd(v_4, q_4);
              v_5 = _mm_add_pd(v_5, q_5);
              v_6 = _mm_add_pd(v_6, q_6);
              v_7 = _mm_add_pd(v_7, q_7);
            }
            s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(v_0, mask_BLP));
            s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(v_1, mask_BLP));
            s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(v_2, mask_BLP));
            s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(v_3, mask_BLP));
            s_buffer[((j * 8) + 4)] = _mm_add_pd(s_buffer[((j * 8) + 4)], _mm_or_pd(v_4, mask_BLP));
            s_buffer[((j * 8) + 5)] = _mm_add_pd(s_buffer[((j * 8) + 5)], _mm_or_pd(v_5, mask_BLP));
            s_buffer[((j * 8) + 6)] = _mm_add_pd(s_buffer[((j * 8) + 6)], _mm_or_pd(v_6, mask_BLP));
            s_buffer[((j * 8) + 7)] = _mm_add_pd(s_buffer[((j * 8) + 7)], _mm_or_pd(v_7, mask_BLP));
          }
          if(i + 4 <= n){
            v_0 = _mm_loadu_pd(v_base);
            v_1 = _mm_loadu_pd(v_base + (incv * 2));
            v_2 = _mm_loadu_pd(v_base + (incv * 4));
            v_3 = _mm_loadu_pd(v_base + (incv * 6));
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 8)];
              s_1 = s_buffer[((j * 8) + 1)];
              s_2 = s_buffer[((j * 8) + 2)];
              s_3 = s_buffer[((j * 8) + 3)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_0, mask_BLP));
              q_1 = _mm_add_pd(s_1, _mm_or_pd(v_1, mask_BLP));
              q_2 = _mm_add_pd(s_2, _mm_or_pd(v_2, mask_BLP));
              q_3 = _mm_add_pd(s_3, _mm_or_pd(v_3, mask_BLP));
              s_buffer[(j * 8)] = q_0;
              s_buffer[((j * 8) + 1)] = q_1;
              s_buffer[((j * 8) + 2)] = q_2;
              s_buffer[((j * 8) + 3)] = q_3;
              q_0 = _mm_sub_pd(s_0, q_0);
              q_1 = _mm_sub_pd(s_1, q_1);
              q_2 = _mm_sub_pd(s_2, q_2);
              q_3 = _mm_sub_pd(s_3, q_3);
              v_0 = _mm_add_pd(v_0, q_0);
              v_1 = _mm_add_pd(v_1, q_1);
              v_2 = _mm_add_pd(v_2, q_2);
              v_3 = _mm_add_pd(v_3, q_3);
            }
            s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(v_0, mask_BLP));
            s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(v_1, mask_BLP));
            s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(v_2, mask_BLP));
            s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(v_3, mask_BLP));
            i += 4, v_base += (incv * 8);
          }
          if(i + 2 <= n){
            v_0 = _mm_loadu_pd(v_base);
            v_1 = _mm_loadu_pd(v_base + (incv * 2));
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 8)];
              s_1 = s_buffer[((j * 8) + 1)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_0, mask_BLP));
              q_1 = _mm_add_pd(s_1, _mm_or_pd(v_1, mask_BLP));
              s_buffer[(j * 8)] = q_0;
              s_buffer[((j * 8) + 1)] = q_1;
              q_0 = _mm_sub_pd(s_0, q_0);
              q_1 = _mm_sub_pd(s_1, q_1);
              v_0 = _mm_add_pd(v_0, q_0);
              v_1 = _mm_add_pd(v_1, q_1);
            }
            s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(v_0, mask_BLP));
            s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(v_1, mask_BLP));
            i += 2, v_base += (incv * 4);
          }
          if(i + 1 <= n){
            v_0 = _mm_loadu_pd(v_base);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 8)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_0, mask_BLP));
              s_buffer[(j * 8)] = q_0;
              q_0 = _mm_sub_pd(s_0, q_0);
              v_0 = _mm_add_pd(v_0, q_0);
            }
            s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(v_0, mask_BLP));
            i += 1, v_base += (incv * 2);
          }
        }
        for(j = 0; j < fold; j += 1){
          q_0 = _mm_loadu_pd(sum_base + (j * 2));
          s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_sub_pd(s_buffer[((j * 8) + 1)], q_0));
          s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_sub_pd(s_buffer[((j * 8) + 2)], q_0));
          s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_sub_pd(s_buffer[((j * 8) + 3)], q_0));
          s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_sub_pd(s_buffer[((j * 8) + 4)], q_0));
          s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_sub_pd(s_buffer[((j * 8) + 5)], q_0));
          s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_sub_pd(s_buffer[((j * 8) + 6)], q_0));
          s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_sub_pd(s_buffer[((j * 8) + 7)], q_0));
          _mm_store_pd((double*)sum + (j * 2), s_buffer[(j * 8)]);
        }
        RESET_DAZ_FLAG
        return;
      }
    }
  }
#else
  void zsumI2(int n, double complex* v, int incv, int fold, double complex* sum){
    long_double tmp_BLP;
    SET_DAZ_FLAG;
    switch(fold){
      case 3:{
        int i;

        double* sum_base = (double*) sum;
        double* v_base = (double*) v;
        double v_0, v_1, v_2, v_3;
        double q_0, q_1, q_2, q_3;
        double s_0_0, s_0_1, s_0_2, s_0_3;
        double s_1_0, s_1_1, s_1_2, s_1_3;
        double s_2_0, s_2_1, s_2_2, s_2_3;

        s_0_0 = s_0_2 = sum_base[0];
        s_0_1 = s_0_3 = sum_base[1];
        s_1_0 = s_1_2 = sum_base[2];
        s_1_1 = s_1_3 = sum_base[3];
        s_2_0 = s_2_2 = sum_base[4];
        s_2_1 = s_2_3 = sum_base[5];
        if(incv == 1){

          for(i = 0; i + 2 <= n; i += 2, v_base += 4){
            v_0 = v_base[0];
            v_1 = v_base[1];
            v_2 = v_base[2];
            v_3 = v_base[3];
            q_0 = s_0_0;
            q_1 = s_0_1;
            q_2 = s_0_2;
            q_3 = s_0_3;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_0_0 = s_0_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_0_1 = s_0_1 + tmp_BLP.d;
            tmp_BLP.d = v_2;
            tmp_BLP.l |= 1;
            s_0_2 = s_0_2 + tmp_BLP.d;
            tmp_BLP.d = v_3;
            tmp_BLP.l |= 1;
            s_0_3 = s_0_3 + tmp_BLP.d;
            q_0 = q_0 - s_0_0;
            q_1 = q_1 - s_0_1;
            q_2 = q_2 - s_0_2;
            q_3 = q_3 - s_0_3;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            v_2 = v_2 + q_2;
            v_3 = v_3 + q_3;
            q_0 = s_1_0;
            q_1 = s_1_1;
            q_2 = s_1_2;
            q_3 = s_1_3;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_1_0 = s_1_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_1_1 = s_1_1 + tmp_BLP.d;
            tmp_BLP.d = v_2;
            tmp_BLP.l |= 1;
            s_1_2 = s_1_2 + tmp_BLP.d;
            tmp_BLP.d = v_3;
            tmp_BLP.l |= 1;
            s_1_3 = s_1_3 + tmp_BLP.d;
            q_0 = q_0 - s_1_0;
            q_1 = q_1 - s_1_1;
            q_2 = q_2 - s_1_2;
            q_3 = q_3 - s_1_3;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            v_2 = v_2 + q_2;
            v_3 = v_3 + q_3;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_2_0 = s_2_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_2_1 = s_2_1 + tmp_BLP.d;
            tmp_BLP.d = v_2;
            tmp_BLP.l |= 1;
            s_2_2 = s_2_2 + tmp_BLP.d;
            tmp_BLP.d = v_3;
            tmp_BLP.l |= 1;
            s_2_3 = s_2_3 + tmp_BLP.d;
          }
          if(i + 1 <= n){
            v_0 = v_base[0];
            v_1 = v_base[1];
            q_0 = s_0_0;
            q_1 = s_0_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_0_0 = s_0_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_0_1 = s_0_1 + tmp_BLP.d;
            q_0 = q_0 - s_0_0;
            q_1 = q_1 - s_0_1;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            q_0 = s_1_0;
            q_1 = s_1_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_1_0 = s_1_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_1_1 = s_1_1 + tmp_BLP.d;
            q_0 = q_0 - s_1_0;
            q_1 = q_1 - s_1_1;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_2_0 = s_2_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_2_1 = s_2_1 + tmp_BLP.d;
            i += 1, v_base += 2;
          }
        }else{

          for(i = 0; i + 2 <= n; i += 2, v_base += (incv * 4)){
            v_0 = v_base[0];
            v_1 = v_base[1];
            v_2 = v_base[(incv * 2)];
            v_3 = v_base[((incv * 2) + 1)];
            q_0 = s_0_0;
            q_1 = s_0_1;
            q_2 = s_0_2;
            q_3 = s_0_3;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_0_0 = s_0_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_0_1 = s_0_1 + tmp_BLP.d;
            tmp_BLP.d = v_2;
            tmp_BLP.l |= 1;
            s_0_2 = s_0_2 + tmp_BLP.d;
            tmp_BLP.d = v_3;
            tmp_BLP.l |= 1;
            s_0_3 = s_0_3 + tmp_BLP.d;
            q_0 = q_0 - s_0_0;
            q_1 = q_1 - s_0_1;
            q_2 = q_2 - s_0_2;
            q_3 = q_3 - s_0_3;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            v_2 = v_2 + q_2;
            v_3 = v_3 + q_3;
            q_0 = s_1_0;
            q_1 = s_1_1;
            q_2 = s_1_2;
            q_3 = s_1_3;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_1_0 = s_1_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_1_1 = s_1_1 + tmp_BLP.d;
            tmp_BLP.d = v_2;
            tmp_BLP.l |= 1;
            s_1_2 = s_1_2 + tmp_BLP.d;
            tmp_BLP.d = v_3;
            tmp_BLP.l |= 1;
            s_1_3 = s_1_3 + tmp_BLP.d;
            q_0 = q_0 - s_1_0;
            q_1 = q_1 - s_1_1;
            q_2 = q_2 - s_1_2;
            q_3 = q_3 - s_1_3;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            v_2 = v_2 + q_2;
            v_3 = v_3 + q_3;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_2_0 = s_2_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_2_1 = s_2_1 + tmp_BLP.d;
            tmp_BLP.d = v_2;
            tmp_BLP.l |= 1;
            s_2_2 = s_2_2 + tmp_BLP.d;
            tmp_BLP.d = v_3;
            tmp_BLP.l |= 1;
            s_2_3 = s_2_3 + tmp_BLP.d;
          }
          if(i + 1 <= n){
            v_0 = v_base[0];
            v_1 = v_base[1];
            q_0 = s_0_0;
            q_1 = s_0_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_0_0 = s_0_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_0_1 = s_0_1 + tmp_BLP.d;
            q_0 = q_0 - s_0_0;
            q_1 = q_1 - s_0_1;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            q_0 = s_1_0;
            q_1 = s_1_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_1_0 = s_1_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_1_1 = s_1_1 + tmp_BLP.d;
            q_0 = q_0 - s_1_0;
            q_1 = q_1 - s_1_1;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_2_0 = s_2_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_2_1 = s_2_1 + tmp_BLP.d;
            i += 1, v_base += (incv * 2);
          }
        }
        q_0 = ((double*)sum)[0];
        s_0_0 = s_0_0 + (s_0_2 - q_0);
        q_0 = ((double*)sum)[1];
        s_0_1 = s_0_1 + (s_0_3 - q_0);
        ((double*)sum)[0] = s_0_0;
        ((double*)sum)[1] = s_0_1;
        q_0 = ((double*)sum)[2];
        s_1_0 = s_1_0 + (s_1_2 - q_0);
        q_0 = ((double*)sum)[3];
        s_1_1 = s_1_1 + (s_1_3 - q_0);
        ((double*)sum)[2] = s_1_0;
        ((double*)sum)[3] = s_1_1;
        q_0 = ((double*)sum)[4];
        s_2_0 = s_2_0 + (s_2_2 - q_0);
        q_0 = ((double*)sum)[5];
        s_2_1 = s_2_1 + (s_2_3 - q_0);
        ((double*)sum)[4] = s_2_0;
        ((double*)sum)[5] = s_2_1;
        RESET_DAZ_FLAG
        return;
      }
      default:{
        int i, j;

        double* sum_base = (double*) sum;
        double* v_base = (double*) v;
        double v_0, v_1, v_2, v_3;
        double q_0, q_1, q_2, q_3;
        double s_0, s_1, s_2, s_3;
        double s_buffer[(MAX_FOLD * 4)];

        for(j = 0; j < fold; j += 1){
          s_buffer[(j * 4)] = s_buffer[((j * 4) + 2)] = sum_base[(j * 2)];
          s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 3)] = sum_base[((j * 2) + 1)];
        }
        if(incv == 1){

          for(i = 0; i + 2 <= n; i += 2, v_base += 4){
            v_0 = v_base[0];
            v_1 = v_base[1];
            v_2 = v_base[2];
            v_3 = v_base[3];
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 4)];
              s_1 = s_buffer[((j * 4) + 1)];
              s_2 = s_buffer[((j * 4) + 2)];
              s_3 = s_buffer[((j * 4) + 3)];
              tmp_BLP.d = v_0;
              tmp_BLP.l |= 1;
              q_0 = s_0 + tmp_BLP.d;
              tmp_BLP.d = v_1;
              tmp_BLP.l |= 1;
              q_1 = s_1 + tmp_BLP.d;
              tmp_BLP.d = v_2;
              tmp_BLP.l |= 1;
              q_2 = s_2 + tmp_BLP.d;
              tmp_BLP.d = v_3;
              tmp_BLP.l |= 1;
              q_3 = s_3 + tmp_BLP.d;
              s_buffer[(j * 4)] = q_0;
              s_buffer[((j * 4) + 1)] = q_1;
              s_buffer[((j * 4) + 2)] = q_2;
              s_buffer[((j * 4) + 3)] = q_3;
              q_0 = s_0 - q_0;
              q_1 = s_1 - q_1;
              q_2 = s_2 - q_2;
              q_3 = s_3 - q_3;
              v_0 = v_0 + q_0;
              v_1 = v_1 + q_1;
              v_2 = v_2 + q_2;
              v_3 = v_3 + q_3;
            }
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_buffer[(j * 4)] = s_buffer[(j * 4)] + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + tmp_BLP.d;
            tmp_BLP.d = v_2;
            tmp_BLP.l |= 1;
            s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + tmp_BLP.d;
            tmp_BLP.d = v_3;
            tmp_BLP.l |= 1;
            s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + tmp_BLP.d;
          }
          if(i + 1 <= n){
            v_0 = v_base[0];
            v_1 = v_base[1];
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 4)];
              s_1 = s_buffer[((j * 4) + 1)];
              tmp_BLP.d = v_0;
              tmp_BLP.l |= 1;
              q_0 = s_0 + tmp_BLP.d;
              tmp_BLP.d = v_1;
              tmp_BLP.l |= 1;
              q_1 = s_1 + tmp_BLP.d;
              s_buffer[(j * 4)] = q_0;
              s_buffer[((j * 4) + 1)] = q_1;
              q_0 = s_0 - q_0;
              q_1 = s_1 - q_1;
              v_0 = v_0 + q_0;
              v_1 = v_1 + q_1;
            }
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_buffer[(j * 4)] = s_buffer[(j * 4)] + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + tmp_BLP.d;
            i += 1, v_base += 2;
          }
        }else{

          for(i = 0; i + 2 <= n; i += 2, v_base += (incv * 4)){
            v_0 = v_base[0];
            v_1 = v_base[1];
            v_2 = v_base[(incv * 2)];
            v_3 = v_base[((incv * 2) + 1)];
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 4)];
              s_1 = s_buffer[((j * 4) + 1)];
              s_2 = s_buffer[((j * 4) + 2)];
              s_3 = s_buffer[((j * 4) + 3)];
              tmp_BLP.d = v_0;
              tmp_BLP.l |= 1;
              q_0 = s_0 + tmp_BLP.d;
              tmp_BLP.d = v_1;
              tmp_BLP.l |= 1;
              q_1 = s_1 + tmp_BLP.d;
              tmp_BLP.d = v_2;
              tmp_BLP.l |= 1;
              q_2 = s_2 + tmp_BLP.d;
              tmp_BLP.d = v_3;
              tmp_BLP.l |= 1;
              q_3 = s_3 + tmp_BLP.d;
              s_buffer[(j * 4)] = q_0;
              s_buffer[((j * 4) + 1)] = q_1;
              s_buffer[((j * 4) + 2)] = q_2;
              s_buffer[((j * 4) + 3)] = q_3;
              q_0 = s_0 - q_0;
              q_1 = s_1 - q_1;
              q_2 = s_2 - q_2;
              q_3 = s_3 - q_3;
              v_0 = v_0 + q_0;
              v_1 = v_1 + q_1;
              v_2 = v_2 + q_2;
              v_3 = v_3 + q_3;
            }
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_buffer[(j * 4)] = s_buffer[(j * 4)] + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + tmp_BLP.d;
            tmp_BLP.d = v_2;
            tmp_BLP.l |= 1;
            s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + tmp_BLP.d;
            tmp_BLP.d = v_3;
            tmp_BLP.l |= 1;
            s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + tmp_BLP.d;
          }
          if(i + 1 <= n){
            v_0 = v_base[0];
            v_1 = v_base[1];
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 4)];
              s_1 = s_buffer[((j * 4) + 1)];
              tmp_BLP.d = v_0;
              tmp_BLP.l |= 1;
              q_0 = s_0 + tmp_BLP.d;
              tmp_BLP.d = v_1;
              tmp_BLP.l |= 1;
              q_1 = s_1 + tmp_BLP.d;
              s_buffer[(j * 4)] = q_0;
              s_buffer[((j * 4) + 1)] = q_1;
              q_0 = s_0 - q_0;
              q_1 = s_1 - q_1;
              v_0 = v_0 + q_0;
              v_1 = v_1 + q_1;
            }
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_buffer[(j * 4)] = s_buffer[(j * 4)] + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + tmp_BLP.d;
            i += 1, v_base += (incv * 2);
          }
        }
        for(j = 0; j < fold; j += 1){
          q_0 = ((double*)sum)[(j * 2)];
          s_buffer[(j * 4)] = s_buffer[(j * 4)] + (s_buffer[((j * 4) + 2)] - q_0);
          q_0 = ((double*)sum)[((j * 2) + 1)];
          s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + (s_buffer[((j * 4) + 3)] - q_0);
          ((double*)sum)[(j * 2)] = s_buffer[(j * 4)];
          ((double*)sum)[((j * 2) + 1)] = s_buffer[((j * 4) + 1)];
        }
        RESET_DAZ_FLAG
        return;
      }
    }
  }
#endif