/****
DIAMOND protein sequence aligner
Copyright (C) 2012-2026 Benjamin J. Buchfink

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
****/
// SPDX-License-Identifier: GPL-3.0-or-later

namespace Stats { namespace DISPATCH_ARCH {

constexpr int N = 20, NP = 24;        // logical rows, padded column stride

#include <immintrin.h>
#include <cmath>

static inline __m256 exp256_ps(__m256 x) {
    const __m256 hi = _mm256_set1_ps(88.3762626647950f);
    const __m256 lo = _mm256_set1_ps(-87.3365478515625f);
    const __m256 LOG2E = _mm256_set1_ps(1.44269504088896341f);
    const __m256 C1 = _mm256_set1_ps(0.693359375f);
    const __m256 C2 = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);

    x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);

    __m256 fx = _mm256_floor_ps(_mm256_fmadd_ps(x, LOG2E, half));
    x = _mm256_fnmadd_ps(fx, C1, x);
    x = _mm256_fnmadd_ps(fx, C2, x);

    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    __m256 x2 = _mm256_mul_ps(x, x);
    y = _mm256_fmadd_ps(y, x2, x);
    y = _mm256_add_ps(y, one);

    __m256i e = _mm256_cvttps_epi32(fx);
    e = _mm256_slli_epi32(_mm256_add_epi32(e, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

static inline __m256 log256_ps(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

    x = _mm256_max_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x00800000)));
    __m256i imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

    x = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(~0x7f800000)));
    x = _mm256_or_ps(x, _mm256_set1_ps(0.5f));

    imm0 = _mm256_sub_epi32(imm0, _mm256_set1_epi32(0x7f));
    __m256 e = _mm256_add_ps(_mm256_cvtepi32_ps(imm0), one);

    __m256 mask = _mm256_cmp_ps(x, _mm256_set1_ps(0.707106781186547524f), _CMP_LT_OS);
    __m256 tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);

    __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(7.0376836292E-2f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-1.1514610310E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.1676998740E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-1.2420140846E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.4249322787E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-1.6668057665E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(2.0000714765E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-2.4999993993E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(3.3333331174E-1f));
    y = _mm256_mul_ps(y, x);
    y = _mm256_mul_ps(y, z);

    y = _mm256_fmadd_ps(e, _mm256_set1_ps(-2.12194440e-4f), y);
    y = _mm256_fnmadd_ps(z, _mm256_set1_ps(0.5f), y);
    x = _mm256_add_ps(x, y);
    x = _mm256_fmadd_ps(e, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_or_ps(x, invalid_mask);
    return x;
}

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehl_ps(lo, lo);
    lo = _mm_add_ps(lo, sh);
    sh = _mm_shuffle_ps(lo, lo, 0x1);
    lo = _mm_add_ss(lo, sh);
    return _mm_cvtss_f32(lo);
}

static inline float hmax256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    __m128 sh = _mm_movehl_ps(lo, lo);
    lo = _mm_max_ps(lo, sh);
    sh = _mm_shuffle_ps(lo, lo, 0x1);
    lo = _mm_max_ss(lo, sh);
    return _mm_cvtss_f32(lo);
}

static inline __m256 chunk(const float* xp, const float* lp,
    __m256 colmask, __m256 zero, __m256 one,
    __m256 acc) {
    __m256 xv = _mm256_loadu_ps(xp);
    __m256 lv = _mm256_loadu_ps(lp);
    
    __m256 pos = _mm256_cmp_ps(xv, zero, _CMP_GT_OQ);
    __m256 mask = _mm256_and_ps(pos, colmask);

    __m256 xsafe = _mm256_blendv_ps(one, xv, mask);
    __m256 lx = log256_ps(xsafe);

    __m256 term = _mm256_mul_ps(xv, _mm256_sub_ps(lx, lv));
    term = _mm256_and_ps(term, mask);

    return _mm256_add_ps(acc, term);
}

static float relative_entropy_avx2(const float* __restrict x, const float* __restrict lnPQ) {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 full = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
    const __m256 tail = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0));

    __m256 a0 = zero, a1 = zero, a2 = zero;

    for (int i = 0; i < N; ++i) {
        const float* xr = x + i * NP;
        const float* lr = lnPQ + i * NP;
        a0 = chunk(xr + 0, lr + 0, full, zero, one, a0);
        a1 = chunk(xr + 8, lr + 8, full, zero, one, a1);
        a2 = chunk(xr + 16, lr + 16, tail, zero, one, a2);
    }

    return hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), a2));
}

static inline void finalize_block(__m256 ssum, const float* Qp, float* bp,
    __m256& residv) {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 m = _mm256_cmp_ps(ssum, zero, _CMP_GT_OQ);
    const __m256 recip = _mm256_div_ps(_mm256_loadu_ps(Qp), ssum);
    const __m256 bj = _mm256_blendv_ps(zero, recip, m);
    const __m256 bold = _mm256_loadu_ps(bp);
    const __m256 adiff = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_sub_ps(bj, bold));
    residv = _mm256_max_ps(residv, adiff);
    _mm256_storeu_ps(bp, bj);
}

static void ras_balance(const float* k, const float* P, const float* Q,
    float* a, float* b, float* x,
    int max_iters = 100, float tol = 1e-7f) {
    for (int it = 0; it < max_iters; ++it) {        
        for (int i = 0; i < N; ++i) {
            const float* ki = k + i * NP;
            __m256 acc = _mm256_mul_ps(_mm256_loadu_ps(ki), _mm256_loadu_ps(b));
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(ki + 8), _mm256_loadu_ps(b + 8), acc);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(ki + 16), _mm256_loadu_ps(b + 16), acc);
            const float s = hsum256(acc);
            a[i] = s > 0.0f ? P[i] / s : 0.0f;
        }

        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps();
        for (int i = 0; i < N; ++i) {
            const float* ki = k + i * NP;
            const __m256 ai = _mm256_set1_ps(a[i]);
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(ki), ai, s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(ki + 8), ai, s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(ki + 16), ai, s2);
        }
        __m256 residv = _mm256_setzero_ps();
        finalize_block(s0, Q, b, residv);
        finalize_block(s1, Q + 8, b + 8, residv);
        finalize_block(s2, Q + 16, b + 16, residv);

        if (hmax256(residv) < tol) break;
    }

    for (int i = 0; i < N; ++i) {
        const float* ki = k + i * NP;
        float* xi = x + i * NP;
        const __m256 ai = _mm256_set1_ps(a[i]);
        _mm256_storeu_ps(xi, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(ki), ai), _mm256_loadu_ps(b)));
        _mm256_storeu_ps(xi + 8, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(ki + 8), ai), _mm256_loadu_ps(b + 8)));
        _mm256_storeu_ps(xi + 16, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(ki + 16), ai), _mm256_loadu_ps(b + 16)));
    }
}

bool matrix_adjust(const float* q, const float* P, const float* Q, float* x, float target_re) {
    alignas(32) float k[N * NP];
    alignas(32) float L[N * NP], lnPQ[N * NP];

    for (int i = 0; i < N; ++i) {
        const __m256 Pi = _mm256_set1_ps(P[i]);
        const int    row = i * NP;
        for (int j = 0; j < N; j += 8) {
            const __m256 pq = _mm256_mul_ps(Pi, _mm256_load_ps(Q + j));
            _mm256_store_ps(lnPQ + row + j, log256_ps(pq));
            const __m256 qv = _mm256_loadu_ps(q + row + j);
            _mm256_store_ps(L + row + j, log256_ps(_mm256_div_ps(qv, pq)));
        }
        std::fill(L + row + N, L + row + NP, 0.0f);
        std::fill(lnPQ + row + N, lnPQ + row + NP, 0.0f);
    }

    float a[NP], b[NP];
    for (int i = 0; i < N; ++i) { a[i] = 1.0; b[i] = 1.0; }
    for (int i = N; i < NP; ++i) { a[i] = 0.0; b[i] = 0.0; }

    auto re_at = [&](float theta) -> float {
        const __m256 vtheta = _mm256_set1_ps(theta);
        __m256 vmax = _mm256_set1_ps(std::numeric_limits<float>::lowest());
        for (int i = 0; i < N; ++i) {
            const int b = i * NP;
            __m256 e0 = _mm256_fmadd_ps(vtheta, _mm256_load_ps(L + b), _mm256_load_ps(lnPQ + b));
            _mm256_store_ps(k + b, e0);
            vmax = _mm256_max_ps(vmax, e0);

            __m256 e1 = _mm256_fmadd_ps(vtheta, _mm256_load_ps(L + b + 8), _mm256_load_ps(lnPQ + b + 8));
            _mm256_store_ps(k + b + 8, e1);
            vmax = _mm256_max_ps(vmax, e1);

            __m256 e2 = _mm256_fmadd_ps(vtheta, _mm256_load_ps(L + b + 16), _mm256_load_ps(lnPQ + b + 16));
            _mm256_store_ps(k + b + 16, e2);
            vmax = _mm256_max_ps(vmax, e2);
        }
        float e_max = hmax256(vmax);

        const __m256 vemax = _mm256_set1_ps(e_max);
        for (int t = 0; t < N * NP; t += 8) {
            __m256 v = _mm256_load_ps(&k[t]);
            v = exp256_ps(_mm256_sub_ps(v, vemax));
            _mm256_store_ps(&k[t], v);
        }

        ras_balance(k, P, Q, a, b, x);
        return relative_entropy_avx2(x, lnPQ);

        };

    const float TMIN = 0.05f, TMAX = 21.0f, RE_TOL = 1e-3f;
    float lo, hi, flo, fhi;
    const float f1 = re_at(1.0) - target_re;
    if (std::fabs(f1) < RE_TOL) return true;

    if (f1 < 0.0f) {        
        lo = 1.0; flo = f1; hi = 0.0f; fhi = 0.0f;
        bool bracketed = false;
        for (float t = 1.0f; t < TMAX; ) {
            t = std::min(t * 2.0f, TMAX);
            const float ft = re_at(t) - target_re;
            if (ft >= 0.0f) { hi = t; fhi = ft; bracketed = true; break; }
            lo = t; flo = ft;
        }
        if (!bracketed) return false;
    }
    else {        
        flo = re_at(TMIN) - target_re;
        if (flo >= 0.0f) return true;
        lo = TMIN; hi = 1.0; fhi = f1;
    }

    for (int it = 0; it < 30; ++it) {
        const float theta = lo - flo * (hi - lo) / (fhi - flo);
        const float f = re_at(theta) - target_re;
        if (std::fabs(f) < RE_TOL) break;
        if (f > 0.0f) { hi = theta; fhi = f; flo *= 0.5f; }
        else { lo = theta; flo = f; fhi *= 0.5f; }
    }
	return true;
}

}}