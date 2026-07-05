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

// SSE4.1 port of the AVX2 relative-entropy matrix-adjustment code.
// Requires: SSE4.1 (_mm_floor_ps, _mm_blendv_ps). Everything else is SSE2/SSE3.
// Compile with e.g. -msse4.1 (SSSE3 is implied and not otherwise needed).
//
// Differences vs. the AVX2 original:
//   * 128-bit lanes: rows are processed as 6 chunks of 4 (NP = 24) instead of
//     3 chunks of 8. Since N = 20 is a multiple of 4, the relative-entropy
//     reduction needs no tail mask at all — the padded columns 20..23 are
//     simply skipped (the AVX2 tail mask excluded them anyway).
//   * FMA is an AVX2-era extension, so all fused multiply-adds are emulated
//     with separate mul+add. Results may differ from the AVX2 build in the
//     last ULP or so due to the extra rounding step.
//   * Loads from caller-provided pointers (q, P, Q, x) use unaligned loads,
//     so no 32-byte alignment contract is imposed on the caller.

constexpr int N = 20, NP = 24;        // logical rows, padded column stride

#include <smmintrin.h>                // SSE4.1 (pulls in SSSE3/SSE3/SSE2)
#include <cmath>
#include <limits>
#include <algorithm>

// ---- FMA emulation ---------------------------------------------------------
static inline __m128 fmadd128(__m128 a, __m128 b, __m128 c) {   // a*b + c
    return _mm_add_ps(_mm_mul_ps(a, b), c);
}
static inline __m128 fnmadd128(__m128 a, __m128 b, __m128 c) {  // c - a*b
    return _mm_sub_ps(c, _mm_mul_ps(a, b));
}

// ---- vectorized exp / log (Cephes-style, same coefficients as the AVX2 code)
static inline __m128 exp128_ps(__m128 x) {
    const __m128 hi    = _mm_set1_ps(88.3762626647950f);
    const __m128 lo    = _mm_set1_ps(-87.3365478515625f);
    const __m128 LOG2E = _mm_set1_ps(1.44269504088896341f);
    const __m128 C1    = _mm_set1_ps(0.693359375f);
    const __m128 C2    = _mm_set1_ps(-2.12194440e-4f);
    const __m128 half  = _mm_set1_ps(0.5f);
    const __m128 one   = _mm_set1_ps(1.0f);

    x = _mm_min_ps(_mm_max_ps(x, lo), hi);

    __m128 fx = _mm_floor_ps(fmadd128(x, LOG2E, half));      // SSE4.1
    x = fnmadd128(fx, C1, x);
    x = fnmadd128(fx, C2, x);

    __m128 y = _mm_set1_ps(1.9875691500e-4f);
    y = fmadd128(y, x, _mm_set1_ps(1.3981999507e-3f));
    y = fmadd128(y, x, _mm_set1_ps(8.3334519073e-3f));
    y = fmadd128(y, x, _mm_set1_ps(4.1665795894e-2f));
    y = fmadd128(y, x, _mm_set1_ps(1.6666665459e-1f));
    y = fmadd128(y, x, _mm_set1_ps(5.0000001201e-1f));
    __m128 x2 = _mm_mul_ps(x, x);
    y = fmadd128(y, x2, x);
    y = _mm_add_ps(y, one);

    __m128i e = _mm_cvttps_epi32(fx);
    e = _mm_slli_epi32(_mm_add_epi32(e, _mm_set1_epi32(127)), 23);
    return _mm_mul_ps(y, _mm_castsi128_ps(e));
}

static inline __m128 log128_ps(__m128 x) {
    const __m128 one = _mm_set1_ps(1.0f);
    __m128 invalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());

    x = _mm_max_ps(x, _mm_castsi128_ps(_mm_set1_epi32(0x00800000)));
    __m128i imm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);

    x = _mm_and_ps(x, _mm_castsi128_ps(_mm_set1_epi32(~0x7f800000)));
    x = _mm_or_ps(x, _mm_set1_ps(0.5f));

    imm0 = _mm_sub_epi32(imm0, _mm_set1_epi32(0x7f));
    __m128 e = _mm_add_ps(_mm_cvtepi32_ps(imm0), one);

    __m128 mask = _mm_cmplt_ps(x, _mm_set1_ps(0.707106781186547524f));
    __m128 tmp = _mm_and_ps(x, mask);
    x = _mm_sub_ps(x, one);
    e = _mm_sub_ps(e, _mm_and_ps(one, mask));
    x = _mm_add_ps(x, tmp);

    __m128 z = _mm_mul_ps(x, x);
    __m128 y = _mm_set1_ps(7.0376836292E-2f);
    y = fmadd128(y, x, _mm_set1_ps(-1.1514610310E-1f));
    y = fmadd128(y, x, _mm_set1_ps(1.1676998740E-1f));
    y = fmadd128(y, x, _mm_set1_ps(-1.2420140846E-1f));
    y = fmadd128(y, x, _mm_set1_ps(1.4249322787E-1f));
    y = fmadd128(y, x, _mm_set1_ps(-1.6668057665E-1f));
    y = fmadd128(y, x, _mm_set1_ps(2.0000714765E-1f));
    y = fmadd128(y, x, _mm_set1_ps(-2.4999993993E-1f));
    y = fmadd128(y, x, _mm_set1_ps(3.3333331174E-1f));
    y = _mm_mul_ps(y, x);
    y = _mm_mul_ps(y, z);

    y = fmadd128(e, _mm_set1_ps(-2.12194440e-4f), y);
    y = fnmadd128(z, _mm_set1_ps(0.5f), y);
    x = _mm_add_ps(x, y);
    x = fmadd128(e, _mm_set1_ps(0.693359375f), x);
    x = _mm_or_ps(x, invalid_mask);                 // NaN for x <= 0
    return x;
}

// ---- horizontal reductions --------------------------------------------------
static inline float hsum128(__m128 v) {
    __m128 sh = _mm_movehl_ps(v, v);                // [2,3,2,3]
    v  = _mm_add_ps(v, sh);                         // [0+2, 1+3, ..]
    sh = _mm_shuffle_ps(v, v, 0x1);
    v  = _mm_add_ss(v, sh);
    return _mm_cvtss_f32(v);
}

static inline float hmax128(__m128 v) {
    __m128 sh = _mm_movehl_ps(v, v);
    v  = _mm_max_ps(v, sh);
    sh = _mm_shuffle_ps(v, v, 0x1);
    v  = _mm_max_ss(v, sh);
    return _mm_cvtss_f32(v);
}

// ---- relative entropy sum(x * (log x - lnPQ)) over x > 0 --------------------
// With 4-wide vectors, N = 20 columns split into exactly 5 full chunks, so no
// column mask is needed (padding columns 20..23 are never touched here).
static inline __m128 chunk(const float* xp, const float* lp,
    __m128 zero, __m128 one, __m128 acc) {
    __m128 xv = _mm_loadu_ps(xp);
    __m128 lv = _mm_loadu_ps(lp);

    __m128 mask = _mm_cmpgt_ps(xv, zero);

    __m128 xsafe = _mm_blendv_ps(one, xv, mask);    // SSE4.1
    __m128 lx = log128_ps(xsafe);

    __m128 term = _mm_mul_ps(xv, _mm_sub_ps(lx, lv));
    term = _mm_and_ps(term, mask);

    return _mm_add_ps(acc, term);
}

static float relative_entropy_sse(const float* __restrict x, const float* __restrict lnPQ) {
    const __m128 zero = _mm_setzero_ps();
    const __m128 one = _mm_set1_ps(1.0f);

    __m128 a0 = zero, a1 = zero, a2 = zero, a3 = zero, a4 = zero;

    for (int i = 0; i < N; ++i) {
        const float* xr = x + i * NP;
        const float* lr = lnPQ + i * NP;
        a0 = chunk(xr + 0,  lr + 0,  zero, one, a0);
        a1 = chunk(xr + 4,  lr + 4,  zero, one, a1);
        a2 = chunk(xr + 8,  lr + 8,  zero, one, a2);
        a3 = chunk(xr + 12, lr + 12, zero, one, a3);
        a4 = chunk(xr + 16, lr + 16, zero, one, a4);
    }

    __m128 s = _mm_add_ps(_mm_add_ps(a0, a1), _mm_add_ps(a2, a3));
    return hsum128(_mm_add_ps(s, a4));
}

// ---- RAS (iterative proportional fitting) -----------------------------------
static inline void finalize_block(__m128 ssum, const float* Qp, float* bp,
    __m128& residv) {
    const __m128 zero = _mm_setzero_ps();
    const __m128 m = _mm_cmpgt_ps(ssum, zero);
    const __m128 recip = _mm_div_ps(_mm_loadu_ps(Qp), ssum);
    const __m128 bj = _mm_blendv_ps(zero, recip, m);
    const __m128 bold = _mm_loadu_ps(bp);
    const __m128 adiff = _mm_andnot_ps(_mm_set1_ps(-0.0f), _mm_sub_ps(bj, bold));
    residv = _mm_max_ps(residv, adiff);
    _mm_storeu_ps(bp, bj);
}

static void ras_balance(const float* k, const float* P, const float* Q,
    float* a, float* b, float* x,
    int max_iters = 100, float tol = 1e-7f) {
    for (int it = 0; it < max_iters; ++it) {
        // row scaling: a[i] = P[i] / sum_j k[i][j] * b[j]
        for (int i = 0; i < N; ++i) {
            const float* ki = k + i * NP;
            __m128 acc = _mm_mul_ps(_mm_load_ps(ki), _mm_loadu_ps(b));
            acc = fmadd128(_mm_load_ps(ki + 4),  _mm_loadu_ps(b + 4),  acc);
            acc = fmadd128(_mm_load_ps(ki + 8),  _mm_loadu_ps(b + 8),  acc);
            acc = fmadd128(_mm_load_ps(ki + 12), _mm_loadu_ps(b + 12), acc);
            acc = fmadd128(_mm_load_ps(ki + 16), _mm_loadu_ps(b + 16), acc);
            acc = fmadd128(_mm_load_ps(ki + 20), _mm_loadu_ps(b + 20), acc);
            const float s = hsum128(acc);
            a[i] = s > 0.0f ? P[i] / s : 0.0f;
        }

        // column sums: s[j] = sum_i a[i] * k[i][j]
        __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps(), s2 = _mm_setzero_ps();
        __m128 s3 = _mm_setzero_ps(), s4 = _mm_setzero_ps(), s5 = _mm_setzero_ps();
        for (int i = 0; i < N; ++i) {
            const float* ki = k + i * NP;
            const __m128 ai = _mm_set1_ps(a[i]);
            s0 = fmadd128(_mm_load_ps(ki),      ai, s0);
            s1 = fmadd128(_mm_load_ps(ki + 4),  ai, s1);
            s2 = fmadd128(_mm_load_ps(ki + 8),  ai, s2);
            s3 = fmadd128(_mm_load_ps(ki + 12), ai, s3);
            s4 = fmadd128(_mm_load_ps(ki + 16), ai, s4);
            s5 = fmadd128(_mm_load_ps(ki + 20), ai, s5);
        }

        __m128 residv = _mm_setzero_ps();
        finalize_block(s0, Q,      b,      residv);
        finalize_block(s1, Q + 4,  b + 4,  residv);
        finalize_block(s2, Q + 8,  b + 8,  residv);
        finalize_block(s3, Q + 12, b + 12, residv);
        finalize_block(s4, Q + 16, b + 16, residv);
        finalize_block(s5, Q + 20, b + 20, residv);

        if (hmax128(residv) < tol) break;
    }

    // x[i][j] = a[i] * k[i][j] * b[j]
    for (int i = 0; i < N; ++i) {
        const float* ki = k + i * NP;
        float* xi = x + i * NP;
        const __m128 ai = _mm_set1_ps(a[i]);
        for (int j = 0; j < NP; j += 4) {
            const __m128 v = _mm_mul_ps(_mm_mul_ps(_mm_load_ps(ki + j), ai),
                                        _mm_loadu_ps(b + j));
            _mm_storeu_ps(xi + j, v);
        }
    }
}

// ---- top-level: find theta so that RE(x(theta)) == target_re ----------------
bool matrix_adjust(const float* q, const float* P, const float* Q, float* x, float target_re) {
    alignas(16) float k[N * NP];
    alignas(16) float L[N * NP], lnPQ[N * NP];

    for (int i = 0; i < N; ++i) {
        const __m128 Pi = _mm_set1_ps(P[i]);
        const int    row = i * NP;
        for (int j = 0; j < N; j += 4) {
            const __m128 pq = _mm_mul_ps(Pi, _mm_loadu_ps(Q + j));
            _mm_store_ps(lnPQ + row + j, log128_ps(pq));
            const __m128 qv = _mm_loadu_ps(q + row + j);
            _mm_store_ps(L + row + j, log128_ps(_mm_div_ps(qv, pq)));
        }
        std::fill(L + row + N, L + row + NP, 0.0f);
        std::fill(lnPQ + row + N, lnPQ + row + NP, 0.0f);
    }

    float a[NP], b[NP];
    for (int i = 0; i < N; ++i) { a[i] = 1.0f; b[i] = 1.0f; }
    for (int i = N; i < NP; ++i) { a[i] = 0.0f; b[i] = 0.0f; }

    auto re_at = [&](float theta) -> float {
        const __m128 vtheta = _mm_set1_ps(theta);
        __m128 vmax = _mm_set1_ps(std::numeric_limits<float>::lowest());
        for (int t = 0; t < N * NP; t += 4) {
            const __m128 e = fmadd128(vtheta, _mm_load_ps(L + t), _mm_load_ps(lnPQ + t));
            _mm_store_ps(k + t, e);
            vmax = _mm_max_ps(vmax, e);
        }
        const float e_max = hmax128(vmax);

        const __m128 vemax = _mm_set1_ps(e_max);
        for (int t = 0; t < N * NP; t += 4) {
            __m128 v = _mm_load_ps(&k[t]);
            v = exp128_ps(_mm_sub_ps(v, vemax));
            _mm_store_ps(&k[t], v);
        }

        ras_balance(k, P, Q, a, b, x);
        return relative_entropy_sse(x, lnPQ);
    };

    const float TMIN = 0.05f, TMAX = 21.0f, RE_TOL = 1e-3f;
    float lo, hi, flo, fhi;
    const float f1 = re_at(1.0f) - target_re;
    if (std::fabs(f1) < RE_TOL) return true;

    if (f1 < 0.0f) {
        lo = 1.0f; flo = f1; hi = 0.0f; fhi = 0.0f;
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
        lo = TMIN; hi = 1.0f; fhi = f1;
    }

    // Illinois-style regula falsi
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