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

#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>

// Baseline ARMv8.0-A AArch64 NEON only (no SVE, no v8.2+ extensions).
// Everything below (vfmaq/vfmsq, vrndmq, vdivq, vaddvq/vmaxvq) is mandatory
// in the base A64 ISA, so this runs on anything from Cortex-A53 up.

// ---- cephes-style exp, 4 lanes ---------------------------------------------
static inline float32x4_t exp_ps(float32x4_t x) {
    const float32x4_t hi    = vdupq_n_f32(88.3762626647950f);
    const float32x4_t lo    = vdupq_n_f32(-87.3365478515625f);
    const float32x4_t LOG2E = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t C1    = vdupq_n_f32(0.693359375f);
    const float32x4_t C2    = vdupq_n_f32(-2.12194440e-4f);
    const float32x4_t half  = vdupq_n_f32(0.5f);
    const float32x4_t one   = vdupq_n_f32(1.0f);

    x = vminq_f32(vmaxq_f32(x, lo), hi);

    // fx = floor(x * log2(e) + 0.5)
    float32x4_t fx = vrndmq_f32(vfmaq_f32(half, x, LOG2E));
    x = vfmsq_f32(x, fx, C1);              // x -= fx * C1
    x = vfmsq_f32(x, fx, C2);              // x -= fx * C2

    float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
    y = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), y, x);
    y = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), y, x);
    y = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), y, x);
    y = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), y, x);
    y = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), y, x);
    const float32x4_t x2 = vmulq_f32(x, x);
    y = vfmaq_f32(x, y, x2);               // y = y*x^2 + x
    y = vaddq_f32(y, one);

    // 2^fx via exponent bits (fx already integral, cvt truncation is exact)
    int32x4_t e = vcvtq_s32_f32(fx);
    e = vshlq_n_s32(vaddq_s32(e, vdupq_n_s32(127)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(e));
}

// ---- cephes-style log, 4 lanes ----------------------------------------------
static inline float32x4_t log_ps(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const uint32x4_t invalid_mask = vcleq_f32(x, vdupq_n_f32(0.0f));

    x = vmaxq_f32(x, vreinterpretq_f32_u32(vdupq_n_u32(0x00800000u))); // clamp to smallest normal

    int32x4_t imm0 = vreinterpretq_s32_u32(
        vshrq_n_u32(vreinterpretq_u32_f32(x), 23));

    // keep mantissa, force exponent to 0.5
    x = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x),
                                        vdupq_n_u32(~0x7f800000u)));
    x = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x),
                                        vreinterpretq_u32_f32(vdupq_n_f32(0.5f))));

    imm0 = vsubq_s32(imm0, vdupq_n_s32(0x7f));
    float32x4_t e = vaddq_f32(vcvtq_f32_s32(imm0), one);

    const uint32x4_t mask = vcltq_f32(x, vdupq_n_f32(0.707106781186547524f));
    const float32x4_t tmp = vreinterpretq_f32_u32(
        vandq_u32(vreinterpretq_u32_f32(x), mask));
    x = vsubq_f32(x, one);
    e = vsubq_f32(e, vreinterpretq_f32_u32(
        vandq_u32(vreinterpretq_u32_f32(one), mask)));
    x = vaddq_f32(x, tmp);

    const float32x4_t z = vmulq_f32(x, x);
    float32x4_t y = vdupq_n_f32(7.0376836292e-2f);
    y = vfmaq_f32(vdupq_n_f32(-1.1514610310e-1f), y, x);
    y = vfmaq_f32(vdupq_n_f32( 1.1676998740e-1f), y, x);
    y = vfmaq_f32(vdupq_n_f32(-1.2420140846e-1f), y, x);
    y = vfmaq_f32(vdupq_n_f32( 1.4249322787e-1f), y, x);
    y = vfmaq_f32(vdupq_n_f32(-1.6668057665e-1f), y, x);
    y = vfmaq_f32(vdupq_n_f32( 2.0000714765e-1f), y, x);
    y = vfmaq_f32(vdupq_n_f32(-2.4999993993e-1f), y, x);
    y = vfmaq_f32(vdupq_n_f32( 3.3333331174e-1f), y, x);
    y = vmulq_f32(y, x);
    y = vmulq_f32(y, z);

    y = vfmaq_f32(y, e, vdupq_n_f32(-2.12194440e-4f));
    y = vfmsq_f32(y, z, vdupq_n_f32(0.5f));
    x = vaddq_f32(x, y);
    x = vfmaq_f32(x, e, vdupq_n_f32(0.693359375f));
    // poison lanes with x <= 0 (all-ones bit pattern = NaN), like the SSE original
    x = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), invalid_mask));
    return x;
}

// ---- horizontal reductions: single instruction on AArch64 -------------------
static inline float hsum4(float32x4_t v) { return vaddvq_f32(v); }
static inline float hmax4(float32x4_t v) { return vmaxvq_f32(v); }

// ---- KL contribution of 4 columns -------------------------------------------
// N (=20) is a multiple of the 4-lane vector width, so no tail/column mask is
// needed: lanes 20..23 of each padded row are never touched by this kernel.
static inline float32x4_t chunk(const float* xp, const float* lp, float32x4_t acc) {
    const float32x4_t xv = vld1q_f32(xp);
    const float32x4_t lv = vld1q_f32(lp);

    const uint32x4_t mask = vcgtq_f32(xv, vdupq_n_f32(0.0f));

    const float32x4_t xsafe = vbslq_f32(mask, xv, vdupq_n_f32(1.0f));
    const float32x4_t lx = log_ps(xsafe);

    float32x4_t term = vmulq_f32(xv, vsubq_f32(lx, lv));
    term = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(term), mask));

    return vaddq_f32(acc, term);
}

static float relative_entropy_neon(const float* __restrict x, const float* __restrict lnPQ) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0, a4 = a0;

    for (int i = 0; i < N; ++i) {
        const float* xr = x + i * NP;
        const float* lr = lnPQ + i * NP;
        a0 = chunk(xr +  0, lr +  0, a0);
        a1 = chunk(xr +  4, lr +  4, a1);
        a2 = chunk(xr +  8, lr +  8, a2);
        a3 = chunk(xr + 12, lr + 12, a3);
        a4 = chunk(xr + 16, lr + 16, a4);
    }

    const float32x4_t s = vaddq_f32(vaddq_f32(vaddq_f32(a0, a1),
                                              vaddq_f32(a2, a3)), a4);
    return hsum4(s);
}

// ---- column-scaling update for 4 columns ------------------------------------
static inline void finalize_block(float32x4_t ssum, const float* Qp, float* bp,
    float32x4_t& residv) {
    const uint32x4_t m = vcgtq_f32(ssum, vdupq_n_f32(0.0f));
    const float32x4_t recip = vdivq_f32(vld1q_f32(Qp), ssum);   // junk lanes discarded by bsl
    const float32x4_t bj = vbslq_f32(m, recip, vdupq_n_f32(0.0f));
    const float32x4_t bold = vld1q_f32(bp);
    residv = vmaxq_f32(residv, vabdq_f32(bj, bold));             // |bj - bold|
    vst1q_f32(bp, bj);
}

static void ras_balance(const float* k, const float* P, const float* Q,
    float* a, float* b, float* x,
    int max_iters = 100, float tol = 1e-7f) {
    for (int it = 0; it < max_iters; ++it) {
        // row scaling: a[i] = P[i] / (k[i,:] . b)
        for (int i = 0; i < N; ++i) {
            const float* ki = k + i * NP;
            float32x4_t acc0 = vmulq_f32(vld1q_f32(ki),      vld1q_f32(b));
            float32x4_t acc1 = vmulq_f32(vld1q_f32(ki +  4), vld1q_f32(b +  4));
            acc0 = vfmaq_f32(acc0, vld1q_f32(ki +  8), vld1q_f32(b +  8));
            acc1 = vfmaq_f32(acc1, vld1q_f32(ki + 12), vld1q_f32(b + 12));
            acc0 = vfmaq_f32(acc0, vld1q_f32(ki + 16), vld1q_f32(b + 16));
            acc1 = vfmaq_f32(acc1, vld1q_f32(ki + 20), vld1q_f32(b + 20));
            const float s = hsum4(vaddq_f32(acc0, acc1));
            a[i] = s > 0.0f ? P[i] / s : 0.0f;
        }

        // column sums: s[j] = sum_i a[i] * k[i,j]
        float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0,
                    s3 = s0, s4 = s0, s5 = s0;
        for (int i = 0; i < N; ++i) {
            const float* ki = k + i * NP;
            const float ai = a[i];
            s0 = vfmaq_n_f32(s0, vld1q_f32(ki),      ai);
            s1 = vfmaq_n_f32(s1, vld1q_f32(ki +  4), ai);
            s2 = vfmaq_n_f32(s2, vld1q_f32(ki +  8), ai);
            s3 = vfmaq_n_f32(s3, vld1q_f32(ki + 12), ai);
            s4 = vfmaq_n_f32(s4, vld1q_f32(ki + 16), ai);
            s5 = vfmaq_n_f32(s5, vld1q_f32(ki + 20), ai);
        }
        float32x4_t residv = vdupq_n_f32(0.0f);
        finalize_block(s0, Q,      b,      residv);
        finalize_block(s1, Q +  4, b +  4, residv);
        finalize_block(s2, Q +  8, b +  8, residv);
        finalize_block(s3, Q + 12, b + 12, residv);
        finalize_block(s4, Q + 16, b + 16, residv);
        finalize_block(s5, Q + 20, b + 20, residv);

        if (hmax4(residv) < tol) break;
    }

    // x[i,j] = a[i] * k[i,j] * b[j]
    for (int i = 0; i < N; ++i) {
        const float* ki = k + i * NP;
        float* xi = x + i * NP;
        const float ai = a[i];
        for (int j = 0; j < NP; j += 4) {
            const float32x4_t t = vmulq_n_f32(vld1q_f32(ki + j), ai);
            vst1q_f32(xi + j, vmulq_f32(t, vld1q_f32(b + j)));
        }
    }
}

bool matrix_adjust(const float* q, const float* P, const float* Q, float* x, float target_re) {
    alignas(16) float k[N * NP];
    alignas(16) float L[N * NP], lnPQ[N * NP];

    for (int i = 0; i < N; ++i) {
        const float32x4_t Pi = vdupq_n_f32(P[i]);
        const int row = i * NP;
        for (int j = 0; j < N; j += 4) {                 // N % 4 == 0
            const float32x4_t pq = vmulq_f32(Pi, vld1q_f32(Q + j));
            vst1q_f32(lnPQ + row + j, log_ps(pq));
            const float32x4_t qv = vld1q_f32(q + row + j);
            vst1q_f32(L + row + j, log_ps(vdivq_f32(qv, pq)));
        }
        std::fill(L + row + N, L + row + NP, 0.0f);
        std::fill(lnPQ + row + N, lnPQ + row + NP, 0.0f);
    }

    float a[NP], b[NP];
    for (int i = 0; i < N; ++i)  { a[i] = 1.0f; b[i] = 1.0f; }
    for (int i = N; i < NP; ++i) { a[i] = 0.0f; b[i] = 0.0f; }

    auto re_at = [&](float theta) -> float {
        float32x4_t vmax = vdupq_n_f32(std::numeric_limits<float>::lowest());
        for (int i = 0; i < N; ++i) {
            const int base = i * NP;
            for (int j = 0; j < NP; j += 4) {
                const float32x4_t e = vfmaq_n_f32(vld1q_f32(lnPQ + base + j),
                                                  vld1q_f32(L + base + j), theta);
                vst1q_f32(k + base + j, e);
                vmax = vmaxq_f32(vmax, e);
            }
        }
        const float e_max = hmax4(vmax);

        const float32x4_t vemax = vdupq_n_f32(e_max);
        for (int t = 0; t < N * NP; t += 4) {
            float32x4_t v = vld1q_f32(&k[t]);
            v = exp_ps(vsubq_f32(v, vemax));
            vst1q_f32(&k[t], v);
        }

        ras_balance(k, P, Q, a, b, x);
        return relative_entropy_neon(x, lnPQ);
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

    for (int it = 0; it < 30; ++it) {
        const float theta = lo - flo * (hi - lo) / (fhi - flo);
        const float f = re_at(theta) - target_re;
        if (std::fabs(f) < RE_TOL) break;
        if (f > 0.0f) { hi = theta; fhi = f; flo *= 0.5f; }
        else          { lo = theta; flo = f; fhi *= 0.5f; }
    }
    return true;
}

}}