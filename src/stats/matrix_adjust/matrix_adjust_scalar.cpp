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

#include <algorithm>
#include <cmath>
#include <limits>

#include "util/simd/dispatch.h"
#include "basic/config.h"
#include "../stats.h"
#include "util/math/math.h"

namespace Stats { namespace DISPATCH_ARCH {

constexpr int N = 20, NP = 24;

template<typename Float, int STRIDE>
static void ras_balance(const Float* k, const Float* P, const Float* Q, Float* a, Float* b, Float* x, int max_iters = 100, Float tol = 1e-7) {
    for (int it = 0; it < max_iters; ++it) {
        for (int i = 0; i < N; ++i) {
            Float s = 0.0;
            for (int j = 0; j < N; ++j) s += k[i * STRIDE + j] * b[j];
            a[i] = s > Float(0.0) ? P[i] / s : Float(0.0);
        }
        Float resid = 0.0;
        for (int j = 0; j < N; ++j) {
            Float s = 0.0;
            for (int i = 0; i < N; ++i) s += k[i * STRIDE + j] * a[i];
            const Float bj = s > Float(0.0) ? Q[j] / s : Float(0.0);
            resid = std::max(resid, std::fabs(bj - b[j]));
            b[j] = bj;
        }
        if (resid < tol) break;
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            x[i * STRIDE + j] = k[i * STRIDE + j] * a[i] * b[j];
        for (int j = N; j < STRIDE; ++j)
            x[i * STRIDE + j] = Float(0.0);
    }
}

template <typename Float, int STRIDE>
static bool matrix_adjust_impl(const Float* q, const Float* P, const Float* Q, Float* x, Float target_re) {
    Float L[N * STRIDE], lnPQ[N * STRIDE], k[N * STRIDE];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            const int ij = i * STRIDE + j;
            const Float pq = P[i] * Q[j];
            lnPQ[ij] = Math::log(pq);
            L[ij] = Math::log(q[ij] / pq);
        }

    Float a[N], b[N];
    for (int i = 0; i < N; ++i) { a[i] = 1.0; b[i] = 1.0; }

    auto re_at = [&](Float theta) -> Float {
        Float emax = std::numeric_limits<Float>::lowest();
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const int ij = i * STRIDE + j;
                const Float e = lnPQ[ij] + theta * L[ij];
                k[ij] = e;
                if (e > emax) emax = e;
            }
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const int ij = i * STRIDE + j;
                k[ij] = Math::exp(k[ij] - emax);
            }
        ras_balance<Float, STRIDE>(k, P, Q, a, b, x);
        Float re = 0.0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const int ij = i * STRIDE + j;
                const Float xij = x[ij];
                if (xij > Float(0.0)) re += xij * (Math::log(xij) - lnPQ[ij]);
            }
        return re;
        };

    const Float TMIN = Float(0.05), TMAX = Float(21.0), RE_TOL = Float(config.relative_entropy_tolerance);

    Float lo, hi, flo, fhi;
    const Float f1 = re_at(1.0) - target_re;
    if (std::fabs(f1) < RE_TOL) return true;

    if (f1 < Float(0.0)) {        
        lo = 1.0; flo = f1; hi = 0.0; fhi = 0.0;
        bool bracketed = false;
        for (Float t = 1.0; t < TMAX; ) {
            t = std::min(t * Float(2.0), TMAX);
            const Float ft = re_at(t) - target_re;
            if (ft >= Float(0.0)) { hi = t; fhi = ft; bracketed = true; break; }
            lo = t; flo = ft;
        }
        if (!bracketed) return false;
    }
    else {        
        flo = re_at(TMIN) - target_re;
        if (flo >= Float(0.0)) return true;
        lo = TMIN; hi = 1.0; fhi = f1;
    }

    for (int it = 0; it < 30; ++it) {
        const Float theta = lo - flo * (hi - lo) / (fhi - flo);
        const Float f = re_at(theta) - target_re;
        if (std::fabs(f) < RE_TOL) break;
        if (f > Float(0.0)) { hi = theta; fhi = f; flo *= Float(0.5); }
        else { lo = theta; flo = f; fhi *= Float(0.5); }
    }
    return true;
}

bool matrix_adjust_scalar(const double* q, const double* P, const double* Q, double* x, double target_re) {
    return matrix_adjust_impl<double, N>(q, P, Q, x, target_re);
}

bool matrix_adjust_scalar(const float* q, const float* P, const float* Q, float* x, float target_re) {
    return matrix_adjust_impl<float, NP>(q, P, Q, x, target_re);
}

}

DISPATCH_5(bool, matrix_adjust_scalar, const double*, q, const double*, P, const double*, Q, double*, x, double, target_re)
DISPATCH_5(bool, matrix_adjust_scalar, const float*, q, const float*, P, const float*, Q, float*, x, float, target_re)

}