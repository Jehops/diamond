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
#include "../stats.h"

#ifdef __AVX2__
#include "kernel_avx2.h"
#elif defined(__SSSE3__) && defined(__POPCNT__) && defined(__SSE4_1__)
#include "kernel_sse41.h"
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include "kernel_arm64.h"
#else
namespace Stats { namespace ARCH_GENERIC {
bool matrix_adjust(const float* q, const float* P, const float* Q, float* x, float target_re) {
	return Stats::matrix_adjust_scalar(q, P, Q, x, target_re);
}
}}
#endif

namespace Stats {

DISPATCH_5(bool, matrix_adjust, const float*, q, const float*, P, const float*, Q, float*, x, float, target_re)

}