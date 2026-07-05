/****
DIAMOND protein sequence aligner
Copyright (C) 2012-2026 Benjamin J. Buchfink
Arm NEON port contributed by Martin Larralde <martin.larralde@embl.de>

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

#pragma once
#include <stdint.h>
#include "../simd.h"

namespace DISPATCH_ARCH { namespace SIMD {

template<>
struct Vector<int8_t> {

	static constexpr size_t LANES = 16;

	Vector()
	{}

	Vector(const signed char *p):
		v(vld1q_s8(p))
	{}

	operator int8x16_t() const {
		return v;
	}

	int8x16_t v;

};

template<>
struct Vector<int16_t> {

	static constexpr size_t LANES = 8;

	Vector()
	{}

	Vector(const int16_t* p) :
		v(vld1q_s16(p))
	{}

	operator int16x8_t() const {
		return v;
	}

	int16x8_t v;

};

template<>
struct Vector<int32_t> {

	static constexpr size_t LANES = 1;

	Vector()
	{}

	Vector(const int32_t* p) :
		v(*p)
	{}

	operator int32_t() const {
		return v;
	}

	int32_t v;

};

template<>
struct Traits<float> {
	static constexpr size_t LANES = 4;
	using Register = float32x4_t;
};

static inline float32x4_t add(float32x4_t a, float32x4_t b) {
	return vaddq_f32(a, b);
}

static inline float32x4_t zero(float32x4_t) {
	return vdupq_n_f32(0.0f);
}

// Sets only lane 0 to x (like _mm_set_ss), other lanes = 0
static inline float32x4_t set(float x, float32x4_t) {
	return vdupq_n_f32(x);
}

static inline float32x4_t unaligned_load(const float* p, float32x4_t) {
	// NEON uses the same intrinsic for aligned/unaligned loads
	return vld1q_f32(p);
}

static inline float32x4_t load(const float* p, float32x4_t) {
	return vld1q_f32(p);
}

static inline void unaligned_store(float32x4_t v, float* p) {
	vst1q_f32(p, v);
}

static inline void store(float32x4_t v, float* p) {
	vst1q_f32(p, v);
}

static inline float32x4_t mul(float32x4_t a, float32x4_t b) {
	return vmulq_f32(a, b);
}

static inline float32x4_t fmadd(float32x4_t a, float32x4_t b, float32x4_t c) {
	// c + a*b  (matches add(mul(a,b), c))
#if defined(__aarch64__)
	return vfmaq_f32(c, a, b);
#else
	// vmlaq_f32 is fused on CPUs with FMA; otherwise it may lower to mul+add
	return vmlaq_f32(c, a, b);
#endif
}

static inline float hsum(float32x4_t v) {
#if defined(__aarch64__)
	return vaddvq_f32(v);
#else
	// pairwise reduce: (v0+v1) + (v2+v3)
	float32x2_t vlow = vget_low_f32(v);
	float32x2_t vhigh = vget_high_f32(v);
	float32x2_t sum2 = vadd_f32(vlow, vhigh);      // [v0+v2, v1+v3]
	float32x2_t sum1 = vpadd_f32(sum2, sum2);      // [ (v0+v2)+(v1+v3), ... ]
	return vget_lane_f32(sum1, 0);
#endif
}

}}