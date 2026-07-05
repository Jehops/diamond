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

#pragma once
#include "util/geo/interval.h"
#include "basic/sequence.h"

namespace Stats {

bool matrix_adjust_scalar(const double* q, const double* P, const double* Q, double* x, double target_re);
bool matrix_adjust_scalar(const float* q, const float* P, const float* Q, float* x, float target_re);
// Float SIMD path: q, Q, and x use a 24-column padded stride.
bool matrix_adjust(const float* q, const float* P, const float* Q, float* x, float target_re);
double approx_id(Score raw_score, Loc range1, Loc range2);
//double approx_id(int score, Interval query_range, Interval target_range, const Sequence& query, const Sequence& target);

int
Blast_OptimizeTargetFrequencies(double x[],
    int alphsize,
    int* iterations,
    const double q[],
    const double row_sums[],
    const double col_sums[],
    int constrain_rel_entropy,
    double relative_entropy,
    double tol,
    int maxits);

}
