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

#include <array>
#include <vector>
#include "util/memory/memory_resource.h"
#include "basic/sequence.h"
#include "basic/statistics.h"
#include "standard_matrix.h"
#include "masking/def.h"

namespace Stats {

using Composition = std::array<double, TRUE_AA>;

Composition composition(const Sequence& s);

/** An collection of constants that specify all rules that may
 *  be used to generate a compositionally adjusted matrix.  */
typedef enum EMatrixAdjustRule {
    eDontAdjustMatrix = (-1),
    eCompoScaleOldMatrix = 0,
    eUnconstrainedRelEntropy = 1,
    eRelEntropyOldMatrixNewContext = 2,
    eRelEntropyOldMatrixOldContext = 3,
    eUserSpecifiedRelEntropy = 4
} EMatrixAdjustRule;

struct TargetMatrix {

    TargetMatrix(const Composition& query_comp, int query_len, unsigned cbs, const Sequence& target, Statistics& stats, std::pmr::monotonic_buffer_resource& pool, EMatrixAdjustRule rule);
    int score_width() const;

    std::pmr::vector<int8_t> scores;
    //std::vector<int32_t> scores32;
    //Scores<int8_t> scores_low, scores_high;

    int score_min, score_max;

};

/** Work arrays used to perform composition-based matrix adjustment */
typedef struct Blast_CompositionWorkspace {
    double** mat_b;       /**< joint probabilities for the matrix in
                                standard context */
    double** mat_final;   /**< optimized target frequencies */

    double* first_standard_freq;     /**< background frequency vector
                                           of the first sequence */
    double* second_standard_freq;    /**< background frequency vector of
                                           the second sequence */
} Blast_CompositionWorkspace;

/** Information about a amino-acid substitution matrix */
typedef struct Blast_MatrixInfo {
    char* matrixName;         /**< name of the matrix */
    int** startMatrix;     /**< Rescaled values of the original matrix */
    double** startFreqRatios;  /**< frequency ratios used to calculate matrix
                                    scores */
    int      rows;             /**< the number of rows in the scoring
                                    matrix. */
    int      cols;             /**< the number of columns in the scoring
                                    matrix, i.e. the alphabet size. */
    int      positionBased;    /**< is the matrix position-based */
    double   ungappedLambda;   /**< ungapped Lambda value for this matrix
                                    in standard context */
} Blast_MatrixInfo;

EMatrixAdjustRule adjust_matrix(const Composition& query_comp, int query_len, unsigned cbs, const Sequence& target);
void Blast_FreqRatioToScore(double** matrix, size_t rows, size_t cols, double Lambda);
void s_RoundScoreMatrix(int** matrix, size_t rows, size_t cols, double** floatScoreMatrix);
int s_GetMatrixScoreProbs(double** scoreProb, int* obs_min, int* obs_max,
    const int* const* matrix, int alphsize,
    const double* subjectProbArray,
    const double* queryProbArray);
double s_CalcLambda(double probs[], int min_score, int max_score, double lambda0);
double ideal_lambda(const int** matrix);
void s_SetXUOScores(double** M, int alphsize, const double row_probs[], const double col_probs[]);
int count_true_aa(const Sequence& s);
bool use_seg_masking(const Sequence& a, const Sequence& b);

EMatrixAdjustRule
s_TestToApplyREAdjustmentConditional(int Len_query,
    int Len_match,
    const double* P_query,
    const double* P_match,
    const double* background_freqs);

struct CBS {
    static bool hauser(unsigned code) {
        switch (code) {
        case 0:
        case MATRIX_ADJUST:
        //case COMP_BASED_STATS:
        case COMP_BASED_STATS_AND_MATRIX_ADJUST:
        case SINKHORN_MATRIX_ADJUST:
            return false;
        case 1:
        case 2:
        case HAUSER_AND_MATRIX_ADJUST:
            return true;
        default:
            throw std::runtime_error("Unknown CBS code.");
        }
    }
    static bool matrix_adjust(unsigned code) {
        switch (code) {
        case DISABLED:
        case HAUSER:
            return false;
        case DEPRECATED1:
        case HAUSER_AND_MATRIX_ADJUST:
        case MATRIX_ADJUST:
        //case COMP_BASED_STATS:
        case COMP_BASED_STATS_AND_MATRIX_ADJUST:
        case SINKHORN_MATRIX_ADJUST:
            return true;
        default:
            throw std::runtime_error("Unknown CBS code.");
        }
    }
    static bool support_translated(unsigned code) {
        switch (code) {
        case DISABLED:
        case HAUSER:
            return true;
        default:
            return false;
        }
    }
    static bool conditioned(unsigned code) {
        switch (code) {
        case DEPRECATED1:
        case HAUSER_AND_MATRIX_ADJUST:
        case COMP_BASED_STATS_AND_MATRIX_ADJUST:
            return true;
        default:
            return false;
        }
    }
    static int tantan(unsigned code) {
        switch (code) {
        case DISABLED:
        case HAUSER:
            return 1;
        default:
            return 0;
        }
    }
    static int target_seg(unsigned code) {
        switch (code) {
        case DISABLED:
        case HAUSER:
            return 0;
        default:
            return 1;
        }
    }
    static MaskingMode masking_mode(unsigned code) {
        switch (code) {
        case DISABLED:
        case HAUSER:
            return MaskingMode::TANTAN;
        case DEPRECATED1:
        case HAUSER_AND_MATRIX_ADJUST:
        case MATRIX_ADJUST:
        case COMP_BASED_STATS_AND_MATRIX_ADJUST:
        case SINKHORN_MATRIX_ADJUST:
            return MaskingMode::BLAST_SEG;
        default:
            throw std::runtime_error("Unknown CBS code.");
        }
    }
    enum {
        DISABLED = 0,
        HAUSER = 1,
        DEPRECATED1 = 2,
        HAUSER_AND_MATRIX_ADJUST = 3,
        MATRIX_ADJUST = 4,
        //COMP_BASED_STATS = 6,
        COMP_BASED_STATS_AND_MATRIX_ADJUST = 5,
        SINKHORN_MATRIX_ADJUST = 6,
        COUNT
    };
    CBS(unsigned code, double query_match_distance_threshold, double length_ratio_threshold, double angle);
    double query_match_distance_threshold;
    double length_ratio_threshold;
    double angle;
};

constexpr unsigned DEFAULT_CBS = CBS::HAUSER;

void Blast_ApplyPseudocounts(double* probs20, int number_of_observations, const double* background_probs20);
void CompositionMatrixAdjust(int query_len, int target_len, const double* query_comp, const double* target_comp, int scale, double ungapped_lambda, const double* joint_probs, const double* background_freqs, std::array<int, AMINO_ACID_COUNT * AMINO_ACID_COUNT>& out, Statistics& stats);
void matrix_adjust(int query_len, int target_len, const double* query_comp, const double* target_comp, int scale, double ungapped_lambda, const double* joint_probs, const double* background_freqs, const float* joint_probs_f, const float* background_freqs_f, std::array<int, AMINO_ACID_COUNT * AMINO_ACID_COUNT>& out, Statistics& stats);
bool CompositionBasedStats(const int* const* matrix_in, const Composition& queryProb, const Composition& resProb, double lambda, const FreqRatios& freq_ratios, std::array<int, AMINO_ACID_COUNT* AMINO_ACID_COUNT>& out);
int Blast_OptimizeTargetFrequencies(double x[],
    int alphsize,
    int* iterations,
    const double q[],
    const double row_sums[],
    const double col_sums[],
    int constrain_rel_entropy,
    double relative_entropy,
    double tol,
    int maxits);

extern const int ALPH_TO_NCBI[];
extern CBS comp_based_stats;

}