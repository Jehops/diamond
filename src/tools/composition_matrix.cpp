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
#include <array>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>
#include "basic/config.h"
#include "basic/statistics.h"
#include "data/sequence_file.h"
#include "stats/cbs.h"
#include "stats/score_matrix.h"

using std::array;
using std::ofstream;
using std::ostream;
using std::pair;
using std::unique_ptr;
using std::vector;

static uint64_t pair_count(const uint64_t n)
{
	if (n < 2)
		return 0;
	uint64_t a = n;
	uint64_t b = n - 1;
	if (a % 2 == 0)
		a /= 2;
	else
		b /= 2;
	if (a > std::numeric_limits<uint64_t>::max() / b)
		throw std::runtime_error("Too many database sequences to enumerate sequence pairs.");
	return a * b;
}

static pair<size_t, size_t> pair_from_index(const uint64_t index, const size_t sequence_count)
{
	const uint64_t total = pair_count(sequence_count);
	size_t low = 0;
	size_t high = sequence_count - 1;
	while (low + 1 < high) {
		const size_t middle = low + (high - low) / 2;
		const uint64_t first_index = total - pair_count(sequence_count - middle);
		if (first_index <= index)
			low = middle;
		else
			high = middle;
	}
	const uint64_t first_index = total - pair_count(sequence_count - low);
	return { low, low + 1 + static_cast<size_t>(index - first_index) };
}

static vector<uint64_t> sample_pair_indices(const uint64_t total, const uint64_t sample_size)
{
	std::random_device random_device;
	std::seed_seq seed{
		random_device(), random_device(), random_device(), random_device(),
		random_device(), random_device(), random_device(), random_device()
	};
	std::mt19937_64 generator(seed);
	std::unordered_set<uint64_t> selected;
	selected.reserve(static_cast<size_t>(sample_size));
	for (uint64_t upper = total - sample_size; upper < total; ++upper) {
		const uint64_t candidate = std::uniform_int_distribution<uint64_t>(0, upper)(generator);
		if (!selected.insert(candidate).second)
			selected.insert(upper);
	}
	vector<uint64_t> result(selected.begin(), selected.end());
	std::sort(result.begin(), result.end());
	return result;
}

void composition_matrix_workflow()
{
	unique_ptr<SequenceFile> db(SequenceFile::auto_create({ config.database }, SequenceFile::Flags::SEQS));
	unique_ptr<Block> block(db->load_seqs(INT64_MAX));
	const SequenceSet& sequences = block->seqs();

	vector<Stats::Composition> compositions;
	vector<int> lengths;
	compositions.reserve(sequences.size());
	lengths.reserve(sequences.size());
	for (size_t i = 0; i < sequences.size(); ++i) {
		compositions.push_back(Stats::composition(sequences[i]));
		lengths.push_back(Stats::count_true_aa(sequences[i]));
	}

	unique_ptr<ofstream> output_file;
	ostream* output = &std::cout;
	if (!config.output_file.empty()) {
		output_file.reset(new ofstream(config.output_file));
		if (!*output_file)
			throw std::runtime_error("Failed to open output file: " + config.output_file);
		output = output_file.get();
	}
	*output << std::setprecision(17);

	Statistics stats;
	array<int, AMINO_ACID_COUNT * AMINO_ACID_COUNT> matrix;
	const auto write_pair = [&](const size_t first, const size_t second) {

		/*array<double, 400> out;
		int iterations;
		double row_probs[20], col_probs[20];
		std::copy(compositions[first].data(), compositions[first].data() + 20, row_probs);
		std::copy(compositions[second].data(), compositions[second].data() + 20, col_probs);
		Stats::Blast_ApplyPseudocounts(row_probs, lengths[first],
			score_matrix.background_freqs());
		Stats::Blast_ApplyPseudocounts(col_probs, lengths[second],
			score_matrix.background_freqs());

		Stats::Blast_OptimizeTargetFrequencies(out.data(), 20, &iterations, score_matrix.joint_probs(), row_probs, col_probs, true, 0.44, config.cbs_err_tolerance,
			config.cbs_it_limit);*/

		if (config.comp_based_stats_.get(Stats::DEFAULT_CBS) == Stats::CBS::SINKHORN_MATRIX_ADJUST) {
			Stats::matrix_adjust(
				lengths[first],
				lengths[second],
				compositions[first].data(),
				compositions[second].data(),
				1,
				score_matrix.ideal_lambda(),
				score_matrix.joint_probs(),
				score_matrix.background_freqs(),
				score_matrix.joint_probs_f(),
				score_matrix.background_freqs_f(),
				matrix,
				stats);
		}
		else {
			Stats::CompositionMatrixAdjust(
				lengths[first],
				lengths[second],
				compositions[first].data(),
				compositions[second].data(),
				1,
				score_matrix.ideal_lambda(),
				score_matrix.joint_probs(),
				score_matrix.background_freqs(),
				matrix,
				stats);
		}

		bool initial_column = true;
		const auto write_column = [&](const double value) {
			if (!initial_column)
				*output << '\t';
			*output << value;
			initial_column = false;
		};

		for (const double value : compositions[first])
			write_column(value);
		for (const double value : compositions[second])
			write_column(value);
		for (size_t row = 0; row < TRUE_AA; ++row)
			for (size_t column = 0; column < TRUE_AA; ++column)
				//write_column(out[row * 20 + column]);
				write_column(matrix[row * AMINO_ACID_COUNT + column]);
		*output << '\n';
	};

	const uint64_t total_pairs = pair_count(sequences.size());
	if (config.composition_matrix_sample_size.present()) {
		const int64_t requested_sample_size = config.composition_matrix_sample_size;
		if (requested_sample_size < 0)
			throw std::runtime_error("Sample size must not be negative.");
		const uint64_t sample_size = static_cast<uint64_t>(requested_sample_size);
		if (sample_size > total_pairs)
			throw std::runtime_error(
				"Requested sample size (" + std::to_string(sample_size) +
				") exceeds the number of sequence pairs (" + std::to_string(total_pairs) + ").");
		for (const uint64_t index : sample_pair_indices(total_pairs, sample_size)) {
			const auto pair = pair_from_index(index, sequences.size());
			write_pair(pair.first, pair.second);
		}
	}
	else {
		for (size_t first = 0; first < sequences.size(); ++first)
			for (size_t second = first + 1; second < sequences.size(); ++second)
				write_pair(first, second);
	}

	if (!*output)
		throw std::runtime_error("Failed writing composition matrix output.");
}
