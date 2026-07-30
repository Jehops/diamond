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
#include <stdexcept>
#include <string>
#include "basic/config.h"
#include "stats/score_matrix.h"
#include "seqindex.h"
#include "util/log_stream.h"

using std::array;
using std::runtime_error;

static constexpr int MAX_KMER_LEN = Seqindex::MAX_KMER_LEN;

struct ScoredLetter {
	Letter letter;
	int score;
};

using SubsRow = array<ScoredLetter, TRUE_AA>;

struct SubstitutionTable {
	SubstitutionTable() {
		for (Letter q = 0; q < TRUE_AA; ++q) {
			SubsRow& r = row[q];
			for (Letter l = 0; l < TRUE_AA; ++l)
				r[l] = { l, score_matrix(q, l) };
			std::sort(r.begin(), r.end(),
				[](const ScoredLetter& lhs, const ScoredLetter& rhs) {
					return lhs.score != rhs.score ? lhs.score > rhs.score : lhs.letter < rhs.letter;
				});
		}
	}
	array<SubsRow, TRUE_AA> row;
};

namespace {

struct Emit {
	void operator()(int pos, int prefix_score, uint32_t code) const {
		if (pos == k) {
			out.push_back({ code, value });
			return;
		}
		for (const ScoredLetter& sl : *subs[pos]) {
			if (prefix_score + sl.score + best_suffix[pos + 1] < min_score)
				break;
			(*this)(pos + 1, prefix_score + sl.score, code * TRUE_AA + (uint32_t)sl.letter);
		}
	}
	const int k, min_score;
	const SubsRow* const* subs;
	const int* best_suffix;
	const Loc value;
	std::pmr::vector<Seqindex::Entry>& out;
};

}

Seqindex::Seqindex(Sequence seq, int k, double word_threshold, std::pmr::memory_resource& pool, Statistics& stats):
	k_(k),
	shift_(31),
	mod_(1),
	entry_(&pool),
	next_(&pool),
	head_(&pool)
{
	if (k < MIN_KMER_LEN || k > MAX_KMER_LEN)
		throw runtime_error("Seqindex k-mer length must be between "
			+ std::to_string(MIN_KMER_LEN) + " and " + std::to_string(MAX_KMER_LEN) + ".");

	TaskTimer timer;
	for (int i = 0; i < k; ++i)
		mod_ *= TRUE_AA;

	if (seq.length() >= k) {
		const SubstitutionTable table;
		const int min_score = score_matrix.rawscore(word_threshold);
		const SubsRow* subs[MAX_KMER_LEN];
		int best_suffix[MAX_KMER_LEN + 1];
		best_suffix[k] = 0;
		for (Loc pos = 0; pos <= seq.length() - k; ++pos) {
			const Letter* mer = seq.data() + pos;
			if (std::any_of(mer, mer + k,
				[](Letter letter) { return letter < 0 || letter >= TRUE_AA; }))
				continue;

			for (int i = 0; i < k; ++i)
				subs[i] = &table.row[(int)mer[i]];
			for (int i = k - 1; i >= 0; --i)
				best_suffix[i] = best_suffix[i + 1] + (*subs[i])[0].score;
			Emit{ k, min_score, subs, best_suffix, pos, entry_ }(0, 0, 0);
		}
	}

	size_t buckets = 2;
	while (buckets < entry_.size() / 4)
		buckets <<= 1;
	shift_ = 32;
	for (size_t b = buckets; b > 1; b >>= 1)
		--shift_;
	head_.assign(buckets, -1);
	next_.resize(entry_.size());
	for (int32_t i = (int32_t)entry_.size() - 1; i >= 0; --i) {
		const uint32_t b = bucket(entry_[i].code);
		next_[i] = head_[b];
		head_[b] = i;
	}

	stats.inc(Statistics::TIME_KEYWORD_TREE, timer.nanoseconds());
}

Seqindex::~Seqindex() {
}

void Seqindex::scan(Sequence seq, std::function<void(Loc, Loc)>& callback) {
	if (entry_.empty())
		return;
	const Letter* s = seq.data();
	const Loc len = seq.length();
	uint32_t code = 0;
	int run = 0;
	for (Loc j = 0; j < len; ++j) {
		const Letter l = s[j];
		if (l < 0 || l >= TRUE_AA) {
			run = 0;
			code = 0;
			continue;
		}
		code = (code * TRUE_AA + (uint32_t)l) % mod_;
		if (++run < k_)
			continue;
		const Loc begin = j - k_ + 1;
		for (int32_t i = head_[bucket(code)]; i != -1; i = next_[i])
			if (entry_[i].code == code)
				callback(entry_[i].pos, begin);
	}
}