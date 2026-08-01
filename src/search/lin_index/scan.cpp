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

#include <array>
#include <limits>
#include <memory>
#include "lin_index.h"
#include "basic/seed.h"
#include "basic/shape_config.h"
#include "data/block/block.h"
#include "stats/score_matrix.h"
#include "search/hamming/finger_print.h"
#include "search/seed_array/enum_seeds.h"
#include "search/stage2.h"
#include "util/log_stream.h"
#include "util/ptr_vector.h"
#include "util/simd/dispatch.h"

using std::array;
using std::vector;
using ::DISPATCH_ARCH::FingerPrint;

namespace Search { namespace DISPATCH_ARCH {

/* Scans the member block against the seed index of the reference block. For
   every seed of the member block the index yields at most one location, the
   occurrence of that seed in the longest sequence of the reference block, which
   takes the role of the query in the resulting seed hit. */

struct ScanCallback {

	ScanCallback(const Context& context, Search::Config& cfg, const LinIndex& index, size_t thread_id) :
		writer(*cfg.seed_hit_buf, thread_id),
		work_set(context, cfg, 0, &writer, nullptr, nullptr),
		index(index),
		query_seqs(cfg.query->seqs()),
		target_seqs(cfg.target->seqs()),
		target_seed_hits(cfg.target_seed_hits.get()),
		hamming_filter_id(cfg.hamming_filter_id),
		self(config.self && cfg.current_ref_block == 0)
	{}

	bool operator()(const uint64_t key, const uint64_t pos, const uint32_t block_id, const uint64_t shape_id) {
		const uint64_t pivot = index(key);
		if (pivot == 0)
			return true;
		work_set.stats.inc(Statistics::SEED_HITS);

		alignas(64) array<char, 48> fq, fs;
		FingerPrint::load(query_seqs.data(pivot), &fq);
		FingerPrint::load(target_seqs.data(pos), &fs);
		if (FingerPrint(fq).match(FingerPrint(fs)) < hamming_filter_id)
			return true;
		work_set.stats.inc(Statistics::TENTATIVE_MATCHES1);

		const std::pair<BlockId, Loc> l = query_seqs.local_position((int64_t)pivot);
		const unsigned query_id = l.first;
		if (self && query_id == block_id)
			return true;
		const Loc seed_offset = l.second;
		const int query_len = query_seqs.length(query_id);
		const int score_cutoff = ungapped_cutoff(query_len, work_set);
		int score = std::numeric_limits<int>::max();
		if (score_cutoff) {
			const int window = ungapped_window(query_len);
			const Letter* query = query_seqs.data(pivot);
			const Sequence query_clipped = Util::Seq::clip(query - window, window * 2, window);
			const int window_left = int(query - query_clipped.data()), window_clipped = (int)query_clipped.length();
			const Letter* subject = target_seqs.data(pos) - window_left;
			DP::window_ungapped_best(query_clipped.data(), &subject, 1, window_clipped, &score);
			if (score <= score_cutoff)
				return true;
		}
		work_set.stats.inc(Statistics::TENTATIVE_MATCHES3);

		if (target_seed_hits)
			target_seed_hits->operator[]((size_t)shape_id).atomic_set(pos);
		if (query_id != last_query_ || seed_offset != last_seed_offset_) {
			writer.new_query(query_id, seed_offset);
			last_query_ = query_id;
			last_seed_offset_ = seed_offset;
		}
		writer.write(query_id, PackedLoc(pos), (uint16_t)score, block_id);
		return true;
	}

	void finish() {}

	HitBuffer::Writer writer;
	WorkSet work_set;
	const LinIndex& index;
	const SequenceSet& query_seqs, & target_seqs;
	std::vector<BitVector>* const target_seed_hits;
	const unsigned hamming_filter_id;
	const bool self;
	unsigned last_query_ = std::numeric_limits<unsigned>::max();
	Loc last_seed_offset_ = std::numeric_limits<Loc>::min();

};

void scan_lin_index(Search::Config& cfg) {
	if (shapes.count() != 1)
		throw std::runtime_error("Seed index search is only supported for a single seed shape.");
	const int threads = std::max(config.threads_, 1);
	const vector<uint32_t> patterns = shapes.patterns(0, 1);
	// A Context holds two PatternMatcher lookup tables of 2^MAX_SHAPE_LEN bytes
	// each and must not be placed on the stack.
	const std::unique_ptr<Context> context(new Context{ { patterns.data(), patterns.data() + patterns.size() - 1 },
		{ patterns.data(), patterns.data() + patterns.size() },
		score_matrix.rawscore(config.short_query_ungapped_bitscore),
		nullptr,
		seedp_mask(cfg.seedp_bits) });

	const auto partition = cfg.target->seqs().partition(threads);
	// Soft masking is applied to the reference block when its index is built. The
	// member block is scanned unmasked so that the fingerprints of both blocks are
	// computed on the same, unmasked sequence data.
	const EnumCfg enum_cfg{ &partition, 0, 1, cfg.seed_encoding, nullptr, false, false, cfg.seed_complexity_cut,
		MaskingAlgo::NONE, cfg.minimizer_window, false, false, cfg.sketch_size, cfg.target_seed_hits.get() };

	PtrVector<ScanCallback> v;
	for (int i = 0; i < threads; ++i)
		v.push_back(new ScanCallback(*context, cfg, *cfg.lin_index, i));
	enum_seeds(*cfg.target, v, &no_filter, enum_cfg);
	for (int i = 0; i < threads; ++i)
		statistics += v[i].work_set.stats;
	v.clear();
}

}

DISPATCH_1V(scan_lin_index, Search::Config&, cfg)

}
