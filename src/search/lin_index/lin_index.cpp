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

#define NOMINMAX
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>
#include "mio/mmap.hpp"
#include "lin_index.h"
#include "basic/config.h"
#include "basic/shape_config.h"
#include "data/block/block.h"
#include "run/config.h"
#include "search/seed_array/enum_seeds.h"
#include "search/seed_complexity.h"
#include "util/io/file.h"
#include "util/log_stream.h"
#include "util/ptr_vector.h"

using std::atomic;
using std::endl;
using std::runtime_error;
using std::string;
using std::unique_ptr;
using std::vector;

namespace Search {

// Counts the seed positions of the block in order to size the hash table.
struct CountCallback {
	bool operator()(uint64_t, uint64_t, uint32_t, uint64_t) {
		++n;
		return true;
	}
	void finish() {}
	uint64_t n = 0;
};

struct InsertCallback {

	InsertCallback(atomic<uint64_t>* table, uint64_t table_size, const SequenceSet& seqs, const Shape& shape, double seed_complexity_cut) :
		table(table),
		table_size(table_size),
		seqs(seqs),
		shape(shape),
		seed_complexity_cut(seed_complexity_cut)
	{}

	bool operator()(const uint64_t key, const uint64_t pos, uint32_t, uint64_t) {
		// Mirrors Search::mask_seeds, which discards seed groups whose pivot is
		// of low complexity.
		if (!seed_is_complex(seqs.data(pos), shape, seed_complexity_cut)) {
			++low_complexity;
			return true;
		}
		insert(key, pos);
		return true;
	}

	void finish() {}

	void insert(const uint64_t key, const uint64_t pos) {
		const uint64_t h = LinIndex::hash(key), f = LinIndex::fingerprint(h), value = (f << LinIndex::POS_BITS) | pos;
		uint64_t i = LinIndex::bucket(h, table_size), probes = 0;
		for (;;) {
			uint64_t cur = table[i].load(std::memory_order_relaxed);
			if (cur == 0) {
				if (table[i].compare_exchange_weak(cur, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
					++inserted;
					return;
				}
				continue;
			}
			if ((cur >> LinIndex::POS_BITS) == f) {
				// The block is sorted by decreasing sequence length, so the
				// smallest position is the occurrence in the longest sequence.
				if ((cur & LinIndex::POS_MASK) <= pos)
					return;
				if (table[i].compare_exchange_weak(cur, value, std::memory_order_relaxed, std::memory_order_relaxed))
					return;
				continue;
			}
			if (++i == table_size)
				i = 0;
			if (++probes == table_size)
				throw runtime_error("Seed index hash table overflow.");
		}
	}

	atomic<uint64_t>* const table;
	const uint64_t table_size;
	const SequenceSet& seqs;
	const Shape& shape;
	const double seed_complexity_cut;
	uint64_t inserted = 0, low_complexity = 0;

};

static void check_length_sorted(const SequenceSet& seqs) {
	const BlockId n = seqs.size();
	for (BlockId i = 1; i < n; ++i)
		if (seqs.length(i) > seqs.length(i - 1))
			throw runtime_error("Seed index requires a block sorted by decreasing sequence length.");
}

void build_lin_index(Block& block, const string& file_name, const Config& cfg, int threads) {
	if (shapes.count() != 1)
		throw runtime_error("Seed index is only supported for a single seed shape.");
	const SequenceSet& seqs = block.seqs();
	if ((uint64_t)seqs.raw_len() > LinIndex::POS_MASK)
		throw runtime_error("Block size exceeds the maximum supported by the seed index.");
	check_length_sorted(seqs);
	threads = std::max(threads, 1);

	TaskTimer timer("Counting seeds");
	const auto partition = block.seqs().partition(threads);
	const EnumCfg enum_cfg{ &partition, 0, 1, cfg.seed_encoding, nullptr, false, false, cfg.seed_complexity_cut,
		cfg.soft_masking, cfg.minimizer_window, false, false, cfg.sketch_size, nullptr };
	uint64_t seed_count = 0;
	{
		PtrVector<CountCallback> v;
		for (int i = 0; i < threads; ++i)
			v.push_back(new CountCallback());
		enum_seeds(block, v, &no_filter, enum_cfg);
		for (int i = 0; i < threads; ++i)
			seed_count += v[i].n;
	}
	timer.finish();

	const uint64_t table_size = std::max<uint64_t>(16, (uint64_t)((double)seed_count / LinIndex::LOAD_FACTOR) + 1);
	*log_stream << "Seed index: seeds=" << seed_count << " table_size=" << table_size
		<< " memory=" << table_size * sizeof(uint64_t) << endl;

	timer.go("Allocating seed index");
	unique_ptr<atomic<uint64_t>[]> table(new atomic<uint64_t>[table_size]);
	{
		vector<std::thread> workers;
		for (int i = 0; i < threads; ++i)
			workers.emplace_back([&table, table_size, threads, i] {
				const uint64_t begin = table_size / threads * i, end = i == threads - 1 ? table_size : table_size / threads * (i + 1);
				for (uint64_t j = begin; j < end; ++j)
					table[j].store(0, std::memory_order_relaxed);
				});
		for (auto& t : workers)
			t.join();
	}

	timer.go("Building seed index");
	uint64_t inserted = 0, low_complexity = 0;
	{
		PtrVector<InsertCallback> v;
		for (int i = 0; i < threads; ++i)
			v.push_back(new InsertCallback(table.get(), table_size, seqs, shapes[0], cfg.seed_complexity_cut));
		enum_seeds(block, v, &no_filter, enum_cfg);
		for (int i = 0; i < threads; ++i) {
			inserted += v[i].inserted;
			low_complexity += v[i].low_complexity;
		}
	}
	*log_stream << "Seed index: entries=" << inserted << " low complexity seeds=" << low_complexity
		<< " load=" << (double)inserted / table_size << endl;

	timer.go("Writing seed index");
	LinIndex::Header header;
	header.magic = LinIndex::MAGIC;
	header.version = LinIndex::VERSION;
	header.shape_count = (uint32_t)shapes.count();
	header.seq_count = (uint64_t)seqs.size();
	header.raw_len = (uint64_t)seqs.raw_len();
	header.table_size = table_size;
	header.entry_count = inserted;
	File out(file_name, "wb");
	out.write(header);
	static const uint64_t BUF = 1 << 20;
	vector<uint64_t> buf(BUF);
	for (uint64_t i = 0; i < table_size; i += BUF) {
		const uint64_t n = std::min(BUF, table_size - i);
		for (uint64_t j = 0; j < n; ++j)
			buf[j] = table[i + j].load(std::memory_order_relaxed);
		out.write(buf.data(), n * sizeof(uint64_t));
	}
	out.close();
	timer.finish();
}

LinIndex::LinIndex(const string& file_name) {
	mmap_.reset(new mio::mmap_source(file_name));
	if (mmap_->length() < sizeof(Header))
		throw runtime_error("Invalid seed index file: " + file_name);
	memcpy(&header_, mmap_->data(), sizeof(Header));
	if (header_.magic != MAGIC)
		throw runtime_error("Invalid seed index file: " + file_name);
	if (header_.version != VERSION)
		throw runtime_error("Invalid seed index file version: " + file_name);
	if (header_.shape_count != (uint32_t)shapes.count())
		throw runtime_error("Seed index has a different number of shapes: " + file_name);
	if (mmap_->length() < sizeof(Header) + header_.table_size * sizeof(uint64_t))
		throw runtime_error("Truncated seed index file: " + file_name);
	// Guarantees that a lookup of a seed that is not in the index terminates on a
	// blank slot instead of probing the table forever.
	if (header_.entry_count >= header_.table_size)
		throw runtime_error("Invalid seed index file: " + file_name);
	table_ = (const uint64_t*)(mmap_->data() + sizeof(Header));
	table_size_ = header_.table_size;
}

LinIndex::~LinIndex() {
}

}
