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
#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>
#include "lin_index.h"
#include "basic/config.h"
#include "basic/reduction.h"
#include "basic/shape_config.h"
#include "data/block/block.h"
#include "run/config.h"
#include "search/seed_array/enum_seeds.h"
#include "search/seed_complexity.h"
#include "util/algo/varint.h"
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

/* Applies the reduction to the raw sequence data on the fly, so that the seed at a
   position can be computed without materializing the reduced sequence. */
struct ReducedSeq {
	Letter operator[](const int i) const {
		return (Letter)Reduction::get_reduction()(letter_mask(ptr[i]));
	}
	const Letter* ptr;
};

// Recomputes the seed of a stored pivot position, mirroring what SeedIterator
// yields for that position in enum_seeds.
static bool seed_at(const Letter* seq, const Shape& shape, uint64_t& key) {
	return shape.set_seed_reduced(key, ReducedSeq{ seq });
}

// The positions of the index refer to the seed encoding they were enumerated with,
// and only the default encoding can be recomputed from a position alone.
static void check_seed_encoding(const Config& cfg) {
	if (cfg.seed_encoding != SeedEncoding::SPACED_FACTOR)
		throw runtime_error("Seed index is only supported for the default seed encoding.");
}

bool LinIndex::insert(atomic<uint64_t>* table, const uint64_t table_size, const uint64_t key, const uint64_t pos) {
	const uint64_t h = hash(key), f = fingerprint(h), value = (f << POS_BITS) | pos;
	uint64_t i = bucket(h, table_size), probes = 0;
	for (;;) {
		uint64_t cur = table[i].load(std::memory_order_relaxed);
		if (cur == 0) {
			if (table[i].compare_exchange_weak(cur, value, std::memory_order_relaxed, std::memory_order_relaxed))
				return true;
			continue;
		}
		if ((cur >> POS_BITS) == f) {
			// The block is sorted by decreasing sequence length, so the smallest
			// position is the occurrence in the longest sequence.
			if ((cur & POS_MASK) <= pos)
				return false;
			if (table[i].compare_exchange_weak(cur, value, std::memory_order_relaxed, std::memory_order_relaxed))
				return false;
			continue;
		}
		if (++i == table_size)
			i = 0;
		if (++probes == table_size)
			throw runtime_error("Seed index hash table overflow.");
	}
}

static unique_ptr<atomic<uint64_t>[]> alloc_table(const uint64_t table_size, const int threads) {
	unique_ptr<atomic<uint64_t>[]> table(new atomic<uint64_t>[table_size]);
	vector<std::thread> workers;
	for (int i = 0; i < threads; ++i)
		workers.emplace_back([&table, table_size, threads, i] {
			const uint64_t begin = table_size / threads * i, end = i == threads - 1 ? table_size : table_size / threads * (i + 1);
			for (uint64_t j = begin; j < end; ++j)
				table[j].store(0, std::memory_order_relaxed);
			});
	for (auto& t : workers)
		t.join();
	return table;
}

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
		if (LinIndex::insert(table, table_size, key, pos))
			++inserted;
		return true;
	}

	void finish() {}

	atomic<uint64_t>* const table;
	const uint64_t table_size;
	const SequenceSet& seqs;
	const Shape& shape;
	const double seed_complexity_cut;
	uint64_t inserted = 0, low_complexity = 0;

};

/* Collects the pivot positions that the finished hash table holds. A position is a
   pivot exactly if looking up its seed yields the position itself, so every table
   entry is reported once. Each thread enumerates a contiguous range of sequences,
   so sorting the output of a thread makes the concatenation of all of them sorted.
   The sort is needed because sketched seeds are enumerated in the order of their
   hash value rather than of their position. */
struct ExtractCallback {

	ExtractCallback(const atomic<uint64_t>* table, uint64_t table_size, const SequenceSet& seqs, const Shape& shape, double seed_complexity_cut) :
		table(table),
		table_size(table_size),
		seqs(seqs),
		shape(shape),
		seed_complexity_cut(seed_complexity_cut)
	{}

	bool operator()(const uint64_t key, const uint64_t pos, uint32_t, uint64_t) {
		if (!seed_is_complex(seqs.data(pos), shape, seed_complexity_cut))
			return true;
		if (LinIndex::lookup(table, table_size, key) == pos)
			out.push_back(pos);
		return true;
	}

	void finish() {
		std::sort(out.begin(), out.end());
	}

	const atomic<uint64_t>* const table;
	const uint64_t table_size;
	const SequenceSet& seqs;
	const Shape& shape;
	const double seed_complexity_cut;
	vector<uint64_t> out;

};

static void check_length_sorted(const SequenceSet& seqs) {
	const BlockId n = seqs.size();
	for (BlockId i = 1; i < n; ++i)
		if (seqs.length(i) > seqs.length(i - 1))
			throw runtime_error("Seed index requires a block sorted by decreasing sequence length.");
}

// Delta encodes the sorted pivot positions into chunks of CHUNK_ENTRIES entries and
// writes them out. Each chunk starts with an absolute position and can therefore be
// decoded independently of the others.
static void write_positions(const PtrVector<ExtractCallback>& v, const string& file_name, LinIndex::Header& header) {
	vector<uint64_t> chunk_offset;
	vector<char> data;
	uint64_t entries = 0, prev = 0, n = 0;
	for (size_t i = 0; i < v.size(); ++i)
		for (const uint64_t pos : v[i].out) {
			if (entries % LinIndex::CHUNK_ENTRIES == 0) {
				chunk_offset.push_back(n);
				prev = 0;
			}
			else if (pos <= prev)
				throw runtime_error("Seed positions are not sorted.");
			// The longest varint encoding of a 64 bit value takes 10 bytes.
			if (n + 10 > (uint64_t)data.size())
				data.resize((size_t)std::max<uint64_t>((uint64_t)data.size() * 2, n + MEGABYTES));
			n = (uint64_t)(write_varuint64(pos - prev, data.data() + n) - data.data());
			prev = pos;
			++entries;
		}

	header.entry_count = entries;
	header.data_size = n;
	File out(file_name, "wb");
	out.write(header);
	out.write(chunk_offset.data(), chunk_offset.size() * sizeof(uint64_t));
	out.write(data.data(), n);
	out.close();
	*log_stream << "Seed index: file size=" << sizeof(LinIndex::Header) + chunk_offset.size() * sizeof(uint64_t) + n
		<< " bytes per entry=" << (entries ? (double)n / entries : 0.0) << endl;
}

void build_lin_index(Block& block, const string& file_name, const Config& cfg, int threads) {
	if (shapes.count() != 1)
		throw runtime_error("Seed index is only supported for a single seed shape.");
	check_seed_encoding(cfg);
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
	unique_ptr<atomic<uint64_t>[]> table(alloc_table(table_size, threads));

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

	timer.go("Extracting seed positions");
	PtrVector<ExtractCallback> v;
	for (int i = 0; i < threads; ++i)
		v.push_back(new ExtractCallback(table.get(), table_size, seqs, shapes[0], cfg.seed_complexity_cut));
	enum_seeds(block, v, &no_filter, enum_cfg);
	table.reset();

	timer.go("Writing seed positions");
	LinIndex::Header header;
	header.magic = LinIndex::MAGIC;
	header.version = LinIndex::VERSION;
	header.shape_count = (uint32_t)shapes.count();
	header.seq_count = (uint64_t)seqs.size();
	header.raw_len = (uint64_t)seqs.raw_len();
	header.table_size = table_size;
	write_positions(v, file_name, header);
	timer.finish();
}

LinIndex::LinIndex(const string& file_name, const Block& block, const Config& cfg, int threads) {
	if (shapes.count() != 1)
		throw runtime_error("Seed index is only supported for a single seed shape.");
	check_seed_encoding(cfg);
	threads = std::max(threads, 1);
	const SequenceSet& seqs = block.seqs();

	TaskTimer timer("Loading seed positions");
	vector<uint64_t> chunk_offset;
	vector<char> data;
	{
		File in(file_name, "rb");
		in.read(header_);
		if (header_.magic != MAGIC)
			throw runtime_error("Invalid seed index file: " + file_name);
		if (header_.version != VERSION)
			throw runtime_error("Invalid seed index file version: " + file_name);
		if (header_.shape_count != (uint32_t)shapes.count())
			throw runtime_error("Seed index has a different number of shapes: " + file_name);
		if (header_.seq_count != (uint64_t)seqs.size() || header_.raw_len != (uint64_t)seqs.raw_len())
			throw runtime_error("Seed index does not match the block: " + file_name);
		// Guarantees that a lookup of a seed that is not in the index terminates on a
		// blank slot instead of probing the table forever.
		if (header_.entry_count >= header_.table_size)
			throw runtime_error("Invalid seed index file: " + file_name);
		chunk_offset.resize((header_.entry_count + CHUNK_ENTRIES - 1) / CHUNK_ENTRIES);
		in.read(chunk_offset.data(), chunk_offset.size() * sizeof(uint64_t));
		data.resize(header_.data_size);
		in.read(data.data(), data.size());
		in.close();
	}

	timer.go("Allocating seed index");
	table_size_ = header_.table_size;
	table_ = alloc_table(table_size_, threads);

	timer.go("Rebuilding seed index");
	{
		const uint64_t chunk_count = (uint64_t)chunk_offset.size(), entry_count = header_.entry_count, raw_len = header_.raw_len;
		atomic<uint64_t> next(0);
		vector<std::thread> workers;
		for (int i = 0; i < threads; ++i)
			workers.emplace_back([&] {
				const Shape& shape = shapes[0];
				uint64_t key;
				for (uint64_t c = next++; c < chunk_count; c = next++) {
					const uint64_t begin = c * CHUNK_ENTRIES, n = std::min(CHUNK_ENTRIES, entry_count - begin);
					const char* p = data.data() + chunk_offset[c];
					uint64_t pos = 0;
					for (uint64_t j = 0; j < n; ++j) {
						const auto d = read_varuint64(p);
						pos += d.first;
						p = d.second;
						if (pos >= raw_len)
							throw runtime_error("Invalid seed position in file " + file_name);
						if (seed_at(seqs.data(pos), shape, key))
							insert(table_.get(), table_size_, key, pos);
					}
				}
				});
		for (auto& t : workers)
			t.join();
	}
	timer.finish();
}

LinIndex::~LinIndex() {
}

}
