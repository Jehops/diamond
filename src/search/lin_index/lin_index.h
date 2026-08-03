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
#include <stdint.h>
#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

struct Block;

namespace Search {

struct Config;

/* Seed index of a single, length sorted sequence block, used by the linear
   clustering rounds with unidirectional coverage. For every seed it holds the
   single occurrence in the longest sequence of the block containing that seed
   (the pivot), which is the only entry the linear stage 1 kernels look at.

   An entry is a 64 bit open addressing hash table slot holding a 24 bit
   fingerprint of the seed and the 40 bit position of the pivot in the raw
   sequence data of the block. An empty slot is zero, which is never a valid
   entry since position 0 lies in the leading padding of a sequence set.

   The hash table itself is never stored. What is written to disk once per block
   is only the sorted list of pivot positions, delta encoded as varints, which is
   an order of magnitude smaller than the table. The table is rebuilt from it in
   memory whenever the block takes part in a block combination as the
   representative (reference) block: the seed of a pivot is recomputed from the
   sequence data at its position, so no keys need to be stored. The positions are
   grouped into chunks of a fixed number of entries, each of which starts with an
   absolute position, so that the file can be decoded by all threads in parallel. */

struct LinIndex {

	static constexpr uint64_t MAGIC = 0x786469646e696c64ull;
	static constexpr uint32_t VERSION = 2;
	static constexpr int POS_BITS = 40;
	static constexpr uint64_t POS_MASK = (uint64_t(1) << POS_BITS) - 1;
	static constexpr double LOAD_FACTOR = 0.7;
	static constexpr uint64_t CHUNK_ENTRIES = 4096;

	struct Header {
		uint64_t magic;
		uint32_t version;
		uint32_t shape_count;
		uint64_t seq_count;
		uint64_t raw_len;
		uint64_t table_size;
		uint64_t entry_count;
		uint64_t data_size;
	};

	// Rebuilds the hash table of the block from its seed position file. The block
	// has to be hard masked in the same way as when the file was written.
	LinIndex(const std::string& file_name, const Block& block, const Config& cfg, int threads);
	~LinIndex();

	// Position of the pivot occurrence of the seed in the raw sequence data of
	// the indexed block, 0 if the seed is not contained in the index.
	uint64_t operator()(const uint64_t key) const {
		return lookup(table_.get(), table_size_, key);
	}

	static uint64_t lookup(const std::atomic<uint64_t>* table, const uint64_t table_size, const uint64_t key) {
		const uint64_t h = hash(key), f = fingerprint(h);
		uint64_t i = bucket(h, table_size);
		for (;;) {
			const uint64_t e = table[i].load(std::memory_order_relaxed);
			if (e == 0)
				return 0;
			if ((e >> POS_BITS) == f)
				return e & POS_MASK;
			if (++i == table_size)
				i = 0;
		}
	}

	// Inserts the pivot of a seed, keeping the smallest position of all entries
	// that map to the same slot. The block is sorted by decreasing sequence
	// length, so the smallest position is the occurrence in the longest sequence.
	// Returns true if a new slot was claimed.
	static bool insert(std::atomic<uint64_t>* table, const uint64_t table_size, const uint64_t key, const uint64_t pos);

	const Header& header() const {
		return header_;
	}

	int64_t size() const {
		return (int64_t)table_size_ * (int64_t)sizeof(uint64_t);
	}

	static uint64_t hash(uint64_t x) {
		// splitmix64 finalizer
		x += 0x9e3779b97f4a7c15ull;
		x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
		x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
		return x ^ (x >> 31);
	}

	// Has to be taken from bits that the bucket index does not use, otherwise it
	// adds no information and entries of distinct seeds get merged.
	static uint64_t fingerprint(const uint64_t h) {
		return std::max(h & ((uint64_t(1) << (64 - POS_BITS)) - 1), uint64_t(1));
	}

	// Maps a hash value to [0, n) without a division (Lemire's fastrange).
	static uint64_t bucket(const uint64_t h, const uint64_t n) {
		return mul_hi(h, n);
	}

	static uint64_t mul_hi(const uint64_t a, const uint64_t b) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
		return __umulh(a, b);
#elif defined(__SIZEOF_INT128__)
		return (uint64_t)(((__uint128_t)a * (__uint128_t)b) >> 64);
#else
		const uint64_t a0 = a & 0xffffffffull, a1 = a >> 32, b0 = b & 0xffffffffull, b1 = b >> 32;
		const uint64_t t = a0 * b0, u = a1 * b0 + (t >> 32), v = a0 * b1 + (u & 0xffffffffull);
		return a1 * b1 + (u >> 32) + (v >> 32);
#endif
	}

private:

	Header header_;
	std::unique_ptr<std::atomic<uint64_t>[]> table_;
	uint64_t table_size_;

};

// Computes the seed positions of a length sorted block and writes them to disk. The
// block has to be hard masked in the same way as at search time.
void build_lin_index(Block& block, const std::string& file_name, const Config& cfg, int threads);

// Scans the member block (cfg.target) against the index of the reference block
// (cfg.query) and writes the resulting seed hits to the hit buffer.
void scan_lin_index(Config& cfg);

}
