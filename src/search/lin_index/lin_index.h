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
#include <memory>
#include <string>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include "mio/forward.h"

struct Block;

namespace Search {

struct Config;

/* Seed index of a single, length sorted sequence block, used by the linear
   clustering rounds with unidirectional coverage. For every seed it holds the
   single occurrence in the longest sequence of the block containing that seed
   (the pivot), which is the only entry the linear stage 1 kernels look at.
   The index is built once per block and written to disk, so that it can be
   reused for every block combination the block takes part in as the
   representative (reference) block.

   An entry is a 64 bit open addressing hash table slot holding a 24 bit
   fingerprint of the seed and the 40 bit position of the pivot in the raw
   sequence data of the block. An empty slot is zero, which is never a valid
   entry since position 0 lies in the leading padding of a sequence set. */

struct LinIndex {

	static constexpr uint64_t MAGIC = 0x786469646e696c64ull;
	static constexpr uint32_t VERSION = 1;
	static constexpr int POS_BITS = 40;
	static constexpr uint64_t POS_MASK = (uint64_t(1) << POS_BITS) - 1;
	static constexpr double LOAD_FACTOR = 0.7;

	struct Header {
		uint64_t magic;
		uint32_t version;
		uint32_t shape_count;
		uint64_t seq_count;
		uint64_t raw_len;
		uint64_t table_size;
		uint64_t entry_count;
	};

	LinIndex(const std::string& file_name);
	~LinIndex();

	// Position of the pivot occurrence of the seed in the raw sequence data of
	// the indexed block, 0 if the seed is not contained in the index.
	uint64_t operator()(const uint64_t key) const {
		const uint64_t h = hash(key), f = fingerprint(h);
		uint64_t i = bucket(h, table_size_);
		for (;;) {
			const uint64_t e = table_[i];
			if (e == 0)
				return 0;
			if ((e >> POS_BITS) == f)
				return e & POS_MASK;
			if (++i == table_size_)
				i = 0;
		}
	}

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
	std::unique_ptr<mio::mmap_source> mmap_;
	const uint64_t* table_;
	uint64_t table_size_;

};

// Builds the seed index of a length sorted block and writes it to disk. The block
// has to be hard masked in the same way as at search time.
void build_lin_index(Block& block, const std::string& file_name, const Config& cfg, int threads);

// Scans the member block (cfg.target) against the index of the reference block
// (cfg.query) and writes the resulting seed hits to the hit buffer.
void scan_lin_index(Config& cfg);

}
