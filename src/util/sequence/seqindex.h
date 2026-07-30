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
#include <cstdint>
#include <functional>
#include <vector>
#include "basic/sequence.h"
#include "util/memory/memory_resource.h"
#include "basic/statistics.h"

struct Seqindex {

	Seqindex(Sequence seq, int k, double word_threshold, std::pmr::memory_resource& pool, Statistics& stats);	
	void scan(Sequence seq, std::function<void(Loc, Loc)>& callback);
	~Seqindex();
	int k() const {
		return k_;
	}

	static constexpr int MIN_KMER_LEN = 3;
	static constexpr int MAX_KMER_LEN = 6;

	struct Entry {
		uint32_t code;
		Loc pos;
	};

private:
	
	uint32_t bucket(uint32_t code) const {
		return (code * 2654435761u) >> shift_;
	}

	int k_;
	unsigned shift_;
	uint32_t mod_;
	std::pmr::vector<Entry> entry_;
	std::pmr::vector<int32_t> next_;
	std::pmr::vector<int32_t> head_;

};