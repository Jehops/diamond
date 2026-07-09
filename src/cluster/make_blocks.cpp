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

#include <inttypes.h>
#include "multinode.h"
#define _REENTRANT
#include "lib/ips4o/ips4o.hpp"

using std::runtime_error;
using std::vector;
using std::pair;
using std::string;
using std::unique_ptr;
using std::tie;
using std::ofstream;
using std::endl;

static bool can_add(uint64_t block_letters, uint64_t block_seqs, uint64_t seq_len, uint64_t block_letter_limit) {
	constexpr uint64_t RAW_LIMIT = (uint64_t(1) << 40) - 1;
	if (seq_len > RAW_LIMIT)
		return false;
	if (block_letters > RAW_LIMIT - seq_len)
		return false;
	const uint64_t raw_len = block_letters + seq_len + block_seqs + 1 + 256;
	if (raw_len > RAW_LIMIT)
		return false;
	if (block_seqs == 0)
		return true;
	if (block_seqs >= std::numeric_limits<BlockId>::max())
		return false;
	return block_letters + seq_len <= block_letter_limit;
}

vector<pair<int, OId>> make_blocks(Job& job, VolumedFile& volumes, vector<unique_ptr<ofstream>>& out, vector<unique_ptr<ofstream>>& acc_out) {
	const bool first_round_linear = job.is_linear_round();
	const string input_parts = job.root_dir() + "input.tsv", input_letters = job.root_dir() + "input_letters.txt";
	job.log("Memory limit = %" PRIu64, job.mem_limit);
	vector<pair<Loc, OId>> lengths;
	uint64_t letters = 0;
	for (vector<Volume>::const_iterator i = volumes.begin(); i != volumes.end(); ++i) {
		job.log("Reading volume %td/%zu", i - volumes.begin() + 1, volumes.size());
		unique_ptr<SequenceFile> volume_file;
		try {
			volume_file.reset(SequenceFile::auto_create({ i->path }, SequenceFile::Flags::NEED_LETTER_COUNT | SequenceFile::Flags::NEED_LENGTH_LOOKUP, amino_acid_traits));
		}
		catch (FormatDetectionError& e) {
			throw runtime_error("Error opening file " + i->path + ": " + e.what());
		}
		string msg = volume_file->open_stats();
		if (volume_file->type() == SequenceFile::Type::DMND)
			job.log("Warning: using legacy loader on .dmnd files");
		else {
			msg.pop_back();
			job.log(msg.c_str());
		}
		letters += volume_file->letters().value();
		const OId n = volume_file->sequence_count().value();
		for (OId i = 0; i < n; ++i)
			lengths.emplace_back(volume_file->seq_length(i), lengths.size());
	}
	ofstream letters_out(input_letters);
	letters_out << letters << endl;
	letters_out << lengths.size() << endl;
	if (!letters_out)
		throw runtime_error("Error writing file " + input_letters);
	letters_out.close();
	job.log("Computing blocks");
	std::ostringstream ss;
	//volumes.set_max_oid(lengths.size() - 1);
	ss << "Sequences in database = " << lengths.size() << endl;
	ss << "Letters in database = " << letters << endl;
	ss << "Database blocks:" << endl;

	double block_gb;
	int index_chunks;
	if (!first_round_linear) {
		block_gb = 1e6;
		index_chunks = 1;
	}
	else
		tie(block_gb, index_chunks) = ::block_size(job.mem_limit, letters, Sensitivity::FAST, true, config.threads_); // TODO take cluster steps into account here
	const uint64_t block_size = gb_to_bytes(block_gb);
	job.log("Block size = %" PRIu64 ", index chunks = %d", block_size, index_chunks);
	ips4o::parallel::sort(lengths.begin(), lengths.end(), std::greater<pair<Loc, OId>>(), config.threads_);
	ofstream idx(input_parts);
	vector<pair<int, OId>> block_mapping(lengths.size());
	int block = 0;
	OId new_oid = 0;
	for (vector<pair<Loc, OId>>::const_iterator i = lengths.begin(); i != lengths.end();) {
		uint64_t block_letters = 0, seqs = 0;
		while (i != lengths.end() && can_add(block_letters, seqs, i->first, block_size)) {
			block_letters += i->first;
			block_mapping[i->second] = { block, new_oid++ };
			++i;
			++seqs;
		}
		if (seqs == 0)
			throw runtime_error("Sequence exceeds supported maximum block size.");
		ss << seqs << '\t' << block_letters << endl;
		const string block_idx = std::to_string(block);
		const string name = job.root_dir() + "input" + block_idx + ".faa";
		out.emplace_back(new ofstream(name));
		acc_out.emplace_back(new ofstream(job.root_dir() + "input" + block_idx + ".tsv"));
		idx << name << '\t' << seqs << endl;
		++block;
	}
	job.log(ss.str().c_str());
	return block_mapping;
}