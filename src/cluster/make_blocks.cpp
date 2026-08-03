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
#include "file_array.h"
#include "input_buffer.h"
#define _REENTRANT
#include "lib/ips4o/ips4o.hpp"

using std::runtime_error;
using std::vector;
using std::string;
using std::unique_ptr;
using std::tie;
using std::ofstream;
using std::endl;

static const Loc len_partition[255] = {
29,
40,
46,
50,
54,
58,
60,
63,
65,
67,
69,
71,
73,
75,
77,
79,
81,
84,
86,
88,
90,
92,
94,
96,
98,
100,
102,
104,
106,
108,
109,
111,
113,
115,
117,
119,
121,
123,
125,
127,
128,
130,
132,
134,
136,
138,
139,
141,
143,
145,
147,
148,
150,
152,
154,
155,
157,
159,
161,
162,
164,
166,
168,
170,
172,
174,
176,
178,
180,
182,
184,
185,
187,
189,
191,
193,
195,
197,
199,
201,
202,
204,
206,
208,
210,
212,
214,
216,
217,
219,
221,
223,
225,
227,
229,
231,
232,
234,
236,
238,
240,
242,
244,
246,
248,
250,
252,
254,
255,
257,
259,
261,
263,
264,
266,
268,
270,
272,
274,
276,
278,
281,
283,
285,
287,
289,
290,
292,
294,
296,
298,
300,
302,
304,
306,
308,
310,
312,
314,
316,
318,
320,
322,
324,
326,
328,
331,
333,
335,
337,
339,
341,
343,
346,
348,
350,
353,
355,
358,
361,
363,
366,
368,
371,
374,
376,
379,
382,
384,
387,
390,
393,
395,
398,
400,
403,
406,
409,
412,
415,
418,
421,
424,
427,
430,
433,
436,
439,
443,
446,
449,
453,
456,
460,
463,
467,
470,
474,
478,
482,
486,
491,
495,
499,
504,
509,
513,
518,
524,
530,
536,
542,
548,
554,
560,
566,
573,
580,
588,
596,
604,
612,
622,
631,
641,
651,
661,
672,
684,
696,
708,
721,
734,
749,
764,
781,
798,
815,
836,
857,
879,
904,
934,
966,
1006,
1045,
1091,
1150,
1216,
1273,
1392,
1568,
1900,
2769,
4416 };

static uint64_t len_bucket(Loc len)
{
	constexpr uint64_t N = sizeof(len_partition) / sizeof(len_partition[0]);
	auto it = std::lower_bound(std::begin(len_partition), std::end(len_partition), len);
	uint64_t ans = static_cast<uint64_t>(std::distance(std::begin(len_partition), it));
	return ans;
}

struct LengthRecord {
	static constexpr bool POD = true;
	uint64_t length;
	OId oid;
};

static_assert(sizeof(LengthRecord) == sizeof(uint64_t) + sizeof(OId), "LengthRecord must not contain padding");

static void serialize(const LengthRecord& record, CompressedBuffer& buf) {
	buf.write(record.length);
	buf.write(record.oid);
}

static bool greater_length(const LengthRecord& lhs, const LengthRecord& rhs) {
	return lhs.length > rhs.length || (lhs.length == rhs.length && lhs.oid > rhs.oid);
}

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

vector<int> make_blocks(Job& job, VolumedFile& volumes, vector<unique_ptr<ofstream>>& out, vector<unique_ptr<ofstream>>& acc_out) {
	const bool first_round_linear = job.is_linear_round();
	const string input_parts = job.root_dir() + "input.tsv", input_letters = job.root_dir() + "input_letters.txt";
	const string length_dir = job.root_dir() + "length_sorted" + PATH_SEPARATOR;
	job.log("Memory limit = %" PRIu64, job.mem_limit);
	job.make_temp_dir(length_dir);
	FileArray length_files(length_dir, RADIX_COUNT, job.worker_id(), false);
	uint64_t letters = 0, sequence_count = 0;
	{
		BufferArray buffers(length_files, RADIX_COUNT);
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
			for (OId j = 0; j < n; ++j) {
				const uint64_t length = volume_file->seq_length(j);				
				const LengthRecord record{ length, sequence_count++ };
				buffers.write(len_bucket(length), record);
			}
		}
	}
	length_files.close();
	const RadixedTable length_buckets = length_files.buckets(16 - RADIX_BITS);
	ofstream letters_out(input_letters);
	letters_out << letters << endl;
	letters_out << sequence_count << endl;
	if (!letters_out)
		throw runtime_error("Error writing file " + input_letters);
	letters_out.close();
	job.log("Computing blocks");
	std::ostringstream ss;
	//volumes.set_max_oid(sequence_count - 1);
	ss << "Sequences in database = " << sequence_count << endl;
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
	ofstream idx(input_parts);
	vector<int> block_mapping(sequence_count);
	int block = 0;
	uint64_t block_letters = 0, seqs = 0;
	auto finish_block = [&]() {
		if (seqs == 0)
			return;
		ss << seqs << '\t' << block_letters << endl;
		const string block_idx = std::to_string(block);
		const string name = job.root_dir() + "input" + block_idx + ".faa";
		out.emplace_back(new ofstream(name));
		acc_out.emplace_back(new ofstream(job.root_dir() + "input" + block_idx + ".tsv"));
		idx << name << '\t' << seqs << endl;
		++block;
		block_letters = 0;
		seqs = 0;
	};
	for (auto bucket = length_buckets.crbegin(); bucket != length_buckets.crend(); ++bucket) {
		VolumedFile files(*bucket);
		InputBuffer<LengthRecord> lengths(files);
		ips4o::parallel::sort(lengths.begin(), lengths.end(), greater_length, config.threads_);
		for (const LengthRecord& record : lengths) {
			if (!can_add(block_letters, seqs, record.length, block_size))
				finish_block();
			if (!can_add(block_letters, seqs, record.length, block_size))
				throw runtime_error("Sequence exceeds supported maximum block size.");
			block_letters += record.length;
			block_mapping[record.oid] = block;
			++seqs;
		}
		files.remove();
	}
	finish_block();
	rmdir(length_dir);
	job.log(ss.str().c_str());
	return block_mapping;
}