/****
DIAMOND protein aligner
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
#include <cstdio>
#include <fstream>
#include <memory>
#include "multinode.h"
#include "basic/shape_config.h"
#include "data/block/block.h"
#include "data/fasta/fasta_file.h"
#include "masking/masking.h"
#include "search/lin_index/lin_index.h"
#include "search/search.h"
#include "util/io/file.h"
#include "util/log_stream.h"
#include "util/parallel/multiprocessing.h"
#include "util/sequence/sequence.h"
#include "util/text_buffer.h"

using std::runtime_error;
using std::string;
using std::unique_ptr;
using std::vector;

static int shape_count(Sensitivity sens) {
	const vector<string>& codes = config.shape_mask.empty() ? Search::shape_codes.at(sens) : config.shape_mask;
	const int n = (int)codes.size();
	return config.shapes == 0 ? n : std::min((int)config.shapes, n);
}

bool use_lin_index(const Job& job) {
	if (!config.seed_index || !job.is_linear_round() || config.mutual_cover.present())
		return false;
	return shape_count(config.sensitivity) == 1;
}

string lin_index_file(const string& volume_path) {
	return volume_path + ".seedidx";
}

static bool length_sorted(const SequenceSet& seqs) {
	const BlockId n = seqs.size();
	for (BlockId i = 1; i < n; ++i)
		if (seqs.length(i) > seqs.length(i - 1))
			return false;
	return true;
}

// Replaces the volume file by the given block in FASTA format.
static void write_block(const Block& block, const string& path, int64_t worker_id) {
	const string tmp_path = path + "." + std::to_string(worker_id) + ".sorted";
	{
		std::ofstream out(tmp_path, std::ios::binary);
		if (!out)
			throw runtime_error("Error opening file " + tmp_path);
		TextBuffer buf;
		const BlockId n = block.seqs().size();
		for (BlockId i = 0; i < n; ++i) {
			Util::Seq::format(block.seqs()[i], block.ids()[i], nullptr, buf, "fasta", amino_acid_traits);
			if (buf.size() >= (int64_t)MEGABYTES) {
				out.write(buf.data(), buf.size());
				buf.clear();
			}
		}
		out.write(buf.data(), buf.size());
		out.close();
		if (!out)
			throw runtime_error("Error writing file " + tmp_path);
	}
	std::remove(path.c_str());
	if (std::rename(tmp_path.c_str(), path.c_str()) != 0)
		throw runtime_error("Error renaming file " + tmp_path);
}

void build_lin_indices(Job& job, const VolumedFile& volumes) {
	const string base_dir = job.base_dir() + PATH_SEPARATOR + "seed_index" + PATH_SEPARATOR;
	job.make_temp_dir(base_dir);
	Atomic q(base_dir + "queue", job), finished(base_dir + "finished", job);
	const int64_t n = (int64_t)volumes.size();
	int64_t v;

	unique_ptr<vector<BitVector>> no_seed_hits;
	Search::Config cfg(no_seed_hits);
	Search::setup_search(config.sensitivity, cfg);
	if (shapes.count() != 1)
		throw runtime_error("Seed index is only supported for a single seed shape.");

	while (job.goon() && (v = q.fetch_add(), v < n)) {
		const string index_file = lin_index_file(volumes[v].path);
		if (file_exists(index_file)) {
			finished.fetch_add();
			continue;
		}
		job.log("Building seed index. Block=%lli/%lli", v + 1, n);
		TaskTimer timer("Loading block");
		unique_ptr<SequenceFile> file(new FastaFile({ volumes[v].path }, SequenceFile::Flags::SEQS | SequenceFile::Flags::TITLES | SequenceFile::Flags::NEED_LETTER_COUNT, amino_acid_traits));
		unique_ptr<Block> block(file->load_seqs(INT64_MAX));
		file->close();
		timer.finish();
		if (!length_sorted(block->seqs())) {
			// The positions stored in the index refer to the order of the block on
			// disk, and the pivot of a seed is the entry with the smallest position.
			// Both only work out if the block is sorted by decreasing length, which
			// the search would otherwise have to redo for every block combination.
			timer.go("Length sorting block");
			block.reset(block->length_sorted(config.threads_));
			timer.finish();
			timer.go("Writing length sorted block");
			write_block(*block, volumes[v].path, job.worker_id());
			timer.finish();
		}
		if (cfg.query_masking != MaskingAlgo::NONE) {
			timer.go("Masking block");
			mask_seqs(block->seqs(), Masking::get(), true, cfg.query_masking);
			timer.finish();
		}
		// The index is written under a temporary name and renamed, so that a
		// worker crashing mid-write does not leave a truncated index behind.
		const string tmp_file = index_file + "." + std::to_string(job.worker_id()) + ".tmp";
		Search::build_lin_index(*block, tmp_file, cfg, config.threads_);
		block.reset();
		if (!file_exists(index_file) && std::rename(tmp_file.c_str(), index_file.c_str()) != 0)
			throw runtime_error("Error renaming seed index file " + tmp_file);
		std::remove(tmp_file.c_str());
		finished.fetch_add();
		job.finish_step();
	}
	if (!job.goon())
		return;
	finished.await(n);
	rmdir(base_dir.c_str());
}

void remove_lin_indices(const VolumedFile& volumes) {
	for (const Volume& v : volumes)
		remove_tmp_file(lin_index_file(v.path));
}
