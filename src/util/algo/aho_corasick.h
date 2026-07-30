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
//
// aho_corasick.hpp
//
// Aho–Corasick automaton mapping pattern strings to lists of 32-bit
// integers, templated on the alphabet size:
//
//     AhoCorasick<>     — full `char` alphabet (256), any byte allowed
//     AhoCorasick<128>  — 7-bit ASCII: half the goto-table footprint
//     AhoCorasick<4>    — e.g. remapped DNA {0,1,2,3}: 32 bytes/node instead of 1 KiB
//
// The contract for AhoCorasick<N>: every byte of every PATTERN must be < N
// (Builder::add throws otherwise). TEXT bytes >= N are allowed and handled
// safely — no pattern can span them, so the scanner just resets to the root
// (and find() reports "absent"). For N == 256 that range check compiles away.
//
// Split into a transient Builder and an immutable, frozen AhoCorasick:
//
//     AhoCorasick<128>::Builder b;
//     b.add("she", {2, 20});
//     b.add("he", 1);
//     AhoCorasick<128> ac = b.build();   // builder can now be discarded
//
// Both the builder scratch and the frozen buffer are allocated from a
// std::pmr::memory_resource, which defaults to the default resource but can
// be a caller-supplied arena:
//
//     AhoCorasick<128>::Builder b(&pool);   // build() output lives in pool too
//
//     auto vals = ac.find("she");        // exact map-style lookup
//     ac.scan(text, [](const auto& m) { ... });  // all occurrences
//
// Memory layout of the frozen automaton — ONE contiguous int32 buffer:
//
//     +--------+----------------------+------------------+---------------+
//     | header |     goto table       |  node metadata   |  value pool   |
//     | 3 ints |     n * N ints       |   n * 4 ints     | total_values  |
//     +--------+----------------------+------------------+---------------+
//      [N,n,V]   row-major, O(1) per    per node:          all int32
//                input byte; bit 31     {dict, depth,      values, packed
//                of an entry = "target  val_begin,         back-to-back in
//                node emits matches"    val_count}         node order
//
//   * Zero per-node allocations; the whole structure is a single heap block.
//   * Nodes are renumbered in BFS order during build(), so the hot upper
//     levels of the trie occupy adjacent goto rows.
//   * The buffer is position-independent (indices only, no pointers): it can
//     be memcpy'd, written to disk, or mmap'd and rehydrated via from_blob().
//     The alphabet size is stored in the header, so a blob cannot be loaded
//     into a mismatched instantiation.
//   * The emit bit lets scan() touch ONLY the goto table for input bytes that
//     produce no match — one 4-byte load per byte of text in the common case.
//
// Complexity: build O(total_pattern_chars * N); scan O(|text| + matches);
// find O(|key|). Footprint: (N + 4) * 4 bytes per node plus the values.
// The dense goto table is the size-for-speed trade; if total size ever
// matters more than scan speed, replace it with a double-array trie — the
// rest of the layout stays the same.
//
// Notes:
//   * Bytes are `unsigned char`: with N == 256, '\0' and non-ASCII bytes are
//     fine anywhere.
//   * Empty patterns are rejected. Re-adding a pattern appends to its list.
//   * The frozen automaton is immutable and safe for concurrent reads.
//
// Requires C++17.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>
#include "../memory/memory_resource.h"

template <unsigned Alphabet = 256>
class AhoCorasick {
    static_assert(Alphabet >= 1 && Alphabet <= 256,
                  "Alphabet must be in [1, 256]");

public:
    using value_type = std::int32_t;   // the "32-bit ints" stored per pattern
    using index_type = std::int32_t;   // node index (max ~2^31 - 1 nodes)

    static constexpr unsigned kAlphabet = Alphabet;

    // Non-owning view into the value pool (the frozen automaton outlives it).
    struct ValueSpan {
        const value_type* ptr = nullptr;
        std::size_t len = 0;

        const value_type* begin() const { return ptr; }
        const value_type* end() const { return ptr + len; }
        std::size_t size() const { return len; }
        bool empty() const { return len == 0; }
        value_type operator[](std::size_t i) const { return ptr[i]; }
    };

    // One occurrence of a pattern inside the scanned text.
    struct Match {
        std::size_t begin;   // byte offset where the pattern starts
        std::size_t length;  // pattern length in bytes
        ValueSpan values;    // the ints mapped to that pattern

        std::size_t end() const { return begin + length; }
    };

    // ---- Builder: temporary, mutation-friendly trie ---------------------
    class Builder {
    public:
        // Every allocation made while building — and the buffer of the frozen
        // automaton returned by build() — comes from `mr`.
        explicit Builder(std::pmr::memory_resource* mr = std::pmr::get_default_resource()) :
            mr_(mr),
            nodes_(mr),
            edges_(mr),
            values_(mr)
        {
            nodes_.push_back(Node());  // node 0 = root
            root_children_.fill(-1);
        }

        // Append `value` to the list mapped to `pattern`.
        void add(std::string_view pattern, value_type value) {
            append_value(insert_path(pattern), value);
        }

        // Append several values to the list mapped to `pattern`.
        void add(std::string_view pattern, const std::vector<value_type>& values) {
            const index_type s = insert_path(pattern);
            for (value_type v : values)
                append_value(s, v);
        }

        // ---- incremental insertion --------------------------------------
        //
        // Callers that emit patterns sharing long prefixes (a DFS over a
        // neighborhood, say) can walk the trie once per prefix instead of
        // once per pattern, which is what add() would do:
        //
        //     descend(descend(root(), 's'), 'h')   // 'sh' walked once
        //
        index_type root() const { return 0; }

        // The child of `node` along `c`, created if it does not exist yet.
        index_type descend(index_type node, unsigned char c) {
            if (c >= kAlphabet)
                throw std::invalid_argument(
                    "AhoCorasick: pattern byte >= Alphabet");
            // The root is walked once per pattern and is the one node whose
            // fanout is the whole alphabet, so its children are also indexed
            // directly. Every other node keeps just the list.
            if (node == 0 && root_children_[c] != -1)
                return root_children_[c];
            // One walk locates the child and, failing that, the position to
            // splice a new edge into: the list stays sorted by character.
            index_type prev = -1, e = nodes_[node].first_edge;
            for (; e != -1 && edges_[e].c < c; prev = e, e = edges_[e].next);
            if (e != -1 && edges_[e].c == c)
                return edges_[e].child;
            const index_type v = static_cast<index_type>(nodes_.size());
            Node child_node;
            child_node.depth = nodes_[node].depth + 1;
            nodes_.push_back(child_node);
            const index_type ne = static_cast<index_type>(edges_.size());
            edges_.push_back(Edge{ v, e, c });
            if (prev == -1)
                nodes_[node].first_edge = ne;
            else
                edges_[prev].next = ne;
            if (node == 0)
                root_children_[c] = v;
            return v;
        }

        // Map `value` to the pattern spelled by `node`.
        void add_value(index_type node, value_type value) {
            append_value(node, value);
        }

        // Freeze into the compact single-buffer automaton by BFS-renumbering
        // the trie straight into the flat buffer. The builder is left
        // untouched (build again after more add()s if you like).
        AhoCorasick build() const {
            const std::size_t n = nodes_.size();
            const std::size_t total_values = values_.size();
            std::size_t patterns = 0;
            for (const Node& nd : nodes_)
                if (nd.value_count > 0) ++patterns;

            AhoCorasick ac(mr_);
            ac.n_ = static_cast<index_type>(n);
            ac.pattern_count_ = patterns;
            ac.blob_.assign(kHeaderInts + n * (kAlphabet + kMetaInts) + total_values, 0);
            ac.blob_[0] = static_cast<std::int32_t>(kAlphabet);
            ac.blob_[1] = static_cast<std::int32_t>(n);
            ac.blob_[2] = static_cast<std::int32_t>(total_values);

            std::int32_t* g = ac.rows();
            value_type* pool = ac.pool();

            std::pmr::vector<index_type> fail(n, mr_);  // indexed by NEW ids; build-time only
            std::pmr::vector<std::pair<index_type, index_type>> bfs(mr_);  // (old id, new id)
            bfs.reserve(n);
            std::size_t bfs_head = 0;
            index_type next_new = 1;
            std::size_t pool_pos = 0;

            // Root (old 0 == new 0): never a pattern end (empty patterns rejected).
            std::int32_t* mr = ac.meta(0);
            mr[kDict] = -1;
            mr[kDepth] = 0;  // kBeg/kCnt stay 0

            // Depth-1 nodes fail to the root; missing root edges loop on the
            // root (rows are zero-initialized, and root == node 0). Builder
            // edge lists are sorted by character, so walking one is the same
            // as scanning the alphabet in order.
            for (index_type e = nodes_[0].first_edge; e != -1; e = edges_[e].next) {
                const index_type v_new = next_new++;
                fail[v_new] = 0;
                g[edges_[e].c] = v_new;
                bfs.emplace_back(edges_[e].child, v_new);
            }

            // FIFO order == new-id order, and fail(u) < u in new ids, so a
            // node's fail row/metadata and its own metadata slot are always
            // ready in time (this also makes the value pool come out sorted
            // by new id).
            while (bfs_head < bfs.size()) {
                const auto [u_old, u_new] = bfs[bfs_head++];
                const Node& nd = nodes_[u_old];
                const index_type f = fail[u_new];

                std::int32_t* mu = ac.meta(u_new);
                const std::int32_t* mf = ac.meta(f);
                mu[kDict] = mf[kCnt] > 0 ? f : mf[kDict];
                mu[kDepth] = static_cast<std::int32_t>(nd.depth);
                mu[kBeg] = static_cast<std::int32_t>(pool_pos);
                mu[kCnt] = nd.value_count;
                for (index_type vi = nd.first_value; vi != -1; vi = values_[vi].next)
                    pool[pool_pos++] = values_[vi].value;

                // Every entry falls back along the suffix link except the few
                // that are real edges, so take the fail row wholesale and
                // then overwrite those. Fanout is typically a handful, while
                // the row is kAlphabet wide.
                std::int32_t* ru = g + std::size_t(u_new) * kAlphabet;
                const std::int32_t* rf = g + std::size_t(f) * kAlphabet;
                std::memcpy(ru, rf, kAlphabet * sizeof(std::int32_t));
                for (index_type e = nd.first_edge; e != -1; e = edges_[e].next) {
                    const unsigned c = edges_[e].c;
                    const index_type v_new = next_new++;
                    fail[v_new] = rf[c];
                    ru[c] = v_new;
                    bfs.emplace_back(edges_[e].child, v_new);
                }
            }

            // Tag every goto entry whose target produces at least one match,
            // so the scan hot loop can skip all metadata loads for
            // non-matching bytes.
            std::pmr::vector<char> emits(n, mr_);
            for (index_type v = 0; v < static_cast<index_type>(n); ++v) {
                const std::int32_t* mv = ac.meta(v);
                emits[std::size_t(v)] = mv[kCnt] > 0 || mv[kDict] != -1;
            }
            for (std::size_t e = 0; e < n * kAlphabet; ++e)
                if (emits[std::size_t(g[e])])
                    g[e] = static_cast<std::int32_t>(
                        static_cast<std::uint32_t>(g[e]) | kEmitBit);

            return ac;
        }

    private:
        // Child edges and values are singly-linked lists threaded through two
        // flat arrays rather than a vector per node: the whole trie is three
        // growing buffers, so a trie of any size costs a handful of
        // allocations from the resource.
        struct Node {
            index_type first_edge = -1;   // head of the child edge list
            index_type first_value = -1;  // head of the value list (insertion order)
            index_type last_value = -1;
            index_type value_count = 0;
            std::uint32_t depth = 0;
        };
        struct Edge {
            index_type child;
            index_type next;
            unsigned char c;
        };
        struct Val {
            value_type value;
            index_type next;
        };

        void append_value(index_type node, value_type value) {
            const index_type i = static_cast<index_type>(values_.size());
            values_.push_back(Val{ value, -1 });
            Node& nd = nodes_[node];
            if (nd.last_value == -1)
                nd.first_value = i;
            else
                values_[nd.last_value].next = i;
            nd.last_value = i;
            ++nd.value_count;
        }

        index_type insert_path(std::string_view pattern) {
            if (pattern.empty())
                throw std::invalid_argument(
                    "AhoCorasick: empty patterns are not supported");
            index_type s = 0;
            for (char ch : pattern)
                s = descend(s, static_cast<unsigned char>(ch));
            return s;
        }

        std::pmr::memory_resource* mr_;
        std::pmr::vector<Node> nodes_;
        std::pmr::vector<Edge> edges_;
        std::pmr::vector<Val> values_;
        std::array<index_type, kAlphabet> root_children_;  // direct index, depth 1
    };

    // Empty automaton: matches nothing, contains nothing. `mr` is the resource
    // the frozen buffer will be allocated from.
    explicit AhoCorasick(std::pmr::memory_resource* mr = std::pmr::get_default_resource()) :
        blob_(mr)
    {}

    // ---- exact lookup (map-style) ---------------------------------------

    // Values mapped to exactly `key`; empty span if `key` was never added.
    //
    // Correctness of the depth check: after consuming a prefix p, the state
    // is the trie node spelling the LONGEST SUFFIX of p that exists in the
    // trie. Its depth equals |p| iff p itself is a trie node, so a final
    // depth of |key| means we followed real edges the whole way.
    ValueSpan find(std::string_view key) const {
        if (n_ == 0) return {};
        const std::int32_t* g = rows();
        index_type s = 0;
        for (char ch : key) {
            const auto c = static_cast<unsigned char>(ch);
            if constexpr (kAlphabet < 256) {
                if (c >= kAlphabet) return {};  // byte can't occur in any pattern
            }
            const auto e = static_cast<std::uint32_t>(
                g[std::size_t(s) * kAlphabet + c]);
            s = static_cast<index_type>(e & kIndexMask);
        }
        const std::int32_t* m = meta(s);
        if (m[kCnt] > 0 &&
            static_cast<std::size_t>(m[kDepth]) == key.size())
            return {pool() + m[kBeg], static_cast<std::size_t>(m[kCnt])};
        return {};
    }

    bool contains(std::string_view key) const { return !find(key).empty(); }

    // ---- Aho–Corasick text scan -------------------------------------------

    // Invoke `on_match(const Match&)` for every occurrence of every pattern,
    // in order of increasing match end. Overlapping/nested matches included.
    // Text bytes >= Alphabet simply reset the automaton (no pattern contains
    // them, so no match can span them).
    template <typename Callback>
    void scan(std::string_view text, Callback&& on_match) const {
        if (n_ == 0) return;
        const std::int32_t* g = rows();
        const std::int32_t* mt = meta(0);
        const value_type* vp = pool();

        index_type s = 0;
        for (std::size_t i = 0; i < text.size(); ++i) {
            const auto c = static_cast<unsigned char>(text[i]);
            if constexpr (kAlphabet < 256) {
                if (c >= kAlphabet) { s = 0; continue; }
            }
            const auto e = static_cast<std::uint32_t>(
                g[std::size_t(s) * kAlphabet + c]);
            s = static_cast<index_type>(e & kIndexMask);
            if (!(e & kEmitBit)) continue;  // hot path: goto table only

            // Report this node if it ends a pattern, then every shorter
            // pattern ending here, via dictionary-suffix (dict) links.
            const std::int32_t* ms = mt + std::size_t(s) * kMetaInts;
            for (index_type t = ms[kCnt] > 0 ? s : ms[kDict]; t != -1;) {
                const std::int32_t* mn = mt + std::size_t(t) * kMetaInts;
                on_match(Match{
                    i + 1 - static_cast<std::size_t>(mn[kDepth]),
                    static_cast<std::size_t>(mn[kDepth]),
                    ValueSpan{vp + mn[kBeg], static_cast<std::size_t>(mn[kCnt])}});
                t = mn[kDict];
            }
        }
    }

    // Convenience overload: collect all matches into a vector.
    std::vector<Match> scan(std::string_view text) const {
        std::vector<Match> out;
        scan(text, [&](const Match& m) { out.push_back(m); });
        return out;
    }

    // ---- introspection / persistence --------------------------------------

    std::size_t node_count() const { return static_cast<std::size_t>(n_); }
    std::size_t pattern_count() const { return pattern_count_; }
    std::size_t memory_bytes() const { return blob_.size() * sizeof(std::int32_t); }

    // The entire automaton as one position-independent int32 buffer.
    const std::pmr::vector<std::int32_t>& blob() const { return blob_; }

    // Rehydrate from a buffer previously obtained via blob() (e.g. read from
    // disk). Rejects blobs built for a different Alphabet. Only structural
    // size checks are performed; feed it trusted data.
    static AhoCorasick from_blob(std::pmr::vector<std::int32_t> blob) {
        if (blob.size() < kHeaderInts)
            throw std::invalid_argument("AhoCorasick::from_blob: truncated");
        if (blob[0] != static_cast<std::int32_t>(kAlphabet))
            throw std::invalid_argument(
                "AhoCorasick::from_blob: alphabet size mismatch");
        const auto n = blob[1];
        const auto values = blob[2];
        if (n < 1 || values < 0 ||
            blob.size() != kHeaderInts +
                               std::size_t(n) * (kAlphabet + kMetaInts) +
                               std::size_t(values))
            throw std::invalid_argument("AhoCorasick::from_blob: bad size");
        AhoCorasick ac(std::move(blob));
        ac.n_ = n;
        for (index_type v = 0; v < n; ++v)
            if (ac.meta(v)[kCnt] > 0) ++ac.pattern_count_;
        return ac;
    }

private:
    // Adopts an existing buffer (and the resource it was allocated from).
    explicit AhoCorasick(std::pmr::vector<std::int32_t>&& blob) :
        blob_(std::move(blob))
    {}

    static constexpr std::size_t kHeaderInts = 3;   // [alphabet, node_count, value_count]
    static constexpr std::size_t kMetaInts = 4;     // ints per node metadata
    // Metadata slots per node:
    static constexpr std::size_t kDict = 0;   // nearest suffix ending a pattern, -1 if none
    static constexpr std::size_t kDepth = 1;  // length of the string this node spells
    static constexpr std::size_t kBeg = 2;    // offset of this node's values in the pool
    static constexpr std::size_t kCnt = 3;    // number of values (0 => not a pattern end)
    // Goto entries: low 31 bits = target node, bit 31 = target emits matches.
    static constexpr std::uint32_t kEmitBit = 0x8000'0000u;
    static constexpr std::uint32_t kIndexMask = 0x7FFF'FFFFu;

    // Section accessors (offsets are pure functions of n_, so the default
    // copy/move operations remain valid).
    const std::int32_t* rows() const { return blob_.data() + kHeaderInts; }
    std::int32_t* rows() { return blob_.data() + kHeaderInts; }
    const std::int32_t* meta(index_type node) const {
        return rows() + std::size_t(n_) * kAlphabet + std::size_t(node) * kMetaInts;
    }
    std::int32_t* meta(index_type node) {
        return rows() + std::size_t(n_) * kAlphabet + std::size_t(node) * kMetaInts;
    }
    const value_type* pool() const { return meta(n_); }
    value_type* pool() { return meta(n_); }

    std::pmr::vector<std::int32_t> blob_;  // the single allocation
    index_type n_ = 0;                // node count (0 => empty/default-constructed)
    std::size_t pattern_count_ = 0;
};