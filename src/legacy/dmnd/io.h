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

#include "util/algo/varint.h"
#include "util/io/file.h"

/*static inline void write_varint(Serializer& s, int32_t x) {
	char buf[5];
	char* end = write_varuint32(x, buf);
	s.write_raw(buf, end - buf);
}*/

static inline void write_varint(File& s, int32_t x) {
	char buf[5];
	char* end = write_varuint32(x, buf);
	s.write(buf, end - buf);
}

/*/template<typename It>
static inline void serialize_varint(Serializer& s, It begin, It end) {
	for (It i = begin; i != end; ++i)
		write_varint(s, *i);
}*/

template<typename It>
static inline void serialize_varint(File& s, It begin, It end) {
	for (It i = begin; i != end; ++i)
		write_varint(s, *i);
}

/*template<typename It>
static inline void serialize(Serializer& s, It begin, It end) {
	for (It i = begin; i != end; ++i)
		s << *i;
}

static inline void serialize(Serializer& s, const std::set<int32_t>& v) {
	write_varint(s, (int32_t)v.size());
	serialize_varint(s, v.cbegin(), v.cend());
}

static inline void serialize(Serializer& s, const std::vector<int32_t>& v) {
	s.write((uint32_t)v.size());
	serialize(s, v.cbegin(), v.cend());
}

static inline void serialize(Serializer& s, const std::vector<std::string>& v) {
	s.write((uint32_t)v.size());
	serialize(s, v.cbegin(), v.cend());
}*/

/*template<typename It>
static inline void serialize(File& s, It begin, It end) {
	for (It i = begin; i != end; ++i)
		s << *i;
}*/

static inline void serialize(File& s, const std::set<int32_t>& v) {
	write_varint(s, (int32_t)v.size());
	serialize_varint(s, v.cbegin(), v.cend());
}

static inline void serialize(File& s, const std::vector<int32_t>& v) {
	const uint32_t size = big_endian_byteswap((uint32_t)v.size());
	s.write(size);
	for (int32_t i : v) {
		i = big_endian_byteswap(i);
		s.write(i);
	}
}

static inline void serialize(File& f, const std::vector<std::string>& v) {
	f.write(big_endian_byteswap((uint32_t)v.size()));
	for (const std::string& s : v)
		f.write_c_str(s.c_str());
}


/*template<typename T1, typename T2>
void serialize(Serializer& s, const std::pair<T1, T2>& p) {
	s << p.first << p.second;
}*/

static inline void serialize(File& f, const std::pair<uint64_t, int32_t>& p) {
	f.write(big_endian_byteswap(p.first));
	f.write(big_endian_byteswap(p.second));
}

static inline void serialize(File& f, const std::pair<std::string, OId>& p) {
	f.write_c_str(p.first.c_str());
	f.write(big_endian_byteswap(p.second));
}


static inline void serialize(File& f, const std::pair<std::string, int32_t>& p) {
	f.write_c_str(p.first.c_str());
	f.write(big_endian_byteswap(p.second));
}

NODISCARD static inline bool deserialize(File& f, std::pair<std::string, OId>& p) {
	f.read_c_str(p.first);
	if (f.read_max(&p.second, sizeof(OId)) != sizeof(OId))
		return false;
	p.second = big_endian_byteswap(p.second);
	return true;
}

NODISCARD static inline bool deserialize(File& f, std::pair<std::string, int32_t>& p) {
	f.read_c_str(p.first);
	if (f.read_max(&p.second, sizeof(int32_t)) != sizeof(int32_t))
		return false;
	p.second = big_endian_byteswap(p.second);
	return true;
}

NODISCARD static inline bool deserialize(File& f, std::pair<uint64_t, int32_t>& p) {
	if (f.read_max(&p.first, sizeof(uint64_t)) != sizeof(uint64_t))
		return false;
	if (f.read_max(&p.second, sizeof(int32_t)) != sizeof(int32_t))
		return false;
	p.first = big_endian_byteswap(p.first);
	p.second = big_endian_byteswap(p.second);
	return true;
}

/*static inline void deserialize(Deserializer& d, std::vector<std::string>& out) {
	uint32_t n;
	d >> n;
	out.clear();
	out.reserve(n);
	std::string s;
	for (uint32_t i = 0; i < n; ++i) {
		d >> s;
		out.push_back(std::move(s));
	}
}

static inline void deserialize(Deserializer& d, std::vector<std::int32_t>& out) {
	uint32_t n;
	d >> n;
	out.clear();
	out.reserve(n);
	int32_t x;
	for (uint32_t i = 0; i < n; ++i) {
		d >> x;
		out.push_back(x);
	}
}*/

static inline void deserialize(File& d, std::vector<std::string>& out) {
	uint32_t n;
	d.read(n);
	n = big_endian_byteswap(n);
	out.clear();
	out.reserve(n);
	std::string s;
	for (uint32_t i = 0; i < n; ++i) {
		d.read_c_str(s);
		out.push_back(std::move(s));
	}
}

static inline void deserialize(File& d, std::vector<std::int32_t>& out) {
	uint32_t n;
	d.read(n);
	n = big_endian_byteswap(n);
	out.clear();
	out.reserve(n);
	int32_t x;
	for (uint32_t i = 0; i < n; ++i) {
		d.read(x);
		x = big_endian_byteswap(x);
		out.push_back(x);
	}
}

/*template<typename T1, typename T2>
void deserialize(Deserializer& s, std::pair<T1, T2>& out) {
	s >> out.first >> out.second;
}*/