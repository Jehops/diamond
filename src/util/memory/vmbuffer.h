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

//
// Backed directly by virtual memory rather than the heap. A fixed range of
// address space is reserved at construction; pages inside it are committed on
// grow and decommitted on shrink. The base pointer is stable for the lifetime
// of the buffer, so neither operation moves data.
//
// Huge page support (HugePages::Transparent):
//   Linux   — MADV_HUGEPAGE on committed ranges, 2 MB-aligned reservation, and
//             commit/decommit rounded to 2 MB so huge pages are never split.
//   macOS   — unavailable. No THP; VM_FLAGS_SUPERPAGE_SIZE_2MB is x86-only and
//             absent on Apple Silicon (whose 16 KB base pages already cut TLB
//             pressure ~4x). Falls back to base pages.
//   Windows — unavailable *by design*. MEM_LARGE_PAGES requires
//             MEM_RESERVE|MEM_COMMIT in a single call and cannot be partially
//             decommitted, which is incompatible with copy-free shrink.
//             Falls back to base pages.
//
// Requesting huge pages is lazy: the reservation is 2 MB-aligned right away
// (address space is free) but the commit granularity only switches to 2 MB once
// the buffer actually asks for that much. A small buffer therefore costs no more
// than it does with base pages — rounding a 1 KB buffer up to a huge page would
// fault in 2 MB of physical memory for it, which no TLB saving pays back.
//
// A caller that only learns after construction whether the buffer will get big
// enough to be worth it can turn huge pages on then, with
// request_huge_pages().
//
// Call huge_pages_active() to learn what you actually got.
//
// Two reservation modes:
//   fixed — constructed with an explicit max_bytes. That reservation is a hard
//           ceiling; growing past it throws std::bad_alloc. The base pointer
//           never moves.
//   auto  — default-constructed (optionally with a HugePages setting). The
//           reservation is established on first growth and enlarged
//           geometrically when outgrown. Enlarging relocates (one memcpy of the
//           live bytes), so the base pointer is only stable between such
//           events; reserve enough up front and it never happens.
//
// VmVector<T> at the bottom of this file wraps a VmBuffer in a std::vector-like
// interface for trivially copyable element types.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iterator>
#include <new>
#include <type_traits>
#include <utility>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <unistd.h>
#  if defined(MAP_NORESERVE)
#    define PB_MAP_NORESERVE MAP_NORESERVE
#  else
#    define PB_MAP_NORESERVE 0   // macOS/BSD: absent, overcommit is implicit
#  endif
#endif

enum class HugePages {
    Off,          // base pages only
    Transparent   // request THP where supported, silently fall back otherwise
};

namespace detail {

inline std::size_t page_size() noexcept {
#if defined(_WIN32)
    static const std::size_t ps = [] {
        SYSTEM_INFO si;
        ::GetSystemInfo(&si);
        return static_cast<std::size_t>(si.dwPageSize);
    }();
#else
    static const std::size_t ps =
        static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
#endif
    return ps;
}

// Size of a huge page, or 0 if this platform cannot use them here.
inline std::size_t huge_page_size() noexcept {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    return std::size_t(2) * 1024 * 1024;   // PMD size on x86-64 and arm64
#else
    return 0;
#endif
}

inline std::size_t round_up(std::size_t n, std::size_t mult) noexcept {
    return ((n + mult - 1) / mult) * mult;
}

#if !defined(_WIN32)
// Reserve `bytes` of address space whose base is a multiple of `align`.
// mmap gives no alignment guarantee, so over-reserve and trim both ends.
inline void* reserve_aligned(std::size_t bytes, std::size_t align) noexcept {
    const std::size_t slack = (align > page_size()) ? align : 0;
    void* raw = ::mmap(nullptr, bytes + slack, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | PB_MAP_NORESERVE, -1, 0);
    if (raw == MAP_FAILED) return nullptr;
    if (slack == 0) return raw;

    const std::uintptr_t lo      = reinterpret_cast<std::uintptr_t>(raw);
    const std::uintptr_t aligned = (lo + align - 1) & ~std::uintptr_t(align - 1);
    if (aligned > lo)
        ::munmap(raw, aligned - lo);                            // trim head
    const std::uintptr_t end     = aligned + bytes;
    const std::uintptr_t raw_end = lo + bytes + slack;
    if (raw_end > end)
        ::munmap(reinterpret_cast<void*>(end), raw_end - end);  // trim tail
    return reinterpret_cast<void*>(aligned);
}
#endif

inline void* reserve(std::size_t bytes, std::size_t align) noexcept {
#if defined(_WIN32)
    (void)align;   // no THP on Windows, so no alignment constraint to satisfy
    return ::VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS);
#else
    return reserve_aligned(bytes, align);
#endif
}

inline void release(void* base, std::size_t bytes) noexcept {
#if defined(_WIN32)
    (void)bytes;
    ::VirtualFree(base, 0, MEM_RELEASE);
#else
    ::munmap(base, bytes);
#endif
}

inline bool commit(void* addr, std::size_t len, bool huge) noexcept {
    if (len == 0) return true;
#if defined(_WIN32)
    (void)huge;
    return ::VirtualAlloc(addr, len, MEM_COMMIT, PAGE_READWRITE) != nullptr;
#else
    if (::mprotect(addr, len, PROT_READ | PROT_WRITE) != 0) return false;
#  if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (huge) ::madvise(addr, len, MADV_HUGEPAGE);   // advisory; ignore failure
#  else
    (void)huge;
#  endif
    return true;
#endif
}

// Hand physical pages back while keeping the address space reserved.
inline void decommit(void* addr, std::size_t len) noexcept {
    if (len == 0) return;
#if defined(_WIN32)
    ::VirtualFree(addr, len, MEM_DECOMMIT);
#else
    // A fresh anonymous PROT_NONE mapping over the range drops the pages
    // immediately. Behaves identically on Linux and macOS, unlike the
    // MADV_DONTNEED / MADV_FREE split.
    ::mmap(addr, len, PROT_NONE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | PB_MAP_NORESERVE, -1, 0);
#endif
}

}  // namespace detail

class VmBuffer {
public:
    // Auto mode: no address space is held until the first growth.
    VmBuffer() noexcept : gran_(detail::page_size()) {}

    explicit VmBuffer(HugePages hp) noexcept { set_mode(hp); }

    // max_bytes is an address-space reservation, not an allocation; on 64-bit
    // it is nearly free, so be generous. It is the hard ceiling on growth.
    explicit VmBuffer(std::size_t max_bytes, HugePages hp = HugePages::Off) {
        set_mode(hp);
        fixed_    = true;
        // Rounded to the alignment, not the current granularity: the ceiling has
        // to still hold max_bytes after a switch to huge-page granularity.
        reserved_ = detail::round_up(max_bytes, align());
        if (reserved_ == 0) return;

        base_ = static_cast<unsigned char*>(detail::reserve(reserved_, align()));
        if (!base_) {
            reserved_ = 0;
            throw std::bad_alloc();
        }
    }

    ~VmBuffer() { reset(); }

    VmBuffer(VmBuffer&& o) noexcept
        : base_(o.base_), size_(o.size_), committed_(o.committed_),
          reserved_(o.reserved_), gran_(o.gran_), hps_(o.hps_), huge_(o.huge_),
          fixed_(o.fixed_) {
        o.disown();
    }

    VmBuffer& operator=(VmBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            base_ = o.base_;  size_ = o.size_;  committed_ = o.committed_;
            reserved_ = o.reserved_;  gran_ = o.gran_;  huge_ = o.huge_;
            hps_ = o.hps_;  fixed_ = o.fixed_;
            o.disown();
        }
        return *this;
    }

    VmBuffer(const VmBuffer&) = delete;
    VmBuffer& operator=(const VmBuffer&) = delete;

    unsigned char*       data()       noexcept { return base_; }
    const unsigned char* data() const noexcept { return base_; }
    std::size_t size()     const noexcept { return size_; }
    std::size_t capacity() const noexcept { return committed_; }
    std::size_t reserved() const noexcept { return reserved_; }
    bool        empty()    const noexcept { return size_ == 0; }

    // True only once the buffer has grown past a huge page and huge pages have
    // actually been requested from the OS. Whether the kernel honoured that also
    // depends on THP being 'always' or 'madvise' and on fragmentation; check
    // AnonHugePages in /proc/self/smaps to confirm.
    bool        huge_pages_active() const noexcept { return huge_; }
    // What was asked for, which survives a buffer being empty or moved from.
    HugePages   huge_pages_mode()   const noexcept {
        return hps_ ? HugePages::Transparent : HugePages::Off;
    }
    std::size_t granularity()       const noexcept { return gran_; }

    // Switch a buffer constructed with HugePages::Off over to
    // HugePages::Transparent, for callers that only learn how big the buffer
    // will get after it exists. Activation stays as lazy as it is for a buffer
    // constructed that way: the granularity switches on the first commit of a
    // whole huge page. Returns false, leaving the buffer untouched, where the
    // platform has no huge pages or the reservation cannot honour them.
    bool request_huge_pages() {
        const std::size_t hps = detail::huge_page_size();
        if (hps_) return true;      // already requested
        if (hps == 0) return false;  // no huge pages on this platform
        if (base_) {
            // A fixed reservation is a hard ceiling and its base cannot move,
            // so neither the alignment nor the ceiling can be made to hold
            // under huge-page rounding.
            if (fixed_) return false;
            if (reinterpret_cast<std::uintptr_t>(base_) & (hps - 1)) {
                // Reserved before the request and so only base-page aligned;
                // move it now, while it is small, rather than on the growth
                // that activates huge pages.
                hps_ = hps;               // align() picks it up
                relocate(committed_);
                return true;
            }
        }
        hps_ = hps;
        return true;
    }

    unsigned char&       operator[](std::size_t i)       noexcept { return base_[i]; }
    const unsigned char& operator[](std::size_t i) const noexcept { return base_[i]; }

    void reserve_bytes(std::size_t n) {
        if (n <= committed_) return;
        const bool activated = activate_huge(n);
        const std::size_t want = detail::round_up(n, gran_);
        if (want > reserved_) {
            if (fixed_) throw std::bad_alloc();
            relocate(want);
            return;
        }
        // On the switch to huge pages re-advise from the base so the range
        // committed before it is covered too; mprotect over live pages is a
        // no-op, and base_ is huge-page aligned whenever hps_ is set.
        const std::size_t from = activated ? 0 : committed_;
        if (!detail::commit(base_ + from, want - from, huge_))
            throw std::bad_alloc();
        committed_ = want;
    }

    void resize(std::size_t n) {
        if (n > size_) reserve_bytes(n);
        size_ = n;
    }

    // Return whole granules past the live region to the OS. Nothing is copied
    // and data() is unchanged; pointers into [0, size()) stay valid.
    void shrink_to_fit() noexcept {
        const std::size_t want = detail::round_up(size_, gran_);
        if (want >= committed_) return;
        detail::decommit(base_ + want, committed_ - want);
        committed_ = want;
    }

    void clear() noexcept { size_ = 0; }

    void append(const void* src, std::size_t n) {
        reserve_bytes(size_ + n);
        std::memcpy(base_ + size_, src, n);
        size_ += n;
    }

    void reset() noexcept {
        if (base_) detail::release(base_, reserved_);
        base_ = nullptr;
        size_ = committed_ = reserved_ = 0;
        // Back to base pages; a reused buffer re-earns huge-page granularity by
        // growing into it again. The requested mode (hps_) is kept.
        huge_ = false;
        gran_ = detail::page_size();
    }

private:
    // Address space is cheap, relocation is not: reserve well past what is
    // asked for so that an unreserved fill pattern rarely has to copy.
    enum : std::size_t {
        RESERVE_FACTOR = 2,
        MIN_RESERVE    = std::size_t(1) << 20
    };

    void set_mode(HugePages hp) noexcept {
        hps_  = (hp == HugePages::Transparent) ? detail::huge_page_size() : 0;
        huge_ = false;
        gran_ = detail::page_size();
    }

    // Reservations are always huge-page aligned when huge pages are on the
    // table, so that activating them later needs no relocation.
    std::size_t align() const noexcept {
        return hps_ ? hps_ : detail::page_size();
    }

    // Switch to huge-page granularity once a commit of at least one whole huge
    // page is asked for. From then on commit/decommit are rounded to the huge
    // page size, because a partial decommit would split the huge page back into
    // base pages and quietly undo the benefit.
    bool activate_huge(std::size_t n) noexcept {
        if (huge_ || hps_ == 0 || n < hps_) return false;
        gran_ = hps_;
        huge_ = true;
        return true;
    }

    void disown() noexcept {
        base_ = nullptr;
        size_ = committed_ = reserved_ = 0;
        huge_ = fixed_ = false;
        gran_ = detail::page_size();   // hps_ kept: the mode outlives the memory
    }

    // Auto mode only: take a larger reservation and move the live bytes there.
    void relocate(std::size_t want_commit) {
        const std::size_t scaled = want_commit <= std::size_t(-1) / RESERVE_FACTOR
            ? want_commit * RESERVE_FACTOR : want_commit;
        const std::size_t res = detail::round_up(
            scaled < MIN_RESERVE ? MIN_RESERVE : scaled, align());
        unsigned char* nb = static_cast<unsigned char*>(detail::reserve(res, align()));
        if (!nb) throw std::bad_alloc();
        if (!detail::commit(nb, want_commit, huge_)) {
            detail::release(nb, res);
            throw std::bad_alloc();
        }
        if (size_) std::memcpy(nb, base_, size_);
        if (base_) detail::release(base_, reserved_);
        base_      = nb;
        reserved_  = res;
        committed_ = want_commit;
    }

    unsigned char* base_      = nullptr;
    std::size_t    size_      = 0;   // logical bytes in use
    std::size_t    committed_ = 0;   // granule-rounded bytes backed by memory
    std::size_t    reserved_  = 0;   // granule-rounded address space held
    std::size_t    gran_      = 0;   // commit/decommit granularity
    std::size_t    hps_       = 0;   // huge page size if requested & available
    bool           huge_      = false;  // gran_ == hps_
    bool           fixed_     = false;  // reservation is a hard ceiling
};

// std::vector-like view of a VmBuffer for trivially copyable elements.
// Elements are never constructed or destroyed individually; growth commits
// pages geometrically and shrink_to_fit() hands whole granules back to the OS
// without copying anything.
template<typename T>
class VmVector {

#if __GNUC__ >= 5 || _MSC_VER || __clang__
    static_assert(std::is_trivially_copyable<T>::value, "VmVector requires a trivially copyable element type");
#endif

public:
    using value_type      = T;
    using size_type       = std::size_t;
    using iterator        = T*;
    using const_iterator  = const T*;

    VmVector() = default;
    explicit VmVector(HugePages hp) noexcept : buf_(hp) {}

    VmVector(size_type n, const T& v) { resize(n, v); }

    VmVector(size_type n, const T& v, HugePages hp) : buf_(hp) { resize(n, v); }

    VmVector(const VmVector& o) : buf_(o.buf_.huge_pages_mode()) {
        assign(o.data(), o.size());
    }

    VmVector& operator=(const VmVector& o) {
        if (this != &o) assign(o.data(), o.size());
        return *this;
    }

    VmVector(VmVector&&) noexcept = default;
    VmVector& operator=(VmVector&&) noexcept = default;

    T*       data()       noexcept { return reinterpret_cast<T*>(buf_.data()); }
    const T* data() const noexcept { return reinterpret_cast<const T*>(buf_.data()); }

    size_type size()     const noexcept { return buf_.size() / sizeof(T); }
    size_type capacity() const noexcept { return buf_.capacity() / sizeof(T); }
    bool      empty()    const noexcept { return buf_.empty(); }

    T&       operator[](size_type i)       noexcept { return data()[i]; }
    const T& operator[](size_type i) const noexcept { return data()[i]; }

    T&       back()       noexcept { return data()[size() - 1]; }
    const T& back() const noexcept { return data()[size() - 1]; }

    iterator       begin()        noexcept { return data(); }
    iterator       end()          noexcept { return data() + size(); }
    const_iterator begin()  const noexcept { return data(); }
    const_iterator end()    const noexcept { return data() + size(); }
    const_iterator cbegin() const noexcept { return begin(); }
    const_iterator cend()   const noexcept { return end(); }

    void reserve(size_type n) { buf_.reserve_bytes(n * sizeof(T)); }

    bool request_huge_pages() { return buf_.request_huge_pages(); }
    bool huge_pages_active() const noexcept { return buf_.huge_pages_active(); }

    void resize(size_type n) {
        const size_type s = size();
        grow(n);
        buf_.resize(n * sizeof(T));
        //if (n > s) std::memset(data() + s, 0, (n - s) * sizeof(T));
    }

    void resize(size_type n, const T& v) {
        const size_type s = size();
        grow(n);
        buf_.resize(n * sizeof(T));
        if (n > s) std::fill(data() + s, data() + n, v);
    }

    void clear() noexcept { buf_.clear(); }
    void shrink_to_fit() noexcept { buf_.shrink_to_fit(); }

    void push_back(const T& v) {
        const size_type s = size();
        grow(s + 1);
        buf_.resize((s + 1) * sizeof(T));
        data()[s] = v;
    }

    template<typename It>
    void append(It first, It last) {
        const size_type n = size_type(std::distance(first, last));
        const size_type s = size();
        grow(s + n);
        buf_.resize((s + n) * sizeof(T));
        std::copy(first, last, data() + s);
    }

    void append(size_type n, const T& v) {
        const size_type s = size();
        grow(s + n);
        buf_.resize((s + n) * sizeof(T));
        std::fill(data() + s, data() + s + n, v);
    }

    void assign(const T* src, size_type n) {
        buf_.clear();
        grow(n);
        buf_.resize(n * sizeof(T));
        if (n) std::memcpy(data(), src, n * sizeof(T));
    }

private:

    // Geometric so that an unreserved fill does not commit one page at a time.
    void grow(size_type n) {
        const size_type cap = capacity();
        if (n <= cap) return;
        reserve(n > cap + cap / 2 ? n : cap + cap / 2);
    }

    VmBuffer buf_;

};