#pragma once

#include "long_fixed.hpp"

namespace longfixed {

// ============================================================================
// 1. Core arithmetic on UInt<N>
// ============================================================================

template<std::size_t N>
inline UInt<N> uadd(const UInt<N>& a, const UInt<N>& b) noexcept {
    UInt<N> r{};
    unsigned __int128 carry = 0;
    for (std::size_t i = 0; i < N; ++i) {
        unsigned __int128 sum =
            static_cast<unsigned __int128>(a.w[i]) +
            static_cast<unsigned __int128>(b.w[i]) +
            carry;
        r.w[i] = static_cast<std::uint64_t>(sum);
        carry  = sum >> 64;
    }
    return r;
}

template<std::size_t N>
inline UInt<N> usub(const UInt<N>& a, const UInt<N>& b) noexcept {
    UInt<N> r{};
    unsigned __int128 borrow = 0;
    for (std::size_t i = 0; i < N; ++i) {
        unsigned __int128 bi =
            static_cast<unsigned __int128>(b.w[i]) + borrow;
        unsigned __int128 ai =
            static_cast<unsigned __int128>(a.w[i]);
        unsigned __int128 diff = ai - bi;
        r.w[i] = static_cast<std::uint64_t>(diff);
        borrow = (ai < bi) ? 1 : 0;
    }
    return r;
}

// Logical shifts (non-negative shift only, for perf)
template<std::size_t N>
inline UInt<N> ushift_left(const UInt<N>& a, int shift) noexcept {
    if (shift <= 0) return a;
    if (shift >= static_cast<int>(N * 64)) return {};

    UInt<N> r{};
    int word_shift = shift / 64;
    int bit_shift  = shift % 64;

    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
        int src = i - word_shift;
        if (src < 0) {
            r.w[static_cast<std::size_t>(i)] = 0;
            continue;
        }
        std::uint64_t lo = a.w[static_cast<std::size_t>(src)];
        std::uint64_t hi = 0;
        if (bit_shift != 0 && src > 0) {
            hi = a.w[static_cast<std::size_t>(src - 1)];
        }
        std::uint64_t v = lo << bit_shift;
        if (bit_shift != 0 && src > 0) {
            v |= (hi >> (64 - bit_shift));
        }
        r.w[static_cast<std::size_t>(i)] = v;
    }
    return r;
}

template<std::size_t N>
inline UInt<N> ushift_right(const UInt<N>& a, int shift) noexcept {
    if (shift <= 0) return a;
    if (shift >= static_cast<int>(N * 64)) return {};

    UInt<N> r{};
    int word_shift = shift / 64;
    int bit_shift  = shift % 64;

    for (std::size_t i = 0; i < N; ++i) {
        int src = static_cast<int>(i) + word_shift;
        if (src >= static_cast<int>(N)) {
            r.w[i] = 0;
            continue;
        }
        std::uint64_t hi = a.w[static_cast<std::size_t>(src)];
        std::uint64_t lo = 0;
        if (bit_shift != 0 && src + 1 < static_cast<int>(N)) {
            lo = a.w[static_cast<std::size_t>(src + 1)];
        }
        std::uint64_t v = hi >> bit_shift;
        if (bit_shift != 0 && src + 1 < static_cast<int>(N)) {
            v |= (lo << (64 - bit_shift));
        }
        r.w[i] = v;
    }
    return r;
}

// Full schoolbook multiplication: UInt<N> * UInt<M> -> UInt<N+M>
template<std::size_t N, std::size_t M>
inline UInt<N + M> umul_full(const UInt<N>& a, const UInt<M>& b) noexcept {
    UInt<N + M> r{};
    for (std::size_t i = 0; i < N; ++i) {
        unsigned __int128 carry = 0;
        for (std::size_t j = 0; j < M; ++j) {
            std::size_t k = i + j;
            unsigned __int128 prod =
                static_cast<unsigned __int128>(a.w[i]) *
                static_cast<unsigned __int128>(b.w[j]) +
                static_cast<unsigned __int128>(r.w[k]) +
                carry;
            r.w[k] = static_cast<std::uint64_t>(prod);
            carry  = prod >> 64;
        }
        r.w[i + M] =
            static_cast<std::uint64_t>(
                static_cast<unsigned __int128>(r.w[i + M]) + carry);
    }
    return r;
}

// Divide by 32-bit unsigned (returns quotient, remainder via out param).
// This is enough for most libm-style scaling / normalization.
template<std::size_t N>
inline UInt<N> udiv_u32(const UInt<N>& a,
                        std::uint32_t d,
                        std::uint32_t& rem_out) noexcept {
    UInt<N> q{};
    std::uint64_t rem = 0;
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
        unsigned __int128 cur =
            (static_cast<unsigned __int128>(rem) << 64) |
            a.w[static_cast<std::size_t>(i)];
        std::uint64_t quot = static_cast<std::uint64_t>(cur / d);
        rem = static_cast<std::uint64_t>(cur % d);
        q.w[static_cast<std::size_t>(i)] = quot;
    }
    rem_out = static_cast<std::uint32_t>(rem);
    return q;
}

// ============================================================================
// 2. Fixed<Tag,N> arithmetic (Small / Big)
// ============================================================================

// Aliases for convenience
template<std::size_t N>
using FixedSmall = Fixed<TagSmall, N>;

template<std::size_t N>
using FixedBig = Fixed<TagBig, N>;

// Add/sub for any Tag
template<typename Tag, std::size_t N>
inline Fixed<Tag, N> fadd(Fixed<Tag, N> a, Fixed<Tag, N> b) noexcept {
    return Fixed<Tag, N>{ uadd(a.val, b.val) };
}

template<typename Tag, std::size_t N>
inline Fixed<Tag, N> fsub(Fixed<Tag, N> a, Fixed<Tag, N> b) noexcept {
    return Fixed<Tag, N>{ usub(a.val, b.val) };
}

// Small * Small -> Small (keep high limbs: divide full product by 2^(64*N))
template<std::size_t N>
inline FixedSmall<N> fmul(FixedSmall<N> a,
                          FixedSmall<N> b) noexcept {
    auto full = umul_full<N, N>(a.val, b.val);
    FixedSmall<N> r{};
    for (std::size_t i = 0; i < N; ++i)
        r.val.w[i] = full.w[i + N];
    return r;
}

// Big * Big -> Big (keep low limbs: usual integer product)
template<std::size_t N>
inline Fixed<TagBig, N> fmul(Fixed<TagBig, N> a, Fixed<TagBig, N> b) {
    constexpr int S = fp_scale_bits<TagBig, N>();   // = 64 * (N - 1)

    // 1) Full 2N-limb product P = U1 * U2
    auto full = umul_full<N, N>(a.val, b.val);      // UInt<2*N>

    // 2) Divide by 2^S -> shift right S bits
    auto shifted = ushift_right(full, S);           // still UInt<2*N>

    // 3) Keep low N limbs as the result
    Fixed<TagBig, N> res{};
    for (std::size_t i = 0; i < N; ++i) {
        res.val.w[i] = shifted.w[i];
    }
    return res;
}

// Mixed cases used in libm:

// Small * Big -> Small  (scale at result level: (a/2^K)*(b/2^Kbig) -> small)
// Implementation: treat Big as Small with different K, or just full product
// then shift appropriately. Here we keep it explicit and fast for N==3,4 etc.
//
// For now, provide a generic "Small x Big -> Small" that keeps high N limbs of
// the full product, assuming both are scaled s.t. result fits in same Small K.
template<std::size_t N>
inline FixedSmall<N> fmul(FixedSmall<N> a,
                          FixedBig<N> b) noexcept {
    auto full = umul_full<N, N>(a.val, b.val);
    FixedSmall<N> r{};
    for (std::size_t i = 0; i < N; ++i)
        r.val.w[i] = full.w[i + N];
    return r;
}

template<std::size_t N>
inline FixedSmall<N> fmul(FixedBig<N> a,
                          FixedSmall<N> b) noexcept {
    return fmul(b, a);
}

// Need to add another mixed-tag primitives, mirroring
// M_u192_mul_u128_as_small / M_u192_mul_u256_as_big, etc.

// ============================================================================
// 3. Helpers used in libm-style code
// ============================================================================

// Normalize "small" fixed-point to mantissa+exponent (similar to reverse_small)
// Idea:
//   input: small value S = U / 2^(64*N)
//   we want: S = 2^e * mant / 2^112, mant bit 112 = 1, output exponent e.
//
// Returns mantissa in U128 (to match binary128 width).
inline U128 reverse_small_to_mantissa_and_exp(const U192& small,
                                              int& exponent_out) noexcept {
    if (small.is_zero()) {
        exponent_out = 0;
        return U128{};
    }

    constexpr int K = 64 * 3; // scale bits for U192 small
    int leading = small.clz();
    int msb_pos = 3 * 64 - 1 - leading;

    // S = U / 2^K = 2^(msb_pos-K) * (U << (112-msb_pos)) / 2^112
    exponent_out = msb_pos - K;
    int shift_needed = 112 - msb_pos;

    U192 shifted{};
    if (shift_needed >= 0) {
        shifted = ushift_left(small, shift_needed);
    } else {
        shifted = ushift_right(small, -shift_needed);
    }

    U128 mant{};
    mant.w[0] = shifted.w[0];
    mant.w[1] = shifted.w[1];
    return mant;
}

} // namespace long

