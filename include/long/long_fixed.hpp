#pragma once

#include <array>
#include <cstdint>
#include <type_traits>
#include <limits>
#include <cmath>    // frexp, ldexp, isnan, isinf, signbit

namespace longfixed {

// ============================================================================
// 1. Generic Multi-Precision Unsigned Integer (data + minimal helpers)
// ============================================================================

template<std::size_t N>
struct UInt {
    // Little-endian limbs: w[0] is least significant 64 bits
    std::array<std::uint64_t, N> w{};

    static constexpr std::size_t bits() noexcept { return N * 64; }

    // ----- basic helpers -----

    constexpr bool is_zero() const noexcept {
        for (auto v : w)
            if (v != 0) return false;
        return true;
    }

    friend constexpr bool operator==(const UInt& a, const UInt& b) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            if (a.w[i] != b.w[i]) return false;
        return true;
    }

    friend constexpr bool operator!=(const UInt& a, const UInt& b) noexcept {
        return !(a == b);
    }

private:
    static int clz64(std::uint64_t x) noexcept {
        if (x == 0) return 64;
    #if defined(__GNUC__) || defined(__clang__)
        return __builtin_clzll(x);
    #else
        int n = 0;
        std::uint64_t mask = 1ull << 63;
        while ((x & mask) == 0) {
            ++n;
            mask >>= 1;
        }
        return n;
    #endif
    }

public:
    // Count leading zeros in the full N*64-bit integer
    int clz() const noexcept {
        for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
            auto v = w[static_cast<std::size_t>(i)];
            if (v != 0) {
                int z = clz64(v);
                int hi_words = static_cast<int>(N) - 1 - i;
                return z + hi_words * 64;
            }
        }
        return static_cast<int>(N * 64);
    }
};

// Convenience aliases
using U128 = UInt<2>;
using U192 = UInt<3>;
using U256 = UInt<4>;

// ============================================================================
// 2. Fixed-Point Tags + Wrapper Type (no arithmetic here)
// ============================================================================

struct TagSmall {};    // value = U / 2^(64*N)
struct TagBig {};      // value = U / 2^(64*(N-1))
struct TagMantissa {}; // normalized ~[1,2) (reserved)

template<typename Tag, std::size_t N>
struct Fixed {
    UInt<N> val{};
};

// Scale in bits for each Tag (used by conversions)
template<typename Tag, std::size_t N>
constexpr int fp_scale_bits() noexcept {
    static_assert(N >= 2, "Need at least 128 bits for binary128 mantissa");
    if constexpr (std::is_same_v<Tag, TagSmall>) {
        return static_cast<int>(64 * N);
    } else if constexpr (std::is_same_v<Tag, TagBig>) {
        return static_cast<int>(64 * (N - 1));
    } else if constexpr (std::is_same_v<Tag, TagMantissa>) {
        return 112; // natural mantissa scale (2^112)
    } else {
        static_assert(sizeof(Tag) == 0, "Unknown Tag for fixed-point scaling");
    }
}

// ============================================================================
// 3. "Binary128" math codec for long double (format-agnostic)
// ============================================================================
//
// We do NOT depend on the actual bit layout of long double.
// Instead we enforce the *mathematical* representation:
//
//   x = sign * 2^exponent * mant / 2^112
//
// where mant is a 113-bit integer stored in U128, with mant in [2^112, 2^113)
// for any non-zero finite x. This is valid even if long double is only 64-bit
// (macOS) and exact if long double is IEEE binary128.
//
// ============================================================================

enum class FPClass : std::uint8_t {
    Zero,
    Normal,     // includes subnormals in practice (we don't care about bit-level status)
    Inf,
    NaN
};

struct DecodedLdbl {
    bool   sign{};      // true if negative
    int    exponent{};  // unbiased exponent in the formula above
    U128   mantissa{};  // integer mant
    FPClass cls{FPClass::Zero};
};

struct Binary128 {
    static constexpr int FracBits = 112; // denominator is 2^112
    static constexpr int MantBits = FracBits + 1; // 113

    static DecodedLdbl decode(long double x) noexcept {
        DecodedLdbl out{};

        if (std::isnan(x)) {
            out.cls = FPClass::NaN;
            out.sign = std::signbit(x);
            return out;
        }
        if (std::isinf(x)) {
            out.cls = FPClass::Inf;
            out.sign = std::signbit(x);
            return out;
        }
        if (x == 0.0L) {
            out.cls = FPClass::Zero;
            out.sign = std::signbit(x);
            return out;
        }

        out.cls  = FPClass::Normal;
        out.sign = std::signbit(x);

        long double ax = out.sign ? -x : x;

        // ax = m * 2^e, 0.5 <= m < 1
        int e;
        long double m = std::frexp(ax, &e);

        // Put mantissa in [1,2): m' = 2*m, e' = e-1
        m *= 2.0L;
        e -= 1;

        // scaled = m' * 2^FracBits, so scaled in [2^FracBits, 2^(FracBits+1))
        long double scaled = std::ldexp(m, FracBits);

        // Convert to 128-bit integer with rounding
        long double rounded = std::floor(scaled + 0.5L);
        if (rounded < std::ldexp(1.0L, FracBits) ||
            rounded >= std::ldexp(2.0L, FracBits)) {
            // Just clamp to range [2^112, 2^113-1] to avoid edge-case overflow
            long double lo = std::ldexp(1.0L, FracBits);
            long double hi = std::ldexp(2.0L, FracBits) - 1.0L;
            if (rounded < lo) rounded = lo;
            if (rounded > hi) rounded = hi;
        }

        unsigned __int128 mant_int = static_cast<unsigned __int128>(rounded);

        U128 mant{};
        mant.w[0] = static_cast<std::uint64_t>(mant_int);
        mant.w[1] = static_cast<std::uint64_t>(mant_int >> 64);

        out.mantissa = mant;
        out.exponent = e;
        return out;
    }

    // Encode back to long double:
    //   x = sign ? -1 : 1 * 2^exponent * mant / 2^112
    static long double encode(bool sign, int exponent, const U128& mant) noexcept {
        if (mant.is_zero()) {
            long double z = 0.0L;
            return sign ? -z : z;
        }

        // Convert mant (U128) to long double
        unsigned __int128 m_int =
            (static_cast<unsigned __int128>(mant.w[1]) << 64) |
             static_cast<unsigned __int128>(mant.w[0]);

        long double m = static_cast<long double>(m_int);
        // m_scaled = mant / 2^112
        long double m_scaled = std::ldexp(m, -FracBits);
        long double x = std::ldexp(m_scaled, exponent);
        return sign ? -x : x;
    }
};

// ============================================================================
// 4. Conversions: long double <-> Fixed<Tag, N> (magnitude only)
// ============================================================================
//
// Binary128::decode gives us:
//   |x| = 2^e * mant / 2^112
//
// For a Fixed<Tag,N> with scale 2^K we store:
//   I = mant * 2^(e + K - 112)
//
// Magnitude only; sign is handled by caller (as in your libm code).
// ============================================================================

template<typename Tag, std::size_t N>
inline Fixed<Tag, N> from_ldbl(long double x) noexcept {
    DecodedLdbl d = Binary128::decode(x);

    Fixed<Tag, N> out{};

    if (d.cls != FPClass::Normal && d.cls != FPClass::Zero) {
        // Inf/NaN -> zero magnitude (caller should handle separately)
        return out;
    }
    if (d.cls == FPClass::Zero) {
        return out;
    }

    constexpr int scale_bits = fp_scale_bits<Tag, N>();
    // I = mant * 2^(e + scale_bits - 112)
    int shift = d.exponent + scale_bits - Binary128::FracBits;

    UInt<N> wide{};
    // inject mant (U128) into low limbs
    wide.w[0] = d.mantissa.w[0];
    if constexpr (N >= 2) {
        wide.w[1] = d.mantissa.w[1];
    }

    UInt<N> res{};

    if (shift >= 0) {
        const int word_shift = shift / 64;
        const int bit_shift  = shift % 64;
        for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
            int src = i - word_shift;
            if (src < 0) {
                res.w[static_cast<std::size_t>(i)] = 0;
                continue;
            }
            std::uint64_t lo = wide.w[static_cast<std::size_t>(src)];
            std::uint64_t hi = 0;
            if (bit_shift != 0 && src > 0) {
                hi = wide.w[static_cast<std::size_t>(src - 1)];
            }
            std::uint64_t v = lo << bit_shift;
            if (bit_shift != 0 && src > 0) {
                v |= (hi >> (64 - bit_shift));
            }
            res.w[static_cast<std::size_t>(i)] = v;
        }
    } else {
        int rshift = -shift;
        const int word_shift = rshift / 64;
        const int bit_shift  = rshift % 64;
        for (std::size_t i = 0; i < N; ++i) {
            int src = static_cast<int>(i) + word_shift;
            if (src >= static_cast<int>(N)) {
                res.w[i] = 0;
                continue;
            }
            std::uint64_t hi = wide.w[static_cast<std::size_t>(src)];
            std::uint64_t lo = 0;
            if (bit_shift != 0 && src + 1 < static_cast<int>(N)) {
                lo = wide.w[static_cast<std::size_t>(src + 1)];
            }
            std::uint64_t v = hi >> bit_shift;
            if (bit_shift != 0 && src + 1 < static_cast<int>(N)) {
                v |= (lo << (64 - bit_shift));
            }
            res.w[i] = v;
        }
    }

    out.val = res;
    return out;
}

// Fixed<Tag,N> -> long double (positive magnitude; sign must be applied by caller)
template<typename Tag, std::size_t N>
inline long double to_ldbl(Fixed<Tag, N> f) noexcept {
    const UInt<N>& I = f.val;
    if (I.is_zero()) {
        return 0.0L;
    }

    constexpr int scale_bits = fp_scale_bits<Tag, N>();

    // I is integer such that x = I / 2^scale_bits.
    // Let msb_pos = floor(log2(I))
    int leading = I.clz();
    int msb_pos = static_cast<int>(N * 64) - 1 - leading;

    // As before, we choose mant & exponent so that:
    //   mant = I * 2^(112 - msb_pos)   (if 112 >= msb_pos, else right-shift)
    //   exp  = msb_pos - scale_bits
    // and Binary128::encode implements x = 2^exp * mant / 2^112.
    int exp = msb_pos - scale_bits;
    int shift_needed = Binary128::FracBits - msb_pos; // 112 - msb_pos

    UInt<N> shifted{};

    if (shift_needed >= 0) {
        const int word_shift = shift_needed / 64;
        const int bit_shift  = shift_needed % 64;
        for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
            int src = i - word_shift;
            if (src < 0) {
                shifted.w[static_cast<std::size_t>(i)] = 0;
                continue;
            }
            std::uint64_t lo = I.w[static_cast<std::size_t>(src)];
            std::uint64_t hi = 0;
            if (bit_shift != 0 && src > 0) {
                hi = I.w[static_cast<std::size_t>(src - 1)];
            }
            std::uint64_t v = lo << bit_shift;
            if (bit_shift != 0 && src > 0) {
                v |= (hi >> (64 - bit_shift));
            }
            shifted.w[static_cast<std::size_t>(i)] = v;
        }
    } else {
        int rshift = -shift_needed;
        const int word_shift = rshift / 64;
        const int bit_shift  = rshift % 64;
        for (std::size_t i = 0; i < N; ++i) {
            int src = static_cast<int>(i) + word_shift;
            if (src >= static_cast<int>(N)) {
                shifted.w[i] = 0;
                continue;
            }
            std::uint64_t hi = I.w[static_cast<std::size_t>(src)];
            std::uint64_t lo = 0;
            if (bit_shift != 0 && src + 1 < static_cast<int>(N)) {
                lo = I.w[static_cast<std::size_t>(src + 1)];
            }
            std::uint64_t v = hi >> bit_shift;
            if (bit_shift != 0 && src + 1 < static_cast<int>(N)) {
                v |= (lo << (64 - bit_shift));
            }
            shifted.w[i] = v;
        }
    }

    U128 mant{};
    mant.w[0] = shifted.w[0];
    if constexpr (N >= 2) {
        mant.w[1] = shifted.w[1];
    } else {
        mant.w[1] = 0;
    }

    return Binary128::encode(false, exp, mant);
}

} // namespace longfixed

