#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <quadmath.h>

#include "long/long_fixed.hpp"  // for U128, etc.

namespace longfixed {

// Binary128 parameters
static constexpr int  BINARY128_EXP_BITS   = 15;
static constexpr int  BINARY128_FRAC_BITS  = 112;
static constexpr int  BINARY128_EXP_BIAS   = 16383;
static constexpr int  BINARY128_SIGN_SHIFT = 127;
static constexpr int  BINARY128_EXP_SHIFT  = 112;
static constexpr std::uint16_t BINARY128_EXP_MAX = (1u << BINARY128_EXP_BITS) - 1u;

// A "decoded" binary128 value:
//   x = (-1)^sign * 2^exponent * (mantissa / 2^112), with mantissa in [2^112, 2^113)
//   except for zero, where mantissa == 0.
struct DecodedQ128 {
    bool sign;
    int  exponent;   // unbiased exponent
    U128 mantissa;   // normalized: msb at bit 112 (for non-zero)
};

// Decode a __float128 into sign/exponent/mantissa.
// For testing we only expect finite normals and zero.
// Subnormals/Inf/NaN can be added if needed.
inline DecodedQ128 decode_q128(__float128 x) {
    static_assert(sizeof(__float128) == 16, "__float128 must be 16 bytes");

    DecodedQ128 d{};
    std::uint64_t raw[2];
    std::memcpy(raw, &x, sizeof(x));
    // Little-endian: raw[0] = low bits, raw[1] = high bits
    std::uint64_t lo = raw[0];
    std::uint64_t hi = raw[1];

    bool sign = (hi >> 63) != 0;
    std::uint16_t exp_bits =
        static_cast<std::uint16_t>((hi >> 48) & 0x7FFFu);
    std::uint64_t frac_hi = hi & 0x0000FFFFFFFFFFFFULL;

    d.sign = sign;

    if (exp_bits == 0) {
        // zero or subnormal
        if (frac_hi == 0 && lo == 0) {
            // ±0
            d.exponent = 0;
            d.mantissa = U128{};  // 0
            return d;
        } else {
            // subnormals: for strict testing you can handle them here.
            // For now, treat as zero in tests or avoid generating them.
            d.exponent = 1 - BINARY128_EXP_BIAS;
            U128 m{};
            m.w[0] = lo;
            m.w[1] = frac_hi;
            d.mantissa = m;
            return d;
        }
    }

    if (exp_bits == BINARY128_EXP_MAX) {
        // Inf or NaN; not expected in normal tests
        d.exponent = 0;
        d.mantissa = U128{};
        return d;
    }

    // Normal number
    int e = static_cast<int>(exp_bits) - BINARY128_EXP_BIAS;

    U128 m{};
    m.w[0] = lo;
    // implicit 1 at bit 112 -> bit 48 of the high limb
    m.w[1] = (1ULL << 48) | frac_hi;

    d.exponent = e;
    d.mantissa = m;
    return d;
}

// Encode back to __float128 from DecodedQ128.
// We *rebuild the bits*, so this is bit-exact for normals and zero.
inline __float128 encode_q128(const DecodedQ128& d) {
    static_assert(sizeof(__float128) == 16, "__float128 must be 16 bytes");

    std::uint64_t lo = d.mantissa.w[0];
    std::uint64_t hi = d.mantissa.w[1];

    // Strip implicit 1 (for non-zero normals)
    if (!(hi == 0 && lo == 0)) {
        hi &= 0x0000FFFFFFFFFFFFULL;
    }

    int exp_bits = 0;
    if (hi == 0 && lo == 0) {
        exp_bits = 0; // zero
    } else {
        exp_bits = d.exponent + BINARY128_EXP_BIAS;
        if (exp_bits <= 0) {
            // subnormal or underflow; tests shouldn't go here normally
            exp_bits = 0;
        } else if (exp_bits >= BINARY128_EXP_MAX) {
            exp_bits = BINARY128_EXP_MAX; // will look like Inf/NaN
        }
    }

    std::uint64_t sign_bit   = d.sign ? (1ULL << 63) : 0ULL;
    std::uint64_t exp_field  =
        static_cast<std::uint64_t>(static_cast<std::uint16_t>(exp_bits))
        << 48;

    hi = (hi & 0x0000FFFFFFFFFFFFULL) | exp_field | sign_bit;

    std::uint64_t raw[2] = { lo, hi };
    __float128 x;
    std::memcpy(&x, raw, sizeof(x));
    return x;
}

} // namespace longfixed

