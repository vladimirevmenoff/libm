// tests/test_long_fixed_quad.cpp
//
// Tests for long_fixed.hpp + long_fixed_arith.hpp using __float128 (quadmath)
// as the reference type, WITHOUT going through long double.
//
// For conversions:
//   - x_ref (float128) is printed
//   - x_fixed = fixed_from_q128_*(x_ref)
//   - x_fixed_q = q_from_fixed_*(x_fixed)
//   - errors are accumulated in float128.
//
// For arithmetic:
//   - ref_op = op(a_ref, b_ref) in float128
//   - fixed_op = op( fixed_from_q128_*(a_ref), fixed_from_q128_*(b_ref) )
//   - result converted back to float128 and compared.
//
// Low-level UInt functions are tested against float128 integer arithmetic
// using restricted ranges to avoid precision loss.

#include <quadmath.h>
#include <random>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstring>

#include "long/long_fixed.hpp"
#include "long/long_fixed_math.hpp"

using namespace longfixed;

// ============================================================
// Helpers for __float128
// ============================================================

static std::string qstr(__float128 x) {
    char buf[128];
    quadmath_snprintf(buf, sizeof(buf), "%+.36Qe", x);
    return std::string(buf);
}

static inline __float128 q_abs(__float128 x) {
    return x < 0 ? -x : x;
}

static inline __float128 q_max(__float128 a, __float128 b) {
    return (a > b) ? a : b;
}

static inline __float128 q_ldexp(__float128 x, int e) {
    return ldexpq(x, e);
}

// ============================================================
// Error accumulation
// ============================================================

struct ErrorStats {
    __float128 max_abs_err = (__float128)0.0L;
    __float128 max_rel_err = (__float128)0.0L;
    std::size_t count = 0;
};

static void update_error(ErrorStats& st,
                         __float128 ref,
                         __float128 approx) {
    __float128 diff    = approx - ref;
    __float128 abs_err = q_abs(diff);
    __float128 abs_ref = q_abs(ref);
    __float128 rel_err =
        (abs_ref > (__float128)0.0L) ? abs_err / abs_ref : abs_err;

    st.max_abs_err = q_max(st.max_abs_err, abs_err);
    st.max_rel_err = q_max(st.max_rel_err, rel_err);
    st.count += 1;
}

static void print_error_stats(const char* name, const ErrorStats& st) {
    std::cout << name << ":\n"
              << "  samples      = " << (unsigned long long)st.count << "\n"
              << "  max abs err  = " << qstr(st.max_abs_err) << "\n"
              << "  max rel err  = " << qstr(st.max_rel_err) << "\n\n";
}

// ============================================================
// Conversions from UInt / Fixed to __float128 (for testing)
// ============================================================

// UInt<N> -> __float128 as integer (exact up to 2^113, approx above)
template<std::size_t N>
__float128 q_from_uint(const UInt<N>& u) {
    __float128 base = q_ldexp((__float128)1.0L, 64); // 2^64
    __float128 r    = (__float128)0.0L;
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
        r = r * base + (__float128)u.w[static_cast<std::size_t>(i)];
    }
    return r;
}

// Fixed<TagSmall,N> -> __float128 (value = U / 2^(64*N))
template<std::size_t N>
__float128 q_from_fixed_small(const Fixed<TagSmall, N>& f) {
    __float128 I = q_from_uint(f.val);
    return q_ldexp(I, -static_cast<int>(64 * N));
}

// Fixed<TagBig,N> -> __float128 (value = U / 2^(64*(N-1)))
template<std::size_t N>
__float128 q_from_fixed_big(const Fixed<TagBig, N>& f) {
    __float128 I = q_from_uint(f.val);
    return q_ldexp(I, -static_cast<int>(64 * (N - 1)));
}

// ============================================================
// Utility: generate restricted UInt<N> so that value < 2^K (for exact-ish q128)
// ============================================================

template<std::size_t N>
void random_uint_bounded(UInt<N>& x, std::mt19937_64& rng, int K_bits) {
    static_assert(N >= 1, "N must be >= 1");
    std::uniform_int_distribution<std::uint64_t> dist(0, ~0ull);

    for (std::size_t i = 0; i < N; ++i) x.w[i] = 0;

    if (K_bits <= 0) {
        return;
    } else if (K_bits <= 64) {
        std::uint64_t mask =
            (K_bits == 64) ? ~0ULL : ((1ULL << K_bits) - 1ULL);
        x.w[0] = dist(rng) & mask;
    } else {
        // K_bits in (64, 128] : use w0 full, w1 partially, others zero
        x.w[0] = dist(rng);
        int high_bits = K_bits - 64;
        if (high_bits > 64) high_bits = 64;
        std::uint64_t mask =
            (high_bits == 64) ? ~0ULL : ((1ULL << high_bits) - 1ULL);
        if (N >= 2) x.w[1] = dist(rng) & mask;
    }
}

// ============================================================
// Binary128 (__float128) bit-level codec for tests
// ============================================================

static constexpr int  BINARY128_EXP_BITS   = 15;
static constexpr int  BINARY128_FRAC_BITS  = 112;
static constexpr int  BINARY128_EXP_BIAS   = 16383;
static constexpr int  BINARY128_EXP_MAXVAL = (1 << BINARY128_EXP_BITS) - 1;

// A decoded binary128 value:
//   x = (-1)^sign * mantissa * 2^(exponent - 112),
// with mantissa having its MSB at bit 112 for non-zero normals.
struct DecodedQ128 {
    bool sign;
    int  exponent;   // unbiased exponent
    U128 mantissa;   // normalized: msb at bit 112 (for non-zero)
};

// Decode a __float128 into sign/exponent/mantissa.
// For tests we only expect finite normals and zero.
static DecodedQ128 decode_q128(__float128 x) {
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
            d.exponent = 0;
            d.mantissa = U128{};  // 0
            return d;
        } else {
            // subnormal: we won't generate these in tests, but handle loosely
            d.exponent = 1 - BINARY128_EXP_BIAS;
            U128 m{};
            m.w[0] = lo;
            m.w[1] = frac_hi;
            d.mantissa = m;
            return d;
        }
    }

    if (exp_bits == BINARY128_EXP_MAXVAL) {
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

// Encode __float128 from DecodedQ128 (exact for normals and zero).
static __float128 encode_q128(const DecodedQ128& d) {
    static_assert(sizeof(__float128) == 16, "__float128 must be 16 bytes");

    std::uint64_t lo = d.mantissa.w[0];
    std::uint64_t hi = d.mantissa.w[1];

    // Strip implicit 1 (for non-zero normals)
    if (!(hi == 0 && lo == 0)) {
        hi &= 0x0000FFFFFFFFFFFFULL;
    }

    int exp_bits = 0;
    if (hi == 0 && lo == 0) {
        exp_bits = 0;    // zero
    } else {
        exp_bits = d.exponent + BINARY128_EXP_BIAS;
        if (exp_bits <= 0) {
            exp_bits = 0; // treat as subnormal/underflow
        } else if (exp_bits >= BINARY128_EXP_MAXVAL) {
            exp_bits = BINARY128_EXP_MAXVAL; // Inf/NaN-ish
        }
    }

    std::uint64_t sign_bit  = d.sign ? (1ULL << 63) : 0ULL;
    std::uint64_t exp_field =
        (static_cast<std::uint64_t>(static_cast<std::uint16_t>(exp_bits)) << 48);

    hi = (hi & 0x0000FFFFFFFFFFFFULL) | exp_field | sign_bit;

    std::uint64_t raw[2] = { lo, hi };
    __float128 x;
    std::memcpy(&x, raw, sizeof(x));
    return x;
}

// ============================================================
// Conversions: __float128 -> Fixed<TagSmall/TagBig,N> (no long double)
// ============================================================

// TagSmall: x = U / 2^(64*N)
template<std::size_t N>
Fixed<TagSmall, N> fixed_from_q128_small(__float128 x) {
    DecodedQ128 d = decode_q128(x);

    Fixed<TagSmall, N> fx{};
    if (d.mantissa.w[0] == 0 && d.mantissa.w[1] == 0) {
        // zero
        return fx;
    }

    int scale_bits = 64 * static_cast<int>(N);
    int shift      = d.exponent + scale_bits - 112;

    UInt<N> U{};
    U.w[0] = d.mantissa.w[0];
    if constexpr (N > 1) {
        U.w[1] = d.mantissa.w[1];
    }
    for (std::size_t i = 2; i < N; ++i) {
        U.w[i] = 0;
    }

    if (shift > 0) {
        U = ushift_left(U, shift);
    } else if (shift < 0) {
        U = ushift_right(U, -shift);
    }

    // ignore sign for now; tests use non-negative ranges
    fx.val = U;
    return fx;
}

// TagBig: x = U / 2^(64*(N-1))
template<std::size_t N>
Fixed<TagBig, N> fixed_from_q128_big(__float128 x) {
    DecodedQ128 d = decode_q128(x);

    Fixed<TagBig, N> fx{};
    if (d.mantissa.w[0] == 0 && d.mantissa.w[1] == 0) {
        return fx;
    }

    int scale_bits = 64 * static_cast<int>(N - 1);
    int shift      = d.exponent + scale_bits - 112;

    UInt<N> U{};
    U.w[0] = d.mantissa.w[0];
    if constexpr (N > 1) {
        U.w[1] = d.mantissa.w[1];
    }
    for (std::size_t i = 2; i < N; ++i) {
        U.w[i] = 0;
    }

    if (shift > 0) {
        U = ushift_left(U, shift);
    } else if (shift < 0) {
        U = ushift_right(U, -shift);
    }

    fx.val = U;
    return fx;
}

// ============================================================
// 1. UInt<N> arithmetic tests: uadd, usub, shifts, mul_full, udiv_u32
// ============================================================

// Test add, sub, shifts for N with values < 2^K_add (no wrap; exact-ish in q128)
template<std::size_t N>
void test_uint_add_sub_shift_q(std::mt19937_64& rng) {
    std::cout << "=== UInt<" << N
              << "> add/sub/shift tests (float128 ref) ===\n";

    constexpr int NUM   = 2000;
    constexpr int K_add = 100;   // ensure sum < 2^101 < 2^113 for exactness

    ErrorStats add_stats{};
    ErrorStats sub_stats{};
    ErrorStats shl_stats{};
    ErrorStats shr_stats{};

    std::uniform_int_distribution<int> shift_dist(0, 10);

    for (int t = 0; t < NUM; ++t) {
        UInt<N> a{}, b{};
        random_uint_bounded(a, rng, K_add);
        random_uint_bounded(b, rng, K_add);

        __float128 aq = q_from_uint(a);
        __float128 bq = q_from_uint(b);

        // --- ADD ---
        auto sum = uadd(a, b);
        __float128 sum_q   = q_from_uint(sum);
        __float128 sum_ref = aq + bq;
        if (t < 5) {
            std::cout << "  [add] sample " << t << ":\n"
                      << "    a_ref      = " << qstr(aq) << "\n"
                      << "    b_ref      = " << qstr(bq) << "\n"
                      << "    sum_ref    = " << qstr(sum_ref) << "\n"
                      << "    sum_uint   = " << qstr(sum_q) << "\n";
        }
        update_error(add_stats, sum_ref, sum_q);

        // --- SUB (ensure a >= b) ---
        UInt<N> as = a, bs = b;
        __float128 asq = aq, bsq = bq;
        if (bsq > asq) {
            std::swap(as, bs);
            std::swap(asq, bsq);
        }
        auto diff = usub(as, bs);
        __float128 diff_q   = q_from_uint(diff);
        __float128 diff_ref = asq - bsq;
        if (t < 5) {
            std::cout << "  [sub] sample " << t << ":\n"
                      << "    as_ref     = " << qstr(asq) << "\n"
                      << "    bs_ref     = " << qstr(bsq) << "\n"
                      << "    diff_ref   = " << qstr(diff_ref) << "\n"
                      << "    diff_uint  = " << qstr(diff_q) << "\n";
        }
        update_error(sub_stats, diff_ref, diff_q);

        // --- SHIFTS ---
        int s = shift_dist(rng);

        auto asl = ushift_left(a, s);
        __float128 asl_q   = q_from_uint(asl);
        __float128 asl_ref = q_ldexp(aq, s);
        if (t < 5) {
            std::cout << "  [shl] sample " << t << " (s=" << s << "):\n"
                      << "    a_ref      = " << qstr(aq) << "\n"
                      << "    shl_ref    = " << qstr(asl_ref) << "\n"
                      << "    shl_uint   = " << qstr(asl_q) << "\n";
        }
        update_error(shl_stats, asl_ref, asl_q);

        auto asr = ushift_right(a, s);
        __float128 asr_q   = q_from_uint(asr);
        __float128 asr_ref = aq / q_ldexp((__float128)1.0L, s);
        __float128 asr_ref_floor = floorq(asr_ref);
        if (t < 5) {
            std::cout << "  [shr] sample " << t << " (s=" << s << "):\n"
                      << "    a_ref      = " << qstr(aq) << "\n"
                      << "    shr_ref    = " << qstr(asr_ref_floor) << "\n"
                      << "    shr_uint   = " << qstr(asr_q) << "\n";
        }
        update_error(shr_stats, asr_ref_floor, asr_q);
    }

    print_error_stats("UInt add", add_stats);
    print_error_stats("UInt sub", sub_stats);
    print_error_stats("UInt shl", shl_stats);
    print_error_stats("UInt shr", shr_stats);
}

// Test mul_full and udiv_u32
template<std::size_t N>
void test_uint_mul_div_q(std::mt19937_64& rng) {
    std::cout << "=== UInt<" << N
              << "> mul_full/div_u32 tests (float128 ref) ===\n";

    constexpr int NUM = 2000;

    ErrorStats mul_stats{};
    ErrorStats div_stats{};

    std::uniform_int_distribution<std::uint64_t> dist64(0, ~0ull);
    std::uniform_int_distribution<std::uint32_t> dist32_denom(1, 1000000);

    for (int t = 0; t < NUM; ++t) {
        // --- MUL ---
        UInt<N> a{}, b{};

        // a: up to ~2^80: w0 full, w1 lower 16 bits (if exists)
        a.w[0] = dist64(rng);
        if constexpr (N >= 2) {
            a.w[1] = dist64(rng) & ((1ULL << 16) - 1ULL);
        }
        for (std::size_t i = 2; i < N; ++i) a.w[i] = 0;

        // b: up to ~2^32: only low limb used
        b.w[0] = dist64(rng) & ((1ULL << 32) - 1ULL);
        for (std::size_t i = 1; i < N; ++i) b.w[i] = 0;

        __float128 aq = q_from_uint(a);
        __float128 bq = q_from_uint(b);

        auto full = umul_full<N, N>(a, b);  // UInt<2N>

        __float128 prod_q   = q_from_uint(full);
        __float128 prod_ref = aq * bq;

        if (t < 5) {
            std::cout << "  [mul] sample " << t << ":\n"
                      << "    a_ref      = " << qstr(aq) << "\n"
                      << "    b_ref      = " << qstr(bq) << "\n"
                      << "    prod_ref   = " << qstr(prod_ref) << "\n"
                      << "    prod_uint  = " << qstr(prod_q) << "\n";
        }
        update_error(mul_stats, prod_ref, prod_q);

        // --- DIV by 32-bit: c = q*d + r ---
        UInt<N> c{};
        random_uint_bounded(c, rng, 96); // < 2^96, safely representable
        __float128 cq = q_from_uint(c);

        std::uint32_t d = dist32_denom(rng);
        std::uint32_t rem = 0;
        auto q = udiv_u32(c, d, rem);

        __float128 q_q   = q_from_uint(q);
        __float128 d_q   = (__float128)d;
        __float128 rem_q = (__float128)rem;
        __float128 recomposed = q_q * d_q + rem_q;

        if (t < 5) {
            std::cout << "  [div] sample " << t << ":\n"
                      << "    c_ref      = " << qstr(cq) << "\n"
                      << "    d          = " << d << "\n"
                      << "    q_uint     = " << qstr(q_q) << "\n"
                      << "    r_uint     = " << qstr(rem_q) << "\n"
                      << "    q*d+r      = " << qstr(recomposed) << "\n";
        }

        update_error(div_stats, cq, recomposed);
    }

    print_error_stats("UInt mul_full", mul_stats);
    print_error_stats("UInt div_u32",  div_stats);
}

// ============================================================
// 2. Conversion tests: __float128 <-> Fixed<TagSmall,3> / TagBig,3
// ============================================================

// TagSmall: x in [0,1)
static void test_conversion_small_q() {
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    constexpr int NUM = 5000;
    ErrorStats stats{};

    std::cout << "=== Conversion test: TagSmall, N=3 (U192 small) ===\n";

    for (int i = 0; i < NUM; ++i) {
        double rd = dist(rng);
        __float128 x_ref = (__float128)rd;

        auto fx = fixed_from_q128_small<3>(x_ref);
        __float128 x_fixed = q_from_fixed_small<3>(fx);

        if (i < 10) {
            std::cout << "  sample " << i << ":\n"
                      << "    x_ref (q128)   = " << qstr(x_ref) << "\n"
                      << "    x_fixed (q128) = " << qstr(x_fixed) << "\n";
        }

        update_error(stats, x_ref, x_fixed);
    }

    print_error_stats("TagSmall conversion", stats);
}

// TagBig: x in [0, 1e6]
static void test_conversion_big_q() {
    std::mt19937_64 rng(54321);
    std::uniform_real_distribution<double> dist(0.0, 1.0e6);

    constexpr int NUM = 5000;
    ErrorStats stats{};

    std::cout << "=== Conversion test: TagBig, N=3 (U192 big) ===\n";

    for (int i = 0; i < NUM; ++i) {
        double rd = dist(rng);
        __float128 x_ref = (__float128)rd;

        auto fx = fixed_from_q128_big<3>(x_ref);
        __float128 x_fixed = q_from_fixed_big<3>(fx);

        if (i < 10) {
            std::cout << "  sample " << i << ":\n"
                      << "    x_ref (q128)   = " << qstr(x_ref) << "\n"
                      << "    x_fixed (q128) = " << qstr(x_fixed) << "\n";
        }

        update_error(stats, x_ref, x_fixed);
    }

    print_error_stats("TagBig conversion", stats);
}

// ============================================================
// 3. Fixed-point arithmetic tests (TagSmall and TagBig) using __float128
// ============================================================

// TagSmall arithmetic: a,b in [0,0.5) so that a+b < 1 (no wrap).
static void test_arith_small_q() {
    std::mt19937_64 rng(777);
    std::uniform_real_distribution<double> dist(0.0, 0.5);

    constexpr int NUM = 5000;
    ErrorStats add_stats{};
    ErrorStats sub_stats{};
    ErrorStats mul_stats{};

    std::cout << "=== Arithmetic test: TagSmall, N=3 (U192 small) ===\n";

    for (int i = 0; i < NUM; ++i) {
        double da = dist(rng);
        double db = dist(rng);

        __float128 a_ref = (__float128)da;
        __float128 b_ref = (__float128)db;

        __float128 add_ref = a_ref + b_ref;
        __float128 mul_ref = a_ref * b_ref;

        // For subtraction, ensure a_sub >= b_sub to avoid wrap in unsigned fixed
        __float128 a_sub = a_ref;
        __float128 b_sub = b_ref;
        if (b_sub > a_sub) std::swap(a_sub, b_sub);
        __float128 sub_ref = a_sub - b_sub;

        auto fa_add = fixed_from_q128_small<3>(a_ref);
        auto fb_add = fixed_from_q128_small<3>(b_ref);

        auto fsum   = fadd(fa_add, fb_add);
        auto fprod  = fmul(fa_add, fb_add);

        auto fa_sub = fixed_from_q128_small<3>(a_sub);
        auto fb_sub = fixed_from_q128_small<3>(b_sub);
        auto fdiff  = fsub(fa_sub, fb_sub);

        __float128 add_fixed_q = q_from_fixed_small<3>(fsum);
        __float128 mul_fixed_q = q_from_fixed_small<3>(fprod);
        __float128 sub_fixed_q = q_from_fixed_small<3>(fdiff);

        if (i < 10) {
            std::cout << "  sample " << i << ":\n"
                      << "    a_ref        = " << qstr(a_ref) << "\n"
                      << "    b_ref        = " << qstr(b_ref) << "\n"
                      << "    add_ref      = " << qstr(add_ref) << "\n"
                      << "    add_fixed    = " << qstr(add_fixed_q) << "\n"
                      << "    sub_ref      = " << qstr(sub_ref) << "\n"
                      << "    sub_fixed    = " << qstr(sub_fixed_q) << "\n"
                      << "    mul_ref      = " << qstr(mul_ref) << "\n"
                      << "    mul_fixed    = " << qstr(mul_fixed_q) << "\n";
        }

        update_error(add_stats, add_ref, add_fixed_q);
        update_error(sub_stats, sub_ref, sub_fixed_q);
        update_error(mul_stats, mul_ref, mul_fixed_q);
    }

    print_error_stats("TagSmall add", add_stats);
    print_error_stats("TagSmall sub", sub_stats);
    print_error_stats("TagSmall mul", mul_stats);
}

// TagBig arithmetic: non-negative values, moderate range
static void test_arith_big_q() {
    std::mt19937_64 rng(888);
    std::uniform_real_distribution<double> dist(0.0, 1.0e6);

    constexpr int NUM = 5000;
    ErrorStats add_stats{};
    ErrorStats sub_stats{};
    ErrorStats mul_stats{};

    std::cout << "=== Arithmetic test: TagBig, N=3 (U192 big) ===\n";

    for (int i = 0; i < NUM; ++i) {
        double da = dist(rng);
        double db = dist(rng);

        __float128 a_ref = (__float128)da;
        __float128 b_ref = (__float128)db;

        __float128 add_ref = a_ref + b_ref;
        __float128 mul_ref = a_ref * b_ref;

        __float128 a_sub = a_ref;
        __float128 b_sub = b_ref;
        if (b_sub > a_sub) std::swap(a_sub, b_sub);
        __float128 sub_ref = a_sub - b_sub;

        auto fa_add = fixed_from_q128_big<3>(a_ref);
        auto fb_add = fixed_from_q128_big<3>(b_ref);

        auto fsum   = fadd(fa_add, fb_add);
        auto fprod  = fmul(fa_add, fb_add);

        auto fa_sub = fixed_from_q128_big<3>(a_sub);
        auto fb_sub = fixed_from_q128_big<3>(b_sub);
        auto fdiff  = fsub(fa_sub, fb_sub);

        __float128 add_fixed_q = q_from_fixed_big<3>(fsum);
        __float128 mul_fixed_q = q_from_fixed_big<3>(fprod);
        __float128 sub_fixed_q = q_from_fixed_big<3>(fdiff);

        if (i < 10) {
            std::cout << "  sample " << i << ":\n"
                      << "    a_ref        = " << qstr(a_ref) << "\n"
                      << "    b_ref        = " << qstr(b_ref) << "\n"
                      << "    add_ref      = " << qstr(add_ref) << "\n"
                      << "    add_fixed    = " << qstr(add_fixed_q) << "\n"
                      << "    sub_ref      = " << qstr(sub_ref) << "\n"
                      << "    sub_fixed    = " << qstr(sub_fixed_q) << "\n"
                      << "    mul_ref      = " << qstr(mul_ref) << "\n"
                      << "    mul_fixed    = " << qstr(mul_fixed_q) << "\n";
        }

        update_error(add_stats, add_ref, add_fixed_q);
        update_error(sub_stats, sub_ref, sub_fixed_q);
        update_error(mul_stats, mul_ref, mul_fixed_q);
    }

    print_error_stats("TagBig add", add_stats);
    print_error_stats("TagBig sub", sub_stats);
    print_error_stats("TagBig mul", mul_stats);
}

// ============================================================
// 4. reverse_small_to_mantissa_and_exp test (U192 small) with __float128
// ============================================================

static void test_reverse_small_mantissa_q() {
    std::mt19937_64 rng(999);
    std::uniform_int_distribution<std::uint64_t> dist64(0, ~0ull);

    constexpr int NUM = 2000;
    ErrorStats stats{};

    std::cout << "=== reverse_small_to_mantissa_and_exp test (U192 small) ===\n";

    for (int i = 0; i < NUM; ++i) {
        U192 u{};
        for (std::size_t k = 0; k < 3; ++k) {
            u.w[k] = dist64(rng);
        }
        if (u.is_zero()) u.w[0] = 1;

        // S = U / 2^192 in [0,1)
        __float128 S =
            q_from_uint(u) * q_ldexp((__float128)1.0L, -192);

        int exp = 0;
        U128 mant = reverse_small_to_mantissa_and_exp(u, exp);

        __float128 mant_q = q_from_uint(mant);
        __float128 S2 =
            mant_q * q_ldexp((__float128)1.0L, exp - 112);

        if (i < 10) {
            std::cout << "  sample " << i << ":\n"
                      << "    S     = " << qstr(S) << "\n"
                      << "    S2    = " << qstr(S2) << "\n"
                      << "    exp   = " << exp << "\n";
        }

        update_error(stats, S, S2);
    }

    print_error_stats("reverse_small_to_mantissa_and_exp", stats);
}

// ============================================================
// main
// ============================================================

int main() {
    try {
        // Low-level UInt tests (all UInt functions)
        {
            std::mt19937_64 rng(123);
            test_uint_add_sub_shift_q<2>(rng);
            test_uint_add_sub_shift_q<3>(rng);
            test_uint_add_sub_shift_q<4>(rng);

            test_uint_mul_div_q<2>(rng);
            test_uint_mul_div_q<3>(rng);
            test_uint_mul_div_q<4>(rng);
        }

        // Conversions
        test_conversion_small_q();
        test_conversion_big_q();

        // Fixed-point arithmetic (TagSmall / TagBig)
        test_arith_small_q();
        test_arith_big_q();

        // reverse_small_to_mantissa_and_exp
        test_reverse_small_mantissa_q();

        std::cout << "All float128-based tests completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }
}

