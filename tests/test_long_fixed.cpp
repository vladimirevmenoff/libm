#include <mpfr.h>
#include <random>
#include <iostream>
#include <stdexcept>
#include <string>
#include <limits>

#include "long/long_fixed.hpp"
#include "long/long_fixed_math.hpp"

using namespace longfixed;

// ============================================================
// Simple MPFR C++ RAII wrapper
// ============================================================

class MpReal {
public:
    static constexpr mpfr_prec_t PREC = 512;

    MpReal() {
        mpfr_init2(v_, PREC);
        mpfr_set_ui(v_, 0, MPFR_RNDN);
    }

    explicit MpReal(unsigned long long x) {
        mpfr_init2(v_, PREC);
        mpfr_set_ui(v_, static_cast<unsigned long>(x), MPFR_RNDN);
    }

    explicit MpReal(long double x) {
        mpfr_init2(v_, PREC);
        mpfr_set_ld(v_, x, MPFR_RNDN);
    }

    MpReal(const MpReal& other) {
        mpfr_init2(v_, PREC);
        mpfr_set(v_, other.v_, MPFR_RNDN);
    }

    MpReal& operator=(const MpReal& other) {
        if (this != &other) {
            mpfr_set(v_, other.v_, MPFR_RNDN);
        }
        return *this;
    }

    ~MpReal() {
        mpfr_clear(v_);
    }

    mpfr_t& raw() { return v_; }
    const mpfr_t& raw() const { return v_; }

    friend MpReal operator+(const MpReal& a, const MpReal& b) {
        MpReal r;
        mpfr_add(r.v_, a.v_, b.v_, MPFR_RNDN);
        return r;
    }

    friend MpReal operator-(const MpReal& a, const MpReal& b) {
        MpReal r;
        mpfr_sub(r.v_, a.v_, b.v_, MPFR_RNDN);
        return r;
    }

    friend MpReal operator*(const MpReal& a, const MpReal& b) {
        MpReal r;
        mpfr_mul(r.v_, a.v_, b.v_, MPFR_RNDN);
        return r;
    }

    friend MpReal operator/(const MpReal& a, const MpReal& b) {
        MpReal r;
        mpfr_div(r.v_, a.v_, b.v_, MPFR_RNDN);
        return r;
    }

    friend MpReal abs(const MpReal& a) {
        MpReal r;
        mpfr_abs(r.v_, a.v_, MPFR_RNDN);
        return r;
    }

    friend bool operator==(const MpReal& a, const MpReal& b) {
        return mpfr_cmp(a.v_, b.v_) == 0;
    }

    friend bool operator!=(const MpReal& a, const MpReal& b) {
        return !(a == b);
    }

    friend bool operator<(const MpReal& a, const MpReal& b) {
        return mpfr_cmp(a.v_, b.v_) < 0;
    }

    friend bool operator>(const MpReal& a, const MpReal& b) {
        return mpfr_cmp(a.v_, b.v_) > 0;
    }

    friend bool operator<=(const MpReal& a, const MpReal& b) {
        return mpfr_cmp(a.v_, b.v_) <= 0;
    }

    friend bool operator>=(const MpReal& a, const MpReal& b) {
        return mpfr_cmp(a.v_, b.v_) >= 0;
    }

    friend MpReal ldexp(const MpReal& a, long k) {
        MpReal r;
        mpfr_mul_2si(r.v_, a.v_, k, MPFR_RNDN);
        return r;
    }

    long double to_long_double() const {
        return mpfr_get_ld(v_, MPFR_RNDN);
    }

private:
    mpfr_t v_;
};

// ============================================================
// Small test infra
// ============================================================

struct TestFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

#define CHECK(cond, msg) \
    do { if (!(cond)) throw TestFailure(std::string("FAILED: ") + (msg)); } while(0)

static MpReal mp_from_ld(long double x) {
    return MpReal(x);
}

// UInt<N> -> MpReal (exact integer)
template<std::size_t N>
MpReal mp_from_uint(const UInt<N>& u) {
    MpReal base;
    mpfr_set_ui(base.raw(), 1, MPFR_RNDN);
    mpfr_mul_2si(base.raw(), base.raw(), 64, MPFR_RNDN); // base = 2^64

    MpReal r;
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
        mpfr_mul(r.raw(), r.raw(), base.raw(), MPFR_RNDN);
        MpReal limb(static_cast<unsigned long long>(u.w[static_cast<std::size_t>(i)]));
        mpfr_add(r.raw(), r.raw(), limb.raw(), MPFR_RNDN);
    }
    return r;
}

// Fixed<TagSmall,N> -> MpReal
template<std::size_t N>
MpReal mp_from_fixed_small(const Fixed<TagSmall, N>& f) {
    MpReal r = mp_from_uint(f.val);
    mpfr_mul_2si(r.raw(), r.raw(), -static_cast<long>(64 * N), MPFR_RNDN);
    return r;
}

// Fixed<TagBig,N> -> MpReal
template<std::size_t N>
MpReal mp_from_fixed_big(const Fixed<TagBig, N>& f) {
    MpReal r = mp_from_uint(f.val);
    mpfr_mul_2si(r.raw(), r.raw(), -static_cast<long>(64 * (N - 1)), MPFR_RNDN);
    return r;
}

static void check_close(const MpReal& a, const MpReal& b,
                        const std::string& msg,
                        long double rel_tol = 1e-30L,
                        long double abs_tol = 1e-30L) {
    MpReal diff = a - b;
    MpReal ad = abs(diff);

    MpReal aa = abs(a);
    MpReal ab = abs(b);
    MpReal maxab = (aa > ab) ? aa : ab;

    MpReal rel = ad / (maxab + MpReal(1e-40L));

    MpReal atol(abs_tol);
    MpReal rtol(rel_tol);

    if (ad > atol && rel > rtol) {
        std::cerr << "a = " << a.to_long_double()
                  << ", b = " << b.to_long_double()
                  << ", |a-b| = " << ad.to_long_double()
                  << "\n";
        throw TestFailure("CLOSE CHECK FAILED: " + msg);
    }
}

// Reduce modulo 2^(64*N) to [0, 2^(64*N)) using only standard MPFR ops
template<std::size_t N>
MpReal mod_2pow(const MpReal& x) {
    MpReal t = x;
    long k = static_cast<long>(64 * N);

    // t_scaled = x / 2^k
    mpfr_mul_2si(t.raw(), t.raw(), -k, MPFR_RNDN);

    // frac = fractional part of t_scaled in (-1,1)
    MpReal frac;
    mpfr_frac(frac.raw(), t.raw(), MPFR_RNDN);

    // r = frac * 2^k  (now in (-2^k, 2^k))
    MpReal r;
    mpfr_mul_2si(r.raw(), frac.raw(), k, MPFR_RNDN);

    // Ensure result is in [0, 2^k)
    if (mpfr_sgn(r.raw()) < 0) {
        MpReal two_pow_k;
        mpfr_set_ui(two_pow_k.raw(), 1, MPFR_RNDN);
        mpfr_mul_2si(two_pow_k.raw(), two_pow_k.raw(), k, MPFR_RNDN);
        mpfr_add(r.raw(), r.raw(), two_pow_k.raw(), MPFR_RNDN);
    }
    return r;
}

// ============================================================
// Tests for UInt<N> arithmetic
// ============================================================

template<std::size_t N>
void test_uint_add_sub_shift_mul_div(std::mt19937_64& rng) {
    std::uniform_int_distribution<std::uint64_t> dist64(0, std::numeric_limits<std::uint64_t>::max());
    std::uniform_int_distribution<std::uint32_t> dist32(1, 1000000);
    std::uniform_int_distribution<int> distShift(0, static_cast<int>(64 * N - 1));

    constexpr int NUM = 2000;

    for (int t = 0; t < NUM; ++t) {
        UInt<N> a{}, b{};
        for (std::size_t i = 0; i < N; ++i) {
            a.w[i] = dist64(rng);
            b.w[i] = dist64(rng);
        }

        MpReal mp_a = mp_from_uint(a);
        MpReal mp_b = mp_from_uint(b);

        // ADD (mod 2^(64N))
        auto sum = uadd(a, b);
        MpReal mp_sum = mp_from_uint(sum);
        MpReal mp_ref_add = mod_2pow<N>(mp_a + mp_b);
        check_close(mp_sum, mp_ref_add, "UInt add (mod 2^K)");

        // SUB: diff = (a + b) - b  (mod 2^(64N))
        auto c    = uadd(a, b);
        auto diff = usub(c, b);
        MpReal mp_c    = mp_from_uint(c);
        MpReal mp_diff = mp_from_uint(diff);
        MpReal mp_ref_sub = mod_2pow<N>(mp_c - mp_b);
        check_close(mp_diff, mp_ref_sub, "UInt sub (mod 2^K)");

        // Left shift: a << s (mod 2^(64N))
        int s = distShift(rng);
        auto asl = ushift_left(a, s);
        MpReal mp_asl = mp_from_uint(asl);
        MpReal mp_ref_shl = mod_2pow<N>(ldexp(mp_a, s));
        check_close(mp_asl, mp_ref_shl, "UInt left shift (mod 2^K)");

        // Right shift: floor(a / 2^s)
        auto asr = ushift_right(a, s);
        MpReal mp_asr = mp_from_uint(asr);
        MpReal mp_tmp = ldexp(mp_a, -s); // exact a / 2^s
        MpReal mp_ref_shr;
        mpfr_floor(mp_ref_shr.raw(), mp_tmp.raw());
        check_close(mp_asr, mp_ref_shr, "UInt right shift");

        // MUL full: exact, no modulus (2N limbs)
        auto full = umul_full<N, N>(a, b);
        MpReal mp_full = mp_from_uint(full);
        check_close(mp_full, mp_a * mp_b, "UInt full mul");

        // DIV by 32-bit: a = q*d + r
        std::uint32_t d = dist32(rng);
        std::uint32_t rem = 0;
        auto q = udiv_u32(a, d, rem);
        MpReal mp_q = mp_from_uint(q);
        MpReal mp_d(static_cast<unsigned long long>(d));
        MpReal mp_r(static_cast<unsigned long long>(rem));

        MpReal recomposed = mp_q * mp_d + mp_r;
        check_close(recomposed, mp_a, "UInt div_u32 recomposition");
    }
}

// ============================================================
// Fixed<TagSmall,N> conversion + arithmetic tests
// ============================================================

void test_fixed_small_conversion() {
    std::mt19937_64 rng(12345);

    // TagSmall represents a fraction: val = U / 2^(64*N) in [0, 1).
    std::uniform_real_distribution<long double> dist(0.0L, 1.0L);

    constexpr int NUM = 2000;

    for (int t = 0; t < NUM; ++t) {
        long double x = dist(rng);  // 0 <= x < 1

        auto fx = from_ldbl<TagSmall, 3>(x);
        MpReal mp_x = mp_from_ld(x);
        MpReal mp_fx = mp_from_fixed_small(fx);

        check_close(mp_fx, mp_x, "from_ldbl<TagSmall,3>");

        long double y = to_ldbl<TagSmall, 3>(fx);
        MpReal mp_y = mp_from_ld(y);

        check_close(mp_y, mp_x, "to_ldbl<TagSmall,3> o from_ldbl<TagSmall,3>");
    }
}

void test_fixed_big_conversion() {
    std::mt19937_64 rng(54321);

    // TagBig is "integer-like" non-negative in our current usage.
    std::uniform_real_distribution<long double> dist(0.0L, 1e4L);

    constexpr int NUM = 2000;

    for (int t = 0; t < NUM; ++t) {
        long double x = dist(rng);  // x >= 0

        auto fx = from_ldbl<TagBig, 3>(x);
        MpReal mp_x = mp_from_ld(x);
        MpReal mp_fx = mp_from_fixed_big(fx);

        check_close(mp_fx, mp_x, "from_ldbl<TagBig,3>");

        long double y = to_ldbl<TagBig, 3>(fx);
        MpReal mp_y = mp_from_ld(y);

        check_close(mp_y, mp_x, "to_ldbl<TagBig,3> o from_ldbl<TagBig,3>");
    }
}

void test_fixed_small_arith() {
    std::mt19937_64 rng(777);

    // For TagSmall arithmetic tests, keep values in [0, 0.5)
    // so that a + b < 1 and no modulo wrap occurs.
    std::uniform_real_distribution<long double> dist(0.0L, 0.5L);

    constexpr int NUM = 2000;

    for (int t = 0; t < NUM; ++t) {
        long double ax = dist(rng);
        long double bx = dist(rng);

        auto fa = from_ldbl<TagSmall, 3>(ax);
        auto fb = from_ldbl<TagSmall, 3>(bx);

        MpReal mp_a = mp_from_fixed_small(fa);
        MpReal mp_b = mp_from_fixed_small(fb);

        // Add: stays in [0,1), so exact in our fractional representation
        auto fsum = fadd(fa, fb);
        MpReal mp_sum = mp_from_fixed_small(fsum);
        MpReal mp_sum_ref = mp_a + mp_b;
        check_close(mp_sum, mp_sum_ref, "FixedSmall add");

        // Mul: stays in [0,0.25), also safe
        auto fprod = fmul(fa, fb);
        MpReal mp_prod = mp_from_fixed_small(fprod);
        MpReal mp_prod_ref = mp_a * mp_b;
        check_close(mp_prod, mp_prod_ref, "FixedSmall mul");
    }
}

// ============================================================
// reverse_small_to_mantissa_and_exp test (U192 small)
// ============================================================

void test_reverse_small_to_mantissa() {
    std::mt19937_64 rng(999);
    std::uniform_int_distribution<std::uint64_t> dist64(0, std::numeric_limits<std::uint64_t>::max());
    constexpr int NUM = 2000;

    for (int t = 0; t < NUM; ++t) {
        U192 u{};
        for (std::size_t i = 0; i < 3; ++i) {
            u.w[i] = dist64(rng);
        }
        if (u.is_zero()) {
            u.w[0] = 1;
        }

        // S = u / 2^192
        MpReal S = mp_from_uint(u);
        mpfr_mul_2si(S.raw(), S.raw(), -192, MPFR_RNDN);

        int exp = 0;
        U128 mant = reverse_small_to_mantissa_and_exp(u, exp);

        // S2 = 2^exp * mant / 2^112
        MpReal M = mp_from_uint(mant);
        mpfr_mul_2si(M.raw(), M.raw(), -112, MPFR_RNDN);
        MpReal S2 = ldexp(M, exp);

        check_close(S, S2, "reverse_small_to_mantissa_and_exp(U192)");
    }
}

// ============================================================
// main
// ============================================================

int main() {
    try {
        std::cout << "Testing UInt<2/3/4> basic arithmetic..." << std::endl;
        {
            std::mt19937_64 rng(123);
            test_uint_add_sub_shift_mul_div<2>(rng);
            test_uint_add_sub_shift_mul_div<3>(rng);
            test_uint_add_sub_shift_mul_div<4>(rng);
        }

        std::cout << "Testing Fixed<TagSmall,3> conversions..." << std::endl;
        test_fixed_small_conversion();

        std::cout << "Testing Fixed<TagBig,3> conversions..." << std::endl;
        test_fixed_big_conversion();

        std::cout << "Testing Fixed<TagSmall,3> arithmetic..." << std::endl;
        test_fixed_small_arith();

        std::cout << "Testing reverse_small_to_mantissa_and_exp (U192)..." << std::endl;
        test_reverse_small_to_mantissa();

        std::cout << "All tests PASSED." << std::endl;
        return 0;
    } catch (const TestFailure& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected exception: " << e.what() << std::endl;
        return 1;
    }
}

