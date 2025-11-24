#include <mpfr.h>
#include <random>
#include <iostream>
#include <stdexcept>
#include <string>
#include <limits>
#include <chrono>
#include <vector>

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

void test_uint_add_performance(std::mt19937_64& rng) {
    std::uniform_real_distribution<long double> dist(0.8L, 1.2L);
    
    constexpr int N = 200000;
    constexpr int n_iters = 20;
    
    std::vector<Fixed<TagSmall, 3>> fixed_nums(N);
    std::vector<MpReal>             mp_fixed_nums(N);
    for (int i = 0; i < N; ++i) {
        long double val  = dist(rng);
        fixed_nums[i]    = from_ldbl<TagSmall, 3>(val);
        mp_fixed_nums[i] = mp_from_fixed_small(fixed_nums[i]);
    }

    using namespace std::chrono;

    auto start = high_resolution_clock::now();
    Fixed<TagSmall, 3> fprod;
    for (int iter = 0; iter < n_iters; ++iter) {
        fprod = from_ldbl<TagSmall, 3>(1);
        for (int t = 0; t < N; ++t) {
            // MpReal mp_a = mp_from_fixed_small(fa);
            // MpReal mp_b = mp_from_fixed_small(fb);

            // Add: stays in [0,1), so exact in our fractional representation
            // fsum = fadd(fa, fb);
            // MpReal mp_sum = mp_from_fixed_small(fsum);
            // MpReal mp_sum_ref = mp_a + mp_b;
            // check_close(mp_sum, mp_sum_ref, "FixedSmall add");

            // Mul: stays in [0,0.25), also safe
            fprod = fmul(fprod, fixed_nums[t]);
            // MpReal mp_prod = mp_from_fixed_small(fprod);
            // MpReal mp_prod_ref = mp_a * mp_b;
            // check_close(mp_prod, mp_prod_ref, "FixedSmall mul");
        }
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "Our implementation: " << duration.count() / n_iters << " ms\n";

    MpReal mp_prod_ref(static_cast<long double>(1.0));
    start = high_resolution_clock::now();
    for (int iter = 0; iter < n_iters; ++iter) {
        mp_prod_ref = MpReal(static_cast<long double>(1.0));
        for (int t = 0; t < N; ++t) {
            // long double ax = dist(mp_rng);
            // long double bx = dist(mp_rng);

            // auto fa = from_ldbl<TagSmall, 3>(ax);
            // auto fb = from_ldbl<TagSmall, 3>(bx);

            // MpReal mp_a(ax);
            // MpReal mp_b(bx);

            // Add: stays in [0,1), so exact in our fractional representation
            // fsum = fadd(fa, fb);
            // MpReal mp_sum = mp_from_fixed_small(fsum);
            // mp_sum_ref = mp_a + mp_b;
            // check_close(mp_sum, mp_sum_ref, "FixedSmall add");

            // Mul: stays in [0,0.25), also safe
            // fprod = fmul(fa, fb);
            // MpReal mp_prod = mp_from_fixed_small(fprod);
            mp_prod_ref = mp_prod_ref * mp_fixed_nums[t];
            // check_close(mp_prod, mp_prod_ref, "FixedSmall mul");
        }
    }

    end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);
    std::cout << "MPFR Implementation: " << duration.count() / n_iters << " ms\n";

    MpReal check_prod = mp_from_fixed_small(fprod);
    check_close(check_prod, mp_prod_ref, "Fixed prod");
}

// ============================================================
// main
// ============================================================

int main() {
    try {
        std::cout << "Testing Fixed<TagSmall,3> addition+multiplication performance..." << std::endl;
        {
            std::mt19937_64 rng(123);
            test_uint_add_performance(rng);
        }

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

