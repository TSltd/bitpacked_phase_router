#define _POSIX_C_SOURCE 199309L

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <random>
#include <chrono>
#include <stdexcept>
#include <string>
#include <omp.h>

namespace py = pybind11;

#define WORD_BITS 64
#define NB(N) (((N) + WORD_BITS - 1) / WORD_BITS)

// ---------------------------------------------------------
// Utilities
// ---------------------------------------------------------

// Timing

static inline double now_ms()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * ts.tv_sec + ts.tv_nsec * 1e-6;
}

// Fast range

static inline uint64_t fast_range(uint64_t x, uint64_t range)
{
    __uint128_t prod = (__uint128_t)x * range;
    return (uint64_t)(prod >> 64);
}

// Splitmix64

static inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Power-of-2 guard

static inline bool is_power_of_2(size_t n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

static inline void require_power_of_2(size_t N)
{
    if (!is_power_of_2(N))
        throw std::invalid_argument(
            "N must be a power of 2 (got N=" + std::to_string(N) + ")");
}

// ---------------------------------------------------------
// Templated Kernels
// ---------------------------------------------------------

template <int NBW>
static inline std::vector<uint64_t> route_interval_templated(
    const uint64_t *bits,
    size_t N)
{
    std::vector<uint64_t> out(N * NBW, 0);

    // ---- prefix sums ----
    std::vector<int> prefix(N, 0);
    int sum = 0;

    for (size_t i = 0; i < N; i++)
    {
        prefix[i] = sum;

#pragma unroll
        for (int w = 0; w < NBW; w++)
            sum += __builtin_popcountll(bits[i * NBW + w]);
    }

    // ---- routing (INTERVAL-BASED, rolling pointer) ----
#pragma omp parallel for schedule(static) if (N >= 512)
    for (size_t i = 0; i < N; i++)
    {
        int j2 = prefix[i] & (N - 1);

        const uint64_t *row = &bits[i * NBW];
        uint64_t *out_row = &out[i * NBW];

        int word_idx = j2 >> 6;
        int bit_pos = j2 & 63;

#pragma unroll
        for (int w = 0; w < NBW; w++)
        {
            uint64_t m = row[w];
            if (m == 0)
                continue;

            int pc = __builtin_popcountll(m);

            if (pc <= 2)
            {
                // Sparse: ctz path (avoids popcount overhead for 1-2 bits)
                while (m)
                {
                    out_row[word_idx] |= (1ULL << bit_pos);

                    if (++bit_pos == 64)
                    {
                        bit_pos = 0;
                        if (++word_idx == NBW)
                            word_idx = 0;
                    }

                    m &= m - 1;
                }
            }
            else
            {
                // Dense: counted loop (no ctz dependency chain)
                for (int t = 0; t < pc; t++)
                {
                    out_row[word_idx] |= (1ULL << bit_pos);

                    if (++bit_pos == 64)
                    {
                        bit_pos = 0;
                        if (++word_idx == NBW)
                            word_idx = 0;
                    }
                }
            }
        }
    }

    return out;
}

template <int NBW>
static inline void and_extract_fused_templated(
    const uint64_t *S,
    const uint64_t *T_routed,
    size_t N,
    size_t k,
    int *routes,
    uint64_t seed_base)
{
#pragma omp parallel for schedule(static) if (N * k >= 16384)
    for (size_t i = 0; i < N; i++)
    {
        const uint64_t *Srow = &S[i * NBW];

        // Hoist transpose-access constants outside inner loop
        const size_t iw = i >> 6;
        const uint64_t ibit = 1ULL << (i & 63);

        int count = 0;

#pragma unroll
        for (int w = 0; w < NBW; w++)
        {
            uint64_t m = Srow[w];

            while (m)
            {
                int b = __builtin_ctzll(m);
                int j = (w << 6) + b;

                // Fused rotate: T_final[i][j] == T_routed[N-1-j][i]
                const uint64_t *Trow = &T_routed[(N - 1 - j) * NBW];
                if (Trow[iw] & ibit)
                {
                    if (count < (int)k)
                    {
                        routes[i * k + count] = j;
                    }
                    else
                    {
                        uint64_t h = splitmix64(
                            seed_base ^
                            (uint64_t(i) << 32) ^
                            (uint64_t(j) << 1) ^
                            count);

                        uint64_t r = fast_range(h, (uint64_t)(count + 1));

                        if (r < (uint64_t)k)
                            routes[i * k + r] = j;
                    }

                    count++;
                }

                m &= m - 1;
            }
        }

        for (int jj = count; jj < (int)k; jj++)
            routes[i * k + jj] = -1;
    }
}

// ------------------------------------------------------------
// Dispatchers
// ------------------------------------------------------------

static std::vector<uint64_t> route_dispatch(
    const uint64_t *bits,
    size_t N,
    size_t NB_words)
{
    switch (NB_words)
    {
    case 4:
        return route_interval_templated<4>(bits, N);
    case 8:
        return route_interval_templated<8>(bits, N);
    case 16:
        return route_interval_templated<16>(bits, N);
    case 32:
        return route_interval_templated<32>(bits, N);
    default:
    {
        // fallback generic
        std::vector<uint64_t> out(N * NB_words, 0);

        std::vector<int> prefix(N, 0);
        int sum = 0;

        for (size_t i = 0; i < N; i++)
        {
            prefix[i] = sum;
            for (size_t w = 0; w < NB_words; w++)
                sum += __builtin_popcountll(bits[i * NB_words + w]);
        }

#pragma omp parallel for schedule(static) if (N >= 512)
        for (size_t i = 0; i < N; i++)
        {
            int j2 = prefix[i] & (N - 1);

            uint64_t *out_row = &out[i * NB_words];
            int word_idx = j2 >> 6;
            int bit_pos = j2 & 63;

            for (size_t w = 0; w < NB_words; w++)
            {
                uint64_t m = bits[i * NB_words + w];
                if (m == 0)
                    continue;

                int pc = __builtin_popcountll(m);

                if (pc <= 2)
                {
                    // Sparse: ctz path
                    while (m)
                    {
                        out_row[word_idx] |= (1ULL << bit_pos);

                        if (++bit_pos == 64)
                        {
                            bit_pos = 0;
                            if (++word_idx == (int)NB_words)
                                word_idx = 0;
                        }

                        m &= m - 1;
                    }
                }
                else
                {
                    // Dense: counted loop
                    for (int t = 0; t < pc; t++)
                    {
                        out_row[word_idx] |= (1ULL << bit_pos);

                        if (++bit_pos == 64)
                        {
                            bit_pos = 0;
                            if (++word_idx == (int)NB_words)
                                word_idx = 0;
                        }
                    }
                }
            }
        }

        return out;
    }
    }
}

static void and_extract_fused_generic(
    const uint64_t *S,
    const uint64_t *T_routed,
    size_t N,
    size_t NB_words,
    size_t k,
    int *routes,
    uint64_t seed_base)
{
#pragma omp parallel for schedule(static) if (N * k >= 16384)
    for (size_t i = 0; i < N; i++)
    {
        const uint64_t *Srow = &S[i * NB_words];

        // Hoist transpose-access constants outside inner loop
        const size_t iw = i >> 6;
        const uint64_t ibit = 1ULL << (i & 63);

        int count = 0;

        for (size_t w = 0; w < NB_words; w++)
        {
            uint64_t m = Srow[w];

            while (m)
            {
                int b = __builtin_ctzll(m);
                int j = (w << 6) + b;

                // Fused rotate: T_final[i][j] == T_routed[N-1-j][i]
                const uint64_t *Trow = &T_routed[(N - 1 - j) * NB_words];
                if (Trow[iw] & ibit)
                {
                    if (count < (int)k)
                    {
                        routes[i * k + count] = j;
                    }
                    else
                    {
                        uint64_t h = splitmix64(
                            seed_base ^
                            (uint64_t(i) << 32) ^
                            (uint64_t(j) << 1) ^
                            count);

                        uint64_t r = fast_range(h, (uint64_t)(count + 1));

                        if (r < (uint64_t)k)
                            routes[i * k + r] = j;
                    }

                    count++;
                }

                m &= m - 1;
            }
        }

        for (int jj = count; jj < (int)k; jj++)
            routes[i * k + jj] = -1;
    }
}

static void and_extract_dispatch(
    const uint64_t *S,
    const uint64_t *T_routed,
    size_t N,
    size_t NB_words,
    size_t k,
    int *routes,
    uint64_t seed_base)
{
    switch (NB_words)
    {
    case 4:
        and_extract_fused_templated<4>(S, T_routed, N, k, routes, seed_base);
        break;
    case 8:
        and_extract_fused_templated<8>(S, T_routed, N, k, routes, seed_base);
        break;
    case 16:
        and_extract_fused_templated<16>(S, T_routed, N, k, routes, seed_base);
        break;
    case 32:
        and_extract_fused_templated<32>(S, T_routed, N, k, routes, seed_base);
        break;
    default:
        and_extract_fused_generic(S, T_routed, N, NB_words, k, routes, seed_base);
    }
}

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

// Rotate T 90° clockwise (retained as utility — not called from hot path)

__attribute__((unused)) static void rotate90_clockwise(
    const uint64_t *T_prepared,
    uint64_t *T_final,
    size_t N,
    size_t NB_words)
{
    std::fill(T_final, T_final + N * NB_words, 0ULL);

#pragma omp parallel for schedule(static) if (N >= 512)
    for (size_t dst_i = 0; dst_i < N; dst_i++)
    {
        size_t src_col = dst_i;

        for (size_t src_i = 0; src_i < N; src_i++)
        {
            size_t src_w = src_col >> 6;
            size_t src_b = src_col & 63;

            if (T_prepared[src_i * NB_words + src_w] & (1ULL << src_b))
            {
                size_t dst_col = N - 1 - src_i;
                size_t dst_w = dst_col >> 6;
                size_t dst_b = dst_col & 63;
                T_final[dst_i * NB_words + dst_w] |= 1ULL << dst_b;
            }
        }
    }
}

// Left-align rows (utility for callers)

py::array_t<uint8_t> left_align_rows(py::array_t<uint8_t> S_np)
{
    auto S = S_np.unchecked<2>();
    size_t N = S.shape(0);
    py::array_t<uint8_t> S_aligned_np({N, N});
    uint8_t *S_aligned = (uint8_t *)S_aligned_np.mutable_data();

#pragma omp parallel for schedule(static) if (N >= 512)
    for (size_t i = 0; i < N; i++)
    {
        size_t ones_count = 0;
        for (size_t j = 0; j < N; j++)
            ones_count += S(i, j);
        for (size_t j = 0; j < ones_count; j++)
            S_aligned[i * N + j] = 1;
        for (size_t j = ones_count; j < N; j++)
            S_aligned[i * N + j] = 0;
    }
    return S_aligned_np;
}

// Pack bits

py::array_t<uint64_t> pack_bits(py::array_t<uint8_t> M_np)
{
    auto M = M_np.unchecked<2>();
    size_t N = M.shape(0);
    size_t NB_words = NB(N);

    py::array_t<uint64_t> bits_np({N, NB_words});
    uint64_t *bits = (uint64_t *)bits_np.mutable_data();

#pragma omp parallel for schedule(static) if (N >= 512)
    for (size_t i = 0; i < N; i++)
    {
        for (size_t w = 0; w < NB_words; w++)
        {
            uint64_t word = 0;

            for (size_t b = 0; b < 64; b++)
            {
                size_t j = w * 64 + b;
                if (j < N && M(i, j))
                    word |= 1ULL << b;
            }

            bits[i * NB_words + w] = word;
        }
    }

    return bits_np;
}

// ------------------------------------------------------------
// MAIN KERNEL - Core router (INDEX-SPACE VERSION)
// ------------------------------------------------------------

static void phase_router_bitpacked(
    size_t N, size_t k, size_t NB_words,
    const uint64_t *S_bits,
    const uint64_t *T_bits,
    const uint64_t * /*row_perm*/,
    const uint64_t * /*col_perm_S*/,
    const uint64_t * /*col_perm_T*/,
    const uint64_t * /*row_perm_T*/,
    int *routes,
    const char * /*debug_prefix*/,
    uint64_t seed_base)
{
    require_power_of_2(N);

    auto S_routed = route_dispatch(S_bits, N, NB_words);

    auto T_routed = route_dispatch(T_bits, N, NB_words);

    // Fused: and_extract now does on-the-fly transpose access into T_routed
    // — no rotate90_clockwise needed
    and_extract_dispatch(
        S_routed.data(),
        T_routed.data(),
        N,
        NB_words,
        k,
        routes,
        seed_base);
}

// ------------------------------------------------------------
// Python API (unchanged)
// ------------------------------------------------------------
py::dict pack_and_route(py::array_t<uint8_t> S_np,
                        py::array_t<uint8_t> T_np,
                        size_t k,
                        py::array_t<int> routes_np,
                        bool /*dump*/ = false,
                        const std::string & /*prefix*/ = "",
                        bool /*validate*/ = false,
                        uint64_t seed = 0)
{
    size_t N = S_np.shape(0);
    size_t NB_words = NB(N);

    double t0_pack = now_ms();

    auto S_bits_np = pack_bits(S_np);
    auto T_bits_np = pack_bits(T_np);

    std::vector<uint64_t> S_bits(N * NB_words);
    std::vector<uint64_t> T_bits(N * NB_words);

    std::memcpy(S_bits.data(), S_bits_np.data(), N * NB_words * sizeof(uint64_t));
    std::memcpy(T_bits.data(), T_bits_np.data(), N * NB_words * sizeof(uint64_t));

    std::vector<uint64_t> row_perm(N), row_perm_T(N);
    for (size_t i = 0; i < N; i++)
    {
        row_perm[i] = i;
        row_perm_T[i] = i;
    }

    uint64_t seed_base = (seed == 0)
                             ? std::chrono::high_resolution_clock::now().time_since_epoch().count()
                             : seed;

    double t0_route = now_ms();

    phase_router_bitpacked(
        N, k, NB_words,
        S_bits.data(),
        T_bits.data(),
        row_perm.data(),
        nullptr, nullptr,
        row_perm_T.data(),
        (int *)routes_np.mutable_data(),
        nullptr,
        seed_base);

    double t1_route = now_ms();

    size_t active = 0;
    for (size_t i = 0; i < N * k; i++)
        active += ((int *)routes_np.data())[i] != -1;

    py::dict d;
    d["N"] = N;
    d["k"] = k;
    d["active_routes"] = active;
    d["packing_time_ms"] = t0_route - t0_pack;
    d["routing_time_ms"] = t1_route - t0_route;
    d["total_time_ms"] = t1_route - t0_pack;
    d["routes_per_row"] = double(active) / double(N);

    return d;
}

// Simple entry point: takes uint8_t matrices, packs and routes

void phase_router_cpp(py::array_t<uint8_t> S_np,
                      py::array_t<uint8_t> T_np,
                      size_t k,
                      py::array_t<int> routes_np,
                      bool /*validate*/ = false,
                      const std::string & /*debug_prefix*/ = "",
                      uint64_t seed = 0)
{
    size_t N = S_np.shape(0);
    size_t NB_words = NB(N);

    std::vector<uint64_t> S_bits(N * NB_words);
    std::vector<uint64_t> T_bits(N * NB_words);

    auto S = S_np.unchecked<2>();
    auto T = T_np.unchecked<2>();

#pragma omp parallel for schedule(static) if (N >= 512)
    for (size_t i = 0; i < N; i++)
        for (size_t w = 0; w < NB_words; w++)
        {
            uint64_t word = 0;
            for (size_t b = 0; b < 64; b++)
            {
                size_t j = w * 64 + b;
                if (j < N && S(i, j))
                    word |= 1ULL << b;
            }
            S_bits[i * NB_words + w] = word;
        }

#pragma omp parallel for schedule(static) if (N >= 512)
    for (size_t i = 0; i < N; i++)
        for (size_t w = 0; w < NB_words; w++)
        {
            uint64_t word = 0;
            for (size_t b = 0; b < 64; b++)
            {
                size_t j = w * 64 + b;
                if (j < N && T(i, j))
                    word |= 1ULL << b;
            }
            T_bits[i * NB_words + w] = word;
        }

    uint64_t seed_base = (seed == 0)
                             ? std::chrono::high_resolution_clock::now().time_since_epoch().count()
                             : seed;

    phase_router_bitpacked(
        N, k, NB_words,
        S_bits.data(),
        T_bits.data(),
        nullptr,
        nullptr, nullptr,
        nullptr,
        (int *)routes_np.mutable_data(),
        nullptr,
        seed_base);
}

// ------------------------------------------------------------
// Pybind module
// ------------------------------------------------------------
PYBIND11_MODULE(router, m)
{
    m.doc() = "Interval-space phase router (structure-preserving, drop-in replacement)";

    m.def("left_align_rows", &left_align_rows);
    m.def("pack_bits", &pack_bits);

    m.def("phase_router_bitpacked", &phase_router_bitpacked);

    m.def("pack_and_route", &pack_and_route,
          py::arg("S_np"),
          py::arg("T_np"),
          py::arg("k"),
          py::arg("routes_np"),
          py::arg("dump") = false,
          py::arg("prefix") = "",
          py::arg("validate") = false,
          py::arg("seed") = 0);

    m.def("router", &phase_router_cpp,
          py::arg("S"), py::arg("T"),
          py::arg("k"), py::arg("routes"),
          py::arg("validate") = false,
          py::arg("debug_prefix") = "",
          py::arg("seed") = 0);
}
