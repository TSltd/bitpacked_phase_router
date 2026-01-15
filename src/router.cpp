#define _POSIX_C_SOURCE 199309L

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <stddef.h>
#include <omp.h>
#include <time.h>
#include <algorithm>
#include <random>
#include <cstdio>
#include <cstring>
#include <fstream> // for std::ofstream
#include <string>  // for std::string
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

#define ROUTER_ENABLE_DUMP 0
#define ROUTER_DUMP_INTERMEDIATE 0

namespace py = pybind11;

namespace fs = std::filesystem;

#define WORD_BITS 64
#define NB(N) (((N) + WORD_BITS - 1) / WORD_BITS)

static inline double now_ms()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * ts.tv_sec + ts.tv_nsec * 1e-6;
}

// Forward declarations for PBM dumps
#if ROUTER_ENABLE_DUMP
static void dump_pbm_bits(const char *filename, const uint64_t *bits, size_t N, size_t NB_words);
static void dump_pbm_routes_full(const char *filename, const int *routes, size_t N, size_t k);
#endif

// Function prototype for validator (implementation below serves as declaration)
static bool validate_phase_router(size_t N, size_t k, size_t NB_words,
                                  const uint64_t *S_bits,
                                  const uint64_t *T_bits,
                                  const uint64_t *row_perm,
                                  const uint64_t *col_perm_S,
                                  const uint64_t *col_perm_T,
                                  const uint64_t *row_perm_T,
                                  const int *routes,
                                  const char *debug_prefix);

#if ROUTER_ENABLE_DUMP
static void dump_column_stats(const char *name,
                              const uint64_t *bits,
                              size_t N,
                              size_t NB_words)
{
    std::vector<size_t> col(N, 0);

    for (size_t i = 0; i < N; i++)
        for (size_t w = 0; w < NB_words; w++)
        {
            uint64_t m = bits[i * NB_words + w];
            while (m)
            {
                size_t b = __builtin_ctzll(m);
                size_t j = w * 64 + b;
                if (j < N)
                    col[j]++;
                m &= m - 1;
            }
        }

    size_t minc = SIZE_MAX, maxc = 0, sum = 0;
    for (size_t j = 0; j < N; j++)
    {
        minc = std::min(minc, col[j]);
        maxc = std::max(maxc, col[j]);
        sum += col[j];
    }

    double mean = double(sum) / double(N);
    double var = 0;
    for (size_t j = 0; j < N; j++)
        var += (col[j] - mean) * (col[j] - mean);
    var /= N;

    fprintf(stderr,
            "COL[%s]: min=%zu max=%zu mean=%.2f stdev=%.2f skew=%.2f\n",
            name, minc, maxc, mean, sqrt(var), double(maxc) / (mean + 1e-9));
}
#endif

/* ===========
   Core kernel
   ===========*/

static void rotate_bits_full(const uint64_t *src, size_t N, size_t NB_words,
                             size_t offset, uint64_t *dst)
{
    if (N == 0)
        return;

    // Precompute mask for last word (works for N < 64 too)
    const uint64_t mask = (N % WORD_BITS == 0) ? ~0ULL : (1ULL << (N % WORD_BITS)) - 1;

    if (offset == 0)
    {
        // No rotation: just copy & mask last word
        dst[0] = src[0] & mask;
        for (size_t w = 1; w < NB_words; w++)
            dst[w] = src[w];
        return;
    }

    // Shortcut for single-word rows (N <= 64)
    if (NB_words == 1)
    {
        dst[0] = ((src[0] << offset) | (src[0] >> (WORD_BITS - offset))) & mask;
        return;
    }

    // General multi-word rotation
    size_t word_shift = offset / WORD_BITS;
    size_t bit_shift = offset % WORD_BITS;

    for (size_t w = 0; w < NB_words; w++)
    {
        size_t src1 = (w + NB_words - word_shift) % NB_words;
        size_t src2 = (w + NB_words - word_shift - 1 + NB_words) % NB_words;

        uint64_t hi = (bit_shift == 0) ? 0 : (src[src2] >> (WORD_BITS - bit_shift));
        uint64_t lo = src[src1] << bit_shift;

        dst[w] = lo | hi;

        // Mask last word only
        if (w == NB_words - 1)
            dst[w] &= mask;
    }
}

/* ==================
   Column permutation
   ================== */

static void permute_columns_bits(const uint64_t *src,
                                 uint64_t *dst,
                                 const uint64_t *col_perm,
                                 size_t N,
                                 size_t NB_words)
{
    std::memset(dst, 0, NB_words * sizeof(uint64_t));

    for (size_t j = 0; j < N; j++)
    {
        size_t src_j = col_perm[j];

        size_t src_w = src_j / WORD_BITS;
        size_t src_b = src_j % WORD_BITS;

        if (src[src_w] & (1ULL << src_b))
        {
            size_t dst_w = j / WORD_BITS;
            size_t dst_b = j % WORD_BITS;
            dst[dst_w] |= 1ULL << dst_b;
        }
    }
}

static inline uint64_t splitmix64(uint64_t &x)
{
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void phase_router_bitpacked(
    size_t N, size_t k, size_t NB_words,
    const uint64_t *S_bits,
    const uint64_t *T_bits,
    const uint64_t *row_perm, // global row perm for S
    const uint64_t *col_perm_S,
    const uint64_t *col_perm_T,
    const uint64_t *row_perm_T, // global row perm for T
    int *routes,
    const char *debug_prefix)

{
    // -------------------- Step 1: Compute cumulative row offsets from ORIGINAL matrices --------------------
    std::vector<size_t> row_offsets_S(N, 0);
    std::vector<size_t> row_offsets_T(N, 0);

    for (size_t i = 1; i < N; i++)
    {
        size_t rs = 0, rt = 0;

        for (size_t w = 0; w < NB_words; w++)
        {
            rs += __builtin_popcountll(S_bits[(i - 1) * NB_words + w]);
            rt += __builtin_popcountll(T_bits[(i - 1) * NB_words + w]);
        }

        row_offsets_S[i] = (row_offsets_S[i - 1] + rs) % N;
        row_offsets_T[i] = (row_offsets_T[i - 1] + rt) % N;
    }

    // -------------------- Step 2: Rotate rows of S and T --------------------
    std::vector<uint64_t> S_rot(N * NB_words);
    std::vector<uint64_t> T_rot(N * NB_words);

#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        rotate_bits_full(&S_bits[i * NB_words], N, NB_words,
                         row_offsets_S[i], &S_rot[i * NB_words]);
        rotate_bits_full(&T_bits[i * NB_words], N, NB_words,
                         row_offsets_T[i], &T_rot[i * NB_words]);
    }

    // -------------------- Step 3: Apply column shuffling --------------------
    std::vector<uint64_t> S_shuf(N * NB_words);
    std::vector<uint64_t> T_shuf(N * NB_words);

#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        permute_columns_bits(&S_rot[i * NB_words],
                             &S_shuf[i * NB_words],
                             col_perm_S, N, NB_words);

        permute_columns_bits(&T_rot[i * NB_words],
                             &T_shuf[i * NB_words],
                             col_perm_T, N, NB_words);
    }

    // -------------------- Step 4: Global row permutation --------------------
    std::vector<uint64_t> S_final(N * NB_words);
    std::vector<uint64_t> T_prepared(N * NB_words);

#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        size_t src_S = row_perm[i];
        size_t src_T = row_perm_T[i];

        std::memcpy(&S_final[i * NB_words], &S_shuf[src_S * NB_words], NB_words * sizeof(uint64_t));
        std::memcpy(&T_prepared[i * NB_words], &T_shuf[src_T * NB_words], NB_words * sizeof(uint64_t));
    }

    // -------------------- Step 5: Rotate T 90° clockwise (pure rotation) --------------------
    std::vector<uint64_t> T_final(N * NB_words, 0);

#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            size_t src_word = j / WORD_BITS;
            size_t src_bit = j % WORD_BITS;

            uint64_t bit = (T_prepared[i * NB_words + src_word] >> src_bit) & 1ULL;

            // 90° clockwise: T_final[j, i] = T_prepared[i, j]
            size_t dst_i = j;
            size_t dst_j = i;

            size_t dst_word = dst_j / WORD_BITS;
            size_t dst_bit = dst_j % WORD_BITS;

            if (bit)
            {
#pragma omp atomic
                T_final[dst_i * NB_words + dst_word] |= 1ULL << dst_bit;
            }
        }
    }

    // -------------------- Step 5: Optional PBM dumps --------------------
#if ROUTER_ENABLE_DUMP && ROUTER_DUMP_INTERMEDIATE
    if (debug_prefix && N <= 4096)
    {
        fs::path dump_folder = debug_prefix; // folder passed from Python
        fs::create_directories(dump_folder); // ensure it exists

        dump_pbm_bits((dump_folder / "S_rot.pbm").string().c_str(), S_rot.data(), N, NB_words);
        dump_pbm_bits((dump_folder / "T_rot.pbm").string().c_str(), T_rot.data(), N, NB_words);
        dump_pbm_bits((dump_folder / "S_shuf.pbm").string().c_str(), S_shuf.data(), N, NB_words);
        dump_pbm_bits((dump_folder / "T_shuf.pbm").string().c_str(), T_shuf.data(), N, NB_words);
        dump_pbm_bits((dump_folder / "T_final.pbm").string().c_str(), T_final.data(), N, NB_words);
    }
#endif

// -------------------- Step 6: Bitwise AND (S_final ∧ T_final) and extract routes --------------------
#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        size_t cnt = 0;

        const uint64_t *Srow = &S_final[i * NB_words];
        const uint64_t *Trow = &T_final[i * NB_words];

        for (size_t w = 0; w < NB_words && cnt < k; w++)
        {
            uint64_t m = Srow[w] & Trow[w];

            while (m && cnt < k)
            {
                size_t b = __builtin_ctzll(m);
                routes[i * k + cnt++] = w * WORD_BITS + b;
                m &= m - 1;
            }
        }

        for (; cnt < k; cnt++)
            routes[i * k + cnt] = -1;
    }

    validate_phase_router(N, k, NB_words,
                          S_bits, T_bits,
                          row_perm, col_perm_S, col_perm_T, row_perm_T,
                          routes, debug_prefix);

    // -------------------- Step 7: Dump routes PBM --------------------
#if ROUTER_ENABLE_DUMP && ROUTER_DUMP_INTERMEDIATE
    if (debug_prefix && N <= 4096)
    {
        fs::path dump_folder = debug_prefix; // recreate here
        fs::create_directories(dump_folder); // ensure folder exists

        // Dump O.pbm
        dump_pbm_routes_full((dump_folder / "O.pbm").string().c_str(), routes, N, k);
        fprintf(stderr, "DEBUG: dumping O.pbm to %s\n", (dump_folder / "O.pbm").string().c_str());
    }

    // -------- Dump stats ------------
    if (debug_prefix)
    {
        // Build O as a bit-matrix
        std::vector<uint64_t> O_bits(N * NB_words, 0);

        for (size_t i = 0; i < N; i++)
            for (size_t r = 0; r < k; r++)
            {
                int j = routes[i * k + r];
                if (j >= 0)
                    O_bits[i * NB_words + (j >> 6)] |= 1ULL << (j & 63);
            }

        dump_column_stats("T_final", T_final.data(), N, NB_words);
        dump_column_stats("O", O_bits.data(), N, NB_words);
    }

#endif
}

// Extract up to k routes per row from bit-packed O
void extract_routes_from_O(size_t N, size_t k, size_t NB_words,
                           const uint64_t *O_bits,
                           int *routes)
{
#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        size_t cnt = 0;
        for (size_t w = 0; w < NB_words && cnt < k; w++)
        {
            uint64_t m = O_bits[i * NB_words + w];
            while (m && cnt < k)
            {
                size_t b = __builtin_ctzll(m);
                routes[i * k + cnt++] = w * 64 + b;
                m &= m - 1;
            }
        }
        for (; cnt < k; cnt++)
            routes[i * k + cnt] = -1;
    }
}

/* =========================
   Alignment functions
   ========================= */

py::array_t<uint8_t> left_align_rows(py::array_t<uint8_t> S_np)
{
    auto S = S_np.unchecked<2>();
    size_t N = S.shape(0);
    py::array_t<uint8_t> S_aligned_np({N, N});
    uint8_t *S_aligned = (uint8_t *)S_aligned_np.mutable_data();

#pragma omp parallel for
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

/* =========================
   Packing functions
   ========================= */

py::array_t<uint64_t> pack_bits(py::array_t<uint8_t> M_np)
{
    auto M = M_np.unchecked<2>();
    size_t N = M.shape(0);
    size_t NB_words = NB(N);
    py::array_t<uint64_t> bits_np({N, NB_words});
    uint64_t *bits = (uint64_t *)bits_np.mutable_data();

#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
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
    return bits_np;
}
// Debug: check if phase-router preserves bit counts
static bool validate_phase_router(size_t N, size_t k, size_t NB_words,
                                  const uint64_t *S_bits,
                                  const uint64_t *T_bits,
                                  const uint64_t *row_perm,
                                  const uint64_t *col_perm_S,
                                  const uint64_t *col_perm_T,
                                  const uint64_t *row_perm_T,
                                  const int *routes,
                                  const char *debug_prefix = nullptr)
{
    // Count total bits in S and T
    size_t total_bits_S = 0, total_bits_T = 0;
    for (size_t i = 0; i < N; i++)
        for (size_t w = 0; w < NB_words; w++)
        {
            total_bits_S += __builtin_popcountll(S_bits[i * NB_words + w]);
            total_bits_T += __builtin_popcountll(T_bits[i * NB_words + w]);
        }

    // Count total active routes
    size_t total_routes = 0;
    for (size_t i = 0; i < N; i++)
        for (size_t r = 0; r < k; r++)
            if (routes[i * k + r] != -1)
                total_routes++;

    if (debug_prefix)
        fprintf(stderr, "DEBUG[%s]: total bits S=%zu, T=%zu, routes=%zu\n",
                debug_prefix, total_bits_S, total_bits_T, total_routes);
    else
        fprintf(stderr, "DEBUG: total bits S=%zu, T=%zu, routes=%zu\n",
                total_bits_S, total_bits_T, total_routes);

    // Validator returns true if all bits are routed
    return total_bits_S <= total_routes && total_bits_T <= total_routes;
}

/* =========================
   Python-friendly routing APIs
   ========================= */

py::dict pack_and_route(py::array_t<uint8_t> S_np,
                        py::array_t<uint8_t> T_np,
                        size_t k,
                        py::array_t<int> routes_np,
                        bool dump = false,
                        const std::string &prefix = "dump",
                        bool validate = false,
                        uint64_t seed = 0) // seed=0 means use time (non-deterministic)
{
    size_t N = S_np.shape(0);
    size_t NB_words = NB(N);

    py::array_t<uint8_t> S_aligned = left_align_rows(S_np);
    py::array_t<uint8_t> T_aligned = left_align_rows(T_np);

    std::vector<uint64_t> S_bits(N * NB_words);
    std::vector<uint64_t> T_bits(N * NB_words);

    double t0_pack = now_ms();

#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        for (size_t w = 0; w < NB_words; w++)
        {
            uint64_t word = 0;
            for (size_t b = 0; b < 64; b++)
            {
                size_t j = w * 64 + b;
                if (j < N && ((uint8_t *)S_aligned.data())[i * N + j])
                    word |= 1ULL << b;
            }
            S_bits[i * NB_words + w] = word;
        }
    }

#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        for (size_t w = 0; w < NB_words; w++)
        {
            uint64_t word = 0;
            for (size_t b = 0; b < 64; b++)
            {
                size_t j = w * 64 + b;
                if (j < N && ((uint8_t *)T_aligned.data())[i * N + j])
                    word |= 1ULL << b;
            }
            T_bits[i * NB_words + w] = word;
        }
    }

    std::vector<uint64_t> row_perm(N), col_perm_S(N), col_perm_T(N), row_perm_T(N);

    for (size_t i = 0; i < N; i++)
    {
        row_perm[i] = i;
        row_perm_T[i] = i;
        col_perm_S[i] = i;
        col_perm_T[i] = i;
    }

    // Use provided seed if non-zero, otherwise use current time
    uint64_t seed_base = (seed == 0)
                             ? std::chrono::high_resolution_clock::now().time_since_epoch().count()
                             : seed;

    std::mt19937_64 rng_rows(seed_base ^ 0xA5A5A5A5A5A5A5A5ULL);
    std::shuffle(row_perm.begin(), row_perm.end(), rng_rows);

    std::mt19937_64 rng_Trows(seed_base ^ 0xC6BC279692B5C323ULL);
    std::shuffle(row_perm_T.begin(), row_perm_T.end(), rng_Trows);

    std::mt19937_64 rng_S(seed_base ^ 0x9E3779B97F4A7C15ULL);
    std::mt19937_64 rng_T(seed_base ^ 0xD1B54A32D192ED03ULL);

    std::shuffle(col_perm_S.begin(), col_perm_S.end(), rng_S);
    std::shuffle(col_perm_T.begin(), col_perm_T.end(), rng_T);

    double t0_route = now_ms();

    phase_router_bitpacked(
        N, k, NB_words,
        S_bits.data(),
        T_bits.data(),
        row_perm.data(),
        col_perm_S.data(),
        col_perm_T.data(),
        row_perm_T.data(),
        (int *)routes_np.mutable_data(),
        prefix.empty() ? nullptr : prefix.c_str());

    double t1_route = now_ms();

    // Optional validator call
    if (validate)
        validate_phase_router(N, k, NB_words,
                              S_bits.data(),
                              T_bits.data(),
                              row_perm.data(),
                              col_perm_S.data(),
                              col_perm_T.data(),
                              row_perm_T.data(),
                              (int *)routes_np.mutable_data(),
                              dump ? prefix.c_str() : nullptr);

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

// Updated route_packed_with_stats with optional validation
py::dict route_packed_with_stats(py::array_t<uint64_t> S_bits_np,
                                 py::array_t<uint64_t> T_bits_np,
                                 py::array_t<uint64_t> row_perm_np,
                                 py::array_t<uint64_t> col_perm_S_np,
                                 py::array_t<uint64_t> col_perm_T_np,
                                 py::array_t<uint64_t> row_perm_T_np,
                                 size_t k,
                                 py::array_t<int> routes_np,
                                 bool validate = false,
                                 const std::string &debug_prefix = "",
                                 uint64_t seed = 0)
{
    size_t N = S_bits_np.shape(0);
    size_t NB_words = S_bits_np.shape(1);

    // Unpack numpy arrays into local vectors
    std::vector<uint64_t> S_bits(N * NB_words);
    std::vector<uint64_t> T_bits(N * NB_words);
    std::vector<uint64_t> row_perm(N);
    std::vector<uint64_t> col_perm_S(N);
    std::vector<uint64_t> col_perm_T(N);
    std::vector<uint64_t> row_perm_T(N);

    // Copy data from numpy arrays to local vectors
    std::memcpy(S_bits.data(), S_bits_np.data(), N * NB_words * sizeof(uint64_t));
    std::memcpy(T_bits.data(), T_bits_np.data(), N * NB_words * sizeof(uint64_t));
    std::memcpy(row_perm.data(), row_perm_np.data(), N * sizeof(uint64_t));
    std::memcpy(col_perm_S.data(), col_perm_S_np.data(), N * sizeof(uint64_t));
    std::memcpy(col_perm_T.data(), col_perm_T_np.data(), N * sizeof(uint64_t));
    std::memcpy(row_perm_T.data(), row_perm_T_np.data(), N * sizeof(uint64_t));

    double t0 = now_ms();

    phase_router_bitpacked(
        N, k, NB_words,
        S_bits.data(),
        T_bits.data(),
        row_perm.data(),
        col_perm_S.data(),
        col_perm_T.data(),
        row_perm_T.data(),
        (int *)routes_np.mutable_data(),
        debug_prefix.empty() ? nullptr : debug_prefix.c_str());

    double t1 = now_ms();

    if (validate)
    {
        validate_phase_router(S_bits_np.shape(0),
                              k, // pass k here, not NB_words
                              S_bits_np.shape(1),
                              (uint64_t *)S_bits_np.data(),
                              (uint64_t *)T_bits_np.data(),
                              (uint64_t *)row_perm_np.data(),
                              (uint64_t *)col_perm_S_np.data(),
                              (uint64_t *)col_perm_T_np.data(),
                              (uint64_t *)row_perm_T_np.data(),
                              (int *)routes_np.mutable_data(),
                              debug_prefix.empty() ? nullptr : debug_prefix.c_str());
    }

    size_t active = 0;
    for (size_t i = 0; i < N * k; i++)
        active += ((int *)routes_np.data())[i] != -1;

    py::dict d;
    d["N"] = N;
    d["k"] = k;
    d["active_routes"] = active;
    d["routing_time_ms"] = t1 - t0;
    d["routes_per_row"] = double(active) / double(N);
    return d;
}

void phase_router_cpp(py::array_t<uint8_t> S_np,
                      py::array_t<uint8_t> T_np,
                      size_t k,
                      py::array_t<int> routes_np,
                      bool validate = false,
                      const std::string &debug_prefix = "",
                      uint64_t seed = 0)
{
    size_t N = S_np.shape(0);
    size_t NB_words = NB(N);

    std::vector<uint64_t> S_bits(N * NB_words);
    std::vector<uint64_t> T_bits(N * NB_words);

    auto S = S_np.unchecked<2>();
    auto T = T_np.unchecked<2>();

#pragma omp parallel for
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

#pragma omp parallel for
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

    std::vector<uint64_t> row_perm(N), row_perm_T(N), col_perm_S(N), col_perm_T(N);

    for (size_t i = 0; i < N; i++)
    {
        row_perm[i] = i;
        row_perm_T[i] = i;
        col_perm_S[i] = i;
        col_perm_T[i] = i;
    }

    // Use provided seed if non-zero, otherwise use current time
    uint64_t seed_base = (seed == 0)
                             ? std::chrono::high_resolution_clock::now().time_since_epoch().count()
                             : seed;

    std::mt19937_64 rng_rows(seed_base ^ 0xA5A5A5A5A5A5A5A5ULL);
    std::shuffle(row_perm.begin(), row_perm.end(), rng_rows);

    std::mt19937_64 rng_Trows(seed_base ^ 0xC6BC279692B5C323ULL);
    std::shuffle(row_perm_T.begin(), row_perm_T.end(), rng_Trows);

    std::mt19937_64 rng_S(seed_base ^ 0x9E3779B97F4A7C15ULL);
    std::mt19937_64 rng_T(seed_base ^ 0xD1B54A32D192ED03ULL);

    std::shuffle(col_perm_S.begin(), col_perm_S.end(), rng_S);
    std::shuffle(col_perm_T.begin(), col_perm_T.end(), rng_T);

    // Run the router
    phase_router_bitpacked(
        N, k, NB_words,
        S_bits.data(),
        T_bits.data(),
        row_perm.data(),
        col_perm_S.data(),
        col_perm_T.data(),
        row_perm_T.data(),
        (int *)routes_np.mutable_data(),
        debug_prefix.empty() ? nullptr : debug_prefix.c_str());

    // Optional validation (fixed argument order)
    if (validate)
    {
        validate_phase_router(N, k, NB_words, // <-- pass k correctly
                              S_bits.data(),
                              T_bits.data(),
                              row_perm.data(),
                              col_perm_S.data(),
                              col_perm_T.data(),
                              row_perm_T.data(),
                              (int *)routes_np.mutable_data(),
                              debug_prefix.empty() ? nullptr : debug_prefix.c_str());
    }
}

#if ROUTER_ENABLE_DUMP
static void dump_pbm_bits(const char *filename, const uint64_t *bits, size_t N, size_t NB_words)
{
    FILE *f = fopen(filename, "wb");
    if (!f)
        return;
    fprintf(f, "P4\n%zu %zu\n", N, N); // binary PBM
    size_t row_bytes = (N + 7) / 8;
    std::vector<uint8_t> row(row_bytes);
    for (size_t i = 0; i < N; i++)
    {
        std::fill(row.begin(), row.end(), 0);
        for (size_t j = 0; j < N; j++)
        {
            size_t w = j / 64;
            size_t b = j % 64;
            if (bits[i * NB_words + w] & (1ULL << b))
                row[j / 8] |= 0x80 >> (j % 8);
        }
        fwrite(row.data(), 1, row_bytes, f);
    }
    fclose(f);
}

static void dump_pbm_routes_full(const char *filename, const int *routes, size_t N, size_t k)
{
    FILE *f = fopen(filename, "wb");
    if (!f)
        return;

    fprintf(f, "P1\n%zu %zu\n", N, N); // full N x N PBM

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            int val = 0;
            for (size_t r = 0; r < k; r++)
            {
                if (routes[i * k + r] == (int)j)
                {
                    val = 1;
                    break;
                }
            }
            fprintf(f, "%d ", val);
        }
        fprintf(f, "\n");
    }

    fclose(f);
}
#endif

/* =========================
   Pybind11 module
   ========================= */
PYBIND11_MODULE(router, m)
{
    m.doc() = "Bit-packed phase router for Python / PyTorch (row-only T with clockwise rotation)";

    m.def("left_align_rows", &left_align_rows);
    m.def("pack_bits", &pack_bits);

    m.def("phase_router_bitpacked", &phase_router_bitpacked,
          py::arg("N"), py::arg("k"), py::arg("NB_words"),
          py::arg("S_bits"), py::arg("T_bits"),
          py::arg("row_perm"),
          py::arg("col_perm_S"), py::arg("col_perm_T"),
          py::arg("row_perm_T"),
          py::arg("routes"),
          py::arg("debug_prefix") = nullptr);

    m.def("pack_and_route", &pack_and_route,
          py::arg("S_np"),
          py::arg("T_np"),
          py::arg("k"),
          py::arg("routes_np"),
          py::arg("dump") = false,
          py::arg("prefix") = "dump",
          py::arg("validate") = false,
          py::arg("seed") = 0);

    m.def("route_packed_with_stats", &route_packed_with_stats,
          py::arg("S_bits"),
          py::arg("T_bits"),
          py::arg("row_perm"),
          py::arg("col_perm_S"),
          py::arg("col_perm_T"),
          py::arg("row_perm_T"),
          py::arg("k"),
          py::arg("routes"),
          py::arg("validate") = false,
          py::arg("debug_prefix") = "",
          py::arg("seed") = 0);

    m.def("router", &phase_router_cpp,
          py::arg("S"), py::arg("T"),
          py::arg("k"), py::arg("routes"),
          py::arg("validate") = false,
          py::arg("debug_prefix") = "",
          py::arg("seed") = 0);
}
