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

#define ROUTER_ENABLE_DUMP 1
#define ROUTER_DUMP_INTERMEDIATE 1

namespace py = pybind11;

#define WORD_BITS 64
#define NB(N) (((N) + WORD_BITS - 1) / WORD_BITS)

static inline double now_ms()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * ts.tv_sec + ts.tv_nsec * 1e-6;
}

// Forward declarations for PBM dumps
// static void dump_pbm_u8(const char *filename, const uint8_t *M, size_t N);
static void dump_pbm_bits(const char *filename, const uint64_t *bits, size_t N, size_t NB_words);
static void dump_pbm_routes_full(const char *filename, const int *routes, size_t N, size_t k);

/* =========================
   Core kernel
   ========================= */

static void rotate_bits_full(const uint64_t *src, size_t N, size_t NB_words,
                             size_t offset, uint64_t *dst)
{
    if (offset == 0)
    {
        for (size_t w = 0; w < NB_words; w++)
            dst[w] = src[w];
        return;
    }

    size_t word_shift = offset / WORD_BITS;
    size_t bit_shift = offset % WORD_BITS;

    for (size_t w = 0; w < NB_words; w++)
    {
        size_t src1 = (w + NB_words - word_shift) % NB_words;
        size_t src2 = (w + NB_words - word_shift - 1 + NB_words) % NB_words; // wrap around safely

        dst[w] = (src[src1] << bit_shift) | (src[src2] >> (WORD_BITS - bit_shift));
    }

    // Mask off extra bits in last word if N not multiple of 64
    size_t rem = N % WORD_BITS;
    if (rem)
        dst[NB_words - 1] &= (1ULL << rem) - 1;
}

/* ==================
   Column permutation
   ================== */

static void permute_columns_bits(const uint64_t *src,
                                 uint64_t *dst,
                                 const size_t *col_perm,
                                 size_t N,
                                 size_t NB_words)
{
    // ✅ clears entire destination row
    std::memset(dst, 0, NB_words * sizeof(uint64_t));

    // iterate over DESTINATION columns
    for (size_t j = 0; j < N; j++)
    {
        // src column index for this destination column
        size_t src_j = col_perm[j];

        // locate source bit
        size_t src_w = src_j / WORD_BITS;
        size_t src_b = src_j % WORD_BITS;

        // test bit
        if (src[src_w] & (1ULL << src_b))
        {
            // locate destination bit
            size_t dst_w = j / WORD_BITS;
            size_t dst_b = j % WORD_BITS;

            // set destination bit
            dst[dst_w] |= 1ULL << dst_b;
        }
    }
}

/* void phase_router_bitpacked(size_t N, size_t k, size_t NB_words,
                            const uint64_t *S_bits,
                            const uint64_t *T_bits,
                            const size_t *col_perm,
                            int *routes,
                            const char *debug_prefix = nullptr)
{
    // -------------------- Step 1: Compute cumulative row offsets --------------------
    std::vector<size_t> row_offsets(N, 0);
    for (size_t i = 1; i < N; i++)
    {
        size_t rs = 0;
        for (size_t w = 0; w < NB_words; w++)
            rs += __builtin_popcountll(S_bits[(i - 1) * NB_words + w]);
        row_offsets[i] = (row_offsets[i - 1] + rs) % N;
    }

    // -------------------- Step 2: Rotate rows of S and T --------------------
    std::vector<uint64_t> S_rot(N * NB_words);
    std::vector<uint64_t> T_rot(N * NB_words);

#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        rotate_bits_full(&S_bits[i * NB_words], N, NB_words, row_offsets[i], &S_rot[i * NB_words]);
        rotate_bits_full(&T_bits[i * NB_words], N, NB_words, row_offsets[i], &T_rot[i * NB_words]);
    }

    // -------------------- Step 3: Apply COLUMN shuffling --------------------
    std::vector<uint64_t> S_shuf(N * NB_words);
    std::vector<uint64_t> T_shuf(N * NB_words);

#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        permute_columns_bits(
            &S_rot[i * NB_words],
            &S_shuf[i * NB_words],
            col_perm,
            N,
            NB_words);

        permute_columns_bits(
            &T_rot[i * NB_words],
            &T_shuf[i * NB_words],
            col_perm,
            N,
            NB_words);
    }

    // -------------------- Step 4: Rotate T 90° clockwise (optimized) --------------------
    std::vector<uint64_t> T_final(N * NB_words, 0);

// First: transpose T_shuf (bitwise)
#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            size_t src_word = j / WORD_BITS;
            size_t src_bit = j % WORD_BITS;
            uint64_t bit = (T_shuf[i * NB_words + src_word] >> src_bit) & 1ULL;

            size_t dst_i = j;
            size_t dst_j = i;
            size_t dst_word = dst_j / WORD_BITS;
            size_t dst_bit = dst_j % WORD_BITS;

            if (bit)
                T_final[dst_i * NB_words + dst_word] |= 1ULL << dst_bit;
        }
    }

// Second: reverse each row for clockwise rotation
#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        for (size_t w = 0; w < NB_words; w++)
        {
            uint64_t x = T_final[i * NB_words + w];
            // reverse bits in 64-bit word
            x = ((x & 0x5555555555555555ULL) << 1) | ((x >> 1) & 0x5555555555555555ULL);
            x = ((x & 0x3333333333333333ULL) << 2) | ((x >> 2) & 0x3333333333333333ULL);
            x = ((x & 0x0F0F0F0F0F0F0F0FULL) << 4) | ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL);
            x = ((x & 0x00FF00FF00FF00FFULL) << 8) | ((x >> 8) & 0x00FF00FF00FF00FFULL);
            x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x >> 16) & 0x0000FFFF0000FFFFULL);
            x = (x << 32) | (x >> 32);
            T_final[i * NB_words + w] = x;
        }

        // If N is not a multiple of 64, mask off excess bits in last word
        size_t rem = N % WORD_BITS;
        if (rem)
            T_final[i * NB_words + NB_words - 1] &= (1ULL << rem) - 1;
    }

    // -------------------- Step 5: Optional PBM dumps --------------------
#if ROUTER_ENABLE_DUMP && ROUTER_DUMP_INTERMEDIATE
    if (debug_prefix && N <= 4096)
    {
        dump_pbm_bits((std::string(debug_prefix) + "_S_rot.pbm").c_str(), S_rot.data(), N, NB_words);
        dump_pbm_bits((std::string(debug_prefix) + "_T_rot.pbm").c_str(), T_rot.data(), N, NB_words);
        dump_pbm_bits((std::string(debug_prefix) + "_S_shuf.pbm").c_str(), S_shuf.data(), N, NB_words);
        dump_pbm_bits((std::string(debug_prefix) + "_T_shuf.pbm").c_str(), T_shuf.data(), N, NB_words);
        dump_pbm_bits((std::string(debug_prefix) + "_T_final.pbm").c_str(), T_final.data(), N, NB_words);
        dump_pbm_routes_full((std::string(debug_prefix) + "_O_.pbm").c_str(), routes, N, k);
    }
#endif

    // -------------------- Step 6: Route each source row --------------------
#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        uint64_t row[NB(N)];
        rotate_bits_full(&S_bits[i * NB_words], N, NB_words, row_offsets[i], row);

        size_t cnt = 0;
        for (size_t j = 0; j < N && cnt < k; j++)
        {
            const uint64_t *col = &T_final[j * NB_words];
            for (size_t w = 0; w < NB_words && cnt < k; w++)
            {
                uint64_t m = row[w] & col[w];
                while (m && cnt < k)
                {
                    size_t b = __builtin_ctzll(m);
                    routes[i * k + cnt++] = w * WORD_BITS + b;
                    m &= m - 1;
                }
            }
        }

        for (; cnt < k; cnt++)
            routes[i * k + cnt] = -1;
    }
}
 */

void phase_router_bitpacked(size_t N, size_t k, size_t NB_words,
                            const uint64_t *S_bits,
                            const uint64_t *T_bits,
                            const size_t *col_perm,
                            int *routes,
                            const char *debug_prefix = nullptr)
{
    // -------------------- Step 1: Compute cumulative row offsets --------------------
    std::vector<size_t> row_offsets(N, 0);
    for (size_t i = 1; i < N; i++)
    {
        size_t rs = 0;
        for (size_t w = 0; w < NB_words; w++)
            rs += __builtin_popcountll(S_bits[(i - 1) * NB_words + w]);
        row_offsets[i] = (row_offsets[i - 1] + rs) % N;
    }

    // -------------------- Step 2: Rotate rows of S and T --------------------
    std::vector<uint64_t> S_rot(N * NB_words);
    std::vector<uint64_t> T_rot(N * NB_words);

#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        rotate_bits_full(&S_bits[i * NB_words], N, NB_words, row_offsets[i], &S_rot[i * NB_words]);
        rotate_bits_full(&T_bits[i * NB_words], N, NB_words, row_offsets[i], &T_rot[i * NB_words]);
    }

    // -------------------- Step 3: Apply column shuffling --------------------
    std::vector<uint64_t> S_shuf(N * NB_words);
    std::vector<uint64_t> T_shuf(N * NB_words);

#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        permute_columns_bits(&S_rot[i * NB_words], &S_shuf[i * NB_words], col_perm, N, NB_words);
        permute_columns_bits(&T_rot[i * NB_words], &T_shuf[i * NB_words], col_perm, N, NB_words);
    }

    // -------------------- Step 4: Rotate T 90° clockwise --------------------
    std::vector<uint64_t> T_final(N * NB_words, 0);

#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            size_t src_word = j / WORD_BITS;
            size_t src_bit = j % WORD_BITS;
            uint64_t bit = (T_shuf[i * NB_words + src_word] >> src_bit) & 1ULL;

            size_t dst_i = j;
            size_t dst_j = i;
            size_t dst_word = dst_j / WORD_BITS;
            size_t dst_bit = dst_j % WORD_BITS;

            if (bit)
                T_final[dst_i * NB_words + dst_word] |= 1ULL << dst_bit;
        }
    }

#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        for (size_t w = 0; w < NB_words; w++)
        {
            uint64_t x = T_final[i * NB_words + w];
            x = ((x & 0x5555555555555555ULL) << 1) | ((x >> 1) & 0x5555555555555555ULL);
            x = ((x & 0x3333333333333333ULL) << 2) | ((x >> 2) & 0x3333333333333333ULL);
            x = ((x & 0x0F0F0F0F0F0F0F0FULL) << 4) | ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL);
            x = ((x & 0x00FF00FF00FF00FFULL) << 8) | ((x >> 8) & 0x00FF00FF00FF00FFULL);
            x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x >> 16) & 0x0000FFFF0000FFFFULL);
            x = (x << 32) | (x >> 32);
            T_final[i * NB_words + w] = x;
        }

        size_t rem = N % WORD_BITS;
        if (rem)
            T_final[i * NB_words + NB_words - 1] &= (1ULL << rem) - 1;
    }

    // -------------------- Step 5: Optional PBM dumps --------------------
#if ROUTER_ENABLE_DUMP && ROUTER_DUMP_INTERMEDIATE
    if (debug_prefix && N <= 4096)
    {
        dump_pbm_bits((std::string(debug_prefix) + "_S_rot.pbm").c_str(), S_rot.data(), N, NB_words);
        dump_pbm_bits((std::string(debug_prefix) + "_T_rot.pbm").c_str(), T_rot.data(), N, NB_words);
        dump_pbm_bits((std::string(debug_prefix) + "_S_shuf.pbm").c_str(), S_shuf.data(), N, NB_words);
        dump_pbm_bits((std::string(debug_prefix) + "_T_shuf.pbm").c_str(), T_shuf.data(), N, NB_words);
        dump_pbm_bits((std::string(debug_prefix) + "_T_final.pbm").c_str(), T_final.data(), N, NB_words);
    }
#endif

// -------------------- Step 6: Compute routes --------------------
#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        size_t cnt = 0;

        // Use the shuffled S row (after column permutation)
        const uint64_t *row = &S_shuf[i * NB_words];

        // Loop over columns of T_final
        for (size_t j = 0; j < N && cnt < k; j++)
        {
            const uint64_t *col = &T_final[j * NB_words];

            for (size_t w = 0; w < NB_words && cnt < k; w++)
            {
                uint64_t m = row[w] & col[w];
                while (m && cnt < k)
                {
                    size_t b = __builtin_ctzll(m);
                    routes[i * k + cnt++] = w * WORD_BITS + b;
                    m &= m - 1;
                }
            }
        }

        for (; cnt < k; cnt++)
            routes[i * k + cnt] = -1;
    }

    // -------------------- Step 7: Dump routes PBM --------------------
#if ROUTER_ENABLE_DUMP && ROUTER_DUMP_INTERMEDIATE
    if (debug_prefix && N <= 4096)
    {
        dump_pbm_routes_full((std::string(debug_prefix) + "_O.pbm").c_str(), routes, N, k);
    }
#endif
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

py::array_t<uint8_t> top_align_rows(py::array_t<uint8_t> T_np)
{
    auto T = T_np.unchecked<2>();
    size_t N = T.shape(0);
    py::array_t<uint8_t> T_aligned_np({N, N});
    uint8_t *T_aligned = (uint8_t *)T_aligned_np.mutable_data();

#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        size_t ones_count = 0;
        for (size_t j = 0; j < N; j++)
            ones_count += T(i, j);
        for (size_t j = 0; j < ones_count; j++)
            T_aligned[i * N + j] = 1;
        for (size_t j = ones_count; j < N; j++)
            T_aligned[i * N + j] = 0;
    }
    return T_aligned_np;
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

/* =========================
   Python-friendly routing APIs
   ========================= */

py::dict pack_and_route(py::array_t<uint8_t> S_np,
                        py::array_t<uint8_t> T_np,
                        size_t k,
                        py::array_t<int> routes_np,
                        bool dump = false,
                        const std::string &prefix = "dump")
{
    size_t N = S_np.shape(0);
    size_t NB_words = NB(N);

    py::array_t<uint8_t> S_aligned = left_align_rows(S_np);
    py::array_t<uint8_t> T_aligned = top_align_rows(T_np);

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

    std::vector<size_t> col_perm(N);
    unsigned int seed = (unsigned int)time(nullptr);
    for (size_t j = 0; j < N; j++)
        col_perm[j] = j;

    for (size_t j = N - 1; j > 0; j--)
    {
        size_t k = rand_r(&seed) % (j + 1);
        std::swap(col_perm[j], col_perm[k]);
    }

    double t0_route = now_ms();

    phase_router_bitpacked(
        N, k, NB_words,
        S_bits.data(),
        T_bits.data(),
        col_perm.data(),
        (int *)routes_np.mutable_data(),
        dump ? prefix.c_str() : nullptr);

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

py::dict route_packed_with_stats(py::array_t<uint64_t> S_bits_np,
                                 py::array_t<uint64_t> T_bits_np,
                                 py::array_t<size_t> col_perm_np,
                                 size_t k,
                                 py::array_t<int> routes_np)
{
    double t0 = now_ms();
    phase_router_bitpacked(S_bits_np.shape(0), k, S_bits_np.shape(1),
                           (uint64_t *)S_bits_np.data(),
                           (uint64_t *)T_bits_np.data(),
                           (size_t *)col_perm_np.data(),
                           (int *)routes_np.mutable_data());
    double t1 = now_ms();

    size_t N = S_bits_np.shape(0);
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
                      py::array_t<int> routes_np)
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

    std::vector<size_t> col_perm(N);
    for (size_t i = 0; i < N; i++)
        col_perm[i] = i;

    phase_router_bitpacked(N, k, NB_words,
                           S_bits.data(), T_bits.data(),
                           col_perm.data(),
                           (int *)routes_np.mutable_data(),
                           nullptr);
}

/* =========================
   PBM dump functions
   ========================= */

/* static void dump_pbm_u8(const char *filename, const uint8_t *M, size_t N)
{
    FILE *f = fopen(filename, "wb");
    if (!f)
        return;
    fprintf(f, "P1\n%zu %zu\n", N, N); // ASCII PBM
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
            fprintf(f, "%d ", M[i * N + j] ? 1 : 0);
        fprintf(f, "\n");
    }
    fclose(f);
}
 */

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

/* =========================
   Pybind11 module
   ========================= */

PYBIND11_MODULE(router, m)
{
    m.def("left_align_rows", &left_align_rows);
    m.def("top_align_rows", &top_align_rows);
    m.def("pack_bits", &pack_bits);
    m.def("phase_router_bitpacked", &phase_router_bitpacked,
          py::arg("N"), py::arg("k"), py::arg("NB_words"),
          py::arg("S_bits"), py::arg("T_bits"), py::arg("col_perm"), py::arg("routes"),
          py::arg("debug_prefix") = nullptr);
    m.def("pack_and_route", &pack_and_route,
          py::arg("S"), py::arg("T"), py::arg("k"), py::arg("routes"),
          py::arg("dump") = false, py::arg("prefix") = "dump");
    m.def("route_packed_with_stats", &route_packed_with_stats);
    m.def("router", &phase_router_cpp,
          py::arg("S"), py::arg("T"), py::arg("k"), py::arg("routes"));
    m.doc() = "Bit-packed phase router for Python / PyTorch (row-only T with clockwise rotation)";
}
