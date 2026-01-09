#define _POSIX_C_SOURCE 199309L
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <stdint.h>
#include <stddef.h>
#include <omp.h>
#include <time.h>

namespace py = pybind11;

#define WORD_BITS 64
#define NB(N) (((N) + WORD_BITS - 1) / WORD_BITS)

static inline double now_ms()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * ts.tv_sec + ts.tv_nsec * 1e-6;
}

/* =========================
   Core kernel
   ========================= */

static void rotate_bits(const uint64_t *src, size_t N, size_t NB_words,
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
        size_t src2 = (w + NB_words - word_shift - 1) % NB_words;
        dst[w] = (src[src1] >> bit_shift) | (src[src2] << (WORD_BITS - bit_shift));
    }

    size_t rem = N % WORD_BITS;
    if (rem)
        dst[NB_words - 1] &= (1ULL << rem) - 1;
}

static void compute_offsets(size_t N, size_t NB_words,
                            const uint64_t *S_bits,
                            const uint64_t *T_bits,
                            size_t *row_offsets,
                            size_t *col_offsets)
{
    row_offsets[0] = col_offsets[0] = 0;
    for (size_t i = 1; i < N; i++)
    {
        size_t rs = 0, cs = 0;
        for (size_t w = 0; w < NB_words; w++)
            rs += __builtin_popcountll(S_bits[(i - 1) * NB_words + w]);
        for (size_t w = 0; w < NB_words; w++)
            cs += __builtin_popcountll(T_bits[(i - 1) * NB_words + w]);
        row_offsets[i] = rs % N;
        col_offsets[i] = cs % N;
    }
}

void phase_router_bitpacked(size_t N, size_t k, size_t NB_words,
                            const uint64_t *S_bits,
                            const uint64_t *T_bits,
                            const size_t *row_perm,
                            int *routes)
{
    std::vector<size_t> row_offsets(N), col_offsets(N);
    compute_offsets(N, NB_words, S_bits, T_bits,
                    row_offsets.data(), col_offsets.data());

    std::vector<uint64_t> T_rot(N * NB_words);

#pragma omp parallel for
    for (size_t j = 0; j < N; j++)
        rotate_bits(&T_bits[j * NB_words], N, NB_words,
                    col_offsets[j], &T_rot[j * NB_words]);

#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        uint64_t row[NB(N)];
        size_t pi = row_perm[i];
        rotate_bits(&S_bits[pi * NB_words], N, NB_words, row_offsets[pi], row);

        size_t cnt = 0;
        for (size_t j = 0; j < N && cnt < k; j++)
        {
            const uint64_t *col = &T_rot[j * NB_words];
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

py::array_t<uint8_t> top_align_columns(py::array_t<uint8_t> T_np)
{
    auto T = T_np.unchecked<2>();
    size_t N = T.shape(0);

    py::array_t<uint8_t> T_aligned_np({N, N});
    uint8_t *T_aligned = (uint8_t *)T_aligned_np.mutable_data();

#pragma omp parallel for
    for (size_t j = 0; j < N; j++)
    {
        size_t ones_count = 0;
        for (size_t i = 0; i < N; i++)
            ones_count += T(i, j);
        for (size_t i = 0; i < ones_count; i++)
            T_aligned[i * N + j] = 1;
        for (size_t i = ones_count; i < N; i++)
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

py::array_t<uint64_t> pack_bits_T_permuted(py::array_t<uint8_t> T_np,
                                           py::array_t<size_t> col_perm_np)
{
    auto T = T_np.unchecked<2>();
    auto col_perm = col_perm_np.unchecked<1>();
    size_t N = T.shape(0);
    size_t NB_words = NB(N);

    py::array_t<uint64_t> T_bits_np({N, NB_words});
    uint64_t *T_bits = (uint64_t *)T_bits_np.mutable_data();

#pragma omp parallel for
    for (size_t j = 0; j < N; j++)
    {
        size_t pj = col_perm[j];
        for (size_t w = 0; w < NB_words; w++)
        {
            uint64_t word = 0;
            for (size_t b = 0; b < 64; b++)
            {
                size_t i = w * 64 + b;
                if (i < N && T(i, pj))
                    word |= 1ULL << b;
            }
            T_bits[j * NB_words + w] = word;
        }
    }

    return T_bits_np;
}

/* =========================
   Packed-only routing API
   ========================= */

void route_packed(py::array_t<uint64_t> S_bits_np,
                  py::array_t<uint64_t> T_bits_np,
                  py::array_t<size_t> row_perm_np,
                  size_t k,
                  py::array_t<int> routes_np)
{
    int *r = (int *)routes_np.mutable_data();
    phase_router_bitpacked(
        S_bits_np.shape(0),
        k,
        S_bits_np.shape(1),
        (uint64_t *)S_bits_np.data(),
        (uint64_t *)T_bits_np.data(),
        (size_t *)row_perm_np.data(),
        r);
}

py::dict route_packed_with_stats(py::array_t<uint64_t> S_bits_np,
                                 py::array_t<uint64_t> T_bits_np,
                                 py::array_t<size_t> row_perm_np,
                                 size_t k,
                                 py::array_t<int> routes_np)
{
    double t0 = now_ms();
    int *r = (int *)routes_np.mutable_data();

    phase_router_bitpacked(
        S_bits_np.shape(0),
        k,
        S_bits_np.shape(1),
        (uint64_t *)S_bits_np.data(),
        (uint64_t *)T_bits_np.data(),
        (size_t *)row_perm_np.data(),
        r);
    double t1 = now_ms();

    size_t active = 0;
    for (size_t i = 0; i < S_bits_np.shape(0) * k; i++)
        active += (r[i] != -1);

    py::dict d;
    d["N"] = S_bits_np.shape(0);
    d["k"] = k;
    d["active_routes"] = active;
    d["routing_time_ms"] = t1 - t0;
    d["routes_per_row"] = double(active) / double(S_bits_np.shape(0));
    return d;
}

// =========================
// Wrapper for raw 0/1 matrices (NumPy / PyTorch)
// =========================
void phase_router_cpp(py::array_t<uint8_t> S_np,
                      py::array_t<uint8_t> T_np,
                      size_t k,
                      py::array_t<int> routes_np)
{
    size_t N = S_np.shape(0);
    size_t NB_words = NB(N);

    const uint8_t *S_data = S_np.data();
    const uint8_t *T_data = T_np.data();
    int *routes = routes_np.mutable_data();

    std::vector<uint64_t> S_bits(N * NB_words);
    std::vector<uint64_t> T_bits(N * NB_words);

    // Pack rows
#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
        for (size_t w = 0; w < NB_words; w++)
        {
            uint64_t word = 0;
            for (size_t b = 0; b < 64; b++)
            {
                size_t j = w * 64 + b;
                if (j < N && S_data[i * N + j])
                    word |= 1ULL << b;
            }
            S_bits[i * NB_words + w] = word;
        }

    // Pack columns
#pragma omp parallel for
    for (size_t j = 0; j < N; j++)
        for (size_t w = 0; w < NB_words; w++)
        {
            uint64_t word = 0;
            for (size_t b = 0; b < 64; b++)
            {
                size_t i = w * 64 + b;
                if (i < N && T_data[i * N + j])
                    word |= 1ULL << b;
            }
            T_bits[j * NB_words + w] = word;
        }

    // Identity row permutation
    std::vector<size_t> row_perm(N);
    for (size_t i = 0; i < N; i++)
        row_perm[i] = i;

    // Call the internal packed router
    phase_router_bitpacked(N, k, NB_words,
                           S_bits.data(), T_bits.data(),
                           row_perm.data(), routes);
}

py::dict pack_and_route(py::array_t<uint8_t> S_np,
                        py::array_t<uint8_t> T_np,
                        size_t k,
                        py::array_t<int> routes_np)
{
    size_t N = S_np.shape(0);
    size_t NB_words = NB(N);

    // Align matrices for proper phase separation
    py::array_t<uint8_t> S_aligned = left_align_rows(S_np);
    py::array_t<uint8_t> T_aligned = top_align_columns(T_np);

    const uint8_t *S_data = (const uint8_t *)S_aligned.data();
    const uint8_t *T_data = (const uint8_t *)T_aligned.data();
    int *routes = routes_np.mutable_data();

    std::vector<uint64_t> S_bits(N * NB_words);
    std::vector<uint64_t> T_bits(N * NB_words);

    double t0_pack = now_ms();

// -------------------- Pack rows in parallel --------------------
#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        for (size_t w = 0; w < NB_words; w++)
        {
            uint64_t word = 0;
            for (size_t b = 0; b < 64; b++)
            {
                size_t j = w * 64 + b;
                if (j < N && S_data[i * N + j])
                    word |= 1ULL << b;
            }
            S_bits[i * NB_words + w] = word;
        }
    }

// -------------------- Pack columns in parallel --------------------
#pragma omp parallel for
    for (size_t j = 0; j < N; j++)
    {
        for (size_t w = 0; w < NB_words; w++)
        {
            uint64_t word = 0;
            for (size_t b = 0; b < 64; b++)
            {
                size_t i = w * 64 + b;
                if (i < N && T_data[i * N + j])
                    word |= 1ULL << b;
            }
            T_bits[j * NB_words + w] = word;
        }
    }

    double t1_pack = now_ms();

    // -------------------- Identity row permutation --------------------
    std::vector<size_t> row_perm(N);
    for (size_t i = 0; i < N; i++)
        row_perm[i] = i;

    // -------------------- Bit-packed routing --------------------
    double t0_route = now_ms();

    phase_router_bitpacked(N, k, NB_words,
                           S_bits.data(), T_bits.data(),
                           row_perm.data(), routes);

    double t1_route = now_ms();

    // -------------------- Compute statistics --------------------
    size_t active = 0;
    for (size_t i = 0; i < N * k; i++)
        active += (routes[i] != -1);

    py::dict d;
    d["N"] = N;
    d["k"] = k;
    d["active_routes"] = active;
    d["packing_time_ms"] = t1_pack - t0_pack;
    d["routing_time_ms"] = t1_route - t0_route;
    d["total_time_ms"] = t1_route - t0_pack;
    d["routes_per_row"] = double(active) / double(N);

    return d;
}

/* =========================
   Pybind11 module
   ========================= */

PYBIND11_MODULE(router, m)
{
    m.def("left_align_rows", &left_align_rows, "Left-align rows of a uint8 matrix");
    m.def("top_align_columns", &top_align_columns, "Top-align columns of a uint8 matrix");

    m.def("pack_bits", &pack_bits, "Pack a uint8 matrix into bit-packed uint64 array");
    m.def("pack_bits_T_permuted", &pack_bits_T_permuted,
          "Pack a uint8 matrix into bit-packed uint64 array with column permutation",
          py::arg("T"), py::arg("col_perm"));

    m.def("route_packed", &route_packed);
    m.def("route_packed_with_stats", &route_packed_with_stats);

    m.def("router", &phase_router_cpp,
          "Compute routes from raw 0/1 matrices S and T",
          py::arg("S"), py::arg("T"), py::arg("k"), py::arg("routes"));

    m.def("pack_and_route", &pack_and_route,
          "Pack raw 0/1 matrices and run bit-packed router in one call (with automatic alignment)",
          py::arg("S"), py::arg("T"), py::arg("k"), py::arg("routes"));

    m.doc() = "Bit-packed phase router for Python / PyTorch";
}
