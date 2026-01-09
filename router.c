#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define WORD_BITS 64
#define NB(N) (((N) + WORD_BITS - 1) / WORD_BITS)

/* --- Timing utility --- */
static inline double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* --- Rotate a bit-packed row/column --- */
static void rotate_bits(const uint64_t *src, size_t N, size_t NB_words, size_t offset, uint64_t *dst)
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
        size_t src2 = (w + NB_words - word_shift - 1 + NB_words) % NB_words;
        dst[w] = (src[src1] >> bit_shift) | (src[src2] << (WORD_BITS - bit_shift));
    }
    /* Mask last word to remove extra bits beyond N */
    size_t rem_bits = N % WORD_BITS;
    if (rem_bits)
        dst[NB_words - 1] &= (1ULL << rem_bits) - 1;
}

/* --- Bit-pack rows --- */
void pack_bits(size_t N, size_t NB_words, const uint8_t *M, uint64_t *bits)
{
    for (size_t i = 0; i < N; i++)
        for (size_t w = 0; w < NB_words; w++)
        {
            uint64_t word = 0;
            for (size_t b = 0; b < 64; b++)
            {
                size_t j = w * 64 + b;
                if (j < N && M[i * N + j])
                    word |= 1ULL << b;
            }
            bits[i * NB_words + w] = word;
        }
}

/* --- Bit-pack columns with pre-permutation --- */
void pack_bits_T_permuted(size_t N, size_t NB_words, const uint8_t *T, const size_t *col_perm, uint64_t *T_bits)
{
    for (size_t j = 0; j < N; j++)
    {
        size_t pj = col_perm[j]; /* pre-permuted column index */
        for (size_t w = 0; w < NB_words; w++)
        {
            uint64_t word = 0;
            for (size_t b = 0; b < 64; b++)
            {
                size_t i = w * 64 + b;
                if (i < N && T[i * N + pj])
                    word |= 1ULL << b;
            }
            T_bits[j * NB_words + w] = word;
        }
    }
}

/* --- Compute row/col offsets with deterministic spreading --- */
static inline size_t hash_index(size_t i, size_t N)
{
    // simple multiplicative hash for deterministic spreading
    return (i * 2654435761UL) % N;
}

static void compute_offsets(size_t N, size_t NB_words,
                            const uint64_t *S_bits, const uint64_t *T_bits,
                            size_t row_offsets[N], size_t col_offsets[N])
{
    row_offsets[0] = col_offsets[0] = 0;
    for (size_t i = 1; i < N; i++)
    {
        size_t row_sum = 0, col_sum = 0;
        // sum of bits in previous row
        for (size_t w = 0; w < NB_words; w++)
            row_sum += __builtin_popcountll(S_bits[(i - 1) * NB_words + w]);

        for (size_t w = 0; w < NB_words; w++)
            for (size_t b = 0; b < WORD_BITS; b++)
            {
                size_t idx = w * WORD_BITS + b;
                if (idx >= N)
                    break;
                if ((T_bits[(i - 1) * NB_words + w] >> b) & 1ULL)
                    col_sum++;
            }

        // combine previous sum with hash for deterministic spread
        row_offsets[i] = (row_sum + hash_index(i, N)) % N;
        col_offsets[i] = (col_sum + hash_index(i, N)) % N;
    }
}

/* --- Main Phase Router --- */
void phase_router_bitpacked(size_t N, size_t k, size_t NB_words,
                            const uint64_t *S_bits, const uint64_t *T_bits,
                            const size_t *row_perm, int *routes)
{
    /* 1. Compute rotation offsets */
    size_t row_offsets[N], col_offsets[N];
    compute_offsets(N, NB_words, S_bits, T_bits, row_offsets, col_offsets);

    /* 2. Pre-rotate columns for efficiency */
    uint64_t *T_rotated = malloc(N * NB_words * sizeof(uint64_t));
#pragma omp parallel for schedule(dynamic)
    for (size_t j = 0; j < N; j++)
        rotate_bits(&T_bits[j * NB_words], N, NB_words, col_offsets[j], &T_rotated[j * NB_words]);

    /* 3. Route each source row */
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < N; i++)
    {
        uint64_t rotated_row[NB(N)]; /* rotated source row */
        size_t pi = row_perm[i];     /* row permutation */
        rotate_bits(&S_bits[pi * NB_words], N, NB_words, row_offsets[pi], rotated_row);

        size_t count = 0;
        for (size_t j = 0; j < N && count < k; j++)
        {
            uint64_t *rot_col = &T_rotated[j * NB_words]; /* rotated target column */
            for (size_t w = 0; w < NB_words && count < k; w++)
            {
                uint64_t mask = rotated_row[w] & rot_col[w]; /* bitwise AND for active targets */
                while (mask && count < k)
                {
                    size_t bit = __builtin_ctzll(mask);
                    size_t pj = w * WORD_BITS + bit;
                    if (pj < N)
                        routes[i * k + count++] = pj;
                    mask &= mask - 1; /* remove least-significant set bit */
                }
            }
        }
        /* 4. Pad remaining entries with -1 if fewer than k targets */
        for (; count < k; count++)
            routes[i * k + count] = -1;
    }

    free(T_rotated);
}

/* --- Main function --- */
int main(void)
{
    const size_t N = 8192, k = 64;
    const size_t NB_words = NB(N);

    /* 1. Allocate matrices */
    uint8_t *S = aligned_alloc(64, N * N * sizeof(uint8_t));
    uint8_t *T = aligned_alloc(64, N * N * sizeof(uint8_t));
    uint64_t *S_bits = aligned_alloc(64, N * NB_words * sizeof(uint64_t));
    uint64_t *T_bits = aligned_alloc(64, N * NB_words * sizeof(uint64_t));
    int *routes = aligned_alloc(64, N * k * sizeof(int));
    size_t *row_perm = malloc(N * sizeof(size_t));
    size_t *col_perm = malloc(N * sizeof(size_t));

    if (!S || !T || !S_bits || !T_bits || !routes || !row_perm || !col_perm)
    {
        perror("malloc");
        return 1;
    }

    /* 2. Populate matrices and permutations */
    srand(1);
    for (size_t i = 0; i < N; i++)
    {
        row_perm[i] = i;
        col_perm[i] = (i * 3) % N;
    }
    for (size_t i = 0; i < N * N; i++)
    {
        S[i] = rand() & 1;
        T[i] = rand() & 1;
    }

    /* 3. Pack matrices into bit-packed representation */
    pack_bits(N, NB_words, S, S_bits);
    pack_bits_T_permuted(N, NB_words, T, col_perm, T_bits);

    /* 4. Run routing */
    double t0 = now_seconds();
    phase_router_bitpacked(N, k, NB_words, S_bits, T_bits, row_perm, routes);
    double t1 = now_seconds();

    printf("Phase Router Bit-packed Test\n----------------------------\n");
    printf("N = %zu, k = %zu\n", N, k);
    printf("Routing time: %.6f ms\n", (t1 - t0) * 1000.0);

    /* optional: basic summary of results */
    size_t total_assigned = 0;
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < k; j++)
            if (routes[i * k + j] != -1)
                total_assigned++;
    printf("Total active routes: %zu\n", total_assigned);
    printf("Average per source: %.2f\n", (double)total_assigned / N);

    /* 5. Free memory */
    free(S);
    free(T);
    free(S_bits);
    free(T_bits);
    free(routes);
    free(row_perm);
    free(col_perm);

    return 0;
}
