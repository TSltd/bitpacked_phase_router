import numpy as np
import router


def generate_fixed_density(N, density, seed):
    rng = np.random.default_rng(seed)
    M = np.zeros((N, N), dtype=np.uint8)
    nnz = max(1, int(density * N))

    for i in range(N):
        cols = rng.choice(N, size=nnz, replace=False)
        M[i, cols] = 1

    return M


def probe(N, density, k):
    S = generate_fixed_density(N, density, 123)
    T = generate_fixed_density(N, density, 456)
    routes = np.empty((N, k), dtype=np.int32)

    stats = router.pack_and_route(S, T, k, routes, seed=999)

    return stats


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

if __name__ == "__main__":
    N = 4096

    for d in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        for k in [64, 512, 2048, 4096]:
            stats = probe(N, d, k)

            print(
                f"d={d:.2f}, k={k:4d} -> {stats['kernel']}, "
                f"events={stats['events']:.0f}, "
                f"words={stats['words_touched']:.0f}, "
                f"fill={stats['fill_ratio']:.3f}"
            )