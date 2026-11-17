import numpy as np
from itertools import product
from definitions import build_ns_matrix, L_polytope_extremal, is_convex_combination_cvxpy, draw, NPA_hierarchy


import time


def main(n_pontos, NPA, measurements, outcomes, seed, progress_every=None):
    """
    seed is now a required, caller-supplied argument instead of being drawn
    from `random.randint` inside the worker. The old approach used the
    unseeded global `random` module, which (a) is not reseeded across forked
    ProcessPoolExecutor workers, risking identical draws across rounds, and
    (b) made every run irreproducible -- there was no way to rerun a failed
    correlation or verify a reported volume. The driver is now responsible
    for handing out a distinct seed per (m, round) pair.

    progress_every: print a progress line every this many correlations,
    within each of the three stages. Defaults to ~20 updates per stage
    (qtde // 20, floored at 1). Each line is tagged with (m, k, seed) so
    that output from multiple ProcessPoolExecutor workers running
    concurrently -- which will interleave on stdout -- stays attributable
    to the round it came from.
    """

    qtde = n_pontos
    level_n_NPA = NPA
    m = measurements
    k = outcomes

    tag = f"[m={m} k={k} seed={seed}]"
    step = progress_every or max(1, qtde // 20)

    NS = build_ns_matrix(m, k)

    print(f"{tag} sampling {qtde} correlations...", flush=True)
    t0 = time.time()
    draw_NS = draw(NS["A_ineq"], NS["b_ineq"], qtde, seed)
    print(f"{tag} sampling done ({time.time() - t0:.1f}s)", flush=True)

    # COMPUTE THE RELATIVE VOLUME OF LOCAL SET
    n_local = 0
    v_new = L_polytope_extremal(m, k)

    t0 = time.time()
    for i in range(qtde):
        result_L = is_convex_combination_cvxpy(draw_NS[0][i], v_new)[0]
        if result_L:
            n_local += 1
        if (i + 1) % step == 0 or (i + 1) == qtde:
            print(f"{tag} LOCAL     {i + 1}/{qtde}  ({time.time() - t0:.1f}s elapsed)", flush=True)

    vrel_l = (100 * n_local / qtde)

    # COMPUTE THE RELATIVE VOLUME OF QUANTUM SET
    n_quantum = 0
    t0 = time.time()
    for i in range(qtde):
        result_Q = NPA_hierarchy(draw_NS[0][i], m, k, level_n_NPA)
        if result_Q == 'optimal':
            n_quantum += 1
        if (i + 1) % step == 0 or (i + 1) == qtde:
            print(f"{tag} QUANTUM   {i + 1}/{qtde}  ({time.time() - t0:.1f}s elapsed)", flush=True)

    vrel_q = (100 * n_quantum / qtde)

    # COMPUTE THE RELATIVE VOLUME OF ALMOST-QUANTUM SET
    n_almost_quantum = 0
    t0 = time.time()
    for i in range(qtde):
        result_AQ = NPA_hierarchy(draw_NS[0][i], m, k, 'AQ')
        if result_AQ == 'optimal':
            n_almost_quantum += 1
        if (i + 1) % step == 0 or (i + 1) == qtde:
            print(f"{tag} A-QUANTUM {i + 1}/{qtde}  ({time.time() - t0:.1f}s elapsed)", flush=True)

    vrel_aq = (100 * n_almost_quantum / qtde)

    return [vrel_l, vrel_q, vrel_aq]
