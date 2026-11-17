import numpy as np
import cvxpy as cp
import sympy as sp
from itertools import product
from scipy.optimize import linprog
import ncpol2sdpa as nc
import hopsy



def get_cg_dim(m, k):
    #Calculates the dimension of the Collins-Gisin space.
    return 2 * m * (k - 1) + (m ** 2) * ((k - 1) ** 2)



def _manual_chebyshev_center(A, b, shrink=1e-7):
    """
    Fallback Chebyshev center via scipy.linprog, used when hopsy's own
    LP-based routine fails numerically (e.g. ValueError: Chebyshev center
    is outside of polytope). b is shrunk slightly so the returned point is
    strictly interior, avoiding a boundary/near-boundary starting point.
    """
    n_vars = A.shape[1]
    norms = np.linalg.norm(A, axis=1)
    norms[norms == 0] = 1.0  # guard against degenerate zero rows

    c = np.zeros(n_vars + 1)
    c[-1] = -1.0  # maximize r  ==  minimize -r

    A_ub = np.hstack([A, norms.reshape(-1, 1)])
    b_ub = b - shrink  # small margin so x_c ends up strictly inside

    bounds = [(None, None)] * n_vars + [(0, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Fallback Chebyshev center LP failed: {res.message}")

    return res.x[:n_vars]



def draw(A, b, n_points, aleatorio=77, max_retries=5):
    # Round the polytope to ensure uniform mixing across anisotropic corners
    problem = hopsy.Problem(A, b)
    rounded_problem = hopsy.round(problem)

    st_point = None
    for attempt in range(max_retries):
        try:
            st_point = hopsy.compute_chebyshev_center(rounded_problem)
            problem = rounded_problem
            break
        except ValueError:
            # Numerically borderline case: rebuilding the rounding can
            # shift the tiny floating-point inconsistency enough to pass.
            rounded_problem = hopsy.round(hopsy.Problem(A, b))

    if st_point is None:
        # Give up on rounding entirely; fall back to a manually computed,
        # strictly-interior Chebyshev center on the original polytope.
        st_point = _manual_chebyshev_center(A, b)
        problem = hopsy.Problem(A, b)

    mc = hopsy.MarkovChain(problem, proposal=hopsy.UniformCoordinateHitAndRunProposal, starting_point=st_point)
    rng = hopsy.RandomNumberGenerator(seed=aleatorio)

    dimensions = A.shape[1]
    optimal_thinning = max(100, dimensions * 10)

    acceptance_rate, states = hopsy.sample(mc, rngs=rng, n_samples=n_points, thinning=optimal_thinning)
    return np.array(states)


def build_ns_matrix(m, k):
    """
    Constructs the Non-Signaling polytope using Collins-Gisin parameterization.
    """
    if m < 1 or k < 2:
        raise ValueError("m must be >= 1 and k must be >= 2")

    n_vars = get_cg_dim(m, k)
    rows = []
    b_ineq = []

    # Helpers to find the correct index in the vector
    def idx_A(x, a): return x * (k - 1) + a
    def idx_B(y, b): return m * (k - 1) + y * (k - 1) + b
    def idx_AB(x, y, a, b):
        return 2 * m * (k - 1) + (x * m + y) * ((k - 1) ** 2) + a * (k - 1) + b

    for x in range(m):
        for y in range(m):
            # 1. P(a,b | x,y) >= 0 (for a, b < k-1)
            for a in range(k - 1):
                for b in range(k - 1):
                    row = np.zeros(n_vars)
                    row[idx_AB(x, y, a, b)] = -1.0
                    rows.append(row)
                    b_ineq.append(0.0)

            # 2. P(a, k-1 | x,y) >= 0 => P_A(a|x) - sum_b P_AB(a,b|x,y) >= 0
            for a in range(k - 1):
                row = np.zeros(n_vars)
                row[idx_A(x, a)] = -1.0
                for b in range(k - 1):
                    row[idx_AB(x, y, a, b)] = 1.0
                rows.append(row)
                b_ineq.append(0.0)

            # 3. P(k-1, b | x,y) >= 0 => P_B(b|y) - sum_a P_AB(a,b|x,y) >= 0
            for b in range(k - 1):
                row = np.zeros(n_vars)
                row[idx_B(y, b)] = -1.0
                for a in range(k - 1):
                    row[idx_AB(x, y, a, b)] = 1.0
                rows.append(row)
                b_ineq.append(0.0)

            # 4. P(k-1, k-1 | x,y) >= 0 => sum_a P_A + sum_b P_B - sum_ab P_AB <= 1
            row = np.zeros(n_vars)
            for a in range(k - 1):
                row[idx_A(x, a)] = 1.0
            for b in range(k - 1):
                row[idx_B(y, b)] = 1.0
            for a in range(k - 1):
                for b in range(k - 1):
                    row[idx_AB(x, y, a, b)] = -1.0
            rows.append(row)
            b_ineq.append(1.0)

    return {
        "A_ineq": np.vstack(rows),
        "b_ineq": np.array(b_ineq)
    }


def L_polytope_extremal(m, k):
    """
    Generates all deterministic local strategies for m measurements and k outcomes.
    """
    strategies = list(product(range(k), repeat=m))
    n_strats = len(strategies)
    n_vars = get_cg_dim(m, k)
    c = np.zeros((n_strats * n_strats, n_vars), dtype=float)

    def idx_A(x, a): return x * (k - 1) + a
    def idx_B(y, b): return m * (k - 1) + y * (k - 1) + b
    def idx_AB(x, y, a, b):
        return 2 * m * (k - 1) + (x * m + y) * ((k - 1) ** 2) + a * (k - 1) + b

    row_idx = 0
    for S_A in strategies:
        for S_B in strategies:
            # Add Alice's marginals
            for x in range(m):
                if S_A[x] < k - 1:
                    c[row_idx, idx_A(x, S_A[x])] = 1.0
            # Add Bob's marginals
            for y in range(m):
                if S_B[y] < k - 1:
                    c[row_idx, idx_B(y, S_B[y])] = 1.0
            # Add Joint distributions
            for x in range(m):
                for y in range(m):
                    if S_A[x] < k - 1 and S_B[y] < k - 1:
                        c[row_idx, idx_AB(x, y, S_A[x], S_B[y])] = 1.0
            row_idx += 1
    return c



def is_convex_combination_cvxpy(V, M, tol=1e-7, return_weights=True):
    """
    Tests whether V lies in the convex hull of the rows of M.

    Reformulated as a phase-1 LP (minimize the L1 residual of the equality
    system over the probability simplex) solved with HiGHS via scipy, rather
    than posing a zero-objective feasibility problem to a conic interior-point
    solver. The original CLARABEL formulation had no objective to provide a
    descent direction, which made it numerically fragile (SolverError /
    "solution may be inaccurate") whenever V sat close to the hull boundary
    -- a common occurrence here since we're testing membership in a polytope
    face, not a generic interior point. The LP residual is always well-posed:
    it degrades gracefully to "small positive distance" instead of failing
    outright, and HiGHS' simplex/IPM path handles this degenerate class of
    problem far more robustly and faster than a conic solver.

    Name kept as `is_convex_combination_cvxpy` for drop-in compatibility with
    existing callers, even though cvxpy is no longer used here.
    """
    V = np.asarray(V, dtype=float)
    M = np.asarray(M, dtype=float)
    k, n = M.shape
    if V.shape[0] != n:
        raise ValueError("Dimensão incompatível.")

    # Variables: [w (k weights on the simplex), s_plus (n), s_minus (n)]
    # Minimize sum(s_plus + s_minus) subject to M.T @ w - V = s_plus - s_minus,
    # sum(w) == 1, w >= 0, s_plus, s_minus >= 0.
    # Optimal value ~ 0  <=>  V is (numerically) in the convex hull of M's rows.
    c = np.concatenate([np.zeros(k), np.ones(2 * n)])

    A_eq = np.zeros((n + 1, k + 2 * n))
    A_eq[:n, :k] = M.T
    A_eq[:n, k:k + n] = -np.eye(n)
    A_eq[:n, k + n:] = np.eye(n)
    A_eq[n, :k] = 1.0

    b_eq = np.concatenate([V, [1.0]])

    try:
        res = linprog(
            c, A_eq=A_eq, b_eq=b_eq,
            bounds=[(0, None)] * (k + 2 * n),
            method="highs",
        )
    except Exception:
        # Extremely defensive: HiGHS should not raise, but guarantee this
        # function never crashes a worker process regardless.
        return (False, None) if return_weights else False

    if not res.success:
        return (False, None) if return_weights else False

    inside = res.fun < tol
    wval = res.x[:k] if inside else None
    return (inside, wval) if return_weights else inside



def NPA_hierarchy(p, m, k, nivel):
    """
    NPA implemented using Projectors for k-outcomes instead of unitary operators.
    """
    num_proj = m * (k - 1)
    E = nc.generate_operators('E', num_proj, hermitian=True)
    F = nc.generate_operators('F', num_proj, hermitian=True)

    def idx_A(x, a): return x * (k - 1) + a
    def idx_B(y, b): return y * (k - 1) + b

    zero = sp.Integer(0)
    subs = {}

    # 1. Projectors square to themselves (E^2 = E) and are orthogonal
    for x in range(m):
        for a in range(k - 1):
            i = idx_A(x, a)
            subs[E[i]**2] = E[i]
            for a_prime in range(k - 1):
                if a != a_prime:
                    subs[E[i]*E[idx_A(x, a_prime)]] = zero

    for y in range(m):
        for b in range(k - 1):
            j = idx_B(y, b)
            subs[F[j]**2] = F[j]
            for b_prime in range(k - 1):
                if b != b_prime:
                    subs[F[j]*F[idx_B(y, b_prime)]] = zero

    # 2. Alice and Bob commute
    for i in range(num_proj):
        for j in range(num_proj):
            subs[E[i]*F[j]] = F[j]*E[i]

    # 3. Equating SDP variables to Collins-Gisin probabilities
    mom_eqs = []
    for x in range(m):
        for a in range(k - 1):
            i = idx_A(x, a)
            mom_eqs.append(E[i] - sp.N(p[i]))

    offset_B = m * (k - 1)
    for y in range(m):
        for b in range(k - 1):
            j = idx_B(y, b)
            mom_eqs.append(F[j] - sp.N(p[offset_B + j]))

    offset_AB = 2 * m * (k - 1)
    for x in range(m):
        for y in range(m):
            for a in range(k - 1):
                for b in range(k - 1):
                    p_idx = offset_AB + (x * m + y) * ((k - 1) ** 2) + a * (k - 1) + b
                    mom_eqs.append(E[idx_A(x, a)] * F[idx_B(y, b)] - sp.N(p[p_idx]))

    sdp = nc.SdpRelaxation(E + F)

    if isinstance(nivel, int):
        level = int(nivel)
        extramonomials = None
    else:
        # Almost-Quantum set using 1+AB basis for Projectors
        level = -1
        one = sp.Integer(1)
        extramonomials = [one] + list(E) + list(F) + [E[i]*F[j] for i in range(num_proj) for j in range(num_proj)]

    sdp.get_relaxation(level=level, objective=0, substitutions=subs, momentequalities=mom_eqs, extramonomials=extramonomials)
    sdp.solve(solver='cvxpy', solverparameters={'solver': cp.CVXOPT})
    return sdp.status



def loading(total_interations, current_step, size=95):

        tamanho_barra = size
        proporcao = current_step / total_interations
        blocos_cheios = int(tamanho_barra * proporcao)
        blocos_vazios = tamanho_barra - blocos_cheios
        barra = ('▌' * blocos_cheios) + (' ' * blocos_vazios)

        return print(f"{barra} {proporcao * 100:.1f}%")
