import numpy as np
import cvxpy as cp
import sympy as sp
from itertools import product
import ncpol2sdpa as nc
import hopsy



def draw(A, b, n_points, aleatorio=77):
    problem = hopsy.Problem(A, b)
    
    # Round the polytope to ensure uniform mixing across anisotropic corners
    problem = hopsy.round(problem) 

    st_point = hopsy.compute_chebyshev_center(problem) 
    
    mc = hopsy.MarkovChain(problem, proposal=hopsy.UniformCoordinateHitAndRunProposal, starting_point=st_point)
    rng = hopsy.RandomNumberGenerator(seed=aleatorio)

    dimensions = A.shape[1]
    optimal_thinning = max(100, dimensions * 10)

    acceptance_rate, states = hopsy.sample(mc, rngs=rng, n_samples=n_points, thinning=optimal_thinning)
    return np.array(states)



def build_ns_matrix(m):
    """
    Constructs the matrix for the inequalities p(a,b|x,y) >= 0
    returning A_ineq and b_ineq in the form A_ineq @ vars <= b_ineq,
    where the constant column was removed from A_ineq and placed in b_ineq.

    Returns a dictionary with:
        - coeffs: original matrix (each row is 1/4*(1 + a A_x + b B_y + ab A_xB_y))
        - A_ineq: matrix without the constant column, shape (4*m*m, n_vars-1)
        - b_ineq: rhs vector, shape (4*m*m,)
        - var_names: variable names (without "const")
        - rows_info: list of (a,b,x,y) for each row
    """
    if m < 1:
        raise ValueError("m must be >= 1")

    n_vars = 1 + 2*m + m*m  # adds 'const'
    rows = []
    rows_info = []

    # variables names (constant name first)
    var_names_full = ["const"] + [f"A_{x}" for x in range(m)] + [f"B_{y}" for y in range(m)]
    for x in range(m):
        for y in range(m):
            var_names_full.append(f"A{x}B{y}")

    # build up coeffs (like before)
    for x in range(m):
        for y in range(m):
            for a in (-1, 1):
                for b in (-1, 1):
                    coeff = np.zeros(n_vars, dtype=float)
                    coeff[0] = 1.0
                    coeff[1 + x] = (1.0) * a
                    coeff[1 + m + y] = (1.0) * b
                    idx_ab = 1 + 2*m + x*m + y
                    coeff[idx_ab] = (1.0) * (a*b)
                    rows.append(coeff)
                    rows_info.append((a, b, x, y))

    coeffs = np.vstack(rows)  # shape (4*m*m, n_vars)

    # move constant to RHS:
    # desired form: A_ineq @ vars <= b_ineq
    A_ineq = -coeffs[:, 1:]        # removes the constant column (first column)
    b_ineq = coeffs[:, 0].copy()   # RHS = constant coefficient

    # variables names without 'const'
    var_names = var_names_full[1:]

    return {
        "coeffs": coeffs,
        "A_ineq": A_ineq,
        "b_ineq": b_ineq,
        "var_names": var_names,
        "rows_info": rows_info
    }



def L_polytope_extremal(m):
    # Generates all hypercube verticies [-1,1]^(2*m)
    K = np.array(list(product([-1, 1], repeat=2*m)), dtype=int)  # shape (2**(2*m), 2*m)
    
    # Divide in two parts
    K1 = K[:, :m]   # first m columns
    K2 = K[:, m:]   # Lasts m columns

    c = np.zeros((2**(2*m), 2*m + m**2), dtype=int)

    c[:, :m] = K1
    c[:, m:2*m] = K2

    for j in range(len(K)):
        for i in range(m):
            c[j,2*m+i*m:(3+i)*m] = K1[j,i]*K2[j,:]
    return c



def is_convex_combination_cvxpy(V, M, tol=1e-7, return_weights=True):
    V = np.asarray(V, dtype=float)
    M = np.asarray(M, dtype=float)
    k, n = M.shape
    if V.shape[0] != n:
        raise ValueError("Dimensão incompatível.")
    w = cp.Variable(k)
    constraints = [cp.sum(w) == 1, w >= 0, V == M.T @ w]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.CLARABEL)  # or another avaliable solver
    if prob.status not in ("infeasible", "infeasible_inaccurate"):
        wval = w.value
        # numerical tolerance
        U = (M.T) @ wval
        if np.allclose(U, V, atol=tol):
            return (True, wval) if return_weights else True
    return (False, None) if return_weights else False



def teste_NPA(p, m, nivel):
    """
    Verifies if the p vector satisfies the NPA level for m mesuraments.
    Returns True is satisfies, False otherwise.
    """
    # Declaring the operators for the parties
    A = nc.generate_operators('A', m, hermitian=True)
    B = nc.generate_operators('B', m, hermitian=True)

    # Making the operators unitary and commutativity
    subs = {A[i] ** 2: 1 for i in range(m)}  # Unitary
    subs.update({B[i] ** 2: 1 for i in range(m)})  # Unitary
    subs.update({A[i]*B[j]: B[j]*A[i] for i in range(m) for j in range(m)})  # Commutativity

    # creating the SDP
    sdp = nc.SdpRelaxation(A + B)

    # special level "1+AB" (almost quantum)
    if type(nivel) != int:

        # Defining the moment equalities based on p vector
        # Assuming p[i] corresponds to the correlators <A_i B_j> 
        cons = np.zeros(2*m + m**2, dtype=object)

        for i in range(m):
            cons[i] = A[i] - p[i]
            cons[m+i] = B[i] - p[m+i]
            for j in range(m):
                idx_ab = 2*m + i*m + j
                cons[idx_ab] = A[i]*B[j] - p[idx_ab]

        # Getting the relaxion due to NPA level
        sdp.get_relaxation(level=1, objective=0, substitutions=subs, momentequalities=cons)
        
        # Solving
        sdp.solve(solver='cvxpy', solverparameters={'solver': cp.CVXOPT})
        
        return sdp.status

    # level "nivel" from hierarchy
    else:
        
        # Defining the moment equalities based on p vector
        # Assuming p[i] corresponds to the correlators <A_i B_j> 
        cons = np.zeros(2*m + m**2, dtype=object)

        for i in range(m):
            cons[i] = A[i] - p[i]
            cons[m+i] = B[i] - p[m+i]
            for j in range(m):
                idx_ab = 2*m + i*m + j
                cons[idx_ab] = A[i]*B[j] - p[idx_ab]

        # Getting the relaxion due to NPA level
        sdp.get_relaxation(level=2, objective=0, substitutions=subs, momentequalities=cons)
        
        # Solving
        sdp.solve(solver='cvxpy', solverparameters={'solver': cp.CVXOPT})
        
        return sdp.status



def NPA_hierarchy(p, m, nivel):
    expected_len = 2*m + m*m
    if len(p) != expected_len:
        raise ValueError(f"p must have length {expected_len}; got {len(p)}")

    A = nc.generate_operators('A', m, hermitian=True)
    B = nc.generate_operators('B', m, hermitian=True)

    one = sp.Integer(1)
    substitutions = {A[i]**2: one for i in range(m)}
    substitutions.update({B[j]**2: one for j in range(m)})
    substitutions.update({A[i]*B[j]: B[j]*A[i] for i in range(m) for j in range(m)})

    mom_eqs = []
    for i in range(m):
        mom_eqs.append(A[i] - sp.N(p[i]))
    for j in range(m):
        mom_eqs.append(B[j] - sp.N(p[m + j]))
    for i in range(m):
        for j in range(m):
            idx = 2*m + i*m + j
            mom_eqs.append(A[i]*B[j] - sp.N(p[idx]))

    sdp = nc.SdpRelaxation(A + B)

    if isinstance(nivel, int):
        # --- QUANTUM SET (NPA Level 2) ---
        level = int(nivel)
        extramonomials = None
    else:
        # --- ALMOST-QUANTUM SET (1+AB) ---
        # level=-1 forces ncpol2sdpa to use ONLY the supplied basis
        level = -1  
        # Basis = {1} + {A_i} + {B_j} + {A_i * B_j}
        extramonomials = [one] + list(A) + list(B) + [A[i]*B[j] for i in range(m) for j in range(m)]

    sdp.get_relaxation(level=level,
                       objective=0,
                       substitutions=substitutions,
                       momentequalities=mom_eqs,
                       extramonomials=extramonomials)

    sdp.solve(solver='cvxpy', solverparameters={'solver': cp.CVXOPT})
    return sdp.status



def loading(total_interations, current_step, size=95):
        
        tamanho_barra = size
        proporcao = current_step / total_interations
        blocos_cheios = int(tamanho_barra * proporcao)
        blocos_vazios = tamanho_barra - blocos_cheios
        barra = ('▌' * blocos_cheios) + (' ' * blocos_vazios)

        return print(f"{barra} {proporcao * 100:.1f}%")
