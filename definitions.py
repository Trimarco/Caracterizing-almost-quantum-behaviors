import numpy as np
import cvxpy as cp
import sympy as sp
from itertools import product
import ncpol2sdpa as nc
import hopsy





def sorteio(A, b, n_points, aleatorio=77):
	# o problema completo é definido pela distribuição alvo
	# e pelo domínio restrito, definido pela desigualdade "Ax <= b"
	problem = hopsy.Problem(A, b)

	# the run object contains and constructs the markov chains. in the default case, the
	# Run object will have a single chain using the Hit-and-Run proposal algorithm and is
	# set to produce 10,000 samples.
	st_point = hopsy.compute_chebyshev_center(problem) # this function computes the Chebyshev center of the polytope and uses it one as a starting point
	mc = hopsy.MarkovChain(problem, proposal=hopsy.UniformHitAndRunProposal, starting_point=st_point)
	rng= hopsy.RandomNumberGenerator(seed=aleatorio)

	# call sample on the mc and rng objects 
	acceptance_rate, states = hopsy.sample(mc, rngs=rng, n_samples=n_points, thinning=2)
	
	
	return np.array(states)



def build_ns_matrix(m):
    """
    Constrói a matriz para as desigualdades p(a,b|x,y) >= 0
    retornando A_ineq e b_ineq na forma A_ineq @ vars <= b_ineq,
    onde a coluna da constante foi removida de A_ineq e colocada em b_ineq.

    Retorna dicionário com:
      - coeffs: matriz original (cada linha é 1/4*(1 + a A_x + b B_y + ab A_xB_y))
      - A_ineq: matriz sem a coluna constante, shape (4*m*m, n_vars-1)
      - b_ineq: vetor rhs shape (4*m*m,)
      - var_names: nomes das variáveis (sem "const")
      - rows_info: lista (a,b,x,y) de cada linha
    """
    if m < 1:
        raise ValueError("m deve ser >= 1")

    n_vars = 1 + 2*m + m*m  # inclui 'const'
    rows = []
    rows_info = []

    # nomes das variáveis (com constante primeiro)
    var_names_full = ["const"] + [f"A_{x}" for x in range(m)] + [f"B_{y}" for y in range(m)]
    for x in range(m):
        for y in range(m):
            var_names_full.append(f"A{x}B{y}")

    # construir coeffs (como antes)
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

    # mover constante para o RHS:
    # forma desejada: A_ineq @ vars <= b_ineq
    A_ineq = -coeffs[:, 1:]        # remove a coluna da constante (1ª coluna)
    b_ineq = coeffs[:, 0].copy()   # RHS = coeficiente da constante

    # nomes das variáveis sem a 'const'
    var_names = var_names_full[1:]

    return {
        "coeffs": coeffs,
        "A_ineq": A_ineq,
        "b_ineq": b_ineq,
        "var_names": var_names,
        "rows_info": rows_info
    }

'''# Exemplo de uso
if __name__ == "__main__":
    m = 3
    M = build_ns_matrix(m)
    np.set_printoptions(suppress=True, precision=3)
    print("shape coeffs:", M["coeffs"].shape)        # (4*m*m, 1+2*m+m*m)
    print("shape A_ineq (sem const):", M["A_ineq"].shape)  # (4*m*m, n_vars-1)
    print("shape b_ineq:", M["b_ineq"].shape)
    print("var names (sem const):", M["var_names"])
    print(M["A_ineq"])
    print(M["b_ineq"])'''



def L_polytope_extremal(m):
    # Gera todos os vértices do hipercubo [-1,1]^(2*m)
    K = np.array(list(product([-1, 1], repeat=2*m)), dtype=int)  # shape (2**(2*m), 2*m)
    
    # Divide os vértices em duas partes
    K1 = K[:, :m]   # Primeiras m colunas
    K2 = K[:, m:]   # Últimas m colunas

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
    prob.solve(solver=cp.SCS)  # ou outro solver disponível
    if prob.status not in ("infeasible", "infeasible_inaccurate"):
        wval = w.value
        # tolerância numérica
        U = (M.T) @ wval
        if np.allclose(U, V, atol=tol):
            return (True, wval) if return_weights else True
    return (False, None) if return_weights else False



def teste_NPA(p, m, nivel):
    """
    Testa se o vetor p satisfaz o nível NPA indicado para m medições.
    Retorna True se satisfaz, False caso contrário.
    """
    # Declarando operadores para as partes
    A = nc.generate_operators('A', m, hermitian=True)
    B = nc.generate_operators('B', m, hermitian=True)

    # Tornando os operadores unitários e comutativos
    subs = {A[i] ** 2: 1 for i in range(m)}  # Unitary
    subs.update({B[i] ** 2: 1 for i in range(m)})  # Unitary
    subs.update({A[i]*B[j]: B[j]*A[i] for i in range(m) for j in range(m)})  # Commutativity

    # Criando o SDP
    sdp = nc.SdpRelaxation(A + B)

    # nível especial "1+AB" (almost quantum)
    if type(nivel) != int:

        # Definindo as igualdades de momentos com base no vetor p
        # Assumindo p[i] corresponde aos correladores <A_i B_j>
        cons = np.zeros(2*m + m**2, dtype=object)

        for i in range(m):
            cons[i] = A[i] - p[i]
            cons[m+i] = B[i] - p[m+i]
            for j in range(m):
                idx_ab = 2*m + i*m + j
                cons[idx_ab] = A[i]*B[j] - p[idx_ab]

        # Obter a relaxação do nível NPA
        sdp.get_relaxation(level=1, objective=0, substitutions=subs, momentequalities=cons)
        
        # Resolver usando MOSEK
        sdp.solve(solver='mosek')
        
        return sdp.status

    # nivel "nivel" da hierarquia
    else:
        
        # Definindo as igualdades de momentos com base no vetor p
        # Assumindo p[i] corresponde aos correladores <A_i B_j>
        cons = np.zeros(2*m + m**2, dtype=object)

        for i in range(m):
            cons[i] = A[i] - p[i]
            cons[m+i] = B[i] - p[m+i]
            for j in range(m):
                idx_ab = 2*m + i*m + j
                cons[idx_ab] = A[i]*B[j] - p[idx_ab]

        # Obter a relaxação do nível NPA
        sdp.get_relaxation(level=2, objective=0, substitutions=subs, momentequalities=cons)
        
        # Resolver usando MOSEK
        sdp.solve(solver='mosek')
        
        return sdp.status



def hierarquia_NPA(p, m, nivel):

    """
    Test membership of p in an NPA relaxation.
    - p: vector with layout [<A_0>.. <A_{m-1}>, <B_0>.. <B_{m-1}>, <A_i B_j> flattened row-major]
         length must be 2*m + m*m.
    - m: number of measurements per party
    - nivel:
        * if isinstance(nivel, int): treat as standard NPA level = nivel (quantum-approximation).
        * otherwise: treat as the '1+AB' (almost-quantum) mixed-level case (level=1 with AB extra monomials).
    Returns: (bool, status_string) where bool=True means SDP found a feasible solution consistent with p.
    """

    # Sanity check
    expected_len = 2*m + m*m
    if len(p) != expected_len:
        raise ValueError(f"p must have length {expected_len}; got {len(p)}")

    # generate operators
    A = nc.generate_operators('A', m, hermitian=True)
    B = nc.generate_operators('B', m, hermitian=True)

    # substitutions (idempotency / unitary for ±1 outcomes)
    one = sp.Integer(1)
    substitutions = {A[i]**2: one for i in range(m)}
    substitutions.update({B[j]**2: one for j in range(m)})
    substitutions.update({A[i]*B[j]: B[j]*A[i] for i in range(m) for j in range(m)})  # Commutativity

    # build moment equalities list
    mom_eqs = []
    for i in range(m):
        mom_eqs.append(A[i] - sp.N(p[i]))          # <A_i> = p[i]
    for j in range(m):
        mom_eqs.append(B[j] - sp.N(p[m + j]))      # <B_j> = p[m+j]
    for i in range(m):
        for j in range(m):
            idx = 2*m + i*m + j
            mom_eqs.append(A[i]*B[j] - sp.N(p[idx]))  # <A_i B_j> = p[idx]

    sdp = nc.SdpRelaxation(A + B)

    if isinstance(nivel, int):
        # -------------------------
        # QUANTUM-SET (NPA) CASE
        # -------------------------
        level = int(nivel)
        extramonomials = None
    else:
        # -------------------------
        # ALMOST-QUANTUM (1+AB) CASE
        # -------------------------
        level = 1
        extramonomials = [A[i]*B[j] for i in range(m) for j in range(m)]

    # Build relaxation (feasibility objective = 0)
    sdp.get_relaxation(level=level,
                       objective=0,
                       substitutions=substitutions,
                       momentequalities=mom_eqs,
                       extramonomials=extramonomials)

    # Solve (use MOSEK if available)
    sdp.solve(solver='mosek')

    return sdp.status
