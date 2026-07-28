import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from itertools import product
from definitions import build_ns_matrix, L_polytope_extremal, is_convex_combination_cvxpy, teste_NPA, draw, NPA_hierarchy
import random



def main(n_pontos, NPA, measurements):

	# DEFINING THE PARAMETERS
	#___________________________________________________________________________________________________________________________________#

	qtde = n_pontos  # numbevr of correlations drawed in non-sig polytope 
	level_n_NPA = NPA  # level "n" of NPA hierarchy
	rng = random.randint(1, 10000)  # random number generator
	m = measurements  # number of possible mesuramentsfor each part A and B (2, m, r)

	#___________________________________________________________________________________________________________________________________#





	# BUILDING THE NON-SIG POLYTOPE
	#___________________________________________________________________________________________________________________________________#

	NS = build_ns_matrix(m)
	#print(NS["A_ineq"].shape)
	#print(A_NS)
	#print(b_NS.shape)
	#print(b_NS)

	#___________________________________________________________________________________________________________________________________#





	# DRAWING THE CORRELATIONS IN NON-SIG POLYTOPE
	#___________________________________________________________________________________________________________________________________#

	draw_NS = draw(NS["A_ineq"], NS["b_ineq"], qtde, rng)

	#___________________________________________________________________________________________________________________________________#





	# COMPUTE THE RELATIVE VOLUME OF LOCAL SET
	#___________________________________________________________________________________________________________________________________#

	n_local = 0
	v_new = L_polytope_extremal(m)

	for i in range(qtde):
		result_L = is_convex_combination_cvxpy(draw_NS[0][i], v_new)[0]
		#print(result_L)
		if result_L:
			n_local += 1
		else:
			pass

	vrel_l = (100*n_local/qtde)
	#print(f"the local set corresponds to {vrel_l}% of non-sig set.")

	#___________________________________________________________________________________________________________________________________#





	# COMPUTE THE RELATIVE VOLUME OF QUANTUM SET
	#___________________________________________________________________________________________________________________________________#

	n_quantum = 0
	for i in range(qtde):
		result_Q = NPA_hierarchy(draw_NS[0][i], m, level_n_NPA)

		if result_Q == 'optimal':
			n_quantum += 1
		else:
			continue

	vrel_q = (100*n_quantum/qtde)
	#print(f"the quantum set corresponds to {vrel_q}% of non-sig set.")

	#___________________________________________________________________________________________________________________________________#





	# COMPUTE THE RELATIVE VOLUME OF ALMOST-QUANTUM SET
	#___________________________________________________________________________________________________________________________________#

	n_almost_quantum = 0
	for i in range(qtde):
		result_AQ = NPA_hierarchy(draw_NS[0][i], m, 'AQ')
		if result_AQ == 'optimal':
			n_almost_quantum += 1
		else:
			continue

	vrel_aq = (100*n_almost_quantum/qtde)
	#print(f"the almost-quantum set corresponds to {vrel_aq}% of non-sig set.")

	#___________________________________________________________________________________________________________________________________#

	return [vrel_l, vrel_q, vrel_aq]