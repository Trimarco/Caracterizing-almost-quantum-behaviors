import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from itertools import product
from definitions import build_ns_matrix, L_polytope_extremal, is_convex_combination_cvxpy, teste_NPA, sorteio, hierarquia_NPA
import random



def main(n_pontos, NPA, measurements):

	# DECLARANDO VARIÁVEIS INICIAIS
	#___________________________________________________________________________________________________________________________________#

	qtde = n_pontos  # número de pontos a serem sorteados dentro do politopo não sinalizante
	nivel_n_NPA = NPA  # nível "p" da hierarquia NPA
	rng = random.randint(1, 100)  # número aleatório entre 1 e 100 para ser usado como "seed" no sorteio
	m = measurements  # número de medições possíveis para cada parte A e B (2, m, r)

	#___________________________________________________________________________________________________________________________________#





	# DEFININDO O POLITPO NÃO SINALIZANTE
	#___________________________________________________________________________________________________________________________________#

	NS = build_ns_matrix(m)
	#print(NS["A_ineq"].shape)
	#print(A_NS)
	#print(b_NS.shape)
	#print(b_NS)

	#___________________________________________________________________________________________________________________________________#





	# SORTEANDO PONTOS DENTRO DO POLITPO NÃO SINALIZANTE
	#___________________________________________________________________________________________________________________________________#

	sorteio_NS = sorteio(NS["A_ineq"], NS["b_ineq"], qtde, rng)
	#print(sorteio_NS)
	#print(type(sorteio_NS))
	#print(sorteio_NS.shape)

	#___________________________________________________________________________________________________________________________________#





	# CALCULA O VOLUME RELATIVO DO POLITOPO LOCAL
	#___________________________________________________________________________________________________________________________________#

	n_locais = 0
	v_new = L_polytope_extremal(m)

	for i in range(qtde):
		result_L = is_convex_combination_cvxpy(sorteio_NS[0][i], v_new)[0]
		#print(result_L)
		if result_L:
			n_locais += 1
		else:
			pass

	vrel_l = (100*n_locais/qtde)
	#print(f"O politopo local corresponde a {vrel_l}% do politopo não-sinalizante.")

	#___________________________________________________________________________________________________________________________________#





	# CALCULA O VOLUME RELATIVO DO CONJUNTO QUÂNTICO
	#___________________________________________________________________________________________________________________________________#

	n_quanticos = 0
	for i in range(qtde):
		result_Q = hierarquia_NPA(sorteio_NS[0][i], m, nivel_n_NPA)

		if result_Q == 'optimal':
			n_quanticos += 1
		else:
			continue

	vrel_q = (100*n_quanticos/qtde)
	#print(f"O conjunto quântico corresponde a {vrel_q}% do politopo não-sinalizante.")

	#___________________________________________________________________________________________________________________________________#





	# CALCULA O VOLUME RELATIVO DO CONJUNTO QUASE-QUÂNTICO
	#___________________________________________________________________________________________________________________________________#

	n_quase_quanticos = 0
	for i in range(qtde):
		result_AQ = hierarquia_NPA(sorteio_NS[0][i], m, 'AQ')
		if result_AQ == 'optimal':
			n_quase_quanticos += 1
		else:
			continue

	vrel_aq = (100*n_quase_quanticos/qtde)
	#print(f"O conjunto quase-quântico corresponde a {vrel_aq}% do politopo não-sinalizante.")

	#___________________________________________________________________________________________________________________________________#

	return [vrel_l, vrel_q, vrel_aq]