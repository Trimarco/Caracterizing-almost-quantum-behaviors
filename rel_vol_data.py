from caracterizing import main

m = [2, 3, 4, 5]
nivel = 2
n_rodadas = 100
qtde_pontos = 300

resultados = [[], [], []]

for i in m:
    local = []
    quantum = []
    almost_quantum = []
    for j in range(1, 1+n_rodadas):
        l, q, aq = main(qtde_pontos, nivel, i)
        local.append(l)
        quantum.append(q)
        almost_quantum.append(aq)
        print(f"____________________RODADA {j} DE {n_rodadas} | m = {i} | {qtde_pontos*j} PONTOS____________________")
        print(f"Volume relativo parcial LOCAL: {sum(local)/j:.2f}")
        print(f"Volume relativo parcial QUATUM: {sum(quantum)/j:.2f}")
        print(f"Volume relativo parcial A-QUANTUM: {sum(almost_quantum)/j:.2f}")
        print()
    resultados[0].append(sum(local)/j)
    resultados[1].append(sum(quantum)/j)
    resultados[2].append(sum(almost_quantum)/j)
    print(resultados)
    
