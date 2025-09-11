from caracterizing import main
from mathplot
import matplotlib.pyplot as plt

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

# Criar o gráfico
plt.plot(m, y[0], marker='o', markersize=3, linestyle='-', linewidth=1, color='blue', label="local")
plt.plot(m, y[1], marker='s', markersize=3, linestyle='--', linewidth=1, color='blueviolet', label="quântico")
plt.plot(m, y[2], marker='^', markersize=3, linestyle='-.', linewidth=1, color='darkgreen', label="quase-quântico")

# Labels e título
plt.xlabel("Qtde de medições por parte")
plt.ylabel("Volume relativo")
plt.title("Variação do volume relativo por conjunto")

# Grid apenas principal
plt.grid(True, which='major', linestyle='-', linewidth=0.5)
plt.minorticks_off()  # desativa subticks

# Ajustar ticks para serem apenas inteiros
plt.xticks([2, 3, 4, 5])
plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Mostrar legenda
plt.legend()

# Salvar o gráfico
plt.savefig("results.pdf")

plt.clf()
# Plot da variação do tamanho dos volumes quantico e quase-quântico
y = np.array(y, dtype=float)
var = y[2]-y[1]

plt.bar(m, var, color='skyblue')

# Adicionando título e rótulos
plt.title('Diferença de volume relativo para o conjunto quase-quântico e quântico')
plt.xlabel('Qtde de medições por parte')
plt.ylabel('Diferença no volume relativo')

# Grid apenas principal
plt.grid(True, which='major', linestyle='-', linewidth=0.5, axis="y", alpha=1)
plt.minorticks_off()  # desativa subticks

# Ajustar ticks para serem apenas inteiros
plt.xticks([2, 3, 4, 5])

# Salvar o gráfico
plt.savefig("variacao.pdf")
