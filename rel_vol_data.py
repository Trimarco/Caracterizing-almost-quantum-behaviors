from caracterizing import main
from definitions import loading
import matplotlib.pyplot as plt
import numpy as np

m = [2, 3, 4, 5]
level = 2
n_rounds = 10
n_correlations = 10000

results = [[], [], []]

for i in m:
    local = []
    quantum = []
    almost_quantum = []
    for j in range(1, 1+n_rounds):
        l, q, aq = main(n_correlations, level, i)
        local.append(l)
        quantum.append(q)
        almost_quantum.append(aq)
        print(f"▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ROUND {j} OF {n_rounds} | m = {i} | {n_correlations*j} CORRELATIONS ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁")
        print()
        print(f"Partial relative volume of LOCAL set: {sum(local)/j:.2f}")
        print(f"Partial relative volume of QUATUM set: {sum(quantum)/j:.2f}")
        print(f"Partial relative volume of A-QUANTUM set: {sum(almost_quantum)/j:.2f}")
        print()
        loading(total_interations=n_rounds, current_step=j)
        print()
    results[0].append(sum(local)/j)
    results[1].append(sum(quantum)/j)
    results[2].append(sum(almost_quantum)/j)
    print(results)

# making the graph
plt.plot(m, results[0], marker='o', markersize=3, linestyle='-', linewidth=1, color='blue', label="local")
plt.plot(m, results[1], marker='s', markersize=3, linestyle='--', linewidth=1, color='blueviolet', label="quantum")
plt.plot(m, results[2], marker='^', markersize=3, linestyle='-.', linewidth=1, color='darkgreen', label="almost-quantum")

# labels e title
plt.xlabel("Number of mesurements")
plt.ylabel("Relative vol (%)")
plt.title("Variation of relative volume by set")

# just main grid
plt.grid(True, which='major', linestyle='-', linewidth=0.5)
plt.minorticks_off()  # unactive subticks

# ajusting ticks to be only integers
plt.xticks([2, 3, 4, 5])
plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# showing caption
plt.legend()

# saving
plt.savefig("results.pdf")

plt.clf()
# size variation of quantum and almost-quantum sets plot
y = np.array(results, dtype=float)
var = y[2]-y[1]

plt.bar(m, var, color='skyblue')

# adding title and labels 
plt.title('Size variation of quantum and almost-quantum sets')
plt.xlabel('Number of mesurements')
plt.ylabel('Relative vol (%)')

# just main grid
plt.grid(True, which='major', linestyle='-', linewidth=0.5, axis="y", alpha=1)
plt.minorticks_off()  # unactive subticks

# ajusting ticks to be only integers
plt.xticks([2, 3, 4, 5])

# Saving
plt.savefig("variation.pdf")
