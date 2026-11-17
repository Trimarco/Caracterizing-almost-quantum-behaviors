import sys
import warnings
import os
import json
import concurrent.futures
from caracterizing import main

os.makedirs(os.path.expanduser("~/.cache/arviz"), exist_ok=True)
warnings.filterwarnings("ignore", message="Constraint #0 contains too many subexpressions")

# colors
GREEN = "\033[92m"
YELLOW = "\033[33m"
RED = "\033[91m"
RESET = "\033[0m"

k_val = 5
level = 2
n_rounds = 48
n_correlations = 5500

# Fixed base so every run is reproducible; combined with m and round index
# to give every (m, round) pair its own distinct seed. Change this only if
# you deliberately want a different sample of correlations.
BASE_SEED = 20260916


def run_simulation(args):
    correlations, lvl, m_val, k, seed = args
    try:
        result = main(correlations, lvl, m_val, k, seed)
        return seed, result, None
    except Exception as e:
        # Never let one bad correlation / solver hiccup take down the whole
        # ProcessPoolExecutor run. Surface the failure and keep going.
        return seed, None, f"{type(e).__name__}: {e}"


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error: Must provide 'm' value.")
        sys.exit(1)

    m_val = int(sys.argv[1])

    local, quantum, almost_quantum = [], [], []
    failures = []

    seeds = [BASE_SEED + 1000 * m_val + r for r in range(n_rounds)]
    args_list = [(n_correlations, level, m_val, k_val, s) for s in seeds]

    # Fetch the CPUs allocated to this specific node
    slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 10))

    partial_path = f"partial_m{m_val}.json"

    with concurrent.futures.ProcessPoolExecutor(max_workers=slurm_cpus) as executor:
        for j, (seed, result, err) in enumerate(executor.map(run_simulation, args_list), start=1):
            if err is not None:
                failures.append({"round": j, "seed": seed, "error": err})
                print(f"\n▁▁ ROUND {RED}{j}{RESET} OF {n_rounds} | m = {m_val} AND k = {k_val} "
                      f"| seed = {seed} ▁▁", flush=True)
                print(f"{RED}FAILED:{RESET} {err}", flush=True)
                continue

            l, q, aq = result
            local.append(l)
            quantum.append(q)
            almost_quantum.append(aq)
            n_ok = len(local)

            print(f"\n▁▁ ROUND {GREEN}{j}{RESET} OF {n_rounds} | m = {m_val} AND k = {k_val} "
                  f"| seed = {seed} | {GREEN}{n_correlations*j}{RESET} OF {n_correlations*n_rounds} CORRELATIONS ▁▁",
                  flush=True)
            print(f"Partial relative volume of LOCAL set: {YELLOW}{sum(local)/n_ok:.2f}{RESET}", flush=True)
            print(f"Partial relative volume of QUANTUM set: {YELLOW}{sum(quantum)/n_ok:.2f}{RESET}", flush=True)
            print(f"Partial relative volume of A-QUANTUM set: {YELLOW}{sum(almost_quantum)/n_ok:.2f}{RESET}", flush=True)

            # Checkpoint after every round so a late failure doesn't lose
            # already-completed rounds.
            with open(partial_path, "w") as f:
                json.dump({
                    "m": m_val, "k": k_val, "level": level,
                    "n_correlations": n_correlations, "n_rounds": n_rounds,
                    "rounds_completed": n_ok, "rounds_failed": len(failures),
                    "local": local, "quantum": quantum, "almost_quantum": almost_quantum,
                    "failures": failures,
                }, f, indent=2)

    n_ok = len(local)
    if n_ok == 0:
        print(f"\n{RED}All {n_rounds} rounds failed for m={m_val}. See {partial_path} for details.{RESET}",
              flush=True)
        sys.exit(1)

    final_results = [sum(local)/n_ok, sum(quantum)/n_ok, sum(almost_quantum)/n_ok]
    print(f"\nFinal Results for m={m_val} ({n_ok}/{n_rounds} rounds succeeded): {final_results}", flush=True)
    if failures:
        print(f"{YELLOW}{len(failures)} round(s) failed -- results averaged over successful rounds only. "
              f"See {partial_path}.{RESET}", flush=True)
