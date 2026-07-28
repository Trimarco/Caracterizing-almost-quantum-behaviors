# **What is this codes?**

This is mine undergraduate project. Focused on characterizing behaviors within the non-signaling set using linear programming and the NPA hierarchy. By applying these two computational tools, we can classify individual behaviors into their respective sets. Subsequently, we calculate the relative volume of each set using the following ratio:

### $$\text{Relative Volume} = \frac{\text{Categorized correlations in target set}}{\text{Total sorted correlations}}$$

# definitions.py


This file contains:

- `draw` - used to draw behaviors in the NS set
- `build_ns_matrix` - used to construct the matrix that defines the NS set
- `L_polytope_extremal` - creates the local set
- `is_convex_combination_cvxpy` - verifies whether a point can be written as a convex combination of given extremal points (used to check whether a correlation is local)
- `teste_NPA` - tests NPA with debug output
- `NPA_hierarchy` - uses NPA to verify whether correlations belong to the quantum / almost-quantum set
- `loading` - creates a progress bar


# characterizing.py

`main` - holds the step-by-step process.

1. Define the parameters.
2. Build the non-signaling polytope.
3. Create the local polytope, verify whether the behavior belongs to it, and compute the relative volume.
4. Verify whether the behavior belongs to the quantum set and compute its relative volume.
5. Verify whether the behavior belongs to the almost-quantum set and compute its relative volume.


# rel_vol_data.py

First, declare the parameters, then loop to run this process for different measurements.

After completing all measurements, plot the results.

# Acknowledgment

This study was financed by the brazilian National Council for Scientific and Technological Development (CNPq) and by the brazilian Coordination for the Improvement of Higher Education Personnel (CAPES)
