# Time scales

It is important to understand the time scales involved in the process of simulating a case and post processing it.

Firstly, the raw results are generated from a CFD simulation using a solver.
Each solver may have a **different time and spatial scales**.
The common time and spatial scales can be listed as:

- LBM scale
- Prototype scale (Fluent and OpenFOAM)
- Full scale (Real scale)
- Normalized scale (Independent)

Output from solvers use LBM or prototype scale, and must be transformed to a normalized scale when generating the coefficients.

The normalization ($t^*$) uses the Convective Scale Time ($CST$):

$$
t^* = \frac{t_{solver}}{CST_{solver}}
$$

And to convert normalized time scale to full scale:

$$
t_{FS} = t^*CST_{FS}
$$

To calculate the CST for time scale transformations, the following expression can be used:

$$
CST = \frac{L_{carac}}{U_{H}} 
$$

Where $L_{carac}$ is the characteristic length scale, and $U_H$ is the velocity at the interest height in the current scaling.
