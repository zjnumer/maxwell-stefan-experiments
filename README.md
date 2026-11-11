# Maxwell–Stefan diffusion experiments

This package reproduces Tables 1–4 and the related numerical checks for a second-order variational scheme for Maxwell–Stefan diffusion.

## Environment

Use Python 3.13.9. Install the tested dependencies from the package directory.

~~~sh
python -m pip install numpy==2.3.5 scipy==1.16.3
~~~

## Recompute the results

~~~sh
python ex1.py
python ex2.py
python ex3.py
python ex4.py
~~~

The scripts write to the same `build/` directory. Tables 1 and 3 share a PDE reference built from 16,384 and 32,768 periodic cells with 512 and 1,024 steps, respectively. Each script can run on its own. Running them in order reuses the computed reference. A fresh checkout starts without a reference cache.

| Table | Script | Settings and checks |
| --- | --- | --- |
| 1 | ex1.py | Three schemes on 2,048 periodic cells with final time 0.01. Compares errors against a common reference and records three advance times per case. Also runs the 20,000-cell benchmark. |
| 2 | ex2.py | Two- and three-dimensional grids with 8, 16, 32 and 64 cells per axis. Also checks the 24-cell smooth and near-vacuum states. The 64-cubed run can take several minutes. |
| 3 | ex3.py | Tight solves and joint stopping on the Table 1 grid, with the same three step sizes. Checks density and projected chemical potential gradient errors against the shared reference. Also tests the mass-repaired one-step certificate. |
| 4 | ex4.py | Controlled residuals on 24 cells, final time 0.04, with residual powers 3 and 2. Compares against a tightly solved trajectory on the same grid and time steps. |

Initial states and model parameters are defined in the scripts. The Cancès finite-volume flux adapts the [reference implementation by Clément Cancès, Virginie Ehrlacher, and Laurent Monasse](https://doi.org/10.5281/zenodo.3934286) ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) to periodic connectivity. The Huo update uses the frozen face mobility. The reference is transferred by conservative cell averaging.

The original times were measured on Windows 11 with one linear algebra thread. Tables 1 and 3 use the median of three serial trials. Table 3 uses one warm-up per mode. The advance time includes assembly and solves, including startup. For joint stopping it also includes mass correction, tangent inverse solves and acceptance tests. New runs record their own times, which depend on the machine.

## Export and check the archived results

~~~sh
python tables.py
python verify.py
~~~

The archive exporter writes `build/archive/table1.csv` through `table4.csv`, corresponding to the files in `reference/`. The CSV and MAT files in `data/` hold original timing records, summary values and trajectories. The verifier checks reported orders, timing medians, reference precision, positivity, the simplex and mass constraints, entropy, and the accepted-trajectory residuals.
