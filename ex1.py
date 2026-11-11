from common import (
    BUILD,
    COMPARISON_CELLS,
    REFERENCE_TIME,
    reference,
    Grid,
    initial,
    l2,
    run,
    write_rows,
)
import numpy as np
from tables import export_table

OUT = BUILD / "temporal"


def accuracy():
    ref_coarse = reference(16384, 512)[-1]
    ref = reference(32768, 1024)[-1]
    grid = Grid(COMPARISON_CELLS)
    r0 = initial(COMPARISON_CELLS)
    check = l2(ref, ref_coarse, grid)
    rows = []
    timings = []
    for method in ("huo", "cances", "second"):
        olderr = None
        for tau in (0.0025, 0.00125, 0.000625):
            ts = []
            for repeat in range(3):
                r, diag, t = run(grid, r0, tau, REFERENCE_TIME, method)
                ts.append(t)
                timings.append(
                    {
                        "method": method,
                        "tau": tau,
                        "repeat": repeat + 1,
                        "loop_seconds": t,
                    }
                )
            err = l2(r, ref, grid)
            rows.append(
                {
                    "method": method,
                    "n": COMPARISON_CELLS,
                    "tau": tau,
                    "error": err,
                    "order": float(np.log2(olderr / err)) if olderr else np.nan,
                    "loop_seconds": float(np.median(ts)),
                    "reference_difference": check,
                    "maximum_residual": max(a["residual"] for a in diag),
                    "minimum": min(a["minimum"] for a in diag),
                    "mass": max(a["mass"] for a in diag),
                    "simplex": max(a["simplex"] for a in diag),
                }
            )
            olderr = err
            print(rows[-1], flush=True)
        write_rows(OUT / "accuracy.csv", rows)
        write_rows(OUT / "timings.csv", timings)
    assert check < 0.01 * min(a["error"] for a in rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    accuracy()
    export_table(1, OUT / "accuracy.csv", BUILD)
