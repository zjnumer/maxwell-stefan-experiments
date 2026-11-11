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
    check = l2(ref, ref_coarse, grid)
    rows = []
    timings = []
    for method in ("huo", "cances", "second"):
        olderr = None
        for tau in (0.0025, 0.00125, 0.000625):
            ts = []
            for repeat in range(3):
                r, diag, t = run(
                    grid, initial(COMPARISON_CELLS), tau, REFERENCE_TIME, method
                )
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


def large():
    n = 20000
    length = 2.0
    x = (np.arange(n) + 0.5) * length / n
    raw = np.select(
        [x < 0.25, x < 0.75, x <= 1.25, x < 1.75],
        [0.8, 1.6 * (0.75 - x), 0.0, 1.6 * (x - 1.25)],
        default=0.8,
    )
    eps = 0.01
    r0 = (1 - 3 * eps) * np.column_stack((raw, np.full(n, 0.2), 0.8 - raw)) + eps
    bij = np.array(
        [
            [0.0, 1 / 0.833, 1 / 0.833],
            [1 / 0.833, 0.0, 1 / 0.168],
            [1 / 0.833, 1 / 0.168, 0.0],
        ]
    )
    grid = Grid(n, length=length, bij=bij)
    rows = []
    timings = []
    for method in ("huo", "cances", "second"):
        for repeat in range(3):
            r, diag, t = run(grid, r0, 0.001, 0.01, method)
            timings.append({"method": method, "repeat": repeat + 1, "loop_seconds": t})
        write_rows(OUT / ("large_" + method + ".csv"), diag)
        write_rows(
            OUT / ("large_profile_" + method + ".csv"),
            [
                {
                    "x": float(xx),
                    "rho1": float(rr[0]),
                    "rho2": float(rr[1]),
                    "rho3": float(rr[2]),
                }
                for xx, rr in zip(x, r)
            ],
        )
        rows.append(
            {
                "method": method,
                "n": n,
                "length": length,
                "tau": 0.001,
                "T": 0.01,
                "loop_seconds": float(
                    np.median(
                        [a["loop_seconds"] for a in timings if a["method"] == method]
                    )
                ),
                "minimum": min(a["minimum"] for a in diag),
                "maximum_residual": max(a["residual"] for a in diag),
                "mass": max(a["mass"] for a in diag),
                "simplex": max(a["simplex"] for a in diag),
                "entropy_change": sum(a["entropy_change"] for a in diag),
                "maximum_entropy_increment": max(a["entropy_change"] for a in diag),
                "entropy_identity": max(abs(a["entropy_identity"]) for a in diag),
            }
        )
        print(rows[-1], flush=True)
        write_rows(OUT / "large.csv", rows)
        write_rows(OUT / "large_timings.csv", timings)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    accuracy()
    export_table(1, OUT / "accuracy.csv", BUILD)
    large()
