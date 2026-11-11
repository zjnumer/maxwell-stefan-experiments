import csv
import math
from common import DATA, Grid, initial, l2, load_arrays, simplex

import numpy as np
from scipy.sparse import eye, kron
from scipy.sparse.linalg import spsolve
from ex3 import BASIS, dual_norm, potential, project


def rows(name):
    with (DATA / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def require(ok, msg):
    if not ok:
        raise AssertionError(msg)


a = rows("validation/accuracy.csv")
require(len(a) == 9, "three methods at three time steps")
require({r["method"] for r in a} == {"huo", "cances", "second"}, "comparison methods")
check = float(a[0]["reference_difference"])
fraction = check / min(float(r["error"]) for r in a)
require(fraction <= 0.01, "1 percent reference rule")
for method in ("huo", "cances", "second"):
    group = [r for r in a if r["method"] == method]
    require(all(int(r["n"]) == 2048 for r in group), "common grid")
    for old, new in zip(group, group[1:]):
        rate = math.log(float(old["error"]) / float(new["error"])) / math.log(
            float(old["tau"]) / float(new["tau"])
        )
        require(abs(rate - float(new["order"])) < 1e-12, "temporal order")
    if method == "second":
        require(all(1.9 < float(r["order"]) < 2.1 for r in group[1:]), "second order")
for own in [r for r in a if r["method"] == "second"]:
    for method in ("huo", "cances"):
        other = next(r for r in a if r["method"] == method and r["tau"] == own["tau"])
        require(
            float(other["error"]) / float(own["error"]) > 50,
            "stable accuracy advantage",
        )
timings = rows("validation/timings.csv")
for r in a:
    times = [
        float(x["loop_seconds"])
        for x in timings
        if x["method"] == r["method"] and x["tau"] == r["tau"]
    ]
    require(
        len(times) == 3 and abs(sorted(times)[1] - float(r["loop_seconds"])) < 1e-12,
        "three-run median",
    )
f = rows("validation/refinement.csv")
require(len(f) == 6, "2D and 3D refinement")
for d in (2, 3):
    group = [r for r in f if int(float(r["d"])) == d]
    require([int(float(r["n"])) for r in group] == [8, 16, 32], "grid levels")
    for old, new in zip(group, group[1:]):
        require(
            abs(
                math.log2(float(old["difference"]) / float(new["difference"]))
                - float(new["ratio_order"])
            )
            < 1e-12,
            "grid order",
        )
    require(float(group[-1]["ratio_order"]) > 1.9, "fine spatial order")
diagnostics = rows("validation/refinement_diagnostics.csv")
require(
    any(int(float(r["d"])) == 3 and int(float(r["n"])) == 64 for r in diagnostics),
    "full 64 cubed run",
)
for r in diagnostics:
    require(
        float(r["minimum"]) > 0 and float(r["simplex"]) < 1e-14,
        "positive multidimensional simplex",
    )
    require(
        float(r["mass"]) < 1e-10 and float(r["entropy_identity"]) < 1e-10,
        "multidimensional structures",
    )
c = rows("validation/certificates.csv")
for r in c:
    require(float(r["minus_error"]) <= float(r["eta"]), "negative norm certificate")
    require(float(r["l2_error"]) <= float(r["l2_bound"]), "L2 certificate")
    require(
        float(r["raw_l2_error"]) <= float(r["raw_l2_bound"]), "raw-state mass repair"
    )
    require(float(r["inverse_residual"]) < 1e-18, "restricted inverse")
for r in rows("structure_and_near_vacuum.csv"):
    require(
        float(r["minimum"]) > 0 and abs(float(r["entropy_identity"])) < 1e-10,
        "retained structures",
    )
print("Archived temporal and spatial results verified")
print("reference_fraction=", fraction)
print("temporal_orders=", [float(r["order"]) for r in a if r["method"] == "second"][1:])
print(
    "fine_grid_orders=",
    [float(r["ratio_order"]) for r in f if int(float(r["n"])) == 32],
)


joint = rows("joint_accuracy_cost/summary.csv")
diagnostic = rows("joint_accuracy_cost/steps.csv")
times = rows("joint_accuracy_cost/timings.csv")
out = DATA / "joint_accuracy_cost"
reference = load_arrays(out / "pde_reference_32768.mat")["states"]
coarse = load_arrays(out / "pde_reference_16384.mat")["states"]
g = Grid(2048)
r_end = l2(reference[-1], coarse[-1], g)
require(
    all(
        math.isclose(r_end, float(r["reference_difference"]), rel_tol=1e-10) for r in a
    ),
    "terminal PDE reference difference",
)
r0 = initial(2048)
q = kron(eye(2048, format="csr"), BASIS, format="csr")


def trajectory_errors(states, ref, tau):
    delta = states - ref
    density = np.sqrt(g.h * np.sum(delta**2, axis=(1, 2))).max()
    total = 0.0
    for k in range(len(states) - 1):
        mu, _ = potential(states[k + 1], states[k], tau)
        mur, _ = potential(ref[k + 1], ref[k], tau)
        z = mu - mur
        grad = (np.roll(z, -1, axis=0) - z) / g.h
        grad -= grad.mean(axis=1, keepdims=True)
        total += tau * g.h * np.sum(grad**2)
    return density, np.sqrt(total)


for row in joint:
    tau = float(row["tau"])
    mode = row["mode"]
    stride = round(tau / 0.000625)
    data = load_arrays(out / f"trajectory_{mode}_{tau:g}.mat")
    states, aux = data["states"], data["auxiliary"]
    error = trajectory_errors(states, reference[::stride], tau)
    check = trajectory_errors(reference[::stride], coarse[::stride], tau)
    for field, value, delta in zip(("density", "gradient"), error, check):
        require(
            abs(float(row[field]) - value) < 1e-15, "trajectory error recomputation"
        )
        require(delta < 0.01 * value, "trajectory reference precision")
    own = [s for s in diagnostic if s["mode"] == mode and float(s["tau"]) == tau]
    require(len(own) == len(states), "all auxiliary and main diagnostics")
    require(
        sum(int(s["iterations"]) for s in own) == int(row["iterations"]),
        "correction count",
    )
    tt = [
        float(s["seconds"])
        for s in times
        if s["mode"] == mode and float(s["tau"]) == tau
    ]
    require(len(tt) == 3 and np.median(tt) == float(row["seconds"]), "joint median")
    for index, stored in enumerate(own):
        isaux = index == 0
        k = index - 1
        dt = tau / 2 if isaux else tau
        old, new = (r0, aux) if isaux else (states[k], states[k + 1])
        star = (
            r0
            if isaux
            else (
                aux
                if k == 0
                else simplex(
                    1.5 * np.log(old[:, :2] / old[:, 2:])
                    - 0.5 * np.log(states[k - 1, :, :2] / states[k - 1, :, 2:])
                )
            )
        )
        op = g.mobility(star)
        mu, _ = potential(new, old, dt, isaux)
        residual = new - old + dt * (op @ mu.ravel()).reshape(-1, 3)
        zeta = dual_norm(residual, g.h)
        require(abs(zeta - float(stored["zeta"])) < 1e-13, "fixed residual norm")
        rhs = np.asarray(q.T @ project(residual).ravel()).ravel()
        reduced = (q.T @ op @ q).tocsr()
        inv = spsolve(reduced[:-2, :-2], rhs[:-2])
        eta = np.sqrt(max(0.0, g.h * np.dot(rhs[:-2], inv)))
        require(abs(eta - float(stored["eta"])) < 1e-12, "weighted residual norm")
        budget = (np.sqrt(tau) if isaux else tau) * (tau**2 + g.h**2)
        diss = g.h * np.dot(mu.ravel(), op @ mu.ravel())
        require(zeta <= budget and eta <= 0.5 * dt * np.sqrt(diss), "joint acceptance")
        dh = g.h * np.sum(new * np.log(new) - old * np.log(old))
        require(dh < 0 and new.min() > 0, "entropy and positivity")
        require(
            np.max(np.abs(g.h * (new - r0).sum(axis=0))) < 1e-12, "trajectory masses"
        )
        require(
            np.max(np.abs(residual - project(residual))) < 1e-12, "tangent residual"
        )
accum = rows("residual_accumulation/summary.csv")
accum_steps = rows("residual_accumulation/steps.csv")
for row in accum:
    tau = float(row["tau"])
    power = int(row["power"])
    z = load_arrays(
        DATA / "residual_accumulation" / f"trajectory_p{power}_tau{tau:g}.mat"
    )
    steps = [
        s for s in accum_steps if float(s["tau"]) == tau and int(s["power"]) == power
    ]
    require(
        [int(s["stage"]) for s in steps] == list(range(-1, len(z["states"]) - 1)),
        "all controlled-residual steps",
    )
    budget = math.sqrt(sum(float(s["zeta"]) ** 2 / tau for s in steps[1:]))
    require(
        math.isclose(budget, float(row["residual_budget"]), rel_tol=1e-12),
        "controlled-residual budget",
    )
    require(
        float(steps[0]["zeta"]) == float(row["zeta_a"])
        and max(float(s["entropy_ratio"]) for s in steps)
        == float(row["maximum_entropy_ratio"]),
        "controlled-residual diagnostics",
    )
    require(
        all(float(s["minimum"]) > 0 and float(s["entropy_change"]) < 0 for s in steps),
        "controlled-residual positivity and entropy decay",
    )
    measured = np.sqrt(
        np.sum((z["states"] - z["reference"]) ** 2, axis=(1, 2)) / 24
    ).max()
    require(
        abs(measured - float(row["maximum_density_difference"])) < 1e-15,
        "accumulated algebraic error",
    )
    require(
        float(row["reference_check_difference"]) < 0.01 * measured,
        "algebraic reference precision",
    )
    previous = next(
        (r for r in accum if int(r["power"]) == power and float(r["tau"]) == 2 * tau),
        None,
    )
    if previous:
        require(
            abs(
                np.log2(float(previous["maximum_density_difference"]) / measured)
                - float(row["order"])
            )
            < 1e-12,
            "algebraic order",
        )
print(
    "Archived stopping and residual trajectories verified",
    len(joint),
    len(diagnostic),
    len(accum),
)
