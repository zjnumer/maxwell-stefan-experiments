import time
import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve, splu
from common import (
    BUILD,
    COMPARISON_CELLS,
    REFERENCE_TIME,
    reference,
    Grid,
    initial,
    simplex,
    secant,
    tangent,
    step,
    l2,
    save_arrays,
    write_rows,
)
from tables import export_table
from scipy.sparse import bmat, csr_matrix

OUT = BUILD / "joint"
TAUS = (0.0025, 0.00125, 0.000625)
T, NCELLS, THETA = REFERENCE_TIME, COMPARISON_CELLS, 0.5
BASIS = np.array(
    [
        [1 / np.sqrt(2), 1 / np.sqrt(6)],
        [-1 / np.sqrt(2), 1 / np.sqrt(6)],
        [0.0, -2 / np.sqrt(6)],
    ]
)


def project(r):
    r = r - r.mean(axis=1, keepdims=True)
    return r - r.mean(axis=0, keepdims=True)


def dual_norm(r, h):
    r = project(r)
    eig = 4 * np.sin(np.pi * np.fft.fftfreq(len(r))) ** 2 / h**2
    eig[0] = 1.0
    coeff = np.fft.fft(r, axis=0)
    coeff[0] = 0.0
    inv = np.fft.ifft(coeff / eig[:, None], axis=0).real
    return np.sqrt(max(0.0, h * np.sum(r * inv)))


def potential(a, b, dt, auxiliary=False):
    if auxiliary:
        return np.log(a) + 1.0, 1 / a
    mu, der = secant(a, b)
    return mu + dt * np.log(a / b), der + dt / a


def feasible(a, r0):
    a = a.copy()
    a[:, :2] -= (a[:, :2] - r0[:, :2]).mean(axis=0)
    a[:, 2] = 1.0 - a[:, 0] - a[:, 1]
    return a


def accepted_step(grid, old, star, dt, tau, r0, q, mode, auxiliary=False):
    op = grid.mobility(star)
    red = (q.T @ op @ q).tocsr()

    inverse = splu(red[:-2, :-2].tocsc())
    eps = tau**2 + grid.h**2
    budget = np.sqrt(tau) * eps if auxiliary else tau * eps
    threshold = max(5e-13, 64 * np.finfo(float).eps * dt / grid.h**2)
    y = np.log(old[:, :2] / old[:, 2:])

    def evaluate(yy, jac=False):
        a = simplex(yy)
        mu, der = potential(a, old, dt, auxiliary)
        residual = a - old + dt * (op @ mu.ravel()).reshape(-1, 3)
        res = residual.ravel()[grid.pick]
        if not jac:
            return a, res
        jr = tangent(a)
        j = jr[grid.pick] + dt * (op @ diags(der.ravel()) @ jr)[grid.pick]
        return a, res, j

    def newton_update(yy, raw, j):
        delta = spsolve(j, -raw).reshape(-1, 2)
        norm = np.linalg.norm(raw)
        scale = 1.0
        while scale >= 2**-20:
            candidate = yy + scale * delta
            if (
                np.max(np.abs(candidate)) < 100
                and np.linalg.norm(evaluate(candidate)[1]) < norm
            ):
                return candidate
            scale /= 2
        raise RuntimeError(f"Line search failed: {mode}, tau={tau}")

    for it in range(40):
        trial, raw, j = evaluate(y, True)
        tight_pass = np.max(np.abs(raw)) <= threshold

        if mode == "tight" and not tight_pass:
            y = newton_update(y, raw, j)
            continue
        a = feasible(trial, r0)
        if a.min() <= 0:
            raise RuntimeError("Mass repair lost positivity")
        mu, _ = potential(a, old, dt, auxiliary)
        residual = a - old + dt * (op @ mu.ravel()).reshape(-1, 3)
        projected = project(residual)
        zeta = dual_norm(projected, grid.h)
        rhs = np.asarray(q.T @ projected.ravel()).ravel()
        psi = inverse.solve(rhs[:-2])
        eta = np.sqrt(max(0.0, grid.h * np.dot(rhs[:-2], psi)))
        dissipation = grid.h * np.dot(mu.ravel(), op @ mu.ravel())
        precision_ratio = zeta / budget
        entropy_ratio = eta / (dt * np.sqrt(max(dissipation, 1e-300)))
        joint_pass = precision_ratio <= 1.0 and entropy_ratio <= THETA
        if joint_pass and (mode == "joint" or tight_pass):
            dh = grid.h * np.sum(a * np.log(a) - old * np.log(old))
            work = grid.h * np.sum(residual * mu)
            if auxiliary:
                bregman = grid.h * np.sum(
                    old * np.log(old) - a * np.log(a) - mu * (old - a)
                )
                identity = dh + bregman + dt * dissipation - work
            else:
                temporal = grid.h * np.sum((a - old) * np.log(a / old))
                identity = dh + dt * temporal + dt * dissipation - work
            inverse_defect = np.max(np.abs(red @ np.r_[psi, 0.0, 0.0] - rhs))
            return a, dict(
                iterations=it,
                residual=float(np.max(np.abs(residual))),
                zeta=float(zeta),
                eta=float(eta),
                precision_ratio=float(precision_ratio),
                entropy_ratio=float(entropy_ratio),
                entropy_change=float(dh),
                identity_defect=float(identity),
                minimum=float(a.min()),
                mass=float(np.max(np.abs(grid.h * (a - r0).sum(axis=0)))),
                simplex=float(np.max(np.abs(a.sum(axis=1) - 1))),
                residual_tangent_defect=float(np.max(np.abs(residual - projected))),
                inverse_defect=float(inverse_defect),
            )
        y = newton_update(y, raw, j)
    raise RuntimeError("Newton failed")


def run(tau, mode):
    grid = Grid(NCELLS)
    r0 = initial(NCELLS)
    q = kron(eye(grid.size, format="csr"), BASIS, format="csr")
    states, rows = [r0.copy()], []
    tic = time.perf_counter()
    auxiliary, diag = accepted_step(grid, r0, r0, tau / 2, tau, r0, q, mode, True)
    rows.append(dict(tau=tau, mode=mode, stage="aux", **diag))
    old, previous = r0.copy(), r0.copy()
    for n in range(round(T / tau)):
        star = (
            auxiliary
            if n == 0
            else simplex(
                1.5 * np.log(old[:, :2] / old[:, 2:])
                - 0.5 * np.log(previous[:, :2] / previous[:, 2:])
            )
        )
        new, diag = accepted_step(grid, old, star, tau, tau, r0, q, mode)
        rows.append(dict(tau=tau, mode=mode, stage=n, **diag))
        assert diag["entropy_change"] <= 1e-13
        assert diag["mass"] < 1e-12 and diag["simplex"] < 1e-14
        assert abs(diag["identity_defect"]) < 1e-12
        assert diag["residual_tangent_defect"] < 1e-12
        assert diag["inverse_defect"] < 1e-9
        previous, old = old, new
        states.append(new.copy())
    elapsed = time.perf_counter() - tic
    return np.array(states), auxiliary, rows, elapsed


def errors(states, ref, tau):
    ref = ref[:: round(tau / TAUS[-1])]
    assert ref.shape == states.shape
    h = 1 / NCELLS
    density = np.sqrt(h * np.sum((states - ref) ** 2, axis=(1, 2))).max()
    total = 0.0
    for n in range(len(states) - 1):
        mu, _ = potential(states[n + 1], states[n], tau)
        exact_mu, _ = potential(ref[n + 1], ref[n], tau)
        error = mu - exact_mu
        grad = (np.roll(error, -1, axis=0) - error) / h
        grad -= grad.mean(axis=1, keepdims=True)
        total += tau * h * np.sum(grad**2)
    return float(density), float(np.sqrt(total))


def check_norms():
    grid = Grid(24)
    rng = np.random.default_rng(7)
    res = project(rng.normal(size=(24, 3)))
    lap = (
        2 * np.eye(24)
        - np.roll(np.eye(24), 1, axis=0)
        - np.roll(np.eye(24), -1, axis=0)
    ) / grid.h**2
    expected = np.sqrt(grid.h * np.sum(res * (np.linalg.pinv(lap) @ res)))
    assert abs(dual_norm(res, grid.h) - expected) < 1e-12
    q = kron(eye(24), BASIS).tocsr()
    red = (q.T @ grid.mobility(initial(24)) @ q).toarray()
    rhs = np.asarray(q.T @ res.ravel()).ravel()
    gauge = np.r_[np.linalg.solve(red[:-2, :-2], rhs[:-2]), 0.0, 0.0]
    expected = grid.h * np.dot(rhs, np.linalg.pinv(red) @ rhs)
    assert abs(grid.h * np.dot(rhs, gauge) - expected) < 1e-12
    print("norm checks passed", flush=True)


def main():
    check_norms()
    ref = reference(32768, 1024)
    ref_coarse = reference(16384, 512)
    comparisons, diagnostics, timings = [], [], []

    run(TAUS[0], "tight")
    run(TAUS[0], "joint")
    for mode in ("tight", "joint"):
        previous = None
        for tau in TAUS:
            local = []
            for repeat in range(3):
                states, auxiliary, rows, elapsed = run(tau, mode)
                local.append(elapsed)
                timings.append(
                    dict(mode=mode, tau=tau, repeat=repeat + 1, seconds=elapsed)
                )
            density, gradient = errors(states, ref, tau)
            drho, dmu = errors(ref[:: round(tau / TAUS[-1])], ref_coarse, tau)
            assert drho < 0.01 * density and dmu < 0.01 * gradient
            rec = dict(
                mode=mode,
                tau=tau,
                density=density,
                gradient=gradient,
                density_order=(
                    np.log2(previous[0] / density) if previous else float("nan")
                ),
                gradient_order=(
                    np.log2(previous[1] / gradient) if previous else float("nan")
                ),
                iterations=sum(r["iterations"] for r in rows),
                seconds=float(np.median(local)),
                reference_density_check=drho,
                reference_gradient_check=dmu,
                max_precision_ratio=max(r["precision_ratio"] for r in rows),
                max_entropy_ratio=max(r["entropy_ratio"] for r in rows),
                minimum=min(r["minimum"] for r in rows),
                maximum_mass=max(r["mass"] for r in rows),
                maximum_simplex=max(r["simplex"] for r in rows),
                maximum_identity_defect=max(abs(r["identity_defect"]) for r in rows),
                maximum_entropy_increment=max(r["entropy_change"] for r in rows),
            )
            rec["residual_budget"] = float(
                np.sqrt(sum(r["zeta"] ** 2 / tau for r in rows if r["stage"] != "aux"))
            )
            rec["auxiliary_residual"] = rows[0]["zeta"]
            comparisons.append(rec)
            diagnostics.extend(rows)
            save_arrays(
                OUT / f"trajectory_{mode}_{tau:g}.mat",
                states=states,
                auxiliary=auxiliary,
            )
            previous = density, gradient
            print(rec, flush=True)
    write_rows(OUT / "summary.csv", comparisons)
    write_rows(OUT / "steps.csv", diagnostics)
    write_rows(OUT / "timings.csv", timings)
    export_table(3, OUT / "summary.csv", BUILD)


def certificates():
    from scipy.sparse import kron

    rows = []
    for case in ("smooth", "vacuum"):
        grid = Grid(24)
        old = initial(24)
        if case == "vacuum":
            x = (np.arange(24) + 0.5) / 24
            old[:, 0] = 1e-6 * (1 + 0.45 * np.cos(2 * np.pi * x))
            old[:, 1] = 0.46 + 0.05 * np.sin(2 * np.pi * x)
            old[:, 2] = 1 - old[:, :2].sum(axis=1)
        tau = 0.00125
        star, _, _ = step(grid, old, old, tau / 2, "huo")
        exact, _, _ = step(grid, old, star, tau, "second")
        u, _, _ = step(grid, old, star, tau, "second", tol=1e-7)
        defect = (u - old).mean(axis=0)
        a = u - defect
        assert a.min() > 0
        op = grid.mobility(star)
        mu, _ = secant(a, old)
        mu += tau * np.log(a / old)
        res = a - old + tau * (op @ mu.ravel()).reshape(-1, 3)
        basis = np.array(
            [
                [1 / np.sqrt(2), 1 / np.sqrt(6)],
                [-1 / np.sqrt(2), 1 / np.sqrt(6)],
                [0.0, -2 / np.sqrt(6)],
            ]
        )
        q = kron(diags(np.ones(24)), basis, format="csr")
        red = q.T @ op @ q
        constraints = kron(np.ones((1, 24)), np.eye(2), format="csr")
        augmented = bmat(
            [[red, constraints.T], [constraints, csr_matrix((2, 2))]], format="csr"
        )
        rhs = np.r_[q.T @ res.ravel(), np.zeros(2)]
        solution = spsolve(augmented, rhs)
        eta = float(np.sqrt(max(0, grid.h * np.dot(rhs[:-2], solution[:-2]))))
        bound = eta / (2 * np.sqrt(tau * (0.5 + tau)))
        actual = l2(a, exact, grid)
        erhs = q.T @ (a - exact).ravel()
        einv = spsolve(augmented, np.r_[erhs, np.zeros(2)])[:-2]
        eminus = float(np.sqrt(max(0, grid.h * np.dot(erhs, einv))))
        assert actual <= bound and eminus <= eta
        rows.append(
            {
                "case": case,
                "eta": eta,
                "minus_error": eminus,
                "l2_bound": bound,
                "l2_error": actual,
                "raw_l2_bound": bound + float(np.linalg.norm(defect)),
                "raw_l2_error": l2(u, exact, grid),
                "mass_repair": float(np.linalg.norm(defect)),
                "minimum": float(a.min()),
                "inverse_residual": float(np.max(np.abs(augmented @ solution - rhs))),
            }
        )
    write_rows(OUT / "certificates.csv", rows)
    print(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    certificates()
    main()
