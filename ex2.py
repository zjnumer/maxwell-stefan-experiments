from common import BIJ, BUILD, Grid, initial, l2, restrict, run, write_rows
import time
import numpy as np
from scipy.optimize import least_squares
from tables import export_table

OUT = BUILD / "spatial"


def logits_to_simplex(y: np.ndarray) -> np.ndarray:
    z = np.concatenate([y, np.zeros((1,) + y.shape[1:])], axis=0)
    z -= np.max(z, axis=0, keepdims=True)
    ez = np.exp(np.maximum(z, -700.0))
    return ez / np.sum(ez, axis=0, keepdims=True)


def simplex_to_logits(rho: np.ndarray) -> np.ndarray:
    return np.log(rho[:-1] / rho[-1:])


def entropy(rho: np.ndarray, h: float) -> float:
    return float(h * np.sum(rho * np.log(rho)))


def entropy_secant(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    out = np.empty_like(a)
    close = np.abs(d) <= 1.0e-10 * np.maximum(1.0, np.abs(b))
    mid = 0.5 * (a[close] + b[close])
    out[close] = np.log(mid) + 1.0
    out[~close] = (a[~close] * np.log(a[~close]) - b[~close] * np.log(b[~close])) / d[
        ~close
    ]
    return out


def positive_predictor(rho_n: np.ndarray, rho_nm1: np.ndarray) -> np.ndarray:
    q = 1.5 * np.log(rho_n) - 0.5 * np.log(rho_nm1)
    q -= np.max(q, axis=0, keepdims=True)
    eq = np.exp(q)
    return eq / np.sum(eq, axis=0, keepdims=True)


def mobility_flux_1d(
    rho_star: np.ndarray,
    mu: np.ndarray,
    bij: np.ndarray,
    h: float,
) -> tuple[np.ndarray, float]:
    m, nx = rho_star.shape
    flux = np.zeros_like(rho_star)
    dissipation = 0.0
    for k in range(nx):
        kp = (k + 1) % nx
        r = 0.5 * (rho_star[:, k] + rho_star[:, kp])
        g = (mu[:, kp] - mu[:, k]) / h
        friction = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                weight = bij[i, j] * r[i] * r[j]
                friction[i, i] += weight
                friction[j, j] += weight
                friction[i, j] -= weight
                friction[j, i] -= weight
        rhs = -r * (g - np.dot(r, g))
        saddle = np.block(
            [
                [friction, r[:, None]],
                [r[None, :], np.zeros((1, 1))],
            ]
        )
        velocity = np.linalg.solve(saddle, np.r_[rhs, 0.0])[:m]
        flux[:, k] = r * velocity
        dissipation += h * float(velocity @ friction @ velocity)
    return flux, dissipation


def divergence_1d(flux: np.ndarray, h: float) -> np.ndarray:
    return (flux - np.roll(flux, 1, axis=1)) / h


def solve_step(
    rho_old: np.ndarray,
    rho_star: np.ndarray,
    tau: float,
    bij: np.ndarray,
    potential: str,
    guess: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    m, nx = rho_old.shape
    h = 1.0 / nx
    if guess is None:
        guess = rho_old
    y0 = simplex_to_logits(guess).ravel()

    def residual(yflat: np.ndarray) -> np.ndarray:
        rho_new = logits_to_simplex(yflat.reshape((m - 1, nx)))
        if potential == "second":
            mu = entropy_secant(rho_new, rho_old)
            mu += tau * np.log(rho_new / rho_old)
        elif potential == "euler":
            mu = np.log(rho_new) + 1.0
        else:
            raise ValueError(potential)
        flux, _ = mobility_flux_1d(rho_star, mu, bij, h)
        equation = rho_new - rho_old + tau * divergence_1d(flux, h)
        return equation[:-1].ravel()

    solution = least_squares(
        residual,
        y0,
        bounds=(-60.0, 60.0),
        xtol=5.0e-12,
        ftol=5.0e-12,
        gtol=5.0e-12,
        max_nfev=800,
    )
    rho_new = logits_to_simplex(solution.x.reshape((m - 1, nx)))
    residual_max = float(np.max(np.abs(residual(solution.x))))
    if (not solution.success) or residual_max > 5.0e-10:
        raise RuntimeError(
            "nonlinear solve failed: "
            f"success={solution.success}, residual={residual_max:.3e}, "
            f"status={solution.status}"
        )
    return rho_new, {
        "residual": residual_max,
        "nfev": float(solution.nfev),
        "success": float(solution.success),
    }


def auxiliary_half_step(
    rho0: np.ndarray,
    tau: float,
    bij: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    return solve_step(rho0, rho0, 0.5 * tau, bij, "euler")


def second_order_step(
    rho_n: np.ndarray,
    rho_nm1: np.ndarray,
    tau: float,
    bij: np.ndarray,
    rho_star_override: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    rho_star = (
        positive_predictor(rho_n, rho_nm1)
        if rho_star_override is None
        else rho_star_override
    )
    rho_new, diagnostic = solve_step(rho_n, rho_star, tau, bij, "second")
    h = 1.0 / rho_n.shape[1]
    mu = entropy_secant(rho_new, rho_n)
    mu += tau * np.log(rho_new / rho_n)
    _, dissipation = mobility_flux_1d(rho_star, mu, bij, h)
    temporal_dissipation = float(
        h * np.sum((rho_new - rho_n) * np.log(rho_new / rho_n))
    )
    entropy_change = entropy(rho_new, h) - entropy(rho_n, h)
    diagnostic.update(
        {
            "entropy": entropy(rho_new, h),
            "entropy_change": entropy_change,
            "entropy_identity": entropy_change
            + tau * temporal_dissipation
            + tau * dissipation,
            "minimum": float(np.min(rho_new)),
            "simplex_error": float(np.max(np.abs(np.sum(rho_new, axis=0) - 1.0))),
        }
    )
    return rho_new, diagnostic


def smooth_initial(nx: int) -> np.ndarray:
    x = (np.arange(nx) + 0.5) / nx
    rho1 = 0.34 + 0.08 * np.cos(2.0 * np.pi * x)
    rho2 = 0.31 + 0.06 * np.sin(2.0 * np.pi * x)
    return np.vstack([rho1, rho2, 1.0 - rho1 - rho2])


def near_vacuum_initial(nx: int) -> np.ndarray:
    x = (np.arange(nx) + 0.5) / nx
    rho1 = 1.0e-6 * (1.0 + 0.45 * np.cos(2.0 * np.pi * x))
    rho2 = 0.46 + 0.05 * np.sin(2.0 * np.pi * x)
    return np.vstack([rho1, rho2, 1.0 - rho1 - rho2])


def run_method(
    rho0: np.ndarray,
    tau: float,
    final_time: float,
    bij: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]], float]:
    steps = int(round(final_time / tau))
    if abs(steps * tau - final_time) > 5.0e-14:
        raise ValueError("final_time must be an integer multiple of tau")
    history: list[dict[str, float]] = []
    h = 1.0 / rho0.shape[1]
    start = time.perf_counter()
    rho_half, auxiliary = auxiliary_half_step(rho0, tau, bij)
    rho1, diagnostic = second_order_step(
        rho0, rho0, tau, bij, rho_star_override=rho_half
    )
    diagnostic["step"] = 1.0
    diagnostic["auxiliary_residual"] = auxiliary["residual"]
    diagnostic["mass_error"] = float(np.max(np.abs(h * np.sum(rho1 - rho0, axis=1))))
    history.append(diagnostic)
    rho_nm1, rho_n = rho0, rho1
    for step in range(2, steps + 1):
        rho_new, diagnostic = second_order_step(rho_n, rho_nm1, tau, bij)
        diagnostic["step"] = float(step)
        diagnostic["mass_error"] = float(
            np.max(np.abs(h * np.sum(rho_new - rho0, axis=1)))
        )
        history.append(diagnostic)
        rho_nm1, rho_n = rho_n, rho_new
    loop_time = time.perf_counter() - start
    return rho_n, history, loop_time


def structure():
    rows = []
    for case, state in (
        ("smooth", smooth_initial(24)),
        ("near_vacuum", near_vacuum_initial(24)),
    ):
        _, history, seconds = run_method(state, 0.00125, 0.01, BIJ)
        rows.extend(dict(case=case, cpuTime=seconds, **row) for row in history)
    write_rows(OUT / "structure_and_near_vacuum.csv", rows)


def refinement():
    rows = []
    diagrows = []
    cases = [(2, (8, 16, 32, 64), 0.002), (3, (8, 16, 32, 64), 0.002)]
    for d, ns, T in cases:
        solutions = []
        for n in ns:
            tau = T / (n // ns[0]) ** 2
            g = Grid(n, d)
            r, diag, t = run(g, initial(n, d), tau, T, "second")
            solutions.append(r)
            print("refinement", d, n, t, flush=True)
            diagrows.append(
                {
                    "d": d,
                    "n": n,
                    "tau": tau,
                    "loop_seconds": t,
                    "minimum": min(a["minimum"] for a in diag),
                    "maximum_residual": max(a["residual"] for a in diag),
                    "simplex": max(a["simplex"] for a in diag),
                    "mass": max(a["mass"] for a in diag),
                    "entropy_identity": max(abs(a["entropy_identity"]) for a in diag),
                }
            )
            write_rows(OUT / ("refinement_diagnostics" + ".csv"), diagrows)
        olddiff = None
        for k, n in enumerate(ns[:-1]):
            diff = l2(
                solutions[k], restrict(solutions[k + 1], ns[k + 1], n, d), Grid(n, d)
            )
            rows.append(
                {
                    "d": d,
                    "n": n,
                    "tau": T / (n // ns[0]) ** 2,
                    "difference": diff,
                    "ratio_order": (
                        float(np.log2(olddiff / diff)) if olddiff else np.nan
                    ),
                }
            )
            olddiff = diff
        write_rows(OUT / ("refinement" + ".csv"), rows)


if __name__ == "__main__":
    structure()
    refinement()
    export_table(2, OUT / "refinement.csv", BUILD)
