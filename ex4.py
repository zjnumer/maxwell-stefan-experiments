import numpy as np
from scipy.linalg import null_space
from common import BUILD, Grid, initial, simplex, secant, step, save_arrays, write_rows
from tables import export_table

OUT = BUILD / "residual"
TAUS = (0.005, 0.0025, 0.00125, 0.000625)
T = 0.04
AMPLITUDE = 0.1


def main():
    grid = Grid(24)
    rho0 = initial(grid.n)
    q = np.kron(null_space(np.ones((1, grid.n))), null_space(np.ones((1, 3))))
    lap = (
        2 * np.eye(grid.n)
        - np.roll(np.eye(grid.n), 1, axis=0)
        - np.roll(np.eye(grid.n), -1, axis=0)
    ) / grid.h**2
    poisson = q.T @ np.kron(lap, np.eye(3)) @ q
    x = (np.arange(grid.n) + 0.5) * grid.h
    mode = np.cos(2 * np.pi * x)[:, None] * np.array([[1.0, -1.0, 0.0]])
    mode -= mode.mean(axis=0)

    def dual(res):
        r = q.T @ res.ravel()
        return np.sqrt(max(0.0, grid.h * r @ np.linalg.solve(poisson, r)))

    mode /= dual(mode)
    rows, diagnostics = [], []

    def advance(tau, power, tolerance):
        old = rho0.copy()
        previous = old.copy()
        states = [old.copy()]
        residuals = []
        local = []
        auxiliary_residual = (
            np.zeros_like(old) if power == 0 else AMPLITUDE * tau**2.5 * mode
        )
        auxiliary, _, _ = step(
            grid, old, old, tau / 2, "huo", tol=tolerance, forcing=auxiliary_residual
        )

        def repair(r):
            r[:, :2] -= (r[:, :2] - rho0[:, :2]).mean(axis=0)
            r[:, 2] = 1.0 - r[:, 0] - r[:, 1]
            assert r.min() > 0
            return r

        auxiliary = repair(auxiliary)
        for n in range(-1, round(T / tau)):
            if n == -1:
                new, star, dt, mu = auxiliary, rho0, tau / 2, np.log(auxiliary) + 1.0
                before = rho0
            else:
                star = (
                    auxiliary
                    if n == 0
                    else simplex(
                        1.5 * np.log(old[:, :2] / old[:, 2:])
                        - 0.5 * np.log(previous[:, :2] / previous[:, 2:])
                    )
                )
                prescribed = (
                    np.zeros_like(old) if power == 0 else AMPLITUDE * tau**power * mode
                )
                new, _, _ = step(
                    grid, old, star, tau, "second", tol=tolerance, forcing=prescribed
                )
                new = repair(new)
                dt, before = tau, old
                mu, _ = secant(new, old)
                mu += tau * np.log(new / old)
            op = grid.mobility(star)
            residual = new - before + dt * (op @ mu.ravel()).reshape(-1, 3)
            projected = q.T @ residual.ravel()
            zeta = dual(residual)
            eta = np.sqrt(
                max(0.0, grid.h * projected @ np.linalg.solve(q.T @ op @ q, projected))
            )
            dissipation = grid.h * mu.ravel() @ (op @ mu.ravel())
            entropy_ratio = eta / (dt * np.sqrt(dissipation))
            dh = grid.h * np.sum(new * np.log(new) - before * np.log(before))
            assert dh <= 1e-13 and entropy_ratio <= 0.5
            assert np.max(np.abs(grid.h * (new - rho0).sum(axis=0))) < 1e-13
            assert np.max(np.abs(residual.ravel() - q @ projected)) < 1e-13
            local.append(
                dict(
                    tau=tau,
                    power=power,
                    stage=n,
                    zeta=float(zeta),
                    eta=float(eta),
                    entropy_ratio=float(entropy_ratio),
                    entropy_change=float(dh),
                    minimum=float(new.min()),
                )
            )
            residuals.append(residual.copy())
            if n >= 0:
                previous, old = old, new
                states.append(new.copy())
        return np.array(states), np.array(residuals), local

    for tau in TAUS:
        reference, _, _ = advance(tau, 0, 2e-14)
        reference_check, _, _ = advance(tau, 0, 5e-13)
        reference_difference = max(
            np.sqrt(grid.h * np.sum((a - b) ** 2))
            for a, b in zip(reference, reference_check)
        )
        for power in (3, 2):
            states, residuals, local = advance(tau, power, 2e-14)
            error = max(
                np.sqrt(grid.h * np.sum((a - b) ** 2))
                for a, b in zip(states, reference)
            )
            assert reference_difference < 0.01 * error
            budget = np.sqrt(sum(r["zeta"] ** 2 / tau for r in local[1:]))
            rows.append(
                dict(
                    tau=tau,
                    power=power,
                    maximum_density_difference=float(error),
                    residual_budget=float(budget),
                    zeta_a=local[0]["zeta"],
                    maximum_entropy_ratio=max(r["entropy_ratio"] for r in local),
                    reference_check_difference=float(reference_difference),
                )
            )
            diagnostics.extend(local)
            save_arrays(
                OUT / f"trajectory_p{power}_tau{tau:g}.mat",
                states=states,
                residuals=residuals,
                reference=reference,
            )
            print(rows[-1], flush=True)
    for row in rows:
        previous = next(
            (
                r
                for r in rows
                if r["power"] == row["power"] and abs(r["tau"] - 2 * row["tau"]) < 1e-15
            ),
            None,
        )
        row["order"] = (
            float(
                np.log2(
                    previous["maximum_density_difference"]
                    / row["maximum_density_difference"]
                )
            )
            if previous
            else float("nan")
        )
    write_rows(OUT / "summary.csv", rows)
    write_rows(OUT / "steps.csv", diagnostics)
    export_table(4, OUT / "summary.csv", BUILD)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
