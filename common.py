from pathlib import Path
import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
import csv
import time
import numpy as np
from scipy.io import loadmat, savemat
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve, gmres, LinearOperator

ROOT = Path(__file__).resolve().parent
BIJ = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 0.7], [3.0, 0.7, 0.0]])
TOL = 5e-13


def simplex(y):
    z = np.column_stack((y, np.zeros(len(y))))
    z -= z.max(axis=1, keepdims=True)
    z = np.exp(z)
    return z / z.sum(axis=1, keepdims=True)


def tangent(r):
    n = len(r)
    j = r[:, :, None] * (np.eye(3)[:, :2][None, :, :] - r[:, None, :2])
    rr = np.repeat(np.arange(n) * 3, 6) + np.tile(np.repeat(np.arange(3), 2), n)
    cc = np.repeat(np.arange(n) * 2, 6) + np.tile(np.tile(np.arange(2), 3), n)
    return coo_matrix((j.ravel(), (rr, cc)), shape=(3 * n, 2 * n)).tocsr()


def secant(a, b):
    z = (a - b) / b
    close = np.abs(z) < 0.01
    s = np.empty_like(a)
    ds = np.empty_like(a)
    zz = z[close]

    val = np.ones_like(zz)
    der = np.zeros_like(zz)
    for k in range(1, 13):
        val += (-1) ** (k + 1) * zz**k / (k * (k + 1))
        der += (-1) ** (k + 1) * zz ** (k - 1) / (k + 1)
    s[close] = np.log(b[close]) + val
    ds[close] = der / b[close]
    zz = z[~close]
    logs = np.log(a[~close]) - np.log(b[~close])
    s[~close] = np.log(b[~close]) + (a[~close] / b[~close]) * logs / zz
    ds[~close] = (zz - logs) / (b[~close] * zz**2)
    return s, ds


class Grid:
    def __init__(self, n, d=1, length=1.0, bij=BIJ):
        self.n = n
        self.d = d
        self.h = length / n
        self.size = n**d
        self.bij = bij
        ids = np.arange(self.size).reshape((n,) * d)
        self.left = np.tile(ids.ravel(), d)
        self.right = np.concatenate(
            [np.roll(ids, -1, axis=k).ravel() for k in range(d)]
        )
        self.pick = (np.arange(self.size)[:, None] * 3 + np.arange(2)).ravel()

    def assembly(self, blocks):
        l = self.left
        r = self.right
        rr = np.concatenate(
            [
                (
                    a[:, None, None] * 3
                    + np.arange(3)[None, :, None]
                    + np.zeros((1, 1, 3), int)
                ).ravel()
                for a in (l, l, r, r)
            ]
        )
        cc = np.concatenate(
            [
                (
                    a[:, None, None] * 3
                    + np.arange(3)[None, None, :]
                    + np.zeros((1, 3, 1), int)
                ).ravel()
                for a in (l, r, l, r)
            ]
        )
        return coo_matrix(
            (np.concatenate([b.ravel() for b in blocks]), (rr, cc)),
            shape=(3 * self.size, 3 * self.size),
        ).tocsr()

    def mobility(self, r):
        f = (r[self.left] + r[self.right]) / 2
        w = self.bij[None, :, :] * f[:, :, None] * f[:, None, :]
        lap = -w
        ii = np.arange(3)
        lap[:, ii, ii] = w.sum(axis=2)
        rr = f[:, :, None] * f[:, None, :]
        s = np.eye(3)[None, :, :] * f[:, :, None] - rr
        m = s @ np.linalg.solve(lap + rr, s)
        v = m / self.h**2
        return self.assembly((v, -v, -v, v))

    def cflux(self, r, jac=False):
        x = r[self.left]
        y = r[self.right]
        z = (y - x) / x
        close = np.abs(z) < 1e-4
        lm = np.empty_like(x)
        lx = np.empty_like(x)
        ly = np.empty_like(x)
        zz = z[close]
        lm[close] = x[close] * (1 + zz / 2 - zz**2 / 12 + zz**3 / 24 - 19 * zz**4 / 720)

        der = 0.5 - zz / 6 + zz**2 / 8 - 19 * zz**3 / 180
        ly[close] = der
        lx[close] = lm[close] / x[close] - (1 + zz) * der
        zz = z[~close]
        log = np.log(y[~close]) - np.log(x[~close])
        lm[~close] = x[~close] * zz / log
        lx[~close] = (-log + zz) / log**2
        ly[~close] = (log - (y[~close] - x[~close]) / y[~close]) / log**2
        cstar = np.min(self.bij[np.triu_indices(3, 1)])
        kb = self.bij - cstar
        np.fill_diagonal(kb, 0)
        a = -lm[:, :, None] * kb[None, :, :]
        ii = np.arange(3)
        a[:, ii, ii] = cstar + lm @ kb.T
        flux = -np.linalg.solve(a, ((y - x) / self.h)[:, :, None])[:, :, 0]
        div = np.zeros_like(r)
        np.add.at(div, self.left, flux / self.h)
        np.add.at(div, self.right, -flux / self.h)
        if not jac:
            return div

        act = kb[None, :, :] * flux[:, :, None]
        act[:, ii, ii] = -(flux @ kb.T)
        dl = np.linalg.solve(a, np.eye(3)[None, :, :] / self.h - act * lx[:, None, :])
        dr = np.linalg.solve(a, -np.eye(3)[None, :, :] / self.h - act * ly[:, None, :])
        return div, self.assembly(
            (dl / self.h, dr / self.h, -dl / self.h, -dr / self.h)
        )


def step(grid, old, star, tau, method, tol=TOL, forcing=None):

    if forcing is not None:
        forcing = np.asarray(forcing)
        if forcing.shape != old.shape or not np.isfinite(forcing).all():
            raise ValueError("Invalid algebraic forcing")
        if (
            max(np.abs(forcing.sum(axis=0)).max(), np.abs(forcing.sum(axis=1)).max())
            > 1e-12
        ):
            raise ValueError("Algebraic forcing must lie in X_h")
    y = np.log(old[:, :2] / old[:, 2:])
    op = None if method == "cances" else grid.mobility(star)

    def evaluate(yy, jac=False):
        r = simplex(yy)
        if method == "cances":
            v = grid.cflux(r, jac)
            if jac:
                div, jop = v
            else:
                div = v
        else:
            if method == "second":
                mu, der = secant(r, old)
                mu += tau * np.log(r / old)
                der += tau / r
            else:
                mu = np.log(r) + 1
                der = 1 / r
            div = (op @ mu.ravel()).reshape(-1, 3)
            if jac:
                jop = op @ diags(der.ravel())
        residual = (r - old + tau * div).ravel()[grid.pick]
        if forcing is not None:
            residual -= forcing.ravel()[grid.pick]
        if jac:
            jr = tangent(r)
            matrix = jr[grid.pick] + tau * (jop @ jr)[grid.pick]
            return r, residual, matrix
        return r, residual

    threshold = max(tol, 64 * np.finfo(float).eps * tau / grid.h**2)
    for it in range(40):
        r, res, j = evaluate(y, True)
        err = np.max(np.abs(res))
        if err <= threshold:
            return r, err, it
        if grid.d == 3:
            ids = np.arange(grid.size) * 2
            blocks = np.stack(
                [
                    np.asarray(j[ids + a, ids + b]).ravel()
                    for a in range(2)
                    for b in range(2)
                ],
                axis=1,
            ).reshape(-1, 2, 2)
            inv = np.linalg.inv(blocks)
            pre = LinearOperator(
                j.shape,
                matvec=lambda z: np.einsum("nij,nj->ni", inv, z.reshape(-1, 2)).ravel(),
            )
            delta, info = gmres(
                j, -res, M=pre, rtol=1e-11, atol=1e-15, restart=30, maxiter=100
            )
            if info:
                raise RuntimeError(f"GMRES failed: {info}")
        else:
            delta = spsolve(j, -res)
        dy = delta.reshape(-1, 2)
        norm = np.linalg.norm(res)
        scale = 1.0
        while scale >= 2**-20:
            yy = y + scale * dy
            if np.max(np.abs(yy)) < 100 and np.linalg.norm(evaluate(yy)[1]) < norm:
                y = yy
                break
            scale /= 2
        else:
            raise RuntimeError(f"Newton line search failed: {method}, {err:.3e}")
    raise RuntimeError(f"Newton failed: {method}, {err:.3e}")


def run(grid, r0, tau, T, method, diagnostics=True):
    steps = round(T / tau)
    assert abs(steps * tau - T) < 1e-13
    r = r0.copy()
    prev = r0.copy()
    rows = []
    tic = time.perf_counter()
    for k in range(steps):
        if method == "second":
            if k == 0:
                star, _, _ = step(grid, r, r, tau / 2, "huo")
            else:
                star = simplex(
                    1.5 * np.log(r[:, :2] / r[:, 2:])
                    - 0.5 * np.log(prev[:, :2] / prev[:, 2:])
                )
        else:
            star = r
        new, res, it = step(grid, r, star, tau, method)
        if diagnostics:
            h = grid.h**grid.d
            dh = h * np.sum(new * np.log(new) - r * np.log(r))
            identity = np.nan
            if method == "second":
                mu, _ = secant(new, r)
                mu += tau * np.log(new / r)
                identity = (
                    dh
                    + tau * h * np.sum((new - r) * np.log(new / r))
                    + tau * h * np.dot(mu.ravel(), grid.mobility(star) @ mu.ravel())
                )
            rows.append(
                {
                    "step": k + 1,
                    "residual": res,
                    "iterations": it,
                    "minimum": float(new.min()),
                    "simplex": float(np.max(np.abs(new.sum(axis=1) - 1))),
                    "mass": float(np.max(np.abs(h * (new - r0).sum(axis=0)))),
                    "entropy_change": float(dh),
                    "entropy_identity": float(identity),
                    "predictor_minimum": float(star.min()),
                }
            )
        prev, r = r, new
    elapsed = time.perf_counter() - tic
    return r, rows, elapsed


def initial(n, d=1, average=True):
    x = np.meshgrid(*[(np.arange(n) + 0.5) / n] * d, indexing="ij")
    sinc = np.sinc(1 / n) if average else 1.0
    f = np.prod([np.cos(2 * np.pi * u) * sinc for u in x], axis=0)
    g = np.sin(2 * np.pi * x[0]) * sinc
    for u in x[1:]:
        g *= np.cos(2 * np.pi * u) * sinc
    a = 0.34 + 0.08 * f
    b = 0.31 + 0.06 * g
    return np.column_stack((a.ravel(), b.ravel(), (1 - a - b).ravel()))


def restrict(r, nfine, ncoarse, d=1):
    assert nfine % ncoarse == 0
    ratio = nfine // ncoarse
    shape = []
    for _ in range(d):
        shape.extend([ncoarse, ratio])
    a = r.reshape((*shape, 3))
    return a.mean(axis=tuple(range(1, 2 * d, 2))).reshape(-1, 3)


def l2(a, b, grid):
    return float(np.sqrt(grid.h**grid.d * np.sum((a - b) ** 2)))


DATA = ROOT / "data"
BUILD = ROOT / "build"


def load_arrays(path):
    return loadmat(path, squeeze_me=True)


def save_arrays(path, **arrays):
    savemat(path, arrays, do_compression=True)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


REFERENCE_TIME = 0.01
COMPARISON_CELLS = 2048


def reference(n, nsteps):
    directory = BUILD / "pde_reference"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"pde_reference_{n}.mat"
    if path.exists():
        return load_arrays(path)["states"]
    grid = Grid(n)
    tau = REFERENCE_TIME / nsteps
    old, previous = initial(n), initial(n)
    states = [restrict(old, n, COMPARISON_CELLS)]
    tic = time.perf_counter()
    stride = nsteps // 16
    for k in range(nsteps):
        star = (
            step(grid, old, old, tau / 2, "huo")[0]
            if k == 0
            else simplex(
                1.5 * np.log(old[:, :2] / old[:, 2:])
                - 0.5 * np.log(previous[:, :2] / previous[:, 2:])
            )
        )
        new, _, _ = step(grid, old, star, tau, "second")
        previous, old = old, new
        if (k + 1) % stride == 0:
            states.append(restrict(new, n, COMPARISON_CELLS))
        if (k + 1) % 64 == 0:
            print(
                f"reference n={n}: {k+1}/{nsteps}, elapsed={time.perf_counter()-tic:.1f}s",
                flush=True,
            )
    states = np.array(states)
    save_arrays(path, states=states, n=n, nsteps=nsteps, T=REFERENCE_TIME)
    return states
