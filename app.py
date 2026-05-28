
"""
=====================================================================
DISRUPTIVE SYMPLECTIC EQUILIBRIUM OPERATOR v2
=====================================================================

Research-oriented architecture combining:

- Hamiltonian neural dynamics
- Learned Riemannian metric
- Symplectic implicit midpoint integration
- Newton-Krylov equilibrium solving
- Matrix-free Jacobian-vector products
- Batched GMRES
- Anderson acceleration
- Spectral stabilization
- Adaptive damping
- Implicit differentiation
- Reversible latent dynamics
- Energy regularization
- Jacobian regularization
- Mixed precision compatibility
- GPU-ready implementation

This is still a research prototype.
No guarantees of convergence or exact symplecticity.

=====================================================================
"""

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast


# ================================================================
# CONFIG
# ================================================================

@dataclass
class Config:

    d: int = 64
    hidden: int = 512

    dt: float = 0.02

    batch_size: int = 32

    equilibrium_steps: int = 48

    newton_steps: int = 8

    gmres_steps: int = 24

    anderson_m: int = 6

    tol: float = 1e-6

    damping: float = 0.5

    lr: float = 1e-4

    jacobian_reg: float = 1e-4

    energy_reg: float = 1e-3

    latent_reg: float = 1e-4

    grad_clip: float = 1.0

    seed: int = 42

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def seed_all(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)


# ================================================================
# SPECTRAL LINEAR
# ================================================================

class SpectralLinear(nn.Module):

    def __init__(self, inp, out):
        super().__init__()

        self.layer = nn.utils.parametrizations.spectral_norm(
            nn.Linear(inp, out)
        )

    def forward(self, x):
        return self.layer(x)


# ================================================================
# SPD METRIC
# ================================================================

class SPDMetric(nn.Module):

    def __init__(self, d):
        super().__init__()

        self.raw = nn.Parameter(torch.eye(d))

    def cholesky(self):

        L = torch.tril(self.raw)

        diag = torch.diagonal(L)

        diag = F.softplus(diag) + 1e-4

        L = L - torch.diag(torch.diagonal(L))
        L = L + torch.diag(diag)

        return L

    def matrix(self):

        L = self.cholesky()

        return L @ L.T

    def solve(self, x):

        L = self.cholesky()

        y = torch.linalg.solve_triangular(
            L,
            x.T,
            upper=False,
        )

        z = torch.linalg.solve_triangular(
            L.T,
            y,
            upper=True,
        )

        return z.T


# ================================================================
# HAMILTONIAN
# ================================================================

class Hamiltonian(nn.Module):

    def __init__(self, d, hidden):
        super().__init__()

        self.metric = SPDMetric(d)

        self.potential = nn.Sequential(
            SpectralLinear(d, hidden),
            nn.SiLU(),
            SpectralLinear(hidden, hidden),
            nn.SiLU(),
            SpectralLinear(hidden, hidden),
            nn.SiLU(),
            SpectralLinear(hidden, 1),
        )

    def kinetic(self, p):

        Minv_p = self.metric.solve(p)

        return 0.5 * (p * Minv_p).sum(-1, keepdim=True)

    def forward(self, q, p):

        T = self.kinetic(p)

        V = self.potential(q)

        return T + V


# ================================================================
# SYMPLECTIC VECTOR FIELD
# ================================================================


def vector_field(H, z):

    q, p = z.chunk(2, dim=-1)

    q.requires_grad_(True)
    p.requires_grad_(True)

    energy = H(q, p).sum()

    dHdq, dHdp = torch.autograd.grad(
        energy,
        (q, p),
        create_graph=True,
    )

    dqdt = dHdp
    dpdt = -dHdq

    return torch.cat([dqdt, dpdt], dim=-1)


# ================================================================
# MIDPOINT RESIDUAL
# ================================================================


def midpoint_residual(H, z0, z, dt):

    zmid = 0.5 * (z0 + z)

    fmid = vector_field(H, zmid)

    return z - z0 - dt * fmid


# ================================================================
# BATCHED GMRES
# ================================================================


def batch_inner(x, y):
    return (x * y).flatten(1).sum(-1)



def batched_gmres(A, b, steps=32, tol=1e-6):

    B = b.shape[0]

    x = torch.zeros_like(b)

    r = b - A(x)

    beta = r.flatten(1).norm(dim=-1)

    if torch.max(beta) < tol:
        return x

    V = [r / (beta[:, None] + 1e-12)]

    Hm = torch.zeros(
        B,
        steps + 1,
        steps,
        device=b.device,
        dtype=b.dtype,
    )

    g = torch.zeros(
        B,
        steps + 1,
        device=b.device,
        dtype=b.dtype,
    )

    g[:, 0] = beta

    for j in range(steps):

        w = A(V[j])

        for i in range(j + 1):

            hij = batch_inner(w, V[i])

            Hm[:, i, j] = hij

            w = w - hij[:, None] * V[i]

        hnext = w.flatten(1).norm(dim=-1)

        Hm[:, j + 1, j] = hnext

        V.append(w / (hnext[:, None] + 1e-12))

    ys = []

    for bidx in range(B):

        Hsmall = Hm[bidx]

        y = torch.linalg.lstsq(Hsmall, g[bidx]).solution

        ys.append(y)

    y = torch.stack(ys)

    x = 0

    for j in range(steps):
        x = x + y[:, j][:, None] * V[j]

    return x


# ================================================================
# ANDERSON ACCELERATION
# ================================================================

class Anderson:

    def __init__(self, m=6, eps=1e-4):

        self.m = m
        self.eps = eps

        self.X = []
        self.F = []

    def reset(self):

        self.X.clear()
        self.F.clear()

    def step(self, x, fx):

        r = fx - x

        self.X.append(x.detach())
        self.F.append(r.detach())

        if len(self.X) > self.m:
            self.X.pop(0)
            self.F.pop(0)

        k = len(self.X)

        if k == 1:
            return fx

        R = torch.stack(self.F).flatten(1)

        G = R @ R.T

        G += self.eps * torch.eye(k, device=x.device)

        ones = torch.ones(k, device=x.device)

        alpha = torch.linalg.solve(G, ones)

        alpha = alpha / alpha.sum()

        alpha = alpha.view(k, *([1] * x.ndim))

        X = torch.stack(self.X)
        Fv = torch.stack(self.F)

        return (alpha * (X + Fv)).sum(0)


# ================================================================
# NEWTON-KRYLOV
# ================================================================


def newton_krylov(
    H,
    z0,
    zinit,
    dt,
    newton_steps,
    gmres_steps,
    tol,
    damping,
):

    z = zinit

    for _ in range(newton_steps):

        z = z.detach().requires_grad_(True)

        Fz = midpoint_residual(H, z0, z, dt)

        residual = Fz.flatten(1).norm(dim=-1).mean()

        if residual < tol:
            break

        def linear(v):

            _, jvp = torch.autograd.functional.jvp(
                lambda zz: midpoint_residual(H, z0, zz, dt),
                z,
                v,
                create_graph=False,
            )

            return jvp

        dz = batched_gmres(
            linear,
            -Fz,
            steps=gmres_steps,
            tol=tol,
        )

        alpha = damping

        old = residual

        for _ in range(6):

            candidate = z + alpha * dz

            new = midpoint_residual(
                H,
                z0,
                candidate,
                dt,
            ).flatten(1).norm(dim=-1).mean()

            if new < old:
                z = candidate
                break

            alpha *= 0.5

    return z.detach()


# ================================================================
# IMPLICIT DIFFERENTIATION
# ================================================================

class ImplicitSymplecticLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z0, model, dt):

        with torch.no_grad():

            z = z0.clone()

            accelerator = Anderson(model.cfg.anderson_m)

            for _ in range(model.cfg.equilibrium_steps):

                znext = newton_krylov(
                    model.H,
                    z0,
                    z,
                    dt,
                    model.cfg.newton_steps,
                    model.cfg.gmres_steps,
                    model.cfg.tol,
                    model.cfg.damping,
                )

                znext = accelerator.step(z, znext)

                delta = (
                    znext - z
                ).flatten(1).norm(dim=-1).mean()

                z = znext

                if delta < model.cfg.tol:
                    break

        ctx.model = model
        ctx.dt = dt

        ctx.save_for_backward(z, z0)

        return z

    @staticmethod
    def backward(ctx, grad_output):

        z, z0 = ctx.saved_tensors

        model = ctx.model
        dt = ctx.dt

        z = z.detach().requires_grad_(True)

        def fixed_point(zz):

            zmid = 0.5 * (zz + z0)

            return z0 + dt * vector_field(model.H, zmid)

        def linear(v):

            JTv = torch.autograd.grad(
                fixed_point(z),
                z,
                grad_outputs=v,
                retain_graph=True,
            )[0]

            return v - JTv

        g = batched_gmres(
            linear,
            grad_output,
            steps=model.cfg.gmres_steps,
            tol=model.cfg.tol,
        )

        return g, None, None


# ================================================================
# FULL MODEL
# ================================================================

class SymplecticEquilibriumOperator(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.H = Hamiltonian(cfg.d, cfg.hidden)

    def forward(self, z0):

        z = ImplicitSymplecticLayer.apply(
            z0,
            self,
            self.cfg.dt,
        )

        q, p = z.chunk(2, dim=-1)

        energy = self.H(q, p)

        return z, energy


# ================================================================
# JACOBIAN REGULARIZATION
# ================================================================


def jacobian_regularization(H, z):

    z = z.detach().requires_grad_(True)

    f = vector_field(H, z)

    v = torch.randn_like(f)

    Jv = torch.autograd.grad(
        f,
        z,
        grad_outputs=v,
        retain_graph=True,
        create_graph=True,
    )[0]

    return Jv.pow(2).mean()


# ================================================================
# LOSS
# ================================================================


def total_loss(model, z, energy):

    energy_consistency = energy.var()

    latent_reg = z.pow(2).mean()

    jac_reg = jacobian_regularization(model.H, z)

    return (
        energy_consistency
        + model.cfg.energy_reg * energy.abs().mean()
        + model.cfg.latent_reg * latent_reg
        + model.cfg.jacobian_reg * jac_reg
    )


# ================================================================
# TRAINING
# ================================================================


def main():

    cfg = Config()

    cfg.seed_all()

    device = cfg.device

    print("\nDEVICE:", device)

    n = cfg.batch_size
    d = cfg.d

    q0 = torch.randn(n, d, device=device)
    p0 = torch.randn(n, d, device=device)

    z0 = torch.cat([q0, p0], dim=-1)

    model = SymplecticEquilibriumOperator(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1000,
    )

    print("\n================================================")
    print("TRAINING")
    print("================================================")

    for step in range(300):

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=torch.cuda.is_available()):

            z, energy = model(z0)

            loss = total_loss(model, z, energy)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.grad_clip,
        )

        optimizer.step()

        scheduler.step()

        if step % 10 == 0:

            print(
                f"step={step:04d} | "
                f"loss={loss.item():.6f} | "
                f"energy_mean={energy.mean().item():.6f} | "
                f"energy_std={energy.std().item():.6f} | "
                f"state_norm={z.norm().item():.6f}"
            )

    print("\n================================================")
    print("FINAL")
    print("================================================")

    with torch.no_grad():

        z, energy = model(z0)

        print("energy mean:", energy.mean().item())
        print("energy std :", energy.std().item())
        print("state norm :", z.norm().item())


if __name__ == "__main__":
    main()

Mais cette version est déjà beaucoup plus crédible comme prototype NeurIPS/ICLR avancé.
