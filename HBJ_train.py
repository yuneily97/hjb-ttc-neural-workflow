"""
HJB value-network training script.

Objective:
    Train an HJB-style neural value function that estimates the remaining collision time/TTC over the OVM state domain using a strong fixed-point approach. 

Required before running:
    None, except the Python packages listed in requirements.txt.

Main outputs:
    outputs_hjb_k1_nosup/value_net.keras
    outputs_hjb_k1_nosup/value_net_latest.keras
    outputs_hjb_k1_nosup/history.json
    outputs_hjb_k1_nosup/run_config.json
    outputs_hjb_k1_nosup/boundary_table_line.npz

Run order:
    Run this script first. Later scripts load
    outputs_hjb_k1_nosup/value_net.keras.

HJB Value Function Approximation
================================

Key features:
    1. No supervised warm start.
    2. Uses on-the-fly Monte-Carlo collocation sampling.
    3. Uses line-method dynamic-boundary detection to define the compact interior region for the HJB residual.
    4. Applies the boundary loss on the collision boundary Delta xi = 0.
    5. Uses a smooth residual neural network with no ReLU activations.
    6. Saves the value network as a Functional Keras model so it can be loaded directly with keras.models.load_model.
    7. Tracks loss components, validation metrics, and test metrics during training.

This is a standalone script.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TANH2 = np.tanh(2.0)
V_MAX = 1.0 + TANH2
XI_MIN = 0.0
XI_MAX = 1.5
LAM_MIN = -V_MAX
LAM_MAX = 0.0
VL_MIN = 0.0
VL_MAX = V_MAX


def optimal_velocity_np(xi):
    return np.tanh(xi - 2.0) + TANH2


def optimal_velocity_tf(xi):
    return tf.tanh(xi - 2.0) + tf.constant(TANH2, dtype=xi.dtype)


def effective_dt(T_hor, dt_ratio=0.001, dt_abs_max=None):
    dt = float(dt_ratio) * float(T_hor)
    if dt_abs_max is not None:
        dt = min(dt, float(dt_abs_max))
    return max(dt, 1e-6)


def sample_vl_lam_old(n, rng):
    lam = rng.uniform(LAM_MIN, LAM_MAX, size=n).astype(np.float32)
    vl_hi = np.maximum(0.0, V_MAX + lam).astype(np.float32)
    vl = (rng.random(n).astype(np.float32) * vl_hi).astype(np.float32)
    return vl, lam


def sample_vl_lam_biased_old(n, rng, lam_power=3.0):
    u_lam = rng.random(n).astype(np.float32) ** lam_power
    lam = (-V_MAX + u_lam * V_MAX).astype(np.float32)
    vl_hi = np.maximum(0.0, V_MAX + lam).astype(np.float32)
    vl = (rng.random(n).astype(np.float32) * vl_hi).astype(np.float32)
    return vl, lam


def sample_lam_vl_lebesgue(n, rng):
    # Uniform on the feasible triangle: LAM_MIN <= lambda <= 0,
    # 0 <= v_l <= V_MAX + lambda.
    u = rng.random(n).astype(np.float32)
    lam = (V_MAX * (np.sqrt(u) - 1.0)).astype(np.float32)
    vl_hi = np.maximum(0.0, V_MAX + lam).astype(np.float32)
    vl = (rng.random(n).astype(np.float32) * vl_hi).astype(np.float32)
    return lam, vl


def rho_from_lam_vl(lam, vl):
    denom = np.maximum(V_MAX + lam, 1e-8)
    return np.clip(vl / denom, 0.0, 1.0)


def vl_from_lam_rho(lam, rho):
    return np.clip(rho, 0.0, 1.0) * np.maximum(0.0, V_MAX + lam)


def rk4_hit_exists(s, xi0, lam0, vl0, T_hor=1.0, alpha=2.0, dt=0.003):
    s = np.asarray(s, dtype=np.float64)
    xi = np.asarray(xi0, dtype=np.float64).copy()
    lam = np.asarray(lam0, dtype=np.float64).copy()
    vl = np.asarray(vl0, dtype=np.float64)
    scalar = xi.ndim == 0
    s = np.broadcast_to(np.atleast_1d(s), np.atleast_1d(xi).shape).astype(np.float64)
    xi = np.atleast_1d(xi).astype(np.float64).copy()
    lam = np.broadcast_to(np.atleast_1d(lam), xi.shape).astype(np.float64).copy()
    vl = np.broadcast_to(np.atleast_1d(vl), xi.shape).astype(np.float64)
    remaining = np.maximum(T_hor - s, 0.0)
    hit = xi <= 0.0
    active = (remaining > 0.0) & ~hit

    max_steps = int(math.ceil(float(np.max(remaining)) / float(dt))) if len(xi) else 0
    t = 0.0
    for _ in range(max_steps):
        idx = active & (t < remaining - 1e-12)
        if not np.any(idx):
            break
        h = np.minimum(dt, remaining - t)
        h = np.where(idx, h, 0.0)
        xi_prev = xi.copy()

        def rhs(x, l):
            d_x = l
            v_f = vl - l
            d_l = -alpha * (optimal_velocity_np(x) - v_f)
            return d_x, d_l

        k1x, k1l = rhs(xi, lam)
        k2x, k2l = rhs(xi + 0.5 * h * k1x, lam + 0.5 * h * k1l)
        k3x, k3l = rhs(xi + 0.5 * h * k2x, lam + 0.5 * h * k2l)
        k4x, k4l = rhs(xi + h * k3x, lam + h * k3l)
        xi = xi + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        lam = lam + (h / 6.0) * (k1l + 2.0 * k2l + 2.0 * k3l + k4l)

        crossed = idx & (xi <= 0.0) & (xi_prev > 0.0)
        hit[crossed] = True
        active[crossed] = False
        t += dt
    return bool(hit[0]) if scalar else hit


def rk4_ttc_np(s, xi0, lam0, vl0, T_hor=1.0, alpha=2.0, dt=0.001):
    s = np.asarray(s, dtype=np.float64)
    xi = np.asarray(xi0, dtype=np.float64).copy()
    lam = np.asarray(lam0, dtype=np.float64).copy()
    vl = np.asarray(vl0, dtype=np.float64)
    xi = np.atleast_1d(xi).astype(np.float64).copy()
    s = np.broadcast_to(np.atleast_1d(s), xi.shape).astype(np.float64)
    lam = np.broadcast_to(np.atleast_1d(lam), xi.shape).astype(np.float64).copy()
    vl = np.broadcast_to(np.atleast_1d(vl), xi.shape).astype(np.float64)
    remaining = np.maximum(T_hor - s, 0.0)
    ttc = remaining.copy()
    hit = xi <= 0.0
    ttc[hit] = 0.0
    active = (remaining > 0.0) & ~hit
    max_steps = int(math.ceil(float(np.max(remaining)) / float(dt))) if len(xi) else 0
    t = 0.0
    for _ in range(max_steps):
        idx = active & (t < remaining - 1e-12)
        if not np.any(idx):
            break
        h = np.minimum(dt, remaining - t)
        h = np.where(idx, h, 0.0)
        xi_prev = xi.copy()

        def rhs(x, l):
            d_x = l
            v_f = vl - l
            d_l = -alpha * (optimal_velocity_np(x) - v_f)
            return d_x, d_l

        k1x, k1l = rhs(xi, lam)
        k2x, k2l = rhs(xi + 0.5 * h * k1x, lam + 0.5 * h * k1l)
        k3x, k3l = rhs(xi + 0.5 * h * k2x, lam + 0.5 * h * k2l)
        k4x, k4l = rhs(xi + h * k3x, lam + h * k3l)
        xi = xi + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        lam = lam + (h / 6.0) * (k1l + 2.0 * k2l + 2.0 * k3l + k4l)

        crossed = idx & (xi <= 0.0) & (xi_prev > 0.0)
        frac = np.clip(xi_prev[crossed] / np.maximum(xi_prev[crossed] - xi[crossed], 1e-12), 0.0, 1.0)
        ttc[crossed] = t + h[crossed] * frac
        active[crossed] = False
        t += dt
    return ttc.astype(np.float32)


def build_eval_set(n, args, seed):
    rng = np.random.default_rng(seed + 123)
    n1 = int(0.40 * n)
    n2 = int(0.20 * n)
    n3 = int(0.20 * n)
    n4 = n - n1 - n2 - n3

    def make_block(m, xi_power=1.0, lam_bias=False, s_mode="mixed"):
        xi = max(XI_MIN, 0.01) + (rng.random(m) ** xi_power) * (XI_MAX - max(XI_MIN, 0.01))
        if lam_bias:
            vl, lam = sample_vl_lam_biased_old(m, rng)
        else:
            vl, lam = sample_vl_lam_old(m, rng)
        if s_mode == "early":
            s = args.T_hor * (rng.random(m) ** 3.0)
        elif s_mode == "late":
            s = args.T_hor * (1.0 - rng.random(m) ** 2.0)
        else:
            u = rng.random(m)
            s = np.where(
                u < 0.40,
                rng.random(m) * args.T_hor,
                np.where(u < 0.75,
                         args.T_hor * (rng.random(m) ** 3.0),
                         args.T_hor * (1.0 - rng.random(m) ** 2.0)),
            )
        s = np.clip(s, 0.0, args.T_hor - args.s_eps)
        return xi.astype(np.float32), lam.astype(np.float32), s.astype(np.float32), vl.astype(np.float32)

    xi1, lam1, s1, vl1 = make_block(n1)
    xi2, lam2, s2, vl2 = make_block(n2, xi_power=3.0, s_mode="early")
    xi3, lam3, s3, vl3 = make_block(n3, lam_bias=True, s_mode="mixed")
    xi4, lam4, s4, vl4 = make_block(n4, xi_power=3.0, lam_bias=True, s_mode="early")
    xi = np.concatenate([xi1, xi2, xi3, xi4])
    lam = np.concatenate([lam1, lam2, lam3, lam4])
    s = np.concatenate([s1, s2, s3, s4])
    vl = np.concatenate([vl1, vl2, vl3, vl4])
    perm = rng.permutation(n)
    x = np.column_stack([s, xi, lam, vl]).astype(np.float32)
    x = x[perm]
    y = rk4_ttc_np(x[:, 0], x[:, 1], x[:, 2], x[:, 3], args.T_hor, args.alpha, args.eval_dt)
    return x, y


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    err = y_pred - y_true
    mse = float(np.mean(err * err))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": float(r2)}


def evaluate_model(model, x_eval, y_eval, batch_size):
    pred = model.predict(x_eval, batch_size=batch_size, verbose=0).ravel()
    return regression_metrics(y_eval, pred)


def line_boundary_for_state(s, lam, vl, args):
    if not rk4_hit_exists(s, XI_MIN, lam, vl, args.T_hor, args.alpha, args.boundary_dt):
        return 0.0
    if rk4_hit_exists(s, XI_MAX, lam, vl, args.T_hor, args.alpha, args.boundary_dt):
        return XI_MAX
    lo, hi = XI_MIN, XI_MAX
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    for _ in range(args.line_iters):
        x = hi - (hi - lo) / phi
        if rk4_hit_exists(s, x, lam, vl, args.T_hor, args.alpha, args.boundary_dt):
            lo = x
        else:
            hi = x
    return 0.5 * (lo + hi)


class BoundaryTable:
    def __init__(self, s_grid, lam_grid, rho_grid, xi_boundary):
        self.s_grid = np.asarray(s_grid, dtype=np.float32)
        self.lam_grid = np.asarray(lam_grid, dtype=np.float32)
        self.rho_grid = np.asarray(rho_grid, dtype=np.float32)
        self.xi_boundary = np.asarray(xi_boundary, dtype=np.float32)

    def lookup(self, s, lam, vl):
        s = np.asarray(s, dtype=np.float32)
        lam = np.asarray(lam, dtype=np.float32)
        rho = rho_from_lam_vl(lam, np.asarray(vl, dtype=np.float32)).astype(np.float32)

        si = np.clip(np.searchsorted(self.s_grid, s, side="left"), 0, len(self.s_grid) - 1)
        li = np.clip(np.searchsorted(self.lam_grid, lam, side="left"), 0, len(self.lam_grid) - 1)
        ri = np.clip(np.searchsorted(self.rho_grid, rho, side="left"), 0, len(self.rho_grid) - 1)

        def nearest(ids, grid, vals):
            ids0 = np.maximum(ids - 1, 0)
            choose0 = np.abs(vals - grid[ids0]) <= np.abs(vals - grid[ids])
            return np.where(choose0, ids0, ids)

        si = nearest(si, self.s_grid, s)
        li = nearest(li, self.lam_grid, lam)
        ri = nearest(ri, self.rho_grid, rho)
        return self.xi_boundary[si, li, ri]

    def save(self, path):
        np.savez_compressed(
            path,
            s_grid=self.s_grid,
            lam_grid=self.lam_grid,
            rho_grid=self.rho_grid,
            xi_boundary=self.xi_boundary,
        )

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["s_grid"], z["lam_grid"], z["rho_grid"], z["xi_boundary"])


def build_line_table(args):
    s_grid = np.linspace(0.0, args.T_hor - args.s_eps, args.boundary_s_n, dtype=np.float32)
    lam_grid = np.linspace(LAM_MIN, LAM_MAX, args.boundary_lam_n, dtype=np.float32)
    rho_grid = np.linspace(0.0, 1.0, args.boundary_rho_n, dtype=np.float32)
    out = np.zeros((len(s_grid), len(lam_grid), len(rho_grid)), dtype=np.float32)
    total = out.size
    k = 0
    t0 = time.time()
    for i, s in enumerate(s_grid):
        for j, lam in enumerate(lam_grid):
            for r, rho in enumerate(rho_grid):
                vl = float(vl_from_lam_rho(lam, rho))
                out[i, j, r] = line_boundary_for_state(float(s), float(lam), vl, args)
                k += 1
        print(f"[boundary-line] s index {i+1}/{len(s_grid)} complete ({k}/{total}) elapsed={time.time()-t0:.1f}s", flush=True)
    return BoundaryTable(s_grid, lam_grid, rho_grid, out)


def build_gp_table(args):
    rng = np.random.default_rng(args.seed + 17)
    # Labels in normalized coordinates z=(s, lam, rho, xi).
    def sample_points(n, near=False):
        s = rng.uniform(0.0, args.T_hor - args.s_eps, size=n).astype(np.float32)
        lam, vl = sample_lam_vl_lebesgue(n, rng)
        rho = rho_from_lam_vl(lam, vl).astype(np.float32)
        if near:
            xi = rng.beta(0.7, 2.0, size=n).astype(np.float32) * XI_MAX
        else:
            xi = rng.uniform(XI_MIN, XI_MAX, size=n).astype(np.float32)
        X = np.column_stack([s, lam, rho, xi]).astype(np.float32)
        y = rk4_hit_exists(s, xi, lam, vl, args.T_hor, args.alpha, args.boundary_dt).astype(np.int32)
        return X, y

    X0, y0 = sample_points(args.gp_init_n, near=False)
    X1, y1 = sample_points(args.gp_init_n // 2, near=True)
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
        length_scale=[0.25, 0.35, 0.35, 0.20],
        length_scale_bounds=(0.03, 3.0),
    ) + WhiteKernel(noise_level=1e-4, noise_level_bounds="fixed")

    for rnd in range(args.gp_rounds):
        model = make_pipeline(
            StandardScaler(),
            GaussianProcessClassifier(kernel=kernel, optimizer=None, max_iter_predict=80, random_state=args.seed),
        ).fit(X, y)
        pool, _ = sample_points(args.gp_pool_n, near=False)
        prob = model.predict_proba(pool)[:, 1]
        idx = np.argsort(np.abs(prob - 0.5))[:args.gp_batch_n]
        Xnew = pool[idx]
        lam = Xnew[:, 1]
        vl = vl_from_lam_rho(lam, Xnew[:, 2])
        ynew = rk4_hit_exists(Xnew[:, 0], Xnew[:, 3], lam, vl, args.T_hor, args.alpha, args.boundary_dt).astype(np.int32)
        X = np.vstack([X, Xnew])
        y = np.concatenate([y, ynew])
        print(f"[boundary-gp] round {rnd+1}/{args.gp_rounds} train={len(y)} pos={int(y.sum())}", flush=True)

    model = make_pipeline(
        StandardScaler(),
        GaussianProcessClassifier(kernel=kernel, optimizer=None, max_iter_predict=80, random_state=args.seed),
    ).fit(X, y)

    s_grid = np.linspace(0.0, args.T_hor - args.s_eps, args.boundary_s_n, dtype=np.float32)
    lam_grid = np.linspace(LAM_MIN, LAM_MAX, args.boundary_lam_n, dtype=np.float32)
    rho_grid = np.linspace(0.0, 1.0, args.boundary_rho_n, dtype=np.float32)
    xi_scan = np.linspace(XI_MIN, XI_MAX, args.gp_xi_scan_n, dtype=np.float32)
    out = np.zeros((len(s_grid), len(lam_grid), len(rho_grid)), dtype=np.float32)
    for i, s in enumerate(s_grid):
        for j, lam in enumerate(lam_grid):
            for r, rho in enumerate(rho_grid):
                pts = np.column_stack([
                    np.full_like(xi_scan, s),
                    np.full_like(xi_scan, lam),
                    np.full_like(xi_scan, rho),
                    xi_scan,
                ])
                prob = model.predict_proba(pts)[:, 1]
                hit = prob >= 0.5
                out[i, j, r] = xi_scan[np.where(hit)[0].max()] if np.any(hit) else 0.0
        print(f"[boundary-gp] table s index {i+1}/{len(s_grid)} complete", flush=True)
    return BoundaryTable(s_grid, lam_grid, rho_grid, out), {"gp_train_n": int(len(y)), "gp_pos": int(y.sum())}


class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, peak_lr=3e-4, warmup_steps=300, floor_lr=1e-6):
        super().__init__()
        self.total_steps = float(max(1, total_steps))
        self.peak_lr = float(peak_lr)
        self.warmup_steps = float(max(1, warmup_steps))
        self.floor_lr = float(floor_lr)

    def __call__(self, step):
        step_f = tf.cast(step, tf.float32)
        warm = self.peak_lr * (step_f / self.warmup_steps)
        denom = tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        progress = tf.clip_by_value((step_f - self.warmup_steps) / denom, 0.0, 1.0)
        cosine = self.floor_lr + 0.5 * (self.peak_lr - self.floor_lr) * (
            1.0 + tf.cos(tf.constant(math.pi, dtype=tf.float32) * progress)
        )
        return tf.where(step_f < self.warmup_steps, warm, cosine)

    def get_config(self):
        return {
            "total_steps": self.total_steps,
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "floor_lr": self.floor_lr,
        }


def build_value_network(
    T_hor=1.0,
    hidden_units=(128, 256, 512, 256, 128, 64),
    activations=("tanh", "swish", "tanh", "tanh", "swish", "tanh"),
    input_gate_units=256,
    smooth_gate_activation="tanh",
    output_bias_init=5.0,
    output_kernel_std=1e-3,
):
    inp = keras.Input(shape=(4,), name="s_xi_lam_vl")
    s = inp[:, 0:1]
    s_n = 2.0 * s / (float(T_hor) + 1e-8) - 1.0
    xi_n = 2.0 * inp[:, 1:2] / (XI_MAX + 1e-8) - 1.0
    lam_n = 2.0 * (inp[:, 2:3] - LAM_MIN) / (LAM_MAX - LAM_MIN + 1e-8) - 1.0
    vl_n = 2.0 * inp[:, 3:4] / (V_MAX + 1e-8) - 1.0
    x = keras.ops.concatenate([s_n, xi_n, lam_n, vl_n], axis=1)
    x = layers.Dense(input_gate_units, activation=None, name="input_gate")(x)
    x = layers.Activation(smooth_gate_activation, name="input_gate_act")(x)
    prev = None
    for i, (units, act) in enumerate(zip(hidden_units, activations)):
        h = layers.Dense(units, activation=act, name=f"hidden_{i+1}")(x)
        if prev is not None:
            prev_units = prev.shape[-1]
            if prev_units == units:
                h = layers.Add(name=f"skip_{i+1}")([prev, h])
            else:
                skip = layers.Dense(units, activation=None, use_bias=False, name=f"proj_{i+1}")(prev)
                h = layers.Add(name=f"skip_{i+1}")([skip, h])
            h = layers.Activation(smooth_gate_activation, name=f"post_residual_tanh_{i+1}")(h)
        prev = h
        x = h
    u_logits = layers.Dense(
        1,
        activation=None,
        name="u_logits",
        kernel_initializer=keras.initializers.RandomNormal(stddev=output_kernel_std),
        bias_initializer=keras.initializers.Constant(output_bias_init),
    )(x)
    u_raw = keras.ops.sigmoid(u_logits)
    remaining = keras.ops.maximum(float(T_hor) - s, 0.0)
    out = u_raw * remaining
    return keras.Model(inputs=inp, outputs=out, name="HJB_ValueNet_Residual")


def F_tf(xi, lam, vl, alpha):
    g_xi = lam
    v_f = vl - lam
    g_lam = -alpha * (optimal_velocity_tf(xi) - v_f)
    return g_xi, g_lam


class GeometryTrainer:
    def __init__(self, value_net, optimizer, args):
        self.value_net = value_net
        self.optimizer = optimizer
        self.args = args

    @tf.function
    def _tdt_inf_target(self, x_int, v_now):
        s = x_int[:, 0]
        xi = x_int[:, 1]
        lam = x_int[:, 2]
        vl = x_int[:, 3]
        remaining = tf.maximum(self.args.T_hor - s, 0.0)
        stop_mask = tf.logical_or(xi <= 0.0, remaining <= 0.0)
        kappa_max = tf.minimum(self.args.dt, remaining)
        candidates = []
        for frac in self.args.tdt_fractions:
            if frac == 0.0:
                candidate = v_now
            else:
                kappa = kappa_max * tf.constant(float(frac), dtype=tf.float32)
                gx, gl = F_tf(xi, lam, vl, self.args.alpha)
                xi_k = xi + kappa * gx
                lam_k = lam + kappa * gl
                s_k = tf.minimum(s + kappa, self.args.T_hor)
                crossed = (xi > 0.0) & (xi_k <= 0.0)
                denom = tf.maximum(xi - xi_k, 1e-12)
                hit_frac = tf.clip_by_value(xi / denom, 0.0, 1.0)
                hit_time = hit_frac * kappa
                x_k = tf.stack([s_k, xi_k, lam_k, vl], axis=1)
                v_next = tf.squeeze(self.value_net(x_k, training=True), axis=1)
                candidate = tf.where(crossed, hit_time, kappa + v_next)
            candidate = tf.where(stop_mask, tf.zeros_like(candidate), tf.minimum(remaining, candidate))
            candidates.append(candidate)
        return tf.reduce_min(tf.stack(candidates, axis=0), axis=0)

    @tf.function
    def train_step(self, x_int, x_coll, y_coll):
        with tf.GradientTape() as tape:
            with tf.GradientTape() as input_tape:
                input_tape.watch(x_int)
                v_int = tf.squeeze(self.value_net(x_int, training=True), axis=1)
            grads_in = input_tape.gradient(v_int, x_int)
            s = x_int[:, 0]
            xi = x_int[:, 1]
            lam = x_int[:, 2]
            vl = x_int[:, 3]
            gx, gl = F_tf(xi, lam, vl, self.args.alpha)
            residual = grads_in[:, 0] + gx * grads_in[:, 1] + gl * grads_in[:, 2] + 1.0
            l_hjb = tf.reduce_mean(tf.square(residual))
            if self.args.dt_loss_weight == 0.0:
                l_dt = tf.zeros((), dtype=l_hjb.dtype)
            else:
                target = self._tdt_inf_target(x_int, v_int)
                l_dt = tf.reduce_mean(tf.square(v_int - target))
            l_interior = self.args.dt_loss_weight * l_dt + self.args.hjb_loss_weight * l_hjb
            v_coll = tf.squeeze(self.value_net(x_coll, training=True), axis=1)
            l_collision = tf.reduce_mean(tf.square(v_coll - y_coll))
            loss = l_interior + self.args.boundary_weight * l_collision
        grads = tape.gradient(loss, self.value_net.trainable_variables)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) if g is not None else tf.zeros_like(v)
                 for g, v in zip(grads, self.value_net.trainable_variables)]
        self.optimizer.apply_gradients(zip(grads, self.value_net.trainable_variables))
        return loss, l_interior, l_dt, l_hjb, l_collision


def sample_interior(n, table, args, rng):
    got = []
    need = n
    attempts = 0
    while need > 0:
        m = max(need * 3, 4096)
        s = rng.uniform(0.0, args.T_hor - args.s_eps, size=m).astype(np.float32)
        lam, vl = sample_lam_vl_lebesgue(m, rng)
        xi = rng.uniform(0.0, XI_MAX, size=m).astype(np.float32)
        b = table.lookup(s, lam, vl)
        mask = (xi > args.margin) & (xi < (b - args.margin))
        if np.any(mask):
            arr = np.column_stack([s[mask], xi[mask], lam[mask], vl[mask]]).astype(np.float32)
            got.append(arr[:need])
            need -= min(need, len(arr))
        attempts += 1
        if attempts > 200:
            raise RuntimeError("Could not sample enough interior compact-set points")
    return np.vstack(got)[:n].astype(np.float32)


def sample_boundary(n, table, args, rng):
    s0 = rng.uniform(0.0, args.T_hor - args.s_eps, size=n).astype(np.float32)
    lam0, vl0 = sample_lam_vl_lebesgue(n, rng)
    x0 = np.column_stack([s0, np.zeros(n, dtype=np.float32), lam0, vl0]).astype(np.float32)
    y0 = np.zeros(n, dtype=np.float32)
    return x0, y0


def save_checkpoint(out_dir, model, history, epoch, stats):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "value_net.keras")
    model.save(out_dir / "value_net_latest.keras")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    meta = {"epoch": int(epoch), **{k: float(v) for k, v in stats.items()}}
    with open(out_dir / "value_net_latest_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def run_training(args, table):
    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)
    args.dt = tf.constant(effective_dt(args.T_hor, args.dt_ratio, args.dt_abs_max), dtype=tf.float32)
    if args.tdt_include_zero:
        args.tdt_fractions = tuple(np.linspace(0.0, 1.0, args.tdt_kappa_count, dtype=np.float32).tolist())
    else:
        args.tdt_fractions = tuple(np.linspace(1.0 / args.tdt_kappa_count, 1.0, args.tdt_kappa_count, dtype=np.float32).tolist())
    model = build_value_network(T_hor=args.T_hor)
    opt = keras.optimizers.Adam(learning_rate=float(args.lr), clipnorm=1.0)
    trainer = GeometryTrainer(model, opt, args)
    x_val, y_val = build_eval_set(args.val_n, args, args.seed + 1000)
    x_test, y_test = build_eval_set(args.test_n, args, args.seed + 2000)
    history = {
        "epoch": [], "loss": [], "l_interior": [], "l_dt": [], "l_hjb": [],
        "l_collision_boundary": [], "lr": [],
        "val_mae": [], "val_mse": [], "val_rmse": [], "val_r2": [],
        "test_mae": [], "test_mse": [], "test_rmse": [], "test_r2": [],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_r2 = -np.inf
    best_step = 0
    stale_evals = 0
    for step in range(1, args.steps + 1):
        if step <= args.lr_warmup_steps:
            warm_lr = float(args.lr) * (float(step) / float(max(1, args.lr_warmup_steps)))
            opt.learning_rate.assign(warm_lr)
        xi = sample_interior(args.n_interior, table, args, rng)
        x_coll, y_coll = sample_boundary(args.n_boundary, table, args, rng)
        loss, l_int, l_dt, l_hjb, l_coll = trainer.train_step(
            tf.convert_to_tensor(xi, dtype=tf.float32),
            tf.convert_to_tensor(x_coll, dtype=tf.float32),
            tf.convert_to_tensor(y_coll, dtype=tf.float32),
        )
        if step == 1 or step % args.print_every == 0:
            current_lr = float(opt.learning_rate.numpy())
            stats = {
                "loss": float(loss.numpy()),
                "l_interior": float(l_int.numpy()),
                "l_dt": float(l_dt.numpy()),
                "l_hjb": float(l_hjb.numpy()),
                "l_collision_boundary": float(l_coll.numpy()),
                "lr": current_lr,
            }
            val_metrics = evaluate_model(model, x_val, y_val, args.eval_batch_size)
            test_metrics = evaluate_model(model, x_test, y_test, args.eval_batch_size)
            stats.update({f"val_{k}": v for k, v in val_metrics.items()})
            stats.update({f"test_{k}": v for k, v in test_metrics.items()})
            if stats["val_r2"] > best_val_r2 + args.lr_min_delta:
                best_val_r2 = stats["val_r2"]
                best_step = step
                stale_evals = 0
                save_checkpoint(out_dir / "best_val", model, history, step, stats)
            else:
                stale_evals += 1
                if stale_evals >= args.lr_patience:
                    new_lr = max(float(opt.learning_rate.numpy()) * args.lr_reduce_factor, args.lr_floor)
                    if new_lr < float(opt.learning_rate.numpy()) - 1e-15:
                        opt.learning_rate.assign(new_lr)
                        print(
                            f"  [adaptive lr] step {step}: val_r2 stale for {stale_evals} evals; "
                            f"lr -> {new_lr:.2e} (best val_r2={best_val_r2:.4f} at step {best_step})",
                            flush=True,
                        )
                    stale_evals = 0
            history["epoch"].append(step)
            for k in ["loss", "l_interior", "l_dt", "l_hjb", "l_collision_boundary", "lr"]:
                history[k].append(stats[k])
            for k in ["mae", "mse", "rmse", "r2"]:
                history[f"val_{k}"].append(stats[f"val_{k}"])
                history[f"test_{k}"].append(stats[f"test_{k}"])
            print(
                f"  epoch {step:5d}/{args.steps}  "
                f"loss={stats['loss']:.4e}  "
                f"l_int={stats['l_interior']:.4e}  "
                f"l_dt={stats['l_dt']:.4e}  "
                f"l_hjb={stats['l_hjb']:.4e}  "
                f"l_coll={stats['l_collision_boundary']:.4e}  "
                f"lr={stats['lr']:.2e}  "
                f"val_mae={stats['val_mae']:.4f}  "
                f"val_rmse={stats['val_rmse']:.4f}  "
                f"val_r2={stats['val_r2']:.4f}  "
                f"test_rmse={stats['test_rmse']:.4f}  "
                f"test_r2={stats['test_r2']:.4f}",
                flush=True,
            )
        if step % args.checkpoint_every == 0:
            save_checkpoint(out_dir, model, history, step, stats)
            print(f"[saved] checkpoint epoch {step} -> {out_dir}", flush=True)
    save_checkpoint(out_dir, model, history, args.steps, stats)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("line", "gp"), default="line")
    p.add_argument("--out-dir", default="outputs_hjb_k1_nosup")
    p.add_argument("--steps", type=int, default=500000)
    p.add_argument("--n-interior", type=int, default=65536)
    p.add_argument("--n-boundary", type=int, default=8192)
    p.add_argument("--boundary-weight", type=float, default=1.0)
    p.add_argument("--T-hor", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--margin", type=float, default=0.00001)
    p.add_argument("--s-eps", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-warmup-steps", type=int, default=300)
    p.add_argument("--lr-floor", type=float, default=1e-6)
    p.add_argument("--hjb-loss-weight", type=float, default=10.0)
    p.add_argument("--dt-loss-weight", type=float, default=0.0)
    p.add_argument("--dt-ratio", type=float, default=0.001)
    p.add_argument("--dt-abs-max", type=float, default=None)
    p.add_argument("--tdt-kappa-count", type=int, default=5)
    p.add_argument("--tdt-exclude-zero", action="store_true")
    p.add_argument("--seed", type=int, default=70)
    p.add_argument("--print-every", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--lr-patience", type=int, default=8)
    p.add_argument("--lr-reduce-factor", type=float, default=0.5)
    p.add_argument("--lr-min-delta", type=float, default=1e-4)
    p.add_argument("--val-n", type=int, default=4096)
    p.add_argument("--test-n", type=int, default=4096)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--eval-dt", type=float, default=0.001)
    p.add_argument("--boundary-dt", type=float, default=0.003)
    p.add_argument("--boundary-s-n", type=int, default=21)
    p.add_argument("--boundary-lam-n", type=int, default=61)
    p.add_argument("--boundary-rho-n", type=int, default=7)
    p.add_argument("--line-iters", type=int, default=16)
    p.add_argument("--gp-init-n", type=int, default=260)
    p.add_argument("--gp-rounds", type=int, default=5)
    p.add_argument("--gp-pool-n", type=int, default=600)
    p.add_argument("--gp-batch-n", type=int, default=40)
    p.add_argument("--gp-xi-scan-n", type=int, default=101)
    args = p.parse_args()
    args.tdt_include_zero = not args.tdt_exclude_zero
    return args


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    table_path = out_dir / f"boundary_table_{args.method}.npz"
    if table_path.exists():
        table = BoundaryTable.load(table_path)
    else:
        if args.method == "line":
            table = build_line_table(args)
            extra = {}
        else:
            table, extra = build_gp_table(args)
            with open(out_dir / "gp_info.json", "w") as f:
                json.dump(extra, f, indent=2)
        table.save(table_path)
    run_training(args, table)


if __name__ == "__main__":
    main()
