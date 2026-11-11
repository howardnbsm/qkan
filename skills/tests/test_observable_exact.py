# Copyright (c) 2026, Jiun-Cheng Jiang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainable-observable acceptance tests: CPU, ``exact`` solver.

Section E of the trainable-observable specification: E1-E5, the CPU parts of
E6, and E7(b)-(c). E7(a) is a suite-level criterion (every other test in
``skills/tests`` keeps passing) and has no test of its own; the CUDA criteria
live in ``test_observable_cute.py``.

Fixture (spec E): ``QKANLayer(3, 4, reps=2, group=g, solver="exact",
ansatz="pz_encoding", ba_trainable=False, c_dtype=torch.complex128,
p_dtype=torch.float64)`` on CPU, ``B = 7``, ``x`` uniform on [-pi, pi], and
``Z``: zeros ``(12, 4)`` with rows 3, 7, 11 set to 1, i.e. O_ji = Z on every
edge. ``g = 3`` gives one circuit per input node (G = 1); ``g = -1`` one per
edge (G = O = 4).

The new API (``forward(x, observable=...)``, ``forward_bloch``, the
``bloch_measure`` solver keyword, ``StateVector.measure_bloch``) is referenced
only inside test bodies, so the module collects on code that predates it.
Tests named ``test_extra_*`` go beyond the criteria of spec E (sections B-D
only) and say so in their docstrings.
"""

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import qkan  # noqa: E402
from qkan.qkan import QKANLayer  # noqa: E402
from qkan.solver import (  # noqa: E402
    QKANSolver,
    get_registry,
    get_solver,
    register,
    torch_exact_solver,
)

B = 7
IN_DIM = 3
OUT_DIM = 4
REPS = 2
GROUPS = (3, -1)
# G of B.2 (theta.shape[0] after the solver's broadcast) for each fixture group.
N_CIRCUITS = {3: 1, -1: OUT_DIM}
SEED = 0
TOL = 1e-12

# The two observable layouts of B.3.
OBS_SHAPES = {
    "shared": (4 * IN_DIM, OUT_DIM),
    "per_sample": (B, 4 * IN_DIM, OUT_DIM),
}


def _gid(g):
    return f"g{g}"


@pytest.fixture(autouse=True)
def _seeded_rng():
    # Every test draws its layer init, inputs and coefficients from a fixed
    # seed; the global RNG state is restored afterwards.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        yield


def make_layer(g, **overrides):
    """The spec-E exact fixture; ``overrides`` replace or add keywords."""
    kwargs = dict(
        reps=REPS,
        group=g,
        solver="exact",
        ansatz="pz_encoding",
        ba_trainable=False,
        c_dtype=torch.complex128,
        p_dtype=torch.float64,
    )
    kwargs.update(overrides)
    return QKANLayer(IN_DIM, OUT_DIM, **kwargs)


def uniform_x(batch=B):
    """Inputs of shape ``(batch, I)``, uniform on [-pi, pi], float64."""
    return torch.empty(batch, IN_DIM, dtype=torch.float64).uniform_(-math.pi, math.pi)


def z_observable():
    """Coefficient p = 3 (Z) of every edge set to 1: rows 3, 7, 11 of (12, 4)."""
    obs = torch.zeros(4 * IN_DIM, OUT_DIM, dtype=torch.float64)
    obs[[3, 7, 11]] = 1.0
    return obs


def randn_observable(kind, dtype=torch.float64):
    return torch.randn(OBS_SHAPES[kind], dtype=dtype)


def key_matrix(layer, x):
    """``K = cat(ones, forward_bloch(x)[:, 0]).reshape(B, 4 * I)`` (E2, E4)."""
    r = layer.forward_bloch(x)[:, 0]
    k = torch.cat([torch.ones_like(r[..., :1]), r], dim=-1)
    return k.reshape(x.shape[0], 4 * IN_DIM)


def max_abs(a, b):
    return (a - b).abs().max().item()


# ---------------------------------------------------------------------------
# E1 Bloch values (exact)
# ---------------------------------------------------------------------------


def _bloch_grid():
    """E1 inputs: 41 points evenly spaced on [-pi, pi], same value in all 3 columns."""
    t = torch.linspace(-math.pi, math.pi, 41, dtype=torch.float64)
    return t, t[:, None].repeat(1, IN_DIM)


def _e1_input(x_kind):
    """E1's 41-point grid ("grid") or the fixture's uniform x, B = 7 ("uniform").

    The random-theta checks of E1 run on both: E1 states the grid as its
    inputs, and the uniform x has distinct columns, which exposes column
    mix-ups that the grid's identical columns cannot.
    """
    return _bloch_grid()[1] if x_kind == "grid" else uniform_x()


def _set_theta(layer, case):
    with torch.no_grad():
        if case == "random":
            layer.theta.copy_(torch.randn_like(layer.theta))
        else:
            layer.theta.zero_()
            if case == "ry_half_pi":
                # layer-1 R_Y angle of every circuit
                layer.theta[..., 1, 1] = math.pi / 2


@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_e1_bloch_theta_zero(g):
    """E1: with theta = 0, forward_bloch(x)[:, 0] = (cos 2x, sin 2x, 0)."""
    layer = make_layer(g)
    _set_theta(layer, "zero")
    t, x = _bloch_grid()
    r = layer.forward_bloch(x)
    assert r.shape == (41, N_CIRCUITS[g], IN_DIM, 3)
    assert r.dtype == torch.float64  # C.2: returned in p_dtype
    expected = torch.stack(
        [torch.cos(2 * t), torch.sin(2 * t), torch.zeros_like(t)], dim=-1
    )
    assert max_abs(r[:, 0], expected[:, None, :]) <= TOL


@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_e1_bloch_layer1_ry_half_pi(g):
    """E1 (algorithm.md T1): theta = 0 except theta[..., 1, 1] = pi/2 gives
    forward_bloch(x)[:, 0] = (-1/2 + cos(2x)/2, sin(2x)/2, -cos x), for both
    theta shapes (3, 3, 2) (g = 3) and (4, 3, 3, 2) (g = -1)."""
    layer = make_layer(g)
    assert tuple(layer.theta.shape) == {3: (3, 3, 2), -1: (4, 3, 3, 2)}[g]
    _set_theta(layer, "ry_half_pi")
    t, x = _bloch_grid()
    r = layer.forward_bloch(x)
    assert r.shape == (41, N_CIRCUITS[g], IN_DIM, 3)
    assert r.dtype == torch.float64
    expected = torch.stack(
        [-0.5 + 0.5 * torch.cos(2 * t), 0.5 * torch.sin(2 * t), -torch.cos(t)],
        dim=-1,
    )
    assert max_abs(r[:, 0], expected[:, None, :]) <= TOL


@pytest.mark.parametrize(
    "case,x_kind",
    [
        ("zero", "grid"),
        ("ry_half_pi", "grid"),
        ("random", "grid"),
        ("random", "uniform"),
    ],
    ids=["zero-grid", "ry_half_pi-grid", "random-grid", "random-uniform"],
)
@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_e1_bloch_vectors_have_unit_norm(g, case, x_kind):
    """E1: ||r|| = 1 within 1e-12 (pure states), for every circuit; the
    closed-form thetas on E1's 41-point grid, random theta on that grid and
    on the fixture's uniform x."""
    layer = make_layer(g)
    _set_theta(layer, case)
    x = _e1_input(x_kind)
    r = layer.forward_bloch(x)
    assert r.shape == (x.shape[0], N_CIRCUITS[g], IN_DIM, 3)
    norm = torch.linalg.vector_norm(r, dim=-1)
    assert (norm - 1).abs().max().item() <= TOL


@pytest.mark.parametrize("x_kind", ["grid", "uniform"])
@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_e1_bloch_z_equals_exact_z_measurement(g, x_kind):
    """E1: for random theta, forward_bloch(x)[..., 2] equals
    torch_exact_solver(..., fast_measure=False) within 1e-12 (B.2: r_z is the
    existing exact Z measurement), on E1's 41-point grid (shapes (41, 1, 3, 3)
    / (41, 4, 3, 3)) and on the fixture's uniform x."""
    layer = make_layer(g)
    _set_theta(layer, "random")
    x = _e1_input(x_kind)
    r = layer.forward_bloch(x)
    assert r.shape == (x.shape[0], N_CIRCUITS[g], IN_DIM, 3)
    with torch.no_grad():
        z = torch_exact_solver(
            x,
            layer._theta_natural(),
            layer._preacts_w_natural(),
            layer._preacts_b_natural(),
            REPS,
            ansatz="pz_encoding",
            fast_measure=False,
            out_dim=OUT_DIM,
            dtype=torch.complex128,
        )
    assert z.shape == r.shape[:-1]
    assert max_abs(r[..., 2], z) <= TOL


# ---------------------------------------------------------------------------
# E2 Readout (exact)
# ---------------------------------------------------------------------------


def _explicit_readout(layer, x, observable):
    """E2 reference: B.3 with explicit loops over b, j, i, p, then the epilogue.

    phi[b, j, i] = sum_p c^(b)[4 i + p, j] * k[b, g(j), i, p] with
    k = (1, r_x, r_y, r_z) from forward_bloch(x) and g(j) = 0 if G == 1 else j;
    y[b, j] = sum_i (phi[b, j, i] + beta[j, i]) * w[j, i] * m[j, i]
              + sum_i u[j, i] * m[j, i] * sigma(x[b, i]),
    with beta applied under the existing condition. The fixture has no batch
    norm, so sigma acts on x itself.
    """
    assert not layer.is_batchnorm
    with torch.no_grad():
        r = layer.forward_bloch(x)
        sigma = layer.base_activation(x)
    batch, n_circuits, n_in, _ = r.shape
    n_out = layer.out_dim
    assert n_circuits in (1, n_out)
    c = observable.detach().to(torch.float64)
    w = layer.postact_weights.detach()
    beta = layer.postact_bias.detach()
    m = layer.mask.detach()
    u = layer.base_weight.detach()
    use_beta = layer.postact_bias_trainable or layer._pb_is_nonzero
    y = torch.zeros(batch, n_out, dtype=torch.float64)
    for b in range(batch):
        cb = c if c.dim() == 2 else c[b]
        for j in range(n_out):
            gj = 0 if n_circuits == 1 else j
            acc = 0.0
            for i in range(n_in):
                k = (
                    1.0,
                    r[b, gj, i, 0].item(),
                    r[b, gj, i, 1].item(),
                    r[b, gj, i, 2].item(),
                )
                phi = 0.0
                for p in range(4):
                    phi += cb[4 * i + p, j].item() * k[p]
                if use_beta:
                    phi += beta[j, i].item()
                acc += phi * w[j, i].item() * m[j, i].item()
                acc += u[j, i].item() * m[j, i].item() * sigma[b, i].item()
            y[b, j] = acc
    return y


@pytest.mark.parametrize("kind", list(OBS_SHAPES))
@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_e2_readout_matches_explicit_loops(g, kind):
    """E2: layer(x, observable) equals B.3 evaluated with explicit loops from
    forward_bloch(x), within 1e-12, for observable randn (12, 4) or (7, 12, 4)."""
    layer = make_layer(g)
    x = uniform_x()
    obs = randn_observable(kind)
    y = layer(x, observable=obs)
    assert y.shape == (B, OUT_DIM)  # C.1
    assert max_abs(y, _explicit_readout(layer, x, obs)) <= TOL


@pytest.mark.parametrize("is_batchnorm", [False, True], ids=["no_bn", "batchnorm"])
@pytest.mark.parametrize("kind", list(OBS_SHAPES))
def test_e2_readout_equals_einsum_g3(kind, is_batchnorm):
    """E2: for g = 3, layer(x, C) equals einsum("kn,bk->bn", C, K) (shared C)
    or einsum("bkn,bk->bn", C, K) (per-sample C) with
    K = cat(ones, forward_bloch(x)[:, 0]).reshape(7, 12), within 1e-12; the
    same identities hold with is_batchnorm=True."""
    layer = make_layer(3, is_batchnorm=is_batchnorm)
    x = uniform_x()
    C = randn_observable(kind)
    y = layer(x, observable=C)
    K = key_matrix(layer, x)
    equation = "kn,bk->bn" if kind == "shared" else "bkn,bk->bn"
    assert max_abs(y, torch.einsum(equation, C, K)) <= TOL


@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_e2_shared_equals_expanded_per_sample(g):
    """E2 (B.4, any strides): a shared C and C.unsqueeze(0).expand(7, -1, -1)
    give equal outputs (within the E2 tolerance 1e-12)."""
    layer = make_layer(g)
    x = uniform_x()
    C = randn_observable("shared")
    expanded = C.unsqueeze(0).expand(B, -1, -1)
    assert expanded.stride(0) == 0
    assert max_abs(layer(x, observable=C), layer(x, observable=expanded)) <= TOL


@pytest.mark.parametrize("kind", list(OBS_SHAPES))
@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_e2_float32_observable_equals_double(g, kind):
    """E2 (B.4): a float32 observable gives a result torch.equal to that of
    observable.double()."""
    layer = make_layer(g)
    x = uniform_x()
    obs32 = randn_observable(kind, dtype=torch.float32)
    y32 = layer(x, observable=obs32)
    y64 = layer(x, observable=obs32.double())
    # torch.equal ignores dtype, so compare it separately.
    assert y32.dtype == y64.dtype
    assert torch.equal(y32, y64)


@pytest.mark.parametrize("requires_grad", [True, False], ids=["leaf_grad", "no_grad"])
@pytest.mark.parametrize("kind", list(OBS_SHAPES))
@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_e2_observable_untouched_by_forward_and_backward(g, kind, requires_grad):
    """E2 (B.5): the values and _version of observable are unchanged after
    forward and backward."""
    layer = make_layer(g)
    x = uniform_x()
    obs = randn_observable(kind).requires_grad_(requires_grad)
    before = obs.detach().clone()
    version = obs._version
    y = layer(x, observable=obs)
    assert obs._version == version
    assert torch.equal(obs.detach(), before)
    y.sum().backward()
    assert obs._version == version
    assert torch.equal(obs.detach(), before)


# ---------------------------------------------------------------------------
# E3 Fixed Z (exact)
# ---------------------------------------------------------------------------

E3_CONFIGS = {
    "ba_trainable": dict(ba_trainable=True),
    "ba_frozen": dict(ba_trainable=False),
    "postacts_trainable_randn": dict(
        ba_trainable=False,
        postact_weight_trainable=True,
        postact_bias_trainable=True,
    ),
}


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("fast_measure", [True, False], ids=["fm_true", "fm_false"])
@pytest.mark.parametrize("config", list(E3_CONFIGS))
@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_e3_observable_z_matches_fixed_z(g, config, fast_measure, mode):
    """E3: a fast_measure=False layer called without observable and a layer
    with the same state_dict and fast_measure in {True, False} called with
    observable=Z give outputs equal within 1e-12, in train() and in eval()."""
    reference = make_layer(g, fast_measure=False, **E3_CONFIGS[config])
    if config == "postacts_trainable_randn":
        with torch.no_grad():
            reference.postact_weights.copy_(torch.randn_like(reference.postact_weights))
            reference.postact_bias.copy_(torch.randn_like(reference.postact_bias))
    layer = make_layer(g, fast_measure=fast_measure, **E3_CONFIGS[config])
    layer.load_state_dict(reference.state_dict())
    reference.train(mode == "train")
    layer.train(mode == "train")
    x = uniform_x()
    expected = reference(x)
    y = layer(x, observable=z_observable())
    assert y.shape == expected.shape
    assert max_abs(y, expected) <= TOL


# ---------------------------------------------------------------------------
# E4 Gradients (exact)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(OBS_SHAPES))
@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_e4a_gradcheck_x_and_observable(g, kind):
    """E4(a): torch.autograd.gradcheck(lambda x, c: layer(x, observable=c),
    (x, c)) passes with default tolerances."""
    layer = make_layer(g)
    x = uniform_x().requires_grad_(True)
    c = randn_observable(kind).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x, c: layer(x, observable=c), (x, c))


@pytest.mark.parametrize("kind", list(OBS_SHAPES))
@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_e4b_theta_grad_matches_central_differences(g, kind):
    """E4(b): for L = sum(layer(x, c) * G) with G randn (7, 4), dL/dtheta
    matches central differences with step 1e-6 within 1e-6."""
    layer = make_layer(g)
    x = uniform_x()
    c = randn_observable(kind).requires_grad_(True)
    upstream = torch.randn(B, OUT_DIM, dtype=torch.float64)  # G of E4

    def loss():
        return (layer(x, observable=c) * upstream).sum()

    (grad,) = torch.autograd.grad(loss(), layer.theta)
    step = 1e-6
    numeric = torch.empty_like(grad)
    with torch.no_grad():
        flat = layer.theta.view(-1)
        for idx in range(flat.numel()):
            orig = flat[idx].item()
            flat[idx] = orig + step
            plus = loss().item()
            flat[idx] = orig - step
            minus = loss().item()
            flat[idx] = orig
            numeric.view(-1)[idx] = (plus - minus) / (2 * step)
    assert max_abs(grad, numeric) <= 1e-6


@pytest.mark.parametrize("kind", list(OBS_SHAPES))
def test_e4c_observable_grad_is_key_times_upstream_g3(kind):
    """E4(c): for g = 3, dL/dc equals K[:, :, None] * G[:, None, :] for a
    per-sample c and its sum over the batch for a shared c, within 1e-12."""
    layer = make_layer(3)
    x = uniform_x()
    c = randn_observable(kind).requires_grad_(True)
    upstream = torch.randn(B, OUT_DIM, dtype=torch.float64)  # G of E4
    (grad_c,) = torch.autograd.grad((layer(x, observable=c) * upstream).sum(), c)
    with torch.no_grad():
        K = key_matrix(layer, x)
    expected = K[:, :, None] * upstream[:, None, :]
    if kind == "shared":
        expected = expected.sum(dim=0)
    assert grad_c.shape == expected.shape
    assert max_abs(grad_c, expected) <= TOL


# ---------------------------------------------------------------------------
# E5 Dynamic coefficients (exact, g = 3)
# ---------------------------------------------------------------------------


def _e5_leaves():
    x1 = uniform_x().requires_grad_(True)
    x2 = uniform_x().requires_grad_(True)
    C1 = torch.randn(4 * IN_DIM, OUT_DIM, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, OUT_DIM, dtype=torch.float64, requires_grad=True)
    return x1, x2, C1, v


def _e5_output(layer, x1, x2, C1, v, detach_key=False):
    """k1 = cat(ones, forward_bloch(x1)[:, 0]).reshape(B, 4I) / sqrt(2I);
    C2 = C1[None] + k1[:, :, None] * v[:, None, :]; y = layer(x2, observable=C2)."""
    r1 = layer.forward_bloch(x1)[:, 0]
    k1 = torch.cat([torch.ones_like(r1[..., :1]), r1], dim=-1)
    k1 = k1.reshape(B, 4 * IN_DIM) / math.sqrt(2 * IN_DIM)
    if detach_key:
        k1 = k1.detach()
    C2 = C1[None] + k1[:, :, None] * v[:, None, :]
    return layer(x2, observable=C2)


def test_e5a_gradcheck_dynamic_coefficients():
    """E5(a): gradcheck over (x1, x2, C1, v) passes."""
    layer = make_layer(3)
    assert torch.autograd.gradcheck(
        lambda x1, x2, C1, v: _e5_output(layer, x1, x2, C1, v), _e5_leaves()
    )


def test_e5b_theta_grad_flows_through_key():
    """E5(b): d(sum y)/dtheta differs from the value computed with
    k1.detach() (max abs difference > 1e-6)."""
    layer = make_layer(3)
    x1, x2, C1, v = _e5_leaves()
    (full,) = torch.autograd.grad(_e5_output(layer, x1, x2, C1, v).sum(), layer.theta)
    (cut,) = torch.autograd.grad(
        _e5_output(layer, x1, x2, C1, v, detach_key=True).sum(), layer.theta
    )
    assert max_abs(full, cut) > 1e-6


# ---------------------------------------------------------------------------
# E6 Errors (CPU part)
# ---------------------------------------------------------------------------


def _call_entry(layer, x, entry):
    if entry == "forward_observable":
        return layer(x, observable=z_observable())
    return layer.forward_bloch(x)


@pytest.mark.parametrize("entry", ["forward_observable", "forward_bloch"])
def test_e6_flash_solver_raises_not_implemented(entry):
    """E6: forward(x, observable=Z) and forward_bloch(x) raise
    NotImplementedError for solver="flash", on CPU, before any kernel call.

    On CPU tensors the flash solver fails with a different exception (Triton
    rejects CPU pointers with a ValueError at launch; ImportError without
    Triton) and its base activation falls back to torch, so a
    NotImplementedError can only come from the up-front check of C.1/C.2.
    """
    layer = make_layer(3, solver="flash")
    x = uniform_x()
    with pytest.raises(NotImplementedError):
        _call_entry(layer, x, entry)


@pytest.mark.parametrize("entry", ["forward_observable", "forward_bloch"])
def test_e6_rpz_ansatz_raises_not_implemented(entry):
    """E6: forward(x, observable=Z) and forward_bloch(x) raise
    NotImplementedError for ansatz="rpz_encoding"."""
    layer = make_layer(3, ansatz="rpz_encoding")
    x = uniform_x()
    with pytest.raises(NotImplementedError):
        _call_entry(layer, x, entry)


def test_e6_exact_solver_rpz_bloch_measure_raises():
    """E6: torch_exact_solver(..., ansatz="rpz_encoding", bloch_measure=True)
    raises NotImplementedError."""
    layer = make_layer(3, ansatz="rpz_encoding")
    x = uniform_x()
    with pytest.raises(NotImplementedError):
        torch_exact_solver(
            x,
            layer._theta_natural(),
            layer._preacts_w_natural(),
            layer._preacts_b_natural(),
            REPS,
            ansatz="rpz_encoding",
            out_dim=OUT_DIM,
            dtype=torch.complex128,
            bloch_measure=True,
        )


@pytest.mark.parametrize(
    "bad_shape", [(12, 5), (12,), (8, 12, 4)], ids=["12x5", "12", "8x12x4"]
)
def test_e6_bad_observable_shape_raises_value_error(bad_shape):
    """E6: forward raises ValueError for observable shapes (12, 5), (12,) and
    (8, 12, 4) with B = 7."""
    layer = make_layer(3)
    x = uniform_x()
    assert x.shape[0] == 7
    with pytest.raises(ValueError):
        layer(x, observable=torch.zeros(bad_shape, dtype=torch.float64))


# ---------------------------------------------------------------------------
# E7 Backward compatibility ((a) is the rest of the suite)
# ---------------------------------------------------------------------------

# 80ef133, the commit the trainable-observable spec was written against.
BASELINE_COMMIT = "80ef133918169721ef28aec00a0c861fefb5683d"

# E7(b) recipe, run once against the baseline copy and once against the code
# under test, each in a fresh interpreter.
E7B_RECIPE = """\
import sys

import torch

import qkan
from qkan import QKAN, QKANLayer

print("QKAN_FILE=" + qkan.__file__, flush=True)
out = {}
torch.manual_seed(0)
x = torch.randn(5, 3)
for g in (-1, 3):
    for fm in (True, False):
        layer = QKANLayer(3, 4, reps=2, group=g, fast_measure=fm, seed=1)
        y = layer(x)  # train route
        y.sum().backward()
        out[f"train_g{g}_fm{fm}"] = y.detach().clone()
        out[f"theta_grad_g{g}_fm{fm}"] = layer.theta.grad.detach().clone()
        layer.eval()
        out[f"eval_g{g}_fm{fm}"] = layer(x).detach().clone()  # eval route
out["qkan_3_4_2"] = QKAN([3, 4, 2], reps=2, seed=1)(x).detach().clone()
torch.save(out, sys.argv[1])
"""


def _run_e7b_recipe(script, src, out_file):
    proc = subprocess.run(
        [sys.executable, str(script), str(out_file)],
        capture_output=True,
        text=True,
        # PYTHONPATH precedes the editable install's .pth entry, so the copy
        # under ``src`` is the one imported; the recipe prints it to prove it.
        env=dict(os.environ, PYTHONPATH=str(src)),
        cwd=str(script.parent),
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"recipe failed with PYTHONPATH={src}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("QKAN_FILE=")]
    assert len(lines) == 1, proc.stdout
    imported = Path(lines[0].split("=", 1)[1]).resolve()
    assert imported.is_relative_to(Path(src).resolve()), (
        f"imported {imported}, expected the copy under {src}"
    )
    return torch.load(out_file)


def test_e7b_default_outputs_bitwise_equal_to_baseline(tmp_path):
    """E7(b): with x = torch.randn(5, 3) after torch.manual_seed(0), for g in
    {-1, 3} and fast_measure in {True, False}, the train-route output of
    QKANLayer(3, 4, reps=2, group=g, fast_measure=fm, seed=1), the theta.grad
    of its sum and its eval-route output, plus QKAN([3, 4, 2], reps=2,
    seed=1)(x), are torch.equal between 80ef133 and the code under test."""
    repo = Path(qkan.__file__).resolve().parents[2]
    if shutil.which("git") is None or shutil.which("tar") is None:
        pytest.skip("git and tar are needed to extract the baseline")
    probe = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{BASELINE_COMMIT}^{{commit}}"],
        capture_output=True,
    )
    if probe.returncode != 0:
        pytest.skip(f"commit {BASELINE_COMMIT[:7]} is not available in {repo}")
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", BASELINE_COMMIT, "src/qkan"],
        capture_output=True,
    )
    if archive.returncode != 0:
        pytest.skip(f"git archive {BASELINE_COMMIT[:7]} failed: {archive.stderr!r}")
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()
    subprocess.run(
        ["tar", "-x", "-C", str(baseline_root)], input=archive.stdout, check=True
    )
    assert (baseline_root / "src" / "qkan" / "__init__.py").is_file()

    script = tmp_path / "e7b_recipe.py"
    script.write_text(E7B_RECIPE)
    baseline = _run_e7b_recipe(script, baseline_root / "src", tmp_path / "old.pt")
    current = _run_e7b_recipe(script, repo / "src", tmp_path / "new.pt")

    assert len(baseline) == 4 * 3 + 1
    assert sorted(current) == sorted(baseline)
    # torch.equal ignores dtype, so compare it separately.
    differing = [
        key
        for key in sorted(baseline)
        if baseline[key].dtype != current[key].dtype
        or not torch.equal(baseline[key], current[key])
    ]
    assert not differing, f"differs from {BASELINE_COMMIT[:7]}: {differing}"


def test_e7c_state_dict_keys_unchanged():
    """E7(c): sorted(QKANLayer(3, 4).state_dict()) is unchanged (A.3: the
    library owns no observable state)."""
    assert sorted(QKANLayer(3, 4).state_dict()) == [
        "base_weight",
        "mask",
        "postact_bias",
        "postact_weights",
        "preacts_bias",
        "preacts_weight",
        "theta",
    ]


# ---------------------------------------------------------------------------
# Extra (not in spec E): cheap direct checks of D.1, C.3/B.4, B.5/D.2,
# D.6 items 2-3, the B.3 epilogue terms that the spec-E fixtures leave at
# their identity values, the p_dtype cast of B.4/C.2, B.2's group=(1, I)
# case and the "pz" alias of C.1/C.3
# ---------------------------------------------------------------------------


def test_extra_state_vector_measure_bloch_known_states():
    """Extra (not in spec E), D.1/B.2: StateVector.measure_bloch returns
    (B, out, in, 3) with (r_x, r_y, r_z) of |0>, |1>, |+> and |+i>."""
    from qkan.torch_qc import StateVector

    s = 1 / math.sqrt(2)
    states = torch.tensor([[1, 0], [0, 1], [s, s], [s, 1j * s]], dtype=torch.complex128)
    expected = torch.tensor(
        [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0]], dtype=torch.float64
    )
    psi = StateVector(4, 1, 1, dtype=torch.complex128)
    psi.state = states[:, None, None, :].clone()
    r = psi.measure_bloch()
    assert r.shape == (4, 1, 1, 3)
    assert r.dtype == torch.float64
    assert max_abs(r[:, 0, 0], expected) <= TOL


@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_extra_exact_solver_bloch_measure_keyword(g):
    """Extra (not in spec E), C.3/B.4/B.5: torch_exact_solver and
    get_solver("exact") accept bloch_measure; True returns (B, G, I, 3) in the
    real dtype of c_dtype and ignores fast_measure; False is the unchanged Z
    path; forward_bloch is that output in p_dtype, whatever fast_measure."""
    layer = make_layer(g)
    x = uniform_x()

    def solve(solver=torch_exact_solver, dtype=torch.complex128, **kwargs):
        return solver(
            x,
            layer._theta_natural(),
            layer._preacts_w_natural(),
            layer._preacts_b_natural(),
            REPS,
            ansatz="pz_encoding",
            out_dim=OUT_DIM,
            dtype=dtype,
            **kwargs,
        )

    with torch.no_grad():
        r = solve(bloch_measure=True, fast_measure=True)
        assert r.shape == (B, N_CIRCUITS[g], IN_DIM, 3)
        assert r.dtype == torch.float64
        assert torch.equal(r, solve(bloch_measure=True, fast_measure=False))
        registry_r = solve(get_solver("exact"), bloch_measure=True, fast_measure=True)
        assert torch.equal(r, registry_r)
        assert solve(dtype=torch.complex64, bloch_measure=True).dtype == torch.float32
        for fm in (True, False):
            z_default = solve(fast_measure=fm)
            assert torch.equal(z_default, solve(bloch_measure=False, fast_measure=fm))
    assert max_abs(layer.forward_bloch(x), r) <= TOL
    twin = make_layer(g, fast_measure=False)
    twin.load_state_dict(layer.state_dict())
    assert torch.equal(twin.forward_bloch(x), layer.forward_bloch(x))


@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_extra_bloch_solver_gradcheck_trainable_preacts(g):
    """Extra (not in spec E), B.5/D.2: the bloch_measure output of
    torch_exact_solver passes gradcheck w.r.t. x, theta and trainable
    preacts (plain autograd)."""
    layer = make_layer(g, preact_trainable=True)
    x = uniform_x().requires_grad_(True)
    theta = layer._theta_natural().detach().clone().requires_grad_(True)
    pw = torch.randn_like(layer._preacts_w_natural()).requires_grad_(True)
    pb = torch.randn_like(layer._preacts_b_natural()).requires_grad_(True)

    def bloch(x, theta, pw, pb):
        return torch_exact_solver(
            x,
            theta,
            pw,
            pb,
            REPS,
            ansatz="pz_encoding",
            preacts_trainable=True,
            out_dim=OUT_DIM,
            dtype=torch.complex128,
            bloch_measure=True,
        )

    # The Z output would pass gradcheck too: make sure this is the Bloch one.
    assert bloch(x, theta, pw, pb).shape == (B, N_CIRCUITS[g], IN_DIM, 3)
    assert torch.autograd.gradcheck(bloch, (x, theta, pw, pb))


def test_extra_z_mode_solver_keywords_unchanged():
    """Extra (not in spec E), D.6 item 3 / A.3: without observable, both
    routes hand a registry solver exactly the baseline keywords (no
    bloch_measure). Uses a spy solver under a fresh registry name that
    delegates to torch_exact_solver and is removed afterwards."""
    calls = []

    class _SpySolver(QKANSolver):
        name = "_spy_test_observable_exact"

        def __call__(self, x, theta, preacts_weight, preacts_bias, reps, **kwargs):
            calls.append(dict(kwargs))
            return torch_exact_solver(
                x, theta, preacts_weight, preacts_bias, reps, **kwargs
            )

    registry = get_registry()
    assert _SpySolver.name not in registry
    register(_SpySolver())
    try:
        for fm in (True, False):
            layer = make_layer(3, solver=_SpySolver.name, fast_measure=fm)
            x = uniform_x()
            layer(x)  # train route
            layer.eval()
            layer(x)  # eval route
    finally:
        registry.pop(_SpySolver.name, None)
    assert _SpySolver.name not in get_registry()
    baseline_keywords = dict(
        device="cpu",
        ansatz="pz_encoding",
        group=(3,),
        preacts_trainable=False,
        out_dim=OUT_DIM,
        dtype=torch.complex128,
        checkpoint_reps=False,
    )
    assert calls == [
        dict(baseline_keywords, fast_measure=fm)
        for fm in (True, False)
        for _route in ("train", "eval")
    ]


@pytest.mark.parametrize("kind", list(OBS_SHAPES))
@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_extra_e2_readout_eval_route(g, kind):
    """Extra (not in spec E), B.3 / D.6 item 2: the E2 explicit-loop readout
    also holds on the eval route (after layer.eval(), whose epilogue uses the
    cached folded weights). Spec E2 runs on a freshly built layer (train
    route), and E3 reaches the eval route only with the shared Z."""
    layer = make_layer(g)
    layer.eval()
    x = uniform_x()
    obs = randn_observable(kind)
    y = layer(x, observable=obs)
    assert y.shape == (B, OUT_DIM)
    assert max_abs(y, _explicit_readout(layer, x, obs)) <= TOL


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("fast_measure", [True, False], ids=["fm_true", "fm_false"])
@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_extra_e3_pruned_mask_frozen_bias(g, fast_measure, mode):
    """Extra (not in spec E), B.3: the mask m and a frozen nonzero
    postact_bias (beta, applied under the existing condition) are kept in
    observable mode. This is the E3 comparison on a layer outside E3's three
    configurations, which all have an all-ones mask and a zero or trainable
    bias. QKANLayer has no pruning method of its own, so the pruned state
    comes in through load_state_dict, which recomputes _mask_is_identity and
    _pb_is_nonzero."""
    reference = make_layer(g, fast_measure=False, ba_trainable=True)
    state = {k: v.clone() for k, v in reference.state_dict().items()}
    state["mask"][1, 2] = 0.0
    state["postact_bias"].copy_(torch.randn_like(state["postact_bias"]))
    reference.load_state_dict(state)
    layer = make_layer(g, fast_measure=fast_measure, ba_trainable=True)
    layer.load_state_dict(state)
    # Preconditions: the masked and frozen-bias branches of the epilogue.
    assert not layer._mask_is_identity
    assert layer._pb_is_nonzero and not layer.postact_bias_trainable
    reference.train(mode == "train")
    layer.train(mode == "train")
    x = uniform_x()
    expected = reference(x)
    y = layer(x, observable=z_observable())
    assert y.shape == expected.shape
    assert max_abs(y, expected) <= TOL


@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_extra_forward_bloch_cast_to_p_dtype(g):
    """Extra (not in spec E), B.4 / C.2: with c_dtype=complex64 the exact
    solver's Bloch output is float32, and forward_bloch returns it cast to
    p_dtype (float64), within 1e-5 of the complex128 fixture. The fixture
    itself cannot detect a missing cast: its solver output is already float64."""
    ref = make_layer(g)
    layer = make_layer(g, c_dtype=torch.complex64)
    layer.load_state_dict(ref.state_dict())
    x = uniform_x()
    with torch.no_grad():
        raw = torch_exact_solver(
            x,
            layer._theta_natural(),
            layer._preacts_w_natural(),
            layer._preacts_b_natural(),
            REPS,
            ansatz="pz_encoding",
            out_dim=OUT_DIM,
            dtype=torch.complex64,
            bloch_measure=True,
        )
    # B.4: the Bloch output (not the Z one, also float32) before the layer's cast
    assert raw.shape == (B, N_CIRCUITS[g], IN_DIM, 3)
    assert raw.dtype == torch.float32
    r = layer.forward_bloch(x)
    assert r.shape == (B, N_CIRCUITS[g], IN_DIM, 3)
    assert r.dtype == torch.float64
    assert max_abs(r, ref.forward_bloch(x)) <= 1e-5


@pytest.mark.parametrize("kind", list(OBS_SHAPES))
def test_extra_e2_group_1_by_in_dim(kind):
    """Extra (not in spec E), B.2: group=(1, I) has a 4-D theta
    (1, I, R+1, 2) but G = theta.shape[0] = 1, so every output node reads
    circuit 0, and the E2 explicit-loop readout holds. The spec fixtures reach
    G = 1 only through group=3 (3-D theta)."""
    layer = make_layer((1, IN_DIM))
    assert tuple(layer.theta.shape) == (1, IN_DIM, REPS + 1, 2)
    x = uniform_x()
    assert layer.forward_bloch(x).shape == (B, 1, IN_DIM, 3)
    obs = randn_observable(kind)
    y = layer(x, observable=obs)
    assert y.shape == (B, OUT_DIM)
    assert max_abs(y, _explicit_readout(layer, x, obs)) <= TOL


@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_extra_pz_alias_accepted(g):
    """Extra (not in spec E), C.1 / C.3: ansatz="pz" is accepted like
    "pz_encoding" by forward(x, observable), forward_bloch and
    torch_exact_solver(..., bloch_measure=True), with the same values."""
    ref = make_layer(g)
    alias = make_layer(g, ansatz="pz")
    alias.load_state_dict(ref.state_dict())
    x = uniform_x()
    obs = randn_observable("per_sample")
    r_alias = alias.forward_bloch(x)
    assert r_alias.shape == (B, N_CIRCUITS[g], IN_DIM, 3)
    assert max_abs(r_alias, ref.forward_bloch(x)) <= TOL
    y_alias = alias(x, observable=obs)
    assert y_alias.shape == (B, OUT_DIM)
    assert max_abs(y_alias, ref(x, observable=obs)) <= TOL

    def solve(ansatz):
        return torch_exact_solver(
            x,
            ref._theta_natural(),
            ref._preacts_w_natural(),
            ref._preacts_b_natural(),
            REPS,
            ansatz=ansatz,
            out_dim=OUT_DIM,
            dtype=torch.complex128,
            bloch_measure=True,
        )

    with torch.no_grad():
        r_solver = solve("pz")
        assert r_solver.shape == (B, N_CIRCUITS[g], IN_DIM, 3)
        assert max_abs(r_solver, solve("pz_encoding")) <= TOL


# ---------------------------------------------------------------------------
# Extra (not in spec E): validation happens before any work, B.4's
# "real-valued" observable, and the CUDA file's skip gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case,error",
    [
        ("bad_shape", ValueError),
        ("rpz_ansatz", NotImplementedError),
        ("complex", ValueError),
        ("other_device", ValueError),
    ],
    ids=["bad_shape", "rpz_ansatz", "complex", "other_device"],
)
def test_extra_rejected_observable_leaves_batchnorm_untouched(case, error):
    """Extra (not in spec E), C.1 / B.4: forward(x, observable) validates
    before batch norm runs, so a rejected call leaves the running statistics
    as they were (and launches no base-activation kernel on CUDA). B.4's
    "real-valued tensor on x.device" is checked up front too; the meta device
    stands in for "another device" without needing a GPU."""
    if case == "rpz_ansatz":
        layer = make_layer(3, ansatz="rpz_encoding", is_batchnorm=True)
        obs = z_observable()
    else:
        layer = make_layer(3, is_batchnorm=True)
        if case == "bad_shape":
            obs = torch.zeros(4 * IN_DIM, OUT_DIM + 1, dtype=torch.float64)
        elif case == "complex":
            obs = z_observable().to(torch.complex128)
        else:
            obs = torch.zeros(4 * IN_DIM, OUT_DIM, dtype=torch.float64, device="meta")
    layer.train()
    before = {k: v.clone() for k, v in layer.bn.state_dict().items()}
    with pytest.raises(error):
        layer(uniform_x(), observable=obs)
    after = layer.bn.state_dict()
    assert sorted(after) == sorted(before)
    assert all(torch.equal(after[k], before[k]) for k in before)


@pytest.mark.parametrize("kind", list(OBS_SHAPES))
def test_extra_complex_observable_raises_value_error(kind):
    """Extra (not in spec E), B.4: the observable is real-valued; a complex
    one is rejected instead of silently losing its imaginary part in
    observable.to(p_dtype)."""
    layer = make_layer(3)
    obs = torch.zeros(OBS_SHAPES[kind], dtype=torch.complex128)
    with pytest.raises(ValueError):
        layer(uniform_x(), observable=obs)


def test_extra_cute_available_whenever_extension_imports():
    """Extra (not in spec E), D.5 item 1: test_observable_cute.py is skipped
    unless qkan.solver._CUTE_AVAILABLE, and cute.py sets that flag to False
    on any ImportError in its import block. A misnamed wrapper there would
    switch the cute solver off for every user and silently skip every CUDA
    test, so check the flag (and the new wrappers) wherever the prebuilt
    extension loads."""
    import importlib

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    try:
        importlib.import_module("qkan._C")
    except ImportError:
        pytest.skip("the prebuilt CuTe extension qkan._C is not installed")
    from qkan.solver import _CUTE_AVAILABLE
    from qkan.solver.cute import cute as cute_module

    assert _CUTE_AVAILABLE
    assert callable(cute_module.cute_pz_bloch_forward)
    assert callable(cute_module.cute_pz_bloch_backward)


# ---------------------------------------------------------------------------
# Extra (not in spec E): solver_kwargs, and layer options outside the E2/E3
# fixtures (batch norm, trainable preacts, p_dim=2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_extra_solver_kwargs_bloch_measure_entry_is_ignored(g, value):
    """Extra (not in spec E), C.3 / A.3: QKANLayer passes bloch_measure only
    from forward(x, observable) and forward_bloch, so a 'bloch_measure' entry
    in solver_kwargs (meaningless before this feature) changes nothing: the
    Z-mode routes, forward_no_sum, observable mode and forward_bloch give the
    same results as without it."""
    ref = make_layer(g)
    layer = make_layer(g, solver_kwargs={"bloch_measure": value})
    layer.load_state_dict(ref.state_dict())
    x = uniform_x()
    obs = randn_observable("per_sample")
    for training in (True, False):
        ref.train(training)
        layer.train(training)
        assert torch.equal(layer(x), ref(x))
        assert torch.equal(layer.forward_no_sum(x), ref.forward_no_sum(x))
        assert torch.equal(layer(x, observable=obs), ref(x, observable=obs))
        assert torch.equal(layer.forward_bloch(x), ref.forward_bloch(x))


E3_EXTRA_CONFIGS = {
    "batchnorm": dict(ba_trainable=True, is_batchnorm=True),
    "preact_trainable": dict(ba_trainable=True, preact_trainable=True),
    "p_dim2": dict(ba_trainable=True, p_dim=2),
}


def _grads(y, upstream, inputs):
    return torch.autograd.grad((y * upstream).sum(), inputs, allow_unused=True)


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("fast_measure", [True, False], ids=["fm_true", "fm_false"])
@pytest.mark.parametrize("config", list(E3_EXTRA_CONFIGS))
@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_extra_e3_layer_variants(g, config, fast_measure, mode):
    """Extra (not in spec E), B.3 / B.5 / D.6: the E3 identity (observable=Z
    gives the fixed-Z fast_measure=False output) and equal gradients w.r.t.
    x and every trainable parameter, for layer options the spec fixtures do
    not use: batch norm (B.3: the circuits and sigma see the normalized x),
    trainable preacts (B.5) and p_dim=2 storage (D.6 items 4.2 and 5 read the
    natural-rank views)."""
    cfg = E3_EXTRA_CONFIGS[config]
    reference = make_layer(g, fast_measure=False, **cfg)
    if cfg.get("preact_trainable"):
        with torch.no_grad():
            reference.preacts_weight.copy_(torch.randn_like(reference.preacts_weight))
            reference.preacts_bias.copy_(torch.randn_like(reference.preacts_bias))
    layer = make_layer(g, fast_measure=fast_measure, **cfg)
    layer.load_state_dict(reference.state_dict())
    reference.train(mode == "train")
    layer.train(mode == "train")
    x_ref = uniform_x().requires_grad_(True)
    x_obs = x_ref.detach().clone().requires_grad_(True)
    expected = reference(x_ref)
    y = layer(x_obs, observable=z_observable())
    assert y.shape == expected.shape
    assert max_abs(y, expected) <= TOL
    upstream = torch.randn(B, OUT_DIM, dtype=torch.float64)
    names = [n for n, p in reference.named_parameters() if p.requires_grad]
    if cfg.get("preact_trainable"):
        assert {"preacts_weight", "preacts_bias"} <= set(names)
    ref_params = dict(reference.named_parameters())
    obs_params = dict(layer.named_parameters())
    ref_grads = _grads(expected, upstream, [x_ref] + [ref_params[n] for n in names])
    obs_grads = _grads(y, upstream, [x_obs] + [obs_params[n] for n in names])
    for name, a, b in zip(["x"] + names, ref_grads, obs_grads):
        assert (a is None) == (b is None), name
        if a is not None:
            assert max_abs(a, b) <= 1e-10, name


@pytest.mark.parametrize("g", GROUPS, ids=_gid)
def test_extra_p_dim2_matches_p_dim4(g):
    """Extra (not in spec E), D.6 items 4.2 and 5: under p_dim=2 the
    parameters are stored flattened, and forward_bloch / forward(x,
    observable) must read the natural-rank views: they match a p_dim=4 layer
    holding the same values, for a random observable and trainable preacts."""
    p4 = make_layer(g, preact_trainable=True)
    p2 = make_layer(g, preact_trainable=True, p_dim=2)
    assert p2.theta.shape != p4.theta.shape  # storage really differs
    with torch.no_grad():
        for name in ("theta", "preacts_weight", "preacts_bias"):
            getattr(p4, name).copy_(torch.randn_like(getattr(p4, name)))
        p2._theta_natural().copy_(p4.theta)
        p2._preacts_w_natural().copy_(p4.preacts_weight)
        p2._preacts_b_natural().copy_(p4.preacts_bias)
        for name in ("base_weight", "postact_weights", "postact_bias", "mask"):
            assert torch.equal(
                p2._as_oi(getattr(p2, name)), p4._as_oi(getattr(p4, name))
            )
    x = uniform_x()
    obs = randn_observable("per_sample")
    r2 = p2.forward_bloch(x)
    assert r2.shape == (B, N_CIRCUITS[g], IN_DIM, 3)
    assert max_abs(r2, p4.forward_bloch(x)) <= TOL
    assert max_abs(p2(x, observable=obs), p4(x, observable=obs)) <= TOL
