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

"""Trainable-observable acceptance tests on the CuTe solver (CUDA).

Covers the CUDA criteria of the trainable-observable specification
(QKAN_TRAINABLE_OBSERVABLE_SPEC.md, section E): the CuTe half of E6, and
E8-E11.  The exact-solver criteria live in ``test_observable_exact.py``.

Fixture (spec E): CUDA, ``in_dim = 5``, ``out_dim = 3``, ``B = 37``,
``p_dtype=torch.float32``, default ``c_dtype``, ``ba_trainable=False``.  The
reference is the same computation with ``solver="exact"`` on CUDA, layer pairs
share parameters through ``load_state_dict``, and inputs are ``torch.randn``
with fixed seeds.  Every compared quantity must satisfy
``eps(e, a) = max|e - a| / (max|e| + 1e-6) < 1e-4``, with ``e`` from ``exact``.
That bound assumes IEEE float32 matmuls, PyTorch's default, so an autouse
fixture undoes a session-wide TF32 setting for each test.

The two raw-binding guard cases of E6 run in their own interpreter, as in
``test_cute_guards.py``: a launcher that gets them wrong can poison the CUDA
context (or, for the int32 case, try to allocate ~16 GiB) for everything else.
"""

import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

import qkan  # noqa: E402
from qkan import QKANLayer  # noqa: E402
from qkan.solver import (  # noqa: E402
    _CUTE_AVAILABLE,
    cute_exact_solver,
    torch_exact_solver,
)

pytestmark = pytest.mark.skipif(
    not (_CUTE_AVAILABLE and torch.cuda.is_available()),
    reason="needs the CuTe extension and a CUDA device",
)


@pytest.fixture(autouse=True)
def _ieee_fp32_cuda_matmul():
    """Run each test with IEEE float32 CUDA matmuls (PyTorch's default).

    Not a spec E criterion: it keeps a session-wide TF32 setting (made by
    another module or a conftest) from failing correct code, since TF32 moves
    the exact reference's complex64 einsums by ~1e-3.  ``fp32_precision``
    is used where it exists: once TF32 was set through it, PyTorch refuses to
    read the legacy ``get_float32_matmul_precision()``.
    """
    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "fp32_precision"):
        previous = matmul.fp32_precision
        matmul.fp32_precision = "ieee"
        yield
        matmul.fp32_precision = previous
    else:  # older PyTorch: the legacy setting is the only one
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        yield
        torch.set_float32_matmul_precision(previous)


DEVICE = "cuda"
BATCH, IN_DIM, OUT_DIM = 37, 5, 3
N_COEF = 4 * IN_DIM  # observable rows: Pauli coefficients (I, X, Y, Z) per input
TOL = 1e-4

# The spec fixes the seed policy (torch.randn, fixed seeds), not the values.
SEED_LAYER = 7
SEED_X, SEED_THETA, SEED_PW, SEED_PB, SEED_COTANGENT = 11, 12, 13, 14, 15
SEED_OBSERVABLE = 16
SEED_X1, SEED_X2, SEED_C1, SEED_V = 21, 22, 23, 24

GROUPS = [pytest.param(IN_DIM, id="group5"), pytest.param(-1, id="group-1")]
OBSERVABLE_SHAPES = {
    "shared": (N_COEF, OUT_DIM),
    "per-sample": (BATCH, N_COEF, OUT_DIM),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _randn(*shape, seed):
    """``torch.randn`` from a fixed seed, drawn on the CPU and moved to CUDA."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator).to(DEVICE)


def _eps(expected, actual):
    """Spec E error metric, with ``expected`` from the exact solver."""
    e = expected.detach().double()
    a = actual.detach().double()
    return ((e - a).abs().max() / (e.abs().max() + 1e-6)).item()


def _assert_parity(expected, actual, what):
    assert expected is not None, f"{what}: the exact reference produced no tensor"
    assert actual is not None, f"{what}: the cute side produced no tensor"
    assert actual.shape == expected.shape, (
        f"{what}: cute shape {tuple(actual.shape)} != exact shape "
        f"{tuple(expected.shape)}"
    )
    err = _eps(expected, actual)
    assert err < TOL, f"{what}: eps = {err:.3e} (limit {TOL:.0e})"


def _assert_shape(tensor, shape, what):
    assert tuple(tensor.shape) == shape, (
        f"{what}: shape {tuple(tensor.shape)}, expected {shape}"
    )


def _assert_grad_parity(ref, out, ref_wrt, out_wrt):
    """Compare d sum(out * G) / d(input) for every named input, G randn."""
    cotangent = _randn(*ref.shape, seed=SEED_COTANGENT)
    names = list(ref_wrt)
    ref_grads = torch.autograd.grad(
        (ref * cotangent).sum(), [ref_wrt[n] for n in names], allow_unused=True
    )
    out_grads = torch.autograd.grad(
        (out * cotangent).sum(), [out_wrt[n] for n in names], allow_unused=True
    )
    for name, expected, actual in zip(names, ref_grads, out_grads):
        _assert_parity(expected, actual, f"gradient w.r.t. {name}")


def _solver_inputs(theta_rows, reps, preacts_trainable):
    """CUDA leaves for the solver-level comparisons (E8, E10)."""
    x = _randn(BATCH, IN_DIM, seed=SEED_X).requires_grad_()
    theta = _randn(theta_rows, IN_DIM, reps + 1, 2, seed=SEED_THETA)
    theta.requires_grad_()
    # Always on the device: the CuTe launcher checks the preacts even unused.
    pw = _randn(theta_rows, IN_DIM, reps, seed=SEED_PW)
    pb = _randn(theta_rows, IN_DIM, reps, seed=SEED_PB)
    pw.requires_grad_(preacts_trainable)
    pb.requires_grad_(preacts_trainable)
    return x, theta, pw, pb


def _solver_pair(theta_rows, reps, preacts_trainable, **mode):
    """torch_exact_solver (complex64) and cute_exact_solver (float32) on the
    same CUDA tensors; returns both outputs and the leaves to differentiate."""
    x, theta, pw, pb = _solver_inputs(theta_rows, reps, preacts_trainable)
    common = dict(
        ansatz="pz_encoding",
        out_dim=theta_rows,
        preacts_trainable=preacts_trainable,
        **mode,
    )
    ref = torch_exact_solver(x, theta, pw, pb, reps, dtype=torch.complex64, **common)
    out = cute_exact_solver(x, theta, pw, pb, reps, dtype=torch.float32, **common)
    wrt = {"x": x, "theta": theta}
    if preacts_trainable:
        wrt.update(preacts_weight=pw, preacts_bias=pb)
    return ref, out, wrt


def _layer_pair(group):
    """Exact/cute ``QKANLayer`` pair of the spec E fixture, sharing parameters."""
    common = dict(
        reps=2,
        group=group,
        device=DEVICE,
        ansatz="pz_encoding",
        ba_trainable=False,
        p_dtype=torch.float32,
    )
    exact = QKANLayer(IN_DIM, OUT_DIM, solver="exact", seed=SEED_LAYER, **common)
    cute = QKANLayer(IN_DIM, OUT_DIM, solver="cute", **common)
    cute.load_state_dict(exact.state_dict())
    return exact, cute


def _n_circuits(group):
    """G of spec B.2: 1 for group = in_dim, out_dim for group = -1."""
    return 1 if group == IN_DIM else OUT_DIM


def _layer_x():
    return _randn(BATCH, IN_DIM, seed=SEED_X).requires_grad_()


def _observable(kind):
    return _randn(*OBSERVABLE_SHAPES[kind], seed=SEED_OBSERVABLE).requires_grad_()


def _dynamic_coefficients(layer, leaves):
    """The E5 construction: the key of ``x1`` adds a rank-1 update to ``C1``,
    which is then the per-sample observable read out on ``x2``."""
    r1 = layer.forward_bloch(leaves["x1"])[:, 0]
    k1 = torch.cat([torch.ones_like(r1[..., :1]), r1], -1).reshape(BATCH, N_COEF)
    k1 = k1 / math.sqrt(2 * IN_DIM)
    c2 = leaves["C1"][None] + k1[:, :, None] * leaves["v"][:, None, :]
    return layer(leaves["x2"], observable=c2)


def _dynamic_leaves():
    """Leaves ``x1``, ``x2`` ``(B, I)``, ``C1`` ``(4*I, O)``, ``v`` ``(B, O)``."""
    return {
        "x1": _randn(BATCH, IN_DIM, seed=SEED_X1).requires_grad_(),
        "x2": _randn(BATCH, IN_DIM, seed=SEED_X2).requires_grad_(),
        "C1": _randn(N_COEF, OUT_DIM, seed=SEED_C1).requires_grad_(),
        "v": _randn(BATCH, OUT_DIM, seed=SEED_V).requires_grad_(),
    }


# Raw-binding cases run in a fresh interpreter against this checkout.
_SNIPPET_PREAMBLE = '''
import torch
from qkan.solver.cute.cute_ops import _get_ext

ext = _get_ext()


def attempt(tag, call):
    """Print one line per call: "<tag>: OK <result>" or "<tag>: RAISED ..."."""
    try:
        result = call()
    except Exception as error:
        lines = str(error).strip().splitlines()
        first = lines[0][:300] if lines else ""
        print(f"{tag}: RAISED {type(error).__name__} | {first}", flush=True)
    else:
        print(f"{tag}: OK {result}", flush=True)
'''


def _run_snippet(body, timeout=300):
    src = Path(qkan.__file__).resolve().parents[2] / "src"
    return subprocess.run(
        [sys.executable, "-c", _SNIPPET_PREAMBLE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(src)),
        timeout=timeout,
    )


def _tagged(result, tag):
    """The text after ``"<tag>: "`` on the snippet's stdout, or None."""
    prefix = f"{tag}: "
    for line in result.stdout.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    return None


def _diagnostics(result):
    return f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


# ---------------------------------------------------------------------------
# E6 (CUDA part): errors raised by the CuTe solver and bindings
# ---------------------------------------------------------------------------


def test_e6_cute_solver_bloch_measure_rejects_rpz_encoding():
    """E6 / C.3 / D.5 item 3: ``cute_exact_solver(..., ansatz="rpz_encoding",
    bloch_measure=True)`` raises ``NotImplementedError``."""
    reps = 2
    x = _randn(BATCH, IN_DIM, seed=SEED_X)
    theta = _randn(OUT_DIM, IN_DIM, reps + 1, 1, seed=SEED_THETA)  # rpz layout
    pw = _randn(OUT_DIM, IN_DIM, reps, seed=SEED_PW)
    pb = _randn(OUT_DIM, IN_DIM, reps, seed=SEED_PB)
    with pytest.raises(NotImplementedError):
        cute_exact_solver(
            x,
            theta,
            pw,
            pb,
            reps,
            ansatz="rpz_encoding",
            out_dim=OUT_DIM,
            dtype=torch.float32,
            bloch_measure=True,
        )


def test_e6_pz_bloch_backward_rejects_z_shaped_grad_output():
    """E6 / C.4 / D.3 item 3: ``pz_bloch_backward`` raises ``RuntimeError``
    when ``grad_output`` has the Z-mode shape ``(B, O, I)``.

    The same call with the Bloch shape ``(B, O, I, 3)`` runs as a control, so
    the error is attributable to the shape check.
    """
    result = _run_snippet(f"""
        B, O, I, R = {BATCH}, {OUT_DIM}, {IN_DIM}, 2
        generator = torch.Generator().manual_seed({SEED_X})

        def randn(*shape):
            return torch.randn(*shape, generator=generator).to("cuda")

        x, theta = randn(B, I), randn(O, I, R + 1, 2)
        pw, pb = randn(O, I, R), randn(O, I, R)
        z_shaped, bloch_shaped = randn(B, O, I), randn(B, O, I, 3)

        def shapes(grads):
            return [None if g is None else tuple(g.shape) for g in grads]

        attempt("z-shaped", lambda: shapes(
            ext.pz_bloch_backward(x, theta, pw, pb, z_shaped, False, 32)))
        attempt("bloch-shaped", lambda: shapes(
            ext.pz_bloch_backward(x, theta, pw, pb, bloch_shaped, False, 32)))
    """)
    diagnostics = _diagnostics(result)
    rejected = _tagged(result, "z-shaped")
    assert rejected is not None, "snippet did not reach the call" + diagnostics
    assert rejected.startswith("RAISED RuntimeError |"), (
        "grad_output of shape (B, O, I) must raise RuntimeError" + diagnostics
    )
    accepted = _tagged(result, "bloch-shaped")
    assert accepted is not None and accepted.startswith("OK"), (
        "control: grad_output of shape (B, O, I, 3) must be accepted" + diagnostics
    )


def test_e6_pz_bloch_forward_int32_guard_not_oom():
    """E6 / D.3 item 4: ``pz_bloch_forward`` raises the int32 ``RuntimeError``
    (not an out-of-memory error) once ``3 * O * I`` exceeds ``2**31 - 1``.

    The inputs are the spec's expanded views, so the test allocates nothing
    sizeable itself.  The snippet caps its allocator at 1 GiB: an
    implementation that allocates (or makes the views contiguous) before the
    guard then fails with OOM, which this test reports, instead of taking
    ~16 GiB of a shared GPU and launching with wrapped offsets.  The same
    inputs at ``N = 4`` run as a control.
    """
    result = _run_snippet("""
        N = 715_827_883
        assert 3 * N > 2**31 - 1 >= 2 * N  # 3*O*I wraps; theta.numel() does not

        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, (1 << 30) / total))

        def inputs(n):
            x = torch.zeros(1, 1, device="cuda").expand(1, n)
            theta = torch.zeros(1, 1, 1, 2, device="cuda").expand(1, n, 1, 2)
            pw = pb = torch.zeros(1, device="cuda")
            return x, theta, pw, pb

        def call(n):
            return tuple(ext.pz_bloch_forward(*inputs(n), False, False).shape)

        def call_backward(n):
            grad_output = torch.zeros(1, 1, 1, 3, device="cuda").expand(1, 1, n, 3)
            grads = ext.pz_bloch_backward(*inputs(n), grad_output, False, 32)
            return [None if g is None else tuple(g.shape) for g in grads]

        attempt("oversized", lambda: call(N))
        attempt("control", lambda: call(4))
        attempt("oversized-backward", lambda: call_backward(N))
        attempt("control-backward", lambda: call_backward(4))
    """)
    diagnostics = _diagnostics(result)
    oversized = _tagged(result, "oversized")
    assert oversized is not None, "snippet did not reach the call" + diagnostics
    assert "out of memory" not in oversized.lower(), (
        "got an out-of-memory error instead of the int32 guard, which must "
        "run before any allocation" + diagnostics
    )
    assert oversized.startswith("RAISED RuntimeError |"), (
        "expected the int32 RuntimeError of D.3 item 4" + diagnostics
    )
    assert "3 * out_dim * in_dim" in oversized and "int32" in oversized, (
        "expected the int32 guard of D.3 item 4, not another check" + diagnostics
    )
    control = _tagged(result, "control")
    assert control == "OK (1, 1, 4, 3)", (
        "control: the same expanded inputs with N = 4 must give (B, O, I, 3)"
        + diagnostics
    )
    # Extra (not in spec E): D.3 item 4 puts the same guard in the backward.
    oversized_bwd = _tagged(result, "oversized-backward")
    assert oversized_bwd is not None, "snippet did not reach the backward" + diagnostics
    assert oversized_bwd.startswith("RAISED RuntimeError |"), (
        "expected the int32 RuntimeError of D.3 item 4 from pz_bloch_backward"
        + diagnostics
    )
    assert "3 * out_dim * in_dim" in oversized_bwd and "int32" in oversized_bwd, (
        "expected the int32 guard of D.3 item 4, not another check" + diagnostics
    )
    control_bwd = _tagged(result, "control-backward")
    assert control_bwd == "OK [(1, 4), (1, 4, 1, 2), None, None]", (
        "control: the backward with N = 4 must run" + diagnostics
    )


# ---------------------------------------------------------------------------
# E8: CuTe solver parity in Bloch mode
# ---------------------------------------------------------------------------


def _e8_parametrize(test):
    """theta (1 or 3, 5, R+1, 2) x preacts_trainable x R, ids in that order."""
    test = pytest.mark.parametrize(
        "theta_rows", [1, 3], ids=["theta-rows1", "theta-rows3"]
    )(test)
    test = pytest.mark.parametrize(
        "preacts_trainable", [False, True], ids=["fixed-preacts", "trainable-preacts"]
    )(test)
    return pytest.mark.parametrize("reps", [1, 2, 3], ids=lambda r: f"R{r}")(test)


@_e8_parametrize
def test_e8_cute_solver_bloch_output_parity(theta_rows, preacts_trainable, reps):
    """E8: with ``bloch_measure=True``, ``cute_exact_solver`` (float32) and
    ``torch_exact_solver`` (complex64) return ``(37, G, 5, 3)`` (G =
    ``theta.shape[0]``, B.4: float32) and agree to eps < 1e-4."""
    ref, out, _ = _solver_pair(theta_rows, reps, preacts_trainable, bloch_measure=True)
    shape = (BATCH, theta_rows, IN_DIM, 3)
    _assert_shape(ref, shape, "torch_exact_solver(bloch_measure=True)")
    _assert_shape(out, shape, "cute_exact_solver(bloch_measure=True)")
    assert ref.dtype == torch.float32, f"exact Bloch dtype {ref.dtype} (B.4)"
    assert out.dtype == torch.float32, f"cute Bloch dtype {out.dtype} (B.4)"
    _assert_parity(ref, out, "Bloch output")


@_e8_parametrize
def test_e8_cute_solver_bloch_gradient_parity(theta_rows, preacts_trainable, reps):
    """E8: gradients of ``sum(out * G)`` (G randn) w.r.t. ``x``, ``theta`` and,
    when trainable, the preacts weight and bias agree to eps < 1e-4."""
    ref, out, wrt = _solver_pair(
        theta_rows, reps, preacts_trainable, bloch_measure=True
    )
    shape = (BATCH, theta_rows, IN_DIM, 3)
    _assert_shape(ref, shape, "torch_exact_solver(bloch_measure=True)")
    _assert_shape(out, shape, "cute_exact_solver(bloch_measure=True)")
    _assert_grad_parity(ref, out, wrt, wrt)


# ---------------------------------------------------------------------------
# E9: CuTe layer parity (train route, autograd)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group", GROUPS)
def test_e9_forward_bloch_output_parity(group):
    """E9 / C.2: ``forward_bloch(x)`` of a cute/exact layer pair; shape
    ``(B, G, I, 3)`` in ``p_dtype``, eps < 1e-4."""
    exact, cute = _layer_pair(group)
    x = _layer_x()
    ref = exact.forward_bloch(x)
    out = cute.forward_bloch(x)
    shape = (BATCH, _n_circuits(group), IN_DIM, 3)
    _assert_shape(ref, shape, "exact forward_bloch(x)")
    _assert_shape(out, shape, "cute forward_bloch(x)")
    assert out.dtype == torch.float32, f"cute forward_bloch dtype {out.dtype}"
    _assert_parity(ref, out, "forward_bloch(x)")


@pytest.mark.parametrize("group", GROUPS)
def test_e9_forward_bloch_gradient_parity(group):
    """E9 / C.2 / B.5: gradients of ``forward_bloch(x)`` w.r.t. ``x`` and
    ``theta`` agree to eps < 1e-4."""
    exact, cute = _layer_pair(group)
    x = _layer_x()
    ref = exact.forward_bloch(x)
    out = cute.forward_bloch(x)
    shape = (BATCH, _n_circuits(group), IN_DIM, 3)
    _assert_shape(ref, shape, "exact forward_bloch(x)")
    _assert_shape(out, shape, "cute forward_bloch(x)")
    _assert_grad_parity(
        ref, out, {"x": x, "theta": exact.theta}, {"x": x, "theta": cute.theta}
    )


@pytest.mark.parametrize("observable_kind", list(OBSERVABLE_SHAPES))
@pytest.mark.parametrize("group", GROUPS)
def test_e9_layer_observable_output_parity(group, observable_kind):
    """E9 / C.1: ``layer(x, observable)`` with a randn ``(20, 3)`` or
    ``(37, 20, 3)`` observable (``requires_grad=True``), eps < 1e-4."""
    exact, cute = _layer_pair(group)
    x = _layer_x()
    observable = _observable(observable_kind)
    ref = exact(x, observable=observable)
    out = cute(x, observable=observable)
    _assert_shape(ref, (BATCH, OUT_DIM), "exact layer(x, observable)")
    _assert_parity(ref, out, "layer(x, observable)")


@pytest.mark.parametrize("observable_kind", list(OBSERVABLE_SHAPES))
@pytest.mark.parametrize("group", GROUPS)
def test_e9_layer_observable_gradient_parity(group, observable_kind):
    """E9 / C.1 / B.5: gradients of ``layer(x, observable)`` w.r.t. ``x``,
    ``theta`` and ``observable`` agree to eps < 1e-4."""
    exact, cute = _layer_pair(group)
    x = _layer_x()
    observable = _observable(observable_kind)
    ref = exact(x, observable=observable)
    out = cute(x, observable=observable)
    _assert_shape(ref, (BATCH, OUT_DIM), "exact layer(x, observable)")
    _assert_shape(out, (BATCH, OUT_DIM), "cute layer(x, observable)")
    _assert_grad_parity(
        ref,
        out,
        {"x": x, "theta": exact.theta, "observable": observable},
        {"x": x, "theta": cute.theta, "observable": observable},
    )


def test_e9_dynamic_coefficients_output_parity():
    """E9 (E5 construction, ``in_dim = 5``, ``g = 5``): the output of
    ``layer(x2, observable=C1 + k1 v)`` agrees to eps < 1e-4."""
    exact, cute = _layer_pair(IN_DIM)
    leaves = _dynamic_leaves()
    ref = _dynamic_coefficients(exact, leaves)
    out = _dynamic_coefficients(cute, leaves)
    _assert_shape(ref, (BATCH, OUT_DIM), "exact E5 output")
    _assert_parity(ref, out, "E5 construction output")


def test_e9_dynamic_coefficients_gradient_parity():
    """E9 (E5 construction, ``in_dim = 5``, ``g = 5``): gradients w.r.t.
    ``x1``, ``x2``, ``C1``, ``v`` and ``theta`` agree to eps < 1e-4."""
    exact, cute = _layer_pair(IN_DIM)
    leaves = _dynamic_leaves()
    ref = _dynamic_coefficients(exact, leaves)
    out = _dynamic_coefficients(cute, leaves)
    _assert_shape(ref, (BATCH, OUT_DIM), "exact E5 output")
    _assert_shape(out, (BATCH, OUT_DIM), "cute E5 output")
    _assert_grad_parity(
        ref, out, dict(leaves, theta=exact.theta), dict(leaves, theta=cute.theta)
    )


# ---------------------------------------------------------------------------
# E10: CuTe Z mode is unchanged (no bloch_measure keyword)
# ---------------------------------------------------------------------------


def _e10_parametrize(test):
    """fast_measure x preacts_trainable, ids in that order."""
    test = pytest.mark.parametrize(
        "fast_measure", [True, False], ids=["fast-measure", "square-measure"]
    )(test)
    return pytest.mark.parametrize(
        "preacts_trainable", [False, True], ids=["fixed-preacts", "trainable-preacts"]
    )(test)


@_e10_parametrize
def test_e10_cute_solver_z_mode_output_parity(fast_measure, preacts_trainable):
    """E10 / A.3: without ``bloch_measure``, ``cute_exact_solver`` matches
    ``torch_exact_solver`` for ``theta`` ``(3, 5, 3, 2)``, eps < 1e-4."""
    ref, out, _ = _solver_pair(3, 2, preacts_trainable, fast_measure=fast_measure)
    _assert_shape(ref, (BATCH, 3, IN_DIM), "torch_exact_solver Z output")
    _assert_parity(ref, out, "Z output")


@_e10_parametrize
def test_e10_cute_solver_z_mode_gradient_parity(fast_measure, preacts_trainable):
    """E10 / A.3: Z-mode gradients w.r.t. ``x``, ``theta`` and the trainable
    preacts agree to eps < 1e-4."""
    ref, out, wrt = _solver_pair(3, 2, preacts_trainable, fast_measure=fast_measure)
    _assert_shape(ref, (BATCH, 3, IN_DIM), "torch_exact_solver Z output")
    _assert_shape(out, (BATCH, 3, IN_DIM), "cute_exact_solver Z output")
    _assert_grad_parity(ref, out, wrt, wrt)


# ---------------------------------------------------------------------------
# E11: CuTe inference path (eval route, no autograd)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group", GROUPS)
def test_e11_inference_forward_bloch_output_parity(group):
    """E11 / D.5 item 3: after ``eval()`` and under ``torch.no_grad()``,
    ``forward_bloch(x)`` of the E9 pair agrees to eps < 1e-4."""
    exact, cute = _layer_pair(group)
    exact.eval()
    cute.eval()
    x = _layer_x()
    with torch.no_grad():
        ref = exact.forward_bloch(x)
        out = cute.forward_bloch(x)
    _assert_shape(ref, (BATCH, _n_circuits(group), IN_DIM, 3), "exact forward_bloch")
    _assert_parity(ref, out, "forward_bloch(x) under eval/no_grad")


@pytest.mark.parametrize("observable_kind", list(OBSERVABLE_SHAPES))
@pytest.mark.parametrize("group", GROUPS)
def test_e11_inference_layer_observable_output_parity(group, observable_kind):
    """E11 / D.5 item 3: after ``eval()`` and under ``torch.no_grad()``,
    ``layer(x, observable)`` of the E9 pair agrees to eps < 1e-4."""
    exact, cute = _layer_pair(group)
    exact.eval()
    cute.eval()
    x = _layer_x()
    observable = _observable(observable_kind)
    with torch.no_grad():
        ref = exact(x, observable=observable)
        out = cute(x, observable=observable)
    _assert_shape(ref, (BATCH, OUT_DIM), "exact layer(x, observable)")
    _assert_parity(ref, out, "layer(x, observable) under eval/no_grad")


# ---------------------------------------------------------------------------
# Extra (not in spec E): cheap contract checks of C.2-C.4 and D.5
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preacts_trainable", [False, True], ids=["fixed-preacts", "trainable-preacts"]
)
def test_extra_pz_bloch_bindings_and_wrappers_contract(preacts_trainable):
    """Extra (not in spec E), C.4: binding and wrapper return shapes/dtypes.

    ``pz_bloch_forward`` returns ``(B, O, I, 3)``; ``pz_bloch_backward``
    returns ``[grad_x (B, I), grad_theta (theta's shape), grad_pw or None,
    grad_pb or None]`` in float32; the Python wrappers return the same.
    """
    from qkan.solver.cute.cute_ops import _get_ext

    ext = _get_ext()
    reps = 2
    x, theta, pw, pb = (
        t.detach() for t in _solver_inputs(OUT_DIM, reps, preacts_trainable)
    )
    grad_output = _randn(BATCH, OUT_DIM, IN_DIM, 3, seed=SEED_COTANGENT)

    out = ext.pz_bloch_forward(x, theta, pw, pb, preacts_trainable, False)
    _assert_shape(out, (BATCH, OUT_DIM, IN_DIM, 3), "pz_bloch_forward")
    assert out.dtype == torch.float32, f"pz_bloch_forward dtype {out.dtype}"

    grads = ext.pz_bloch_backward(x, theta, pw, pb, grad_output, preacts_trainable, 32)
    assert len(grads) == 4, f"pz_bloch_backward returned {len(grads)} items"
    expected = [(BATCH, IN_DIM), tuple(theta.shape)]
    expected += [tuple(pw.shape), tuple(pb.shape)] if preacts_trainable else [None] * 2
    got = [None if g is None else tuple(g.shape) for g in grads]
    assert got == expected, f"pz_bloch_backward shapes {got}, expected {expected}"
    for g in grads:
        assert g is None or g.dtype == torch.float32, f"gradient dtype {g.dtype}"

    from qkan.solver.cute.cute_ops import cute_pz_bloch_backward, cute_pz_bloch_forward

    wrapped = cute_pz_bloch_forward(x, theta, pw, pb, preacts_trainable)
    assert torch.equal(wrapped, out), "cute_pz_bloch_forward != pz_bloch_forward"
    wrapped_grads = cute_pz_bloch_backward(
        x, theta, pw, pb, grad_output, preacts_trainable
    )
    assert isinstance(wrapped_grads, tuple) and len(wrapped_grads) == 4
    got = [None if g is None else tuple(g.shape) for g in wrapped_grads]
    assert got == expected, f"cute_pz_bloch_backward shapes {got}"


def test_extra_cute_bloch_measure_ignores_fast_measure():
    """Extra (not in spec E), C.3: with ``bloch_measure=True``, ``fast_measure``
    has no effect on ``cute_exact_solver``."""
    reps = 2
    x, theta, pw, pb = (t.detach() for t in _solver_inputs(OUT_DIM, reps, False))
    common = dict(
        ansatz="pz_encoding", out_dim=OUT_DIM, dtype=torch.float32, bloch_measure=True
    )
    with torch.no_grad():
        fast = cute_exact_solver(x, theta, pw, pb, reps, fast_measure=True, **common)
        square = cute_exact_solver(x, theta, pw, pb, reps, fast_measure=False, **common)
    _assert_shape(fast, (BATCH, OUT_DIM, IN_DIM, 3), "cute_exact_solver(bloch)")
    assert torch.equal(fast, square), "fast_measure changed the Bloch output"


@_e8_parametrize
def test_extra_cute_solver_bloch_no_grad_branch_parity(
    theta_rows, preacts_trainable, reps
):
    """Extra (not in spec E), D.5 item 3: the ``needs_grad=False`` Bloch
    branch of ``cute_exact_solver`` (taken under ``torch.no_grad()``) on the
    E8 inputs, including trainable preacts, which E11's layers never use."""
    with torch.no_grad():
        ref, out, _ = _solver_pair(
            theta_rows, reps, preacts_trainable, bloch_measure=True
        )
    _assert_shape(
        out,
        (BATCH, theta_rows, IN_DIM, 3),
        "cute_exact_solver(bloch_measure=True) under no_grad",
    )
    _assert_parity(ref, out, "Bloch output (needs_grad=False branch)")


@pytest.mark.parametrize("group", GROUPS)
def test_extra_cute_bloch_follows_p_dtype(group):
    """Extra (not in spec E), C.2 / B.4: with ``p_dtype=torch.bfloat16``,
    ``forward_bloch(x)`` returns ``p_dtype`` although the kernel writes
    float32, and ``layer(x, observable)`` keeps the dtype of the Z-mode
    output (the output dtype rule of ``forward`` is unchanged).

    Only shapes and dtypes are checked; the spec E parity fixture is float32.
    """
    layer = QKANLayer(
        IN_DIM,
        OUT_DIM,
        reps=2,
        group=group,
        device=DEVICE,
        solver="cute",
        ansatz="pz_encoding",
        ba_trainable=False,
        p_dtype=torch.bfloat16,
        seed=SEED_LAYER,
    )
    x = _randn(BATCH, IN_DIM, seed=SEED_X).to(torch.bfloat16).requires_grad_()
    r = layer.forward_bloch(x)
    _assert_shape(r, (BATCH, _n_circuits(group), IN_DIM, 3), "cute forward_bloch(x)")
    assert r.dtype == torch.bfloat16, f"forward_bloch dtype {r.dtype}, expected p_dtype"
    y = layer(x, observable=_observable("shared"))
    _assert_shape(y, (BATCH, OUT_DIM), "cute layer(x, observable)")
    z_mode = layer(x)
    assert y.dtype == z_mode.dtype == torch.bfloat16, (
        f"layer(x, observable) dtype {y.dtype}; Z mode gives {z_mode.dtype}"
    )


# ---------------------------------------------------------------------------
# Extra (not in spec E): the kernels themselves, every BLOCK_B instance, the
# bf16 / fp8 paths, the dispatch of D.5 item 3, and autocast
# ---------------------------------------------------------------------------


def _exact_bloch(x, theta, pw, pb, reps, preacts_trainable):
    return torch_exact_solver(
        x,
        theta,
        pw,
        pb,
        reps,
        ansatz="pz_encoding",
        out_dim=theta.shape[0],
        preacts_trainable=preacts_trainable,
        dtype=torch.complex64,
        bloch_measure=True,
    )


@pytest.mark.parametrize(
    "preacts_trainable", [False, True], ids=["fixed-preacts", "trainable-preacts"]
)
@pytest.mark.parametrize("theta_rows", [1, 3], ids=["theta-rows1", "theta-rows3"])
def test_extra_pz_bloch_bindings_value_parity(theta_rows, preacts_trainable):
    """Extra (not in spec E), C.4 / D.3: the raw bindings, not only
    cute_exact_solver, agree with torch_exact_solver's Bloch output and its
    autograd gradients (eps < 1e-4), so a solver that stopped calling the
    kernels could not hide a kernel bug."""
    from qkan.solver.cute.cute_ops import _get_ext

    ext = _get_ext()
    reps = 2
    x, theta, pw, pb = _solver_inputs(theta_rows, reps, preacts_trainable)
    pw.requires_grad_(True)
    pb.requires_grad_(True)
    ref = _exact_bloch(x, theta, pw, pb, reps, preacts_trainable)
    cotangent = _randn(*ref.shape, seed=SEED_COTANGENT)
    ref_grads = torch.autograd.grad(
        (ref * cotangent).sum(), [x, theta, pw, pb], allow_unused=True
    )
    out = ext.pz_bloch_forward(
        x.detach(), theta.detach(), pw.detach(), pb.detach(), preacts_trainable, False
    )
    _assert_parity(ref, out, "pz_bloch_forward")
    grads = ext.pz_bloch_backward(
        x.detach(),
        theta.detach(),
        pw.detach(),
        pb.detach(),
        cotangent,
        preacts_trainable,
        32,
    )
    names = ["x", "theta", "preacts_weight", "preacts_bias"]
    for name, expected, actual in zip(names, ref_grads, grads):
        if name.startswith("preacts") and not preacts_trainable:
            assert actual is None, f"pz_bloch_backward returned a {name} gradient"
            continue
        _assert_parity(expected, actual, f"pz_bloch_backward gradient w.r.t. {name}")


def test_extra_cute_solver_dispatches_to_bloch_kernels(monkeypatch):
    """Extra (not in spec E), D.5 item 3: with grad enabled the Bloch branch
    goes through _CuTePzBlochFunction; under no_grad it calls
    cute_pz_bloch_forward directly; neither falls back to the exact solver."""
    from qkan.solver.cute import cute as cute_module

    def no_fallback(*args, **kwargs):
        raise AssertionError("the cute Bloch path fell back to torch_exact_solver")

    monkeypatch.setattr(cute_module, "torch_exact_solver", no_fallback)
    reps = 2
    x, theta, pw, pb = _solver_inputs(OUT_DIM, reps, False)
    common = dict(
        ansatz="pz_encoding", out_dim=OUT_DIM, dtype=torch.float32, bloch_measure=True
    )
    out = cute_exact_solver(x, theta, pw, pb, reps, **common)
    assert "CuTePzBlochFunction" in type(out.grad_fn).__name__, out.grad_fn

    calls = []
    direct = cute_module.cute_pz_bloch_forward

    def spy(*args, **kwargs):
        calls.append(1)
        return direct(*args, **kwargs)

    def no_function(*args, **kwargs):
        raise AssertionError("needs_grad=False must not go through the Function")

    monkeypatch.setattr(cute_module, "cute_pz_bloch_forward", spy)
    monkeypatch.setattr(cute_module._CuTePzBlochFunction, "apply", no_function)
    with torch.no_grad():
        out_nograd = cute_exact_solver(x, theta, pw, pb, reps, **common)
    assert calls == [1]
    assert torch.equal(out_nograd, out.detach())


@pytest.mark.parametrize("batch", [37, 100, 300, 600], ids=lambda b: f"B{b}")
def test_extra_cute_bloch_parity_across_block_sizes(batch):
    """Extra (not in spec E), D.3: the launchers pick BLOCK_B in {32, 64, 128,
    256} from the batch size (37, 100, 300, 600 hit each one); the Bloch
    output and its gradients agree with the exact solver for every instance
    (eps < 1e-4), with trainable preacts and G = O."""
    reps = 2
    gen = torch.Generator().manual_seed(SEED_X + batch)
    x = torch.randn(batch, IN_DIM, generator=gen).to(DEVICE).requires_grad_()
    theta = torch.randn(OUT_DIM, IN_DIM, reps + 1, 2, generator=gen).to(DEVICE)
    pw = torch.randn(OUT_DIM, IN_DIM, reps, generator=gen).to(DEVICE)
    pb = torch.randn(OUT_DIM, IN_DIM, reps, generator=gen).to(DEVICE)
    for t in (theta, pw, pb):
        t.requires_grad_()
    common = dict(
        ansatz="pz_encoding",
        out_dim=OUT_DIM,
        preacts_trainable=True,
        bloch_measure=True,
    )
    ref = torch_exact_solver(x, theta, pw, pb, reps, dtype=torch.complex64, **common)
    out = cute_exact_solver(x, theta, pw, pb, reps, dtype=torch.float32, **common)
    _assert_shape(out, (batch, OUT_DIM, IN_DIM, 3), "cute_exact_solver(bloch)")
    _assert_parity(ref, out, f"Bloch output, batch {batch}")
    wrt = {"x": x, "theta": theta, "preacts_weight": pw, "preacts_bias": pb}
    _assert_grad_parity(ref, out, wrt, wrt)


LOW_PRECISION_EPS = {torch.bfloat16: 0.2, torch.float8_e4m3fn: 0.5}


@pytest.mark.parametrize("c_dtype", list(LOW_PRECISION_EPS), ids=["bf16", "fp8"])
@pytest.mark.parametrize(
    "preacts_trainable", [False, True], ids=["fixed-preacts", "trainable-preacts"]
)
def test_extra_cute_bloch_low_precision_smoke(preacts_trainable, c_dtype):
    """Extra (not in spec E), D.3 / D.5: the bf16-I/O forward and the bf16 /
    fp8 state-checkpoint backward run the Bloch branch too. A.4 sets no
    accuracy target for them, so the bound is loose (correct code stays below
    about 0.04 for bf16 and 0.13 for fp8); it still catches a dispatch that
    loses the Bloch flag, which gives eps of order 1."""
    reps = 2
    x, theta, pw, pb = _solver_inputs(OUT_DIM, reps, preacts_trainable)
    common = dict(
        ansatz="pz_encoding",
        out_dim=OUT_DIM,
        preacts_trainable=preacts_trainable,
        bloch_measure=True,
    )
    ref = torch_exact_solver(x, theta, pw, pb, reps, dtype=torch.complex64, **common)
    out = cute_exact_solver(x, theta, pw, pb, reps, dtype=c_dtype, **common)
    assert out.dtype == torch.bfloat16, f"low-precision Bloch output dtype {out.dtype}"
    bound = LOW_PRECISION_EPS[c_dtype]
    err = _eps(ref, out)
    assert err < bound, f"Bloch output eps {err:.3e} (limit {bound})"
    cotangent = _randn(*ref.shape, seed=SEED_COTANGENT)
    wrt = [x, theta] + ([pw, pb] if preacts_trainable else [])
    ref_grads = torch.autograd.grad((ref * cotangent).sum(), wrt)
    out_grads = torch.autograd.grad((out.float() * cotangent).sum(), wrt)
    for expected, actual in zip(ref_grads, out_grads):
        assert actual.dtype == torch.float32, f"gradient dtype {actual.dtype}"
        err = _eps(expected, actual)
        assert err < bound, f"gradient eps {err:.3e} (limit {bound})"


@pytest.mark.parametrize(
    "fused", [False, True], ids=["eager-epilogue", "fused-epilogue"]
)
@pytest.mark.parametrize("solver", ["exact", "cute"])
def test_extra_autocast_readout_stays_in_p_dtype(solver, fused):
    """Extra (not in spec E), B.4: under torch.autocast the observable
    readout stays in p_dtype, like the Z-mode postacts, so the output dtype
    matches Z mode and the backward through the (opt-in) fused epilogue works
    in train mode."""
    from qkan.qkan_epilogue import _TRITON_AVAILABLE

    if fused and not _TRITON_AVAILABLE:
        pytest.skip("the fused epilogue needs Triton")
    layer = QKANLayer(
        IN_DIM,
        OUT_DIM,
        reps=2,
        group=-1,
        device=DEVICE,
        solver=solver,
        ansatz="pz_encoding",
        ba_trainable=True,
        p_dtype=torch.float32,
        seed=SEED_LAYER,
    )
    layer.set_fused_epilogue(fused)
    x = _layer_x()
    observable = _observable("per-sample")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        postacts = layer._observable_postacts(x, observable)
        y = layer(x, observable=observable)
        y_z = layer(x)
    assert postacts.dtype == torch.float32, f"readout dtype {postacts.dtype}"
    assert y.dtype == y_z.dtype, f"observable mode {y.dtype}, Z mode {y_z.dtype}"
    y.float().sum().backward()
    for name, grad in (
        ("x", x.grad),
        ("observable", observable.grad),
        ("theta", layer.theta.grad),
    ):
        assert grad is not None and torch.isfinite(grad).all(), name
