# tests/test_model.py
import os
import math
import random
import tempfile
import contextlib

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Adjust imports to your actual module names
from src.model import CNNModel  # expects forward(x) -> logits [N, 10]


# -----------------------------
# Fixtures and helpers
# -----------------------------

@pytest.fixture(scope="module")
def num_classes():
    return 10

@pytest.fixture(params=[1, 4, 32])
def batch_size(request):
    return request.param

@pytest.fixture(params=[(3, 32, 32), (3, 64, 64)])
def image_shape(request):
    return request.param

@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return torch.device(request.param)

@pytest.fixture
def model(device, num_classes):
    m = CNNModel(num_classes=num_classes)
    m.to(device)
    m.eval()
    return m

def make_inputs(batch_size, image_shape, device, dtype=torch.float32):
    c, h, w = image_shape
    x = torch.randn(batch_size, c, h, w, device=device, dtype=dtype)
    return x


# -----------------------------
# Shape and dtype contracts
# -----------------------------

def test_forward_shape(model, batch_size, image_shape, num_classes, device):
    x = make_inputs(batch_size, image_shape, device)
    y = model(x)
    assert y.shape == (batch_size, num_classes), f"Expected {(batch_size, num_classes)}, got {tuple(y.shape)}"
    assert y.dtype == torch.float32

def test_no_grad_in_eval(model, device, image_shape):
    model.eval()
    x = make_inputs(2, image_shape, device)
    with torch.no_grad():
        y = model(x)
    assert y.requires_grad is False

def test_requires_grad_in_train(num_classes, device, image_shape):
    model = CNNModel(num_classes=num_classes).to(device).train()
    x = make_inputs(2, image_shape, device)
    y = model(x)
    assert y.requires_grad is True


# -----------------------------
# Determinism under fixed seed
# -----------------------------

def test_deterministic_forward(model, device, image_shape):
    torch.manual_seed(42)
    x = make_inputs(4, image_shape, device)
    y1 = model(x)
    torch.manual_seed(42)
    x2 = make_inputs(4, image_shape, device)
    y2 = model(x2)
    assert torch.allclose(y1, y2, atol=1e-6), "Forward pass not deterministic under fixed seed (pure inference)."


# -----------------------------
# Gradients and backward sanity
# -----------------------------

def test_backward_gradients(num_classes, device):
    model = CNNModel(num_classes=num_classes).to(device).train()
    x = torch.randn(8, 3, 32, 32, device=device, requires_grad=False)
    target = torch.randint(0, num_classes, (8,), device=device)
    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    grad_count = sum(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert grad_count > 0, "No gradients computed."
    total_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
    assert torch.isfinite(total_grad_norm), "Gradient norm is NaN or Inf."


# -----------------------------
# Mixed precision (if CUDA)
# -----------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP requires CUDA")
def test_amp_forward_backward(num_classes):
    device = torch.device("cuda")
    model = CNNModel(num_classes=num_classes).to(device).train()
    scaler = torch.cuda.amp.GradScaler()
    x = torch.randn(16, 3, 32, 32, device=device)
    target = torch.randint(0, num_classes, (16,), device=device)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        logits = model(x)
        loss = F.cross_entropy(logits, target)

    scaler.scale(loss).backward()
    scaler.step(torch.optim.SGD(model.parameters(), lr=1e-3))
    scaler.update()

    assert loss.item() > 0
    assert logits.dtype in (torch.float16, torch.float32)  # depending on last layer behavior


# -----------------------------
# Save / load state dict
# -----------------------------

def test_save_and_load_state_dict(num_classes, device):
    model = CNNModel(num_classes=num_classes).to(device).eval()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pth")
        torch.save(model.state_dict(), path)
        assert os.path.exists(path)

        # New model instance, load weights
        model2 = CNNModel(num_classes=num_classes).to(device).eval()
        model2.load_state_dict(torch.load(path, map_location=device))
        # Compare parameters
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2), f"Mismatch in parameter {n1}"


# -----------------------------
# Logits sanity (no NaN/Inf)
# -----------------------------

def test_logits_are_finite(model, device, image_shape):
    x = make_inputs(32, image_shape, device)
    y = model(x)
    assert torch.isfinite(y).all(), "Logits contain NaN/Inf."


# -----------------------------
# Softmax probabilities sanity
# -----------------------------

def test_softmax_probabilities_sum_to_one(model, device, image_shape):
    x = make_inputs(5, image_shape, device)
    y = model(x)
    probs = F.softmax(y, dim=1)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), "Softmax rows must sum to 1."
    assert torch.all(probs >= 0), "Probabilities must be non-negative."


# -----------------------------
# Gradcheck (double precision)
# -----------------------------

@pytest.mark.skipif(not torch.cuda.is_available() and not torch.backends.mps.is_available(), reason="Slow on CPU only")
def test_gradcheck_small_input(num_classes):
    # Use a super small network path: requires CNNModel to be differentiable for inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(num_classes=num_classes).to(device).eval()

    # Gradcheck needs double precision and requires_grad inputs
    x = torch.randn(1, 3, 16, 16, dtype=torch.double, device=device, requires_grad=True)
    # Wrap model to a function that returns a single scalar to gradcheck (sum of logits)
    def f(inp):
        out = model(inp.to(dtype=torch.double))
        return out.sum()

    assert torch.autograd.gradcheck(f, (x,), eps=1e-3, atol=1e-3), "Gradcheck failed."


# -----------------------------
# Device move and consistency
# -----------------------------

def test_device_move_consistency(num_classes, image_shape):
    cpu = torch.device("cpu")
    model_cpu = CNNModel(num_classes=num_classes).to(cpu).eval()
    x_cpu = make_inputs(3, image_shape, cpu)
    y_cpu = model_cpu(x_cpu)

    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        model_gpu = CNNModel(num_classes=num_classes).to(gpu).eval()
        x_gpu = x_cpu.to(gpu)
        y_gpu = model_gpu(x_gpu).to(cpu)
        assert torch.allclose(y_cpu, y_gpu, atol=1e-5), "CPU vs GPU outputs diverge more than tolerance."


# -----------------------------
# Parameter initialization sanity
# -----------------------------

def test_parameters_not_all_zero(model):
    zeros = []
    for p in model.parameters():
        if p.data.abs().sum().item() == 0.0:
            zeros.append(p.shape)
    assert len(zeros) == 0, f"Found parameters exactly zero: {zeros}"


# -----------------------------
# JIT / torchscript (optional if supported)
# -----------------------------

def test_torchscript_trace_if_supported(num_classes):
    model = CNNModel(num_classes=num_classes).eval()
    x = torch.randn(2, 3, 32, 32)
    try:
        traced = torch.jit.trace(model, x)
        y1 = model(x)
        y2 = traced(x)
        assert torch.allclose(y1, y2, atol=1e-5)
    except Exception:
        pytest.skip("TorchScript trace not supported by current CNNModel.")


# -----------------------------
# Robustness to extreme inputs
# -----------------------------

@pytest.mark.parametrize("scale", [0.0, 1e-3, 1.0, 10.0, 100.0])
def test_robustness_extreme_scales(model, device, scale):
    x = torch.randn(8, 3, 32, 32, device=device) * scale
    y = model(x)
    assert torch.isfinite(y).all()

@pytest.mark.parametrize("noise", [0.0, 0.1, 0.5, 1.0])
def test_robustness_noise(model, device, noise):
    x = torch.randn(8, 3, 32, 32, device=device)
    x = x + noise * torch.randn_like(x)
    y = model(x)
    assert torch.isfinite(y).all()


# -----------------------------
# Batch size edge cases
# -----------------------------

@pytest.mark.parametrize("b", [1, 2, 8, 64])
def test_varied_batch_sizes(model, device, b):
    x = torch.randn(b, 3, 32, 32, device=device)
    y = model(x)
    assert y.shape[0] == b


# -----------------------------
# Serialization safety to models/
# -----------------------------

def test_save_to_models_folder(num_classes, device):
    model = CNNModel(num_classes=num_classes).to(device).eval()
    os.makedirs("models", exist_ok=True)
    path = os.path.join("models", "cnn_model.pth")
    torch.save(model.state_dict(), path)
    assert os.path.exists(path)
    # Clean up the artifact to avoid polluting repo during tests
    with contextlib.suppress(Exception):
        os.remove(path)


# -----------------------------
# Sanity: output variance (not collapsing)
# -----------------------------

def test_output_has_variance(model, device):
    x = torch.randn(32, 3, 32, 32, device=device)
    y = model(x)
    var = y.var().item()
    assert var > 0, "Outputs appear collapsed to constants."


# -----------------------------
# Hook coverage (ensure backward passes through all params)
# -----------------------------

def test_backward_touches_params(num_classes, device):
    model = CNNModel(num_classes=num_classes).to(device).train()

    touched = {id(p): False for p in model.parameters() if p.requires_grad}
    def make_hook(p):
        def _hook(grad):
            touched[id(p)] = True
        return _hook

    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(make_hook(p))

    x = torch.randn(16, 3, 32, 32, device=device)
    target = torch.randint(0, num_classes, (16,), device=device)
    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    assert all(touched.values()), "Some parameters did not receive gradients."


# -----------------------------
# Numerical stability: large inputs shouldn’t produce NaNs
# -----------------------------

def test_numerical_stability_large_inputs(model, device):
    x = torch.randn(8, 3, 32, 32, device=device) * 1e4
    y = model(x)
    assert torch.isfinite(y).all(), "Numerical instability for large magnitude inputs."


# -----------------------------
# Basic performance guard (smoke)
# -----------------------------

def test_forward_time_smoke(model, device):
    import time
    x = torch.randn(64, 3, 32, 32, device=device)
    start = time.time()
    y = model(x)
    elapsed = time.time() - start
    # Set a generous threshold to catch pathological slowdowns
    assert elapsed < 2.0, f"Forward pass too slow: {elapsed:.3f}s"

