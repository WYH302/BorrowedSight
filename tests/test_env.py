"""Phase 0 environment lock: CUDA 3090, bf16, ResidualSimVQ backward."""

import torch
from vector_quantize_pytorch import ResidualSimVQ


def _quantized_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "quantized"):
        return out.quantized
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise AssertionError(f"unexpected ResidualSimVQ output: {type(out)!r}")


def test_cuda_3090_and_bf16():
    assert torch.cuda.is_available()
    assert torch.version.cuda
    assert "+cpu" not in torch.__version__
    name = torch.cuda.get_device_name(0)
    assert name == "NVIDIA GeForce RTX 3090", name
    assert torch.cuda.is_bf16_supported()

    ones = torch.ones(4, device="cuda", dtype=torch.bfloat16)
    doubled = ones + ones
    assert doubled.dtype == torch.bfloat16
    assert doubled.device.type == "cuda"
    assert torch.all(doubled == 2)


def test_residual_simvq_rotation_trick_forward_backward():
    torch.manual_seed(0)
    model = ResidualSimVQ(
        dim=16,
        num_quantizers=2,
        codebook_size=32,
        rotation_trick=True,
    ).to("cuda")
    model.train()

    x = torch.randn(2, 4, 16, device="cuda", requires_grad=True)
    quantized = _quantized_tensor(model(x))
    assert quantized.shape == x.shape
    assert quantized.is_cuda

    loss = quantized.float().square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
