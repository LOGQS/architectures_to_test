# tct_test.py 

import torch
import torch.nn as nn
import time
from tct import ThousandColumns, RMSNorm, SwiGLU, ColumnCore
import os
import tempfile

def test_basic_shapes():
    """Test that all tensor shapes are correct throughout the forward pass."""
    print("Testing basic tensor shapes...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ThousandColumns(
        N=1024,
        d_state=64,
        d_input=256,
        d_vote=64,
        active_pct=0.25,
        device=device
    ).to(device)
    
    # Test different batch sizes and sequence lengths
    test_cases = [
        (1, 1, 256),    # Single sample, single timestep
        (2, 8, 256),    # Small batch, short sequence
        (4, 32, 256),   # Medium batch, medium sequence
        (1, 128, 256),  # Long sequence
    ]
    
    for B, T, d_input in test_cases:
        x = torch.randn(B, T, d_input, device=device, dtype=torch.float32)
        
        # Test forward pass
        with torch.no_grad():
            y = model(x)
        
        # Verify output shape
        assert y.shape == (B, T, d_input), f"Expected {(B, T, d_input)}, got {y.shape}"
        
        # Verify state shape
        assert model.state.shape == (B, model.N, model.d_state), \
            f"State shape mismatch: expected {(B, model.N, model.d_state)}, got {model.state.shape}"
    
    print("All shape tests passed")

def test_gradient_flow():
    """Test that gradients flow properly through the model."""
    print("Testing gradient flow...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ThousandColumns(
        N=512,
        d_state=32,
        d_input=128,
        d_vote=32,
        active_pct=0.2,
        device=device
    ).to(device)
    
    x = torch.randn(2, 4, 128, device=device, dtype=torch.float32, requires_grad=True)
    target = torch.randn(2, 4, 128, device=device, dtype=torch.float32)
    
    # Forward pass
    y = model(x)
    loss = nn.MSELoss()(y, target)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist for all parameters
    grad_count = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            grad_count += 1
            # Check gradient isn't all zeros
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
    
    print(f"Gradients computed for {grad_count}/{total_params} parameters")

def test_state_persistence():
    """Test that state is properly managed across calls."""
    print("Testing state persistence...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ThousandColumns(
        N=256,
        d_state=32,
        d_input=64,
        device=device
    ).to(device)
    
    B = 2
    x1 = torch.randn(B, 4, 64, device=device, dtype=torch.float32)
    x2 = torch.randn(B, 4, 64, device=device, dtype=torch.float32)
    
    # First forward pass
    model.eval()  # Set to eval mode to prevent dropout randomness
    with torch.no_grad():
        y1 = model(x1)
        state_after_first = model.state.clone()
    
    # Second forward pass (should use updated state)
    with torch.no_grad():
        y2 = model(x2)
        state_after_second = model.state.clone()
    
    # States should be different
    assert not torch.equal(state_after_first, state_after_second), \
        "State should change between forward passes"
    
    # Test state reset
    model.reset_state(B)
    assert model.state.shape == (B, model.N, model.d_state), \
        "State shape incorrect after reset"
    
    # State should be zeros after reset
    assert torch.allclose(model.state, torch.zeros_like(model.state)), \
        "State should be zeros after reset"
    
    print("State persistence tests passed")

def test_inference_speed():
    """Benchmark inference speed for different configurations."""
    print("Testing inference speed...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    configs = [
        {"N": 512, "d_state": 32, "active_pct": 0.25, "name": "Small"},
        {"N": 1024, "d_state": 64, "active_pct": 0.25, "name": "Medium"},
        {"N": 2048, "d_state": 64, "active_pct": 0.2, "name": "Large"},
    ]
    
    for config in configs:
        model = ThousandColumns(
            N=config["N"],
            d_state=config["d_state"],
            d_input=256,
            d_vote=64,
            active_pct=config["active_pct"],
            device=device
        ).to(device).eval()
        
        # Warm up
        x = torch.randn(1, 32, 256, device=device, dtype=torch.float32)
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)
        
        # Time multiple runs
        times = []
        with torch.no_grad():
            for _ in range(10):
                start = time.time()
                _ = model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        throughput = 32 / avg_time  # tokens per second
        
        print(f"  {config['name']}: {avg_time*1000:.2f}ms/step, {throughput:.1f} tokens/s")
    
    print("Speed benchmarks completed")

def test_memory_usage():
    """Test memory usage scaling."""
    print("Testing memory usage...")
    
    if not torch.cuda.is_available():
        print("  Skipping memory tests (CUDA not available)")
        return
    
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()
    
    model = ThousandColumns(
        N=1024,
        d_state=64,
        d_input=256,
        d_vote=64,
        device=device
    ).to(device)
    
    x = torch.randn(4, 64, 256, device=device, dtype=torch.float32)
    
    # Measure memory before forward pass
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        y = model(x)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    print(f"  Peak memory usage: {peak_memory:.1f} MB")
    
    # Test memory doesn't grow with multiple forward passes
    initial_memory = torch.cuda.memory_allocated()
    
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    
    final_memory = torch.cuda.memory_allocated()
    memory_growth = (final_memory - initial_memory) / 1024**2
    
    print(f"  Memory growth over 5 steps: {memory_growth:.1f} MB")
    print("Memory usage tests completed")

def test_attention_patterns():
    """Test that attention mechanisms work correctly."""
    print("Testing attention patterns...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ThousandColumns(
        N=256,
        d_state=32,
        d_input=64,
        d_vote=32,
        active_pct=0.3,
        device=device
    ).to(device)
    
    x = torch.randn(2, 8, 64, device=device, dtype=torch.float32)
    
    # Hook to capture attention weights
    attention_weights = []
    
    def attention_hook(module, input, output):
        # MultiheadAttention returns (output, attention_weights)
        if len(output) == 2 and output[1] is not None:
            attention_weights.append(output[1])
    
    # Register hook
    hook = model.bus_attn.register_forward_hook(attention_hook)
    
    with torch.no_grad():
        y = model(x)
    
    hook.remove()
    
    # Check that attention weights were captured
    assert len(attention_weights) > 0, "No attention weights captured"
    
    # Check attention weights sum to 1
    for attn in attention_weights:
        if attn is not None:
            attn_sum = attn.sum(dim=-1)
            assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-6), \
                "Attention weights don't sum to 1"
    
    print("Attention pattern tests passed")

def test_component_shapes():
    """Test individual component shapes."""
    print("Testing component shapes...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test RMSNorm
    norm = RMSNorm(64).to(device)
    x = torch.randn(2, 10, 64, device=device, dtype=torch.float32)
    y = norm(x)
    assert y.shape == x.shape, f"RMSNorm shape mismatch: {y.shape} != {x.shape}"
    
    # Test SwiGLU
    swiglu = SwiGLU(64, 128).to(device)
    x = torch.randn(2, 10, 64, device=device, dtype=torch.float32)
    y = swiglu(x)
    assert y.shape == (2, 10, 128), f"SwiGLU shape mismatch: {y.shape} != {(2, 10, 128)}"
    
    # Test ColumnCore
    core = ColumnCore(64, 32, 16).to(device)
    h = torch.randn(2, 10, 64, device=device, dtype=torch.float32)
    x = torch.randn(2, 10, 32, device=device, dtype=torch.float32)
    m = torch.randn(2, 10, 16, device=device, dtype=torch.float32)
    y = core(h, x, m)
    assert y.shape == h.shape, f"ColumnCore shape mismatch: {y.shape} != {h.shape}"
    
    print("Component shape tests passed")

def test_numerical_stability():
    """Test numerical stability with edge cases."""
    print("Testing numerical stability...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ThousandColumns(
        N=128,
        d_state=32,
        d_input=64,
        device=device
    ).to(device)
    
    # Test with very small inputs
    x_small = torch.randn(1, 4, 64, device=device, dtype=torch.float32) * 1e-6
    with torch.no_grad():
        y_small = model(x_small)
    assert torch.isfinite(y_small).all(), "Model produced non-finite values with small inputs"
    
    # Test with large inputs
    x_large = torch.randn(1, 4, 64, device=device, dtype=torch.float32) * 100
    with torch.no_grad():
        y_large = model(x_large)
    assert torch.isfinite(y_large).all(), "Model produced non-finite values with large inputs"
    
    # Test with zero inputs
    x_zero = torch.zeros(1, 4, 64, device=device, dtype=torch.float32)
    with torch.no_grad():
        y_zero = model(x_zero)
    assert torch.isfinite(y_zero).all(), "Model produced non-finite values with zero inputs"
    
    print("Numerical stability tests passed")

def test_state_dict_integrity():
    """Test that state buffer remains in state_dict after forward passes."""
    print("Testing state_dict integrity...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ThousandColumns(N=128, d_state=32, d_input=64, device=device).to(device)
    
    # Check initial state_dict
    initial_state_dict = model.state_dict()
    assert 'state' in initial_state_dict, "State buffer missing from initial state_dict"
    
    # Forward pass
    x = torch.randn(1, 4, 64, device=device, dtype=torch.float32)
    with torch.no_grad():
        _ = model(x)
    
    # Check state_dict after forward pass
    final_state_dict = model.state_dict()
    assert 'state' in final_state_dict, "State buffer missing from state_dict after forward pass"
    
    # Test save/load
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name
    try:
        torch.save(model.state_dict(), temp_path)
        loaded_state_dict = torch.load(temp_path)
        assert 'state' in loaded_state_dict, "State buffer missing from loaded state_dict"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    print("State dict integrity tests passed")

def test_gradient_coverage():
    """Test that all parameters receive gradients during training."""
    print("Testing gradient coverage...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ThousandColumns(N=128, d_state=32, d_input=64, device=device).to(device)
    model.train()  # Important: set to training mode
    
    x = torch.randn(2, 4, 64, device=device, dtype=torch.float32, requires_grad=True)
    target = torch.randn(2, 4, 64, device=device, dtype=torch.float32)
    
    # Forward pass
    y = model(x)
    loss = nn.MSELoss()(y, target)
    
    # Backward pass
    loss.backward()
    
    # Detailed gradient analysis by module
    gradient_info = {}
    params_with_grad = 0
    total_params = 0
    missing_grad_params = []
    
    for name, param in model.named_parameters():
        total_params += 1
        module_name = name.split('.')[0]  # Get top-level module name
        if module_name not in gradient_info:
            gradient_info[module_name] = {'total': 0, 'with_grad': 0, 'params': []}
        
        gradient_info[module_name]['total'] += 1
        gradient_info[module_name]['params'].append(name)
        
        if param.grad is not None and param.grad.abs().sum() > 0:
            params_with_grad += 1
            gradient_info[module_name]['with_grad'] += 1
        else:
            missing_grad_params.append(name)
    
    print(f"  Parameters with gradients: {params_with_grad}/{total_params}")
    print("  Gradient breakdown by module:")
    for module_name, info in gradient_info.items():
        status = "✓" if info['with_grad'] == info['total'] else "✗"
        print(f"    {status} {module_name}: {info['with_grad']}/{info['total']}")
        if info['with_grad'] < info['total']:
            missing_in_module = [p for p in info['params'] if p in missing_grad_params]
            print(f"      Missing: {missing_in_module}")
    
    if missing_grad_params:
        print(f"  Overall missing gradients: {missing_grad_params}")
    
    assert params_with_grad == total_params, f"Missing gradients for: {missing_grad_params}"
    
    print("Gradient coverage tests passed")

def test_large_scale_memory():
    """Test memory efficiency with large N."""
    print("Testing large scale memory efficiency...")
    
    if not torch.cuda.is_available():
        print("  Skipping large scale tests (CUDA not available)")
        return
    
    device = "cuda"
    
    # Test with large N
    model = ThousandColumns(N=4096, d_state=64, d_input=256, active_pct=0.1, device=device).to(device)
    model._optimize_active = True  # Enable optimizations
    
    x = torch.randn(2, 16, 256, device=device, dtype=torch.float32)
    
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        y = model(x)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    print(f"  Peak memory with N=4096: {peak_memory:.1f} MB")
    
    # Memory should be reasonable (< 1.5GB for this config)
    assert peak_memory < 1536, f"Memory usage too high: {peak_memory:.1f} MB"
    
    print("Large scale memory tests passed")

def test_dtype_consistency():
    """Test that all tensors maintain consistent dtype."""
    print("Testing dtype consistency...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ThousandColumns(N=128, d_state=32, d_input=64, device=device).to(device)
    
    # Check state buffer dtype
    assert model.state.dtype == torch.float32, f"State dtype should be float32, got {model.state.dtype}"
    
    # Test forward pass with mixed precision
    x = torch.randn(1, 4, 64, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        y = model(x)
    
    # Output should maintain input dtype
    assert y.dtype == torch.float32, f"Output dtype should be float32, got {y.dtype}"
    
    print("Dtype consistency tests passed")

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running TCT Architecture Tests")
    print("=" * 60)
    
    tests = [
        test_basic_shapes,
        test_gradient_flow,
        test_state_persistence,
        test_component_shapes,
        test_numerical_stability,
        test_attention_patterns,
        test_inference_speed,
        test_memory_usage,
        test_state_dict_integrity,
        test_gradient_coverage,
        test_large_scale_memory,
        test_dtype_consistency,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"{test.__name__} FAILED: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)