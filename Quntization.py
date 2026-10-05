import numpy as np

def get_quantization_params(x, num_bits=8):
    """Calculates the scale and zero-point based on the min/max values of the tensor."""
    qmin = 0
    qmax = (2 ** num_bits) - 1  # For 8-bit, this is 255
    
    x_min, x_max = x.min(), x.max()
    
    # Prevent division by zero if all elements are the same
    if x_min == x_max:
        return 1.0, 0
    
    # Calculate scale
    scale = (x_max - x_min) / (qmax - qmin)
    
    # Calculate zero-point and clip it to stay within valid [0, 255] boundaries
    zero_point = qmin - (x_min / scale)
    zero_point = np.clip(np.round(zero_point), qmin, qmax).astype(np.int32)
    
    return float(scale), int(zero_point)

def quantize(x, scale, zero_point, num_bits=8):
    """Converts FP32 numbers into INT8 discrete values."""
    qmin = 0
    qmax = (2 ** num_bits) - 1
    
    # Scale, shift, round, and clamp to the 8-bit range
    q = np.round(x / scale) + zero_point
    q = np.clip(q, qmin, qmax).astype(np.uint8)
    return q

def dequantize(q, scale, zero_point):
    """Reconstructs FP32 values back from INT8 data."""
    return (q.astype(np.float32) - zero_point) * scale

# --- Demonstration ---
if __name__ == "__main__":
    # 1. Create a simulated layer weight tensor (FP32)
    np.random.seed(42)
    original_weights = np.random.uniform(-5.0, 12.0, size=(5,))
    print(f"Original FP32 Weights: \n{original_weights}\n")
    
    # 2. Get scale and zero point
    scale, zero_point = get_quantization_params(original_weights, num_bits=8)
    print(f"Calculated Scale:      {scale:.5f}")
    print(f"Calculated Zero-Point: {zero_point}\n")
    
    # 3. Quantize to INT8
    quantized_int8 = quantize(original_weights, scale, zero_point)
    print(f"Quantized INT8 Array:  \n{quantized_int8}  (Data type: {quantized_int8.dtype})\n")
    
    # 4. Dequantize back to FP32
    reconstructed_weights = dequantize(quantized_int8, scale, zero_point)
    print(f"Dequantized Weights:   \n{reconstructed_weights}\n")
    
    # 5. Measure the Quantization Error (Loss of precision)
    errors = np.abs(original_weights - reconstructed_weights)
    print(f"Quantization Error per element: \n{errors}")
    print(f"Mean Absolute Error:            {errors.mean():.5f}")
