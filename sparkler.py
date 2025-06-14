import torch


def sparkler(values, log=True):
    """Sparkler quantization

    Args:
        values (tensor): torch tensor of values to be quantized

    Returns:
        tensor: quantized values
    """
    values = values.half().cpu()
    bits = values.view(torch.uint16).to(torch.int32)
    bits = (bits >> 1) << 1
    exponent_bits = (bits >> 10) & 0x1F  # 5 bits
    top3_mantissa_bits = (bits >> 7) & 0x7  # top 3 of 10 mantissa bits
    sign_bit = bits & 0x8000  # bit 15 (1 bit sign)

    unsparklers = torch.zeros_like(values)

    sparkler_indices = (exponent_bits > 11) & (exponent_bits < 20)
    unsparklers[~sparkler_indices] = values[~sparkler_indices]

    # Extract only values at sparkler_indices
    sparkler_exponent_bits = exponent_bits[sparkler_indices]
    sparkler_top3_mantissa_bits = top3_mantissa_bits[sparkler_indices]
    sparkler_sign_bit = sign_bit[sparkler_indices]

    # Reconstruct only the relevant elements
    mantissa_reconstructed = sparkler_top3_mantissa_bits << 7
    exponent_reconstructed = sparkler_exponent_bits << 10
    reconstructed_bits = (
        sparkler_sign_bit | exponent_reconstructed | mantissa_reconstructed
    )
    reconstructed_tensor = reconstructed_bits.to(torch.uint16).view(torch.float16)

    # Assign back to unsparklers
    unsparklers[sparkler_indices] = reconstructed_tensor

    # Logging compression
    if log:
        sparkler_count = sparkler_indices.sum().item()
        total_el = torch.numel(values)
        print(sparkler_count, ", ", total_el)

    return unsparklers


# def sparkler_count(tensor_fp32):
#     tensor_fp16 = tensor_fp32.half()
#     bits = tensor_fp16.view(torch.uint16).to(torch.int32)
#     exponent_bits = (bits >> 10) & 0x1F  # 5 bits
#     top3_mantissa_bits = (bits >> 7) & 0x7  # top 3 of 10 mantissa bits
#     sign_bit = bits & 0x8000  # bit 15 (1 bit sign)

#     unsparklers = torch.zeros_like(tensor_fp16)

#     sparkler_indices = (exponent_bits > 11) & (exponent_bits < 20)
#     sparkler_count = sparkler_indices.sum().item()
#     return sparkler_count
