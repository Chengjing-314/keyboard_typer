# This is bascially torch version of dm_control reward util
import torch


def _sigmoids_torch(d, value_at_margin, sigmoid_type):
    """
    Applies a sigmoid-like decay to the normalized distance tensor `d`.

    Args:
      d: torch.Tensor containing the normalized distances (d = distance/margin).
      value_at_margin: A float between 0 and 1 representing the desired value when d == 1.
      sigmoid_type: String specifying the type of sigmoid. Valid types are:
         'gaussian', 'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.

    Returns:
      torch.Tensor with the sigmoid function applied elementwise.
    """
    if sigmoid_type == "gaussian":
        # Choose a scaling factor c such that f(1) = exp(-c^2) equals value_at_margin.
        # Handle potential issues when value_at_margin <= 0.
        if value_at_margin <= 0:
            c = torch.tensor(1e6, dtype=d.dtype, device=d.device)
        else:
            c = torch.sqrt(
                -torch.log(torch.tensor(value_at_margin, dtype=d.dtype, device=d.device))
            )
        return torch.exp(-((d * c) ** 2))

    elif sigmoid_type == "linear":
        # Linearly decrease from 1 at d=0 to value_at_margin at d=1,
        # and clamp to 0 for larger distances.
        return torch.clamp(1 - (1 - value_at_margin) * d, min=0.0)

    elif sigmoid_type == "hyperbolic":
        # Use a hyperbolic decay: f(d) = 1 / (1 + a * d), where a is chosen so that f(1)=value_at_margin.
        a = 1 / value_at_margin - 1 if value_at_margin > 0 else 1e6
        return 1 / (1 + a * d)

    elif sigmoid_type == "long_tail":
        # Use a long-tail decay: f(d) = 1 / (1 + (d/c)^2)
        # Solve 1/(1 + (1/c)^2) = value_at_margin to get c.
        if value_at_margin <= 0:
            c = torch.tensor(1e6, dtype=d.dtype, device=d.device)
        else:
            c = 1 / torch.sqrt(
                torch.tensor(1 / value_at_margin - 1, dtype=d.dtype, device=d.device)
            )
        return 1 / (1 + (d / c) ** 2)

    elif sigmoid_type == "cosine":
        # For d <= 1, use a cosine decay that gives 1 at d=0 and value_at_margin at d=1.
        # For d > 1, continue with an exponential decay.
        f = torch.where(
            d <= 1,
            (1 - value_at_margin) * torch.cos(d * torch.pi / 2) + value_at_margin,
            value_at_margin * torch.exp(-(d - 1)),
        )
        return f

    elif sigmoid_type == "tanh_squared":
        # Use a tanh-squared decay with a fixed steepness factor (here chosen as 5).
        factor = 5.0
        return (1 - value_at_margin) * torch.tanh(factor * (1 - d)) ** 2 + value_at_margin

    else:
        raise ValueError(f"Unknown sigmoid type: {sigmoid_type}")


def tolerance_torch(x, bounds=(0.0, 0.0), margin=0.0, sigmoid="gaussian", value_at_margin=0.5):
    """
    Returns 1 when `x` is inside the bounds, and a value between 0 and 1 when `x` is out-of-bounds.
    Works with PyTorch tensors and handles inputs of any shape (e.g. n x 1 tensors).

    Args:
      x: A scalar or torch.Tensor.
      bounds: Tuple (lower, upper) specifying inclusive bounds.
      margin: Float controlling how steeply the output decreases as x moves out-of-bounds.
              * If margin == 0, output is 0 for x outside the bounds.
              * If margin > 0, output decreases sigmoidally with increasing distance from the nearest bound.
      sigmoid: String specifying the type of sigmoid function. Options include:
               'gaussian', 'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
      value_at_margin: Float in [0, 1] that is the output value when the distance from x to the nearest bound equals margin.

    Returns:
      torch.Tensor with the same shape as x containing values between 0.0 and 1.0.

    Raises:
      ValueError: If lower bound > upper bound or if margin is negative.
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")
    if margin < 0:
        raise ValueError("`margin` must be non-negative.")

    # Convert x to a tensor if it isn't already.
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    # Compute a boolean tensor indicating which elements are within bounds.
    in_bounds = (x >= lower) & (x <= upper)

    if margin == 0:
        # Outside bounds get a value of 0.
        value = torch.where(
            in_bounds,
            torch.tensor(1.0, dtype=x.dtype, device=x.device),
            torch.tensor(0.0, dtype=x.dtype, device=x.device),
        )
    else:
        # Compute the normalized distance from the nearest bound.
        # For values below lower, use (lower - x); for values above upper, use (x - upper); else 0.
        d = torch.where(
            x < lower,
            lower - x,
            torch.where(x > upper, x - upper, torch.tensor(0.0, dtype=x.dtype, device=x.device)),
        )
        d = d / margin

        # Compute the decay using the chosen sigmoid.
        sig = _sigmoids_torch(d, value_at_margin, sigmoid)
        value = torch.where(in_bounds, torch.tensor(1.0, dtype=x.dtype, device=x.device), sig)
    return value


# Example usage:
if __name__ == "__main__":
    # Create a sample tensor (n x 1)
    x = torch.linspace(-2, 4, steps=30).unsqueeze(1)  # shape: [30, 1]
    result = tolerance_torch(
        x, bounds=(0.0, 2.0), margin=1.0, sigmoid="gaussian", value_at_margin=0.5
    )
    print(result)
