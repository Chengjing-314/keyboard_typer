import torch


def quaternion_batch_distance(q1, q2_batch):
    """
    Compute the quaternion distance between a single quaternion and a batch of quaternions.

    Args:
        q1 (Tensor): Shape (4,) in WXYZ format.
        q2_batch (Tensor): Shape (N, 4) in WXYZ format.

    Returns:
        Tensor: Angular distance in radians, shape (N,)
    """
    # Normalize quaternions to ensure they are unit quaternions
    q1 = q1 / torch.norm(q1)
    q2_batch = q2_batch / torch.norm(q2_batch, dim=1, keepdim=True)

    # Compute the dot product between q1 and each quaternion in q2_batch
    dot_products = torch.abs(torch.sum(q1 * q2_batch, dim=1))

    dot_products = torch.clamp(dot_products, -1.0, 1.0)

    distances = 2 * torch.acos(dot_products)

    return distances
