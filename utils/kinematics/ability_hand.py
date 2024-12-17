import numpy as np
import pytransform3d.batch_rotations
import torch
import pytransform3d
from scipy.spatial.transform import Rotation as R


def batch_euler_distance(euler1, euler2):
    """
    Calculate the angular distance between two batches of Euler angles.

    Args:
        euler1 (torch.Tensor): Tensor of shape (N, 3) containing the first set of Euler angles.
        euler2 (torch.Tensor): Tensor of shape (N, M, 3) containing M sets of Euler angles to compare against.

    Returns:
        torch.Tensor: Tensor of shape (N, M) containing the angular distances.
    """
    # Compute the element-wise difference
    diff = euler1.unsqueeze(1) - euler2  # Shape: (N, M, 3)

    # Wrap the differences to the range [-pi, pi] to get the shortest distance
    diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi

    # Compute the Euclidean distance for each pair
    distance = torch.norm(diff, dim=-1)  # Shape: (N, M)

    return distance


def batch_closest_euler_from_quaternion(input_eulers, quaternions):
    """
    Find the Euler angles (intrinsic XYZ) closest to each input Euler angles in a batch,
    given a batch of quaternions.

    Args:
        input_eulers (torch.Tensor): Tensor of shape (N, 3) containing input Euler angles in radians.
        quaternions (torch.Tensor): Tensor of shape (N, 4) containing quaternions (w, x, y, z).

    Returns:
        torch.Tensor: Tensor of shape (N, 3) containing the closest Euler angles in radians.
    """
    # Normalize the quaternions to ensure they are valid
    quaternions = quaternions / quaternions.norm(p=2, dim=1, keepdim=True)  # Shape: (N, 4)

    # Convert quaternions to Euler angles (intrinsic XYZ) for each quaternion in the batch
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # Compute Euler angles from quaternions
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)  # Clamp to avoid out-of-range errors
    pitch_y = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    # Base Euler angles
    base_euler = torch.stack((roll_x, pitch_y, yaw_z), dim=1)  # Shape: (N, 3)

    # Generate alternative Euler representations for each angle in the batch
    alternatives = torch.stack(
        [
            base_euler,
            base_euler + torch.tensor([2 * torch.pi, 0, 0]),
            base_euler - torch.tensor([2 * torch.pi, 0, 0]),
            base_euler + torch.tensor([0, 2 * torch.pi, 0]),
            base_euler - torch.tensor([0, 2 * torch.pi, 0]),
            base_euler + torch.tensor([0, 0, 2 * torch.pi]),
            base_euler - torch.tensor([0, 0, 2 * torch.pi]),
        ],
        dim=1,
    )  # Shape: (N, 7, 3)

    # Compute distances between each input Euler and all alternatives
    distances = batch_euler_distance(input_eulers, alternatives)  # Shape: (N, 7)

    # Find the index of the minimum distance for each batch element
    min_indices = torch.argmin(distances, dim=1)  # Shape: (N,)

    # Gather the closest Euler angles using the minimum indices
    closest_euler = torch.gather(
        alternatives, 1, min_indices.view(-1, 1, 1).expand(-1, 1, 3)
    ).squeeze(
        1
    )  # Shape: (N, 3)

    return closest_euler


def unwrap_angles(euler_angles):
    return torch.unwrap(euler_angles, dim=0)


def batch_quaternion_to_euler_intrinsic_xyz(quaternions):
    """
    Convert a batch of quaternions to Euler angles (intrinsic XYZ) in PyTorch.

    Args:
        quaternions (torch.Tensor): Tensor of shape (N, 4) containing quaternions (w, x, y, z).
        order (str): Order of the Euler angles ('xyz').

    Returns:
        torch.Tensor: Tensor of shape (N, 3) containing Euler angles in radians.
    """
    assert quaternions.shape[1] == 4, "Input tensor must be of shape (N, 4) for quaternions."

    # Normalize the quaternion to avoid any numerical errors
    quaternions = quaternions / quaternions.norm(p=2, dim=1, keepdim=True)

    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # Compute the Euler angles from the quaternion
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(
        t2, -1.0, 1.0
    )  # Clamp to avoid out-of-range errors due to floating-point precision
    pitch_y = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    euler_angles = torch.stack((roll_x, pitch_y, yaw_z), dim=1)
    euler_angles = (euler_angles + torch.pi) % (2 * torch.pi) - torch.pi  # Normalize to [-pi, pi]

    return euler_angles


def recover_residual(residual, residual_low, residual_high):
    # residual is between -1 and 1
    return residual * (residual_high - residual_low) / 2 + (residual_high + residual_low) / 2


def apply_residuals_to_delta_pose(
    delta_left, delta_right, residual_left, residual_right, residual_low, residual_high
):

    # assume the residuals are between -1 and 1
    residual_left = recover_residual(
        torch.clamp(residual_left, -1, 1), residual_low, residual_high
    )
    residual_right = recover_residual(
        torch.clamp(residual_right, -1, 1), residual_low, residual_high
    )

    delta_left[:, :3] += residual_left[:, :3]  # xyz can be added directly
    delta_right[:, :3] += residual_right[:, :3]

    delta_left_rpy, delta_right_rpy = delta_left[:, 3:], delta_right[:, 3:]
    residual_left_rpy, residual_right_rpy = residual_left[:, 3:], residual_right[:, 3:]

    delta_left_rpy = delta_left_rpy.cpu().numpy()
    delta_right_rpy = delta_right_rpy.cpu().numpy()

    residual_left_rpy = residual_left_rpy.cpu().numpy()
    residual_right_rpy = residual_right_rpy.cpu().numpy()

    delta_left_new_R = R.from_euler("XYZ", delta_left_rpy)
    delta_right_new_R = R.from_euler("XYZ", delta_right_rpy)

    residual_left_R = R.from_euler("XYZ", residual_left_rpy)
    residual_right_R = R.from_euler("XYZ", residual_right_rpy)

    delta_left_new_rpy = torch.tensor((residual_left_R * delta_left_new_R).as_euler("XYZ"))
    delta_right_new_rpy = torch.tensor((residual_right_R * delta_right_new_R).as_euler("XYZ"))

    delta_left[:, 3:6] = delta_left_new_rpy
    delta_right[:, 3:6] = delta_right_new_rpy

    return delta_left, delta_right


def get_intersection_circles_batch(o0, r0, o1, r1):
    # Calculate the distance between the origins of the circles
    # o0 = 1 x 3, o1 = num_envs x 3
    d = torch.sqrt(torch.sum((o0 - o1) ** 2, dim=1)).unsqueeze(-1)
    # d = num_envs x 1

    r0_sq = r0**2
    r1_sq = r1**2
    d_sq = d**2

    # Solve for 'a', a = num_envs x 1
    a = (r0_sq - r1_sq + d_sq) / (2 * d)

    # Solve for 'h', h = num_envs x 1
    h_sq = r0_sq - a**2
    h = torch.sqrt(h_sq)

    # Find p2, p2 = num_envs x 3
    p2 = o0 + a * (o1 - o0) / d

    t1 = h * (o1[:, 1] - o0[:, 1]).unsqueeze(-1) / d
    t2 = h * (o1[:, 0] - o0[:, 0]).unsqueeze(-1) / d

    # Calculate the two solutions
    sol0 = torch.zeros(p2.shape[0], 2)
    sol1 = torch.zeros(p2.shape[0], 2)

    sol0[:, 0] = p2[:, 0] + t1.squeeze(-1)
    sol0[:, 1] = p2[:, 1] - t2.squeeze(-1)

    sol1[:, 0] = p2[:, 0] - t1.squeeze(-1)
    sol1[:, 1] = p2[:, 1] + t2.squeeze(-1)

    return sol0, sol1


def get_abh_4bar_driven_angle_batch(q1):

    # q1 is num_envs x 1, it is easier to just iterate over columns

    q1 = q1 + 0.084474  # factor in offset imposed by our choice of link frame attachments

    L1 = 38.6104
    L2 = 36.875
    L3 = 9.1241
    p3 = torch.tensor([9.47966, -0.62133, 0], dtype=q1.dtype, device=q1.device).unsqueeze(
        0
    )  # expand for broadcasting

    cq1 = torch.cos(q1)
    sq1 = torch.sin(q1)
    p1 = torch.stack(
        [
            L1 * cq1,
            L1 * sq1,
            torch.zeros_like(cq1),
        ],
        dim=1,
    ).squeeze(-1)

    p1 = p1.to(q1.device)

    # Get the intersection points for each batch
    sol0, sol1 = get_intersection_circles_batch(p3, L2, p1, L3)

    if torch.any(torch.isnan(sol0)) or torch.any(torch.isnan(sol1)):
        for i in range(sol0.shape[0]):
            if torch.any(torch.isnan(sol0[i])) or torch.any(torch.isnan(sol1[i])):
                print(f"Batch {i} has nan values in sol0 or sol1.")
                print(q1[i])
        raise ValueError("Nan values found in sol0 or sol1. Check input values.")

    p2 = sol1

    p2 = p2.to(q1.device)

    # Calculate the linkage intermediate angle
    q2pq1 = torch.atan2(p2[:, 1] - L1 * sq1.squeeze(-1), p2[:, 0] - L1 * cq1.squeeze(-1))
    q2 = q2pq1 - q1.squeeze(-1)
    q2 = torch.remainder(q2 + torch.pi, 2 * torch.pi) - torch.pi

    # check for nan values
    if torch.any(torch.isnan(q2)):
        raise ValueError("Nan values found in q2. Check input values.")

    return q2.unsqueeze(-1)


def map_to_q_pos(nn_output):
    # nn_output: num_envs x 6, assume not normalized but clamped

    thumb_q1 = nn_output[:, 0].unsqueeze(-1)
    thumb_q2 = nn_output[:, 5].unsqueeze(-1)
    fingers = nn_output[:, 1:5]

    output = []

    for i in range(4):
        output.append(get_abh_4bar_driven_angle_batch(fingers[:, i].unsqueeze(-1)))

    output = torch.cat(output, dim=1)

    return torch.cat(
        (thumb_q1, fingers, thumb_q2, output),
        dim=-1,
    )


def obtain_hand_actual_qpos_normalized(env, hand_action):
    """_summary_
        This function directly takes in nn output and maps it to the actual qpos of the hand.
    Args:
        env (): _description_
        hand_action (_type_): _description_

    Returns:
        _type_: _description_
    """

    # ** return is left then right
    with torch.no_grad():

        hand_action = torch.clamp(hand_action, -1, 1)

        r_action, l_action = env.unwrapped.recover_hand_qpos_6d(
            hand_action[:, 6:], hand_action[:, :6]
        )

        r_action = map_to_q_pos(r_action)
        l_action = map_to_q_pos(l_action)

        r_action, l_action = env.unwrapped.normalize_hand_qpos(r_action, l_action)

        action = torch.cat((l_action, r_action), dim=-1)

    return action


def obtain_hand_actual_qpos(hand_action):

    with torch.no_grad():

        r_action, l_action = hand_action[:, 6:], hand_action[:, :6]

        r_action = map_to_q_pos(r_action)
        l_action = map_to_q_pos(l_action)

        action = torch.cat((l_action, r_action), dim=-1)

    return action


def test_get_abh_4bar_driven_angle_batch():
    test_angles = torch.tensor(
        [[0.1], [0.2], [0.3], [0.4], [0.5], [0.1], [0.8], [1.1]], dtype=torch.float32
    )

    q2_batch = get_abh_4bar_driven_angle_batch(test_angles)

    q2_individual = []
    for angle in test_angles:
        q2_individual.append(get_abh_4bar_driven_angle(angle.item()))

    q2_individual = torch.tensor(q2_individual, dtype=torch.float32).unsqueeze(1)

    assert torch.allclose(
        q2_batch, q2_individual, atol=1e-6
    ), f"Mismatch found! Batch: {q2_batch}, Individual: {q2_individual}"

    print("All tests passed successfully.")


#### Legacy Code from Lixing


def get_abh_4bar_driven_angle(q1):
    q1 = q1 + 0.084474  # factor in offset imposed by our choice of link frame attachments

    # L0 = 9.5
    L1 = 38.6104
    L2 = 36.875
    L3 = 9.1241
    p3 = np.array(
        [9.47966, -0.62133, 0]
    )  # if X of the base frame was coincident with L3, p3 = [9.5, 0 0]. However, our frame choices are different to make the 0 references for the fingers nice, so this location is a little less convenient.

    cq1 = np.cos(q1)
    sq1 = np.sin(q1)
    p1 = np.array([L1 * cq1, L1 * sq1, 0])

    sol0, sol1 = get_intersection_circles(p3, L2, p1, L3)

    # copy_vect3(&p2, &sols[1])
    p2 = sol1

    # calculate the linkage intermediate angle!
    q2pq1 = np.arctan2(p2[1] - L1 * sq1, p2[0] - L1 * cq1)
    q2 = q2pq1 - q1
    q2 = np.mod(q2 + np.pi, 2 * np.pi) - np.pi
    return q2


"""
	Helper function for the above function. 
	Solves for the intersection of two circles.
	
	Behavior of this function for circles that intersect at only one point, or 
	circles that do not intersect, is not defined.
	
	INPUTS: 
		o0: origin of circle 0
		r0: radius of circle 0
		o1: origin of circle 1
		r1: origin of circle 1
	OUTPUS:
		sol0: 2d position of the first intersection point
		sol1: 2d position of the second intersection point
"""


def get_intersection_circles(o0, r0, o1, r1):

    d = np.sqrt(np.sum((o0 - o1) ** 2))

    sol0 = np.zeros(2)
    sol1 = np.zeros(2)

    r0_sq = r0 * r0
    r1_sq = r1 * r1
    d_sq = d * d

    # solve for a
    a = (r0_sq - r1_sq + d_sq) / (2 * d)

    # solve for h
    h_sq = r0_sq - a * a
    h = np.sqrt(h_sq)

    # find p2
    p2 = o0 + a * (o1 - o0) / d

    t1 = h * (o1[1] - o0[1]) / d
    t2 = h * (o1[0] - o0[0]) / d

    sol0[0] = p2[0] + t1
    sol0[1] = p2[1] - t2

    sol1[0] = p2[0] - t1
    sol1[1] = p2[1] + t2

    return sol0, sol1


# Ability Hand Real Order:
# index, middle, ring, pinky, thumb l2, thumb l1

# Sapien Order:
# index, middle, ring, pinky, thumb l1, thumb l2


def main():
    # test_get_abh_4bar_driven_angle_batch()

    test_output = np.ones((1024, 6)) * 0


if __name__ == "__main__":
    main()
