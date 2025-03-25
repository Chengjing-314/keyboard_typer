import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator
import h5py


class LPFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None
        self.is_init = False

    def next(self, x):
        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self):
        self.y = None
        self.is_init = False


class TactileZeroDriftCalibrator:
    def __init__(self, path, alpha=0.2) -> None:
        with h5py.File(path, "r") as f:
            self.left_hand_qpos = f["left"]["qpos"][:]
            self.left_hand_tactile = f["left"]["tactile"][:]
            self.right_hand_qpos = f["right"]["qpos"][:]
            self.right_hand_tactile = f["right"]["tactile"][:]

        # sequence [index, middle, ring, pinky, thumb]
        self.left_hand_tactile_interpolator = [
            interp1d(
                self.left_hand_qpos[..., ii],
                self.left_hand_tactile[..., ii * 6 : (ii + 1) * 6],
                axis=0,
            )
            for ii in range(4)
        ]

        self.left_hand_tactile_interpolator.append(
            LinearNDInterpolator(
                self.left_hand_qpos[..., -2:], self.left_hand_tactile[..., -6:]
            )
        )

        self.right_hand_tactile_interpolator = [
            interp1d(
                self.right_hand_qpos[..., ii],
                self.right_hand_tactile[..., ii * 6 : (ii + 1) * 6],
                axis=0,
            )
            for ii in range(4)
        ]
        self.right_hand_tactile_interpolator.append(
            LinearNDInterpolator(
                self.right_hand_qpos[..., -2:], self.right_hand_tactile[..., -6:]
            )
        )

    def calibrate_batch(self, qpos, tactile, is_right=False):
        hand_tactile_interpolator = (
            self.right_hand_tactile_interpolator
            if is_right
            else self.left_hand_tactile_interpolator
        )
        new_tactile = tactile.copy()
        # print('qpos', qpos)
        for i in range(tactile.shape[-1] // 6):
            _tactile = tactile[..., i * 6 : (i + 1) * 6]
            _qpos = qpos[..., i] if i < 4 else qpos[:, -2:]
            # print('_qpos', _qpos.shape, _qpos)
            if i < 4:
                _qpos = np.clip(
                    _qpos,
                    a_min=hand_tactile_interpolator[i].x.min() + 1e-3,
                    a_max=hand_tactile_interpolator[i].x.max() - 1e-3,
                )
                # zero_drift_idx = np.argmin(np.abs(hand_tactile_interpolator[i].x - _qpos))
                # zero_drift = hand_tactile_interpolator[i].y[zero_drift_idx]
            else:
                _qpos = np.clip(
                    _qpos,
                    a_min=hand_tactile_interpolator[i].points.min(0) + 1e-3,
                    a_max=hand_tactile_interpolator[i].points.max(0) - 1e-3,
                )
            zero_drift = hand_tactile_interpolator[i](_qpos)
            if np.isnan(zero_drift).any():
                # assign a reasonable value to zero-drift
                zero_drift = hand_tactile_interpolator[i](
                    hand_tactile_interpolator[i].points.mean(0))
            new_tactile[..., i * 6 : (i + 1) * 6] = np.clip(
                _tactile - zero_drift, a_min=0, a_max=None
            )
        return new_tactile


class TactileProcessor:
    def __init__(self, tactile_data_path="assets/tactile_data_0527.h5", alpha=0.2) -> None:
        self.tactile_zero_drift_calibrator = TactileZeroDriftCalibrator(
            tactile_data_path
        )
        self.filters = [LPFilter(alpha), LPFilter(alpha)]
        self.shift = []

    def process_tatile(self, hand0, hand1):
        hand0_tatile = hand0[:, 12:]
        hand1_tatile = hand1[:, 12:]
        calibrated_hand0_tatile = self.tactile_zero_drift_calibrator.calibrate_batch(
            hand0[:, :6], hand0_tatile, is_right=False
        )
        calibrated_hand1_tatile = self.tactile_zero_drift_calibrator.calibrate_batch(
            hand1[:, :6], hand1_tatile, is_right=True
        )
        if len(self.shift) >= 30 and hand0.shape[0] == 1:
            calibrated_hand0_tatile = np.clip(
                calibrated_hand0_tatile - np.array(self.shift)[..., :30].mean(0), a_min=0, a_max=None)
            calibrated_hand1_tatile = np.clip(
                calibrated_hand1_tatile - np.array(self.shift)[..., 30:].mean(0), a_min=0, a_max=None)
        filtered_hand0_tatile = np.array([self.filters[0].next(x) for x in calibrated_hand0_tatile])
        filtered_hand1_tatile = np.array([self.filters[1].next(x) for x in calibrated_hand1_tatile])
        hand0[:, 12:] = filtered_hand0_tatile
        hand1[:, 12:] = filtered_hand1_tatile
        if len(self.shift) < 30 and hand0.shape[0] == 1:
            self.shift.append(np.concatenate((calibrated_hand0_tatile, calibrated_hand1_tatile), axis=-1))
        # print(filtered_hand0_tatile.astype(int), filtered_hand1_tatile.astype(int))
        # print(len(self.shift))
        return hand0, hand1

    def process_tactile_single_hand(self, hand, is_right=False):
        hand_tatile = hand[:, 12:]
        calibrated_hand_tatile = self.tactile_zero_drift_calibrator.calibrate_batch(
            hand[:, :6], hand_tatile, is_right=is_right
        )
        if len(self.shift) >= 30:
            calibrated_hand_tatile = np.clip(
                calibrated_hand_tatile - np.array(self.shift).mean(0), a_min=0, a_max=None)
        hand_idx = 1 if is_right else 0
        filtered_hand_tatile = np.array([self.filters[hand_idx].next(x) for x in calibrated_hand_tatile])
        hand[:, 12:] = filtered_hand_tatile
        if len(self.shift) < 30:
            self.shift.append(calibrated_hand_tatile)
            # print(len(self.shift))
        
        return hand

    def reset_filter(self):
        for filter in self.filters:
            filter.reset()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import h5py

    tactile_data_path = "assets/tactile_data_0527.h5"
    tactile_zero_drift_calibrator = TactileZeroDriftCalibrator(tactile_data_path)

    path = "teleop_data/pour/robot_data_raw/episode_45/data.h5"

    with h5py.File(path, "r") as f:
        hand0 = f["hand0"][:]
        hand1 = f["hand1"][:]

    ori_hand0_tatile = hand0[:, 12:]
    ori_hand1_tatile = hand1[:, 12:]

    calibrated_hand0_tatile = tactile_zero_drift_calibrator.calibrate_batch(
        hand0[:, :6], ori_hand0_tatile, is_right=False
    )
    calibrated_hand1_tatile = tactile_zero_drift_calibrator.calibrate_batch(
        hand1[:, :6], ori_hand1_tatile, is_right=True
    )

    fig, axs = plt.subplots(10, 3, figsize=(10, 5))
    for idx in range(0, 30):
        x = np.arange(hand0.shape[0])
        _ori_hand0_tatile = ori_hand0_tatile[:, idx]
        _ori_hand1_tatile = ori_hand1_tatile[:, idx]
        hand0_tatile = calibrated_hand0_tatile[:, idx]
        hand1_tatile = calibrated_hand1_tatile[:, idx]
        # hand0_tatile_filtered = apply_filter(hand0_tatile, alpha=0.2)
        # print("origin", "max:", np.max(_hand0_tatile), "min:", np.min(hand0_tatile))
        # print("calibrated", "max:", np.max(hand0_tatile), "min:", np.min(hand0_tatile))
        # print(
        #     "filtered",
        #     "max:",
        #     np.max(hand0_tatile_filtered),
        #     "min:",
        #     np.min(hand0_tatile_filtered),
        # )

        # import ipdb; ipdb.set_trace(context=10)
        axs[idx // 3, idx % 3].plot(x, _ori_hand1_tatile, label="original")
        axs[idx // 3, idx % 3].plot(x, hand1_tatile, label="calibrated")
        axs[idx // 3, idx % 3].set_ylim(0, 3000)
        # plt.plot(x, hand0_tatile, label='original')
        # plt.plot(x, hand0_tatile_filtered, label='filtered')
    # plt.ylim(0, 3000)
    plt.legend()
    plt.show()
