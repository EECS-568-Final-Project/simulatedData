from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.random import default_rng
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from typing import NamedTuple, Protocol
from dataclasses import dataclass
from math import sin, cos, floor
from tqdm import tqdm

generator = default_rng()
type FloatMat = npt.NDArray[np.floating]


class Vec3(NamedTuple):
    x: float
    y: float
    z: float

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
        )

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z,
        )

    def __mul__(self, other: float) -> Vec3:
        return Vec3(
            x=self.x * other,
            y=self.y * other,
            z=self.z * other,
        )

    def __rmul__(self, other: float) -> Vec3:
        return self * other

    def __truediv__(self, other: float) -> Vec3:
        return self * (1 / other)

    @staticmethod
    def from_matrix(matrix: FloatMat) -> Vec3:
        return Vec3(matrix[0], matrix[1], matrix[2])

    def as_matrix(self) -> FloatMat:
        return np.array([self.x, self.y, self.z])


@dataclass
class Pose:
    matrix: FloatMat  # 4x4

    @staticmethod
    def from_rotation(pos: Vec3, rotation: FloatMat) -> Pose:  # 3x3
        return Pose(
            np.block(
                [
                    [rotation, pos.as_matrix()[:, None]],
                    [np.zeros((1, 3)), 1],
                ]
            )
        )

    @staticmethod
    def from_euler(pos: Vec3, euler: Vec3) -> Pose:
        rotation = Rotation.from_euler("zyx", euler, degrees=True).as_matrix()
        return Pose.from_rotation(pos, rotation)

    @property
    def position(self) -> Vec3:
        return Vec3.from_matrix(self.matrix[:3, 3].reshape((3,)))

    @property
    def rotation(self) -> FloatMat:
        return self.matrix[:3, :3]

    def inverse(self) -> Pose:
        rt = self.rotation.T
        return Pose.from_rotation(Vec3.from_matrix(-rt @ self.position.as_matrix()), rt)


def vee3(mat: FloatMat) -> Vec3:  # 3x3
    return Vec3(mat[2, 1], mat[0, 2], mat[1, 0])


def hat3(vec: Vec3) -> FloatMat:  # 3x3
    return np.array(
        [
            [0, -vec.z, vec.y],
            [vec.z, 0, -vec.x],
            [-vec.y, vec.x, 0],
        ]
    )


@dataclass
class SensorNoise:
    dvl_cov: FloatMat  # 3x3
    lin_acc_cov: FloatMat  # 3x3
    ang_vel_cov: FloatMat  # 3x3
    depth_std: float
    lin_acc_bias_cov: FloatMat  # 3x3
    ang_vel_bias_cov: FloatMat  # 3x3
    lin_acc_bias: Vec3 = Vec3(0, 0, 0)
    ang_vel_bias: Vec3 = Vec3(0, 0, 0)

    def add_noise(self, data: SensorData):
        zero = np.array([0, 0, 0])

        def vec3_noise(cov: FloatMat) -> Vec3:
            return Vec3.from_matrix(generator.multivariate_normal(zero, cov))

        data.dvl += vec3_noise(self.dvl_cov)
        self.lin_acc_bias += vec3_noise(self.lin_acc_bias_cov)
        data.lin_acc += vec3_noise(self.lin_acc_cov) + self.lin_acc_bias
        self.ang_vel_bias += vec3_noise(self.ang_vel_bias_cov)
        data.ang_vel += vec3_noise(self.ang_vel_cov) + self.ang_vel_bias
        data.depth += generator.normal(0, self.depth_std)


@dataclass
class SensorData:
    time: float
    dvl: Vec3
    lin_acc: Vec3
    ang_vel: Vec3
    depth: float

    def floatify(self):
        self.dvl = Vec3(*map(float, self.dvl))
        self.lin_acc = Vec3(*map(float, self.lin_acc))
        self.ang_vel = Vec3(*map(float, self.ang_vel))
        self.depth = float(self.depth)


class Path(Protocol):
    def sample(self, t: float) -> Pose: ...


class SimplePath:
    def sample(self, t: float) -> Pose:
        return Pose.from_euler(Vec3(t, t, 0), Vec3(0, 0, 0))


class SinPath:
    def sample(self, t: float) -> Pose:
        return Pose.from_euler(Vec3(sin(t), cos(t), sin(t + 1)), Vec3(0, 0, 0))


def generate_data(
    path: Path,
    sensor_noise: SensorNoise,
    sample_period: float,
    start_time: float = 0,
    end_time: float = 10,
) -> list[SensorData]:
    sensor_data = []

    duration = end_time - start_time
    num_samples = floor(duration / sample_period)
    assert num_samples >= 3, "Must have 3 or more samples"

    for i in tqdm(range(0, num_samples)):
        time = start_time + sample_period * i

        pose = path.sample(time)
        prev = path.sample(time - sample_period)
        next = path.sample(time + sample_period)

        derivative = (next.matrix - pose.matrix) / sample_period
        vel = Vec3.from_matrix(derivative[:3, 3])
        prev_vel = (pose.position - prev.position) / sample_period

        lin_acc = (vel - prev_vel) / sample_period
        twist = derivative @ pose.inverse().matrix
        ang_vel = vee3(twist[:3, :3])

        datum = SensorData(
            time=time,
            dvl=vel,
            lin_acc=lin_acc,
            ang_vel=ang_vel,
            depth=pose.position.z,
        )
        sensor_noise.add_noise(datum)
        datum.floatify()
        sensor_data.append(datum)

    return sensor_data


def plotPath(estimatedPoses: list[Vec3], truePoses: list[Vec3]) -> None:

    # Plot each axis separately
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle("Noisy Robot Path")
    fig.subplots_adjust(hspace=0.5)
    labels = ["X", "Y", "Z"]

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("X")
    axs[0].plot(
        [pose.x for pose in estimatedPoses],
        label="Estimated Path",
    )
    axs[0].plot(
        [pose.x for pose in truePoses],
        label="True Path",
    )

    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Y")
    axs[1].plot(
        [pose.y for pose in estimatedPoses],
        label="Estimated Path",
    )
    axs[1].plot(
        [pose.y for pose in truePoses],
        label="True Path",
    )

    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Z")
    axs[2].plot(
        [pose.z for pose in estimatedPoses],
        label="Estimated Path",
    )
    axs[2].plot(
        [pose.z for pose in truePoses],
        label="True Path",
    )
    for i, ax in enumerate(axs):
        ax.set_title(f"Path in {labels[i]} direction")
        ax.legend()
        ax.grid()
    plt.show()


def estimate_poses(data: list[SensorData], start_pose: Pose) -> list[Vec3]:
    current_position = start_pose.position
    estimated_poses = [current_position]
    for i in range(len(data) - 1):
        datum = data[i]
        next_datum = data[i + 1]
        current_position += datum.dvl * (next_datum.time - datum.time)
        estimated_poses.append(current_position)
    return estimated_poses


def main():
    large_cov = True

    if large_cov:
        # Define larger covariance matrices and standard deviation for some noise
        low_noise_cov = np.eye(3) * 0.1
        low_depth_std = 0.1
        noise = SensorNoise(
            low_noise_cov,
            low_noise_cov,
            low_noise_cov,
            low_depth_std,
            low_noise_cov,
            low_noise_cov,
        )

    else:  # Use default noise (no noise)
        zero = np.zeros((3, 3))
        noise = SensorNoise(zero, zero, zero, 0, zero, zero)

    path = SinPath()

    data = generate_data(path, noise, 0.1)

    estimatedPoses = estimate_poses(data, path.sample(0.0))
    # True path is path.sample(time) for each time in data
    truePoses = [path.sample(datum.time).position for datum in data]

    print("Generated data:")
    for datum in data:
        print(
            f"t: {datum.time:.2f}, dvl: {datum.dvl}, lin_acc: {datum.lin_acc}, ang_vel: {datum.ang_vel}, depth: {datum.depth:.2f}"
        )

    plotPath(estimatedPoses, truePoses)
    print("Done generating data")

    # Save data to file
    with open("data/sensor_data_largeCov.csv", "w") as f:
        f.write(
            "time, dvl_x, dvl_y, dvl_z, lin_acc_x, lin_acc_y, lin_acc_z, ang_vel_x, ang_vel_y, ang_vel_z, depth\n"
        )
        for datum in data:
            f.write(
                f"{datum.time:.2f}, {datum.dvl.x:.2f}, {datum.dvl.y:.2f}, {datum.dvl.z:.2f}, "
                f"{datum.lin_acc.x:.2f}, {datum.lin_acc.y:.2f}, {datum.lin_acc.z:.2f}, "
                f"{datum.ang_vel.x:.2f}, {datum.ang_vel.y:.2f}, {datum.ang_vel.z:.2f}, "
                f"{datum.depth:.2f}\n"
            )


if __name__ == "__main__":
    main()
