import numpy as np
import numpy.typing as npt
from numpy.random import default_rng
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from typing import NamedTuple, Protocol
from dataclasses import dataclass
from math import sin, cos

generator = default_rng()

class Vec3(NamedTuple):
    x: float
    y: float
    z: float

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(
            x = self.x + other.x,
            y = self.y + other.y,
            z = self.z + other.z,
        )
    
    @staticmethod
    def from_matrix(matrix: npt.NDArray[float]) -> Vec3:
        return Vec3(matrix[0], matrix[1], matrix[2])    
def as_matrix(self) -> npt.NDArray[float]:
        return np.array([self.x, self.y, self.z])

class Pose:
    matrix: npt.NDArray[float]

    @staticmethod
    def from_euler(pos: Vec3, euler: Vec3) -> Pose:
        rotation = Rotation.from_euler('zyx', euler, degrees=True).as_matrix()
        return Pose(np.block([
            [rotation, pos.as_matrix()[:, None]],
            [np.zeros((1, 3)), 1]
        ]))

class SensorNoise(NamedTuple):
    dvl_cov: npt.NDArray[float] # 3x3
    lin_acc_cov: npt.NDArray[float] # 3x3
    ang_vel_cov: npt.NDArray[float] # 3x3
    depth_std: float

    def add_noise(self, data: SensorData):
        data.dvl += generator.multivariate_normal(self.dvl_cov)
        data.lin_acc += Vec3.from_matrix(generator.multivariate_normal(self.lin_acc_cov))
        data.ang_vel += Vec3.from_matrix(generator.multivariate_normal(self.ang_vel_cov))
        data.depth += Vec3.from_matrix(generator.normal(self.depth_std))

@dataclass
class SensorData:
    time: float
    dvl: Vec3
    lin_acc: Vec3
    ang_vel: Vec3
    depth: float
    
class Path(Protocol):
    def sample(self, t: float) -> Pose:
        pass
    
class SimplePath:
    def sample(self, t: float) -> Pose:
        return Pose.from_euler(np.zero((3,)), Vec3(t, t, 0))
    
class SinPath:
    def sample(self, t: float) -> Pose:
        return Pose.from_euler(np.zero((3,)), Vec3(sin(t), cos(t), sin(t + 1)))


def generate_data(path: Path, sensor_noise: SensorNoise, sample_rate: float) -> list[SensorData]:

    def velocity(pose1: Pose, pose2: Pose, dt: float):
        pos1 = pose1.matrix[:3, 3]
        pos2 = pose2.matrix[:3, 3]
        return (pos2 - pos1) / dt

    pose_data = []
    NOISY_sensorData = []
    
    for i in range(0, 5, 1/sample_rate):
        pose = path.sample(i)
        pose_data.append(pose)

    

    # TODO: Need to do angular velocity
    '''
        * Currently just taking the differnece between 2 points to find sensor data
        * We can change it to use derivatives instead if we want
    '''
    prev_pose, prev_velocity = None
    for time, i in enumerate(pose_data):
        # Is it the first point of data?
        if prev_pose is None:
            prev_pose = i
            continue

        curVelocity = velocity(prev_pose, i, 1/sample_rate)

        # Can we calculate linear acceleration
        if prev_velocity is None:
            prev_velocity = curVelocity
            continue

        acc = (curVelocity - prev_velocity) / (1/sample_rate)


        # Reset Vars for next iteration
        SensorData = SensorData(time/sample_rate,
                                curVelocity,
                                acc,
                                Vec3(0, 0, 0),
                                0)
        prev_pose = i
        prev_velocity = curVelocity


        # Do we want the noise to propogate through the data?
        # Currently noise is applied on true sensor data
        NOISY_sensorData.append(sensor_noise.add_noise(SensorData))



    return NOISY_sensorData

        

def plotPath(estimatedPoses, truePoses):

    # TODO: fix this to be 3 separate graphs for rotation
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(*zip(*[(p.x, p.y, p.z) for p in estimatedPoses]), label="Esimtated Path")
    ax.plot(*zip(*[(p.x, p.y, p.z) for p in truePoses]), label="True Path")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title('Noisy Robot Path')
    ax.legend()
    plt.show()
        

def runPATH(path: Path, sensorNoise: SensorNoise, sample_rate: float):
    truePose = []
    noisyPath = []

    for i in range(0, 5*sample_rate + 1):
        time = i / sample_rate
        truePose.append(path.sample(time))

    noisyData = generate_data(path, sensorNoise, sample_rate)

    # Run Filter
    # noisyPath = runFilter(noisyData)

    plotPath(noisyPath, truePose)



runPATH(SimplePath, SensorNoise, 0.05)