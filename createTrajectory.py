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
    NOISY_sensorData = []

    BIAS = Vec3(x=0.01, y=0.01, z=0.01)
    
    for i in range(0, 5 * sample_rate + 1):
        time = i / sample_rate
        pose = path.sample(time)
        noisyPosition, noisyData = sensor_noise.add_noise(BIAS, )
        NOISY_sensorData.append(noisyData)
        
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



runPATH(SimplePath, SensorNoise, 0.05)runPATH(SimplePath, SensorNoise)amth(noisyPath, truePos, se)



runPATH(SimplePath, SensorNoise)