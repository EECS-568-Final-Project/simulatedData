from bagpy import bagreader
import pandas as pd

b= bagreader('night_2.bag')

print(b.topic_table)

depth = b.message_by_topic('/depth/raw_depth')
dvl = b.message_by_topic('/dvl/raw_dvl')
imu = b.message_by_topic('/imu_PIMU')
ahrs = b.message_by_topic('/imu_INS')

depth
dvl
imu
ahrs

# ahrsData = b.message_by_topic('/imu_PIMU')
# print(ahrsData)