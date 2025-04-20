import pandas as pd
import numpy as np
import os
import random
import shutil

def convert_to_sensor_files(input_file, output_dir="./output", dropout_rate=0.15, copy_to_iekf=True):
    """
    Convert simulated data to sensor files with random data dropout
    
    Parameters:
    - input_file: Path to input CSV file
    - output_dir: Directory to save output files
    - dropout_rate: Probability of dropping entire rows of data (0-1) for non-IMU sensors
    - copy_to_iekf: If True, copy the generated files to the AUV-InEKF/data directory
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(input_file, 'r') as file:
            firstLine = file.readline().strip()
        
        skiprows = 1 if firstLine.startswith("//") else 0

        df = pd.read_csv(input_file, skiprows=skiprows, header=None,
                         names=["time", "dvl_x", "dvl_y", "dvl_z", "lin_acc_x", 
                               "lin_acc_y", "lin_acc_z", "ang_vel_x", "ang_vel_y", 
                               "ang_vel_z", "depth"])
        
        # Convert ALL numeric columns to float, not just time
        numeric_columns = ["time", "dvl_x", "dvl_y", "dvl_z", "lin_acc_x", 
                          "lin_acc_y", "lin_acc_z", "ang_vel_x", "ang_vel_y", 
                          "ang_vel_z", "depth"]
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Drop rows with NaN values
        if df.isna().any().any():
            print("Warning: Some values could not be converted to numbers")
            df = df.dropna()

    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Calculate time step
    if len(df) >= 2:
        # Use fixed time step if data is evenly spaced
        dt = df['time'].iloc[1] - df['time'].iloc[0]
    else:
        dt = 0.1  # Default time step if not enough data points
    
    df['dt'] = dt

    # Create IMU file (angular velocity + linear acceleration + dt)
    # IMU data is kept intact (no dropout)
    imu_df = pd.DataFrame({
        'Time': df['time'],
        'dtheta.x': df['ang_vel_x'],
        'dtheta.y': df['ang_vel_y'], 
        'dtheta.z': df['ang_vel_z'],
        'dvel.x': df['lin_acc_x'],
        'dvel.y': df['lin_acc_y'],
        'dvel.z': df['lin_acc_z'],
        'dt': df['dt']
    })
    imu_path = f"{output_dir}/simulated_imu.csv"
    imu_df.to_csv(imu_path, index=False)
    print(f"Created IMU file with {len(imu_df)} records")

    # Create DVL file with random row dropouts
    # Generate mask for which rows to keep (True) or drop (False)
    dvl_mask = np.random.random(len(df)) >= dropout_rate
    # Ensure first and last rows are kept for better initialization
    dvl_mask[0] = True
    dvl_mask[-1] = True
    
    dvl_df = pd.DataFrame({
        'Time': df['time'][dvl_mask],
        'velocityA': df['dvl_x'][dvl_mask],
        'velocityB': df['dvl_y'][dvl_mask],
        'velocityC': df['dvl_z'][dvl_mask]
    })
    
    dvl_path = f"{output_dir}/simulated_dvl.csv"
    dvl_df.to_csv(dvl_path, index=False)
    print(f"Created DVL file with {len(dvl_df)} records (removed {len(df) - len(dvl_df)} readings)")

    # Create DEPTH file with random row dropouts
    depth_mask = np.random.random(len(df)) >= dropout_rate
    # Ensure first and last rows are kept
    depth_mask[0] = True
    depth_mask[-1] = True
    
    depth_df = pd.DataFrame({
        'Time': df['time'][depth_mask],
        'data': df['depth'][depth_mask]
    })
    
    depth_path = f"{output_dir}/simulated_depth.csv"
    depth_df.to_csv(depth_path, index=False)
    print(f"Created DEPTH file with {len(depth_df)} records (removed {len(df) - len(depth_df)} readings)")

    # Create AHRS file with orientation data
    # Use a more stable approach to generate orientation data
    
    # 1. Compute initial orientation (start at zero or some reference)
    initial_roll = 0.0
    initial_pitch = 0.0
    initial_yaw = 0.0
    
    # 2. Integrate angular velocities
    roll_angles = [initial_roll]
    pitch_angles = [initial_pitch]
    yaw_angles = [initial_yaw]
    
    for i in range(1, len(df)):
        # Get angular velocities and time step
        wx = df['ang_vel_x'].iloc[i]
        wy = df['ang_vel_y'].iloc[i]
        wz = df['ang_vel_z'].iloc[i]
        delta_t = df['dt'].iloc[i]
        
        # Simple integration
        new_roll = roll_angles[-1] + wx * delta_t
        new_pitch = pitch_angles[-1] + wy * delta_t
        new_yaw = yaw_angles[-1] + wz * delta_t
        
        # Normalize angles to keep them within [0, 2π)
        roll_angles.append(new_roll % (2 * np.pi))
        pitch_angles.append(new_pitch % (2 * np.pi))
        yaw_angles.append(new_yaw % (2 * np.pi))
    
    # Generate mask for which rows to keep (True) or drop (False)
    # Using lower dropout rate for AHRS to ensure we have enough orientation data
    ahrs_mask = np.random.random(len(df)) >= (dropout_rate * 0.7)
    # Ensure first and last rows are kept
    ahrs_mask[0] = True
    ahrs_mask[-1] = True
    
    ahrs_df = pd.DataFrame({
        'Time': df['time'][ahrs_mask],
        'theta_0': np.array(roll_angles)[ahrs_mask],  # Roll
        'theta_1': np.array(pitch_angles)[ahrs_mask], # Pitch
        'theta_2': np.array(yaw_angles)[ahrs_mask]    # Yaw
    })
    
    ahrs_path = f"{output_dir}/simulated_ahrs.csv"
    ahrs_df.to_csv(ahrs_path, index=False)
    print(f"Created AHRS file with {len(ahrs_df)} records (removed {len(df) - len(ahrs_df)} readings)")
    
    print("\nNote: Sensor data (except IMU) has random row dropouts to simulate real-world conditions.")
    print(f"Dropout rate used: {dropout_rate*100:.1f}% (less for AHRS: {dropout_rate*0.7*100:.1f}%)")
    print("Note on AHRS data: The orientation angles are generated by integrating angular velocities.")
    
    # Copy files to the AUV-InEKF/data directory if requested
    if copy_to_iekf:
        iekf_data_dir = "../AUV-InEKF/data"
        if os.path.exists(iekf_data_dir):
            print("\nCopying files to AUV-InEKF/data directory...")
            try:
                shutil.copy(imu_path, os.path.join(iekf_data_dir, "simulated_imu.csv"))
                shutil.copy(dvl_path, os.path.join(iekf_data_dir, "simulated_dvl.csv"))
                shutil.copy(depth_path, os.path.join(iekf_data_dir, "simulated_depth.csv"))
                shutil.copy(ahrs_path, os.path.join(iekf_data_dir, "simulated_ahrs.csv"))
                print("Files copied successfully.")
            except Exception as e:
                print(f"Error copying files: {e}")
        else:
            print(f"Warning: AUV-InEKF data directory not found at {iekf_data_dir}")
            print("Files were not copied. You may need to manually copy them.")

if __name__ == "__main__":
    input_file = "data/sensor_data_largeCov.csv"
    convert_to_sensor_files(input_file, dropout_rate=0.15)  # 15% dropout rate by default