
import pandas as pd
import numpy as np
import os

def convert_to_sensor_files(input_file, output_dir="./output"):
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

    except:
        print("Error reading file")
        return

    # Calculate time step
    if len(df) >= 2:
        # Use fixed time step if data is evenly spaced
        dt = df['time'].iloc[1] - df['time'].iloc[0]
    else:
        dt = 0.1  # Default time step if not enough data points
    
    df['dt'] = dt

    # Create IMU file (angular velocity + linear acceleration + dt)
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
    imu_df.to_csv(f"{output_dir}/simulated_imu.csv", index=False)
    print(f"Created IMU file with {len(imu_df)} records")

    # Create DVL file (velocity measurements)
    dvl_df = pd.DataFrame({
        'Time': df['time'],
        'velocityA': df['dvl_x'],
        'velocityB': df['dvl_y'],
        'velocityC': df['dvl_z']
    })
    dvl_df.to_csv(f"{output_dir}/simulated_dvl.csv", index=False)
    print(f"Created DVL file with {len(dvl_df)} records")

    # Create DEPTH file
    depth_df = pd.DataFrame({
        'Time': df['time'],
        'data': df['depth']
    })
    depth_df.to_csv(f"{output_dir}/simulated_depth.csv", index=False)
    print(f"Created DEPTH file with {len(depth_df)} records")

    # Create AHRS file (need to generate theta values from angular velocities)
    # This is an approximation by cumulative integration of angular velocities
    # Convert to Euler angles using simple integration (for demonstration)
    roll = np.cumsum(df['ang_vel_x'] * df['dt'])
    pitch = np.cumsum(df['ang_vel_y'] * df['dt'])
    yaw = np.cumsum(df['ang_vel_z'] * df['dt'])
    
    # Normalize angles to stay within reasonable ranges
    roll = roll % (2 * np.pi)
    pitch = pitch % (2 * np.pi)
    yaw = yaw % (2 * np.pi)
    
    ahrs_df = pd.DataFrame({
        'Time': df['time'],
        'theta_0': roll,  # Roll
        'theta_1': pitch, # Pitch
        'theta_2': yaw    # Yaw
    })
    ahrs_df.to_csv(f"{output_dir}/simulated_ahrs.csv", index=False)
    print(f"Created AHRS file with {len(ahrs_df)} records")
    
    print("\nNote on AHRS data: The orientation angles are approximated by integrating angular velocities.")
    print("For more accurate results, consider using proper orientation calculation methods.")

if __name__ == "__main__":
    input_file = "data/sensor_data_largeCov.csv"
    convert_to_sensor_files(input_file)