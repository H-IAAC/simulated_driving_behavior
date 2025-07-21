import traci
import traci.constants as tc
import numpy as np
import os
import shutil

from . import sumo_helper
import csv
import time as _time

sumoBinary = "/usr/bin/sumo-gui"
sumoCmd = [sumoBinary, "-c", "osm.sumocfg"]

# Code of each variable to subscribe:
SPEED = 64
POSITION = 66
ACCELERATION = 114
ANGLE = 67
DEPATURE = 58


def get_all_variables(folder_path, veh_ids, delta_time=0.05, end_hours=0, use_gui=False, convert_geo=True, freeze_traffic_lights=False):
    """Get all variables from the simulation.
    Args:
        folder_path (str): Path to the folder where the SUMO configuration file is located.
        veh_ids (list): List of vehicle IDs to subscribe to.
        delta_time (float): Time step for the simulation in seconds.
        end_hours (int): Number of hours to run the simulation. If 0, runs until simulation is over.
        use_gui (bool): Whether to use the SUMO GUI or not.
        convert_geo (bool): Whether to convert the coordinates from SUMO to latitude/longitude.
        freeze_traffic_lights (bool): If true, traffic lights are green all the time.
    Returns:
        dict: Dictionary with vehicle variables indexed by time.
    Raises:
        Exception: If there is an error starting the SUMO simulation or subscribing to vehicles.
    """

    if use_gui:
        traci.start(["sumo-gui", "-c", f"{folder_path}/osm.sumocfg"])
    else:
        traci.start(["sumo", "-c", f"{folder_path}/osm.sumocfg"])
    v_variables = {}

    if end_hours != 0:
        end_time = f'{end_hours * 60 * (60/delta_time)}'  # 24 hours
    else:
        end_time = ''

    sumo_helper.add_xml_child(
        f'{folder_path}/osm.sumocfg', 'time', 'step-length', f'{delta_time}', replace=True)
    sumo_helper.add_xml_child(
        f'{folder_path}/osm.sumocfg', 'time', 'end', end_time, replace=True)

    # Get all traffic light IDs
    if freeze_traffic_lights:
        tls_ids = traci.trafficlight.getIDList()

        # Set all traffic lights to constant green
        for tls_id in tls_ids:
            # Get logic info
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[
                0]

            # Create a new green-only phase
            green_state = 'G' * len(logic.phases[0].state)
            green_phase = traci.trafficlight.Phase(
                duration=9999999, state=green_state)

            # Replace the logic with only this green phase
            new_logic = traci.trafficlight.Logic(
                logic.programID, logic.type, logic.currentPhaseIndex, [
                    green_phase]
            )
            traci.trafficlight.setCompleteRedYellowGreenDefinition(
                tls_id, new_logic)

    start_time = _time.time()

    time = 0
    while traci.simulation.getMinExpectedNumber() > 0:

        # Subscribe to vehicles that have justdeparted
        for veh_id in (set(traci.simulation.getDepartedIDList()) & set(veh_ids)):
            print(f"Vehicle {veh_id} has departed")
            traci.vehicle.subscribe(
                veh_id, [tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ACCELERATION, tc.VAR_ANGLE])

        results = traci.vehicle.getAllSubscriptionResults().copy()

        for veh_id in results.keys():
            # Converting from x, y sumo coordinates to lat, lon
            if convert_geo:
                x, y = results[veh_id][tc.VAR_POSITION]
                lon, lat = traci.simulation.convert_geo(x, y, fromGeo=False)
                results[veh_id]['longitude'] = lon
                results[veh_id]['latitude'] = lat

        v_variables[time] = results

        time += delta_time
        traci.simulationStep()

    traci.close()
    end_time = _time.time()

    return v_variables, (end_time - start_time)


def save_data(veh_variables, data_folder_path, delta_time, exec_time, new_dir=False, verify=True, use_lat_lon=True, speed_threshold=6, acc_threshold=6, derivative_threshold=3):
    """
    Saves the data from the simulation in a folder with the given name.
    Args:
        veh_variables (dict): Dictionary with the variables from the simulation.
        data_folder_path (str): Path of the folder where the data will be saved.
        delta_time (float): Time step of the simulation.
        new_dir (bool): If True, creates a new directory for the data.
        verify (bool): If True, verifies the data and corrects outliers.
        use_lat_lon (bool): If True, uses latitude and longitude instead of x and y coordinates.
        speed_threshold (float): Threshold for speed verification.
        acc_threshold (float): Threshold for acceleration verification.
        derivative_threshold (int): Number of timesteps to use for derivative calculations.
    Returns:
        None
    """

    if new_dir:
        if not os.path.exists(data_folder_path):
            os.mkdir(data_folder_path)
        else:
            shutil.rmtree(data_folder_path)
            os.mkdir(data_folder_path)

    for timestep, data in veh_variables.items():
        for veh_id, veh_data in data.items():

            if use_lat_lon:
                columns = 'timestamp,latitude,longitude,speed,speed_x,speed_y,acceleration,acceleration_x,acceleration_y,angle,acc_diff,gyroscope_z'
            else:
                columns = 'timestamp,x_pos,y_pos,speed,speed_x,speed_y,acc,acc_x,acc_y,angle,acc_diff,gyro_z'

            nolabel_path = f'{data_folder_path}/{veh_id}.csv'

            if not os.path.exists(nolabel_path):
                with open(nolabel_path, 'w') as f:
                    f.write(f'{columns}\n')

            write_speed = veh_data[SPEED]
            write_angle = veh_data[ANGLE]
            write_acc = veh_data[ACCELERATION]
            write_x = veh_data[POSITION][0]
            write_y = veh_data[POSITION][1]

            if verify:

                try:
                    derivative_speed = (
                        veh_variables[timestep][veh_id][SPEED] - veh_variables[timestep-derivative_threshold][veh_id][SPEED]) / derivative_threshold
                    derivative_acceleration = (
                        veh_variables[timestep][veh_id][ACCELERATION] - veh_variables[timestep-derivative_threshold][veh_id][ACCELERATION]) / derivative_threshold
                except KeyError:  # If the previous timestep does not exist
                    derivative_speed = 0
                    derivative_acceleration = 0

                # Making verification to ensure there are no outliers
                if derivative_speed > speed_threshold or derivative_speed < -speed_threshold:
                    last_speed = veh_variables[timestep -
                                               delta_time][veh_id][SPEED]
                    print(
                        f'Vehicle {veh_id} at timestep {timestep} had a speed of {veh_data[SPEED]}, it was changed to {last_speed}')
                    veh_variables[timestep][veh_id][SPEED] = last_speed
                    write_speed = last_speed

                if derivative_acceleration > acc_threshold or derivative_acceleration < -acc_threshold:
                    last_acc = veh_variables[timestep -
                                             delta_time][veh_id][ACCELERATION]
                    print(
                        f'Vehicle {veh_id} at timestep {timestep} had an acceleration of {veh_data[ACCELERATION]}, it was changed to {last_acc}')
                    veh_variables[timestep][veh_id][ACCELERATION] = last_acc
                    write_acc = last_acc

                if veh_data[ANGLE] < 0 or veh_data[ANGLE] > 360:
                    last_angle = veh_variables[timestep -
                                               delta_time][veh_id][ANGLE]
                    print(
                        f'Vehicle {veh_id} at timestep {timestep} had an angle of {veh_data[ANGLE]}, it was changed to {last_angle}')
                    veh_variables[timestep][veh_id][ANGLE] = last_angle
                    write_angle = last_angle

            if write_speed < -100:  # Invalid values
                write_speed = 0
                print(
                    f'Vehicle {veh_id} at timestep {timestep} had a INVALID speed of {veh_data[SPEED]}, it was changed to 0')
            if write_acc < -100 or write_acc > 100:  # Invalid values
                write_acc = 0
                print(
                    f'Vehicle {veh_id} at timestep {timestep} had a INVALID acceleration of {veh_data[ACCELERATION]}, it was changed to 0')

            # Calculating the decomposed acceleration and speed
            write_speed_x = write_speed * np.cos(np.radians(write_angle))
            write_speed_y = write_speed * np.sin(np.radians(write_angle))
            write_acc_x = write_acc * np.cos(np.radians(write_angle))
            write_acc_y = write_acc * np.sin(np.radians(write_angle))

            try:
                acc_diff = np.abs((veh_variables[timestep][veh_id][ACCELERATION] -
                                   veh_variables[timestep - delta_time][veh_id][ACCELERATION]) / delta_time)

                current_angle = np.radians(
                    veh_variables[timestep][veh_id][ANGLE])
                previous_angle = np.radians(
                    veh_variables[timestep - delta_time][veh_id][ANGLE])

                # Proper angle wrapping to handle transitions like 359° -> 0°
                angle_diff = np.arctan2(np.sin(current_angle - previous_angle),
                                        np.cos(current_angle - previous_angle))

                gyroscope_z = angle_diff / delta_time  # radians/second
            except KeyError:  # If the previous timestep does not exist
                acc_diff = np.abs(write_acc)
                gyroscope_z = 0.0

            if use_lat_lon:
                line = f'{timestep},{veh_data["latitude"]},{veh_data["longitude"]},{write_speed},{write_speed_x},{write_speed_y},{write_acc},{write_acc_x},{write_acc_y},{write_angle},{acc_diff},{gyroscope_z}'
            else:
                line = f'{timestep},{write_x},{write_y},{write_speed},{write_speed_x},{write_speed_y},{write_acc},{write_acc_x},{write_acc_y},{write_angle},{acc_diff},{gyroscope_z}'

            with open(nolabel_path, 'a') as f:
                f.write(f'{line}\n')

    print(
        f"Data saved in {data_folder_path} with delta time {delta_time} seconds.")

    # Save metadata to a CSV file
    metadata = {
        "delta_time": delta_time,
        "verify": verify,
        "use_lat_lon": use_lat_lon,
        "speed_threshold": speed_threshold,
        "acc_threshold": acc_threshold,
        "derivative_threshold": derivative_threshold,
        "vehicle_ids": list(veh_variables[list(veh_variables.keys())[0]].keys()) if veh_variables else [],
        "timesteps": len(veh_variables),
        "execution_time_sec": exec_time,
    }

    metadata_path = os.path.join(data_folder_path, "metadata.csv")
    with open(metadata_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for key, value in metadata.items():
            writer.writerow([key, value])
    print(f"Metadata saved in {metadata_path}")
