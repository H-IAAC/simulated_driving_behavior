# from agents.navigation.global_route_planner import GlobalRoutePlanner
import sys
import os
import time as _time

import carla
import numpy as np
import pandas as pd
import csv
import scipy.stats as stats

# sys.path.append('/opt/carla-simulator/PythonAPI/carla')


def imu_callback(data):
    """Callback function to handle IMU sensor data.
    Args:
        data: The IMU sensor data containing timestamp, accelerometer, gyroscope, and compass values.
    """

    global imu_df
    timestamp = data.timestamp
    acc = data.accelerometer
    gyro = data.gyroscope
    compass = data.compass

    values = [[timestamp, acc.x, acc.y, acc.z,
               gyro.x, gyro.y, gyro.z, np.rad2deg(compass)]]
    new_data = pd.DataFrame(values, columns=imu_df.columns)

    imu_df = pd.concat([imu_df, new_data], ignore_index=True)


def set_attributes_imu(imu_bp, sensor_tick=None, acc_noise=0.000, gyro_std=0.000, gyro_mean=0.000):
    """Set attributes for the IMU sensor blueprint.

    Args:
        imu_bp: The IMU sensor blueprint.
        sensor_tick: Time in seconds between sensor captures (optional).
        acc_noise: Standard deviation for acceleration noise (default is 0.000).
        gyro_std: Standard deviation for gyroscope noise (default is 0.000).
        gyro_mean: Mean for gyroscope noise (default is 0.000).
    """

    if sensor_tick is not None:
        # Time in seconds between sensor captures.
        imu_bp.set_attribute('sensor_tick', f'{sensor_tick}')

    imu_bp.set_attribute('noise_accel_stddev_x', f'{acc_noise}')
    imu_bp.set_attribute('noise_accel_stddev_y', f'{acc_noise}')
    imu_bp.set_attribute('noise_accel_stddev_z', f'{acc_noise}')
    imu_bp.set_attribute('noise_gyro_stddev_x', f'{gyro_std}')
    imu_bp.set_attribute('noise_gyro_stddev_y', f'{gyro_std}')
    imu_bp.set_attribute('noise_gyro_stddev_z', f'{gyro_std}')
    imu_bp.set_attribute('noise_gyro_bias_x', f'{gyro_mean}')
    imu_bp.set_attribute('noise_gyro_bias_y', f'{gyro_mean}')
    imu_bp.set_attribute('noise_gyro_bias_z', f'{gyro_mean}')


def gnss_callback(data):
    """Callback function to handle GNSS sensor data.
    Args:
        data: The GNSS sensor data containing timestamp, latitude, and longitude.
    """

    global gnss_df
    timestamp = data.timestamp
    latitude = data.latitude
    longitude = data.longitude

    values = [[timestamp, latitude, longitude]]
    new_data = pd.DataFrame(values, columns=gnss_df.columns)

    gnss_df = pd.concat([gnss_df, new_data], ignore_index=True)


def set_attributes_gnss(gnss_bp, sensor_tick=None, lat_bias=0, lat_sttdev=0, lon_bias=0, lon_stddev=0):
    """Set attributes for the GNSS sensor blueprint.
    Args:
        gnss_bp: The GNSS sensor blueprint.
        sensor_tick: Time in seconds between sensor captures (optional).
        lat_bias: Mean parameter in the noise model for latitude (default is 0).
        lat_sttdev: Standard deviation parameter in the noise model for latitude (default is 0).
        lon_bias: Mean parameter in the noise model for longitude (default is 0).
        lon_stddev: Standard deviation parameter in the noise model for longitude (default is 0).
    """

    if sensor_tick is not None:
        gnss_bp.set_attribute('sensor_tick', f'{sensor_tick}')

    gnss_bp.set_attribute('noise_lat_bias', f'{lat_bias}')
    gnss_bp.set_attribute('noise_lat_stddev', f'{lat_sttdev}')
    gnss_bp.set_attribute('noise_lon_bias', f'{lon_bias}')
    gnss_bp.set_attribute('noise_lon_stddev', f'{lon_stddev}')


def spawn_vehicles_tm(client, vehicles_bp, spawn_points, n_vehicles, dtlv=5, psd=80, hpm=True, hpr=50):
    """
    Spawn vehicles in the CARLA world using Traffic Manager.
    Args:
        client: CARLA client instance.
        vehicles_bp: List of vehicle blueprints to spawn.
        spawn_points: List of spawn points for the vehicles.
        n_vehicles: Number of vehicles to spawn.
        dtlv: Distance to leading vehicle for all TM managed vehicles.
        psd: Percentage speed difference for all TM managed vehicles.
        hpm: Hybrid physics mode. If True, simulate full physics only for vehicles near the ego vehicle (or camera).
        hpr: Radius in meters for hybrid physics mode.
    Returns:
        tm_vehs (list): List of spawned vehicles.
        tm_port (int): Traffic Manager port.
    Raises:
        RuntimeError: If no vehicles could be spawned.
    Description:
        This function spawns a specified number of vehicles in the CARLA world using the Traffic Manager.
        It sets the autopilot for each vehicle and configures the Traffic Manager with the specified parameters.
    """

    world = client.get_world()
    tm = client.get_trafficmanager()
    tm_port = tm.get_port()
    tm_vehs = []

    tm.set_synchronous_mode(True)
    tm.set_hybrid_physics_mode(hpm)

    if hpm:
        tm.set_hybrid_physics_radius(hpr)

    for _ in range(n_vehicles):
        veh = world.try_spawn_actor(np.random.choice(
            vehicles_bp), np.random.choice(spawn_points))
        if veh:
            tm_vehs.append(veh)
            veh.set_autopilot(True, tm_port)
            tm.distance_to_leading_vehicle(veh, dtlv)
            tm.vehicle_percentage_speed_difference(veh, psd)

    print(f"Spawned {len(tm_vehs)} vehicles with Traffic Manager")

    return tm_vehs, tm_port


def get_spawn_points_from_csv(csv_file):
    """Read spawn points from a CSV file and return them as a dictionary.
    Args:
        csv_file (str): Path to the CSV file containing spawn points.
    Returns:
        dict: A dictionary where keys are names and values are carla.Transform objects.
    """

    spawn_points = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            loc = carla.Location(
                float(row['x']), float(row['y']), float(row['z']))
            rot = carla.Rotation(float(row['pitch']), float(
                row['yaw']), float(row['roll']))
            spawn_points[row['Name']] = carla.Transform(loc, rot)

    return spawn_points


def destroy_all_vehicles(client):
    """Destroy all vehicles in the CARLA world.
    Args:
        client: CARLA client instance.
    """

    print("Destroying all vehicles...")
    world = client.get_world()
    # Destroy all vehicles
    for actor in world.get_actors().filter('vehicle.*'):
        actor.destroy()


def draw_route(world, route, life_time=999999):
    for i, w in enumerate(route):
        # Raise the text above the ground
        location = w.location + carla.Location(z=i * 1.0)
        world.debug.draw_string(location, f'PONTO_{i}', draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=life_time,
                                persistent_lines=True)


def generate_route(grp, desired_spawn_points):
    route = []
    for i in range(len(desired_spawn_points) - 1):
        route.append(grp.trace_route(
            desired_spawn_points[i].location, desired_spawn_points[i + 1].location))

    return route


def create_ego_vehicle(client, vehicle_bp, route_sps, draw_debug_route=True):
    """Create an ego vehicle in the CARLA world and sets its route.
    Args:
        client: CARLA client instance.
        vehicle_bp: Blueprint of the vehicle to be spawned.
        route_sps: List of spawn points to follow the route.
        draw_debug_route: If True, the route will be drawn in the world (default is True).
    Returns:
        vehicle: The spawned vehicle actor.
    """

    tm = client.get_trafficmanager()
    # Important for carlaviz and for TM
    vehicle_bp.set_attribute('role_name', 'hero')
    # Spawn the vehicle at the first spawn point
    vehicle = client.get_world().spawn_actor(vehicle_bp, route_sps[0])
    vehicle.set_autopilot(True, tm.get_port())  # Let TM control this vehicle

    # Assign the global plan to TM
    tm.set_path(vehicle, [wp.location for wp in route_sps])

    if draw_debug_route:
        world = client.get_world()
        # Draw the route in the world
        draw_route(world, route_sps)

    print(f"Ego vehicle created with ID: {vehicle.id}")

    return vehicle


def configure_tm_vehicle(tm, vehicle, config: dict):
    """
    Configure CARLA Traffic Manager parameters for a given vehicle using a config dictionary.
    Args:
        tm (carla.TrafficManager): The Traffic Manager instance.
        vehicle (carla.Actor): The vehicle actor to configure.
        config (dict): Dictionary of parameters to apply.
    """

    # Speed difference from speed limit (in percent).
    # Negative => faster than limit; Positive => slower.

    # Distance to keep from the leading vehicle (in meters).
    if 'distance_to_leading_vehicle' in config:
        tm.distance_to_leading_vehicle(
            vehicle, config['distance_to_leading_vehicle'])

    # Probability (0–100%) of ignoring traffic lights.
    if 'ignore_lights_percentage' in config:
        tm.ignore_lights_percentage(
            vehicle, config['ignore_lights_percentage'])

    # Probability (0–100%) of ignoring stop signs.
    if 'ignore_signs_percentage' in config:
        tm.ignore_signs_percentage(vehicle, config['ignore_signs_percentage'])

    # Probability (0–100%) of ignoring other vehicles.
    if 'ignore_vehicles_percentage' in config:
        tm.ignore_vehicles_percentage(
            vehicle, config['ignore_vehicles_percentage'])

    # Probability (0–100%) of randomly performing a left lane change.
    if 'random_left_lanechange_percentage' in config:
        tm.random_left_lanechange_percentage(
            vehicle, config['random_left_lanechange_percentage'])

    # Probability (0–100%) of randomly performing a right lane change.
    if 'random_right_lanechange_percentage' in config:
        tm.random_right_lanechange_percentage(
            vehicle, config['random_right_lanechange_percentage'])

    if 'vehicle_percentage_speed_difference' in config:
        # Percentage speed difference from the speed limit.
        tm.vehicle_percentage_speed_difference(
            vehicle, config['vehicle_percentage_speed_difference'])


def set_sync_mode(client, delta_time: float, render: bool):
    """Set the CARLA world to synchronous mode with a fixed delta time.
    Args:
        client: CARLA client instance.
        delta_time: Time step for the simulation (default is 0.05).
        render: If True, rendering will be enabled (default is True).
    Returns:
        settings: The world settings after applying synchronous mode.
    """

    world = client.get_world()
    tm = client.get_trafficmanager()
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enable sync mode
    settings.fixed_delta_seconds = delta_time  # Set the fixed delta time
    settings.no_rendering_mode = not render
    tm.set_synchronous_mode(True)  # Enable synchronous mode in Traffic Manager
    world.apply_settings(settings)
    print(f"Delta time set to {delta_time} seconds. Synchronous mode enabled.")
    return settings


def set_freeze_traffic_lights(client):
    """Freeze all traffic lights in the CARLA world.

    Args:
        client: CARLA client instance.
    """

    world = client.get_world()
    print("Freezing all traffic lights...")
    for tl in world.get_actors().filter('traffic.traffic_light'):
        tl.set_state(carla.TrafficLightState.Green)
        tl.freeze(True)


def set_up_sensors(client, sensors_bp, vehicle):
    """Set up sensors for the vehicle in the CARLA world.
    Args:
        client: CARLA client instance.
        sensors_bp: List of sensor blueprints to be attached to the vehicle.
        vehicle: The vehicle actor to which the sensors will be attached.
    Returns:
        sensor_actors: List of spawned sensor actors.
    """

    world = client.get_world()
    sensor_actors = []  # Hold references to sensor actors to prevent garbage collection
    print(f"Spawning {len(sensors_bp)} sensors...")
    for sensor_bp in sensors_bp:
        # Setting the sensors at the windshield of the vehicle
        sensor = world.spawn_actor(sensor_bp, carla.Transform(
            carla.Location(z=1.3, x=0.3)), attach_to=vehicle)
        sensor.set_simulate_physics(False)
        sensor_actors.append(sensor)
    print(f"Spawned {len(sensor_actors)} sensors.")

    for sensor in sensor_actors:
        if sensor.type_id == 'sensor.other.imu':
            print("IMU sensor attached.")
            sensor.listen(lambda data: imu_callback(data))
        elif sensor.type_id == 'sensor.other.gnss':
            print("GNSS sensor attached.")
            sensor.listen(lambda data: gnss_callback(data))

    return sensor_actors


def goal_reached(vehicle, end_location, threshold=5):
    """Check if the vehicle has reached the goal location within a threshold.

    Args:
        vehicle: The vehicle actor.
        end_location: The target location to reach.
        threshold: Distance threshold to consider the goal reached (default is 1.0 meters).

    Returns:
        bool: True if the vehicle is within the threshold of the goal location, False otherwise.
    """
    current_location = vehicle.get_location()
    distance = current_location.distance(end_location)
    return distance < threshold


def follow_route(client, vehicle_bp, sensors_bp, route_sps: list, agent_params: dict, delta_time: float = 0.01, freeze_traffic_lights: bool = False, n_extra_vehicles: int = 30, fixed_spectator: bool = True, draw_debug_route: bool = True, render: bool = True) -> None:
    """
    Follow the route using the vehicle.
    Args:
        client: CARLA client instance.
        vehicle_bp: Blueprint of the vehicle to be spawned.
        sensors_bp: List of sensor blueprints to be attached to the vehicle.
        route_sps: List of spawn points to follow the route.
        agent_params: Dictionary of parameters for the agent.
        delta_time: Time step for the simulation (default is 0.05).
        stop_time: How much time to stop at each waypoint (default is 0).
        n_extra_vehicles: Number of vehicles to spawn (default is 30).
        fixed_spectator: If True, the spectator will follow the vehicle (default is True).
        draw_debug_route: If True, the route will be drawn in the world (default is True).
        render: If True, rendering will be enabled (default is True).
    """
    world = client.get_world()
    # Set up spectator
    spectator = world.get_spectator()
    spectator.set_transform(route_sps[0])

    if freeze_traffic_lights:
        set_freeze_traffic_lights(client)

    # Set synchronous mode with the specified delta time
    settings = set_sync_mode(client, delta_time, render)

    vehicle = create_ego_vehicle(
        client, vehicle_bp, route_sps, draw_debug_route)  # Create the ego vehicle and set its route

    # Configure Traffic Manager for the vehicle
    configure_tm_vehicle(client.get_trafficmanager(), vehicle, agent_params)

    sensor_actors = set_up_sensors(
        client, sensors_bp, vehicle)  # Setting up sensors

    # Initializing other cars in the simulation
    if n_extra_vehicles > 0:
        spawn_vehicles_tm(client, world.get_blueprint_library().filter(
            'vehicle.*'), world.get_map().get_spawn_points(), n_vehicles=n_extra_vehicles, dtlv=5, psd=-0, hpm=True, hpr=50)

    checkpoints_hit = 0
    stuck_counter = 0
    try:
        while True:

            # Checking if vehicle is stuck
            velocity = vehicle.get_velocity()
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            if speed < 0.1:
                stuck_counter += 1
                # If the vehicle is stuck for more than 20 seconds
                if stuck_counter > (1 / delta_time) * 20:
                    print(
                        "Vehicle is stuck. Stopping this routine early, saving the data, and moving to the next one.")
                    break
            else:
                stuck_counter = 0

            # Check if vehicle has reached its goal
            if goal_reached(vehicle, route_sps[checkpoints_hit + 1].location):
                if checkpoints_hit + 1 == len(route_sps) - 1:
                    print("Vehicle has reached the final destination.")
                    break

                else:
                    checkpoints_hit += 1
                    print(f"Checkpoint {checkpoints_hit} reached!")

            # Update spectator position
            if fixed_spectator:
                # Spectator follows the vehicle
                spectator.set_transform(carla.Transform(vehicle.get_transform(
                ).location + carla.Location(z=40), carla.Rotation(pitch=-90)))

            world.tick()

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")

    finally:
        for sensor in sensor_actors:
            sensor.destroy()

        destroy_all_vehicles(client)
        print("All actors destroyed.")

        # Disable sync mode
        settings.synchronous_mode = False
        world.apply_settings(settings)


def save_data_to_csv(veh_id: str, imu_df: pd.DataFrame, gnss_df: pd.DataFrame, folder_path: str) -> None:
    """Save IMU and GNSS data to CSV files.
    Args:
        veh_id: Identifier for the vehicle.
        imu_df: DataFrame containing IMU data.
        gnss_df: DataFrame containing GNSS data.
        folder_path: Path to the folder where the CSV files will be saved.
    """

    whole_df = pd.DataFrame()
    whole_df = pd.merge(imu_df, gnss_df, on='timestamp',
                        how='outer', validate='one_to_one')[3:]
    whole_df['timestamp'] = whole_df['timestamp'] - \
        whole_df['timestamp'].iloc[0]  # Normalize time to start from
    os.makedirs(folder_path, exist_ok=True)
    whole_df.to_csv(f'{folder_path}/{veh_id}.csv', index=False)


def run_simulation(client, sim_params: dict, sps_routines: list, output_folder: str) -> None:
    """ Run the simulation with the specified parameters.

    Args:
        client: CARLA client instance.
        sim_params: Dictionary containing simulation parameters.
            - vehicle_bp: Blueprint of the vehicle to be spawned.
            - sensor_bps: List of sensor blueprints to be attached to the vehicle.
            - agent_params: Dictionary of parameters for the agent behaviors.
            - freeze_traffic_lights: If True, traffic lights will be frozen to green.
            - n_extra_vehicles: Number of extra vehicles to spawn in the simulation.
            - delta_time: Time step for the simulation (0.01 is 100 Hz).
            - fixed_spectator: If True, the spectator will follow the vehicle.
            - draw_debug_route: If True, the route will be drawn in the world.
            - render: If True, rendering will be enabled.
        sps_routines: List of routes to follow in the simulation.
        output_folder: Folder where the simulation data will be saved.

    Description:
        This function runs the simulation by spawning vehicles, setting up sensors, and following the specified routes.
        It saves the IMU and GNSS data to CSV files for each vehicle.

    Raises:
        KeyboardInterrupt: If the simulation is interrupted by the user.
    """
    end_times = []
    agent_params = sim_params['agent_params']
    for beh in agent_params.keys():
        print(f"Performing {len(sps_routines)} routines for behavior: {beh}")
        for i, sps in enumerate(sps_routines):
            print(
                f"\n------------------ ROUTINE {i} - BEHAVIOR: {beh} ---------------------------\n")
            try:
                # timestamp: simulation time - s
                # acc: m/s^2
                # gyro: rad/s
                # compass: degrees
                global imu_df, gnss_df
                imu_df = pd.DataFrame(columns=[
                                      'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'compass'])

                # orientation: rad
                gnss_df = pd.DataFrame(
                    columns=['timestamp', 'latitude', 'longitude'])

                agent_behavior = beh
                id = f'veh_{i}_{agent_behavior}'

                start_time = _time.time()
                # Here traffic lights are frozen to green in order to reduce simulation time
                follow_route(
                    client,
                    vehicle_bp=sim_params['vehicle_bp'],
                    sensors_bp=sim_params['sensor_bps'],
                    route_sps=sps,
                    agent_params=agent_params[beh],
                    freeze_traffic_lights=sim_params['freeze_traffic_lights'],
                    n_extra_vehicles=sim_params['n_extra_vehicles'],
                    delta_time=sim_params['delta_time'],
                    fixed_spectator=sim_params['fixed_spectator'],
                    draw_debug_route=sim_params['draw_debug_route'],
                    render=sim_params['render'])

                end_times.append(_time.time() - start_time)

                save_data_to_csv(
                    id, imu_df, gnss_df, output_folder)

                print(
                    f"\n------------------ VEHICLE {id} FINISHED ---------------------------\n")

            except KeyboardInterrupt as e:
                print(f"Simulation interrupted by user. Error: {e}")
                break

    print("Simulation finished. All data saved to CSV files.")

    with open(os.path.join(output_folder, "metadata.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for key, value in sim_params.items():
            writer.writerow([key, value])
        writer.writerow(
            ["execution_time_sec", sum(end_times) / len(end_times)])
    print(f"Metadata saved in {os.path.join(output_folder, 'metadata.csv')}")


def get_param_value(param_dict: dict, parameter: str, style: str) -> tuple:
    """ Generates a random value for a given parameter based on a Gaussian distribution defined in the param_dict.
    Args:
        param_dict (dict): A dictionary containing parameters for each style.
        parameter (str): The parameter for which to generate a value.
        style (str): The style of the vehicle (e.g., 'agg', 'norm').
    Returns:
        tuple: A tuple containing the generated value and its probability.
    """

    m = (param_dict[parameter][style]['min'] +
         param_dict[parameter][style]['max'])/2
    # Finding the standard deviation for 95% of the data to be within the range
    s = (param_dict[parameter][style]['max'] - m) / stats.norm.ppf(0.975)

    rng = np.random.default_rng()
    value = np.round(rng.normal(m, s), 2)

    if s <= 0:
        print(f"Error: The standard deviation for {parameter} is {s}")

    cdf = stats.norm.cdf(value, loc=m, scale=s)
    if value > m:
        probability = 1 - cdf
    else:
        probability = cdf

    return value, probability


def generate_vehicle_types(param_dict: dict, styles: list, n: int) -> dict:
    """
    Generates vehicle types based on the given parameter dictionary and styles and assigns a probability to each vType based on how likely it is to be real.
    Args:
        param_dict (dict): A dictionary containing parameters for each style.
        styles (list): A list of styles to be generated (e.g., 'agg', 'norm').
        n (int): The number of vehicle types to generate for each style.
    Returns:
        dict: A dictionary containing the generated vehicle types for each style.
    """
    vtypes_dist = {}
    # Keeps the probability score for each of the generated vTypes
    param_probs = np.zeros(n)
    for style in styles:
        vtypes_dist[f'veh_{style}'] = {}
        for i in range(n):
            vtypes_dist[f'veh_{style}'][f'veh_{style}{i}'] = {}
            prob = 0
            for parameter in list(param_dict.keys()):
                # Gets value for parameter and the probability of getting that value
                value, probability = get_param_value(
                    param_dict, parameter, style)
                vtypes_dist[f'veh_{style}'][f'veh_{style}{i}'][parameter] = float(
                    value)
                prob += probability  # Sum of probabilities for each parameter

            param_probs[i] = prob

        # Softmax function to normalize the probabilities
        softm = np.exp(param_probs) / np.sum(np.exp(param_probs))

        for i in range(n):
            # Assigning the normalized probability to each vType
            vtypes_dist[f'veh_{style}'][f'veh_{style}{i}']["probability"] = softm[i]

    return vtypes_dist


def get_random_types(vtypes_dist, behaviors):

    dicts = []
    for behavior in behaviors:
        rd = np.random.randint(0, len(vtypes_dist[f'veh_{behavior}']))
        dicts.append(vtypes_dist[f'veh_{behavior}'][f'veh_{behavior}{rd}'])

        print(
            f"Selected vehicle type **veh_{behavior}{rd}** for behavior **{behavior}**")

    return dicts[0], dicts[1]
