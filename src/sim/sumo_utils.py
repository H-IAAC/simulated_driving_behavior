from typing import Union
import sys

import random
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from . import osm_utils


def flush_print():
    sys.stdout.write("\r")  # Move the cursor to the beginning of the line
    sys.stdout.write(" " * 50)  # Overwrite with spaces to clear the line
    sys.stdout.write("\r")  # Move back to the beginning again


def has_parking_spot(lanes: list, parking_areas: list) -> Union[str, None]:
    """ Checks if there is a parking spot in the given lanes.
    Args:
        lanes (list): List of lane objects to check for parking spots.
        parking_areas (list): List of parking area objects to check against.
    Returns:
        str or bool: Returns the parking area id if a parking spot is found in the lanes, otherwise returns False.
    """

    # Example of parkingArea:
    # <parkingArea id="pa-1046248579#0" lane="-1046248579#0_0" roadsideCapacity="94" length="5.00"/>

    # Returns parkingArea id if there is a parking spot in the lane
    lane_ids = [lane.getID() for lane in lanes]
    for park in parking_areas:
        if park.lane in lane_ids:
            return park.id

    return None


def get_closest_edges(net: object, lat: float, lon: float, radius: float, max_edges: int = 10, convert_to_xy: bool = True) -> list:
    """ Gets the closest edges to the given latitude and longitude within a specified radius.
    Args:
        net (object): The SUMO network object.
        lat (float): Latitude of the point of interest.
        lon (float): Longitude of the point of interest.
        radius (float): Radius in meters to search for edges.
        max_edges (int): Maximum number of edges to return.
        convert_to_xy (bool): Whether to convert latitude and longitude to XY coordinates. Used when the network is in XY coordinates, e.g., not from Open Street Maps.
    Returns:
        list: A list of the closest edges that allow passenger cars, or None if no edges are found.
    """
    # SUMO uses XY coordinates to find the closest edges, so if the network is in lat, lon format, it converts the coordinates to XY.
    if convert_to_xy:
        x, y = net.convertLonLat2XY(lon, lat)
    else:
        x, y = lon, lat

    edges = net.getNeighboringEdges(x, y, radius)
    closest_edges = []
    if (len(edges) > 0):
        distance_edges = sorted([(dist, edge)
                                for edge, dist in edges], key=lambda x: x[0])

        # Checking if the edge found can be used by passenger car
        for dist, edge in distance_edges[:max_edges]:
            if edge.allows('passenger'):
                closest_edges.append(edge)

    if len(edges) == 0:
        print(
            f'No edges found for {x}, {y}. Perhaps location is not inside the network or there are no viable roads inside the radius.')
        return []

    return closest_edges


def get_parking_spot(net: object, lat: float, lon: float, radius: int, parking_areas: list) -> Union[str, None]:
    """ Gets the parking spot closest to the given latitude and longitude within a specified radius.
    Args:
        net (object): The SUMO network object.
        lat (float): Latitude of the point of interest.
        lon (float): Longitude of the point of interest.
        radius (float): Radius in meters to search for parking spots.
        parking_areas (list): List of parking area objects to check against.
    Returns:
        str or None: Returns the parking area id if a parking spot is found, otherwise returns None.
    """

    # Get the parking spot closest to the given lat, lon
    # Used to set stops for the vehicles

    edges = get_closest_edges(net, lat, lon, radius)

    # Look for parking spots
    for i in range(len(edges)):
        parking_spot = has_parking_spot(edges[i].getLanes(), parking_areas)
        if parking_spot:
            return parking_spot
    print(
        f"No parking spot found close to {lat}, {lon}. Perhaps decrease the radius?")

    return None


def get_path(location_time_list: dict, net: object, steps_per_stop: int, parking_areas: list = None, radius: int = 100, use_carla_routine: bool = False) -> tuple:
    """ Creates a path for the trip based on the locations and parking areas.
        All that is needed to create the trip are the stops and the start and end edges.
        The duarouter is responsible for finding the path between the edges going through the stops.

    Args:
        location_time_list (dict): A dictionary with the locations and their respective times.
        net (object): The SUMO network object.
        steps_per_stop (int): The number of simulation steps that the vehicle will stay at each stop.
        radius (int): The radius in meters to search for parking spots.
        parking_areas (list, optional): A list of parking area objects to check against.
        use_carla_routine (bool): If True, the path is treated as a sequence of edges without parking areas.
    Returns:
        tuple: A tuple containing the path (list of edges and parking spots) and the stop durations (list of integers).
    """

    # 'coordinates' is a list of tuples with the latitude and longitude of the points of interest, for example IC, FEEC, IC means that
    # the vehicle will start from IC, stop at a parking lot close to FEEC, and then back to IC.

    if parking_areas is not None:
        if use_carla_routine:
            print(
                "Parking areas provided, but will not use them to find parking spots between locations. Cannot use them with the carla routine.")
        else:
            print(
                "Parking areas provided, will use them to find parking spots between locations.")

    stop_durations = []
    departures = list(location_time_list.keys())

    # Indicates the first place is an edge and not a parking spot
    stop_durations.append(-1)

    path = []
    coords = [location_time_list[k]['coords']
              for k in location_time_list.keys()]

    convert_to_xy = True  # Coordinates are in lat, lon format
    if use_carla_routine:
        convert_to_xy = False  # Routines from CARLA are already in xy coordinates

    home = get_closest_edges(
        net, *coords[0], radius, convert_to_xy=convert_to_xy)[0].getID()

    path.append(home)

    for i in range(1, len(coords)-1):

        stop_durations.append(
            steps_per_stop * (departures[i + 1] - departures[i]))

        if parking_areas is None:  # When using the carla routine, we don't have parking spots
            ps = get_closest_edges(
                net, *coords[i], radius, convert_to_xy=convert_to_xy)[0].getID()
        else:
            ps = get_parking_spot(net, *coords[i], radius, parking_areas)

        if ps is not None:
            path.append(ps)

        else:
            raise ValueError(
                f"Could not find parking spot for {coords[i]}. Maybe there are none in the radius of {radius} meters?")

    path.append(home)
    # Indicates the last place is an edge and not a parking spot
    stop_durations.append(-1)

    return path, stop_durations


def osm_search(local: str, center_lat: float, certer_lon: float, start_radius: int, step_radius: int, limit_radius: int, n_options: int = 3, names: dict = None) -> Union[dict, None]:
    """ Searches for a location in the OpenStreetMap API.
    Args:
        local (str): The name of the location to search for.
        center_lat (float): Latitude of the search center.
        certer_lon (float): Longitude of the search center.
        start_radius (int): The initial radius in meters for searching locations.
        step_radius (int): The amount in meters to increase the search radius if the location is not found.
        limit_radius (int): The maximum radius in meters for searching locations.
        n_options (int): The number of options to find for each location.
        names (dict, optional): A dictionary to store names of locations. Defaults to None.
    Returns:
        dict or None: Returns a dictionary with the latitude and longitude of the location if found,
                      otherwise returns None.
    """

    found = False
    print(f"Looking for {local}...", end='', flush=True)
    # Storing the previous location to use as a reference for the next search

    names[local] = local
    result = osm_utils.find_nearby_building(
        center_lat, certer_lon, local, radius=start_radius)

    expanded = start_radius

    if len(result) > 0:  # Found at least one option
        found = True

    # If the location is not found or there are less optios than expected, expand the search radius
    while len(result) == 0 or len(result) < n_options:
        expanded += step_radius
        if expanded > limit_radius:
            break

        result = osm_utils.find_nearby_building(
            center_lat, certer_lon, local, radius=expanded)

        if found == False and (len(result) > 0):
            found = True  # Found at least one option, but will keep looking for more until the limit is reached

    if found == False:
        return None

    else:
        # Randomize the results to avoid always getting the absolute closest building
        random.shuffle(result)
        return result[0]


def get_coords(trip: dict, sulfix: str, institutes: list, center_lat: float, certer_lon: float, home_lat: float, home_lon: float, start_radius: int, step_radius: int, limit_radius: int, n_options: int = 3) -> tuple:
    """
    Gets the coordinates of the locations in the trip.
    Args:
        trip (dict): A dictionary containing the trip location fror each timestep.
        sulfixo (str): A suffix to be added to every location name for better search results, such as SP, Brazil.
        institutes (list): A list of institute names to be used for geocoding.
        start_radius (int): The initial radius in meters for searching locations.
        step_radius (int): The amount in meters to increase the search radius if the location is not found.
        limit_radius (int): The maximum radius in meters for searching locations.
        center_lat (float): Latitude of the search center.
        certer_lon (float): Longitude of the search center.
        n_options (int): The number of options to find for each location.
        restaurants (list, optional): A list of restaurant names to be used for geocoding. Defaults to None.
    Returns:
        tuple: A tuple containing two dictionaries:
            - coords: A dictionary with the latitude and longitude of the locations of interest.
            - names: A dictionary with the names of the locations.
    """

    # The suffix is the name of the state, city and neighborhood that will be added to the end of each location to improve the search
    # 'start_radius' is the initial radius of the search, 'step_radius' is the amount that will be added to the radius if the location is not found and 'limit_radius' is the maximum radius that will be used. After that, the student will choose not to leave the place he is at.
    # 'n_options' is the number of options of places we ideally want to find to choose from. This only applies while the limit_radius is not exceeded

    coords = {}  # Coordinates for every place the student will visit
    names = {}
    for i in range(len(trip)):
        local = trip[f'{i + 7}']['location']
        local_comp = local + ", " + sulfix

        if local in coords.keys():  # If the location is already in the dictionary, use the coordinates from there
            lat, lon = coords[local]
            continue

        if local == 'home':  # If the location is home, use the home coordinates
            coords['home'] = (home_lat, home_lon)
            names['home'] = 'home'
            continue

        # If the location is an institute, use the coordinates from the API
        elif local in institutes:
            result = osm_utils.geocode_address(local_comp)

            if not result:
                raise ValueError(
                    f"Could not get coordinates for {local}, maybe its name is not correct")

            name = local
            lat, lon = result[0]['latitude'], result[0]['longitude']
            print(f"\033[1mFound {local} at {lat}, {lon}.\033[0m")

        else:  # If the location is not an institute, we have to search for it in the OSM API
            result = osm_search(local, center_lat, certer_lon, start_radius,
                                step_radius, limit_radius, n_options=n_options, names=names)

            if not result:
                flush_print()
                print(
                    f"Could not find {local} in a radius of {limit_radius} meters. The student will not leave the place he is currently at.")

            else:
                lat, lon = result[0]['latitude'], result[0]['longitude']
                name = result[0]['name']
                flush_print()
                print(
                    f"Found {len(result)} options for {local}: {[x['name'] for x in result]}")
                print(
                    f"\033[1m{local} picked: {result[0]['name']} at {lat}, {lon}.\033[0m")

        coords[f'{local}'] = (lat, lon)
        names[f'{local}'] = name

    return coords, names


def coords_to_trip(trip: dict, coords: dict, names: dict = None) -> dict:
    """
    Converts the trip dictionary to a dictionary with locations and their coordinates.
    Args:
        trip (dict): A dictionary containing the trip location for each timestep.
        coords (dict): A dictionary with the latitude and longitude of the locations of interest.
        names (dict, optional): A dictionary with the names of the locations. Defaults to None.
    Returns:
        dict: A dictionary with the locations and their coordinates.
    """

    location_time = {}
    # The first location is always home
    location_time[7] = {}
    location_time[7]['location'] = 'home'
    location_time[7]['coords'] = coords['home']
    location_time[7]['name'] = 'home'
    last = coords['home']

    for j in range(1, len(trip)):
        location = trip[f'{j + 7}']['location']
        location_coords = coords[location]

        if names:
            location_names = names[location]
        else:
            location_names = location

        if location_coords != last:
            # If the coordinates are the same as the last one, skip this location
            location_time[j + 7] = {}
            location_time[j + 7]['location'] = location
            location_time[j + 7]['coords'] = location_coords
            location_time[j + 7]['name'] = location_names
            last = location_coords

    return location_time


def randomtrips_get_args(net_path: str, output_path: str, end_time: int, departure_step: int, additional_path: str = None) -> list:
    """
    Generates the arguments for the randomTrips.py script.
    Args:
        net_path (str): Path to the SUMO network file.
        additional_path (str): Path to the additional files for SUMO.
        output_path (str): Path where the generated trips will be saved.
        end_time (int): The end time for the simulation.
        departure_step (int): The step interval for departures.
    Returns:
        list: A list of arguments.
    """
    # Generates the arguments for the randomTrips.py script
    args = [
        "-n", net_path,
        "-r", f"{output_path}",
        "-o", f"{'/'.join(output_path.split('/')[:-1])}/{output_path.split('/')[-1].split('.')[0].split('.')[0]}.trips.xml",
        "-e", str(end_time),
        "-p", str(departure_step),
        "--validate",
    ]
    if additional_path:
        args += ["--additional", additional_path]

    return args


def get_random_trips(net_path: str,  end_time: int, vtype_data: dict, output_file_path: str, departure_step: int = 10, add_path: str = None) -> dict:
    """
    Generates random trips for a SUMO simulation and writes to output_file_path.
    Args:
        net_path (str): Path to the SUMO network file.
        add_path (str): Path to the additional files for SUMO.
        end_time (int): The end time for the simulation.
        vtype_data (dict): A dictionary containing vehicle type data, where keys are class names
                            and values are lists of vehicle type IDs.
        departure_step (int): The step interval for departures.
        output_file_path (str): Path where the generated trips will be saved.
    Returns:
        dict: A dictionary where keys are class names and values are lists of vehicle IDs for each class.
    Raises:
        subprocess.CalledProcessError: If the randomTrips.py script fails to execute.
    """

    # Generates random trips with the given number of trips per class and vehicle types
    # Writes the trips to a <output_file_name> and returns the vehicle ids for each class
    randomtrips_path = "/usr/share/sumo/tools/randomTrips.py"
    subprocess.run(['python3', randomtrips_path] + randomtrips_get_args(net_path,
                   output_file_path, end_time, departure_step, additional_path=add_path), check=True)  # Generates random trips without vtype

    tree = ET.parse(output_file_path)
    root = tree.getroot()
    vehicles = root.findall('vehicle')

    print(f"Generated {len(vehicles)} vehicles.")

    n_trips_per_class = len(vehicles) // len(vtype_data)

    type_id = {}
    index = 0
    cls = list(vtype_data.keys())[0]
    type_id[cls] = []

    for i in range(len(vehicles)):

        # Guarantees that the number of vehicles of each type is respected
        if i < (n_trips_per_class * len(vtype_data)) and i >= n_trips_per_class * (index + 1):
            index += 1
            cls = list(vtype_data.keys())[index]
            type_id[cls] = []

        # Choosing a random vehicle type from that class

        rng = np.random.default_rng(seed=42)  # For reproducibility
        rd = rng.integers(0, len(vtype_data[cls]))
        type_id[cls].append(vehicles[i].get('id'))  # Storing the vehicle id

        # Setting the vehicle type
        vehicles[i].set('type', vtype_data[cls][rd])

    tree.write(output_file_path, encoding="UTF-8",
               xml_declaration=True, method="xml")

    return type_id


def get_coords_from_spawnpoints(routine_points: dict, net_offset: tuple) -> dict:
    """
    Get the edges corresponding to the spawn points of the vehicles in CARLA.
    The calculation is done by getting the closest edge to the spawn point.
    Args:
        routine_points (dict): A dictionary with the routine spawnpoints.
    Returns:
        dict: A dictionary with the coordinates of the routine points that SUMO can use.
    """
    carla_routine_coords = {}
    for name, transf in routine_points.items():
        loc = transf.location
        x, y = loc.x + net_offset[0], -loc.y + net_offset[1]
        # Coordinates are reversed to match the SUMO coordinates
        carla_routine_coords[name] = (y, x)

    return carla_routine_coords


def draw_debug_points(coords: dict, output_path: str, size: int = 5):
    """ Writes the XML file to draw debug points at the coordinates.
    Args:
        coords (dict): A dictionary with the coordinates of the routine points.
        path (str): The path to the XML file to be created.
        size (int): The size of the points to be drawn.
    Returns:
        None
    """

    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<additional>\n'

    for name, coord in coords.items():
        xml += f'<poi id="{name}" x="{coord[1]}" y="{coord[0]}" color="0,255,0" layer="30"/>\n'

    xml += '</additional>\n'

    with open(output_path, 'w') as f:
        f.write(xml)


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

    rng = np.random.default_rng(seed=42)
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
