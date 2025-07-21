import os
import shutil
import re

from pandas import DataFrame
import csv
import carla
from pandas import DataFrame, read_csv
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from xml.etree.ElementTree import tostring


def add_xml_child(file_path: str, parent_tag: str, child_tag: str, child_value: str, replace: bool = True) -> bool:
    """
    Adds a new child parameter inside a specified parent tag in the XML configuration file.
    If the parent tag does not exist, it creates a new parent tag (<parameter>) with the child.
    It also checks if the child element already exists to prevent duplicates.

    Args:
        file_path (str): Path to the XML configuration file.
        parent_tag (str): The parent tag under which to add the child (e.g., 'input').
        child_tag (str): The child tag to add (e.g., 'additional-files').
        child_value (str): The value to set for the new child tag.
        replace (bool): If True, replaces the existing child tag with the new value.
                        If False, adds another child value.
    Returns:
        bool: True if the addition was successful, False otherwise.
    """
    import xml.dom.minidom

    def pretty_write(tree, file_path):
        # Convert ElementTree to string, then pretty print and write
        rough_string = ET.tostring(tree.getroot(), encoding='utf-8')
        reparsed = xml.dom.minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        # Remove blank lines
        pretty_xml = "\n".join(
            [line for line in pretty_xml.splitlines() if line.strip()])
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Find the parent element by tag
        parent_elem = root.find(parent_tag)
        if parent_elem is None:
            print(
                f"Parent tag '{parent_tag}' not found. Creating new parent tag.")
            parent_elem = ET.Element(parent_tag)
            root.append(parent_elem)
            print(f"Created new parent tag <{parent_tag}>.")

        # Check if the child element already exists inside the parent element
        existing_child = parent_elem.find(child_tag)
        if existing_child is not None:

            if existing_child.get('value') == child_value:
                print(
                    f"Child <{child_tag}> with value '{child_value}' already exists. Skipping addition.")
                return False
            else:
                if replace:
                    print(
                        f"Child <{child_tag}> already exists. Updating value to '{child_value}'.")
                    existing_child.set('value', child_value)
                else:
                    if child_value in existing_child.get('value').split(', '):
                        print(
                            f"Child <{child_tag}> with value '{child_value}' already exists. Skipping addition.")
                        return False

                    print(
                        f"Child <{child_tag}> already exists. Adding another child with value '{child_value}'.")
                    existing_child.set(
                        'value', f'{existing_child.get("value")}, {child_value}')

                pretty_write(tree, file_path)
                print("XML file updated and formatted successfully.")
                return True

        # Create the new child element and set its value
        new_child = ET.Element(child_tag)
        new_child.set('value', child_value)
        print(f"Created <{child_tag}> with value '{child_value}'.")

        # Add the new child to the parent element
        parent_elem.append(new_child)
        print(f"Added <{child_tag}> to <{parent_tag}>.")

        # Write the updated XML to the file with pretty formatting
        pretty_write(tree, file_path)
        print("XML file updated and formatted successfully.")
        return True

    except ET.ParseError as e:
        print(f"XML Parsing error: {e}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def add_missing_vtypes(routes_file: str, types_file: str, output_file: str) -> bool:
    """
    Adds missing vehicle types from types_file to routes_file.
    If a vehicle type in types_file is not present in routes_file, it will be added.
    The updated routes_file will be saved to output_file.
    Args:
        routes_file (str): Path to the SUMO routes file (XML).
        types_file (str): Path to the SUMO types file (XML).
        output_file (str): Path to save the updated routes file.
    Returns:
        bool: True if the operation was successful, False otherwise.
    """

    # Parse the XML files
    routes_tree = ET.parse(routes_file)
    routes_root = routes_tree.getroot()
    types_tree = ET.parse(types_file)
    types_root = types_tree.getroot()

    # Extract existing vType IDs in routes_file
    existing_vtypes = {vtype.get('id')
                       for vtype in routes_root.findall('vType')}

    # Find vTypes in types_file that are not in routes_file
    for vtype in types_root.findall(".//vType"):
        if vtype.get('id') not in existing_vtypes:
            routes_root.insert(0, vtype)

    # Write the updated routes file
    routes_tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Updated ROUTES_FILE saved as {output_file}")

    return True


def extract_vtype_distribution(file_path: str) -> dict:
    """
    Extracts the vTypeDistribution from a SUMO XML file and returns a dictionary
    where keys are distribution IDs and values are lists of vehicle type IDs.
    Args:
        file_path (str): Path to the SUMO XML file containing vTypeDistribution.
    Returns:
        dict: A dictionary with distribution IDs as keys and lists of vehicle type IDs as values.
    """

    tree = ET.parse(file_path)
    root = tree.getroot()

    vtype_dict = {}
    for dist in root.findall("vTypeDistribution"):
        dist_id = dist.get("id")
        vtype_ids = [vType.get("id")
                     for vType in dist.findall("vType") if vType.get("id")]
        vtype_dict[dist_id] = vtype_ids

    return vtype_dict


def path_to_xml(path: list[str], vehicle_id: str, veh_type: str, departure_time: int, stop_durations: list[int], no_parking: bool = False) -> str:
    """
    Converts a path to an XML format.
    Args:
        path (list): List of edges or parking areas in the trip.
        vehicle_id (str): Unique identifier for the vehicle.
        veh_type (str): Type of the vehicle.
        departure_time (int): Departure time for the trip.
        stop_durations (list): List of durations for each stop in the trip.
        no_parking (bool): If True, the path is treated as a sequence of edges
                            without parking areas. If False, the path is treated as a sequence of parking areas.
    Returns:
        xml (str): XML string representing the trip.
    """
    xml = f'<trip id="{vehicle_id}" type="{veh_type}" depart="{departure_time}" from="{path[0]}" to="{path[-1]}">\n'
    for i in range(1, len(path)-1):
        if no_parking:
            xml += f'\t<stop edge="{path[i]}" duration="{stop_durations[i]}"/>\n'
        else:
            xml += f'\t<stop parkingArea="{path[i]}" duration="{stop_durations[i]}"/>\n'

    xml += '</trip>'
    return xml


def add_trip_xml(path: list, stop_durations: list[int], veh_id: int, veh_type: str, departure_time: int, out_file_path: str, use_carla_routine: bool = False):
    """
    Creates the XML for n_trips for each provided v_type based on the provided parameters.

    Args:
        path (list): List of edges or parking areas in the trip.
        stop_durations (list): List of durations for each stop in the trip.
        veh_id (int): Unique identifier for the vehicle.
        veh_type (str): Vehicle type.
        departure_time (int): Departure time for the trip.
        use_carla_routine (bool): If True, the path is treated as a sequence of edges without parking areas.
        out_file_path (str): Path to save the generated XML file.
    Returns:
        bool: True if the operation was successful, False otherwise.
    """

    if not os.path.exists(out_file_path):
        # If the file does not exist, create the root element
        xml = '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n'
        xml += '</routes>'
        with open(out_file_path, 'w') as f:
            f.write(xml)

    # when using the carla routine, there will be no parking, so durations are zero and the vehicle will go through edges
    xml = path_to_xml(path, veh_id, veh_type, departure_time,
                      stop_durations, no_parking=use_carla_routine) + '\n'

    tree = ET.parse(out_file_path)
    root = tree.getroot()

    # Parse the new trip XML string to an Element
    new_trip_elem = ET.fromstring(xml)
    root.append(new_trip_elem)

    # Write back to file
    tree.write(out_file_path, encoding='utf-8', xml_declaration=True)

    return True


def parse_veh_dist_xml(param_dict, vtypes_dist, styles, output_path, car_follow_model="IDM", lc_model="SL2015"):
    """
    Generates an XML file for vehicle type distributions based on the provided parameters and styles.
    Args:
        param_dict (dict): Dictionary containing parameters for each style.
        vtypes_dist (dict): Dictionary containing vehicle types and their parameters.
        styles (list): List of styles for which the vehicle types are generated.
        output_path (str): Folder where the XML file will be saved.
        car_follow_model (str): Car-following model to be used for the vehicle types.
        lc_model (str): Lane change model to be used for the vehicle types.
    Returns:
        xml (str): The generated XML string.
    """

    xml = "<root>\n"
    for style in styles:
        xml += f'<vTypeDistribution id=\"{style}\">\n'

        for vtype in vtypes_dist[f'veh_{style}']:
            xml += f'\t<vType id=\"{vtype}\" carFollowModel=\"{car_follow_model}\" laneChangeModel=\"{lc_model}\" '

            for parameter in param_dict.keys():
                xml += f"{parameter}=\"{vtypes_dist['veh_{}'.format(style)][vtype][parameter]}\" "

            xml += f"probability=\"{vtypes_dist['veh_{}'.format(style)][vtype]['probability']}\">\n"
            xml += '\t\t<param key="device.rerouting.probability" value="1.0"/>\n'
            xml += '\t\t<param key="device.rerouting.adaptation-steps" value="18"/>\n'
            xml += '\t\t<param key="device.rerouting.adaptation-interval" value="10"/>\n'
            xml += '\t</vType>\n'

        xml += "</vTypeDistribution>\n"

    xml += "</root>\n"

    with open(output_path, "w") as f:
        f.write(xml)
    return xml


def parse_veh_fixed_xml(param_dict, styles, output_path, car_follow_model="IDM", lc_model="SL2015"):
    """
    Generates an XML file for fixed vehicle types based on the provided parameters and styles.
    Args:
        param_dict (dict): Dictionary containing fixed parameters for each style.
        styles (list): List of styles for which the vehicle types are generated.
        output_path (str): Folder where the XML file will be saved.
        car_follow_model (str): Car-following model to be used for the vehicle types.
        lc_model (str): Lane change model to be used for the vehicle types.
    Returns:
        xml (str): The generated XML string.
    """

    xml = "<root>\n"
    for style in styles:
        xml += f'<vTypeDistribution id=\"{style}\">\n'

        xml += f'\t<vType id=\"veh_{style}\" carFollowModel=\"{car_follow_model}\" laneChangeModel=\"{lc_model}\" '

        for parameter in param_dict[styles[0]].keys():
            xml += f"{parameter}=\"{param_dict[style][parameter]}\" "

        xml += 'probability="1.0">\n'
        xml += '\t\t<param key="device.rerouting.probability" value="1.0"/>\n'
        xml += '\t\t<param key="device.rerouting.adaptation-steps" value="18"/>\n'
        xml += '\t\t<param key="device.rerouting.adaptation-interval" value="10"/>\n'
        xml += '\t</vType>\n'

        xml += "</vTypeDistribution>\n"

    xml += "</root>\n"

    with open(output_path, "w") as f:
        f.write(xml)
    return xml


def merge_routes(routine_routes_path, random_routes_path, output_file_path):

    # Create the root <routes> element
    routes_root = ET.Element('routes')
    routes_root.set('xmlns:xsi', "http://www.w3.org/2001/XMLSchema-instance")
    routes_root.set('xsi:noNamespaceSchemaLocation',
                    "http://sumo.dlr.de/xsd/routes_file.xsd")

    elements = []
    root_routines = ET.parse(routine_routes_path).getroot()
    root_random = ET.parse(random_routes_path).getroot()
    for elem in root_routines:
        if elem.tag in ('trip', 'vehicle'):
            elements.append(elem)
        elif elem.tag == 'vType':
            routes_root.append(elem)

    for elem in root_random:
        if elem.tag in ('trip', 'vehicle'):
            elements.append(elem)
        elif elem.tag == 'vType':
            # Check if the vType already exists in the routes_root
            existing_vtype = routes_root.find(
                f"vType[@id='{elem.get('id')}']")
            if existing_vtype is None:
                routes_root.append(elem)

    # Sort the elements by their 'depart' attribute, required by SUMO
    elements.sort(key=lambda x: float(x.get('depart', 0)))

    # Append sorted elements to the routes_root
    for elem in elements:
        routes_root.append(elem)

    with open(output_file_path, "w", encoding="utf-8") as f:
        pretty = parseString(
            tostring(routes_root, encoding="utf-8")).toprettyxml(indent="  ")
        f.write("\n".join(line for line in pretty.splitlines() if line.strip()))


def make_output_file(output_file_path, final_trips_file_path=None, random_trips_file_path=None) -> str:
    # Creates the output file with the merged routes:
    # if final_trips is not provided, only the random trips will be used
    # if random_trips is not provided, only the final trips will be used
    # if both are provided, they will be merged
    """
    Generates the output file with the merged routes. If you want to use the alternative routes, you can set final_trips_file_path to the alternative routes path.
    Args:
        output_file_name (str): Name of the output file.
        final_trips_file_path (str): Name of the final trips file.
        random_trips_file_path (str): Name of the random trips file.
    Returns:
        str: Path to the output file.
    Raises:
        ValueError: If both final_trips_file_path and random_trips_file_path are not provided.
    """

    if not final_trips_file_path and not random_trips_file_path:
        raise ValueError(
            "At least one of final_trips_file_path or random_trips_file_path must be provided.")

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    if final_trips_file_path and random_trips_file_path:
        merge_routes(final_trips_file_path,
                     random_trips_file_path, output_file_path)

    elif final_trips_file_path:
        shutil.copyfile(final_trips_file_path, output_file_path)

    else:
        shutil.copyfile(random_trips_file_path, output_file_path)

    print(f"Output file created at {output_file_path}")
    return output_file_path


def clean_response(institutes: list[str]) -> list[str]:
    """
    Cleans the list of institutes by removing parenthesis and everything after it,
    splitting by hyphen, and taking the longest slice. If there are multiple parts
    separated by '/', it adds both parts to the cleaned institutes list.
    Args:
        institutes (list): List of institute names to be cleaned.
    Returns:
        list: A cleaned list of institute names.
    """

    cleaned_institutes = []
    for institute in institutes:
        # Remove parenthesis and everything after it
        institute = re.sub(r'\(.*', '', institute).strip()

        # Split by hyphen and take the longest slice
        parts = institute.split('-')
        longest_part = max(parts, key=len).strip()

        # Split by '/' and add both parts to the cleaned_institutes list
        if '/' in longest_part:
            parts = longest_part.split('/')
            for part in parts:
                part = part.strip()
                if part not in cleaned_institutes:
                    cleaned_institutes.append(part)

        else:
            if longest_part not in cleaned_institutes:
                cleaned_institutes.append(longest_part)

    return cleaned_institutes


def get_spawn_points_from_csv(csv_file: str) -> dict:
    """ Reads a CSV file with spawn points and returns a dictionary with the spawn points.
    Args:
        csv_file (str): The path to the CSV file containing the spawn points.
    Returns:
        dict: A dictionary with the spawn points, where keys are the names and values are carla.Transform objects.
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


def get_routines_from_csv(llm_routines_folder: str) -> list:
    """ Reads the LLM routines from CSV files in the specified folder and returns a list of
    routines, where each routine is a dictionary with time as keys and the corresponding
    vehicle information as values.
    Args:
        llm_routines_folder (str): The path to the folder containing the LLM routines.
    Returns:
        list: A list of routines, where each routine is a dictionary with time as keys and
              the corresponding vehicle information as values.
    """

    llm_files = os.listdir(llm_routines_folder)

    routines = []
    for file in llm_files:
        df = read_csv(f'{llm_routines_folder}/{file}')
        routine = {f"{row['time']}": {k: row[k]
                                      for k in df.columns if k != 'time'} for _, row in df.iterrows()}
        routines.append(routine)

    return routines


def save_routines_csv(location_time_list: list[dict], veh_ids: list, dir_path: str, use_lat_lon: bool = True):
    """ Saves the provided location_time_list as CSV files in a specified directory.
    Args:
        location_time_list (list): A list of dictionaries, where each dictionary contains
                                   vehicle information with time as keys.
        veh_ids (list): A list of vehicle IDs corresponding to the location_time_list.
        dir_name (str): The name of the directory where the CSV files will be saved.
        use_lat_lon (bool): If True, the coordinates will be saved as latitude and longitude.
                            If False, they will be saved as x_pos and y_pos.
    Returns:
        None
    """

    os.makedirs(dir_path, exist_ok=True)

    # Save each dictionary in location_time_list as a CSV file
    for idx, location_time in enumerate(location_time_list):
        df = DataFrame.from_dict(location_time, orient='index')
        # Separate the coords tuple into latitude and longitude columns
        if use_lat_lon:
            df[['latitude', 'longitude']] = DataFrame(
                df['coords'].tolist(), index=df.index)
        else:
            df[['x_pos', 'y_pos']] = DataFrame(
                df['coords'].tolist(), index=df.index)

        # Drop the original coords column
        df.drop(columns=['coords'], inplace=True)
        csv_path = f'{dir_path}/{veh_ids[idx]}.csv'
        df.to_csv(csv_path, index_label='Time')
        print(f"Route saved to: {csv_path}")


def save_ids_styles_csv(veh_ids: list[str], veh_styles: list[str], output_path: str):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'style'])
        for rid, style in zip(veh_ids, veh_styles):
            writer.writerow([rid, style])
    print(f"Saved to {output_path}")
