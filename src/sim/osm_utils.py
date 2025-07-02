from typing import Union
import requests

building_query_template = """
    [out:json];
    (
      node["amenity"="{build_type}"](around:{radius},{latitude},{longitude});
      way["amenity"="{build_type}"](around:{radius},{latitude},{longitude});
      relation["amenity"="{build_type}"](around:{radius},{latitude},{longitude});
      node["shop"="{build_type}"](around:{radius},{latitude},{longitude});
      way["shop"="{build_type}"](around:{radius},{latitude},{longitude});
      relation["shop"="{build_type}"](around:{radius},{latitude},{longitude});
      node["leisure"="{build_type}"](around:{radius},{latitude},{longitude});
      way["leisure"="{build_type}"](around:{radius},{latitude},{longitude});
      relation["leisure"="{build_type}"](around:{radius},{latitude},{longitude});
    );
    out center;
    """

uni_query_template = """
    [out:json];
    (
      node["amenity"="college"](around:{radius},{latitude},{longitude});
      way["amenity"="college"](around:{radius},{latitude},{longitude});
      relation["amenity"="college"](around:{radius},{latitude},{longitude});
      node["building"="university"](around:{radius},{latitude},{longitude});
      way["building"="university"](around:{radius},{latitude},{longitude});
      relation["building"="university"](around:{radius},{latitude},{longitude});
    );
    out center;
    """


def geocode_address(address: str) -> Union[list, None]:
    """
    Geocodes an address using the Nominatim API from OpenStreetMap.
    Args:
        address (str): The address to geocode.

    Returns:
        list: A list of dictionaries with formatted address, latitude, and longitude.
    """
    # URL for the Nominatim API
    url = 'https://nominatim.openstreetmap.org/search'

    # Parameters for the API request
    params = {
        'q': address,           # The address to search
        'format': 'json',       # Response format
        'addressdetails': 1     # Include detailed address information
    }

    headers = {
        'User-Agent': 'Student SUMO project (r244808@dac.unicamp.br)'
    }

    # Make the request
    response = requests.get(url, params=params, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        if data:
            # Extract and return formatted address, latitude, and longitude
            results = []
            for result in data:
                results.append({
                    "formatted_address": result['display_name'],
                    "latitude": float(result['lat']),
                    "longitude": float(result['lon'])
                })
            return results
        else:
            return []

    else:
        print(f"Error: {response.status_code}")
        return None


def find_nearby_buildings(latitude: float, longitude: float, radius: int, build_type: str = None, university: bool = False, filters: list[str] = None):
    """    Finds nearby buildings of a specific type or universities using the Overpass API.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        radius (int): Search radius in meters.
        build_type (str, optional): Type of building to search for (e.g., 'school', 'hospital', 'restaurant'). If None, searches for universities.
        university (bool, optional): If True, searches for universities instead of a specific building type.
        filters (list, optional): List of strings to filter results by name. If None, no filtering is applied.
    Returns:
        list: A list of dictionaries containing the name, latitude, longitude, and type of each building found.
    Raises:
        ValueError: If neither 'build_type' is specified nor 'university' is True, or if both are specified.
    """

    # Overpass API endpoint
    url = "https://overpass-api.de/api/interpreter"

    if build_type is None and not university:
        raise ValueError(
            "Either 'build_type' must be specified or 'university' must be True.")

    if build_type is not None and university:
        raise ValueError(
            "Cannot specify 'build_type' and 'university' at the same time. Choose one.")

    # Overpass API query
    if university:
        # Use the university query template
        query = uni_query_template.format(
            radius=radius, latitude=latitude, longitude=longitude)
    else:
        # Use the building query template
        query = building_query_template.format(
            build_type=build_type, radius=radius, latitude=latitude, longitude=longitude)

    # Make the API request
    response = requests.get(url, params={"data": query})

    if response.status_code == 200:
        data = response.json()
        results = []

        for element in data.get('elements', []):
            if element['type'] == 'node':
                # Extract coordinates directly for nodes
                lat = element['lat']
                lon = element['lon']
            elif element['type'] in ['way', 'relation']:
                # Use the 'center' key for ways and relations
                center = element.get('center')
                if center:
                    lat = center['lat']
                    lon = center['lon']
                else:
                    # Skip if no center is available
                    continue
            else:
                # Skip unsupported types
                continue

            # Add the result
            name = element.get("tags", {}).get("name")
            if name and any(filter_str.lower() in name.lower() for filter_str in filters):

                if not any(result['name'] == name for result in results):
                    results.append({
                        "name": name,
                        "latitude": lat,
                        "longitude": lon,
                        "type": element['type']
                    })
        return results

    else:
        print(f"Error: {response.status_code}")
        return []
