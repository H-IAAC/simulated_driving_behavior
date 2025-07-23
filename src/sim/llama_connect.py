import os
import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from groq import Groq
from tqdm import tqdm
from pandas import DataFrame
from typing import List

client = Groq(
    # Initialize the Groq client with the API key from environment variables
    # Ensure that the environment variable GROQ_API_KEY is set with your Groq API key
    # You can set this in your terminal or in a .env file
    api_key=os.getenv("GROQ_API_KEY"),
)


def get_response(message: str) -> str:
    """
    Generates a response from the LLAMA model using the Groq API.
    Args:
        message (str): The message to be sent to the LLAMA model.
    Returns:
        str: The response from the LLAMA model.
    """
    instructions_content = (f"{message}"
                            )
    message = [
        {
            "role": "user",
            "content": instructions_content,
        }
    ]

    chat_completion = client.chat.completions.create(
        messages=message, model="llama3-8b-8192")

    return chat_completion.choices[0].message.content


def get_trip_carla(places, short=False):
    instructions_content = ("Plan the information of a person.")
    if short:
        message = [
            {
                "role": "system",
                "content": "You need to plan the routine of a person who has access to the following places: {places}. This persons must go through all the places throughout the day, but never visit them more than two times a day. Consider this person does not spend more than one hour at each place and when there are no more places left, they return home. Your response should be in a JSON format showing only the current location and current activity. Never include locations to activities. Always start the day at time 7 at home and end the day at home, after having visited all the places at least one. Update every hour. Locations always have a single place, never two. Follow the example where the first number is the time: {{'7': {{'location':'home', 'activity':'wake up'}}, '8': {{'location':'school', 'activity':'study'}}, '9':{{'location':'cafe', 'activity':'breakfast".format(
                    places=", ".join(places),
                )
            },
            {
                "role": "user",
                "content": instructions_content,
            }
        ]

    else:
        message = [
            {
                "role": "system",
                "content": "You need to plan the routine of a person who has access to the following places: {places}. This person goes to a lot of places during the day and never stays more than an hour anywhere. Be creative and consider this person does not like to stay to much time at the same place. Maybe they go to the gym in the morning and in the evening, they have to take their kids to and from school, go to the bar and to the movies at night, for example. This person always gets home very late. Your response should be in a JSON format showing only the current location and current activity. Never include locations to activities. Always start the day at time 7 at home and end the day at time 23 at home, update every hour. Locations always have a single place, never two. Follow the example where the first number is the time: {{'7': {{'location':'home', 'activity':'wake up'}}, '8': {{'location':'school', 'activity':'study'}}, '9':{{'location':'cafe', 'activity':'breakfast".format(
                    places=", ".join(places),
                )
            },
            {
                "role": "user",
                "content": instructions_content,
            }
        ]

    chat_completion = client.chat.completions.create(
        messages=message, model="llama3-8b-8192", temperature=1, response_format={"type": "json_object"})

    return chat_completion.choices[0].message.content


def get_trip_sumo(student_info: str, places: dict) -> str:
    """ Generates a response for a student's daily routine trip using the LLAMA model.
    Args:
        student_info (str): Information about the student, such as their field of study.
        places (dict): A dictionary containing various places categorized by their type, such as leisure,
                       eating, shopping, sports, institutes, and university.
    Returns:
        str: A JSON formatted string representing the student's daily routine, including their current location
              and activity for each hour of the day."""

    institute = places['institute']
    leisure = places['leisure']
    eating = places['eating']
    shopping = places['shopping']
    sports = places['sports']

    instructions_content = (f"Plan the information of the following student: {student_info}."
                            )
    message = [
        {
            "role": "system",
            "content": "You need to plan the routine of a student. I want to know what they do during the day, including having lunch, studying, leisure, shopping and practicing sports. Whenever students are having classes or studying, they MUST be at one of the following places, according to what they are studying which are separated by OR: {institutes}. A biology student would most likely be at the Institute of Biology or at the Institute of Geociences, for example. A civil engineering student would most likely be at the Faculty of Civil Engineering. Whenever students are having lunch, breakfast or dinner, they MUST be at one of the following places: {eating}. Whenever students are having leisure, they MUST be at one of the following places: {leisure}. Whenever students are shopping, they MUST be at one of the following places: {shopping}. Whenever students are praticing sports, they MUST be at the following places: {sports}. Do not use any location that has not been provided. Students have lunch between 12 and 14 and dinner between 18 and 21. Students do not go to any institute after 20. Students only go to the gym once every day. Some days, students go the gas_station. Do not name the places, just choose one of the options. Students spend most of their day having classes at an institute. Your response should be in a JSON format showing only the current location and current activity. Never include locations to activities. Always start the day at time 7 at home and end the day at time 23 at home, update every hour. Locations always have a single place, never two. Follow the example where the first number is the time: {{'7': {{'location':'home', 'activity':'wake up'}}, '8': {{'location':'{institute}', 'activity':'study'}}, '9':{{'location':'cafe', 'activity':'breakfast'}}}}.".format(
                institutes=" or ".join(institute),
                institute=institute[0],
                leisure=", ".join(leisure),
                shopping=", ".join(shopping),
                sports=", ".join(sports),
                eating=", ".join(eating)
            )
        },
        {
            "role": "user",
            "content": instructions_content,
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=message, model="llama3-8b-8192", temperature=1, response_format={"type": "json_object"})

    return chat_completion.choices[0].message.content


def response_check(response: str, possible_locations: list, short: bool = False) -> bool:
    """ Checks if the response from the LLM makes sense.
    Args:
        response (str): The response from the LLM in JSON format.
        possible_locations (list): A list of valid locations that the LLM can generate.
    Returns:
        bool: True if the response is valid, False otherwise.
    """

    response = json.loads(response)
    if not short:
        if len(response) < 17:
            print("Error: The response is missing some hours")
            return False

        if len(response) > 24:
            print("Error: The response has too many hours")
            return False

    for item in response:
        try:
            local = response[item]['location']
            if local not in possible_locations or ',' in local:
                print(f"Error: Invalid location generated '{local}'")
                return False

        except KeyError:
            print("Error: The response is missing a key")
            return False

    return True


def generate_response_trips(student_info: str, places: dict, number_of_trips: int = 5) -> List[str]:
    """ Generates and verifies a list of responses for the student's daily routine trips using the LLAMA model.
    Args:
        student_info (str): Information about the student, such as their field of study.
        places (dict): A dictionary containing various places categorized by their type, such as leisure,
                       eating, shopping, sports, institutes, and university.
        number_of_trips (int): The number of trips to generate for the student.
    Returns:
        list: A list of JSON formatted strings representing the student's daily routine trips.
    """

    responses = []
    i = 0
    total_locations = sum(places.values(), []) + ['home']

    with tqdm(total=number_of_trips) as pbar:
        while i < number_of_trips:
            try:
                response = get_trip_sumo(student_info, places)

                if response_check(response, total_locations):
                    responses.append(response)
                    i += 1
                    pbar.update(1)
                else:
                    print("Invalid response. Generating a new one.")

            except Exception as e:
                print(f"Error generating response {e}. Trying a new one.")

    return responses


def generate_range_parameters(parameters: str, styles: List[str]) -> str:
    """ Generates the range of parameters for the vehicle types using the LLAMA model.
    Args:
        parameters (str): A string containing the parameters to be used in the generation.
        styles (list): A list of styles for which the parameters will be generated.
    Returns:
        str: A JSON formatted string representing the range of values for each parameter and style.
    """

    instructions_content = (f"Give the range of values for the following styles: {styles}."
                            )
    message = [
        {
            "role": "system",
            "content": "You need to return range of values in JSON for every one of the parameters that represent how a driver behaves in traffic, give an explanation for why you picked each value. Following, there is a list of parameter, default value, range [minimum-maximum] and description:\n {parameters}.\nThe more aggressive a driver is, the less they tend to cooperate in traffic and the more selfish they are. ALWAYS BE INSIDE THE RANGE LIMIT. Consider the default value for each parameter as a basis for a normal driver. Consider the answers you gave to the previous parameters when giving your answer. One parameter range of values must not be a subrange of any other parameter range, meaning you should not give overlap the range of other styles, if a aggressive style is given 'min': 0.2 and max: '0.5' for some paramter, another style can not have 'min':0.3, 'max':0.4' for this same paramters, because the ranges overlap each other. Keep the same distance between min and max for every style for each parameter. ALL THE PARAMETERS PROVIDED and BE ALWAYS IN THE SAME FOLLOWING FORMAT containing the parameter name, the style, the min and max values and the reason you picked those values. Note that every parameter has the same JSON structure that may NOT be changed. ALWAYS BE INSIDE THE RANGE LIMIT. PARAMETERS WITH ARE FACTORS WILL ALWAYS BE BETWEEN 0 AND 1. Example of proper JSON: {{'parameter': {{'style': {{'explanation': 'string', 'min': value, 'max': value}}}}}}. For example, if the styles are aggressive and normal: {{'lcCooperative': {{'aggressive': {{'explanation': 'aggresive drivers are not very cooperative', 'min': 0.2, 'max': 0.5}}, 'normal': {{'explanation': 'normal drivers are cooperative', 'min': 0.5, 'max': 0.8}}}}}}.".format(parameters=parameters)
        },
        {
            "role": "user",
            "content": instructions_content,
        }
    ]
    try:
        chat_completion = client.chat.completions.create(
            messages=message, model="llama3-8b-8192", temperature=0.5, response_format={"type": "json_object"})

    except Exception as e:
        print(f"Error generating response {e}. Trying a new one.")
        return generate_range_parameters(parameters, styles)

    return chat_completion.choices[0].message.content


def get_range_parameters(data: DataFrame, params: str, styles: List[str]) -> dict:
    """ Generates the range of parameters for the vehicles using the LLAMA model.
    Args:
        data (pd.DataFrame): A DataFrame containing the parameters to be used in the generation.
        params (str): A string containing the parameters to be used in the generation.
        styles (list): A list of styles for which the parameters will be generated.
    Returns:
        dict: A dictionary with the range of values for each parameter and style.
    """

    # Generates the parameters for the vehicles
    veh_parameters = generate_range_parameters(params, styles)
    param_dict = json.loads(veh_parameters)
    missing_params = [param for param in data['Parameter']
                      if param not in list(param_dict.keys())]

    while missing_params:
        print(
            f"Missing parameters in param_dict: {missing_params}. Trying new response.")
        # Generates the parameters for the vehicles
        veh_parameters = generate_range_parameters(params, styles)
        param_dict = json.loads(veh_parameters)
        missing_params = [param for param in data['Parameter']
                          if param not in list(param_dict.keys())]

    return param_dict


def verify_parameters(parameters_dict: dict, styles: list, separate_distributions: bool = False):
    """ Verifies the parameters generated by the LLAMA model to ensure they are within valid ranges and do not overlap. If they do, they are adjusted directly in the dictionary.
    Args:
        parameters_dict (dict): A dictionary containing the parameters and their ranges for each style.
        styles (list): A list of styles for which the parameters are defined.
        separate_distributions (bool): If True, the function will ensure that the distributions of parameters
                                       for different styles do not overlap. If False, it will not check for this.
    Returns:
        None
    """

    # Trying to better separate the distributions of the parameteres
    if separate_distributions:
        for i in range(len(styles)):
            for param in parameters_dict.keys():

                if param == 'speedFactor':
                    continue

                values = parameters_dict[param][f'{styles[i]}']
                if i < len(styles) - 1 and parameters_dict[param][f'{styles[i + 1]}']['min'] < values['max']:

                    if values['max'] > parameters_dict[param][f'{styles[i + 1]}']['max']:
                        print(
                            f"{param} of {styles[i + 1]} is contained inside {styles[i]}")
                        continue

                    parameters_dict[param][f'{styles[i + 1]}']['min'] = values['max']

                if float(values['max']) == float(values['min']):
                    # If the min and max are the same, we need to adjust them to create a range
                    values['min'] -= 0.2
                    values['max'] += 0.2

                if param.endswith('Factor'):
                    # Factors should be between 0.1 and 0.9
                    print(f"Verifying {param} for {styles[i]}: {values}")
                    if float(values['min']) < 0.1 or float(values['min']) > 0.9:
                        values['min'] = 0.1
                        values['max'] = 0.9

                    if float(values['max']) > 0.9:
                        values['max'] = 0.9

                    if float(values['min']) > float(values['max']):
                        print(
                            f"Warning: {param} for {styles[i]} has min > max. Inverting values.")
                        values['min'], values['max'] = values['max'], values['min']

                    if float(values['min']) == float(values['max']):
                        values['min'] -= 0.1
                        values['max'] += 0.1

                if param == 'startupDelay':
                    # Startup delay should be between 0 and 5 seconds
                    print(f"Verifying {param} for {styles[i]}: {values}")
                    if float(values['min']) < 0.3:
                        values['min'] = 0.3
                    if float(values['max']) > 5:
                        values['max'] = 5

    # Ensure maxSpeed accommodates the maximum speed implied by the distribution:
    # max speed from distribution = speedFactor × default speed + 3 × deviation
    values_speed = parameters_dict['maxSpeed']

    for style in styles:
        speed = values_speed[f'{style}']['max']
        expected_mean_speed = (
            values_speed[f'{style}']['max'] - values_speed[f'{style}']['min']) / 2
        max_speedfactor = (speed - 10) / speed

        print(
            f"Expected mean speed: {expected_mean_speed}, max speedfactor: {max_speedfactor}, max_speed: {speed}")

        parameters_dict['speedFactor'][f'{style}']['max'] = max_speedfactor + 0.1
        parameters_dict['speedFactor'][f'{style}']['min'] = max_speedfactor - 0.1

    print("Verification complete!")


def save_response(response, generated_routines, routines_folder):
    if isinstance(response, str):
        response = json.loads(response)
    df = pd.DataFrame(response).T
    df.index.name = 'time'
    os.makedirs(routines_folder, exist_ok=True)
    df.to_csv(f'{routines_folder}/llm_routine_{generated_routines}.csv')


def generate_routines(places, n_of_routines, routines_folder, short=False):
    """ Generates a number of routines using the LLAMA model and saves them to a CSV file
    Args:
        places (list): A list of places that the LLM can generate routines for.
        n_of_routines (int): The number of routines to generate.
        routines_folder (str): The folder where the generated routines will be saved.
        short (bool): If True, the routines will be shorter (vehicle visits each place one time).
    Returns:
        None: The generated routines are saved to a CSV file in the specified folder.
    """

    generated_routines = 0
    while (generated_routines < n_of_routines):
        try:

            # Generate a full routine where the vehicle can visit each place multiple times
            response = get_trip_carla(places, short=short)
            # Getting a valid response from the LLM
            while not response_check(response, places, short=short):
                response = json.loads(get_trip_carla(places, short=short))

            # Save the response to a CSV file
            save_response(response, generated_routines,
                          routines_folder=routines_folder)
            print(response)
            generated_routines += 1

        except Exception as e:
            print(f"Error generating routine. Trying again. Error: {e}")
            continue


def csv_str(data: DataFrame) -> str:
    """ Converts a pandas DataFrame to a string representation of its contents.
    Args:
        data (pd.DataFrame): The DataFrame to convert.
    Returns:
        str: A string representation of the DataFrame, formatted as a list of parameters with their ranges and descriptions.
    """
    s = ''
    for i in range(len(data)):
        s += f"Parameter: {data['Parameter'][i]}; Range: {data['Range'][i]}; Description: {data['Description'][i]}."
    return s


def show_gaussians(param_dict: dict, parameters: list, styles: list):
    """ Displays the Gaussian distributions for the given parameters and styles.
    Args:
        param_dict (dict): A dictionary containing parameters for each style.
        parameters (list): A list of parameters to be displayed.
        styles (list): A list of styles to be displayed (e.g., 'agg', 'norm').
    Returns:
        None: Displays the Gaussian distributions for each parameter and style.
    """

    num_params = len(parameters)
    num_rows = (num_params + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 6 * num_rows))

    axes = axes.flatten()

    for ax, parameter in zip(axes, parameters):
        for style in styles:
            m = (param_dict[parameter][style]['min'] +
                 param_dict[parameter][style]['max']) / 2

            # Finding the standard deviation for 95% of the data to be within the range
            s = (param_dict[parameter][style]['max'] - m) / \
                stats.norm.ppf(0.975)

            # Generate data
            rng = np.random.default_rng(seed=42)  # For reproducibility
            data = rng.normal(m, s, 5000)

            # Plot the data
            ax.hist(data, bins=30, density=True,
                    alpha=0.6, label=f'{style} style')

            # Plot the Gaussian distribution
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
            ax.plot(x, p, linewidth=2)

        ax.set_title(f'Gaussian Distribution for {parameter}')
        ax.legend()

    # Hide any unused subplots
    for i in range(len(parameters), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
