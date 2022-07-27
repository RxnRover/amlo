from typing import List, Dict

import math


def validate_config(config: Dict) -> None:
    """Validates the configuration dictionary for generating grids.

    :param config: Configuration to be checked
    :type config: Dict

    :raises ValueError: At least one given bound is invalid.
    :raises ValueError: At least one given resolution is invalid.
    """

    # Check for invalid bounds
    for bound in config["bounds"]:
        if (bound[0] > bound[1]):
            msg = "Max bound must be greater than or equal to the min "
            msg += "bound. Given bounds: Min = {}, Max = {}".format(
                bound[0], bound[1])
            raise (ValueError(msg))

    # Check for invalid resolutions
    for resolution in config["resolutions"]:
        if (resolution <= 0):
            msg = "Resolutions must all be positive, nonzero values. "
            msg += "Given resolutions: {}".format(config["resolutions"])
            raise (ValueError(msg))


def get_next_combo(resolutions, bounds, curr_values,
                   full_combo_list: List[List[float]],
                   feature_idx: int) -> List[List[float]]:
    """Recursive function to genereting all the parameter combinations. 

    :param resolutions: Minimum resolution of parameter values.
    :type resolutions: float
    :param bounds: Min and Max bounds for each parameter.
    :type bounds: float
    :param curr_values: Current parameter value at each recursive cycle 
    :type curr_values: float
    :param full_combo_list: All the parameter combinations.
    :type full_combo_list: List[List[float]]
    :param feature_idx: Feature index from parameter list
    :type feature_idx: int
    :return: All the parameter combinations
    :rtype: List[List[float]]
    """
    tmp_values = [x for x in curr_values]

    if (feature_idx >= len(tmp_values)):
        tmp_value = [x for x in tmp_values]
        full_combo_list.append(tmp_value)
        return full_combo_list

    # Reset current feature value to minimum
    tmp_values[feature_idx] = bounds[feature_idx][0]

    while (math.isclose(tmp_values[feature_idx], bounds[feature_idx][1])
           or tmp_values[feature_idx] < bounds[feature_idx][1]):
        full_combo_list = get_next_combo(resolutions, bounds, tmp_values,
                                         full_combo_list, feature_idx + 1)

        tmp_values[feature_idx] += resolutions[feature_idx]

    return full_combo_list


def generate_uniform_grid(config: Dict) -> List[List[float]]:
    """Genereting the parameter grid using the parameter bounds and their resolution.

    :param config: Dictionary of parameters, their bounds and resolution.
    :type config: Dict
    :return: List of all the parameter combinations within bounds.
    :rtype: List[List[float]]
    """

    validate_config(config)

    # Initialize current values to lower bounds
    curr_values = [bound[0] for bound in config["bounds"]]

    return get_next_combo(config["resolutions"], config["bounds"], curr_values,
                          [], 0)


def optimizer_iteration(full_combo, training_set, parameters=[], rxn_yield=0):
    pass


def main():

    config = {
        "bounds": [[0, 0.3], [0, 0.3], [-10, 20]],
        "resolutions": [0.1, 0.1, 5],
        "feature_names": ["f1", "f2", "f3"],
        "feature_count": 3
    }

    full_combo_list = generate_uniform_grid(config)

    for combo in full_combo_list:
        print(combo)


if __name__ == "__main__":

    main()
