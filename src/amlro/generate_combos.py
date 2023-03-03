from typing import Any, Dict, List

import math


def validate_config(config: Dict) -> None:
    """Validates the configuration dictionary for generating grids.

    :param config: Configuration to be checked
    :type config: Dict

    :raises ValueError: At least one given bound is invalid.
    :raises ValueError: At least one given resolution is invalid.
    """

    # Check for invalid bounds
    for bound in config["continuous"]["bounds"]:
        if bound[0] > bound[1]:
            msg = "Max bound must be greater than or equal to the min "
            msg += "bound. Given bounds: Min = {}, Max = {}".format(
                bound[0], bound[1]
            )
            raise (ValueError(msg))

    # Check for invalid resolutions
    for resolution in config["continuous"]["resolutions"]:
        if resolution <= 0:
            msg = "Resolutions must all be positive, nonzero values. "
            msg += "Given resolutions: {}".format(
                config["continuous"]["resolutions"]
            )
            raise (ValueError(msg))


def get_combos(features: List[List[Any]]) -> List[List[Any]]:
    """Generates all combinations of the values for the features given. The
    order of the combinations is not guaranteed by this function.

    :param features: Discretized features to generate combinations of, with
                     one feature per row.
    :type features: List[List[Any]]

    :return: All combinations of the features, with one combination per row.
    :rtype: List[List[Any]]
    """

    return _get_next_combo(features, [], [], 0)


def _get_next_combo(
    features: List[List[Any]],
    curr_combo: List[Any],
    full_combo_list: List[List[Any]],
    feature_idx: int,
) -> List[List[Any]]:
    """Internal implementation for the recursive function to generate all
    combinations for a given set of discretized features. This helps to hide
    the extra parameters from the user.

    This implementation will iterate over the features with the first feature
    moving slowest and the last feature moving fastest in the generated
    combinations. This means combinations for the features [0, 1] and [1, 2]
    will be generated in the following order: [0, 1], [0, 2], [1, 1], [1, 2].

    :param features: Discretized features to generate combinations of, with
                     one feature per row.
    :type features: List[List[Any]]
    :param curr_combo: Current combination being generated. In the initial
                       call this should be an empty list ([])
    :type curr_combo: List[Any], optional
    :param full_combo_list: Full list of combinations. In the initial
                       call this should be an empty list ([])
    :type full_combo_list: List[List[float]], optional
    :param feature_idx: Current feature index. In the initial
                       call this should be zero (0)
    :type feature_idx: int

    :return: All combinations of the features, with one combination per row.
    :rtype: List[List[Any]]
    """

    # Make a copy of the curr_combo list
    tmp_combo = [x for x in curr_combo]

    # Base case
    # We have passed the last feature and a full combination has been
    # generated.
    if feature_idx >= len(features):
        full_combo_list.append(tmp_combo)
        return full_combo_list

    # Loop over each value of the current feature, adding it to the current
    # combination and passing it to the next parameter
    for value in features[feature_idx]:
        tmp_combo.append(value)

        full_combo_list = _get_next_combo(
            features, tmp_combo, full_combo_list, feature_idx + 1
        )

        del tmp_combo[-1]

    return full_combo_list


def generate_uniform_grid(config: Dict) -> List[List[float]]:
    """Genereting the parameter grid using the parameter bounds and their resolution.

    :param config: Dictionary of parameters, their bounds and resolution.
    :type config: Dict
    :return: List of all the parameter combinations within bounds.
    :rtype: List[List[float]]
    """

    validate_config(config)

    features = []

    if "categorical" in config and len(config["categorical"]):
        for i in range(len(config["categorical"]["feature_names"])):
            config["continuous"]["feature_names"].append(config["categorical"]["feature_names"][i])
            config["continuous"]["bounds"].append([0, len(config["categorical"]["values"][i]) - 1])
            config["continuous"]["resolutions"].append(1)

    for i in range(len(config["continuous"]["bounds"])):
        lower_bound = config["continuous"]["bounds"][i][0]
        upper_bound = config["continuous"]["bounds"][i][1]

        feature = []

        next_value = lower_bound

        # This while condition is essentially `while (next_value <= upper_bound)`,
        # but math.isclose() is used to account for slight variation in stored
        # float values (0.3 might be stored as 0.30000000000000004)
        while next_value < upper_bound or math.isclose(next_value, upper_bound):
            feature.append(next_value)

            next_value += config["continuous"]["resolutions"][i]

        features.append(feature)

    return get_combos(features)


def optimizer_iteration(full_combo, training_set, parameters=[], rxn_yield=0):
    pass


def main():
    config = {
        "continuous": {
            "feature_names": ["f1", "f2", "f3"],
            "bounds": [[0, 0.3], [0, 0.3], [-10, 20]],
            "resolutions": [0.1, 0.1, 5],
        },
        "categorical": {
            "feature_names": ["animal", "color"],
            "values": [["cat", "dog"], ["brown", "black", "yellow"]],
        },
    }

    full_combo_list = generate_uniform_grid(config)

    for combo in full_combo_list:
        print(combo)


if __name__ == "__main__":
    main()
