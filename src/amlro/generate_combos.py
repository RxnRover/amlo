from typing import List, Dict

import math


def get_next_combo(resolutions, bounds, curr_values, full_combo_list: List[List[float]], feature_idx: int) -> List[List[float]]:
    """_summary_

    Args:
        resolutions (_type_): _description_
        bounds (_type_): _description_
        curr_values (_type_): _description_
        full_combo_list (List[List[float]]): _description_
        feature_idx (int): _description_

    Returns:
        List[List[float]]: _description_
    """
    tmp_values = [x for x in curr_values]

    if (feature_idx >= len(tmp_values)):
        tmp_value = [x for x in tmp_values]
        full_combo_list.append(tmp_value)
        return full_combo_list

    # Reset current feature value to minimum
    tmp_values[feature_idx] = bounds[feature_idx][0]

    while(math.isclose(tmp_values[feature_idx], bounds[feature_idx][1]) or tmp_values[feature_idx] < bounds[feature_idx][1]):
        full_combo_list = get_next_combo(
            resolutions, bounds, tmp_values, full_combo_list, feature_idx + 1)

        tmp_values[feature_idx] += resolutions[feature_idx]

    return full_combo_list


def generate_uniform_grid(config: Dict):
    """_summary_

    Args:
        config (Dict): _description_

    Returns:
        _type_: _description_
    """

    # Initialize current values to lower bounds
    curr_values = [bound[0] for bound in config["bounds"]]
    print("Initial values: {}".format(curr_values))

    return get_next_combo(config["resolutions"], config["bounds"], curr_values, [], 0)


def optimizer_iteration(full_combo, training_set, parameters=[], rxn_yield=0):
    pass


def main():

    config = {
        "bounds": [[0, 0.3],
                   [0, 0.3],
                   [-10, 20]],
        "resolutions": [0.1, 0.1, 5],
        "feature_names": ["f1", "f2", "f3"],
        "feature_count": 3
    }

    full_combo_list = generate_uniform_grid(config)

    for combo in full_combo_list:
        print(combo)


if __name__ == "__main__":

    main()
