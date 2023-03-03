import unittest

from amlro import generate_combos


class TestConfigValidation(unittest.TestCase):

    def test_max_bound_lt_min_bound(self):
        """If the bounds are invalid, the function should say so."""

        config = {
            "continuous": {
                "bounds": [[0, -1], [0, -1]],
                "resolutions": [1, 1],
                "feature_names": ["f1", "f2"],
            }
        }

        self.assertRaises(ValueError, generate_combos.validate_config, config)

    def test_negative_resolution(self):
        """Only positive resolutions make progress from the min to max bound,
        so a negative resolution should be rejected.
        """

        config = {
            "continuous": {
                "bounds": [[0, 1], [0, 1]],
                "resolutions": [-1, 1],
                "feature_names": ["f1", "f2"],
            }
        }

        self.assertRaises(ValueError, generate_combos.validate_config, config)

    def test_zero_resolution(self):
        """A resolution of zero would cause an infinite loop with no progress,
        so it should not be accepted.
        """

        config = {
            "continuous": {
                "bounds": [[0, 1], [0, 1]],
                "resolutions": [0, 1],
                "feature_names": ["f1", "f2"],
            }
        }

        self.assertRaises(ValueError, generate_combos.validate_config, config)


class TestGetCombos(unittest.TestCase):

    def test_small_list(self):
        features = [[0, 1], [0, 1]]

        corr = [[0, 0], [0, 1], [1, 0], [1, 1]]

        combos = generate_combos.get_combos(features)

        self.assertEqual(combos, corr)

    def test_categorical_features(self):
        features = [["cat", "dog"], ["black", "grey"]]

        # May not be in this order
        corr = [
            ["cat", "black"],
            ["cat", "grey"],
            ["dog", "black"],
            ["dog", "grey"],
        ]

        combos = generate_combos.get_combos(features)

        self.assertEqual(combos, corr)

    def test_mixed_features(self):
        features = [[0, 1], ["cat", "dog"], ["black", "grey"]]

        # May not be in this order
        corr = [
            [0, "cat", "black"],
            [0, "cat", "grey"],
            [0, "dog", "black"],
            [0, "dog", "grey"],
            [1, "cat", "black"],
            [1, "cat", "grey"],
            [1, "dog", "black"],
            [1, "dog", "grey"],
        ]

        combos = generate_combos.get_combos(features)

        self.assertEqual(combos, corr)


class TestUniformGrid(unittest.TestCase):

    def test_generate_uniform_grid_with_no_space_in_bounds(self):
        """When the min and max bounds are equal, only one combo should
        be generated.
        """

        config = {
            "continuous": {
                "bounds": [[0, 0], [0, 0]],
                "resolutions": [1, 1],
                "feature_names": ["f1", "f2"],
            },
        }

        corr = [[0, 0]]

        combo_list = generate_combos.generate_uniform_grid(config)

        self.assertEqual(combo_list, corr)

    def test_generate_uniform_grid_small_binary_grid(self):
        """Test that a small grid of binary numbers can be generated."""

        config = {
            "continuous": {
                "bounds": [[0, 1], [0, 1]],
                "resolutions": [1, 1],
                "feature_names": ["f1", "f2"],
            }
        }

        corr = [[0, 0], [0, 1], [1, 0], [1, 1]]

        combo_list = generate_combos.generate_uniform_grid(config)

        self.assertEqual(combo_list, corr)

    def test_generate_uniform_grid_larger_grid_with_nonintegers(self):
        """Test that a larger grid with non-integer values, negative
        bounds, and a resolution which the bounds are not divisible by can
        be generated correctly."""

        config = {
            "continuous": {
                "bounds": [[0, 0.5], [-2, 2.5]],
                "resolutions": [0.26, 0.5],
                "feature_names": ["f1", "f2"],
            }
        }

        corr = [
            [0, -2],
            [0, -1.5],
            [0, -1.0],
            [0, -0.5],
            [0, 0.0],
            [0, 0.5],
            [0, 1.0],
            [0, 1.5],
            [0, 2.0],
            [0, 2.5],
            [0.26, -2],
            [0.26, -1.5],
            [0.26, -1.0],
            [0.26, -0.5],
            [0.26, 0.0],
            [0.26, 0.5],
            [0.26, 1.0],
            [0.26, 1.5],
            [0.26, 2.0],
            [0.26, 2.5],
        ]

        combo_list = generate_combos.generate_uniform_grid(config)
        for pair in combo_list:
            print(pair)

        self.assertEqual(combo_list, corr)

    def test_generate_uniform_grid_mixed_variables(self):
        """Test that a small grid of binary numbers can be generated."""

        config = {
            "continuous": {
                "bounds": [[0, 1], [0, 1]],
                "resolutions": [1, 1],
                "feature_names": ["f1", "f2"],
            },
            "categorical": {
                "feature_names": ["animal"],
                "values": [["cat", "dog"]]
            }
        }

        corr = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

        combo_list = generate_combos.generate_uniform_grid(config)

        self.assertEqual(combo_list, corr)