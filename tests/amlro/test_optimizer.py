import unittest

import amlro.optimizer as optimizer


class TestUniformGrid(unittest.TestCase):

    def test_categorical_feature_decoding(self):
        """Test that a encoded parameter list decoding back."""

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

        decoded = [1,1,'dog']
        encoded = [1,1,1]

        decoded_list = optimizer.categorical_feature_decoding(config,encoded)

        self.assertListEqual(list(decoded_list), decoded)

    def test_categorical_feature_encoding(self):
        """Test that a decoded parameter list convert into encoded list."""

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

        decoded = [1,1,'dog']
        encoded = [1,1,1]

        encoded_list = optimizer.categorical_feature_encoding(config,decoded)

        self.assertListEqual(list(encoded_list), encoded)