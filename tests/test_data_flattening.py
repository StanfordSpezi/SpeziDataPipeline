#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module contains unit tests for classes that manage and flatten FHIR (Fast Healthcare
Interoperability Resources) data structures using pandas DataFrames, particularly focusing
on the FHIRDataFrame and ObservationFlattener classes from the data_flattening library.

The tests ensure the proper initialization, validation, and functionality of these classes.
For the FHIRDataFrame, tests validate the correct setup and validation of data within a DataFrame
tailored for FHIR resources, ensuring that data conforms to expected formats and structures. For
the ObservationFlattener, tests confirm the accurate transformation of complex FHIR Observation
resources into a simplified DataFrame format, suitable for further analysis or processing.

These classes are crucial for handling healthcare data efficiently in a standardized format, and
the tests help ensure robustness and correctness in their implementation.

Classes:
    TestFHIRDataFrame: Tests initialization and validation of FHIRDataFrame instances.
    TestObservationFlattener: Tests the functionality of the ObservationFlattener class in
        converting FHIR observations into a simplified DataFrame format.
"""

# Standard library imports
import datetime
from pathlib import Path

# Related third-party imports
import unittest
from unittest.mock import MagicMock
import pandas as pd


# Local application/library specific imports
from data_flattening.fhir_resources_flattener import (
    FHIRDataFrame,
    FHIRResourceType,
    ColumnNames,
    ObservationFlattener,
)


class TestFHIRDataFrame(unittest.TestCase):  # pylint: disable=unused-variable
    """
    A test case for the FHIRDataFrame class, which is designed to handle and validate
    FHIR data in a pandas DataFrame format.

    Attributes:
        None.
    """

    def test_initialization_and_validation(self):
        """
        Tests the initialization and validation of the FHIRDataFrame object. This includes loading
        sample FHIR data from a CSV file, initializing a FHIRDataFrame with this data, and
        validating the correct setup of the DataFrame and its columns.

        It checks that the data frame is correctly initialized with the given FHIR resource type,
        and validates the columns of the data frame to ensure they meet expected criteria.

        Steps:
        - Load data from a CSV file using pandas, with custom date parsing.
        - Initialize the FHIRDataFrame with loaded data and specify the resource type as
            OBSERVATION.
        - Validate that the data frame and resource type are correctly set.
        - Check column validation is successful.
        """

        def custom_date_parser(date_str):
            """Parses a date string into a datetime.date object."""
            return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()

        data_file = Path(__file__).parent.parent / "sample_data" / "sample_df.csv"
        data = pd.read_csv(
            data_file, parse_dates=["EffectiveDateTime"], date_parser=custom_date_parser
        )

        # Initialize FHIRDataFrame with sample data and a resource type
        df = FHIRDataFrame(data, FHIRResourceType.OBSERVATION)

        # Check that the DataFrame and resource type are correctly set
        self.assertTrue(isinstance(df.data_frame, pd.DataFrame))
        self.assertEqual(df.resource_type, FHIRResourceType.OBSERVATION)

        # Validate columns
        self.assertTrue(df.validate_columns())


class TestObservationFlattener(unittest.TestCase):  # pylint: disable=unused-variable
    """
    A test case for the ObservationFlattener class, responsible for flattening FHIR Observation
    resources into a simplified DataFrame structure.

    Attributes:
        None.
    """

    def test_flatten(self):
        """
        Tests the functionality of the ObservationFlattener's flatten method. This method is
        supposed to take a list of mocked FHIR Observation instances, flatten them into a simpler
        data structure, and validate the output.

        Steps:
        - Mock FHIR Observation instances with predetermined data.
        - Flatten these observations using the ObservationFlattener.
        - Validate the structure and contents of the resulting FHIRDataFrame to ensure it matches
            expectations.
        """
        # Assuming we create or mock FHIR Observation instances
        # observations = [mock_observation1, mock_observation2]

        flattener = ObservationFlattener()

        # Mock the Observation instances
        mock_observation1 = MagicMock()
        mock_observation2 = MagicMock()

        # Set return values for dict() to simulate observation data
        mock_observation1.dict.return_value = {
            "subject": {"id": "user1"},
            "effectiveDateTime": "2021-01-01",
            "valueQuantity": {"value": 100, "unit": "beats/min"},
            "code": {"coding": [{"code": "1234-5", "display": "Heart Rate"}]},
        }
        mock_observation2.dict.return_value = {
            "subject": {"id": "user2"},
            "effectiveDateTime": "2021-01-02",
            "valueQuantity": {"value": 120, "unit": "beats/min"},
            "code": {"coding": [{"code": "1234-5", "display": "Heart Rate"}]},
        }

        # Flatten the mock observations
        flattened_df = flattener.flatten([mock_observation1, mock_observation2])

        # Validate the flattened DataFrame
        self.assertTrue(isinstance(flattened_df, FHIRDataFrame))
        self.assertEqual(
            flattened_df.df.shape[0], 2
        )  # Assuming 2 observations were flattened
        self.assertTrue(ColumnNames.USER_ID.value in flattened_df.df.columns)


if __name__ == "__main__":
    unittest.main()
