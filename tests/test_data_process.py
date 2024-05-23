#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Unit tests for the FHIRDataProcessor class in the Spezi Data Pipeline.

This module contains unit tests for the FHIRDataProcessor class, ensuring that data processing,
including outlier filtering and processing based on specific LOINC code mappings, behaves as 
expected.

Classes:
    TestFHIRDataProcessor: Contains unit tests for the FHIRDataProcessor class.

Functions:
    setUp(self): Initializes any pre-requisites for the tests.
    test_process_fhir_data_valid_input(self): Tests processing of valid FHIR data.
    test_filter_outliers(self): Tests outlier filtering based on specific value ranges.
    test_select_data_by_user(self): Verifies the user ID filtering functionality.
    test_select_data_by_dates(self): Verifies the date filtering functionality.

Constants:
    USER_ID1 (str): Example user ID used in tests.
    OUTLIER_VALUE (float): Example outlier value used in tests.
    LOWER_THRESHOLD (int): Lower threshold for outlier detection.
    UPPER_THRESOLD (int): Upper threshold for outlier detection.
"""

# Standard library imports
from pathlib import Path
import random

# Related third-party imports
import unittest
from unittest.mock import MagicMock
import pandas as pd

# Local application/library specific imports
from spezi_data_pipeline.data_flattening.fhir_resources_flattener import (
    FHIRDataFrame,
    FHIRResourceType,
    ColumnNames,
)
from spezi_data_pipeline.data_processing.data_processor import (
    FHIRDataProcessor,
    select_data_by_dates,
    select_data_by_user,
)

from spezi_data_pipeline.data_processing.observation_processor import (
    calculate_daily_data,
)

USER_ID1 = "XrftRMc358NndzcRWEQ7P2MxvabZ"
OUTLIER_VALUE = 1e10
LOWER_THRESHOLD = 0
UPPER_THRESOLD = 15000


class TestFHIRDataProcessor(unittest.TestCase):  # pylint: disable=unused-variable
    """
    This class contains unit tests for the FHIRDataProcessor class, ensuring that data processing,
    including outlier filtering and processing based on specific LOINC code mappings, behaves as
    expected.
    """

    def setUp(self):
        """Initialize any pre-requisites for the tests."""
        self.processor = FHIRDataProcessor()
        data_file = Path(__file__).parent.parent / "sample_data" / "sample_df.csv"
        self.sample_data = pd.read_csv(data_file)

        # Ensure the EffectiveDateTime is converted correctly
        self.sample_data[ColumnNames.EFFECTIVE_DATE_TIME.value] = pd.to_datetime(
            self.sample_data[ColumnNames.EFFECTIVE_DATE_TIME.value]
        ).dt.date
        self.fhir_df = FHIRDataFrame(
            pd.DataFrame(self.sample_data), resource_type=FHIRResourceType.OBSERVATION
        )

        # Set a random row's QUANTITY_VALUE to OUTLIER_VALUE as an outlier
        random_index = random.choice(self.fhir_df.df.index)
        self.fhir_df.df.at[random_index, ColumnNames.QUANTITY_VALUE.value] = (
            OUTLIER_VALUE
        )

    def test_process_fhir_data_valid_input(self):
        """Test processing of valid FHIR data."""
        self.processor.code_processor = MagicMock()
        self.processor.code_processor.default_value_ranges = {
            "55423-8": (LOWER_THRESHOLD, UPPER_THRESOLD)
        }
        self.processor.code_processor.code_to_function = {
            "55423-8": calculate_daily_data
        }

        processed_df = self.processor.process_fhir_data(self.fhir_df)
        self.assertIsNotNone(processed_df)
        self.assertIsInstance(processed_df, FHIRDataFrame)

    def test_filter_outliers(self):
        """Test outlier filtering based on specific value ranges."""
        filtered_df = self.processor.filter_outliers(
            self.fhir_df, (LOWER_THRESHOLD, UPPER_THRESOLD)
        )

        # Check that the row with the outlier value has been removed
        self.assertTrue(
            all(
                value <= UPPER_THRESOLD
                for value in filtered_df.df[ColumnNames.QUANTITY_VALUE.value]
            )
        )
        self.assertTrue(
            OUTLIER_VALUE not in filtered_df.df[ColumnNames.QUANTITY_VALUE.value]
        )

        # Check that the number of rows is less than the original if there was indeed an outlier
        if OUTLIER_VALUE in self.fhir_df.df[ColumnNames.QUANTITY_VALUE.value].values:
            self.assertLess(len(filtered_df.df), len(self.fhir_df.df))

    def test_select_data_by_user(self):
        """Verify the user ID filtering functionality."""
        print("DataFrame before filtering by user:", self.fhir_df.df)
        selected_data = select_data_by_user(self.fhir_df, USER_ID1)
        print("DataFrame after filtering by user:", selected_data.df)
        self.assertEqual(
            len(selected_data.df),
            2,
            "Data selection by user ID did not work as expected",
        )
        self.assertTrue(
            (selected_data.df[ColumnNames.USER_ID.value] == USER_ID1).all(),
            "User ID filtering issue.",
        )

    def test_select_data_by_dates(self):
        """Verify the date filtering functionality."""
        print("DataFrame before filtering by dates:", self.fhir_df.df)
        selected_data = select_data_by_dates(self.fhir_df, "2023-01-01", "2024-01-02")
        print("DataFrame after filtering by dates:", selected_data.df)

        # Check that the dates in the filtered data are within the expected range
        expected_start_date = pd.to_datetime("2023-01-01").date()
        expected_end_date = pd.to_datetime("2024-01-02").date()
        selected_dates = pd.to_datetime(selected_data.df["EffectiveDateTime"]).dt.date
        self.assertTrue(
            (selected_dates >= expected_start_date).all()
            and (selected_dates <= expected_end_date).all(),
            "Data filtering by dates did not work as expected",
        )

        # Check if the filtered data contains the expected number of rows
        expected_number_of_rows = (
            3  # Change this to the number of expected rows in the sample_data DataFrame
        )
        self.assertEqual(
            len(selected_data.df),
            expected_number_of_rows,
            f"The number of rows after filtering should be exactly {expected_number_of_rows}.",
        )


if __name__ == "__main__":
    unittest.main()
