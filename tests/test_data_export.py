#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module contains unit tests for the DataExporter class, which is part of the data_export
module. The DataExporter class is responsible for exporting healthcare data structured as FHIR 
(Fast Healthcare Interoperability Resources) data into CSV files and generating visual plots based
on this data.

The tests in this module verify the functionality of the DataExporter by ensuring it can handle
various scenarios related to data export and visualization. This includes tests for exporting data
for single and multiple users,  and tests for behavior when no user ID is specified. Additional
tests cover the functionality of saving plots with specific settings, such as file format and
resolution.

These tests use the unittest framework and include the use of patches and mocks to isolate the
functionalities being tested, ensuring that the tests remain focused on the logic of the
DataExporter class without dependency on external files or the data visualization module's full
implementation.

Classes:
    TestDataExporter: Contains all unit tests for testing the functionalities of the DataExporter
        class.
"""


# Standard library imports
from pathlib import Path

# Related third-party imports
import unittest
from unittest.mock import patch
import pandas as pd

# Local application/library specific imports
from data_flattening.fhir_resources_flattener import FHIRDataFrame, FHIRResourceType
from data_export.data_exporter import DataExporter


class TestDataExporter(unittest.TestCase):  # pylint: disable=unused-variable
    """
    Test suite for the DataExporter class.

    This class tests the functionality of the DataExporter class, which includes exporting FHIR data
    to a CSV file and creating visualizations based on the data. The tests cover various scenarios
    including exporting data with single and multiple user IDs, and ensuring proper handling when
    no user ID is specified.
    """

    def setUp(self):
        data_file = Path(__file__).parent.parent / "sample_data" / "sample_df.csv"
        self.sample_data = pd.read_csv(data_file)
        self.fhir_df = FHIRDataFrame(self.sample_data, FHIRResourceType.OBSERVATION)
        self.exporter = DataExporter(self.fhir_df)
        self.exporter.user_ids = [
            "XrftRMc358NndzcRWEQ7P2MxvabZ",
            "sEmijWpn0vXe1cj60GO5kkjkrdT4",
        ]

    def test_initialization(self):
        """
        This test checks whether the DataExporter object is initialized correctly. It asserts that
        the flattened_fhir_dataframe attribute of the exporter is set to the FHIRDataFrame object
        passed during initialization, and it verifies that the user_ids attribute is set correctly.
        """
        self.assertIs(self.exporter.flattened_fhir_dataframe, self.fhir_df)
        self.assertEqual(
            self.exporter.user_ids,
            ["XrftRMc358NndzcRWEQ7P2MxvabZ", "sEmijWpn0vXe1cj60GO5kkjkrdT4"],
        )

    @patch("pandas.DataFrame.to_csv")
    def test_export_to_csv(self, mock_to_csv):
        """
        This test ensures that the export_to_csv method of the DataExporter class correctly calls
        the to_csv method of the pandas.DataFrame object with the specified filename and
        index=False. It uses the patch decorator from the unittest.mock module to mock the to_csv
        method.
        """
        filename = "test.csv"
        self.exporter.export_to_csv(filename)
        mock_to_csv.assert_called_once_with(filename, index=False)

    def test_create_filename(self):
        # Check filename creation with index
        filename = self.exporter.create_filename("base", "user1", 1)
        expected_filename = "base_user_user1_all_dates_fig1.png"
        self.assertEqual(filename, expected_filename)

        # Check without index
        filename = self.exporter.create_filename("base", "user1")
        expected_filename = "base_user_user1_all_dates.png"
        self.assertEqual(filename, expected_filename)

    @patch("matplotlib.figure.Figure.savefig")
    def test_create_and_save_plot(self, mock_savefig):
        self.exporter.create_and_save_plot("plot_base")
        # As plots creation depends on data, check calls based on expected conditions
        # This assumes at least one plot should be created given the loaded data and user_ids
        if self.exporter.user_ids:
            mock_savefig.assert_called()
        else:
            mock_savefig.assert_not_called()


if __name__ == "__main__":
    unittest.main()
