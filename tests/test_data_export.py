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
from unittest.mock import patch, MagicMock
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

    @patch("pandas.DataFrame.to_csv")
    def test_export_to_csv(self, mock_to_csv):
        """Test exporting FHIR data to a CSV file."""
        exporter = DataExporter(self.fhir_df)
        exporter.export_to_csv("test.csv")

        mock_to_csv.assert_called_once_with("test.csv", index=False)

    @patch("data_visualization.data_visualizer.DataVisualizer.create_static_plot")
    def test_create_and_save_plot_single_user(self, mock_create_static_plot):
        """
        Test saving a plot for a single user.

        Ensures that the DataExporter can create and save a plot for a single specified user ID,
        checking that the plot is saved with the specified filename and dpi.
        """
        fig_mock = MagicMock()
        mock_create_static_plot.return_value = fig_mock

        exporter = DataExporter(self.fhir_df)
        exporter.user_ids = ["XrftRMc358NndzcRWEQ7P2MxvabZ"]
        exporter.create_and_save_plot("plot.png")

        fig_mock.savefig.assert_called_once_with("plot.png", dpi=300)
        mock_create_static_plot.assert_called_once()

    @patch("data_visualization.data_visualizer.DataVisualizer.create_static_plot")
    def test_create_and_save_plot_multiple_users(self, mock_create_static_plot):
        """
        Test attempting to save a plot with multiple user IDs specified.

        Checks that no plot is created or saved when multiple user IDs are specified, as the current
        implementation does not support visualizing data for more than one user at a time.
        """
        exporter = DataExporter(self.fhir_df)
        exporter.user_ids = [
            "XrftRMc358NndzcRWEQ7P2MxvabZ",
            "sEmijWpn0vXe1cj60GO5kkjkrdT4",
        ]
        exporter.create_and_save_plot("plot.png")

        mock_create_static_plot.assert_not_called()

    @patch("data_visualization.data_visualizer.DataVisualizer.create_static_plot")
    def test_create_and_save_plot_no_user(self, mock_create_static_plot):
        """
        Test attempting to save a plot with no user ID specified.

        Ensures that no plot is created or saved when no user ID is specified, highlighting the
        requirement for specifying at least one user ID for plot generation.
        """
        exporter = DataExporter(self.fhir_df)
        exporter.create_and_save_plot("plot.png")

        mock_create_static_plot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
