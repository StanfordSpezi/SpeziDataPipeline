#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module provides test cases for the DataVisualizer class from the data_visualization module.

The tests focus on the initialization and configuration of the DataVisualizer instance, including
the capabilities to set specific date ranges, user IDs, and y-axis boundaries for data
visualization plots. It includes methods to test both the setting of parameters and the generation
of static plots, ensuring the correct application of combined and individual plotting based on the
instance's state.

The module makes use of unittest framework for structuring the test cases, employing mocks to
isolate and test functionality effectively without dependence on external data or the actual
plotting libraries.

Classes:
    TestDataVisualizer: Contains all the unit tests for testing the DataVisualizer functionalities.
"""

# Standard library imports
from pathlib import Path

# Related third-party imports
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

# Local application/library specific imports
from data_visualization.data_visualizer import DataVisualizer


class TestDataVisualizer(unittest.TestCase):  # pylint: disable=unused-variable
    """
    Test suite for the DataVisualizer class.

    This class tests the functionality of setting up a DataVisualizer instance, including setting
    date ranges, user IDs, and y-axis bounds for visualizations. It also tests the logic behind
    deciding whether to create combined or individual plots based on the DataVisualizer's state.
    """

    def test_set_date_range(self):
        """
        Test setting the date range for data visualization.

        Ensures that the start and end dates are correctly set in the DataVisualizer instance.
        """
        visualizer = DataVisualizer()
        start_date = "2021-01-01"
        end_date = "2021-01-31"
        visualizer.set_date_range(start_date, end_date)
        self.assertEqual(visualizer.start_date.strftime("%Y-%m-%d"), start_date)
        self.assertEqual(visualizer.end_date.strftime("%Y-%m-%d"), end_date)

    def test_set_user_ids(self):
        """
        Test setting the user IDs for data visualization.

        Verifies that the list of user IDs is correctly set in the DataVisualizer instance.
        """
        visualizer = DataVisualizer()
        user_ids = ["XrftRMc358NndzcRWEQ7P2MxvabZ", "sEmijWpn0vXe1cj60GO5kkjkrdT4"]
        visualizer.set_user_ids(user_ids)
        self.assertEqual(visualizer.user_ids, user_ids)

    def test_set_y_bounds(self):
        """
        Test setting the y-axis bounds for the plot.

        Checks that the lower and upper bounds of the y-axis are properly set in the
        DataVisualizer instance.
        """
        visualizer = DataVisualizer()
        visualizer.set_y_bounds(0, 500)
        self.assertEqual(visualizer.y_lower, 0)
        self.assertEqual(visualizer.y_upper, 500)

    @patch("data_visualization.data_visualizer.DataVisualizer.plot_combined")
    @patch("data_visualization.data_visualizer.DataVisualizer.plot_individual")
    def test_create_static_plot_combined(  # pylint: disable=no-self-use
        self, mock_plot_individual, mock_plot_combined
    ):
        """
        Test creating a combined static plot.

        Verifies that when the 'combine_plots' flag is True, a combined plot is created using the
        'plot_combined' method, and the 'plot_individual' method is not called.
        """
        visualizer = DataVisualizer()
        visualizer.combine_plots = True

        data_file = Path(__file__).parent.parent / "sample_data" / "sample_df.csv"
        df = pd.read_csv(data_file)
        mock_fhir_df = MagicMock()
        mock_fhir_df.df = df

        visualizer.create_static_plot(mock_fhir_df)
        mock_plot_combined.assert_called_once()
        mock_plot_individual.assert_not_called()

    # Add tests to handle incorrect date formats, empty user IDs, and edge cases


if __name__ == "__main__":
    unittest.main()
