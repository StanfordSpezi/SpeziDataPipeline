#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module provides test cases for the `DataExplorer` class from the data_exploration module.

The tests focus on the initialization and configuration of the `DataExplorer` instance, including
the capabilities to set specific date ranges, user IDs, and y-axis boundaries for data
exploration plots. It includes methods to test both the setting of parameters and the generation
of static plots, ensuring the correct application of combined and individual plotting based on the
instance's state.

The module makes use of unittest framework for structuring the test cases, employing mocks to
isolate and test functionality effectively without dependence on external data or the actual
plotting libraries.

Classes:
    `TestDataExplorer`: Contains all the unit tests for testing the `DataExplorer` functionalities.
"""

# Standard library imports
from pathlib import Path
from datetime import datetime

# Related third-party imports
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import matplotlib.pyplot as plt


# Local application/library specific imports
from spezi_data_pipeline.data_flattening.fhir_resources_flattener import (
    FHIRResourceType,
    FHIRDataFrame,
    ColumnNames,
)
from spezi_data_pipeline.data_exploration.data_explorer import (
    DataExplorer,
    QuestionnaireExplorer,
)


class TestDataExplorer(unittest.TestCase):  # pylint: disable=unused-variable
    """
    Test suite for the DataExplorer class.

    This class tests the functionality of setting up a DataExplorer instance, including setting
    date ranges, user IDs, and y-axis bounds for explorations. It also tests the logic behind
    deciding whether to create combined or individual plots based on the DataExplorer's state.
    """

    def test_set_date_range(self):
        """
        Test setting the date range for data exploration.

        Ensures that the start and end dates are correctly set in the DataExplorer instance.
        """
        visualizer = DataExplorer()
        start_date = "2021-01-01"
        end_date = "2021-01-31"
        visualizer.set_date_range(start_date, end_date)
        self.assertEqual(visualizer.start_date.strftime("%Y-%m-%d"), start_date)
        self.assertEqual(visualizer.end_date.strftime("%Y-%m-%d"), end_date)

    def test_set_user_ids(self):
        """
        Test setting the user IDs for data exploration.

        Verifies that the list of user IDs is correctly set in the DataExplorer instance.
        """
        visualizer = DataExplorer()
        user_ids = ["XrftRMc358NndzcRWEQ7P2MxvabZ", "sEmijWpn0vXe1cj60GO5kkjkrdT4"]
        visualizer.set_user_ids(user_ids)
        self.assertEqual(visualizer.user_ids, user_ids)

    def test_set_y_bounds(self):
        """
        Test setting the y-axis bounds for the plot.

        Checks that the lower and upper bounds of the y-axis are properly set in the
        DataExplorer instance.
        """
        visualizer = DataExplorer()
        visualizer.set_y_bounds(0, 500)
        self.assertEqual(visualizer.y_lower, 0)
        self.assertEqual(visualizer.y_upper, 500)

    @patch(
        "spezi_data_pipeline.data_exploration.data_explorer.DataExplorer.plot_combined"
    )
    @patch(
        "spezi_data_pipeline.data_exploration.data_explorer.DataExplorer.plot_individual"
    )
    def test_create_static_plot_combined(  # pylint: disable=no-self-use
        self, mock_plot_individual, mock_plot_combined
    ):
        """
        Test creating a combined static plot.

        Verifies that when the 'combine_plots' flag is True, a combined plot is created using the
        'plot_combined' method, and the 'plot_individual' method is not called.
        """
        visualizer = DataExplorer()
        visualizer.combine_plots = True

        data_file = Path(__file__).parent.parent / "sample_data" / "sample_df.csv"
        df = pd.read_csv(data_file)
        mock_fhir_df = MagicMock()
        mock_fhir_df.df = df

        visualizer.create_static_plot(mock_fhir_df)
        mock_plot_combined.assert_called_once()
        mock_plot_individual.assert_not_called()


class TestQuestionnaireExplorer(unittest.TestCase): # pylint: disable=unused-variable
    """
    Test suite for the QuestionnaireExplorer class.

    This class tests the functionality of setting up a QuestionnaireExplorer instance, including
    setting date ranges, user IDs, and generating score plots based on filtered data.
    """

    def setUp(self):
        self.explorer = QuestionnaireExplorer("Test Questionnaire")
        data = {
            ColumnNames.USER_ID.value: ["user1", "user2", "user1", "user2"],
            ColumnNames.AUTHORED_DATE.value: [
                "2023-01-01",
                "2023-01-02",
                "2023-02-01",
                "2023-02-02",
            ],
            ColumnNames.RISK_SCORE.value: [5, 6, 7, 8],
            ColumnNames.RESOURCE_ID.value: ["r1", "r2", "r3", "r4"],
            ColumnNames.SURVEY_TITLE.value: [
                "Survey1",
                "Survey1",
                "Survey1",
                "Survey1",
            ],
            ColumnNames.QUESTION_ID.value: ["q1", "q2", "q1", "q2"],
            ColumnNames.QUESTION_TEXT.value: [
                "Question 1",
                "Question 2",
                "Question 1",
                "Question 2",
            ],
            ColumnNames.ANSWER_CODE.value: [1, 2, 3, 4],
            ColumnNames.ANSWER_TEXT.value: [
                "Answer 1",
                "Answer 2",
                "Answer 3",
                "Answer 4",
            ],
        }
        df = pd.DataFrame(data)
        self.fhir_dataframe = FHIRDataFrame(
            df, resource_type=FHIRResourceType.QUESTIONNAIRE_RESPONSE
        )

    def test_set_date_range(self):
        self.explorer.set_date_range("2023-01-01", "2023-01-31")
        self.assertEqual(
            self.explorer.start_date, datetime.strptime("2023-01-01", "%Y-%m-%d").date()
        )
        self.assertEqual(
            self.explorer.end_date, datetime.strptime("2023-01-31", "%Y-%m-%d").date()
        )

    def test_set_user_ids(self):
        self.explorer.set_user_ids(["user1", "user2"])
        self.assertEqual(self.explorer.user_ids, ["user1", "user2"])

    def test_create_score_plot(self):
        self.explorer.set_date_range("2023-01-01", "2023-01-31")
        self.explorer.set_user_ids(["user1"])

        fig = self.explorer.create_score_plot(self.fhir_dataframe)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(
            len(fig.axes[0].lines), 1
        )  # Only one user's data should be plotted

    def test_no_data(self):
        self.explorer.set_date_range("2024-01-01", "2024-01-31")
        self.explorer.set_user_ids(["user3"])

        fig = self.explorer.create_score_plot(self.fhir_dataframe)
        self.assertIsNone(fig)


if __name__ == "__main__":
    unittest.main()
