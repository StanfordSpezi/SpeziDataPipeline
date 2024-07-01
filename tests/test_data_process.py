#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module contains unit tests for various components of the Spezi Data Pipeline, 
focusing on the processing and analysis of FHIR (Fast Healthcare Interoperability Resources) data. 
The tests ensure the correct functionality of key data processing methods, including outlier
filtering, user-specific data selection, date-specific data selection, and the calculation of
risk scores from  questionnaire responses.

The module defines the following unit test classes:
1. `TestFHIRDataProcessor`: Tests for `FHIRDataProcessor` class, including:
   - Processing FHIR data with valid inputs.
   - Filtering outliers based on value ranges.
   - Selecting data by user ID.
   - Selecting data by date range.

2. `TestCalculateRiskScore`: Tests for `calculate_risk_score` function, including:
   - Calculating risk scores for PHQ-9, GAD-7, and WIQ questionnaires.
   - Handling unsupported questionnaires.

Constants used in the tests:
- USER_ID1: A sample user ID for testing.
- OUTLIER_VALUE: A value used to simulate outliers in the data.
- LOWER_THRESHOLD, UPPER_THRESHOLD: Threshold values for filtering outliers.
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

from spezi_data_pipeline.data_processing.questionnaire_processor import (
    calculate_risk_score,
    SupportedQuestionnaires,
    DepressionSeverity,
    AnxietySeverity,
    ImpairmentSeverity,
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

        self.sample_data[ColumnNames.EFFECTIVE_DATE_TIME.value] = pd.to_datetime(
            self.sample_data[ColumnNames.EFFECTIVE_DATE_TIME.value]
        ).dt.date
        self.fhir_df = FHIRDataFrame(
            pd.DataFrame(self.sample_data), resource_type=FHIRResourceType.OBSERVATION
        )

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

        self.assertTrue(
            all(
                value <= UPPER_THRESOLD
                for value in filtered_df.df[ColumnNames.QUANTITY_VALUE.value]
            )
        )
        self.assertTrue(
            OUTLIER_VALUE not in filtered_df.df[ColumnNames.QUANTITY_VALUE.value]
        )

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

        expected_start_date = pd.to_datetime("2023-01-01").date()
        expected_end_date = pd.to_datetime("2024-01-02").date()
        selected_dates = pd.to_datetime(selected_data.df["EffectiveDateTime"]).dt.date
        self.assertTrue(
            (selected_dates >= expected_start_date).all()
            and (selected_dates <= expected_end_date).all(),
            "Data filtering by dates did not work as expected",
        )

        expected_number_of_rows = 3
        self.assertEqual(
            len(selected_data.df),
            expected_number_of_rows,
            f"The number of rows after filtering should be exactly {expected_number_of_rows}.",
        )


class TestCalculateRiskScore(unittest.TestCase):  # pylint: disable=unused-variable
    """
    This class contains unit tests for the calculate_risk_score function, 
    which computes risk scores based on different questionnaire responses.

    The setUp method initializes sample data for testing purposes, including
    data for PHQ-9, GAD-7, and WIQ questionnaires. The data is formatted into 
    pandas DataFrames and further wrapped into FHIRDataFrames.

    Methods:
    --------
    setUp():
        Initializes sample data for PHQ-9, GAD-7, and WIQ questionnaires.
    
    test_calculate_phq9_score():
        Tests the calculate_risk_score function for PHQ-9 questionnaire responses.
    
    test_calculate_gad7_score():
        Tests the calculate_risk_score function for GAD-7 questionnaire responses.
    
    test_calculate_wiq_score():
        Tests the calculate_risk_score function for WIQ questionnaire responses.
    
    test_unsupported_questionnaire():
        Tests the calculate_risk_score function for handling unsupported questionnaire titles.
    """
    def setUp(self):
        # Sample data for PHQ-9 and GAD-7
        self.data_phq_gad = {
            ColumnNames.USER_ID.value: [1, 1, 1, 1],
            ColumnNames.AUTHORED_DATE.value: [
                "2023-01-01",
                "2023-01-01",
                "2023-01-01",
                "2023-01-01",
            ],
            ColumnNames.SURVEY_TITLE.value: ["PHQ-9", "PHQ-9", "PHQ-9", "PHQ-9"],
            ColumnNames.ANSWER_CODE.value: [2, 3, 1, 4],
        }
        self.df_phq_gad = pd.DataFrame(self.data_phq_gad)
        self.fhir_df_phq_gad = FHIRDataFrame(
            self.df_phq_gad, resource_type=FHIRResourceType("QuestionnaireResponse")
        )

        # Sample data for WIQ
        self.data_wiq = {
            ColumnNames.USER_ID.value: [1, 1, 1, 1],
            ColumnNames.AUTHORED_DATE.value: [
                "2023-01-01",
                "2023-01-01",
                "2023-01-01",
                "2023-01-01",
            ],
            ColumnNames.SURVEY_TITLE.value: ["WIQ", "WIQ", "WIQ", "WIQ"],
            ColumnNames.ANSWER_CODE.value: [50, 150, 300, 600],
        }
        self.df_wiq = pd.DataFrame(self.data_wiq)
        self.fhir_df_wiq = FHIRDataFrame(
            self.df_wiq, resource_type=FHIRResourceType("QuestionnaireResponse")
        )

    def test_calculate_phq9_score(self):
        result_df = calculate_risk_score(
            self.fhir_df_phq_gad, SupportedQuestionnaires.PHQ_9.value
        )
        expected_score = 10
        expected_interpretation = DepressionSeverity.MODERATE.description

        self.assertEqual(result_df.df["RiskScore"].iloc[0], expected_score)
        self.assertEqual(
            result_df.df["ScoreInterpretation"].iloc[0], expected_interpretation
        )

    def test_calculate_gad7_score(self):
        self.fhir_df_phq_gad.df[ColumnNames.SURVEY_TITLE.value] = "GAD-7"
        result_df = calculate_risk_score(
            self.fhir_df_phq_gad, SupportedQuestionnaires.GAD_7.value
        )
        expected_score = 10
        expected_interpretation = AnxietySeverity.MODERATE.description

        self.assertEqual(result_df.df["RiskScore"].iloc[0], expected_score)
        self.assertEqual(
            result_df.df["ScoreInterpretation"].iloc[0], expected_interpretation
        )

    def test_calculate_wiq_score(self):
        result_df = calculate_risk_score(
            self.fhir_df_wiq, SupportedQuestionnaires.WIQ.value
        )
        expected_score = (0 + 10 + 20 + 50) / 4
        expected_interpretation = ImpairmentSeverity.MODERATE_IMPAIRMENT.description

        self.assertEqual(result_df.df["RiskScore"].iloc[0], expected_score)
        self.assertEqual(
            result_df.df["ScoreInterpretation"].iloc[0], expected_interpretation
        )

    def test_unsupported_questionnaire(self):
        with self.assertRaises(ValueError) as context:
            calculate_risk_score(self.fhir_df_phq_gad, "Unsupported")

        actual_message = " ".join(str(context.exception).split())
        expected_message = (
            "Unsupported questionnaire title: Unsupported. Available options: "
            "PHQ-9, GAD-7, WIQ"
        )
        self.assertEqual(expected_message, actual_message)


if __name__ == "__main__":
    unittest.main()
