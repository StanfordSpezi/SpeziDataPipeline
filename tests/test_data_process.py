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

2. `TestExtractLatestUserInteraction`: Tests for `extract_latest_user_interaction` function,
   including:
   - Extracting the latest interaction per user.
   - Ensuring the input DataFrame is not mutated.
   - Writing the result to a configurable CSV file.
   - Handling empty input.

3. `TestCalculateRiskScore`: Tests for `calculate_risk_score` function, including:
   - Calculating risk scores for PHQ-9, GAD-7, and WIQ questionnaires.
   - Handling unsupported questionnaires.

4. `TestCalculateActivityIndex`: Tests for `calculate_activity_index` function, including:
   - Calculating per-user n-day moving averages over gap-filled daily data.
   - Leaving the input DataFrame's row order unchanged.
   - Rejecting duplicate entries and non-step-count data.

Constants used in the tests:
- USER_ID1: A sample user ID for testing.
- OUTLIER_VALUE: A value used to simulate outliers in the data.
- LOWER_THRESHOLD, UPPER_THRESHOLD: Threshold values for filtering outliers.
"""

# Standard library imports
from pathlib import Path
import random
import tempfile

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
    extract_latest_user_interaction,
)

from spezi_data_pipeline.data_processing.observation_processor import (
    calculate_activity_index,
    calculate_daily_data,
    STEP_COUNT_LOINC_CODE,
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
WINDOW_DAYS = 2


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


class TestExtractLatestUserInteraction(  # pylint: disable=unused-variable
    unittest.TestCase
):
    """
    This class contains unit tests for the extract_latest_user_interaction function, which
    summarizes the most recent interaction recorded for each user in a FHIRDataFrame.
    """

    def setUp(self):
        """Initialize sample interaction data spanning multiple users and dates."""
        self.data = {
            ColumnNames.USER_ID.value: [
                "user1",
                "user1",
                "user2",
                "user2",
                "user2",
            ],
            ColumnNames.EFFECTIVE_DATE_TIME.value: [
                "2023-01-01",
                "2023-06-15",
                "2023-03-10",
                "2023-12-31",
                "2023-11-01",
            ],
        }
        self.fhir_df = FHIRDataFrame(
            pd.DataFrame(self.data), resource_type=FHIRResourceType.OBSERVATION
        )

    def test_extract_latest_user_interaction(self):
        """Verify that the latest interaction per user is returned."""
        result_df = extract_latest_user_interaction(self.fhir_df)

        self.assertEqual(len(result_df), 2)
        self.assertListEqual(
            list(result_df.columns),
            [ColumnNames.USER_ID.value, ColumnNames.LAST_USER_INTERACTION.value],
        )

        latest_by_user = dict(
            zip(
                result_df[ColumnNames.USER_ID.value],
                result_df[ColumnNames.LAST_USER_INTERACTION.value],
            )
        )
        self.assertEqual(latest_by_user["user1"], pd.Timestamp("2023-06-15"))
        self.assertEqual(latest_by_user["user2"], pd.Timestamp("2023-12-31"))

    def test_extract_latest_user_interaction_does_not_mutate_input(self):
        """Ensure the source FHIRDataFrame is left unchanged."""
        extract_latest_user_interaction(self.fhir_df)
        self.assertEqual(
            list(self.fhir_df.df[ColumnNames.EFFECTIVE_DATE_TIME.value]),
            self.data[ColumnNames.EFFECTIVE_DATE_TIME.value],
        )

    def test_extract_latest_user_interaction_writes_csv(self):
        """Verify that the result is written to the specified output file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file_path = Path(tmp_dir) / "latest_interactions.csv"
            result_df = extract_latest_user_interaction(
                self.fhir_df, output_file_path=str(output_file_path)
            )

            self.assertTrue(output_file_path.exists())
            written_df = pd.read_csv(output_file_path)
            self.assertEqual(len(written_df), len(result_df))
            self.assertListEqual(
                list(written_df.columns),
                [ColumnNames.USER_ID.value, ColumnNames.LAST_USER_INTERACTION.value],
            )

    def test_extract_latest_user_interaction_empty_input(self):
        """Verify that an empty input yields an empty result and writes no file."""
        empty_fhir_df = FHIRDataFrame(
            pd.DataFrame(
                columns=[
                    ColumnNames.USER_ID.value,
                    ColumnNames.EFFECTIVE_DATE_TIME.value,
                ]
            ),
            resource_type=FHIRResourceType.OBSERVATION,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file_path = Path(tmp_dir) / "should_not_exist.csv"
            result_df = extract_latest_user_interaction(
                empty_fhir_df, output_file_path=str(output_file_path)
            )

            self.assertTrue(result_df.empty)
            self.assertFalse(output_file_path.exists())


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
            ColumnNames.QUESTIONNAIRE_TITLE.value: ["PHQ-9", "PHQ-9", "PHQ-9", "PHQ-9"],
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
            ColumnNames.QUESTIONNAIRE_TITLE.value: ["WIQ", "WIQ", "WIQ", "WIQ"],
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
        self.fhir_df_phq_gad.df[ColumnNames.QUESTIONNAIRE_TITLE.value] = "GAD-7"
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


class TestCalculateActivityIndex(unittest.TestCase):  # pylint: disable=unused-variable
    """
    Tests for the calculate_activity_index function, which computes per-user n-day
    moving averages of step counts.
    """

    @staticmethod
    def make_step_data(rows):
        """Builds a step-count observation DataFrame from (user, date, value) tuples."""
        return pd.DataFrame(
            {
                ColumnNames.USER_ID.value: [user for user, _, _ in rows],
                ColumnNames.RESOURCE_ID.value: [
                    f"{user}-{date}" for user, date, _ in rows
                ],
                ColumnNames.EFFECTIVE_DATE_TIME.value: pd.to_datetime(
                    [date for _, date, _ in rows]
                ),
                ColumnNames.QUANTITY_NAME.value: "Step Count",
                ColumnNames.QUANTITY_UNIT.value: "steps",
                ColumnNames.QUANTITY_VALUE.value: [value for _, _, value in rows],
                ColumnNames.LOINC_CODE.value: STEP_COUNT_LOINC_CODE,
                ColumnNames.DISPLAY.value: "Number of steps",
                ColumnNames.APPLE_HEALTH_KIT_CODE.value: (
                    "HKQuantityTypeIdentifierStepCount"
                ),
            }
        )

    def setUp(self):
        # Rows are deliberately out of order, and userA has no entry for 2023-01-02.
        self.input_df = self.make_step_data(
            [
                ("userB", "2023-01-06", 150),
                ("userA", "2023-01-03", 300),
                ("userB", "2023-01-05", 50),
                ("userA", "2023-01-01", 100),
            ]
        )

    def test_moving_average_per_user(self):
        result = calculate_activity_index(
            FHIRDataFrame(self.input_df, FHIRResourceType.OBSERVATION),
            n=WINDOW_DAYS,
        )
        df = result.df.sort_values(
            [ColumnNames.USER_ID.value, ColumnNames.EFFECTIVE_DATE_TIME.value]
        ).reset_index(drop=True)

        self.assertEqual(result.resource_type, FHIRResourceType.OBSERVATION)
        self.assertEqual(
            list(df[ColumnNames.USER_ID.value]),
            ["userA", "userA", "userA", "userB", "userB"],
        )
        self.assertEqual(
            list(df[ColumnNames.EFFECTIVE_DATE_TIME.value]),
            list(
                pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-03",
                        "2023-01-05",
                        "2023-01-06",
                    ]
                )
            ),
        )
        # The missing day stays empty and the rolling mean skips it.
        self.assertEqual(
            list(df[ColumnNames.QUANTITY_VALUE.value]), [100, 100, 300, 50, 100]
        )
        self.assertTrue(
            (
                df[ColumnNames.QUANTITY_NAME.value]
                == f"{WINDOW_DAYS}-day moving average Step Count"
            ).all()
        )
        self.assertTrue(
            (df[ColumnNames.LOINC_CODE.value] == STEP_COUNT_LOINC_CODE).all()
        )

    def test_does_not_reorder_input(self):
        original_order = list(self.input_df[ColumnNames.RESOURCE_ID.value])
        calculate_activity_index(
            FHIRDataFrame(self.input_df, FHIRResourceType.OBSERVATION),
            n=WINDOW_DAYS,
        )
        self.assertEqual(
            list(self.input_df[ColumnNames.RESOURCE_ID.value]), original_order
        )

    def test_duplicate_entries_return_none(self):
        df = self.make_step_data(
            [("userA", "2023-01-01", 100), ("userA", "2023-01-01", 200)]
        )
        self.assertIsNone(
            calculate_activity_index(FHIRDataFrame(df, FHIRResourceType.OBSERVATION))
        )

    def test_non_step_data_returns_none(self):
        df = self.make_step_data([("userA", "2023-01-01", 100)])
        df[ColumnNames.LOINC_CODE.value] = "8867-4"
        self.assertIsNone(
            calculate_activity_index(FHIRDataFrame(df, FHIRResourceType.OBSERVATION))
        )


if __name__ == "__main__":
    unittest.main()
