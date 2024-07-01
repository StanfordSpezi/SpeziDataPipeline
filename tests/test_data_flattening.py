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
on the `FHIRDataFrame` and `ObservationFlattener` classes from the data_flattening library.

The tests ensure the proper initialization, validation, and functionality of these classes.
For the `FHIRDataFrame`, tests validate the correct setup and validation of data within a
`DataFrame` tailored for FHIR resources, ensuring that data conforms to expected formats and
structures. For the ObservationFlattener, tests confirm the accurate transformation of complex
FHIR Observation resources into a simplified `DataFrame` format, suitable for further analysis
or processing.

These classes are crucial for handling healthcare data efficiently in a standardized format, and
the tests help ensure robustness and correctness in their implementation.

Classes:
    `TestFHIRDataFrame`: Tests initialization and validation of `FHIRDataFrame` instances.
    `TestObservationFlattener`: Tests the functionality of the `ObservationFlattener` class in
        converting FHIR observations into a simplified `DataFrame` format.
"""

# Standard library imports
import json
from pathlib import Path

# Related third-party imports
import unittest
from unittest.mock import MagicMock, patch

# pylint: disable=duplicate-code
import pandas as pd
from fhir.resources.R4B.observation import Observation
from fhir.resources.R4B.reference import Reference
from fhir.resources.R4B.questionnaireresponse import (
    QuestionnaireResponse,
    QuestionnaireResponseItem,
    QuestionnaireResponseItemAnswer,
)
from fhir.resources.R4B.coding import Coding

# Local application/library specific imports
from spezi_data_pipeline.data_flattening.fhir_resources_flattener import (
    FHIRDataFrame,
    FHIRResourceType,
    ColumnNames,
    ECGObservation,
    ObservationFlattener,
    ECGObservationFlattener,
    QuestionnaireResponseFlattener,
    extract_questionnaire_mappings,
    flatten_fhir_resources,
    get_answer_code_and_value,
    get_survey_title,
)

# pylint: enable=duplicate-code


class TestFHIRDataFrame(unittest.TestCase):  # pylint: disable=unused-variable
    """
    A test case for the FHIRDataFrame class, which is designed to handle and validate
    FHIR data in a pandas DataFrame format. This class ensures that the FHIRDataFrame
    can correctly initialize with data from a CSV file and that it contains all required columns.
    """

    def test_initialization(self):
        """
        Test the initialization of FHIRDataFrame with data loaded from a CSV file.
        Ensures that the DataFrame can be created and is recognized as an instance of FHIRDataFrame.
        """
        data_file = (
            Path(__file__).resolve().parent.parent / "sample_data" / "sample_df.csv"
        )

        data = pd.read_csv(data_file, dtype={"EffectiveDateTime": "object"})
        data["EffectiveDateTime"] = pd.to_datetime(
            data["EffectiveDateTime"], errors="coerce"
        )

        df = FHIRDataFrame(data, FHIRResourceType.OBSERVATION)
        self.assertIsInstance(df, FHIRDataFrame)

    def test_column_validation(self):
        """
        Tests the column validation of FHIRDataFrame to ensure that it correctly raises
        an error when required columns are missing or incorrect.
        """
        data_file = (
            Path(__file__).resolve().parent.parent / "sample_data" / "sample_df.csv"
        )
        data = pd.read_csv(data_file, dtype={"EffectiveDateTime": "object"})
        data["EffectiveDateTime"] = pd.to_datetime(
            data["EffectiveDateTime"], errors="coerce"
        )

        # Remove a required column to simulate the error condition
        data.drop(ColumnNames.USER_ID.value, axis=1, inplace=True)

        df = FHIRDataFrame(data, FHIRResourceType.OBSERVATION)
        with self.assertRaises(ValueError):
            df.validate_columns()


class TestObservationFlattener(unittest.TestCase):  # pylint: disable=unused-variable
    """
    Tests for the `ObservationFlattener` class, which processes a collection of Observation objects
    and converts them into a structured pandas `DataFrame`.
    """

    def test_flatten_observations(self):
        """
        Ensures that the `ObservationFlattener` can correctly flatten a list of Observation objects
        into a pandas `DataFrame`, verifying both the structure and content of the `DataFrame`.
        """
        resources = create_mock_observations()

        if isinstance(resources, str):
            self.fail(f"Failed to create mock observations: {resources}")

        flattener = ObservationFlattener()
        result = flattener.flatten(resources)

        self.assertEqual(len(result.df), 2)
        self.assertTrue(ColumnNames.USER_ID.value in result.df.columns)


class TestECGObservationFlattener(unittest.TestCase):  # pylint: disable=unused-variable
    """
    Tests for the `ECGObservationFlattener` class, specifically designed to handle ECG observations,
    converting them into a structured `DataFrame` with appropriate ECG-specific fields.
    """

    def test_flatten_ecg_observations(self):
        """
        Validates that `ECGObservationFlattener` correctly processes ECG observations, ensuring that
        the resulting `DataFrame` contains the correct number of rows and the specific column
        "ECGRecording".
        """
        resources = create_mock_ecg_observations()

        flattener = ECGObservationFlattener()
        result = flattener.flatten(resources)
        self.assertEqual(len(result.df), 2)
        self.assertTrue(ColumnNames.ECG_RECORDING.value in result.df.columns)


class TestQuestionnaireResponseFlattener(  # pylint: disable=unused-variable
    unittest.TestCase
):
    """
    A test suite for the QuestionnaireResponseFlattener class, which is responsible for
    transforming FHIR QuestionnaireResponse resources into a structured DataFrame format.

    The tests ensure that the flattening process correctly handles multiple questionnaire
    responses, translating them into a single DataFrame with accurate representation of
    each response item as a separate row.
    """

    def setUp(self):
        self.flattener = QuestionnaireResponseFlattener()

    @patch(
        "spezi_data_pipeline.data_flattening.fhir_resources_flattener"
        ".extract_questionnaire_mappings"
    )
    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data='{"title": "Test Survey"}',
    )
    def test_flatten(
        self, mock_open, mock_extract_mappings  # pylint: disable=unused-argument
    ):
        """
        Tests the flattening of a single QuestionnaireResponse resource.
        """
        mock_extract_mappings.return_value = (
            {"q1": "Question 1"},
            {"q1": {"1": "Answer 1"}},
        )

        mock_response = MagicMock(spec=QuestionnaireResponse)
        mock_response.subject = MagicMock()
        mock_response.subject.id = "user1"
        mock_response.id = "response1"
        mock_response.authored = "2024-06-01"
        mock_response.item = [
            MagicMock(
                spec=QuestionnaireResponseItem,
                linkId="q1",
                answer=[MagicMock(valueString="1")],
            )
        ]

        with patch(
            "spezi_data_pipeline.data_flattening.fhir_resources_flattener"
            ".get_answer_code_and_value",
            return_value={"code": "1", "text": "Answer 1"},
        ):
            result = self.flattener.flatten([mock_response], "survey_path.json")

        self.assertIsInstance(result, FHIRDataFrame)
        self.assertEqual(result.df.shape[0], 1)
        self.assertIn("QuestionText", result.df.columns)
        self.assertEqual(result.df.iloc[0]["QuestionText"], "Question 1")
        self.assertEqual(result.df.iloc[0]["AnswerText"], "Answer 1")

    def test_flatten_questionnaire_responses(self):
        """
        Tests the functionality of the QuestionnaireResponseFlattener flatten() method
        to ensure it accurately processes a list of `QuestionnaireResponse` objects into
        a structured `DataFrame`.

        This test checks:
        - That the flatten operation does not return None, confirming successful processing.
        - The number of rows in the resulting `DataFrame` matches the total number of items
          across all provided QuestionnaireResponse resources, ensuring that each item is
          correctly represented as a separate row in the `DataFrame`.

        The resources are created using a mock function, `create_mock_questionnaire_responses()`,
        which should return a list of QuestionnaireResponse objects or an error message if
        the resources cannot be generated.
        """
        resources = create_mock_questionnaire_responses()
        if isinstance(resources, str):
            self.fail(f"Failed to create mock questionnaire responses: {resources}")

        flattener = QuestionnaireResponseFlattener()
        result = flattener.flatten(
            resources, survey_path="Resources/SocialSupportQuestionnaire.json"
        )

        self.assertIsNotNone(result, "The resulting DataFrame should not be None")

        expected_rows = sum(
            len(response.item) for response in resources if hasattr(response, "item")
        )
        self.assertEqual(
            len(result.df),
            expected_rows,
            "The number of rows in the DataFrame should match the total number"
            "of items across all resources",
        )

    def test_get_answer_code_and_value(self):
        """
        Tests the get_answer_code_and_value function to ensure it retrieves the correct
        answer code and text for a given QuestionnaireResponseItem.
        """
        mock_coding = MagicMock(spec=Coding)
        mock_coding.code = "test"
        mock_coding.display = "Test Answer"

        mock_answer = MagicMock(spec=QuestionnaireResponseItemAnswer)
        mock_answer.valueCoding = mock_coding

        item = MagicMock(spec=QuestionnaireResponseItem)
        item.linkId = "q1"
        item.answer = [mock_answer]

        answer_map = {"q1": {"test": "Test Answer"}}

        result = get_answer_code_and_value(item, answer_map)

        self.assertEqual(result["code"], "test")
        self.assertEqual(result["text"], "Test Answer")

    @patch(
        "spezi_data_pipeline.data_flattening.fhir_resources_flattener.get_survey_title"
    )
    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data='{"title": "Test Survey"}',
    )
    @patch("json.load")
    def test_get_survey_title(
        self,
        mock_json_load,
        mock_open,  # pylint: disable=unused-argument
        mock_get_survey_title,  # pylint: disable=unused-argument
    ):
        """
        Tests the get_survey_title function to ensure it correctly retrieves the survey title
        from a Phoenix-generated JSON survey file.
        """
        mock_json_load.return_value = {"title": "Test Survey"}
        result = get_survey_title("survey_path.json")
        self.assertEqual(result, "Test Survey")

    @patch(
        "spezi_data_pipeline.data_flattening.fhir_resources_flattener.open",
        new_callable=unittest.mock.mock_open,
        read_data="{}",
    )
    @patch("json.load")
    def test_extract_questionnaire_mappings(
        self, mock_json_load, mock_open  # pylint: disable=unused-argument
    ):
        """
        Tests the extract_questionnaire_mappings function to ensure it correctly extracts
        question and answer mappings from a FHIR Questionnaire JSON file.
        """
        mock_json_load.return_value = {
            "item": [
                {
                    "linkId": "q1",
                    "text": "Question 1",
                    "answerOption": [{"valueString": "Answer 1"}],
                }
            ]
        }
        question_map, answer_map = extract_questionnaire_mappings("survey_path.json")
        self.assertEqual(question_map["q1"], "Question 1")
        self.assertEqual(answer_map["q1"]["Answer 1"], "Answer 1")

    @patch(
        "spezi_data_pipeline.data_flattening.fhir_resources_flattener"
        ".extract_questionnaire_mappings"
    )
    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="{}")
    def test_flatten_fhir_resources(
        self, mock_open, mock_extract_mappings  # pylint: disable=unused-argument
    ):
        """
        Tests the flatten_fhir_resources function to ensure it correctly flattens a list
        of FHIR resources into a structured DataFrame, handling empty and valid inputs.
        """
        mock_extract_mappings.return_value = (
            {"q1": "Question 1"},
            {"q1": {"1": "Answer 1"}},
        )

        result = flatten_fhir_resources([], "survey_path.json")
        self.assertIsNone(result)

        mock_resource = MagicMock(spec=QuestionnaireResponse)
        mock_resource.resource_type = "QuestionnaireResponse"
        mock_resource.subject = MagicMock()
        mock_resource.subject.id = "user1"
        mock_resource.authored = "2024-06-01"
        mock_resource.item = [
            MagicMock(
                spec=QuestionnaireResponseItem,
                linkId="q1",
                answer=[MagicMock(valueString="1")],
            )
        ]

        with patch(
            "spezi_data_pipeline.data_flattening.fhir_resources_flattener"
            ".get_answer_code_and_value",
            return_value={"code": "1", "text": "Answer 1"},
        ):
            result = flatten_fhir_resources([mock_resource], "survey_path.json")

        self.assertIsInstance(result, FHIRDataFrame)
        self.assertEqual(result.df.shape[0], 1)
        self.assertEqual(result.df.iloc[0]["QuestionText"], "Question 1")
        self.assertEqual(result.df.iloc[0]["AnswerText"], "Answer 1")


def create_mock_observations() -> list[Observation] | str:
    """
    Simulates the creation of `Observation` objects from JSON files. This function reads multiple
    JSON files, each representing a mock observation, and converts them into `Observation`
    instances.

    Returns:
        List[Observation]: A list of `Observation` objects if successful.
        str: Error message if the files cannot be read or parsed.
    """
    file_paths = [
        "sample_data/XrftRMc358NndzcRWEQ7P2MxvabZ_sample_data1.json",
        "sample_data/XrftRMc358NndzcRWEQ7P2MxvabZ_sample_data2.json",
    ]

    observations = []

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

                resource_str = json.dumps(data)

                resource_obj = Observation.parse_raw(resource_str)
                resource_obj.subject = Reference(id="XrftRMc358NndzcRWEQ7P2MxvabZ")

                observations.append(resource_obj)
        except FileNotFoundError:
            return f"The file {file_path} was not found."
        except json.JSONDecodeError:
            return "Failed to decode JSON from file."

    return observations


def create_mock_ecg_observations() -> list[ECGObservation] | str:
    """
    Creates a list of ECGObservation objects by reading and parsing JSON files that contain
    ECG data.

    Each JSON file is expected to represent an ECG observation, which includes various attributes
    necessary for constructing an ECGObservation object.

    If any file is not found or contains invalid JSON, the function will return an error message
    indicating the nature of the issue.

    Returns:
        List[ECGObservation]: A list of ECGObservation objects if all files are successfully read
            and parsed.
        str: An error message if an exception is encountered (e.g., file not found, JSON decode
            error).
    """
    file_paths = [
        "sample_data/3aX1qRKWQKTRDQZqr5vg5N7yWU12_sample_ecg_data1.json",
        "sample_data/3aX1qRKWQKTRDQZqr5vg5N7yWU12_sample_ecg_data2.json",
    ]

    ecg_observations = []

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                data.pop("issued", None)
                data.pop("document_id", None)
                data.pop("physicianAssignedDiagnosis", None)
                data.pop("physician", None)
                data.pop("tracingQuality", None)

                resource_str = json.dumps(data)

                resource_obj = Observation.parse_raw(resource_str)
                resource_obj.subject = Reference(id="3aX1qRKWQKTRDQZqr5vg5N7yWU12")
                ecg_resource_obj = ECGObservation(resource_obj)
                ecg_observations.append(ecg_resource_obj)
        except FileNotFoundError:
            return f"The file {file_path} was not found."
        except json.JSONDecodeError:
            return "Failed to decode JSON from file."

    return ecg_observations


def create_mock_questionnaire_responses() -> list[QuestionnaireResponse] | str:
    """
    Simulates the creation of QuestionnaireResponse objects from JSON files. This function
    reads multiple JSON files, each representing a mock questionnaire response, and converts
    them into QuestionnaireResponse instances.

    Returns:
        List[QuestionnaireResponse]: A list of QuestionnaireResponse objects if successful.
        str: Error message if the files cannot be read or parsed.
    """
    file_paths = [
        "sample_data/5tTYsEWMIKNq4EJEf24suVINGI12_sample_questionnaire_response_data1.json",
        "sample_data/5tTYsEWMIKNq4EJEf24suVINGI12_sample_questionnaire_response_data2.json",
    ]

    questionnaire_responses = []

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

                resource_str = json.dumps(data)

                resource_obj = QuestionnaireResponse.parse_raw(resource_str)
                resource_obj.subject = Reference(id="5tTYsEWMIKNq4EJEf24suVINGI12")

                questionnaire_responses.append(resource_obj)
        except FileNotFoundError:
            return f"The file {file_path} was not found."
        except json.JSONDecodeError:
            return "Failed to decode JSON from file."

    return questionnaire_responses


if __name__ == "__main__":
    unittest.main()
