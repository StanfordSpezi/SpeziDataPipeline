#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#


"""
This module provides utilities and classes for transforming FHIR (Fast Healthcare Interoperability
Resources) data from its native hierarchical format into a flattened, tabular format. This
transformation facilitates easier data manipulation, analysis, and visualization by converting
complex FHIR resources into a more accessible and straightforward pandas DataFrame structure.

This module includes classes for processing specific types of FHIR resources, extracting relevant
information and standardizing it into a consistent format. Additionally, it defines enums and helper
functions to aid in the extraction and organization of data from these resources.

Main Components:
- `KeyNames` and `ColumnNames`: Enums that define standardized keys and column names used in the
                                flattening process, ensuring consistency across the application.
- `ResourceFlattener`: An abstract base class designed to be extended for specific FHIR resource
                       types, providing a common interface for the flattening operation.
- `ObservationFlattener` and `ECGObservationFlattener`: Concrete implementations of
                                                        `ResourceFlattener` tailored to handle
                                                        Observation and ECG Observation resources,
                                                        respectively.
- `extract_coding_info` and `extract_component_info`: Helper functions for extracting detailed
                                                      information from Observation components and
                                                      codings.
- `flatten_fhir_resources`: A utility function that orchestrates the flattening process, dynamically
                            selecting the appropriate flattener based on the resource type.
"""

# Standard library imports
from datetime import date
from enum import Enum
import re
import json

# Related third-party imports
from dataclasses import dataclass
from typing import Any
import pandas as pd
from fhir.resources.R4B.observation import Observation
from fhir.resources.R4B.questionnaireresponse import QuestionnaireResponse


SOCIAL_SUPPORT_QUESTIONNAIRE_PATH = "Resources/SocialSupportQuestionnaire.json"
PHQ9_PATH = "Resources/PHQ-9.json"
WALKING_IMPAIRMENT_QUESTIONNAIRE_PATH = "Resources/PAD Walking Impairment.json"
PHQ9_CODE_SYSTEM = "http://hl7.org/fhir/uv/sdc/CodeSystem/CSPHQ9"

ENCODING = "utf-8"
UNKNOWN_ANSWER_STRING = "Unknown Answer"
UNKNOWN_QUESTION_STRING = "Unknown Question"


class KeyNames(Enum):
    """
    Enumerates standardized key names in FHIR resources document snapshots.

    These keys represent common attributes found within FHIR resources. This
    enumeration facilitates the consistent access to these attributes
    across various parts of the codebase.

    Attributes:
        EFFECTIVE_DATE_TIME: The effective date and time of an observation or event.
        EFFECTIVE_PERIOD: The effective period over which an observation or event occurred.
        START: Start time within an effective period.
        CODE: The code that identifies the observation or measurement.
        CODING: The coding system used for the code attribute.
        DISPLAY: The display text associated with the code.
        VALUE_QUANTITY: The quantitative value of an observation.
        UNIT: The units of the VALUE_QUANTITY.
        VALUE: The actual value for VALUE_QUANTITY.
        ORIGIN: The origin point for VALUE_SAMPLED_DATA.
        DATA: The sampled data points in VALUE_SAMPLED_DATA.
        VALUE_SAMPLED_DATA: The complex datatype for sampled data.
        COMPONENT: The component part of an observation.
        VALUE_STRING: A string value associated with an observation.
        SYSTEM: The system that defines the coding system.
        RESOURCE_TYPE: The type of the FHIR resource.
        VALUE_CODING: Tthe answer to a question where the answer is a coded value.
        VALUE_INTEGER: A string value used when the response to a question is a free text string.
        ITEM: The index of the question item in a FHIR QuestionnaireResponse instance.
        LINK_ID: A key linking an answer to the corresponding question in a Questionnaire resource.
        TEXT: The text associated to an question or answer ID.
        ANSWER_OPTION: An answer option from a list of possible answers to a specific question.
    """

    EFFECTIVE_DATE_TIME = "effectiveDateTime"
    EFFECTIVE_PERIOD = "effectivePeriod"
    START = "start"
    ID = "id"
    CODE = "code"
    CODING = "coding"
    DISPLAY = "display"
    VALUE_QUANTITY = "valueQuantity"
    UNIT = "unit"
    VALUE = "value"
    ORIGIN = "origin"
    DATA = "data"
    VALUE_SAMPLED_DATA = "valueSampledData"
    COMPONENT = "component"
    VALUE_STRING = "valueString"
    SYSTEM = "system"
    RESOURCE_TYPE = "resourceType"
    VALUE_CODING = "valueCoding"
    VALUE_INTEGER = "valueInteger"
    ITEM = "item"
    LINK_ID = "linkId"
    TEXT = "text"
    ANSWER_OPTION = "answerOption"

    CONTAINED = "contained"
    VALUE_SET = "ValueSet"
    COMPOSE = "compose"
    INCLUDE = "include"
    CONCEPT = "concept"
    EXTENSION = "extension"
    VALUE_DECIMAL = "valueDecimal"
    TITLE = "title"
    URL = "url"


class ColumnNames(Enum):
    """
    Enumerates standardized column names for use in flattened DataFrames.

    These column names are designed for handling data extracted from FHIR resources, especially for
    observations related to ECG and other health metrics. Standardizing these names helps ensure
    consistency in data processing and analysis tasks.

    Attributes:
        USER_ID: Identifier for the patient or subject of the observation.
        RESOURCE_ID: Identifier for the resource.
        EFFECTIVE_DATE_TIME: The datetime when the observation was effective.
        QUANTITY_NAME: Name or description of the observed quantity.
        QUANTITY_UNIT: Units of the observed quantity.
        QUANTITY_VALUE: Numeric value of the observed quantity.
        LOINC_CODE: LOINC code associated with the observation.
        DISPLAY: Display text associated with the LOINC or other code.
        APPLE_HEALTH_KIT_CODE: Specific code used in Apple HealthKit.
        NUMBER_OF_MEASUREMENTS: Number of measurements taken.
        SAMPLING_FREQUENCY: Frequency at which data was sampled.
        SAMPLING_FREQUENCY_UNIT: Unit for the sampling frequency.
        ELECTROCARDIOGRAM_CLASSIFICATION: Classification of the ECG observation.
        HEART_RATE: Observed heart rate.
        HEART_RATE_UNIT: Unit of the observed heart rate.
        ECG_RECORDING_UNIT: Unit for ECG recording data.
        ECG_RECORDING: ECG recording data.
        AUTHORED_DATE: The date when the QuestionnaireResponse was authored or completed.
        SURVEY_TITLE: The title of the survey or questionnaire.
        SURVEY_DATE: The date when the survey was conducted or the date relevant to the survey.
        QUESTION_ID: The unique identifier for a specific question within the survey.
        QUESTION_TEXT: The text or content of the question being asked.
        ANSWER_TEXT: The text or content of the respondent's answer.
    """

    USER_ID = "UserId"
    RESOURCE_ID = "ResourceId"
    EFFECTIVE_DATE_TIME = "EffectiveDateTime"
    QUANTITY_NAME = "QuantityName"
    QUANTITY_UNIT = "QuantityUnit"
    QUANTITY_VALUE = "QuantityValue"
    LOINC_CODE = "LoincCode"
    DISPLAY = "Display"
    APPLE_HEALTH_KIT_CODE = "AppleHealthKitCode"
    NUMBER_OF_MEASUREMENTS = "NumberOfMeasurements"
    SAMPLING_FREQUENCY = "SamplingFrequency"
    SAMPLING_FREQUENCY_UNIT = "SamplingFrequencyUnit"
    ELECTROCARDIOGRAM_CLASSIFICATION = "ElectrocardiogramClassification"
    HEART_RATE = "HeartRate"
    HEART_RATE_UNIT = "HeartRateUnit"
    ECG_RECORDING_UNIT = "ECGDataRecordingUnit"
    ECG_RECORDING = "ECGRecording"
    AUTHORED_DATE = "AuthoredDate"
    SURVEY_TITLE = "SurveyTitle"
    SURVEY_DATE = "Date"
    QUESTION_ID = "QuestionId"
    QUESTION_TEXT = "QuestionText"
    ANSWER_CODE = "AnswerCode"
    ANSWER_TEXT = "AnswerText"


# pylint: disable=too-few-public-methods
class ECGObservation:  # pylint: disable=unused-variable
    """
    A wrapper class for FHIR ECG observations.

    This class provides a convenient interface to interact with ECG observation data encapsulated
    within FHIR resources. It abstracts away the complexities of the FHIR data model, offering
    direct access to attributes relevant for ECG data processing and visualization.

    Parameters:
        observation (Obervation): The original FHIR observation object containing ECG data.
    """

    def __init__(self, observation: Any):
        """
        Initializes an ECGObservation wrapper for FHIR ECG observations.

        Parameters:
            resource (Any): The original FHIR observation object.
            resource_type (FHIRResourceType): An enum member representing the
                                                       specific type of the FHIR resource.
        """
        self.observation = observation
        self.resource_type = FHIRResourceType.ECG_OBSERVATION.value

    def __getattr__(self, name):
        """
        Allows attribute access to be delegated to the underlying FHIR observation object.

        This method provides dynamic access to the observation's attributes, making it easier
        to retrieve values for standard and custom attributes defined within the FHIR resource.

        Parameters:
            name (str): The attribute name to access from the observation object.

        Returns:
            The value of the attribute if it exists; otherwise, AttributeError is raised.
        """
        return getattr(self.observation, name)


class FHIRResourceType(Enum):
    """
    Enumeration of FHIR resource types relevant to the application.

    This enum helps ensure that the application works with a consistent set of FHIR resource
    types, reducing the risk of errors due to typos and providing a centralized definition
    of resource types used throughout the codebase.

    Attributes:
        OBSERVATION: Represents observation resources, commonly used for measurements and
            findings.
        ECG_OBSERVATION: A specialized observation type for electrocardiogram data.
        QUESTIONNAIRE_RESPONSE: Represents responses to questionnaires.
    """

    OBSERVATION = "Observation"
    ECG_OBSERVATION = "ElectrocardiographicObservation"
    QUESTIONNAIRE_RESPONSE = "QuestionnaireResponse"


@dataclass
class FHIRDataFrame:
    """
    Represents a DataFrame specifically designed to handle FHIR data.

    This class wraps around a pandas DataFrame, organizing FHIR data into a structured,
    tabular format that is easy to manipulate and analyze. It includes methods for validating
    the data structure and ensuring it meets expected requirements for FHIR-based analysis.

    Parameters:
        data (pd.DataFrame): The DataFrame containing structured FHIR data.
        resource_type (FHIRResourceType): The type of FHIR resources represented in the
            DataFrame.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        resource_type: FHIRResourceType,
    ) -> None:
        """
        Initializes the FHIRDataFrame with structured FHIR data and resource type.
        """
        self.data_frame = data
        self.resource_type = resource_type

        flattener = ResourceFlattener(resource_type)
        self.resource_columns = flattener.resource_columns

        if resource_type not in self.resource_columns:
            raise ValueError(f"Unsupported resource type: {resource_type}")

    @property
    def df(self) -> pd.DataFrame:
        """
        A property to access the underlying pandas DataFrame containing FHIR data.

        Returns:
            pd.DataFrame: The pandas DataFrame storing the FHIR data.
        """
        return self.data_frame

    def validate_columns(self) -> bool:
        """
        Validates that the DataFrame contains all required columns for processing and
        checks the data type of specific columns. Raises a ValueError if any required
        column is missing or if a column does not have the expected data type.

        Returns:
            bool: True if all columns are present and correctly formatted, otherwise raises
                an error.
        """

        required_columns = [
            col.value for col in self.resource_columns.get(self.resource_type, [])
        ]

        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]
        if missing_columns:
            formatted_missing_columns = ", ".join(missing_columns)
            raise ValueError(
                f"The DataFrame is missing required columns: {formatted_missing_columns}"
            )

        if ColumnNames.EFFECTIVE_DATE_TIME.value in self.df.columns:
            if not all(
                isinstance(d, date)
                for d in self.df[ColumnNames.EFFECTIVE_DATE_TIME.value]
            ):
                raise ValueError(
                    f"The {ColumnNames.EFFECTIVE_DATE_TIME.value} column is not of type"
                    "datetime.date."
                )

        return True


@dataclass
class ResourceFlattener:
    """
    Base class for transforming FHIR resources into a structured DataFrame format.

    Subclasses of ResourceFlattener implement specific logic for flattening
    different types of FHIR resources, converting them into a format that's suitable
    for data analysis and processing tasks.

    Parameters:
        resource_type (FHIRResourceType): The type of FHIR resources this flattener handles.
    """

    def __init__(self, resource_type: FHIRResourceType):
        """
        Initializes the ResourceFlattener with a specific FHIR resource type.

        This method sets up the resource type and prepares a dictionary mapping the resource type
        to its relevant columns for the flattening process. It validates the resource type to ensure
        that it is supported for flattening.

        Parameters:
            resource_type (FHIRResourceType): The type of FHIR resource this flattener will handle.

        Raises:
            ValueError: If the specified resource type is unsupported.
        """
        self.resource_type = resource_type
        self.resource_columns = {
            FHIRResourceType.OBSERVATION: [
                ColumnNames.USER_ID,
                ColumnNames.RESOURCE_ID,
                ColumnNames.EFFECTIVE_DATE_TIME,
                ColumnNames.QUANTITY_NAME,
                ColumnNames.QUANTITY_UNIT,
                ColumnNames.QUANTITY_VALUE,
                ColumnNames.LOINC_CODE,
                ColumnNames.DISPLAY,
                ColumnNames.APPLE_HEALTH_KIT_CODE,
            ],
            FHIRResourceType.ECG_OBSERVATION: [
                ColumnNames.USER_ID,
                ColumnNames.RESOURCE_ID,
                ColumnNames.EFFECTIVE_DATE_TIME,
                ColumnNames.QUANTITY_NAME,
                ColumnNames.NUMBER_OF_MEASUREMENTS,
                ColumnNames.SAMPLING_FREQUENCY,
                ColumnNames.SAMPLING_FREQUENCY_UNIT,
                ColumnNames.ELECTROCARDIOGRAM_CLASSIFICATION,
                ColumnNames.HEART_RATE,
                ColumnNames.HEART_RATE_UNIT,
                ColumnNames.ECG_RECORDING_UNIT,
                ColumnNames.ECG_RECORDING,
                ColumnNames.LOINC_CODE,
                ColumnNames.DISPLAY,
                ColumnNames.APPLE_HEALTH_KIT_CODE,
            ],
            FHIRResourceType.QUESTIONNAIRE_RESPONSE: [
                ColumnNames.USER_ID,
                ColumnNames.RESOURCE_ID,
                ColumnNames.AUTHORED_DATE,
                ColumnNames.SURVEY_TITLE,
                ColumnNames.SURVEY_DATE,
                ColumnNames.QUESTION_ID,
                ColumnNames.QUESTION_TEXT,
                ColumnNames.ANSWER_CODE,
                ColumnNames.ANSWER_TEXT,
            ],
        }

        if resource_type not in self.resource_columns:
            raise ValueError(f"Unsupported resource type: {resource_type.name}")

    def flatten(
        self, resources: list[Any], survey_path: str | None = None
    ) -> FHIRDataFrame:
        """
        Abstract method intended to transform a list of FHIR resources into a flattened
        FHIRDataFrame.

        This method should be implemented by subclasses to process specific types of FHIR resources,
        extracting relevant data and converting it into a structured, tabular format. The
        implementation will vary depending on the resource type, focusing on the extraction of key
        information suited for analysis and further processing.

        Parameters:
            resources (list[Any]): A list of FHIR resource objects to be flattened. The exact type
                                    of objects in the list should correspond to the FHIR resource
                                    type the subclass is designed to handle.
            survey_path (str): Survey path used in the QuestionnaireResponseFlattener to
                create survey mappings.

        Returns:
            FHIRDataFrame: A FHIRDataFrame object containing the flattened data from the resources,
                            ready for analysis and processing.

        Raises:
            NotImplementedError: Indicates that this method needs to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")


@dataclass
class ObservationFlattener(ResourceFlattener):
    """
    Handles the flattening of FHIR Observation resources into a structured DataFrame
    format, facilitating analysis and visualization. It extracts crucial information such
    as patient IDs, observation timestamps, and measurement values, converting them from
    the FHIR format into a more accessible tabular form.

    Inherits:
        ResourceFlattener: The base class for resource flatteners, providing common
        functionality and attributes.
    """

    def __init__(self):
        """
        Initializes the ObservationFlattener for processing FHIR Observation resources.
        """
        super().__init__(FHIRResourceType.OBSERVATION)

    def flatten(
        self, resources: list[Observation], survey_path: str | None = None
    ) -> FHIRDataFrame:
        """
        Converts a list of FHIR Observation resources into a structured DataFrame.
        Extracts and organizes information from observations into a format suitable for analysis.

        Parameters:
            resources (list[Observation]): A collection of FHIR Observation resources to be
                flattened.
            survey_path (str): Survey path not relevant to Observation flattener.

        Returns:
            FHIRDataFrame: A DataFrame containing structured data extracted from the input
                resources.
        """

        flattened_data = []
        for observation in resources:

            if not (
                effective_datetime := observation.dict().get(
                    KeyNames.EFFECTIVE_DATE_TIME.value
                )
            ):
                effective_period = observation.dict().get(
                    KeyNames.EFFECTIVE_PERIOD.value, {}
                )
                effective_datetime = effective_period.get(KeyNames.START.value, None)

            coding_info = extract_coding_info(observation)

            flattened_entry = {
                ColumnNames.USER_ID.value: observation.subject.id,
                ColumnNames.RESOURCE_ID.value: observation.id,
                ColumnNames.EFFECTIVE_DATE_TIME.value: (
                    effective_datetime if effective_datetime else None
                ),
                **coding_info,
                ColumnNames.QUANTITY_UNIT.value: observation.dict()
                .get(KeyNames.VALUE_QUANTITY.value, {})
                .get(KeyNames.UNIT.value, None),
                ColumnNames.QUANTITY_VALUE.value: observation.dict()
                .get(KeyNames.VALUE_QUANTITY.value, {})
                .get(KeyNames.VALUE.value, None),
            }

            flattened_data.append(flattened_entry)

        flattened_df = pd.DataFrame(flattened_data)

        # Convert to UTC, remove timezone info, and then extract the date
        flattened_df[ColumnNames.EFFECTIVE_DATE_TIME.value] = (
            pd.to_datetime(
                flattened_df[ColumnNames.EFFECTIVE_DATE_TIME.value],
                errors="coerce",
                utc=True,
            )
            .dt.tz_convert(None)
            .dt.date
        )

        return FHIRDataFrame(flattened_df, FHIRResourceType.OBSERVATION)


@dataclass
class ECGObservationFlattener(ResourceFlattener):
    """
    Specializes in flattening FHIR ECG Observation resources, transforming complex ECG data
    into a structured DataFrame. This includes handling multi-component observations and
    extracting relevant metrics for further analysis.

    Inherits:
        ResourceFlattener: The base class providing foundational flattening functionality.
    """

    def __init__(self):
        """
        Initializes the ECGObservationFlattener specifically for ECG Observation resources.
        """
        super().__init__(FHIRResourceType.ECG_OBSERVATION)

    def flatten(
        self, resources: list[ECGObservation], survey_path: str = None
    ) -> FHIRDataFrame:
        """
        Flattens a list of ECG Observation resources, extracting ECG data and related metrics
        into a structured, analyzable DataFrame format.

        Parameters:
            resources (list[ECGObservation]): A collection of FHIR ECG Observation resources.
            survey_path (str): Survey path not relevant to ECG flattener.

        Returns:
            FHIRDataFrame: A DataFrame containing structured ECG data from the input resources.
        """
        flattened_data = []
        for observation in resources:

            if not (
                effective_datetime := observation.dict().get(
                    KeyNames.EFFECTIVE_DATE_TIME.value
                )
            ):
                effective_period = observation.dict().get(
                    KeyNames.EFFECTIVE_PERIOD.value, {}
                )
                effective_datetime = effective_period.get(KeyNames.START.value, None)

            coding_info = extract_coding_info(observation)
            component_info = extract_component_info(observation)

            flattened_entry = {
                ColumnNames.USER_ID.value: observation.subject.id,
                ColumnNames.RESOURCE_ID.value: observation.id,
                ColumnNames.EFFECTIVE_DATE_TIME.value: effective_datetime,
                ColumnNames.NUMBER_OF_MEASUREMENTS.value: observation.dict()
                .get(KeyNames.COMPONENT.value, [{}])[0]
                .get(KeyNames.VALUE_QUANTITY.value, {})
                .get(KeyNames.VALUE.value, None),
                ColumnNames.SAMPLING_FREQUENCY.value: observation.dict()
                .get(KeyNames.COMPONENT.value, [{}])[1]
                .get(KeyNames.VALUE_QUANTITY.value, {})
                .get(KeyNames.VALUE.value, None),
                ColumnNames.SAMPLING_FREQUENCY_UNIT.value: observation.dict()
                .get(KeyNames.COMPONENT.value, [{}])[1]
                .get(KeyNames.VALUE_QUANTITY.value, {})
                .get(KeyNames.UNIT.value, None),
                ColumnNames.ELECTROCARDIOGRAM_CLASSIFICATION.value: observation.dict()
                .get(KeyNames.COMPONENT.value, [{}])[2]
                .get(KeyNames.VALUE_STRING.value, None),
                ColumnNames.HEART_RATE.value: observation.dict()
                .get(KeyNames.COMPONENT.value, [{}])[3]
                .get(KeyNames.VALUE_QUANTITY.value, {})
                .get(KeyNames.VALUE.value, None),
                ColumnNames.HEART_RATE_UNIT.value: observation.dict()
                .get(KeyNames.COMPONENT.value, [{}])[3]
                .get(KeyNames.VALUE_QUANTITY.value, {})
                .get(KeyNames.UNIT.value, None),
                **coding_info,
                **component_info,
            }
            flattened_data.append(flattened_entry)

        flattened_df = pd.DataFrame(flattened_data)
        flattened_df[ColumnNames.EFFECTIVE_DATE_TIME.value] = (
            pd.to_datetime(
                flattened_df[ColumnNames.EFFECTIVE_DATE_TIME.value],
                errors="coerce",
                utc=True,
            )
            .dt.tz_convert(None)
            .dt.date
        )

        return FHIRDataFrame(flattened_df, FHIRResourceType.ECG_OBSERVATION)


def extract_coding_info(observation: Observation | ECGObservation) -> dict:
    """
    Extracts coding information from an Observation resource, focusing on key details
    like LOINC codes and Apple HealthKit codes.

    Parameters:
        observation (Observation | ECGObservation): The FHIR Observation resource from
            which to extract coding information.

    Returns:
        dict: A dictionary containing extracted coding details such as LOINC code,
            Apple HealthKit code, and display text.
    """
    coding = (
        observation.dict().get(KeyNames.CODE.value, {}).get(KeyNames.CODING.value, [])
    )

    loinc_code = None
    apple_health_kit_code = None
    display_text = None
    quantity_name = None

    if coding:
        quantity_name = (
            coding[1][KeyNames.DISPLAY.value]
            if len(coding) > 1
            else (coding[0][KeyNames.DISPLAY.value] if len(coding) > 0 else None)
        )

    for code_info in coding:
        code = code_info.get(KeyNames.CODE.value, "")
        display = code_info.get(KeyNames.DISPLAY.value, "")

        if re.fullmatch(r"\d+(-\d+)?", code):
            loinc_code = code
            display_text = display

        elif re.fullmatch(r"[A-Za-z]+", code):
            apple_health_kit_code = code

    return {
        ColumnNames.QUANTITY_NAME.value: quantity_name,
        ColumnNames.LOINC_CODE.value: loinc_code,
        ColumnNames.DISPLAY.value: display_text,
        ColumnNames.APPLE_HEALTH_KIT_CODE.value: apple_health_kit_code,
    }


def extract_component_info(observation: ECGObservation) -> dict:
    """
    Extracts information from components of an ECG Observation, relevant for detailed ECG
    data analysis.

    Parameters:
        observation (ECGObservation): The FHIR ECG Observation resource containing component data.

    Returns:
        dict: A dictionary with structured information extracted from ECG components,
        including a single merged ECG recording data string and the unit of measurement.
    """
    component_info = {}
    components = observation.dict().get(KeyNames.COMPONENT.value, [])

    merged_ecg_data = ""
    unit = None

    for component in components:
        value_sampled_data = component.get(KeyNames.VALUE_SAMPLED_DATA.value, {})

        data = value_sampled_data.get(KeyNames.DATA.value, None)
        if data is not None:  # pylint: disable=consider-using-assignment-expr
            merged_ecg_data += data + " "  # Adding a space for separation

        if unit is None:
            origin = value_sampled_data.get(KeyNames.ORIGIN.value, {})
            unit = origin.get(KeyNames.UNIT.value, None)

    component_info[ColumnNames.ECG_RECORDING.value] = merged_ecg_data.strip()
    component_info[ColumnNames.ECG_RECORDING_UNIT.value] = unit

    return component_info


@dataclass
class QuestionnaireResponseFlattener(ResourceFlattener):
    """
    Flattens QuestionnaireResponse and creates a structured DataFrame by mapping
    question and answer IDs to the original Phoenix-generated surveys.

    Inherits:
        ResourceFlattener: The base class providing foundational flattening functionality.
    """

    def __init__(self):
        """
        Initializes the ECGObservationFlattener specifically for ECG Observation resources.
        """
        super().__init__(FHIRResourceType.QUESTIONNAIRE_RESPONSE)

    def flatten(
        self, resources: list[QuestionnaireResponse], survey_path: str = None
    ) -> FHIRDataFrame:
        """
        Flattens a list of QuestionnaireResponse resources, extracting all question and answers
        for each user and converting it into text to create the DataFrame

        Parameters:
            resources (list[QuestionnaireResponse]): A collection of QuestionnaireResponse
                resources.
            survey_path (str): The path to Phoenix-generated JSON survey used to extract
                relevant mappings.

        Returns:
            FHIRDataFrame: A DataFrame containing structured QuestionnaireResponse data from the
                input resources.
        """

        if not survey_path:
            print("The path(s) to Phoenix-generated JSON surveys is missing.")
            return None
        all_question_mappings, all_answer_mappings = extract_mappings(survey_path)
        flattened_data = []

        for response in resources:
            for item in response.item:
                question_id = item.linkId
                question_text = all_question_mappings.get(
                    question_id, UNKNOWN_QUESTION_STRING
                )

                answer_code, answer_value = get_answer_code_and_value(
                    item, all_answer_mappings, survey_path
                )

                flattened_entry = {
                    ColumnNames.USER_ID.value: getattr(
                        response.subject, KeyNames.ID.value, None
                    ),
                    ColumnNames.RESOURCE_ID.value: getattr(
                        response, KeyNames.ID.value, None
                    ),
                    ColumnNames.AUTHORED_DATE.value: response.authored,
                    ColumnNames.SURVEY_TITLE.value: get_survey_title(survey_path),
                    ColumnNames.QUESTION_ID.value: question_id,
                    ColumnNames.QUESTION_TEXT.value: question_text,
                    ColumnNames.ANSWER_CODE.value: answer_code,
                    ColumnNames.ANSWER_TEXT.value: answer_value,
                }

                flattened_data.append(flattened_entry)

        flattened_df = pd.DataFrame(flattened_data)
        return FHIRDataFrame(flattened_df, FHIRResourceType.QUESTIONNAIRE_RESPONSE)


def extract_mappings(survey_path: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    Extracts question and answer mappings from Phoenix-generated JSON surveys.

    Parameters:
        survey_path (str): The path to Phoenix-generated JSON survey.

    Returns:
        tuple[dict[str, str], dict[str, str]]: A tuple containing question and answer mappings.
    """
    question_mapping = {}
    answer_mapping = {}

    with open(survey_path, "r", encoding=ENCODING) as file:
        try:
            json_content = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {survey_path}: {e}")

        if survey_path == SOCIAL_SUPPORT_QUESTIONNAIRE_PATH:
            question_mapping, answer_mapping = process_social_support_questionnaire(
                json_content
            )
        if survey_path == PHQ9_PATH:
            question_mapping, answer_mapping = process_phq9_questionnaire(json_content)
        elif survey_path == WALKING_IMPAIRMENT_QUESTIONNAIRE_PATH:
            question_mapping, answer_mapping = process_wiq_questionnaire(json_content)
        else:
            print("Unknown Questionnaire Type")

        return question_mapping, answer_mapping


def process_wiq_questionnaire(json_content):
    """
    Performs special handling to extract the mappings for Walking Impairment
    Questionnaire (WIQ).
    """

    question_mapping = {}
    answer_mapping = {}

    for contained in json_content[KeyNames.CONTAINED.value]:
        if contained.get(KeyNames.RESOURCE_TYPE.value) == KeyNames.VALUE_SET.value:
            system = (
                contained.get(KeyNames.COMPOSE.value, {})
                .get(KeyNames.INCLUDE.value, [{}])[0]
                .get(KeyNames.SYSTEM.value)
            )
            for concept in (
                contained.get(KeyNames.COMPOSE.value, {})
                .get(KeyNames.INCLUDE.value, [{}])[0]
                .get(KeyNames.CONCEPT.value, [])
            ):
                code = concept.get(KeyNames.CODE.value)
                display = concept.get(KeyNames.DISPLAY.value)
                if code and display and system:
                    answer_mapping[f"{system}|{code}"] = display

    for section in json_content.get(KeyNames.ITEM.value, []):
        question_id = section.get(KeyNames.LINK_ID.value)
        question_text = section.get(KeyNames.TEXT.value)

        if question_id and question_text:
            question_mapping[question_id] = question_text

    return question_mapping, answer_mapping


def process_social_support_questionnaire(json_content):
    """Performs special handling to extract the mappings for Social Support Questionnaire."""
    question_mapping = {}
    answer_mapping = {}

    for section in json_content.get(KeyNames.ITEM.value, []):
        question_id = section.get(KeyNames.LINK_ID.value)
        question_text = section.get(KeyNames.TEXT.value, None)

        if question_id:
            question_mapping[question_id] = question_text
            if question_text is not None:
                for answer_option in section.get(KeyNames.ANSWER_OPTION.value, []):
                    value = answer_option.get(KeyNames.VALUE_CODING.value, {})
                    code = value.get(KeyNames.CODE.value, None)
                    display = value.get(KeyNames.DISPLAY.value, None)
                    system = value.get(KeyNames.SYSTEM.value, None)

                    if code and display and system:
                        answer_mapping[f"{system}|{code}"] = display
    return question_mapping, answer_mapping


def process_phq9_questionnaire(json_content):
    """Extracts question and answer mappings from a PHQ-9 questionnaire JSON."""

    question_mapping = {}
    answer_mapping = {}

    for section in json_content.get(KeyNames.ITEM.value, []):
        question_id = section.get(KeyNames.LINK_ID.value)
        question_text = section.get(KeyNames.TEXT.value, None)
        if question_id:
            question_mapping[question_id] = question_text
            for sub_item in section.get(KeyNames.ITEM.value, []):
                if sub_question_id := sub_item.get(KeyNames.LINK_ID.value):
                    question_mapping[sub_question_id] = sub_item.get("text", None)

    for contained in json_content.get(KeyNames.CONTAINED.value, []):
        if contained.get(KeyNames.RESOURCE_TYPE.value) == KeyNames.VALUE_SET.value:
            for include in contained.get(KeyNames.COMPOSE.value, {}).get(
                KeyNames.INCLUDE.value, []
            ):
                for concept in include.get(KeyNames.CONCEPT.value, []):
                    display = concept.get(KeyNames.DISPLAY.value)
                    system = include.get(KeyNames.SYSTEM.value)
                    value_decimal = next(
                        (
                            ext.get(KeyNames.VALUE_DECIMAL.value)
                            for ext in concept.get(KeyNames.EXTENSION.value, [])
                        ),
                        None,
                    )
                    if value_decimal is not None and display and system:
                        answer_mapping[f"{system}|{value_decimal}"] = display

    return question_mapping, answer_mapping


def get_answer_code_and_value(
    item: QuestionnaireResponse, all_answer_mappings: dict[str, str], survey_path: str
) -> tuple[str, str]:
    """
    Gets the answer value for a QuestionnaireResponse item.

    Parameters:
        item: The QuestionnaireResponse item.
        all_answer_mappings: A dictionary containing all answer mappings.
        survey_path (str): The path to Phoenix-generated JSON surveys.

    Returns:
        str: The answer code.
        str: The answer value.
    """
    answer_code = UNKNOWN_ANSWER_STRING
    answer_value = UNKNOWN_ANSWER_STRING

    if item.answer and len(item.answer) > 0:
        answer = item.answer[0].dict()

        if answer.get(KeyNames.VALUE_CODING.value) is not None:
            system_id = answer[KeyNames.VALUE_CODING.value].get(KeyNames.SYSTEM.value)
            code = answer[KeyNames.VALUE_CODING.value].get(KeyNames.CODE.value)
            if system_id and code:
                combined_id = f"{system_id}|{code}"
                answer_code = code
                answer_value = all_answer_mappings.get(
                    combined_id, UNKNOWN_ANSWER_STRING
                )
        elif answer.get(KeyNames.VALUE_STRING.value) is not None:
            answer_code = answer_value = answer[KeyNames.VALUE_STRING.value]
        elif answer.get(KeyNames.VALUE_INTEGER.value) is not None:
            answer_code = str(answer.get(KeyNames.VALUE_INTEGER.value))

            if survey_path == PHQ9_PATH:
                answer_value = all_answer_mappings.get(
                    f"{PHQ9_CODE_SYSTEM}|{answer_code}",
                    UNKNOWN_ANSWER_STRING,
                )
            else:
                answer_value = answer.get(KeyNames.VALUE_INTEGER.value)

    return answer_code, answer_value


def get_survey_title(survey_path: str) -> str:
    """
    Gets the survey title from a Phoenix-generated JSON survey file.

    Parameters:
        survey_path (str): The path to Phoenix-generated JSON survey.

    Returns:
        str: The survey title.
    """
    with open(survey_path, "r", encoding=ENCODING) as file:
        title_json = json.load(file)
        if title := title_json.get(KeyNames.TITLE.value):
            return title
    return ""


def flatten_fhir_resources(  # pylint: disable=unused-variable
    resources: list[Any],
    survey_path: str = None,
) -> FHIRDataFrame | None:
    """
    Flattens a list of FHIR resources into a structured DataFrame.

    This function determines the appropriate ResourceFlattener subclass to use
    based on the type of the first resource in the list. It then uses that flattener
    to transform the list of resources into a FHIRDataFrame.

    Parameters:
        resources (list[Any]): A list of FHIR resource objects to be flattened.
        survey_path (str): The path to Phoenix-generated JSON survey used to
            extract relevant mappings.

    Returns:
        FHIRDataFrame | None: A structured DataFrame containing the flattened FHIR data,
                               or None if the resources list is empty or unsupported.

    Raises:
        ValueError: If no suitable flattener is found for the resource type.
    """
    if not resources:
        print("No data available.")
        return None

    flattener_classes = {
        FHIRResourceType.OBSERVATION: ObservationFlattener,
        FHIRResourceType.ECG_OBSERVATION: ECGObservationFlattener,
        FHIRResourceType.QUESTIONNAIRE_RESPONSE: QuestionnaireResponseFlattener,
    }

    resource_type = FHIRResourceType(
        resources[0].resource_type
    )  # Assuming each resource has a resource_type attribute

    if resource_type in flattener_classes:
        flattener_class = flattener_classes[resource_type]

        flattener = flattener_class()
    else:
        raise ValueError(f"No flattener found for resource type: {resource_type}")

    return flattener.flatten(resources, survey_path)
