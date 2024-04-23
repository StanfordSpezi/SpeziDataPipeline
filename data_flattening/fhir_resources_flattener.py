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

This module includes classes for processing specific types of FHIR resources, such as general
Observations and ECG Observations, extracting relevant information and standardizing it into a
consistent format. Additionally, it defines enums and helper functions to aid in the extraction
and organization of data from these resources.

Main Components:
- `KeyNames` and `ColumnNames`: Enums that define standardized keys and column names used in the
    flattening process, ensuring consistency
  across the application.
- `ResourceFlattener`: An abstract base class designed to be extended for specific FHIR resource
    types, providing a common interface
  for the flattening operation.
- `ObservationFlattener` and `ECGObservationFlattener`: Concrete implementations of
    `ResourceFlattener` tailored to handle Observation
  and ECG Observation resources, respectively.
- `extract_coding_info` and `extract_component_info`: Helper functions for extracting detailed
    information from Observation components and codings.
- `flatten_fhir_resources`: A utility function that orchestrates the flattening process,
    dynamically selecting the appropriate flattener based on the resource type.

Usage:
The module is intended for developers and data analysts working with FHIR data, particularly those
looking to analyze health data within pandas or similar data analysis tools. By providing a
standardized way to flatten FHIR resources, this module aims to lower the barrier to entry for
healthcare data analysis and support a wide range of analytical applications.
"""

# Standard library imports
from datetime import date
from enum import Enum
import re

# Related third-party imports
from dataclasses import dataclass
from typing import Any
import pandas as pd
from fhir.resources.R4B.observation import Observation


class KeyNames(Enum):
    """
    Enumerates standardized key names for fetching FHIR resources.

    These keys represent common attributes found within FHIR resources that are relevant for ECG and
    other health data manipulations. This enumeration facilitates the consistent access to these
    attributes across various parts of an application dealing with FHIR data.

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
    """

    EFFECTIVE_DATE_TIME = "effectiveDateTime"
    EFFECTIVE_PERIOD = "effectivePeriod"
    START = "start"
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


class ColumnNames(Enum):
    """
    Enumerates standardized column names for use in flattened DataFrames.

    These column names are designed for handling data extracted from FHIR resources, especially for
    observations related to ECG and other health metrics. Standardizing these names helps ensure
    consistency in data processing and analysis tasks.

    Attributes:
        USER_ID: Identifier for the patient or subject of the observation.
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
        ECG_RECORDINGJ: Represents a series of ECG recording columns.
        ECG_RECORDING1: First set of ECG recording data.
        ECG_RECORDING2: Second set of ECG recording data.
        ECG_RECORDING3: Third set of ECG recording data.
    """

    USER_ID = "UserId"
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
    
    SURVEY_TITLE = "SurveyTitle"
    SURVEY_DATE = "Date"
    SURVEY_ID = "SurveyID"
    QUESTION_ID = "QuestionID"
    QUESTION_TEXT = "QuestionText"
    ANSWER_TEXT = "AnswerText"

class ECGObservation:    # pylint: disable=unused-variable
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

    Subclasses of ResourceFlattener should implement specific logic for flattening
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
                ColumnNames.SURVEY_TITLE,
                ColumnNames.SURVEY_DATE,
                ColumnNames.SURVEY_ID,
                ColumnNames.QUESTION_ID,
                ColumnNames.QUESTION_TEXT,
                ColumnNames.ANSWER_TEXT
            ],
        }

        if resource_type not in self.resource_columns:
            raise ValueError(f"Unsupported resource type: {resource_type.name}")

    def flatten(self, resources: list[Any], survey_path: list[str] = None) -> FHIRDataFrame:
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

    def flatten(self, resources: list[Observation]) -> FHIRDataFrame:
        """
        Converts a list of FHIR Observation resources into a structured DataFrame.
        Extracts and organizes information from observations into a format suitable for analysis.

        Parameters:
            resources (list[Observation]): A collection of FHIR Observation resources to be
                flattened.

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

    def flatten(self, resources: list[ECGObservation]) -> FHIRDataFrame:
        """
        Flattens a list of ECG Observation resources, extracting ECG data and related metrics
        into a structured, analyzable DataFrame format.

        Parameters:
            resources (list[ECGObservation]): A collection of FHIR ECG Observation resources.

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
    like LOINC codes (numeric) and Apple HealthKit codes (alphabetic).

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

class QuestionnaireResponseFlattener:

    def flatten(self, resources: list[Any], survey_paths: list[str]) -> FHIRDataFrame:
        flattened_data = []
        def extract_mappings(json_content):

            question_mapping = {}
            answer_mapping = {}

            for section in json_content.get("item", []):
                question_id = section.get("linkId")
                question_mapping[question_id] = section.get("text", "Unknown Question")

                for answer_option in section.get("answerOption", []):
                    value = answer_option.get("valueCoding", {})
                    code = value.get("code")
                    display = value.get("display")

                    system = value.get("system")

                    if code and display and system:
                        combined_id = f"{system}|{code}"
                        answer_mapping[combined_id] = display

            return question_mapping, answer_mapping

        all_question_mappings = {}
        all_answer_mappings = {}

        for file_path in survey_paths:
            with open(file_path, "r") as file:
                question_mapping, answer_mapping = extract_mappings(json.load(file))
                all_question_mappings.update(question_mapping)
                all_answer_mappings.update(answer_mapping)
        
        for file_path in survey_paths:
            with open(file_path, 'r') as file:
                title_json = json.load(file)
                title = title_json.get('title')
        
        for response in resources:
            for item in response.item:
                question_id = item.linkId
                question_text = all_question_mappings.get(
                    question_id, "Unknown Question"
                )

                answer_value = "No Answer"

                if item.answer and len(item.answer) > 0:

                    answer = item.answer[0]

                    answer = answer.dict()

                    if "valueCoding" in answer:
                        system_id = answer["valueCoding"].get("system")
                        code = answer["valueCoding"].get("code")
                        if system_id and code:
                            combined_id = f"{system_id}|{code}"
                            answer_value = all_answer_mappings.get(
                                combined_id, "No Answer"
                            )

                    elif "valueString" in answer:
                        answer_value = answer["valueString"]
                    elif "valueInteger" in answer:
                        answer_value = answer["valueInteger"]
                    else:
                        answer_value = "No Answer"

                responseDict = response.dict()
                date = responseDict['authored']
                    
                user_id = getattr(response.subject, 'id', None)
                
                flattened_entry = {
                    ColumnNames.USER_ID.value: user_id,
                    ColumnNames.SURVEY_TITLE.value: title,
                    ColumnNames.SURVEY_ID.value: response.id,
                    ColumnNames.QUESTION_ID.value: question_id,
                    ColumnNames.QUESTION_TEXT.value: question_text,
                    ColumnNames.ANSWER_TEXT.value: answer_value
                }
                
                flattened_data.append(flattened_entry)

        return FHIRDataFrame(flattened_df, FHIRResourceType.QUESTIONNAIRE_RESPONSE)


def flatten_fhir_resources(
    resources: list[Any],
    survey_path: list[str] = None,
) -> FHIRDataFrame | None:
    """
    Flattens a list of FHIR resources into a structured DataFrame.

    This function determines the appropriate ResourceFlattener subclass to use
    based on the type of the first resource in the list. It then uses that flattener
    to transform the list of resources into a FHIRDataFrame.

    Parameters:
        resources (list[Any]): A list of FHIR resource objects to be flattened.

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
        FHIRResourceType.QUESTIONNAIRE_RESPONSE: QuestionnaireResponseFlattener
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

