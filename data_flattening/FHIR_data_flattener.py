#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""Module for flattening FHIR data.
 This module provides functionalities to flatten Fast Healthcare Interoperability
 Resources (FHIR) data for easier processing and analysis. It includes functions
 to transform nested FHIR JSON structures into flattened tabular formats.
 """

# Standard library imports
from datetime import date
from enum import Enum

# Related third-party imports
from dataclasses import dataclass
import pandas as pd
from pandas.api.types import is_string_dtype, is_object_dtype

# Local application/library specific imports
from fhir.resources.observation import Observation


class ColumnNames(Enum):
    USER_ID = "UserId"
    EFFECTIVE_DATE_TIME = "EffectiveDateTime"
    QUANTITY_NAME = "QuantityName"
    QUANTITY_UNIT = "QuantityUnit"
    QUANTITY_VALUE = "QuantityValue"
    LOINC_CODE = "LoincCode"
    DISPLAY = "Display"
    APPLE_HEALTH_KIT_CODE = "AppleHealthKitCode"


class FHIRResourceType(Enum):
    """
    Enumeration of FHIR resource types.

    This enum provides a list of FHIR resource types used in the application, ensuring
    consistency and preventing typos in resource type handling.

    Attributes:
        OBSERVATION (str): Represents an observation resource type.

    Note:
        The `.value` attribute is used to access the string value of the enum members.
    """

    OBSERVATION = "Observation"


@dataclass
class FHIRDataFrame:
    """
    Represents a DataFrame specifically designed to handle FHIR data. This class provides
    a structured format for FHIR data, facilitating easier manipulation and analysis of
    health-related information encoded in FHIR resources.

    Attributes:
        data_frame (pd.DataFrame): The underlying pandas DataFrame that stores the FHIR data.
        resource_type (str): Indicates the type of FHIR resources contained in the DataFrame,
                             such as 'Observation', to provide context for the data.

    Parameters:
        data (pd.DataFrame): A pandas DataFrame containing the FHIR data.
        resource_type (str, optional): The type of FHIR resources contained. Defaults to
                                    'Observation'.
    """

    data_frame: pd.DataFrame
    resource_type: str = FHIRResourceType.OBSERVATION.value
    resource_columns = {
        FHIRResourceType.OBSERVATION.value: [
            ColumnNames.USER_ID.value,
            ColumnNames.EFFECTIVE_DATE_TIME.value,
            ColumnNames.QUANTITY_NAME.value,
            ColumnNames.QUANTITY_UNIT.value,
            ColumnNames.QUANTITY_VALUE.value,
            ColumnNames.LOINC_CODE.value,
            ColumnNames.DISPLAY.value,
            ColumnNames.APPLE_HEALTH_KIT_CODE.value,
        ],
        # Add mappings for other resource types
    }

    def __init__(
        self,
        data: pd.DataFrame,
        resource_type: str = FHIRResourceType.OBSERVATION.value,
    ) -> None:
        """
        Initializes a FHIRDataFrame with given data and resource type.

        Parameters:
            data (pd.DataFrame): The pandas DataFrame containing FHIR data.
            resource_type (str, optional): The type of FHIR resource. Defaults to 'Observation'.
        """
        self.data_frame = data
        self.resource_type = resource_type
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

    def validate_columns(self) -> None:
        """
        Validates that the DataFrame contains all required columns for processing.
        Raises a ValueError if any required column is missing.
        """
        required_columns = self.resource_columns.get(self.resource_type, [])

        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The DataFrame is missing required columns: {missing_columns}"
            )

        if ColumnNames.EFFECTIVE_DATE_TIME.value in self.df.columns:
            if not all(
                isinstance(d, date)
                for d in self.df[ColumnNames.EFFECTIVE_DATE_TIME.value]
            ):
                raise ValueError(
                    f"The {ColumnNames.EFFECTIVE_DATE_TIME.value}"
                    "column is not of type datetime.date."
                )

        for column in set(required_columns) - {ColumnNames.EFFECTIVE_DATE_TIME.value}:
            if column in self.df.columns:
                if not (
                    is_string_dtype(self.df[column]) or is_object_dtype(self.df[column])
                ):
                    raise ValueError(f"The '{column}' column does not contain strings.")


def flatten_fhir_resources(  # pylint: disable=unused-variable
    fhir_resources: list[Observation],
) -> FHIRDataFrame | None:
    """
    Transforms a list of Observation objects into a flattened pandas DataFrame
    structure, making it easier to manipulate and analyze the FHIR data.

    Parameters:
        fhir_resources (list[Observation]): A list of Observation objects
                                                    containing FHIR data to be flattened.

    Returns:
        FHIRDataFrame: A FHIRDataFrame object containing the flattened FHIR data, suitable
                       for further data processing and analysis.
    """
    if not fhir_resources:
        print("No data available.")
        return None

    if not all(isinstance(resource, Observation) for resource in fhir_resources):
        print("Not all FHIR resources are of the Observation type.")
        return None

    flattened_data = []

    for fhir_obj in fhir_resources:
        effective_datetime = (
            fhir_obj.dict()["effectivePeriod"]["start"]
            or fhir_obj.dict()["effectiveDateTime"]
        )
        coding = fhir_obj.dict()["code"]["coding"]
        loinc_code = coding[0]["code"] if len(coding) > 0 else ""
        display = coding[0]["display"] if len(coding) > 0 else ""
        apple_healthkit_code = (
            coding[1]["code"]
            if len(coding) > 1
            else (coding[0]["code"] if len(coding) > 0 else "")
        )
        quantity_name = (
            coding[1]["display"]
            if len(coding) > 1
            else (coding[0]["display"] if len(coding) > 0 else "")
        )

        flattened_entry = {
            "UserId": fhir_obj.subject.id,
            "DocumentId": fhir_obj.dict()["id"],
            "EffectiveDateTime": effective_datetime if effective_datetime else None,
            "QuantityName": quantity_name,
            "QuantityUnit": fhir_obj.dict()["valueQuantity"]["unit"],
            "QuantityValue": fhir_obj.dict()["valueQuantity"]["value"],
            "LoincCode": loinc_code,
            "Display": display,
            "AppleHealthKitCode": apple_healthkit_code,
        }

        flattened_data.append(flattened_entry)

    flattened_df = pd.DataFrame(flattened_data)
    flattened_df["EffectiveDateTime"] = pd.to_datetime(
        flattened_df["EffectiveDateTime"], errors="coerce"
    ).dt.date

    return FHIRDataFrame(flattened_df, "Observation")
