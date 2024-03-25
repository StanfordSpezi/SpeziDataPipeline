#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

# Related third-party imports
import pandas as pd

# Local application/library specific imports
from data_access.firebase_FHIR_data_access import EnhancedObservation


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
        resource_type (str, optional): The type of FHIR resources contained. Defaults to 'Observation'.
    """

    def __init__(self, data: pd.DataFrame, resource_type: str = "Observation") -> None:
        """
        Initializes a FHIRDataFrame with given data and resource type.

        Parameters:
            data (pd.DataFrame): The pandas DataFrame containing FHIR data.
            resource_type (str, optional): The type of FHIR resource (e.g., 'Observation'). Defaults to 'Observation'.
        """
        self.data_frame = data
        self.resource_type = resource_type

    @property
    def df(self) -> pd.DataFrame:
        """
        A property to access the underlying pandas DataFrame containing FHIR data.

        Returns:
            pd.DataFrame: The pandas DataFrame storing the FHIR data.
        """
        return self.data_frame


def flatten_FHIR_resources(FHIR_resources: list[EnhancedObservation]) -> FHIRDataFrame:
    """
    Transforms a list of EnhancedObservation objects into a flattened pandas DataFrame
    structure, making it easier to manipulate and analyze the FHIR data.

    Parameters:
        FHIR_resources (list[EnhancedObservation]): A list of EnhancedObservation objects
                                                    containing FHIR data to be flattened.

    Returns:
        FHIRDataFrame: A FHIRDataFrame object containing the flattened FHIR data, suitable
                       for further data processing and analysis.
    """
    flattened_data = []

    for FHIR_obj in FHIR_resources:
        effective_datetime = (
            FHIR_obj.observation.dict()["effectivePeriod"]["start"]
            or FHIR_obj.dict()["effectiveDateTime"]
        )
        coding = FHIR_obj.observation.dict()["code"]["coding"]
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
            "UserId": FHIR_obj.user_id,
            "DocumentId": FHIR_obj.observation.dict()["id"],
            "EffectiveDateTime": effective_datetime if effective_datetime else None,
            "QuantityName": quantity_name,
            "QuantityUnit": FHIR_obj.observation.dict()["valueQuantity"]["unit"],
            "QuantityValue": FHIR_obj.observation.dict()["valueQuantity"]["value"],
            "LoincCode": loinc_code,
            "Display": display,
            "AppleHealthKitCode": apple_healthkit_code,
        }

        flattened_data.append(flattened_entry)

    flattened_df = pd.DataFrame(flattened_data)
    flattened_df["EffectiveDateTime"] = pd.to_datetime(
        flattened_df["EffectiveDateTime"], errors="coerce"
    )

    return FHIRDataFrame(flattened_df)
