#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

# Standard library imports
from datetime import datetime
from typing import List, Dict, Any

# Related third-party imports
import pandas as pd

# Local application/library specific imports
from data_access.firebase_FHIR_data_access import EnhancedObservation


class FHIRDataFrame:
    """
    A class to represent a DataFrame specifically designed for FHIR data.

    Attributes:
        data_frame (pd.DataFrame): The underlying pandas DataFrame containing FHIR data.
        resource_type (str): The type of FHIR resources contained, e.g., 'Observation'.
    """

    def __init__(self, data: pd.DataFrame, resource_type: str = "Observation") -> None:
        self.data_frame = data
        self.resource_type = resource_type

    @property
    def df(self) -> pd.DataFrame:
        return self.data_frame


def flatten_FHIR_resources(FHIR_resources: List[EnhancedObservation]) -> FHIRDataFrame:
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
            "UserId": FHIR_obj.UserId,
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
