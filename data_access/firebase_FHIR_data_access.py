#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module provides functionalities for accessing and fetching FHIR data from a Firebase
database. It includes the FirebaseFHIRAccess class which handles the connection to the Firebase
database and provides methods for fetching FHIR data based on user ID and LOINC codes. The module
is designed to support both development (using the Firestore emulator) and production environments.
"""

# Standard library imports
import json
import os
from typing import Tuple, Optional

# Related third-party imports
from firebase_admin import credentials, firestore
import firebase_admin
from google.cloud.firestore_v1.base_query import FieldFilter

# Local application/library specific imports
from fhir.resources.observation import Observation


class EnhancedObservation:
    def __init__(self, observation: Observation, user_id: str = None):
        self.observation = observation
        self.user_id = user_id


class FirebaseFHIRAccess:
    def __init__(self, service_account_key_file: str, project_id: str) -> None:
        self.service_account_key_file = service_account_key_file
        self.project_id = project_id
        self.db = None

    def connect(self) -> None:
        if (
            os.getenv("CI") or "FIRESTORE_EMULATOR_HOST" in os.environ
        ):  # Check if running in CI environment
            # Point to the emulator for CI tests
            os.environ["FIRESTORE_EMULATOR_HOST"] = "localhost:8080"
            os.environ["GCLOUD_PROJECT"] = self.project_id
            if not firebase_admin._apps:
                firebase_admin.initialize_app(options={"projectId": self.project_id})
            self.db = firestore.Client(project=self.project_id)
        else:  # Connect to the production environment
            if not firebase_admin._apps:
                cred = credentials.Certificate(self.service_account_key_file)
                firebase_admin.initialize_app(cred, {"projectId": self.project_id})
            self.db = firestore.client()

    def fetch_data(
        self,
        collection_name: str = "users",
        subcollection_name: str = "HealthKit",
        loinc_codes: Optional[list[str]] = None,
    ) -> list[EnhancedObservation]:
        resources = []
        users = self.db.collection(collection_name).stream()

        for user in users:
            query = (
                self.db.collection(collection_name)
                .document(user.id)
                .collection(subcollection_name)
            )
            if loinc_codes:
                for code in loinc_codes:
                    display_str, code_str, system_str = self.get_code_details(code)

                    FHIR_docs = query.where(
                        filter=FieldFilter(
                            "code.coding",
                            "array_contains",
                            {
                                "display": display_str,
                                "system": system_str,
                                "code": code_str,
                            },
                        )
                    ).stream()

                    for doc in FHIR_docs:
                        observation_str = json.dumps(doc.to_dict())
                        fhir_obj = Observation.parse_raw(observation_str)
                        enhanced_fhir_obj = EnhancedObservation(
                            observation=fhir_obj, user_id=user.id
                        )
                        resources.append(enhanced_fhir_obj)
            else:
                FHIR_docs = query.stream()
                for doc in FHIR_docs:
                    observation_str = json.dumps(doc.to_dict())
                    fhir_obj = Observation.parse_raw(observation_str)
                    enhanced_fhir_obj = EnhancedObservation(
                        observation=fhir_obj, user_id=user.id
                    )
                    resources.append(enhanced_fhir_obj)

        return resources

    def get_code_details(
        self, code: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        code_mappings = {
            "9052-2": ("Calorie intake total", "9052-2", "http://loinc.org"),
            "55423-8": (
                "Number of steps in unspecified time Pedometer",
                "55423-8",
                "http://loinc.org",
            ),
            "HKQuantityTypeIdentifierDietaryProtein": (
                "Dietary Protein",
                "HKQuantityTypeIdentifierDietaryProtein",
                "http://developer.apple.com/documentation/healthkit",
            ),
        }

        return code_mappings.get(code, (None, None, None))
