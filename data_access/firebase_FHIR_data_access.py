#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

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
    """
    A wrapper class for FHIR Observation resources with an optional user identifier.

    This class is designed to enhance the Observation object by associating it with
    a specific user, allowing for easier tracking and organization of observations
    within user-centric applications.

    Attributes:
        observation (Observation): The FHIR Observation resource object.
        user_id (str, optional): The identifier for the user associated with this
                                 observation. Defaults to None.

    """
    def __init__(self, observation: Observation, user_id: str = None):
        """
        Initializes the EnhancedObservation instance with a FHIR Observation and an optional user ID.

        Parameters:
            observation (Observation): The FHIR Observation resource.
            user_id (str, optional): A unique identifier for the user associated with
                                     this observation. Defaults to None.
        """
        self.observation = observation
        self.user_id = user_id


class FirebaseFHIRAccess:
    """
    Provides access to FHIR resources stored in Firebase Firestore, allowing for
    operations such as connecting to the database and fetching data based on
    specific criteria like LOINC codes.

    This class abstracts the complexity of interacting with Firestore for FHIR-related
    operations, offering a simplified interface for fetching and manipulating FHIR
    Observations.

    Attributes:
        service_account_key_file (str): Path to the Firebase service account key file.
        project_id (str): The Firebase project ID.
        db (Optional[firestore.Client]): The Firestore client instance, initialized upon
                                         connection. Defaults to None.

    """
    def __init__(self, service_account_key_file: str, project_id: str) -> None:
        """
        Initializes the FirebaseFHIRAccess instance with Firebase service account
        credentials and project ID.

        Parameters:
            service_account_key_file (str): Path to the Firebase service account key file.
            project_id (str): The Firebase project ID.
        """
        self.service_account_key_file = service_account_key_file
        self.project_id = project_id
        self.db = None

    def connect(self) -> None:
        """
        Establishes a connection to the Firebase Firestore database.

        Depending on the environment, it connects to either the Firestore emulator
        for CI tests or the production Firestore service. It initializes the Firestore
        client and sets it to the `db` attribute.
        """
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
        """
            Fetches FHIR Observation data from Firestore based on the given collection names
            and optional LOINC codes.

            Parameters:
                collection_name (str): The name of the Firestore collection to query. Defaults to "users".
                subcollection_name (str): The name of the Firestore subcollection to query. Defaults to "HealthKit".
                loinc_codes (Optional[list[str]]): A list of LOINC codes to filter the Observations. Defaults to None.

            Returns:
                list[EnhancedObservation]: A list of EnhancedObservation objects representing the fetched FHIR Observations.
        """
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

def get_code_details(code: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Retrieves the details associated with a given LOINC code or custom code.

    This method looks up the code in a predefined mapping and returns the display
    string, code string, and system string associated with the code.

    Parameters:
        code (str): The LOINC code or custom code to look up.

    Returns:
        Tuple[Optional[str], Optional[str], Optional[str]]: A tuple containing the display string, code string,
                                                            and system string for the code, or (None, None, None)
                                                            if the code is not found in the mapping.
    """
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
