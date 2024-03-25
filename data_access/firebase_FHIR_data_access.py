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

# Related third-party imports
from firebase_admin import credentials, firestore
import firebase_admin
from google.cloud.firestore import (
    CollectionReference,
    DocumentReference,
    DocumentSnapshot,
)
from google.cloud.firestore_v1.base_query import FieldFilter

# Local application/library specific imports
from fhir.resources.observation import Observation
from fhir.resources.reference import Reference


FIRESTORE_EMULATOR_HOST_KEY = "FIRESTORE_EMULATOR_HOST"


class FirebaseFHIRAccess:  # pylint: disable=unused-variable
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

        try:
            # Attempt to retrieve the default app.
            firebase_admin.get_app()
        except ValueError:
            # If it raises a ValueError, then the app hasn't been initialized.

            if (
                os.getenv("CI") or FIRESTORE_EMULATOR_HOST_KEY in os.environ
            ):  # Check if running in CI environment
                # Point to the emulator for CI tests
                os.environ[FIRESTORE_EMULATOR_HOST_KEY] = "localhost:8080"
                os.environ["GCLOUD_PROJECT"] = self.project_id

                firebase_admin.initialize_app(options={"projectId": self.project_id})

            else:  # Connect to the production environment
                cred = credentials.Certificate(self.service_account_key_file)
                firebase_admin.initialize_app(cred, {"projectId": self.project_id})

            self.db = firestore.client()

    def fetch_data(
        self,
        collection_name: str = "users",
        subcollection_name: str = "HealthKit",
        loinc_codes: list[str] | None = None,
    ) -> list[Observation]:
        """
        Fetches FHIR Observation data from Firestore based on the given collection names
        and optional LOINC codes.

        Parameters:
            collection_name (str): The name of the Firestore collection to query.
                                Defaults to "users".
            subcollection_name (str): The name of the Firestore subcollection to query.
                                    Defaults to "HealthKit".
            loinc_codes (list[str] | None): A list of LOINC codes to filter the Observations.
                                        Defaults to None.

        Returns:
            list[Observation]: A list of Observation objects representing
                                    the fetched FHIR Observations.
        """

        if self.db is None:
            print("Error: Firebase app is not initialized.")
            return []

        resources = []
        users = self.db.collection(collection_name).stream()
        for user in users:
            user_resources = self._fetch_user_resources(
                user, collection_name, subcollection_name, loinc_codes
            )
            resources.extend(user_resources)
        return resources

    def _fetch_user_resources(
        self,
        user: DocumentReference,
        collection_name: str,
        subcollection_name: str,
        loinc_codes: list[str] | None,
    ) -> list[Observation]:
        """
        Fetches resources for a specific user from Firestore based on the given collection and
        subcollection names, optionally filtering by LOINC codes.

        Parameters:
            user (str): The user document reference from Firestore.
            collection_name (str): The name of the Firestore collection to query.
            subcollection_name (str): The name of the Firestore subcollection to query.
            loinc_codes (list[str] | None): A list of LOINC codes to filter the observations by,
            or None to fetch all documents.

        Returns:
            list[Observation]: A list of Observation objects for the specified user.
        """
        resources = []
        query = (
            self.db.collection(collection_name)
            .document(user.id)
            .collection(subcollection_name)
        )
        if loinc_codes:
            resources.extend(_process_loinc_codes(query, user, loinc_codes))
        else:
            resources.extend(_process_all_documents(query, user))
        return resources


def _process_loinc_codes(
    query: CollectionReference,
    user: DocumentReference,
    loinc_codes: list[str],
) -> list[Observation]:
    """
    Processes documents from a Firestore query by filtering them based on a list of LOINC codes.

    Parameters:
        query: The Firestore query object for a specific user and subcollection.
        user: The user document reference from Firestore.
        loinc_codes (list[str]): A list of LOINC codes to filter the documents by.

    Returns:
        list[Observation]: A list of Observation objects that match the
        LOINC codes.
    """
    resources = []
    for code in loinc_codes:
        display_str, code_str, system_str = get_code_details(code)
        fhir_docs = query.where(
            filter=FieldFilter(
                "code.coding",
                "array_contains",
                {"display": display_str, "system": system_str, "code": code_str},
            )
        ).stream()
        resources.extend(_create_enhanced_observations(fhir_docs, user))
    return resources


def _process_all_documents(
    query: CollectionReference, user: DocumentReference
) -> list[Observation]:
    """
    Processes all documents from a Firestore query for a specific user and subcollection.

    Parameters:
        query: The Firestore query object for a specific user and subcollection.
        user: The user document reference from Firestore.

    Returns:
        list[Observation]: A list of Observation objects for all
        documents in the query.
    """
    fhir_docs = query.stream()
    return _create_enhanced_observations(fhir_docs, user)


def _create_enhanced_observations(
    fhir_docs: list[DocumentSnapshot], user: DocumentReference
) -> list[Observation]:
    """
    Creates Observation objects from Firestore documents.

    Parameters:
        fhir_docs: An iterable of Firestore document references containing FHIR observation
                data.
        user: The user document reference from Firestore.

    Returns:
        list[Observation]: A list of Observation objects created from the
        Firestore documents.
    """
    resources = []
    for doc in fhir_docs:
        observation_str = json.dumps(doc.to_dict())
        fhir_obj = Observation.parse_raw(observation_str)
        fhir_obj.subject = Reference(id=user.id)
        resources.append(fhir_obj)
    return resources


def get_code_details(code: str) -> tuple[str, str, str]:
    """
    Retrieves the details associated with a given LOINC code or custom code.

    This method looks up the code in a predefined mapping and returns the display
    string, code string, and system string associated with the code.

    Parameters:
        code (str): The LOINC code or custom code to look up.

    Returns:
        tuple[str | None, str | None, str | None]: A tuple containing the display string,
                                                        code string, and system string for the
                                                        code,or (None, None, None) if the code is
                                                        not found in the mapping.
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
