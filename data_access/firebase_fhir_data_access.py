#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module provides functionalities for accessing and managing FHIR (Fast Healthcare
Interoperability Resources) data within a Firebase Firestore database. It supports both
development and production environments, including the use of the Firestore emulator for
development and testing purposes. The primary class, `FirebaseFHIRAccess`, abstracts the
complexity of Firestore connections and queries, offering simplified methods to fetch FHIR
data based on user IDs and LOINC (Logical Observation Identifiers Names and Codes) codes.

Classes:
    FirebaseFHIRAccess: Manages access to FHIR resources stored in Firebase Firestore,
    allowing for operations such as connecting to the database and fetching data based on
    specific criteria like LOINC codes.

Functions:
    _fetch_user_resources: Fetches resources for a specific user from Firestore based on 
        the given collection and subcollection names, optionally filtering by LOINC codes.
    _process_loinc_codes: Filters documents based on LOINC codes from a Firestore collection
        reference, converting matching documents into FHIR Observation instances.
    _process_all_documents: Fetches and processes all documents from a Firestore collection
        reference for a specific user, converting each document to a FHIR Observation instance.
    _create_resources: Converts Firestore documents into FHIR Observation instances, associating
        each with the corresponding user's Firestore document ID.
    get_code_mappings: Retrieves mappings for a given LOINC code or custom code, supporting the
        translation of codes for FHIR resource creation and querying.
"""

# Standard library imports
import json
import os
from typing import Any

# Related third-party imports
from firebase_admin import credentials, firestore
import firebase_admin
from google.cloud.firestore import (
    CollectionReference,
    DocumentReference,
    DocumentSnapshot,
)
from google.cloud.firestore_v1.base_query import FieldFilter
from fhir.resources.observation import Observation
from fhir.resources.reference import Reference
from data_processing.code_mapping import CodeProcessor


FIRESTORE_EMULATOR_HOST_KEY = "FIRESTORE_EMULATOR_HOST"


class FirebaseFHIRAccess:  # pylint: disable=unused-variable
    """
    Manages access and operations on FHIR resources within Firebase Firestore. This class
    facilitates the connection to Firestore, supporting both development (via the Firestore
    emulator) and production environments. It offers methods to fetch FHIR data based on user
    IDs and LOINC codes, abstracting the complexity of Firestore queries and FHIR data handling.

    Attributes:
        service_account_key_file (str): Path to the Firebase service account key file for
                                        authentication.
        project_id (str): Identifier of the Firebase project.
        db (Optional[firestore.Client]): A Firestore client instance for database operations,
                                          initialized upon successful connection.
    """

    def __init__(self, service_account_key_file: str, project_id: str) -> None:
        """
        Initializes the FirebaseFHIRAccess instance with Firebase service account
        credentials and project ID.
        """
        self.service_account_key_file = service_account_key_file
        self.project_id = project_id
        self.db = None

    def connect(self) -> None:
        """
        Establishes a connection to the Firebase Firestore database.
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
                self.db = firestore.Client(  # pylint: disable=no-member
                    project=self.project_id
                )

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
        Retrieves FHIR Observation data for specified LOINC codes from Firestore.
        Data is fetched from the given collection and subcollection, optionally
        filtered by the provided LOINC codes.

        Parameters:
            collection_name (str): The name of the Firestore collection.
                Defaults to "users".
            subcollection_name (str): The name of the Firestore subcollection.
                Defaults to "HealthKit".
            loinc_codes (list[str] | None): Optional list of LOINC codes to filter
                observations. If None, all observations in the subcollection are fetched.

        Returns:
            list[Observation]: A list of FHIR Observation instances matching the query criteria.
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
        Private method to fetch FHIR Observation resources for a specific user,
        ptionally filtered by LOINC codes. Queries Firestore based on specified collection
        and subcollection names.

        Parameters:
            user (DocumentReference): Firestore reference to the user document.
            collection_name (str): Name of the Firestore collection.
            subcollection_name (str): Name of the Firestore subcollection.
            loinc_codes (list[str] | None): Optional list of LOINC codes to filter observations.

        Returns:
            list[Observation]: List of FHIR Observation objects corresponding to the user and
                                optional LOINC codes filter.
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
    Filters documents based on LOINC codes from a Firestore collection reference. This function
    processes and converts matching Firestore documents into FHIR Observation instances.

    Parameters:
        query (CollectionReference): Firestore query object for a user's subcollection.
        user (DocumentReference): Firestore reference to the user document.
        loinc_codes (list[str]): List of LOINC codes to filter documents.

    Returns:
        list[Observation]: A list of FHIR Observation instances that match the specified
            LOINC codes.
    """
    resources = []
    for code in loinc_codes:
        display_str, code_str, system_str = get_code_mappings(code)
        fhir_docs = query.where(
            filter=FieldFilter(
                "code.coding",
                "array_contains",
                {"display": display_str, "system": system_str, "code": code_str},
            )
        ).stream()
        resources.extend(_create_resources(fhir_docs, user))
    return resources


def _process_all_documents(
    query: CollectionReference, user: DocumentReference
) -> list[Observation]:
    """
    Fetches and processes all documents from a Firestore collection reference for a specific user,
    converting each Firestore document to a FHIR Observation instance.

    Parameters:
        query (CollectionReference): Firestore query object for a user's subcollection.
        user (DocumentReference): Firestore reference to the user document.

    Returns:
        list[Observation]: List of FHIR Observation instances for all documents in the user's
            subcollection.
    """
    fhir_docs = query.stream()
    return _create_resources(fhir_docs, user)


def _create_resources(
    fhir_docs: list[DocumentSnapshot], user: DocumentReference
) -> list[Any]:
    """
    Converts Firestore documents into FHIR Observation instances, setting the subject reference
    to the user's Firestore document ID.

    Parameters:
        fhir_docs (list[DocumentSnapshot]): Iterable of Firestore document snapshots containing
            FHIR observation data.
        user (DocumentReference): Firestore reference to the user document.

    Returns:
        list[Any]: List of FHIR Observation instances created from the Firestore documents.
    """
    resources = []
    for doc in fhir_docs:
        resource_str = json.dumps(doc.to_dict())
        resource_obj = Observation.parse_raw(resource_str)
        resource_obj.subject = Reference(id=user.id)
        resources.append(resource_obj)
    return resources


def get_code_mappings(code: str) -> tuple[str, str, str]:
    """
    Retrieves display, code, and system strings associated with a given LOINC code or custom code
    from a predefined mapping. This function is intended to support the translation of codes
    for use in FHIR resource creation and querying.

    Parameters:
        code (str): The LOINC code or custom code to look up.

    Returns:
        tuple[str, str, str]: A tuple containing the display string, code string, and system string
                               for the code. Returns (None, None, None) if the code is not found.
    """
    loinc_processor = CodeProcessor()
    code_mappings = loinc_processor.code_mappings.get(code)

    if (code_mappings := loinc_processor.code_mappings.get(code)) is None:
        print(f"This LOINC code '{code}' is not supported.")
        return (None, None, None)

    return code_mappings
