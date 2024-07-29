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
    `FirebaseFHIRAccess`: Manages access to FHIR resources stored in Firebase Firestore,
    allowing for operations such as connecting to the database and fetching data based on
    specific criteria like LOINC codes.

Functions:
    `_fetch_user_resources`: Fetches resources for a specific user from Firestore based on 
        the given collection and subcollection names, optionally filtering by LOINC codes.
    `_process_loinc_codes`: Filters documents based on LOINC codes from a Firestore collection
        reference, converting matching documents into FHIR Resource instances.
    `_process_all_documents`: Fetches and processes all documents from a Firestore collection
        reference for a specific user, converting each document to a FHIR Resource instance.
    `create_resources`: Converts Firestore documents into FHIR Resources instances, associating
        each with the corresponding user's Firestore document ID.
    `get_code_mappings`: Retrieves mappings for a given LOINC code or custom code, supporting the
        translation of codes for FHIR resource creation and querying.
"""

# Standard library imports
import json
import os
from typing import Any, Optional

# Related third-party imports
from dataclasses import dataclass
from firebase_admin import credentials, firestore
import firebase_admin
from google.cloud.firestore import (
    CollectionReference,
    DocumentReference,
    DocumentSnapshot,
)
from google.cloud.firestore_v1.base_query import FieldFilter
from fhir.resources.R4B.resource import Resource
from fhir.resources.R4B.observation import Observation
from fhir.resources.R4B.reference import Reference
from fhir.resources.R4B.questionnaireresponse import QuestionnaireResponse

# Local application/library specific imports
from spezi_data_pipeline.data_flattening.fhir_resources_flattener import (
    ECGObservation,
    FHIRResourceType,
    KeyNames,
)
from spezi_data_pipeline.data_processing.code_mapping import CodeProcessor

FIRESTORE_EMULATOR_HOST_KEY = "FIRESTORE_EMULATOR_HOST"
CI_STRING = "CI"
LOCAL_HOST_URL = "localhost:8080"
GCLOUD_PROJECT_STRING = "GCLOUD_PROJECT"
FIREBASE_PROJECT_ID_PARAM_STRING = "projectId"
ECG_RECORDING_LOINC_CODE = "131328"


class FirebaseFHIRAccess:  # pylint: disable=unused-variable
    """
    Manages access and operations on FHIR resources within Firebase Firestore. This class
    facilitates the connection to Firestore, supporting both development (via the Firestore
    emulator) and production environments. It offers methods to fetch FHIR data based on user
    IDs and LOINC codes.

    Attributes:
        project_id (str): Identifier of the Firebase project.
        service_account_key_file (str | None): Path to the Firebase service account key file for
                                        authentication.
        db (firestore.Client | None): A Firestore client instance for database operations,
                                          initialized upon successful connection.
    """

    def __init__(
        self,
        project_id: Optional[  # pylint: disable=consider-alternative-union-syntax
            str
        ] = None,
        service_account_key_file: Optional[  # pylint: disable=consider-alternative-union-syntax
            str
        ] = None,
        db: Optional[  # pylint: disable=consider-alternative-union-syntax
            firestore.client
        ] = None,
    ) -> None:
        """
        Initializes the FirebaseFHIRAccess instance with Firebase service account
        credentials and project ID.
        """
        self.project_id = project_id
        self.service_account_key_file = service_account_key_file
        self.db = db

    def connect(self) -> None:
        """
        Establishes a connection to the Firebase Firestore database.
        """
        if self.db is not None:
            return

        try:
            # Attempt to retrieve the default app.
            app = firebase_admin.get_app()
            self.db = firestore.client(app=app)
        except ValueError as exc:
            # If it raises a ValueError, then the app hasn't been initialized.
            if (
                os.getenv(CI_STRING)
                or FIRESTORE_EMULATOR_HOST_KEY in os.environ
                or not os.path.exists(self.service_account_key_file)
            ):  # Check if running in CI environment
                # Point to the emulator for CI tests
                os.environ[FIRESTORE_EMULATOR_HOST_KEY] = LOCAL_HOST_URL
                os.environ[GCLOUD_PROJECT_STRING] = self.project_id

                firebase_admin.initialize_app(
                    options={FIREBASE_PROJECT_ID_PARAM_STRING: self.project_id}
                )
                self.db = firestore.Client(  # pylint: disable=no-member
                    project=self.project_id
                )
            else:  # Connect to the production environment
                cred = credentials.Certificate(self.service_account_key_file)
                firebase_admin.initialize_app(
                    cred, {FIREBASE_PROJECT_ID_PARAM_STRING: self.project_id}
                )
                self.db = firestore.client()

    def fetch_data(
        self,
        collection_name: str,
        subcollection_name: str,
        loinc_codes: list[str] | None = None,
    ) -> list[Resource]:
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
                resources. If None, all resources in the subcollection are fetched.

        Returns:
            list[Resource]: A list of FHIR resources instances matching the query criteria.
        """

        if self.db is None:
            print("Reinitialize the Firebase app.")
            return None

        if (
            loinc_codes is not None
            and loinc_codes.count(ECG_RECORDING_LOINC_CODE) > 0
            and len(loinc_codes) > 1
        ):
            print("HealthKit quantity types and ECG recordings cannot be downloaded ")
            print("simultaneously. Please review and adjust your selection to include ")
            print("only the necessary LOINC codes.")
            return None
        resources = []
        users = self.db.collection(collection_name).stream()
        for user in users:
            user_resources = self._fetch_user_resources(
                user, collection_name, subcollection_name, loinc_codes
            )
            resources.extend(user_resources)
        return resources

    def fetch_data_path(
        self,
        full_path: str,
        loinc_codes: list[str] | None = None,
        index_name: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None
    ) -> list[Resource]:
        """
        Retrieves FHIR Observation data for specified LOINC codes from Firestore.
        Data is fetched from the given collection and subcollection, optionally
        filtered by the provided LOINC codes.

        Parameters:
            full_path (str): The fully specified path to a Firestore collection.
            loinc_codes (list[str] | None): Optional list of LOINC codes to filter
                resources. If None, all resources in the subcollection are fetched.
            index_name (str | None): The name of the Firebase index that has a registered filter
            start_date (str | None): The start date for Firestore query index filter
            end_date (str | None): The end date for Firestore query index filter

        Returns:
            list[Resource]: A list of FHIR resources instances matching the query criteria.
        """

        if self.db is None:
            print("Reinitialize the Firebase app.")
            return None

        if (
            loinc_codes is not None
            and loinc_codes.count(ECG_RECORDING_LOINC_CODE) > 0
            and len(loinc_codes) > 1
        ):
            print("HealthKit quantity types and ECG recordings cannot be downloaded ")
            print("simultaneously. Please review and adjust your selection to include ")
            print("only the necessary LOINC codes.")
            return None

        path_ref = self.db.collection(full_path)
        resources = []
        if start_date:
            path_ref = path_ref.where(index_name, ">=", start_date)
        if end_date:
            path_ref = path_ref.where(index_name, "<=", end_date)
        if loinc_codes:
            resources.extend(_process_loinc_codes(path_ref, None, loinc_codes))
        else:
            resources.extend(_process_all_documents(path_ref, None))

        return resources

    def _fetch_user_resources(
        self,
        user: DocumentReference,
        collection_name: str,
        subcollection_name: str,
        loinc_codes: list[str] | None,
    ) -> list[Resource]:
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
            list[Resource]: List of FHIR resources corresponding to the user and
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
) -> list[Resource]:
    """
    Filters documents based on LOINC codes from a Firestore collection reference. This function
    processes and converts matching Firestore documents into FHIR Observation instances.

    Parameters:
        query (CollectionReference): Firestore query object for a user's subcollection.
        user (DocumentReference): Firestore reference to the user document.
        loinc_codes (list[str]): List of LOINC codes to filter documents.

    Returns:
        list[Resource]: A list of FHIR resources that match the specified LOINC codes.
    """

    resources = []
    for code in loinc_codes:
        display_str, code_str, system_str = get_code_mappings(code)
        fhir_docs = list(
            query.where(
                filter=FieldFilter(
                    "code.coding",
                    "array_contains",
                    {
                        KeyNames.DISPLAY.value: display_str,
                        KeyNames.SYSTEM.value: system_str,
                        KeyNames.CODE.value: code_str,
                    },
                )
            ).stream()
        )

        if not fhir_docs:
            continue

        first_doc_dict = fhir_docs[0].to_dict()
        resource_type = first_doc_dict[KeyNames.RESOURCE_TYPE.value]

        if resource_type == FHIRResourceType.OBSERVATION.value:
            creator = ObservationCreator()
            resources.extend(creator.create_resources(fhir_docs, user))
        elif resource_type == FHIRResourceType.QUESTIONNAIRE_RESPONSE.value:
            creator = QuestionnaireResponseCreator()
            resources.extend(creator.create_resources(fhir_docs, user))
        else:
            raise ValueError(f"Unsupported resource type: {resource_type}")

    return resources


def _process_all_documents(
    query: CollectionReference, user: DocumentReference
) -> list[Resource]:
    """
    Fetches and processes all documents from a Firestore collection reference for a specific user,
    converting each Firestore document to a FHIR Resource instance.

    Parameters:
        query (CollectionReference): Firestore query object for a user's subcollection.
        user (DocumentReference): Firestore reference to the user document.

    Returns:
        list[Resource]: List of FHIR resources for all documents in the user's subcollection.
    """
    resources = []

    if not (fhir_docs := list(query.stream())):
        return resources

    first_doc_dict = fhir_docs[0].to_dict()
    resource_type = first_doc_dict[KeyNames.RESOURCE_TYPE.value]

    if resource_type == FHIRResourceType.OBSERVATION.value:
        creator = ObservationCreator()
        resources = creator.create_resources(fhir_docs, user)
    elif resource_type == FHIRResourceType.QUESTIONNAIRE_RESPONSE.value:
        creator = QuestionnaireResponseCreator()
        resources = creator.create_resources(fhir_docs, user)
    else:
        raise ValueError(f"Unsupported resource type: {resource_type}")

    return resources


@dataclass
class ResourceCreator:
    """Abstract base class for creating FHIR resources based on the resource type"""

    def __init__(self, resource_type: FHIRResourceType):
        self.resource_type = resource_type

    def create_resources(
        self, fhir_docs: list[DocumentSnapshot], user: DocumentReference
    ) -> list[Any]:
        raise NotImplementedError("Subclasses should implement this method.")


@dataclass
class ObservationCreator(ResourceCreator):
    """
    A specialized resource creator that converts Firestore document snapshots into FHIR Observation
    instances. Each Observation instance is associated with a specific user, identified by the
    Firestore document ID of the user, which is set as the subject reference of the Observation.

    Inherits from:
        ResourceCreator: A base class for creating FHIR resources.

    Initialization:
        Calls the superclass initializer with FHIRResourceType.OBSERVATION to specify the type of
        FHIR resource.

    Methods:
        create_resources: Converts an iterable of Firestore document snapshots into a list of
        Observation instances, handling specific data fields and ensuring each Observation
        references the correct user.
    """

    def __init__(self):
        super().__init__(FHIRResourceType.OBSERVATION)

    def create_resources(
        self, fhir_docs: list[DocumentSnapshot], user: DocumentReference
    ) -> list[Observation]:
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
            doc_dict = doc.to_dict()
            # The following removals will be omitted
            doc_dict.pop("issued", None)
            doc_dict.pop("document_id", None)
            doc_dict.pop("physicianAssignedDiagnosis", None)
            doc_dict.pop("physician", None)
            doc_dict.pop("tracingQuality", None)

            resource_str = json.dumps(doc_dict)
            resource_obj = Observation.parse_raw(resource_str)
            if user:
                resource_obj.subject = Reference(id=user.id)

            # Special handling for ECG data
            if (
                len(resource_obj.code.coding) > 1
                and hasattr(
                    resource_obj.code.coding[1],  # pylint: disable=no-member
                    KeyNames.CODE.value,
                )
                and resource_obj.code.coding[1].code  # pylint: disable=no-member
                == ECG_RECORDING_LOINC_CODE
            ):
                ecg_resource_obj = ECGObservation(resource_obj)
                resources.append(ecg_resource_obj)
            else:
                resources.append(resource_obj)
        return resources


@dataclass
class QuestionnaireResponseCreator(ResourceCreator):
    """
    A specialized resource creator that converts Firestore document snapshots into FHIR
    QuestionnaireResponse instances. Each QuestionnaireResponse instance is associated with
    a specific user, identified by the Firestore document ID of the user, which is set as the
    subject reference of the QuestionnaireResponse.

    Inherits from:
        ResourceCreator: A base class for creating FHIR resources.

    Initialization:
        Calls the superclass initializer with FHIRResourceType.QUESTIONNAIRE_RESPONSE to specify
        the type of FHIR resource.

    Methods:
        create_resources: Converts an iterable of Firestore document snapshots into a list of
        QuestionnaireResponse instances, ensuring each QuestionnaireResponse references the
        correct user.
    """

    def __init__(self):
        super().__init__(FHIRResourceType.QUESTIONNAIRE_RESPONSE)

    def create_resources(
        self, fhir_docs: list[DocumentSnapshot], user: DocumentReference
    ) -> list[QuestionnaireResponse]:
        """
        Converts Firestore documents into FHIR QuestionnaireResponse instances, setting the
        subject reference to the user's Firestore document ID.

        Parameters:
            fhir_docs (list[DocumentSnapshot]): Iterable of Firestore document snapshots containing
                FHIR questionnaire data.
            user (DocumentReference): Firestore reference to the user document.

        Returns:
            list[QuestionnaireResponse]: List of FHIR QuestionnaireResponse instances created from
            the Firestore documents.
        """
        resources = []
        for doc in fhir_docs:
            doc_dict = doc.to_dict()
            resource_str = json.dumps(doc_dict)
            resource_obj = QuestionnaireResponse.parse_raw(resource_str)
            if user:
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
