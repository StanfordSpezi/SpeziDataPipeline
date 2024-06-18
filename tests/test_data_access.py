#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module contains unit tests for the `FirebaseFHIRAccess` class, which is responsible for
managing connections to the Firebase Firestore database in a healthcare data context for
FHIR (Fast Healthcare Interoperability Resources) data.

The tests in this module ensure that the `FirebaseFHIRAccess` class can handle the setup and
initialization of connections to Firestore using Firebase project credentials. It checks both
the scenarios where the service account key file is valid and invalid, verifying the correct
handling of authentication and connection establishment.

These tests utilize the unittest framework and apply mocking to the Firebase Admin SDK components
to isolate the tests from actual Firebase infrastructure. This approach ensures that the tests
can be run in any environment without needing access to real Firebase project credentials.

Classes:
    `TestFirebaseFHIRAccess`: Contains all unit tests for testing the connectivity and
                              initialization capabilities of the `FirebaseFHIRAccess` class.
"""


# Related third-party imports
import unittest
from unittest.mock import patch, MagicMock
import json
from fhir.resources.R4B.observation import Observation
from fhir.resources.R4B.questionnaireresponse import QuestionnaireResponse

# Local application/library specific imports
from spezi_data_pipeline.data_access.firebase_fhir_data_access import (
    FirebaseFHIRAccess,
    ObservationCreator,
    ECGObservation,
    QuestionnaireResponseCreator,
)

FIRESTORE_EMULATOR_HOST_KEY = "FIRESTORE_EMULATOR_HOST"
LOCAL_HOST_URL = "localhost:8080"
GCLOUD_PROJECT_STRING = "GCLOUD_PROJECT"
ECG_RECORDING_LOINC_CODE = "131328"


class TestFirebaseFHIRAccess(unittest.TestCase):  # pylint: disable=unused-variable
    """
    Unit tests for the FirebaseFHIRAccess class.

    These tests verify the connection handling of the FirebaseFHIRAccess class,
    ensuring it correctly manages Firestore connections under various scenarios including:
    - Connection attempts in CI environments without a service account.
    - Connection attempts with a valid service account.
    - Data fetching from Firestore with valid and invalid LOINC code combinations.
    """

    def setUp(self):
        self.project_id = "test-project"
        self.service_account_key_file = "/path/to/service/account.json"

    @patch("os.path.exists")
    @patch("os.environ")
    @patch("firebase_admin.initialize_app")
    @patch("firebase_admin.get_app")
    def test_connect_no_service_account_in_ci(
        self,
        mock_get_app,  # pylint: disable=unused-argument
        mock_init_app,
        mock_environ,
        mock_exists,
    ):
        # Setup environment and file check to simulate CI with no service account file
        mock_environ.get.return_value = True
        mock_exists.return_value = False
        mock_environ.__contains__.return_value = False

        # Instantiate and connect
        firebase_access = FirebaseFHIRAccess(self.project_id)
        firebase_access.connect()

        # Verify the initialization is pointed to the emulator and project ID is set correctly
        calls = [
            ((FIRESTORE_EMULATOR_HOST_KEY, LOCAL_HOST_URL),),
            ((GCLOUD_PROJECT_STRING, self.project_id),),
        ]
        mock_environ.__setitem__.assert_has_calls(calls, any_order=True)
        mock_init_app.assert_called_once()

    @patch("spezi_data_pipeline.data_access.firebase_fhir_data_access.firestore.client")
    @patch(
        "spezi_data_pipeline.data_access.firebase_fhir_data_access.credentials.Certificate"
    )
    @patch(
        "spezi_data_pipeline.data_access.firebase_fhir_data_access.firebase_admin.initialize_app"
    )
    @patch(
        "spezi_data_pipeline.data_access.firebase_fhir_data_access.firebase_admin.get_app",
        side_effect=ValueError,
    )
    @patch(
        "spezi_data_pipeline.data_access.firebase_fhir_data_access.os.getenv",
        return_value=None,
    )
    @patch(
        "spezi_data_pipeline.data_access.firebase_fhir_data_access.os.path.exists",
        return_value=True,
    )
    # pylint: disable=too-many-arguments
    def test_connect_production_with_valid_key(
        self,
        mock_exists,
        mock_getenv,  # pylint: disable=unused-argument
        mock_get_app,  # pylint: disable=unused-argument
        mock_initialize_app,
        mock_certificate,
        mock_client,  # pylint: disable=unused-argument
    ):
        """
        Tests the FirebaseFHIRAccess connection in a production environment with a valid service
        account key.

        This test ensures that when provided with a valid service account key file path, the
        FirebaseFHIRAccess object can successfully initialize and connect to the Firebase Firestore
        database.
        """
        # dummy_key_path = "dummy_service_account_key.json"
        access = FirebaseFHIRAccess(
            project_id=self.project_id,
            service_account_key_file=self.service_account_key_file,
        )
        access.connect()

        mock_exists.assert_called_once_with(self.service_account_key_file)
        mock_certificate.assert_called_once_with(self.service_account_key_file)
        mock_initialize_app.assert_called_once()

    @patch("firebase_admin.firestore")
    def test_fetch_data_invalid_loinc_combination(self, mock_firestore):
        # Setup for error on invalid LOINC code combination
        mock_db = MagicMock()
        mock_firestore.client.return_value = mock_db
        firebase_access = FirebaseFHIRAccess(self.project_id)
        firebase_access.db = mock_db

        result = firebase_access.fetch_data(
            "users", "HealthKit", [ECG_RECORDING_LOINC_CODE, "9999-4"]
        )

        # Verify expected failure on invalid combination
        self.assertIsNone(result)

    @patch("firebase_admin.firestore")
    def test_fetch_data_valid_loinc_code(self, mock_firestore):
        mock_db = MagicMock()
        mock_firestore.client.return_value = mock_db
        firebase_access = FirebaseFHIRAccess(self.project_id)
        firebase_access.db = mock_db

        # Mock collection, document, and subcollection retrieval
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection
        mock_user_doc = MagicMock()
        mock_user_stream = iter([mock_user_doc])
        mock_collection.stream.return_value = mock_user_stream
        mock_subcollection = MagicMock()
        mock_user_doc.collection.return_value = mock_subcollection
        mock_subcollection.stream.return_value = iter(
            []
        )  # Simulate no data within "HealthKit"

        result = firebase_access.fetch_data(
            "users", "HealthKit", [ECG_RECORDING_LOINC_CODE]
        )

        # Verify
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 0)


class TestObservationCreator(unittest.TestCase):  # pylint: disable=unused-variable
    """
    Test the successful creation of an Observation object.
    """

    def test_create_resources(self):
        doc_snapshot = MagicMock()
        file_path = "sample_data/XrftRMc358NndzcRWEQ7P2MxvabZ_sample_data1.json"
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        doc_snapshot.to_dict.return_value = data
        user_ref = MagicMock()
        user_ref.id = "XrftRMc358NndzcRWEQ7P2MxvabZ"

        creator = ObservationCreator()

        results = creator.create_resources([doc_snapshot], user_ref)

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Observation)
        self.assertEqual(results[0].subject.id, user_ref.id)

    def test_ecg_resources(self):
        doc_snapshot = MagicMock()
        file_path = "sample_data/3aX1qRKWQKTRDQZqr5vg5N7yWU12_sample_ecg_data1.json"
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        doc_snapshot.to_dict.return_value = data
        user_ref = MagicMock()
        user_ref.id = "3aX1qRKWQKTRDQZqr5vg5N7yWU12"

        creator = ObservationCreator()

        results = creator.create_resources([doc_snapshot], user_ref)

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], ECGObservation)
        self.assertEqual(results[0].subject.id, user_ref.id)


class TestQuestionnaireResponseCreator(
    unittest.TestCase
):  # pylint: disable=unused-variable
    """
    Test the successful creation of a QuestionnaireResponse object.
    """

    def test_create_resources(self):
        doc_snapshot = MagicMock()
        user_id = "5tTYsEWMIKNq4EJEf24suVINGI12"
        file_path = f"sample_data/{user_id}_sample_questionnaire_response_data1.json"
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        doc_snapshot.to_dict.return_value = data
        user_ref = MagicMock()
        user_ref.id = user_id

        creator = QuestionnaireResponseCreator()

        results = creator.create_resources([doc_snapshot], user_ref)

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], QuestionnaireResponse)
        self.assertEqual(results[0].subject.id, user_ref.id)


if __name__ == "__main__":
    unittest.main()
