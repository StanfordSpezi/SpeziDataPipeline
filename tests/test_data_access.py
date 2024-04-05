#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Docstring to be added.
"""

# Related third-party imports
import unittest
from unittest.mock import patch

# Local application/library specific imports
from data_access.firebase_fhir_data_access import FirebaseFHIRAccess


class TestFirebaseFHIRAccess(unittest.TestCase):  # pylint: disable=unused-variable
    """
    Test suite for testing the FirebaseFHIRAccess class.

    This class is designed to test the connectivity and initialization of the FirebaseFHIRAccess
    class, ensuring that the Firebase Firestore database can be accessed correctly using the
    provided Firebase project credentials.
    """

    @patch("data_access.firebase_fhir_data_access.firestore.client")
    @patch("data_access.firebase_fhir_data_access.credentials.Certificate")
    @patch("data_access.firebase_fhir_data_access.firebase_admin.initialize_app")
    @patch(
        "data_access.firebase_fhir_data_access.firebase_admin.get_app",
        side_effect=ValueError,
    )
    @patch("data_access.firebase_fhir_data_access.os.getenv", return_value=None)
    @patch("data_access.firebase_fhir_data_access.os.path.exists", return_value=True)
    def test_connect_production_with_valid_key(  # pylint: disable
        self,
        mock_exists,
        mock_initialize_app,
        mock_certificate,
    ):
        """
        Tests the FirebaseFHIRAccess connection in a production environment with a valid service
        account key.

        This test ensures that when provided with a valid service account key file path, the
        FirebaseFHIRAccess object can successfully initialize and connect to the Firebase Firestore
        database.
        """
        dummy_key_path = "dummy_service_account_key.json"
        access = FirebaseFHIRAccess(
            project_id="test_project", service_account_key_file=dummy_key_path
        )
        access.connect()

        mock_exists.assert_called_once_with(dummy_key_path)
        mock_certificate.assert_called_once_with(dummy_key_path)
        mock_initialize_app.assert_called_once()

    @patch(
        "data_access.firebase_fhir_data_access.firebase_admin.get_app",
        side_effect=ValueError,
    )
    @patch("data_access.firebase_fhir_data_access.os.path.exists", return_value=False)
    def test_connect_without_valid_key_raises_error(self):
        """
        Tests that attempting to connect without a valid service account key raises a
        FileNotFoundError.

        This test verifies that if the FirebaseFHIRAccess object is provided with an
        invalid path to the service account key file, it properly raises a FileNotFoundError
        upon attempting to connect.
        """
        dummy_project_id = "dummy_project"
        dummy_key_path = "invalid/path/to/service_account_key.json"

        access = FirebaseFHIRAccess(
            project_id=dummy_project_id, service_account_key_file=dummy_key_path
        )

        with self.assertRaises(FileNotFoundError):
            access.connect()


if __name__ == "__main__":
    unittest.main()