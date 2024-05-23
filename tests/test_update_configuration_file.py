#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Unit tests for verifying contributors parsing and Git tag version updating in the pyproject.toml
configuration file.

This module contains unit tests for:
1. Parsing authors from the CONTRIBUTORS.md file and updating the pyproject.toml configuration file.
2. Retrieving the latest Git tag and updating the version in the pyproject.toml configuration file.

Classes:
    TestContributorsFunctions: Contains unit tests for parsing and updating contributors 
                               in the pyproject.toml file.
    TestVersionFunctions: Contains unit tests for the functions get_latest_git_tag and 
                          update_hatch_version.

Functions:
    test_parse_contributors(self): Tests parsing of authors from the CONTRIBUTORS.md file in 
                                   TestContributorsFunctions.
    test_update_pyproject_toml(self): Tests updating of the pyproject.toml file with parsed authors
                                      in TestContributorsFunctions.
    test_get_latest_git_tag(self, mock_run): Tests the retrieval of the latest Git tag using the git
                                             command in TestVersionFunctions.
    test_update_hatch_version(self, mock_file): Tests the updating of the hatch version in the 
                                                pyproject.toml file in TestVersionFunctions.
"""
# Related third-party imports
import unittest
from unittest.mock import mock_open, patch

import re

# Local application/library specific imports
from scripts.update_authors import parse_contributors, update_pyproject_toml
from scripts.update_release_version import get_latest_git_tag, update_hatch_version


class TestContributorsFunctions(unittest.TestCase):  # pylint: disable=unused-variable
    """
    Tests that authors from the CONNTRIBUTORS.md file are correctely parsed in the
    pyproject.toml configuration file.
    """

    def test_parse_contributors(self):
        mock_contributors_md = """
        * [Alice](https://github.com/alice)
        * [Bob](https://github.com/bob)
        * [Charlie](https://github.com/charlie)
        """

        expected_output = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Charlie"}]

        with patch("builtins.open", mock_open(read_data=mock_contributors_md)):
            result = parse_contributors("CONTRIBUTORS.md")
            self.assertEqual(result, expected_output)

    def test_update_pyproject_toml(self):
        mock_pyproject_toml = """
        [project]
        name = "example_project"
        version = "0.1.0"
        """

        contributors = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Charlie"}]

        with patch(
            "builtins.open", mock_open(read_data=mock_pyproject_toml)
        ) as mock_file:
            update_pyproject_toml("pyproject.toml", contributors)
            mock_file.assert_called_with("pyproject.toml", "w", encoding="utf-8")
            written_data = mock_file().write.call_args[0][0]
            self.assertIn('name = "Alice"', written_data)
            self.assertIn('name = "Bob"', written_data)
            self.assertIn('name = "Charlie"', written_data)


class TestVersionFunctions(unittest.TestCase):  # pylint: disable=unused-variable
    """
    Contains unit tests for the functions get_latest_git_tag and update_hatch_version.

    Functions:
        test_get_latest_git_tag(self, mock_run): Tests the retrieval of the latest Git
                                                 tag using the git command.
        test_update_hatch_version(self, mock_file): Tests the updating of the hatch
                                                    version in the pyproject.toml file.
    """

    @patch("subprocess.run")
    def test_get_latest_git_tag(self, mock_run):
        mock_run.return_value.stdout = "v1.2.3\n"
        expected_tag = "v1.2.3"
        actual_tag = get_latest_git_tag()
        self.assertEqual(expected_tag, actual_tag)

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='[tool.hatch.metadata]\nversion = "0.1.0"\n',
    )
    def test_update_hatch_version(self, mock_file):  # pylint: disable=no-self-use
        new_tag = "v1.2.3"
        expected_output = '[tool.hatch.metadata]\nversion = "v1.2.3"\n'

        update_hatch_version(new_tag)

        mock_file.assert_called_with("pyproject.toml", "w", encoding="utf-8")
        mock_file().write.assert_called_once_with(expected_output)


if __name__ == "__main__":
    latest_tag = get_latest_git_tag()
    if re.match(r"^\d+\.\d+\.\d+$", latest_tag):
        update_hatch_version(latest_tag)
        print(f"Updated pyproject.toml with version {latest_tag}")
    else:
        print(f"Invalid tag format: {latest_tag}")

    unittest.main()
