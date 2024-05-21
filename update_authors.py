#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This script updates the 'authors' section of the pyproject.toml file based on the CONTRIBUTORS.md
file.

It parses the CONTRIBUTORS.md file to extract the list of contributors and updates the 
pyproject.toml file to reflect these contributors in the 'authors' section.

Functions:
- parse_contributors(file_path): Parses the CONTRIBUTORS.md file to extract author names.
- update_pyproject_toml(pyproject_path, contributors): Updates the pyproject.toml file with the 
    list of authors.
"""

import re
import toml


def parse_contributors(file_path):
    """
    Parse the CONTRIBUTORS.md file to extract author names.

    Args:
        file_path (str): The path to the CONTRIBUTORS.md file.

    Returns:
        list: A list of dictionaries with author names.
    """
    contributors = []
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            if match := re.match(r"\*\s\[(.*)\]\(.*\)", line.strip()):
                name = match.group(1)
                contributors.append({"name": name})
    return contributors


def update_pyproject_toml(pyproject_path, contributors):
    """
    Update the pyproject.toml file with the list of authors.

    Args:
        pyproject_path (str): The path to the pyproject.toml file.
        contributors (list): A list of dictionaries with author names.
    """
    with open(pyproject_path, "r", encoding="utf-8") as file:
        pyproject_data = toml.load(file)

    pyproject_data["project"]["authors"] = contributors

    with open(pyproject_path, "w", encoding="utf-8") as file:
        toml.dump(pyproject_data, file)


if __name__ == "__main__":
    contributors_file_path = "CONTRIBUTORS.md"
    pyproject_toml_path = "pyproject.toml"

    contributors_list = parse_contributors(contributors_file_path)
    update_pyproject_toml(pyproject_toml_path, contributors_list)
