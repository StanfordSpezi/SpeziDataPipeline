#
# This source file is part of the Stanford Spezi open source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Module to automatically update the Hatch project version based on the latest Git tag.

This module contains functions to retrieve the latest Git tag and update the 
Hatch version in the `pyproject.toml` configuration file. The script is intended 
to be run as a standalone program.

Functions:
    get_latest_git_tag(): Retrieves the latest Git tag from the repository.
    update_hatch_version(tag): Updates the version in the Hatch configuration file 
                               with the specified tag.

Usage:
    Run this script from the command line to automatically update the version in 
    the `pyproject.toml` file based on the latest Git tag:
    
    $ python script_name.py
    
    The script will validate that the tag is in a valid semantic versioning format 
    (e.g., "1.0.0"). If the tag is valid, it will update the version in 
    `pyproject.toml`. If the tag is not valid, it will print an error message.
"""

import os
import subprocess
import re
import toml


def get_latest_git_tag():
    """
    Retrieves the latest Git tag from the repository.

    This function runs the Git command `git describe --tags` to get the latest tag
    in the repository and returns it as a string.

    Returns:
        str: The latest Git tag or None if no tag is found.
    """
    try:
        result = subprocess.run(
            ["git", "describe", "--tags"], stdout=subprocess.PIPE, text=True, check=True
        )
        tag = result.stdout.strip()
        return tag
    except subprocess.CalledProcessError:
        return None


def update_hatch_version(tag):
    """Function to update the version in the hatch configuration file"""
    with open("pyproject.toml", "r", encoding="utf-8") as file:
        config = toml.load(file)

    config["tool"]["hatch"]["metadata"]["version"] = tag

    with open("pyproject.toml", "w", encoding="utf-8") as file:
        toml.dump(config, file)


if __name__ == "__main__":
    # Check if a tag is provided from the workflow_dispatch input
    provided_tag = os.getenv("INPUT_TAG_NAME")

    # Updated regex to match versions with a pattern [0-9].[0-9].[0-9] followed by any characters
    version_regex = r"^\d+\.\d+\.\d+.*$"

    if provided_tag and re.match(version_regex, provided_tag):
        update_hatch_version(provided_tag)
        print(f"Updated pyproject.toml with version {provided_tag}")
    else:
        latest_tag = get_latest_git_tag()
        if latest_tag and re.match(version_regex, latest_tag):
            update_hatch_version(latest_tag)
            print(f"Updated pyproject.toml with version {latest_tag}")
        else:
            print("No valid tag found. Unable to update version.")
            