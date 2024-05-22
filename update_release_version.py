#
# This source file is part of the Stanford Spezi open source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

import subprocess
import re
import toml


def get_latest_git_tag():
    """Function to get the latest git tag"""
    result = subprocess.run(
        ["git", "describe", "--tags"], stdout=subprocess.PIPE, text=True
    )
    tag = result.stdout.strip()
    return tag


def update_hatch_version(tag):
    """Function to update the version in the hatch configuration file"""

    with open("pyproject.toml", "r") as file:
        config = toml.load(file)

    config["tool"]["hatch"]["metadata"]["version"] = tag

    with open("pyproject.toml", "w") as file:
        toml.dump(config, file)


if __name__ == "__main__":
    latest_tag = get_latest_git_tag()
    if re.match(r"^\d+\.\d+\.\d+$", latest_tag):
        update_hatch_version(latest_tag)
        print(f"Updated pyproject.toml with version {latest_tag}")
    else:
        print(f"Invalid tag format: {latest_tag}")
