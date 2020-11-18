#!/usr/bin/env bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Change directory to repository root
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# Run linter with following changes to default rules:
# - We allow assignment of lambda, thus ignore E731 error: https://www.flake8rules.com/rules/E731.html
# - Line break should occur before binary operator, thus between W503 and W504 ignore W503 and follow W504,
#   https://www.flake8rules.com/rules/W503.html
# - Set max line length to 120 characters
# - Separately lint __init__.py and other files, otherwise flake8 complains about unused imports in __init__.py
flake8 --exclude=__init__.py --max-line-length=120 --ignore=E731,W503 objax/ || exit 1
flake8 --filename=__init__.py --max-line-length=120 --ignore=E731,W503 objax/ || exit 1
flake8 --max-line-length=120 --ignore=E731,W503 tests/ || exit 1
