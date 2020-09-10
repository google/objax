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

if python3 -c "import pytest" &> /dev/null ; then
  # If pytest is installed then use it to run tests
  # Pytest has nicer output compared to unittest package and also it's used
  # to run automatic unit tests on GitHub.
  CUDA_VISIBLE_DEVICES= pytest tests/*.py
else
  # If pytest is not installed then use default unittest to run tests.
  for i in tests/*.py; do
    CUDA_VISIBLE_DEVICES= python3 -m unittest $i >&$i.log &
  done
  wait
  fgrep FAILED tests/*.log
fi
