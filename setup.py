# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

README_FILE = 'README.md'
REQUIREMENTS_FILE = 'requirements.txt'
VERSION_FILE = 'objax/_version.py'
VERSION_REGEXP = r'^__version__ = \'(\d+\.\d+\.\d+)\''

r = re.search(VERSION_REGEXP, open(VERSION_FILE).read(), re.M)
if r is None:
    raise RuntimeError(f'Unable to find version string in {VERSION_FILE}.')

version = r.group(1)
long_description = open(README_FILE, encoding='utf-8').read()
install_requires = [str(r) for r in parse_requirements(open(REQUIREMENTS_FILE, 'rt'))]

setup(
    name='objax',
    version=version,
    description='Objax is a machine learning framework that provides an Object Oriented layer for JAX.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Objax team',
    author_email='objax-dev@google.com',
    url='https://github.com/google/objax',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=install_requires,
)
