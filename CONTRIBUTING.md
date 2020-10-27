# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

Below are some basic requirements, for a more detailed discussion see the
[setup guide](https://objax.readthedocs.io/en/latest/dev/setup.html).

In addition to them take a look at a
[guide on adding new modules](https://objax.readthedocs.io/en/latest/dev/adding_module.html).

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Running Tests

Before submitting a PR, it can help to run the unit tests and linter. Install the linter
```bash
pip install flake8
```

and then run
```bash
./tests/run_linter.sh
./tests/run_tests.sh
```
to confirm that the tests all pass.

A single test can be run with
```bash
CUDA_VISIBLE_DEVICES= python3 -m unittest tests/jit.py
```

## Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google.com/conduct/).
