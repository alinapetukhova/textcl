
# Developer's guide

Contributing to **TextCL** is easy. First, clone this repository and `cd` into the project's folder:

```text
git clone https://github.com/alinapetukhova/textcl.git
cd textcl
```

Then create a virtual development environment to test and experiment with the package:

```text
python3 -m venv env
source env/bin/activate
pip install -e .
```

The [pytest](https://docs.pytest.org/en/stable/), [pytest-cov](https://pypi.org/project/pytest-cov/) and [pdoc3](https://pdoc3.github.io/pdoc/) packages are required for testing **TextCL** and generating its documentation:

```text
pip install pytest pytest-cov pdoc3
```

Running the unit tests can be done with the following command from the project's root folder:

```text
pytest
```

To check test coverage, execute the following command:

```text
pytest --cov=textcl --cov-report=html
```

Project documentation can be generated with [pdoc3](https://pdoc3.github.io/pdoc/). For example, running the following command in the project's root folder generates the HTML documentation and places it in the `docs` folder:

```text
pdoc3 --html --output-dir docs textcl/
```
