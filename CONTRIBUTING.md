# Contributing

Welcome to the `open_nipals` contributor's guide.

## Code Contributions

Code contributions should be made in a new feature branch. After a feature is completed please open a pull request and tag at least two people for review. In your request include:

* The purpose of the feature
* Background on why the feature is needed
* How to test the feature (e.g. example code)

This will follow the [semantic versioning scheme](https://semver.org/). If the changes are not reflected in the version number, please use `git tag`. See [this page](https://pyscaffold.org/en/stable/faq.html#best-practices-and-common-errors-with-version-numbers) for more information.

### Reviewing Code contributions

As a reviewer please ensure the following is true:
* Tests are functional
* New Pull Request features are documented (i.e. docstrings and type hints)
* the code is formatted with `ruff` and a line length of 79 characters per [PEP 8](https://peps.python.org/pep-0008/)

## Documentation Improvements

You can help improve `open_nipals` docs by making them more readable and coherent, or by adding missing information and correcting mistakes.

`open_nipals` documentation uses [Sphinx] as its main documentation compiler.
This means that the docs are kept in the same repository as the project code, and
that any documentation update is done in the same way was a code contribution.

When working on documentation changes in your local machine, you can
compile them using [tox] :

```
tox -e docs
```

and use Python's built-in web server for a preview in your web browser
(`http://localhost:8000`):

```
python3 -m http.server --directory 'docs/_build/html'
```

### Create an environment

Before you start coding, we recommend creating an isolated [virtual environment]
to avoid any problems with your installed Python packages.
This can easily be done via either [virtualenv]:

```
virtualenv <PATH TO VENV>
source <PATH TO VENV>/bin/activate
```

or [Miniconda]:

```
conda create -n open_nipals python=3 six virtualenv pytest pytest-cov
conda activate open_nipals
```

## Installation

1. Ensure you have the [git client installed](https://git-scm.com/downloads).
2. Clone the repository to your local computer.
3. Install the open_nipals package from the local folder:

```pip install -e .[testing]```

### Implement your changes

1. Create a branch to hold your changes:

   ```
   git checkout -b my-feature
   ```

   and start making changes. Never work on the main branch!

2. Start your work on this branch. Don't forget to add [docstrings] to new
   functions, modules and classes, especially if they are part of public APIs.

3. Add yourself to the list of contributors in `AUTHORS.md`.

4. When you’re done editing, do:

   ```
   git add <MODIFIED FILES>
   git commit
   ```

   to record your changes in [git].

5. Please check that your changes don't break any unit tests with:

   ```
   tox
   ```

   (after having installed [tox] with `pip install tox` or `pipx`).

   You can also use [tox] to run several other pre-configured tasks in the
   repository. Try `tox -av` to see a list of the available checks.

### Testing

Testing should begin with `pytest` and, eventually, end with `tox`.
`pytest` will illuminate any issues with the code itself while `tox` will
highlight dependency conflicts. It's *much* more expensive to run.

Prior to executing `tox` the first time, you will need to have all of the
python environments specified in the [tox.ini](./tox.ini) file loaded. No
need to build each combination, just a flat python 3.9, 3.10, 3.11, etc 
environment will suffice; `tox` should detect that and then build from there.

All PRs attempting to merge into the `main` branch of the public repository 
will trigger a github actions pipeline running `flake8`-linting and `tox`. 
Therefore, running `tox` is not strictly necessary for these kind of commits.
However, if in doubt, please run `tox` locally before pushing to the public 
repo. The github actions pipeline lives in the 
(./github/workflows/python-app.py)-file. If new python versions are included 
into the `tox` run, please do not only include them into the `tox.ini`, but 
also in the python matrix in the workflow file.

If you hit errors with `tox`, first try deleting the .tox folder.

### Troubleshooting

The following tips can be used when facing problems to build or test the
package:

1. Make sure to fetch all the tags from the upstream [repository].
   The command `git describe --abbrev=0 --tags` should return the version you
   are expecting. If you are trying to run CI scripts in a fork repository,
   make sure to push all the tags.
   You can also try to remove all the egg files or the complete egg folder, i.e.,
   `.eggs`, as well as the `*.egg-info` folders in the `src` folder or
   potentially in the root of your project.

2. Sometimes [tox] misses out when new dependencies are added, especially to
   `setup.cfg` and `docs/requirements.txt`. If you find any problems with
   missing dependencies when running a command with [tox], try to recreate the
   `tox` environment using the `-r` flag. For example, instead of:

   ```
   tox -e docs
   ```

   Try running:

   ```
   tox -r -e docs
   ```

3. Make sure to have a reliable [tox] installation that uses the correct
   Python version (e.g., 3.7+). When in doubt you can run:

   ```
   tox --version
   # OR
   which tox
   ```

   If you have trouble and are seeing weird errors upon running [tox], you can
   also try to create a dedicated [virtual environment] with a [tox] binary
   freshly installed. For example:

   ```
   virtualenv .venv
   source .venv/bin/activate
   .venv/bin/pip install tox
   .venv/bin/tox -e all
   ```

4. [Pytest can drop you] in an interactive session in the case an error occurs.
   In order to do that you need to pass a `--pdb` option (for example by
   running `tox -- -k <NAME OF THE FALLING TEST> --pdb`).
   You can also setup breakpoints manually instead of using the `--pdb` option.