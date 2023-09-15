# Documentation

We use [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
along with [Sphinx](https://www.sphinx-doc.org/en/master/index.html) for generating documentation.


## Commands

Below are some useful commands to use for project documentation.

```bash
cd docs
# Cleans previous docs builds
poetry run make clean
# Write the documents output to HTML files in "/docs/build/html" (open index.html to see documentation)
poetry run make html
# Check broken links in the documentation
poetry run make linkcheck
# Auto build documentation when file is updated
poetry run python -m sphinx_autobuild ./source/ build/html/
```
