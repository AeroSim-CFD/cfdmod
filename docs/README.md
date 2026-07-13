# Documentation

We use [MyST Markdown](https://myst-parser.readthedocs.io/en/latest/)
along with [Sphinx](https://www.sphinx-doc.org/en/master/index.html) for generating documentation.
Pages are authored as `.md` (MyST); the tutorial pages are Jupyter notebooks (`.ipynb`).


## Commands

Below are some useful commands to use for project documentation.

```bash
cd docs
# Cleans previous docs builds
uv run make clean
# Write the documents output to HTML files in "/docs/build/html" (open index.html to see documentation)
uv run make html
# Check broken links in the documentation
uv run make linkcheck
# Auto build documentation when file is updated
uv run python -m sphinx_autobuild ./source/ build/html/
```
