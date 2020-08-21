# Documentation folder

The document uses `.rst` format which stands for reStructuredText
(reST)](https://docutils.sourceforge.io/docs/user/rst/quickstart.html).

[Cheat sheet](http://openalea.gforge.inria.fr/doc/openalea/doc/_build/html/source/sphinx/rest_syntax.html)
for reST.

## Initial setup

```bash
# Install python libraries
pip install --upgrade -r docs/requirements.txt

# Install pandoc, see also https://pandoc.org/installing.html
sudo apt install pandoc
```

## Building

```bash
cd docs
make clean
PYTHONPATH=$PYTHONPATH:.. make html
```
