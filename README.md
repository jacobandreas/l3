# L3

To use:
- download the data from http://nlp.stanford.edu/data/muj/shapeworld_4k.tar.gz, and untar inside `data` folder, to create `data/shapeworld` folder
- use python 2, e.g.
```
virtualenv -p python2 .venv
source .venv/bin/activate
```
- install requirements.txt, e.g.
```
pip install -r requirements.txt
```
- choose a command-line from `exp` folder, eg `cls_hint` is for L3
    - `ex` is Meta, `gold` is ground-truth descriptions, `hint` is L^3, `joint` is Meta+Joint, `sim` is another baseline that didn't make it into the paper, and `vis` is just visualization.
    - For the other two experiments, evaluation is done in a different script from training; those have the `_eval` suffix.
