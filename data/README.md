~ ShapeWorld data can be downloaded at http://people.eecs.berkeley.edu/~jda/data/shapeworld.tar.gz~

Unfortunately, Berkeley seems to have deleted the version of the dataset used in
the original paper. Jesse Mu created a new version, which can be downloaded
[here](https://github.com/jayelm/lsl/tree/master/shapeworld) and gives similar
results; [his paper](https://arxiv.org/abs/1911.02683) reports accuracies for
both L^3 and the (better) LSL model.

If you want to generate a new version of the dataset yourself, code can be found
in the `ShapeWorld` directory here. To use it, download the
[ShapeWorld](https://github.com/AlexKuhnle/ShapeWorld) repository (I think you
need to go back to [this
commit](https://github.com/AlexKuhnle/ShapeWorld/tree/da452fc49f0a0dd18517153bf0f47ce7588065d7)),
place `spatial_jda.py` in `shapeworld/datasets/agreement`, then call
`generate.py`.

I'm still figuring out how to distribute the navigation data; for now, email me
at jda@mit.edu.
