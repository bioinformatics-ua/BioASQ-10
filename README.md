# BioASQ-10 task b and synergy

### Deep Learning solutions based on fixed contextualized embeddings from PubMedBERT on BioASQ 10b and traditional IR in Synergy

This repository exposes, in a reproducible way, our submission code for the 10th edition of the BioASQ. Given that we enrolled in task b and synergy, we organized the code separately for both challenges in the `taskb` and `synergy` folders respectively.

## Installation with docker (easy approach)

TODO, will be available when everything is configured inside a docker image

## Standard installation

First, make sure you have at least python3.8 installed. (We used python 3.8.10).
Then, we recommend creating a virtual environment using the venv package, so if not installed please install the package python3.8-venv.

Next, create a virtual python environment and install the requirements needed for this repository. For instance, the code below creates, activates and updates an environment called "py-bioasq"

```
$ python3.8 -m venv py-bioasq
$ source py-bioasq/bin/activate
$ python -m pip install --upgrade pip
$ pip install -r requirements.txt
```

After this step, run the `setup.sh` script in order to download the reminder larger files that are required for these repositories. Note that this step will probably take a long time since it will download the collections checkpoints, models checkpoints, cache files, datasets and testsets.

```
$ ./setup.sh
```

At this point, everything is set up and ready to run.