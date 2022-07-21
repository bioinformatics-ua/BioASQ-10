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

At this point, everything is set up and ready to run. To reproduce our submission runs for both tasks just execute `run.sh`. Furthermore, what this script does is just executing the `run.sh` script present in the synergy and taskb folders. So, if you are just interested in one of the task, just go to the respective folder and execute the local `run.sh` script. 

### Directory structure

```
.
├── synergy/
│       Code base related to the synergy challenge.
│
└── taskb/
        Code base related to the taskb challenge (document retrieval and yes or no answering).

```

## System specifications

For computational reference, our experiments were performed on a server machine with the following characteristics:

- Operating system: Ubuntu 18.04
- CPU: Intel Xeon E5-2630 v4 (40) @ 3.1GHz
- GPU: NVIDIA Tesla K80
- 128 GB RAM

## Team
  * Tiago Almeida<sup id="a1">[1](#f1)</sup>
  * André Pinho<sup id="a1">[1](#f1)</sup>
  * Rodrigo Pereira<sup id="a1">[1](#f1)</sup>
  * Sérgio Matos<sup id="a1">[1](#f1)</sup>

1. <small id="f1"> University of Aveiro, Department of Electronics, Telecommunications and Informatics (DETI), Institute of Electronics and Informatics Engineering of Aveiro (IEETA), Aveiro, Portugal </small> [↩](#a1)

## Reference

Please cite our paper if you use this code in your work:

TODO: Fix this reference when the paper becomes available.

```
@article{almeida2022a,
  author    = {Almeida, Tiago and Pinho, André and Pereira, Rodrigo and Matos, S{\'e}rgio},
  title     = {Deep Learning solutions based on fixed contextualized embeddings from PubMedBERT on BioASQ 10b and traditional IR in Synergy},
  url       = {https://github.com/bioinformatics-ua/BioASQ-10},
  volume    = {2022},
  year      = {2022},
}
```