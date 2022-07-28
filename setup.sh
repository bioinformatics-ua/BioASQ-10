#!/bin/bash

set -e

# create a virtual env

python -m venv py-bioasq
PYTHON=$(readlink -f py-bioasq/bin/python)
PIP=$(readlink -f py-bioasq/bin/pip)

# upgrade pip
$PIP install --upgrade pip

$PIP install -r requirements.txt

$PYTHON -m spacy download en_core_web_lg

# prepare the T-UPWM
cd taskb/local-T-UPWM
python -m venv t-upwm-venv
PYTHON=$(readlink -f t-upwm-venv/bin/python)
PIP=$(readlink -f t-upwm-venv/bin/pip)
$PIP install --upgrade pip

cd -

cd taskb/local-T-UPWM/nir/
if [ -d "./dist" ]
then
	rm -r ./dist
fi

$PYTHON setup.py sdist
$PIP install ./dist/nir-0.0.1.tar.gz
cd -

cd taskb/local-T-UPWM/mmnrm-lib
if [ -d "./dist" ]
then
	rm -r ./dist
fi

$PYTHON setup.py sdist
$PIP install ./dist/mmnrm-0.0.2.tar.gz

cd -


# auxiliary functions for donwloading the data
CordDownload() {
    
    local GZ_FILE="cord-19_"$1".tar.gz"
    
    if [ ! -f $1"/metadata.csv" ]; then
        echo "Download and untar $GZ_FILE"
        wget -c "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/"$GZ_FILE
        tar -xf $GZ_FILE
        rm $1"/changelog"
        rm $1"/document_parses.tar.gz"
        rm $GZ_FILE
        if [ -f $1"/cord_19_embeddings.tar.gz" ]; then
            rm $1"/cord_19_embeddings.tar.gz"
        fi
    fi
    
}

SHARE_BIOINFORMATICS_DOWNLOAD() {

    local URL_ZIP_FILE="https://share.bioinformatics-ua.pt/share.cgi?ssid=$2&fid=$2&filename=$1&openfolder=forcedownload&ep="

    if [ ! -f $3 ]; then
        echo "Download and unzip $1"
        wget -c -O $1 $URL_ZIP_FILE
        unzip -u $1
        rm $1
    fi
}



# prepare cord-19
echo "Preparing the cord-19 corpus"
mkdir -p synergy/data/cord-19


cd synergy/data/cord-19

CordDownload "2021-11-15"
CordDownload "2021-12-20"
CordDownload "2022-01-03"
CordDownload "2022-01-31"

cd -

echo "Preparing the BioASQ synergy dataset"
mkdir -p synergy/data/dataset

cd synergy/data/dataset

SHARE_BIOINFORMATICS_DOWNLOAD "BioASQ_Synergy9_v1.zip" "6a56d4b1f2ea4630a441c8c241bc6073" "BioASQ_Synergy9_v1/feedback_final.json"

SHARE_BIOINFORMATICS_DOWNLOAD "BioASQ_Synergy9_v2.zip" "476777f538e8409680157b0f950958c9" "BioASQ_Synergy9_v2/feedback_accompanying_round_4.json"

SHARE_BIOINFORMATICS_DOWNLOAD "BioASQ_Synergy10.zip" "5ecd70480ce349bfb587a61c4fab5fc2" "BioASQ_Synergy10/BioASQ-taskSynergy_v2022-testset4"

cd -

echo "Preparing the BioASQ synergy sterilizer cache"
cd synergy

SHARE_BIOINFORMATICS_DOWNLOAD "cache.zip" "c66caa635c2644d3be6ed48e83da1a5d" "cache/cord19_quest_2022-01-31_merge.cache"

cd -


### BIOASQ 10B



#### bioasq 10 dataset
cd taskb/data
SHARE_BIOINFORMATICS_DOWNLOAD "dataset.zip" "e699f2750f404ded95791791e4f5a7a3" "dataset/training10b.json"
cd -

#### Pubmed index
cd taskb/cache/indexes

SHARE_BIOINFORMATICS_DOWNLOAD "pubmed_all_index.zip" "8a451062f81d4f9f9bb15fb53d16f580" "pubmed/_7_Lucene80_0.dvd"

cd -

cd taskb/cache

SHARE_BIOINFORMATICS_DOWNLOAD "parade_model.zip" "173250350c0640939168ce7c6cc717d5" "saved_models/pleasant-deluge-1.h5"

cd -
