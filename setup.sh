#!/bin/bash

set -e

# prepare cord-19
echo "Preparing the cord-19 corpus"
mkdir -p synergy/data/cord-19

URL_BASE_CORD="https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/"

CORD_RND1_GZ_FILE="cord-19_2021-11-15.tar.gz"
URL_CORD_RND1_GZ_FILE=$URL_BASE_CORD$CORD_RND1_GZ_FILE

CORD_RND2_GZ_FILE="cord-19_2021-12-20.tar.gz"
URL_CORD_RND2_GZ_FILE=$URL_BASE_CORD$CORD_RND2_GZ_FILE

CORD_RND3_GZ_FILE="cord-19_2022-01-03.tar.gz"
URL_CORD_RND3_GZ_FILE=$URL_BASE_CORD$CORD_RND3_GZ_FILE

CORD_RND4_GZ_FILE="cord-19_2022-01-31.tar.gz"
URL_CORD_RND4_GZ_FILE=$URL_BASE_CORD$CORD_RND4_GZ_FILE

cd synergy/data/cord-19

if [ ! -f "2021-11-15/metadata.csv" ]; then
    echo "Download and untar $CORD_RND1_GZ_FILE"
    wget -c $URL_CORD_RND1_GZ_FILE
    tar -xf $CORD_RND1_GZ_FILE
    rm "2021-11-15/changelog"
    rm "2021-11-15/cord_19_embeddings.tar.gz"
    rm "2021-11-15/document_parses.tar.gz"
    rm $CORD_RND1_GZ_FILE
fi

if [ ! -f "2021-12-20/metadata.csv" ]; then
    echo "Download and untar $CORD_RND2_GZ_FILE"
    wget -c $URL_CORD_RND2_GZ_FILE
    tar -xf $CORD_RND2_GZ_FILE
    rm "2021-12-20/changelog"
    rm "2021-12-20/cord_19_embeddings.tar.gz"
    rm "2021-12-20/document_parses.tar.gz"
    rm $CORD_RND2_GZ_FILE
fi

if [ ! -f "2022-01-03/metadata.csv" ]; then
    echo "Download and untar $CORD_RND3_GZ_FILE"
    wget -c $URL_CORD_RND3_GZ_FILE
    tar -xf $CORD_RND3_GZ_FILE
    rm "2022-01-03/changelog"
    rm "2022-01-03/cord_19_embeddings.tar.gz"
    rm "2022-01-03/document_parses.tar.gz"
    rm $CORD_RND3_GZ_FILE
fi

if [ ! -f "2022-01-31/metadata.csv" ]; then
    echo "Download and untar $CORD_RND4_GZ_FILE"
    wget -c $URL_CORD_RND4_GZ_FILE
    tar -xf $CORD_RND4_GZ_FILE
    rm "2022-01-31/changelog"
    rm "2022-01-31/document_parses.tar.gz"
    rm $CORD_RND4_GZ_FILE
fi

cd -

echo "Preparing the BioASQ synergy dataset"
mkdir -p synergy/data/dataset

cd synergy/data/dataset

DATA_SYNERGY9B_V1_ZIP_FILE="BioASQ_Synergy9_v1.zip"
URL_DATA_SYNERGY9B_V1_ZIP_FILE="https://share.bioinformatics-ua.pt/share.cgi?ssid=6a56d4b1f2ea4630a441c8c241bc6073&fid=6a56d4b1f2ea4630a441c8c241bc6073&filename=$DATA_SYNERGY9B_V1_ZIP_FILE&openfolder=forcedownload&ep="

if [ ! -f "BioASQ_Synergy9_v1/feedback_final.json" ]; then
    echo "Download and unzip $DATA_SYNERGY9B_V1_ZIP_FILE"
    wget -c -O $DATA_SYNERGY9B_V1_ZIP_FILE $URL_DATA_SYNERGY9B_V1_ZIP_FILE
    unzip -u $DATA_SYNERGY9B_V1_ZIP_FILE
    rm $DATA_SYNERGY9B_V1_ZIP_FILE
fi


DATA_SYNERGY9B_V2_ZIP_FILE="BioASQ_Synergy9_v2.zip"
URL_DATA_SYNERGY9B_V2_ZIP_FILE="https://share.bioinformatics-ua.pt/share.cgi?ssid=476777f538e8409680157b0f950958c9&fid=476777f538e8409680157b0f950958c9&filename=$DATA_SYNERGY9B_V2_ZIP_FILE&openfolder=forcedownload&ep="

if [ ! -f "BioASQ_Synergy9_v2/feedback_accompanying_round_4.json" ]; then
    echo "Download and unzip $DATA_SYNERGY9B_V2_ZIP_FILE"
    wget -c -O $DATA_SYNERGY9B_V2_ZIP_FILE $URL_DATA_SYNERGY9B_V2_ZIP_FILE
    unzip -u $DATA_SYNERGY9B_V2_ZIP_FILE
    rm $DATA_SYNERGY9B_V2_ZIP_FILE
fi

DATA_SYNERGY10_ZIP_FILE="BioASQ_Synergy10.zip"
URL_DATA_SYNERGY10_ZIP_FILE="https://share.bioinformatics-ua.pt/share.cgi?ssid=5ecd70480ce349bfb587a61c4fab5fc2&fid=5ecd70480ce349bfb587a61c4fab5fc2&filename=$DATA_SYNERGY10_ZIP_FILE&openfolder=forcedownload&ep="

if [ ! -f "BioASQ_Synergy10/BioASQ-taskSynergy_v2022-testset4" ]; then
    echo "Download and unzip $DATA_SYNERGY10_ZIP_FILE"
    wget -c -O $DATA_SYNERGY10_ZIP_FILE $URL_DATA_SYNERGY10_ZIP_FILE
    unzip -u $DATA_SYNERGY10_ZIP_FILE
    rm $DATA_SYNERGY10_ZIP_FILE
fi
cd -

echo "Preparing the BioASQ synergy sterilizer cache"
cd synergy

DATA_STERILIZER_ZIP_FILE="cache.zip"
URL_DATA_STERILIZER_ZIP_FILE="https://share.bioinformatics-ua.pt/share.cgi?ssid=c66caa635c2644d3be6ed48e83da1a5d&fid=c66caa635c2644d3be6ed48e83da1a5d&filename=$DATA_STERILIZER_ZIP_FILE&openfolder=forcedownload&ep="

if [ ! -f "cache/cord19_quest_2022-01-31_merge.cache" ]; then
    echo "Download and unzip $DATA_STERILIZER_ZIP_FILE"
    wget -c -O $DATA_STERILIZER_ZIP_FILE $URL_DATA_STERILIZER_ZIP_FILE
    unzip -u $DATA_STERILIZER_ZIP_FILE
    rm $DATA_STERILIZER_ZIP_FILE
fi

cd -


### BIOASQ 10B
