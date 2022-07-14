#!/bin/bash

set -e

# prepare cord-19
echo "Preparing the cord-19 corpus"
mkdir -p data/cord-19

URL_BASE_CORD="https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/"

CORD_RND1_GZ_FILE="cord-19_2021-11-15.tar.gz"
URL_CORD_RND1_GZ_FILE=$URL_BASE_CORD$CORD_RND1_GZ_FILE

CORD_RND2_GZ_FILE="cord-19_2021-12-20.tar.gz"
URL_CORD_RND2_GZ_FILE=$URL_BASE_CORD$CORD_RND2_GZ_FILE

CORD_RND3_GZ_FILE="cord-19_2022-01-03.tar.gz"
URL_CORD_RND3_GZ_FILE=$URL_BASE_CORD$CORD_RND3_GZ_FILE

CORD_RND4_GZ_FILE="cord-19_2022-01-31.tar.gz"
URL_CORD_RND4_GZ_FILE=$URL_BASE_CORD$CORD_RND4_GZ_FILE

cd data/cord-19

echo "Download and untar $CORD_RND1_GZ_FILE"
wget -c $URL_CORD_RND1_GZ_FILE
tar -xf $CORD_RND1_GZ_FILE
rm "2021-11-15/changelog"
rm "2021-11-15/cord_19_embeddings.tar.gz"
rm "2021-11-15/document_parses.tar.gz"

echo "Download and untar $CORD_RND2_GZ_FILE"
wget -c $URL_CORD_RND2_GZ_FILE
tar -xf $CORD_RND2_GZ_FILE
rm "2021-12-20/changelog"
rm "2021-12-20/cord_19_embeddings.tar.gz"
rm "2021-12-20/document_parses.tar.gz"

echo "Download and untar $CORD_RND3_GZ_FILE"
wget -c $URL_CORD_RND3_GZ_FILE
tar -xf $CORD_RND3_GZ_FILE
rm "2022-01-03/changelog"
rm "2022-01-03/cord_19_embeddings.tar.gz"
rm "2022-01-03/document_parses.tar.gz"

echo "Download and untar $CORD_RND4_GZ_FILE"
wget -c $URL_CORD_RND4_GZ_FILE
tar -xf $CORD_RND4_GZ_FILE
rm "2022-01-31/changelog"
rm "2022-01-31/cord_19_embeddings.tar.gz"
rm "2022-01-31/document_parses.tar.gz"

cd -

echo "Preparing the BioASQ synergy dataset"

