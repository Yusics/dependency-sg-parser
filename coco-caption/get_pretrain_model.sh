#!/usr/bin/env sh
# This script downloads the Stanford CoreNLP models.

PRETRAIN=SG_weights
SPICELIB=pycocoevalcap/spice

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

git clone https://gitlab.com/Yusics/SG_weights.git 

mv $PRETRAIN/barchybrid.model4_1e-2_256_200.tmp $SPICELIB/
mv $PRETRAIN/params.pickle $SPICELIB/
rm -rf $SPICELIB/$PRETRAIN/

echo "Done."
