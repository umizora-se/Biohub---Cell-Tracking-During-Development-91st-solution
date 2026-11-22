#!/bin/bash
# Download the competition data and the public weight datasets used by the base pipeline.
# Requires Kaggle API credentials (~/.config/kaggle/kaggle.json) and acceptance of the competition rules.
set -euo pipefail
cd "$(dirname "$0")/.."
KAGGLE=.venv/bin/kaggle

# competition data -> data/train/{name}.zarr, {name}.geff  (~83 GB unpacked)
if [ ! -d data/train ]; then
  mkdir -p data
  $KAGGLE competitions download -c biohub-cell-tracking-during-development -p data
  unzip -q data/biohub-cell-tracking-during-development.zip -d data && rm data/biohub-cell-tracking-during-development.zip
fi

# public datasets of the base notebook -> inputs/<slug>, linked into the local Kaggle mirror pipe/kaggle/input/datasets/<owner>/<slug>
for ref in pilkwang/biohub-deepcenter-unet3d-center-prior-v1 pilkwang/biohub-temporal-unet3d-seed314159-v1 \
           pilkwang/biohub-tracking-support-pack-50ep-v1 anvithpothula/biohub-v1284-head-s075; do
  owner=${ref%%/*}; slug=${ref##*/}
  if [ ! -d inputs/$slug ]; then
    $KAGGLE datasets download -d $ref -p inputs/$slug --unzip
  fi
  mkdir -p pipe/kaggle/input/datasets/$owner
  ln -sfn "$(pwd)/inputs/$slug" pipe/kaggle/input/datasets/$owner/$slug
done
echo "download done"
