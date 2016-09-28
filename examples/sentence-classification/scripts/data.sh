#!/bin/bash

if [ -z "$1" ]; then
  echo "need an argument (for directory to store the data)"
  exit 0
fi
dir=$1

mkdir -p "$dir"
pushd "$dir"

if [ -f train ]; then
    echo " data already processed"
    exit 0
fi

curl https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/stsa.fine.phrases.train > train
curl https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/stsa.fine.train > small-train
curl https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/stsa.fine.dev > valid
curl https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/stsa.fine.test > test

for file in "small-train" "train" "valid" "test"; do
    awk '{print $1}' < "$file" > "${file}.lbls"
    awk '{for (i=2; i<NF; i++) printf $i " "; print $NF}' < "$file" > "${file}.sens"
done

popd
