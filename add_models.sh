#/usr/bin/bash

#read -p "Substring of models to add: " fullname
substring=$1
git add -f data/models/*$substring*/*.png
git add -f data/models/*$substring*/*_final.weights.h5
git add -f data/models/*$substring*/*_config.json
git add -f data/models/*$substring*/*_prog.csv
git add -f data/models/*$substring*/*_summary.txt
