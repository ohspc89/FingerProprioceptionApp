#!/bin/zsh

for f in *.json
do
	python3 -mjson.tool $f >> "${f%%.*}"_tidy.json
done
