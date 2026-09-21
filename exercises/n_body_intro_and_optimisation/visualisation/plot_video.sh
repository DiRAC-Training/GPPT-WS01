#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

find -name "*.csv" -print0 | xargs -0 -t -I % -P $(nproc) $SCRIPT_DIR/plot.py %

ffmpeg -framerate 60 -y -i %04d.csv.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p out.mp4
