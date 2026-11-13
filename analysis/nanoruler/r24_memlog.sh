#!/bin/bash
# R24: log the GPU memory of all compute processes every 5 s while a marker file exists. usage: r24_memlog.sh <marker> <log>
while [ -f "$1" ]; do echo "$(date +%T) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr '\n' ';')" >> "$2"; sleep 5; done
