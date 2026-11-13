#!/bin/bash
# move a cache directory of out/nanoruler to /mnt/data/dmdsf/simcode2_out/nanoruler/<name> and leave a symlink (user rule 2026-11-13)
# usage: move_cache.sh <name>   (rsync at idle I/O priority, 60 MB/s; verified by file count and bytes before the source is removed)
set -e
S=/home/jc/simcode2/out/nanoruler/$1; D=/mnt/data/dmdsf/simcode2_out/nanoruler/$1
[ -L $S ] && { echo "$S is already a symlink"; exit 0; }
mkdir -p $D
ionice -c3 nice -n 19 rsync -a --bwlimit=60000 $S/ $D/
n1=$(find $S -type f | wc -l); n2=$(find $D -type f | wc -l); b1=$(du -sb --apparent-size $S | cut -f1); b2=$(du -sb --apparent-size $D | cut -f1)
f1=$(find $S -type f -printf '%s\n' | awk '{s+=$1} END {print s}'); f2=$(find $D -type f -printf '%s\n' | awk '{s+=$1} END {print s}')
echo "files $n1 / $n2, file bytes $f1 / $f2"
[ "$n1" = "$n2" ] && [ "$f1" = "$f2" ] || { echo "MISMATCH, source kept"; exit 1; }
rm -rf $S; ln -s $D $S; echo "moved $1 -> $D (symlink left) $(date)"
