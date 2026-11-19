#!/bin/bash
# GPU rebalancing: when the m1809 chain is done (GPU 2 free but for its eval), move the m25 chain from GPU 1 (shared with DECODE
# m25) to GPU 2 at the next emitter-chain stage boundary after its detector chain is complete: kill the chain process group right
# after the stage start is logged, restart chain.sh m25 2 (finished stages are skipped; the killed stage restarts from scratch,
# a few seconds lost).
T=/home/jc/simcode2/out/train_v3
until grep -q "m1809: chain done" $T/stages.log; do sleep 60; done
until [ -f $T/runs/m25/detector_pass2_cont/final.pt ]; do grep -q "m25: chain done" $T/stages.log && exit 0; sleep 60; done
n0=$(grep -c "m25: .* start on GPU 1" $T/stages.log)
while true; do
  grep -q "m25: chain done" $T/stages.log && { echo "$(date) m25 chain already done"; exit 0; }
  n=$(grep -c "m25: .* start on GPU 1" $T/stages.log)
  [ $n -gt $n0 ] && break
  sleep 10
done
st=$(grep "m25: .* start on GPU 1" $T/stages.log | tail -1)
pg=$(ps -eo pgid,args | grep "bash chain.sh m25 1" | grep -v grep | awk '{print $1}' | head -1)
echo "$(date) moving m25 at: $st (pgid $pg)"
[ -n "$pg" ] && kill -TERM -- -$pg; sleep 20
echo "$(date +%F_%T) m25: chain moved to GPU 2 (killed at: ${st#* m25: })" >> $T/stages.log
cd $T && setsid nohup bash chain.sh m25 2 >> logs/chain_m25.log 2>&1 < /dev/null &
echo "$(date) restarted on GPU 2"
