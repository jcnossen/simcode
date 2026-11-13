#!/bin/bash
# Paper-simulation tables with the new main SIMCODE v2 row v2sig (sigma <= 12 nm): metrics of v2sig merged into the existing
# metrics jsons (--only v2sig), then the summary tables / figures. Old outputs are kept as *_u045main.* copies first.
set -e
cd "$(dirname "$0")/../.."
O=out/papersim
for f in $O/summary_density_D_m1809 $O/summary_density_Dexp_m1809 $O/summary_density_D_gauss_g13 $O/repro_table_g13; do
  for e in md json; do [ -f $f.$e ] && [ ! -f ${f}_u045main.$e ] && cp $f.$e ${f}_u045main.$e; done
done
for m in moving moving_exp moving_gauss; do ns=$([ $m = moving_gauss ] && echo g13 || echo m1809)
  for b in moving_table moving_matched; do for e in md json; do f=$O/$m/full/${b}_$ns; [ -f $f.$e ] && [ ! -f ${f}_u045main.$e ] && cp $f.$e ${f}_u045main.$e; done; done
done
for d in 0.5 1 2 4; do
  python analysis/papersim/metrics.py $O/D$d/full --movie $O/D$d --netset m1809 --only v2sig
  python analysis/papersim/metrics.py $O/Dexp$d/full --movie $O/Dexp$d --netset m1809 --only v2sig
  python analysis/papersim/metrics.py $O/D${d}_gauss/full --movie $O/D${d}_gauss --netset g13 --only v2sig
done
python analysis/papersim/fig_density.py --netset m1809 --prefix D
python analysis/papersim/fig_density.py --netset m1809 --prefix Dexp
python analysis/papersim/fig_density.py --netset g13 --prefix D --suffix _gauss
python analysis/papersim/fig_overview.py --prefix D --netset m1809
python analysis/papersim/fig_overview.py --prefix Dexp --netset m1809
python analysis/papersim/fig_overview.py --prefix D --suffix _gauss --netset g13
for m in moving moving_exp moving_gauss; do ns=$([ $m = moving_gauss ] && echo g13 || echo m1809)
  python analysis/papersim/metrics.py $O/$m/full --movie $O/$m --netset $ns --only v2sig
  python analysis/papersim/moving_fig.py $O/$m --netset $ns
  python analysis/papersim/moving_matched.py $O/$m --netset $ns
done
python analysis/papersim/repro_table.py --netset g13
echo PAPERSIM DONE
