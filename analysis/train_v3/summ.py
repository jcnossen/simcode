import json, sys
R=json.load(open(sys.argv[1]))
for n,r in R.items():
    print('==',n)
    for k,v in r.items():
        if isinstance(v,dict) and 'max_abs_over_peak' in v: print(f'  {k:36s} peak-norm {v["max_abs_over_peak"]:.2e}  rel(px>1e-3pk) {v["max_rel_px"]:.2e}')
        elif k=='E_decode': print('  E_decode', v['desc'], v['lut_shape'], v.get('module'), '\n     vs direct: peak-norm %.2e rel %.2e'%(v['vs_direct']['max_abs_over_peak'],v['vs_direct']['max_rel_px']), ' vs fit: peak-norm %.2e rel %.2e'%(v['vs_fit_mean']['max_abs_over_peak'],v['vs_fit_mean']['max_rel_px']))
        else: print('  ',k, json.dumps(v)[:200])
