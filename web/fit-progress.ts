export function fitProgress(mode: 'slide' | 'gel') {
  const host = document.createElement('div');
  host.className = 'fit-progress';
  host.innerHTML = '<progress max="1" value="0" aria-label="Overall estimation progress"></progress><output></output>';
  const bar = host.querySelector('progress')!, caption = host.querySelector('output')!;
  const phaseNames = ['Reading movie','Calibrating optics',mode==='gel'?'Refining bead volume':'Refining bead positions','Fitting intensities','Writing results'];
  const key = 'simcode_fit_stage_ms_v1_'+mode;
  // Milliseconds on the local NVIDIA adapter: 64 x 64 pixels, 18 slide frames
  // or 51 gel focus stops. Successful runs replace these starting measurements.
  let weights = mode==='gel' ? [402,5422,126673,1629,1137] : [143,4421,955,1554,254];
  try {
    const saved = JSON.parse(localStorage.getItem(key) ?? 'null');
    if(Array.isArray(saved) && saved.length===5 && saved.every(x=>Number.isFinite(x)&&x>0)) weights=saved;
  } catch {}
  const durations=weights.map(()=>0);
  let stage=0, since=0, started=0, active=false, reported=0;
  let timer: number | undefined;
  function render() {
    const elapsed = performance.now()-since;
    // A timed estimate bridges expensive operations between measured work updates.
    const estimated = Math.min(.9, elapsed/weights[stage], stage===1 || stage===4 ? 1 : reported+.12);
    const fraction = Math.max(reported,estimated);
    const sum=weights.reduce((a,b)=>a+b,0);
    const value=(weights.slice(0,stage).reduce((a,b)=>a+b,0)+weights[stage]*fraction)/sum;
    bar.value=Math.max(bar.value,Math.min(.99,value));
    caption.value=phaseNames[stage]+' · '+Math.floor(bar.value*100)+'% · '+Math.floor((performance.now()-started)/1000)+' s';
  }
  function update(next:number, fraction=0) {
    if(!active || next<stage) return;
    if(next>stage) {
      const now=performance.now();
      durations[stage]+=now-since;
      since=now; stage=next; reported=0;
    }
    reported = Math.max(reported, Math.max(0,Math.min(1,fraction)));
    render();
  }
  function finish() {
    if(!active)return;
    durations[stage]+=performance.now()-since;
    window.clearInterval(timer); active=false; bar.value=1; caption.value='Complete · 100%';
    const measured=durations.map(v=>Math.max(1,v));
    if(durations.every(v=>v>0)) {
      weights=measured;
      try { localStorage.setItem(key,JSON.stringify(measured)); } catch {}
    }
  }
  return {host,update,finish,
    start(){host.hidden=false;active=true; stage=0; reported=0; started=since=performance.now();window.clearInterval(timer);timer=window.setInterval(render,500);durations.fill(0);bar.value=0;update(0);},
    stop(){window.clearInterval(timer);if(active)caption.value='Stopped · '+Math.floor(bar.value*100)+'%';active=false;},
  };
}
