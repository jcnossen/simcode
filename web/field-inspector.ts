import { carrierSpectra } from './modulation';
import { axisName } from './schedule';
import type { PatternField } from './pattern-field';

type Field = PatternField & { maps: Record<string, number>[] };

export function fieldInspector(host: HTMLElement) {
  let x = 16, y = 16;
  return (fit?: Field, truth?: Field) => {
    const field = fit ?? truth;
    host.hidden = !field;
    if (!field) return;
    host.innerHTML = '<h3>Modulation across the field</h3><div class="field-layout"><div class="field-tiles"></div><div><p class="field-position"></p><table class="field-values"></table></div></div>';
    const tiles = host.querySelector<HTMLElement>('.field-tiles')!;
    const table = host.querySelector<HTMLTableElement>('.field-values')!;
    const fields = [
      ...field.patterns.map((_,i) => ['depth_'+axisName(i), 'Depth '+axisName(i)]),
      ...field.ratios.map((_,i) => ['ratio_A_over_'+axisName(i+1), 'Intensity '+axisName(i+1)+' / A']),
    ];
    function read(row:Record<string,number> | undefined,key:string) {
      const v=row?.[key];
      return key.startsWith('ratio_') ? (v!==undefined && v>0 ? 1/v : NaN) : v;
    }
    const other = field.patterns.flatMap((_,i) => {
      const a=axisName(i);
      return [['pitch_'+a+'_nm','Pitch '+a+' (nm)'],...(field.patterns[i].kz_cycles_um===undefined ? [] : [['kz_'+a,'Axial frequency '+a+' (cycles/µm)']]),['orientation_'+a+'_deg','Angle '+a+' (°)'], ['phase_offset_'+a+'_rad','Phase offset '+a+' (rad)'], ['phase_'+a+'_rad','Phase '+a+' (rad)']];
    });
    const header=table.createTHead().insertRow();
    for (const text of ['Parameter','Fitted',...(truth?['True']:[])]) {
      const th=document.createElement('th'); th.textContent=text; header.append(th);
    }
    const body=table.createTBody();
    const cells=[...fields,...other].map(([key,label])=>{
      const row=body.insertRow(); row.insertCell().textContent=label;
      return {key, fitted:row.insertCell(), actual: truth ? row.insertCell() : undefined};
    });
    const markers:HTMLElement[]=[];
    function select(col:number,row:number) {
      x=Math.max(0,Math.min(31,col)); y=Math.max(0,Math.min(31,row));
      const i=y*32+x, position=field!.maps[i];
      host.querySelector('.field-position')!.textContent='Camera ('+position.x_px.toFixed(1)+', '+position.y_px.toFixed(1)+') px';
      for (const {key,fitted,actual} of cells) {
        const value=(v?:number)=>v!==undefined && Number.isFinite(v) ? v.toFixed(4) : '—';
        fitted.textContent=value(read(fit?.maps[i],key));
        if (actual) actual.textContent=value(read(truth?.maps[i],key));
      }
      for(const marker of markers) {
        marker.style.left=(x+.5)*100/32+'%';
        marker.style.top=(y+.5)*100/32+'%';
      }
    }
    function fieldTile([key,label]:string[]) {
      const fig=document.createElement('figure'), title=document.createElement('figcaption');
      title.textContent=(fit?'Fitted ':'True ')+label;
      const tile=document.createElement('div'); tile.className='field-tile';
      const cv=document.createElement('canvas'); cv.width=cv.height=32; cv.tabIndex=0;
      cv.setAttribute('aria-label',title.textContent+'. Drag or use arrow keys to inspect.');
      const ctx=cv.getContext('2d')!, values=field!.maps.map(v=>read(v,key) ?? NaN);
      const finite=values.filter(Number.isFinite), lo=Math.min(...finite), hi=Math.max(...finite);
      values.forEach((v,i)=>{
        const f=hi>lo?(v-lo)/(hi-lo):.5;
        ctx.fillStyle='rgb('+[26+209*f,58+150*f,100-38*f].map(Math.round).join(',')+')';
        if (!Number.isFinite(v)) ctx.fillStyle="#888";
        ctx.fillRect(i%32,Math.floor(i/32),1,1);
      });
      const marker=document.createElement('i'); markers.push(marker);
      const range=document.createElement('small'); range.textContent=Number.isFinite(lo) && Number.isFinite(hi) ? lo.toFixed(3)+' – '+hi.toFixed(3) : 'Unresolved';
      tile.append(cv,marker); fig.append(title,tile,range);
      function point(e:PointerEvent) {
        const box=cv.getBoundingClientRect();
        select(Math.floor((e.clientX-box.left)*32/box.width),Math.floor((e.clientY-box.top)*32/box.height));
      }
      cv.onpointerdown=e=>{cv.setPointerCapture(e.pointerId);point(e);};
      cv.onpointermove=point;
      cv.onkeydown=e=>{
        const step:Record<string,number[]>={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]};
        if(step[e.key]) { e.preventDefault(); select(x+step[e.key][0],y+step[e.key][1]); }
      };
      return fig;
    }
    function fftTile(axis:number, zoom:boolean) {
      const fig=document.createElement('figure'), title=document.createElement('figcaption');
      title.textContent=zoom?'FFT peak zoom':'FFT peaks';
      fig.append(title);
      const spectrum=fit && carrierSpectra.get(fit.patterns[axis]);
      if (!spectrum) {fig.append('Awaiting spectrum');return fig;}
      const values=zoom?spectrum.zoom:spectrum.overview, side=Math.sqrt(values.length);
      const cv=document.createElement('canvas'); cv.width=cv.height=side;
      const ctx=cv.getContext('2d')!, image=ctx.createImageData(side,side);
      const peak=Math.max(...values);
      for (let i=0;i<values.length;i++) {
        const gray=Math.round(255*Math.log1p(20*values[i]/(peak||1))/Math.log(21));
        image.data.set([gray,gray,gray,255],4*i);
      }
      ctx.putImageData(image,0,0);
      if (!zoom) {
        ctx.strokeStyle='#ff9d46';
        ctx.strokeRect((spectrum.center[0]+1)*side/2-3,(spectrum.center[1]+1)*side/2-3,6,6);
      }
      const note=document.createElement('small');
      note.textContent=zoom?'Peak ('+spectrum.center.map(v=>v.toFixed(3)).join(', ')+') cycles/px':'-1 to +1 cycles/px';
      fig.append(cv,note);return fig;
    }
    field.patterns.forEach((_,i)=>{
      const row=document.createElement('div'); row.className='axis-field-row';
      row.append(fieldTile(fields[i]),fftTile(i,false),fftTile(i,true));
      if(i) row.append(fieldTile(fields[field.patterns.length+i-1]));
      tiles.append(row);
    });
    select(x,y);
  };
}
