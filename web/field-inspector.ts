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
      ...field.ratios.map((_,i) => ['ratio_A_over_'+axisName(i+1), 'Intensity A / '+axisName(i+1)]),
    ];
    const other = field.patterns.flatMap((_,i) => {
      const a=axisName(i);
      return [['orientation_'+a+'_deg','Angle '+a+' (°)'], ['phase_offset_'+a+'_rad','Phase offset '+a+' (rad)'], ['phase_'+a+'_rad','Phase '+a+' (rad)']];
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
        fitted.textContent=fit?.maps[i][key]?.toFixed(4) ?? '—';
        if (actual) actual.textContent=truth?.maps[i][key]?.toFixed(4) ?? '—';
      }
      for(const marker of markers) {
        marker.style.left=(x+.5)*100/32+'%';
        marker.style.top=(y+.5)*100/32+'%';
      }
    }
    for(const [key,label] of fields) {
      const fig=document.createElement('figure'), title=document.createElement('figcaption');
      title.textContent=(fit?'Fitted ':'True ')+label;
      const tile=document.createElement('div'); tile.className='field-tile';
      const cv=document.createElement('canvas'); cv.width=cv.height=32; cv.tabIndex=0;
      cv.setAttribute('aria-label',title.textContent+'. Drag or use arrow keys to inspect.');
      const ctx=cv.getContext('2d')!, values=field.maps.map(v=>v[key]);
      const lo=Math.min(...values), hi=Math.max(...values);
      values.forEach((v,i)=>{
        const f=hi>lo?(v-lo)/(hi-lo):.5;
        ctx.fillStyle='rgb('+[26+209*f,58+150*f,100-38*f].map(Math.round).join(',')+')';
        ctx.fillRect(i%32,Math.floor(i/32),1,1);
      });
      const marker=document.createElement('i'); markers.push(marker);
      const range=document.createElement('small'); range.textContent=lo.toFixed(3)+' – '+hi.toFixed(3);
      tile.append(cv,marker); fig.append(title,tile,range); tiles.append(fig);
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
    }
    select(x,y);
  };
}
