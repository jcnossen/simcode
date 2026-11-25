import { fitMovie, predictMovieFrame, type MovieFit, type Settings } from 'torch-dense/movie';
import {refineRawPupil} from 'torch-dense/raw-pupil';
import { getDevice } from 'torch-web/gpu/device';
import { TiffMovie } from 'torch-web/io/tiff';
import { tiffStackBlob } from 'torch-web/psfgrab/tiffstack';

import {refinePhotometry} from './pixel-photometry';
import { modulation, refinedModulation } from './modulation';
import { axisAt, phaseAt, type Schedule } from './schedule';

type Pattern = ReturnType<typeof modulation>;
function field(c:number[],u:number,v:number){
  return c[0]+c[1]*u+c[2]*v+c[3]*u*u+c[4]*u*v+c[5]*v*v;
}
function illumination(fit:MovieFit,pattern:Pattern,order:Schedule,j:number,t:number,drift:number[]){
  const [x,y]=fit.xy[j], u=2*x/(fit.width-1)-1,v=2*y/(fit.height-1)-1;
  const axis=axisAt(t,order), p=pattern.patterns[axis];
  const ratio=axis ? Math.max(.01,field(pattern.ratios[axis-1],u,v)) : 1;
  const angle=p.kx_rad_px*(x+drift[0])+p.ky_rad_px*(y+drift[1])+field(p.phase,u,v)+p.phase_steps_rad[phaseAt(t,order)];
  const q=Math.hypot(p.kx_rad_px,p.ky_rad_px)*fit.settings.diameterNm/(2000*fit.settings.optics.pixel_um);
  const attenuation=q<.001 ? 1 : 3*(Math.sin(q)-q*Math.cos(q))/q**3;
  return Math.max(.001,(1+attenuation*field(p.depth,u,v)*Math.cos(angle))/ratio);
}
function deviance(observed:Float32Array,estimated:Float32Array,settings:Settings){
  let total=0;
  for(let i=0;i<observed.length;i++){
    const y=Math.max(0,(observed[i]-settings.offset)*settings.gain), mu=Math.max(.001,estimated[i]);
    total+=2*(mu-y+(y>0?y*Math.log(y/mu):0));
  }
  return total/observed.length;
}
// Each fold drops a whole interior cycle before any pupil, drift, catalog, depth,
// phase or brightness fit. Adjacent retained frames predict the missing motion.
export async function validateMovie(file:File,settings:Settings,order:Schedule,log:(s:string)=>void){
  const {device}=await getDevice(), movie=await TiffMovie.open([file]);
  const period=order.axes*order.phases, cycles=Math.floor(movie.n/period);
  if(cycles<4)throw Error('Cross-validation needs at least four complete illumination cycles.');
  const held=[Math.max(1,Math.floor(cycles/3)),Math.min(cycles-2,Math.ceil(2*cycles/3)-1)];
  const folds=[];
  for(const cycle of new Set(held)){
    const start=cycle*period, end=start+period, train:Float32Array[]=[];
    log('Fitting training frames with cycle '+(cycle+1)+' held out…');
    for(let t=0;t<movie.n;t++) if(t<start || t>=end)train.push(await movie.read(t,t+1));
    const training=new File([tiffStackBlob(train,train.length,1,movie.H,movie.W)],'training-fold.tif');
    train.length=0;
    let fit=await fitMovie(device,training,null,settings,()=>{});
    let pattern:Pattern|undefined, patternError='';
    if(order.phases>=3)try{
      fit=await refinePhotometry(device,training,fit,log);
      pattern=refinedModulation(fit,order);
    }catch(e){patternError=String(e);}
    const nb=fit.xy.length, counts=Array(period).fill(0);
    const phaseFlux=Array.from({length:period},()=>Array<number>(nb).fill(0));
    const mean=Array<number>(nb).fill(0),num=mean.slice(),den=mean.slice();
    fit.intensity.forEach((row,t)=>{
      const slot=t%period; counts[slot]++;
      for(let j=0;j<nb;j++){
        phaseFlux[slot][j]+=row[j]; mean[j]+=row[j]/fit.intensity.length;
        if(pattern){
          const factor=illumination(fit,pattern,order,j,t,fit.drift[t]);
          num[j]+=factor*row[j]; den[j]+=factor*factor;
        }
      }
    });
    phaseFlux.forEach((row,k)=>row.forEach((v,j)=>{row[j]=v/counts[k];}));
    const flat={...fit,model:{...fit.model,zcoef:fit.model.noll.map(()=>0)}};
    log('Refining a second pupil on raw training patches…');
    const candidate={...fit,model:{...fit.model,zcoef:await refineRawPupil(device,training,fit)}};
    const scores={rawPupil:0,psf:0,flatPupil:0,pattern:0,constantFlux:0};
    log('Predicting withheld pixels in cycle '+(cycle+1)+'…');
    for(let slot=0;slot<period;slot++){
      const alpha=(slot+1)/(period+1);
      const interp=(a:number,b:number)=>a+(b-a)*alpha;
      const drift=fit.drift[start-1].map((v,j)=>interp(v,fit.drift[start][j]));
      const bg=interp(fit.background[start-1],fit.background[start]);
      const flux=[phaseFlux[slot]];
      if(pattern)flux.push(num.map((v,j)=>v/Math.max(1e-12,den[j])*illumination(fit,pattern,order,j,start+slot,drift)),mean);
      const models=await predictMovieFrame(device,fit,drift,flux,bg);
      const [baseline]=await predictMovieFrame(device,flat,drift,[phaseFlux[slot]],bg);
      const pixels=await movie.read(start+slot,start+slot+1);
      const [raw]=await predictMovieFrame(device,candidate,drift,[phaseFlux[slot]],bg);
      scores.rawPupil+=deviance(pixels,raw,settings)/period;
      scores.psf+=deviance(pixels,models[0],settings)/period;
      scores.flatPupil+=deviance(pixels,baseline,settings)/period;
      if(pattern){
        scores.pattern+=deviance(pixels,models[1],settings)/period;
        scores.constantFlux+=deviance(pixels,models[2],settings)/period;
      }
    }
    folds.push({cycle:cycle+1,trainingFrames:fit.drift.length,testFrames:period,psf:scores.psf,rawPupil:scores.rawPupil,rawCoefficientsWaves:candidate.model.zcoef,flatPupil:scores.flatPupil,
      pattern:pattern?scores.pattern:null,constantFlux:pattern?scores.constantFlux:null,patternError,
      diagnostics:fit.pupilDiagnostics,noll:fit.model.noll,coefficientsWaves:Array.from(fit.model.zcoef),patterns:pattern?.patterns});
  }
  return folds;
}

export function validationPanel(out:HTMLElement,file:File,settings:Settings,order:Schedule){
  const box=document.createElement('section'),button=document.createElement('button'),status=document.createElement('p');
  button.type='button';button.textContent='Cross-validate PSF and modulation';
  box.append(button,status);out.append(box);
  button.onclick=async()=>{
    button.disabled=true;
    try{
      const folds=await validateMovie(file,settings,order,s=>{status.textContent=s;});
      const table=document.createElement('table');
      table.innerHTML='<tr><th>Held-out cycle</th><th>Fitted pupil</th><th>Raw-pixel refinement</th><th>Flat pupil</th><th>Fitted pattern</th><th>Constant brightness</th></tr>';
      for(const f of folds){
        const row=table.insertRow();
        for(const v of [String(f.cycle),f.psf.toFixed(3),f.rawPupil.toFixed(3),f.flatPupil.toFixed(3),f.pattern?.toFixed(3)??'—',f.constantFlux?.toFixed(3)??'—'])row.insertCell().textContent=v;
      }
      status.textContent='Held-out Poisson deviance per pixel (lower is better). Each row refits the other cycles. Drift is interpolated across the omitted cycle. '+folds.map(f=>f.patternError).filter(Boolean).join(' ');
      box.append(table);
      const a=document.createElement('a');a.download='cross-validation.json';a.textContent=a.download;
      a.href=URL.createObjectURL(new Blob([JSON.stringify(folds)],{type:'application/json'}));box.append(a);
    }catch(e){status.textContent=String(e);button.disabled=false;}
  };
}
