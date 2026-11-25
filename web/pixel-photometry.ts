import {VecEngine} from 'torch-web/gpu/ops';
import {TiffMovie} from 'torch-web/io/tiff';
import type {MovieFit} from 'torch-dense/movie';
import {FourierDevice} from './dense-fourier';

// The optical bank has the same 33 px normalization and 27 px footprint as Python.
const side=27, radius=13, area=side*side;
async function opticalBank(engine:VecEngine,fit:MovieFit,focusNm:number){
  const r=fit.settings.diameterNm/2000, d=r*Math.sqrt(.6), px=fit.settings.optics.pixel_um;
  const nodes=[[d/px,0,0],[-d/px,0,0],[0,d/px,0],[0,-d/px,0],[0,0,d],[0,0,-d]];
  const params=[];
  for(let y=0;y<7;y++)for(let x=0;x<7;x++)for(const [dx,dy,dz] of nodes)
    params.push({x:x/6-.5+dx,y:y/6-.5+dy,ze:r+dz,f:r+focusNm/1000,I:1,bg:0});
  const bank=new Float32Array(49*area);
  for(let i=0;i<params.length;i+=48){
    const batch=params.slice(i,i+48);
    engine.setImages(batch);await engine.run(batch.length,'psf');
    const data=await engine.readPsf(batch.length);
    for(let j=0;j<batch.length;j++)for(let y=0;y<side;y++)for(let x=0;x<side;x++)
      bank[Math.floor((i+j)/6)*area+y*side+x]+=data[j*1089+(y+3)*33+x+3]/6;
  }
  return bank;
}

const bind=(i:number,name:string,type:string,write=false)=>
  '@group(0) @binding('+i+') var<storage,'+(write?'read_write':'read')+'> '+name+':array<'+type+'>;\n';
const thread='@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){let i=id.x;';
const stamps=bind(0,'bank','f32')+bind(1,'geom','vec4f')+bind(2,'K','f32',true)+thread+`
  let bead=i/64u;let g=geom[bead];for(var j=i%64u;j<729u;j+=64u){let c=clamp((g.zw+.5)*6.,vec2f(0.),vec2f(6.));
  let lo=min(vec2u(c),vec2u(5u));let d=c-vec2f(lo);let s=(lo.y*7u+lo.x)*729u+j;
  K[bead*729u+j]=mix(mix(bank[s],bank[s+729u],d.x),mix(bank[s+5103u],bank[s+5832u],d.x),d.y);}}`;

function projection(w:number,h:number,direction:boolean,robust=false){
  const cols=Math.ceil(w/16);
  return bind(0,'image','f32')+bind(1,'tiles','u32')+bind(2,'geom','vec4f')+bind(3,'K','f32')+
    bind(4,'beads','vec2f')+bind(5,'state','f32')+bind(6,'pixel','vec4f',true)+thread+`
    if(i>=arrayLength(&image)){return;}
    let pos=vec2i(vec2u(i%$W,i/$W));let tile=u32(pos.y/16)*$C+u32(pos.x/16);
    var mu=state[$MODE];
    for(var j=tiles[tile];j<tiles[tile+1u];j++){
      let b=tiles[j];let d=pos-vec2i(geom[b].xy)+13;
      if(all(d>=vec2i(0))&&all(d<vec2i(27))){mu+=K[b*729u+u32(d.y*27+d.x)]*beads[b][$MODE];}
    }
    $BODY
  }`.replaceAll('$W',w+'u').replaceAll('$C',cols+'u').replaceAll('$MODE',direction?'1':'0')
    .replace('$BODY',direction?'pixel[i].z=mu;':`mu=max(1.,mu);let res=image[i]-mu;
      var wt=1./(mu+4.);${robust?'wt*=min(1.,4.*sqrt(mu+4.)/max(1.,abs(res)));':''}
      pixel[i]=vec4f(res,wt,0.,0.);`);
}

function beadStats(w:number,h:number,geometry:boolean){
  return bind(0,'geom','vec4f')+bind(1,'K','f32')+bind(2,'pixel','vec4f')+
    bind(3,'beads','vec2f',true)+bind(4,'normal','f32',true)+`
  var<workgroup> s:array<vec4f,64>;var<workgroup> q:array<f32,64>;
  @compute @workgroup_size(64) fn main(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
    let b=group.x;var sum=vec4f(0.);var yy=0.;
    for(var j=lane;j<729u;j+=64u){
      let x=j%27u;let y=j/27u;let pos=vec2i(geom[b].xy)+vec2i(vec2u(x,y))-13;
      if(any(pos<vec2i(0))||pos.x>=${w}||pos.y>=${h}){continue;}
      let p=pixel[u32(pos.y)*${w}u+u32(pos.x)];let i=729u*b+j;let k=K[i];
      ${geometry?`let l=select(1u,0u,x==0u);let r=select(1u,0u,x==26u);
        let u=select(27u,0u,y==0u);let d=select(27u,0u,y==26u);
        let gx=(K[i-l]-K[i+r])*beads[b].x/f32(l+r);
        let gy=(K[i-u]-K[i+d])*beads[b].x/f32((d+u)/27u);
        sum+=p.y*vec4f(gx*p.x,gy*p.x,gx*gx,gx*gy);yy+=p.y*gy*gy;`:
        'sum+=vec4f(k*p.y*p.x,k*k*p.y,0.,0.);'}
    }
    s[lane]=sum;q[lane]=yy;workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){s[lane]+=s[lane+stride];q[lane]+=q[lane+stride];}workgroupBarrier();}
    if(lane==0u){${geometry?`normal[b*5u]=s[0].x;normal[b*5u+1u]=s[0].y;normal[b*5u+2u]=s[0].z;
      normal[b*5u+3u]=s[0].w;normal[b*5u+4u]=q[0];`:'beads[b].y=max(1.-beads[b].x,s[0].x/max(s[0].y,1e-8));'}}
  }`;
}

function partial(direction:boolean){
  return bind(0,'pixel','vec4f')+bind(1,'part','vec2f',true)+thread+`
    if(i>=arrayLength(&part)){return;}var acc=vec2f(0.);
    for(var j=i*256u;j<min(arrayLength(&pixel),(i+1u)*256u);j++){
      let p=pixel[j];acc+=${direction?'vec2f(p.y*p.x*p.z,p.y*p.z*p.z)':'vec2f(p.y*p.x,p.y)'};}
    part[i]=acc;}`;
}
function total(direction:boolean){
  return bind(0,'part','vec2f')+bind(1,'state','f32',true)+thread+`
    if(i!=0u){return;}var acc=vec2f(0.);
    for(var j=0u;j<arrayLength(&part);j++){acc+=part[j];}
    ${direction?'state[2]=clamp(acc.x/max(acc.y,1e-8),0.,.7);state[0]=max(0.,state[0]+state[2]*state[1]);':'state[1]=acc.x/max(1e-12,acc.y);'}}`;
}
const zeroBg=bind(0,'state','f32',true)+thread+'if(i==0u){state[1]=0.;}}';
const update=bind(0,'beads','vec2f',true)+bind(1,'state','f32')+thread+
  'if(i<arrayLength(&beads)){beads[i].x+=state[2]*beads[i].y;}}';

async function exposure(gpu:FourierDevice,bank:Float32Array,image:Float32Array,xy:number[][],motion:number[],initial:number[],w:number,h:number,needResidual:boolean){
  const n=xy.length,cols=Math.ceil(w/16),rows=Math.ceil(h/16);
  const lists=Array.from({length:cols*rows},()=>[] as number[]);
  const geom=new Float32Array(n*4),beads=new Float32Array(n*2);
  xy.forEach(([x0,y0],i)=>{
    const x=x0+motion[0],y=y0+motion[1],ax=Math.round(x),ay=Math.round(y);
    geom.set([ax,ay,x-ax,y-ay],4*i);beads[2*i]=Math.max(1,initial[i]);
    for(let v=Math.max(0,(ay-radius)>>4);v<=Math.min(rows-1,(ay+radius)>>4);v++)
      for(let u=Math.max(0,(ax-radius)>>4);u<=Math.min(cols-1,(ax+radius)>>4);u++)lists[v*cols+u].push(i);
  });
  const tiles=new Uint32Array(lists.length+1+lists.reduce((s,v)=>s+v.length,0));let k=lists.length+1;
  lists.forEach((l,i)=>{tiles[i]=k;tiles.set(l,k);k+=l.length;});tiles[lists.length]=k;
  const stride=Math.max(1,Math.floor(image.length/16000));
  const sample=Float32Array.from({length:Math.ceil(image.length/stride)},(_,i)=>image[i*stride]).sort();
  const p=.01*(sample.length-1),bg=sample[Math.floor(p)]+(p%1)*(sample[Math.ceil(p)]-sample[Math.floor(p)]);
  const buffers:GPUBuffer[]=[];
  const buf=(v:number|Float32Array)=>{const b=gpu.buffer(v);buffers.push(b);return b;};
  try{
    const lib=buf(bank),G=buf(geom),K=buf(n*area*4),I=buf(image),T=buf(new Float32Array(tiles.buffer)),F=buf(beads);
    const S=buf(Float32Array.of(Math.max(0,bg),0,0)),P=buf(16*w*h),N=buf(20*n);
    const chunks=Math.ceil(w*h/256),Q=buf(8*chunks);
    const enc=gpu.device.createCommandEncoder();
    const run=(shader:string,bs:GPUBuffer[],count:number)=>gpu.dispatch(enc,shader,bs,count);
    run(stamps,[lib,G,K],n*64);
    const proj=[I,T,G,K,F,S,P];
    for(let it=0;it<=24;it++){
      run(projection(w,h,false,it>=6),proj,w*h);
      if(it===24){run(beadStats(w,h,true),[G,K,P,F,N],n*64);break;}
      // The gradient kernel needs no output normal matrix until the last iteration.
      run(beadStats(w,h,false),[G,K,P,F],n*64);
      if(it%3===0){run(partial(false),[P,Q],chunks);run(total(false),[Q,S],1);}
      else run(zeroBg,[S],1);
      run(projection(w,h,true),proj,w*h);
      run(partial(true),[P,Q],chunks);run(total(true),[Q,S],1);run(update,[F,S],n);
    }
    gpu.device.queue.submit([enc.finish()]);
    const [params,normal,state]=await Promise.all([gpu.read(F),gpu.read(N),gpu.read(S)]);
    const residual=needResidual?await gpu.read(P):undefined;
    return {flux:Array.from({length:n},(_,i)=>params[2*i]),normal,residual,background:state[0]};
  }finally{buffers.forEach(b=>b.destroy());}
}

function addRegistered(sum:Float32Array,packed:Float32Array,w:number,h:number,dx:number,dy:number,n:number){
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const u=x+dx,v=y+dy;
    if(u<0||u>w-1||v<0||v>h-1)continue;
    const left=Math.floor(u),top=Math.floor(v),right=Math.min(w-1,left+1),bottom=Math.min(h-1,top+1);
    const a=u-left,b=v-top;
    sum[y*w+x]+=((1-b)*((1-a)*packed[4*(top*w+left)]+a*packed[4*(top*w+right)])+
       b*((1-a)*packed[4*(bottom*w+left)]+a*packed[4*(bottom*w+right)]))/n;
  }
}

function blur(image:Float32Array,w:number,h:number,sigma:number){
  const r=Math.round(4*sigma),kernel=Array.from({length:2*r+1},(_,i)=>Math.exp(-.5*((i-r)/sigma)**2));
  const mass=kernel.reduce((a,b)=>a+b);const weights=kernel.map(k=>k/mass);
  const mirror=(i:number,n:number)=>{let j=((i%(2*n))+2*n)%(2*n);return j<n?j:2*n-1-j;};
  let a=image,b=new Float32Array(a.length);
  for(let dim=0;dim<2;dim++){
    for(let y=0;y<h;y++)for(let x=0;x<w;x++){
      let value=0;for(let i=-r;i<=r;i++)value+=weights[i+r]*a[dim===0?mirror(y+i,h)*w+x:y*w+mirror(x+i,w)];
      b[y*w+x]=value;
    }
    a=b;b=new Float32Array(a.length);
  }
  return a;
}
function median(values:Float32Array){
  const a=values.sort(),i=a.length>>1;return a.length%2?a[i]:(a[i-1]+a[i])/2;
}
function residualPeaks(image:Float32Array,w:number,h:number,xy:number[][]){
  const narrow=blur(image,w,h,.9),broad=blur(image,w,h,3);
  const score=narrow.map((v,i)=>v-broad[i]),center=median(score.slice());
  const threshold=Math.max(3,5*median(score.map(v=>Math.abs(v-center))));
  const peaks:number[][]=[];
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const v=score[y*w+x];if(v<=threshold)continue;
    let peak=true;
    for(let yy=Math.max(0,y-1);yy<=Math.min(h-1,y+1);yy++)
      for(let xx=Math.max(0,x-1);xx<=Math.min(w-1,x+1);xx++)if(score[yy*w+xx]>v)peak=false;
    if(peak && !xy.some(([u,v])=>(x-u)**2+(y-v)**2<=1.65**2))peaks.push([x,y]);
  }
  return peaks;
}

export async function refinePhotometry(device:GPUDevice,file:File,fit:MovieFit,log=(message:string,fraction:number)=>{}){
  const gpu=new FourierDevice(device),movie=await TiffMovie.open([file]);
  const {width:w,height:h,model,settings}=fit;
  if(movie.n!==fit.drift.length||movie.W!==w||movie.H!==h)throw Error('The photometry catalog and movie dimensions differ.');
  const engine=new VecEngine(device,{...model.optics,roisize:33,npupil:model.npupil,oversample:model.oversample,zernike_noll:model.noll},[],{maxChunk:48});
  const banks:Float32Array[]=[];
  try{
    engine.setZernike(model.zcoef);
    for(const drift of fit.drift)banks.push(await opticalBank(engine,fit,drift[2]));
  }finally{engine.destroy();}
  const xy=fit.xy.map(p=>p.slice()),intensity=fit.intensity.map(row=>row.slice()),background=fit.background.slice();
  for(let round=0;round<6;round++){
    const normal=new Float64Array(5*xy.length);
    const residual=round===1?new Float32Array(h*w):undefined;
    for(let t=0;t<movie.n;t++){
      log('Refining bead photometry · pass '+(round+1)+' / 6 · frame '+(t+1)+' / '+movie.n,(round+t/movie.n)/6);
      const image=await movie.read(t,t+1);
      for(let i=0;i<image.length;i++)image[i]=Math.max(0,(image[i]-settings.offset)*settings.gain);
      device.pushErrorScope('validation');
      let r;
      try{r=await exposure(gpu,banks[t],image,xy,fit.drift[t],intensity[t],w,h,!!residual);}
      finally{const err=await device.popErrorScope();if(err)throw Error(err.message);}
      intensity[t]=r.flux;
      background[t]=r.background;
      for(let i=0;i<normal.length;i++)normal[i]+=r.normal[i];
      if(residual && r.residual)addRegistered(residual,r.residual,w,h,fit.drift[t][0],fit.drift[t][1],movie.n);
    }
    if(round===5)break;
    xy.forEach((p,j)=>{
      const [gx,gy,a,b,c]=normal.subarray(5*j,5*j+5);
      const det=Math.max(1e-8,a*c-b*b);
      p[0]+=Math.max(-.5,Math.min(.5,(c*gx-b*gy)/det));
      p[1]+=Math.max(-.5,Math.min(.5,(a*gy-b*gx)/det));
    });
    if(residual){
      const peaks=residualPeaks(residual,w,h,xy);
      if(peaks.length){
        const sorted=Float32Array.from(intensity.flat()).sort();
        const initial=.25*Math.max(1,sorted[(sorted.length-1)>>1]);
        for(const p of peaks)xy.push(p);
        for(const row of intensity)for(let j=0;j<peaks.length;j++)row.push(initial);
      }
    }
  }
  return {...fit,xy,intensity,background,backgroundSlopes:undefined};
}
