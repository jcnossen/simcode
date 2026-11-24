import { footprint } from 'torch-dense/gel';

// Sample the same finite-bead PSF used for image formation and fitting.
export function volumePsfSlices(bank:Float32Array) {
  const side=31, center=15, dz=.24;
  const planes=new Float32Array(side**3), axial=new Float32Array(side**2);
  let peak=0, peakRow=center;
  for(let row=0;row<side;row++) {
    const plane=planes.subarray(row*side**2,(row+1)*side**2);
    footprint(side,side,bank,center,center,(center-row)*dz,(p,v)=>{
      plane[p]=v;
      if(v>peak){peak=v;peakRow=row;}
    });
    axial.set(plane.subarray(center*side,(center+1)*side),row*side);
  }
  return {planes,axial,side,peakRow,dz};
}
export type VolumeSlices=ReturnType<typeof volumePsfSlices>;
