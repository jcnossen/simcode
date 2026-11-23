import { axisName } from './schedule';
export interface PatternField {
  patterns: { pitch_nm:number; orientation_deg:number; kx_rad_px:number; ky_rad_px:number; depth:number[]; phase:number[]; phase_steps_rad:number[] }[];
  ratios:number[][];
}
const dot = (a:number[], b:number[]) => a.reduce((sum,v,i)=>sum+v*b[i],0);
export function fieldMaps({patterns,ratios}:PatternField,width:number,height:number) {
  return Array.from({ length: 32 * 32 }, (_, i) => {
    const x = ((i % 32) * (width - 1)) / 31,
      y = (Math.floor(i / 32) * (height - 1)) / 31,
      u = 2*x/(width-1)-1, v = 2*y/(height-1)-1, b = [1,u,v,u*u,u*v,v*v];
    const angles = patterns.map((p) => {
      const c = p.phase,
        u = b[1],
        v = b[2];
      const kx = p.kx_rad_px + (2 * (c[1] + 2 * c[3] * u + c[4] * v)) / (width - 1);
      const ky = p.ky_rad_px + (2 * (c[2] + c[4] * u + 2 * c[5] * v)) / (height - 1);
      return ((Math.atan2(ky, kx) * 180) / Math.PI + 360) % 360;
    });
    const row: Record<string, number> = { x_px: x, y_px: y };
    patterns.forEach((p, i) => {
      const label = axisName(i);
      row['orientation_' + label + '_deg'] = angles[i];
      row['depth_' + label] = dot(p.depth, b);
      row['phase_offset_' + label + '_rad'] = dot(p.phase,b);
      row['phase_' + label + '_rad'] = p.kx_rad_px * x + p.ky_rad_px * y + dot(p.phase, b);
      if (i) row['ratio_A_over_' + label] = dot(ratios[i - 1], b);
    });
    return row;
  });
}
