export const defaultAberration = 150;

// Orthonormal Noll coefficients combine in quadrature to give pupil RMS.
export function exampleCoefficients(modes: number[], milliWaves: number) {
  const shape = modes.map(j => {
    const magnitude = j === 6 ? .10 + .04 * Math.random()
      : j === 7 || j === 8 ? .05 + .04 * Math.random() : .015 + .04 * Math.random();
    return (Math.random() < .5 ? -1 : 1) * magnitude;
  });
  const scale = milliWaves / (1000 * Math.hypot(...shape));
  return shape.map(v => v * scale);
}
