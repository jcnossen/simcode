export const defaultPhotons = {signal: 4500, background: 5};

// Exact Poisson sampling for dim pixels; the normal approximation is accurate at high counts.
export function photonCount(mean: number, normal: number) {
  if (mean >= 25) return Math.max(0, Math.round(mean + Math.sqrt(mean) * normal));
  if (mean === 0) return 0;
  const limit = Math.exp(-mean);
  let product = 1, count = 0;
  do { product *= Math.random(); count++; } while (product > limit);
  return count - 1;
}
