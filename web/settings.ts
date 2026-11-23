const keys = [
  'pixelNm',
  'na',
  'wavelengthNm',
  'immersionIndex',
  'sampleIndex',
  'diameterNm',
  'stepNm',
  'offset',
  'gain',
] as const;
const cookieName = 'simcode_acquisition_v1';
export type Acquisition = Record<(typeof keys)[number], number>;

export function acquisitionSettings() {
  const section = document.querySelector<HTMLElement>('#settings')!;
  const inputs = Array.from(section.querySelectorAll('input'));
  // An empty stack step means read the calibration TIFF metadata.
  function valid(key: (typeof keys)[number], value: string) {
    if (!value.trim()) return key === 'stepNm';
    const n = Number(value);
    if (!Number.isFinite(n)) return false;
    if (key === 'offset') return true;
    if (key === 'stepNm') return n !== 0;
    return key === 'diameterNm' ? n >= 0 : n > 0;
  }
  const values = () => Object.fromEntries(keys.map((key, i) => [key, inputs[i].value]));
  try {
    const cookie = document.cookie.split('; ').find((c) => c.startsWith(cookieName + '='));
    const saved = cookie && JSON.parse(decodeURIComponent(cookie.slice(cookieName.length + 1)));
    keys.forEach((key, i) => {
      if (typeof saved?.[key] === 'string' && valid(key, saved[key])) inputs[i].value = saved[key];
    });
  } catch {
    /* Browser privacy settings can disable cookies. */
  }
  section.addEventListener('input', () => {
    const data = values();
    if (!keys.every((key) => valid(key, data[key])) || +data.na > +data.immersionIndex) return;
    const secure = location.protocol === 'https:' ? '; Secure' : '';
    try {
      document.cookie =
        cookieName +
        '=' +
        encodeURIComponent(JSON.stringify(data)) +
        '; Max-Age=31536000; Path=/; SameSite=Lax' +
        secure;
    } catch {}
  });
  return () => {
    const data = values();
    if (!keys.every((key) => valid(key, data[key])) || +data.na > +data.immersionIndex)
      throw Error('Check the acquisition settings before generating a movie.');
    return Object.fromEntries(keys.map((key) => [key, Number(data[key])])) as Acquisition;
  };
}
