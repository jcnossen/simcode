import { axisAt, phaseDegrees, defaultSchedule, type Schedule } from './schedule';
export type Pattern = (x: number, y: number, frame: number) => number;

// Cache display pixels only. The uploaded TIFF and the estimator retain their native sampling.
export function movieView(
  container: HTMLElement,
  zoom: HTMLElement,
  label: string,
  onSelect?: (frame: number) => void,
) {
  container.innerHTML =
    '<canvas class="movie-frame" tabindex="0"></canvas><label class="scrub">Frame <input type="range" min="1" max="1" value="1" step="1"></label><small class="frame-number"></small> <input class="follow" type="button" value="Follow processing" hidden>';
  zoom.innerHTML = '<figcaption>Pattern zoom</figcaption><canvas></canvas><small></small>';
  const canvas = container.querySelector('canvas')!,
    ctx = canvas.getContext('2d')!;
  canvas.setAttribute('aria-label', label);
  const slider = container.querySelector<HTMLInputElement>('input[type=range]')!,
    text = container.querySelector('small')!;
  const follow = container.querySelector<HTMLInputElement>('.follow')!;
  const detail = zoom.querySelector('canvas')!,
    zc = detail.getContext('2d')!,
    caption = zoom.querySelector('small')!;
  detail.width = detail.height = 160;
  let images: Uint8ClampedArray[] = [],
    w = 0,
    h = 0,
    bin = 1,
    last = 0,
    current = 0,
    auto = true,
    scale = 1,
    px = 0.5,
    py = 0.5;
  let renderPattern: Pattern | null = null,
    patternLabel = '',
    stage = '',
    total = 0;
  let schedule = defaultSchedule;
  let image: ImageData | undefined;
  function updateZoom() {
    const a = images[current];
    if (!a) return;
    const n = detail.width,
      side = Math.min(20, w, h);
    const left = Math.max(0, Math.min(w - side, px * w - side / 2)),
      top = Math.max(0, Math.min(h - side, py * h - side / 2));
    const rgba = zc.createImageData(n, n);
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const x = left + ((i + 0.5) * side) / n - 0.5,
          y = top + ((j + 0.5) * side) / n - 0.5;
        const ix = Math.min(canvas.width - 1, Math.floor(Math.max(0, Math.round(x)) / bin)),
          iy = Math.min(canvas.height - 1, Math.floor(Math.max(0, Math.round(y)) / bin));
        const gray = a[iy * canvas.width + ix],
          alpha = renderPattern?.(x, y, current) ?? 0;
        const blend = 0.53 * Math.max(0, Math.min(1, alpha)),
          k = 4 * (j * n + i);
        rgba.data[k] = gray * (1 - blend);
        rgba.data[k + 1] = gray * (1 - blend) + 245 * blend;
        rgba.data[k + 2] = gray * (1 - blend) + 212 * blend;
        rgba.data[k + 3] = 255;
      }
    zc.putImageData(rgba, 0, 0);
    caption.textContent =
      (renderPattern
        ? patternLabel + ' · ' + 'axis ' + axisAt(current, schedule) + ' · ' + Number(phaseDegrees(current, schedule).toFixed(2)) + '°'
        : 'Movie ROI · awaiting fitted pattern') +
      ' · x ' +
      Math.round(left + side / 2) +
      ', y ' +
      Math.round(top + side / 2) +
      ' px';
  }
  function show(i: number) {
    if (!images[i] || !image) return;
    current = i;
    slider.value = String(i + 1);
    const a = images[i];
    for (let k = 0; k < a.length; k++) {
      image.data[k * 4] = image.data[k * 4 + 1] = image.data[k * 4 + 2] = a[k];
      image.data[k * 4 + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
    text.textContent = (auto ? stage + ' frame ' : 'Frame ') + (i + 1) + ' / ' + total;
    updateZoom();
    onSelect?.(i);
  }
  function select(i: number) {
    auto = false;
    follow.hidden = false;
    show(Math.max(0, Math.min(images.length - 1, i)));
  }
  slider.oninput = () => select(Number(slider.value) - 1);
  follow.onclick = () => {
    auto = true;
    follow.hidden = true;
    show(last);
  };
  canvas.addEventListener(
    'wheel',
    (e) => {
      if (images.length) {
        e.preventDefault();
        select(current + Math.sign(e.deltaY));
      }
    },
    { passive: false },
  );
  canvas.onpointermove = (e) => {
    const b = canvas.getBoundingClientRect();
    px = (e.clientX - b.left) / b.width;
    py = (e.clientY - b.top) / b.height;
    updateZoom();
  };
  return {
    frame(
      pixels: Float32Array,
      width: number,
      height: number,
      index: number,
      count: number,
      action: string,
      offset = 0,
    ) {
      w = width;
      h = height;
      total = count;
      last = index;
      stage = action;
      if (!image) {
        bin = Math.max(1, Math.ceil(Math.sqrt((count * w * h) / (32 * 1024 * 1024))));
        canvas.width = Math.ceil(w / bin);
        canvas.height = Math.ceil(h / bin);
        image = ctx.createImageData(canvas.width, canvas.height);
        const stride = Math.max(1, Math.floor(pixels.length / 4096));
        const sample = Float32Array.from(
          { length: Math.ceil(pixels.length / stride) },
          (_, i) => pixels[i * stride],
        ).sort();
        scale = Math.max(1, sample[Math.floor((sample.length - 1) * 0.995)] - offset);
      }
      if (!images[index]) {
        const a = new Uint8ClampedArray(canvas.width * canvas.height);
        for (let y = 0; y < canvas.height; y++)
          for (let x = 0; x < canvas.width; x++)
            a[y * canvas.width + x] =
              255 * Math.sqrt(Math.max(0, (pixels[y * bin * w + x * bin] - offset) / scale));
        images[index] = a;
      }
      slider.max = String(images.length);
      if (auto) show(index);
    },
    setPattern(pattern: Pattern | null, name: string, order: Schedule) {
      schedule = order;
      renderPattern = pattern;
      patternLabel = name;
      updateZoom();
    },
    reset() {
      images = [];
      image = undefined;
      auto = true;
      follow.hidden = true;
      slider.value = slider.max = '1';
      text.textContent = '';
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      zc.clearRect(0, 0, detail.width, detail.height);
    },
  };
}
