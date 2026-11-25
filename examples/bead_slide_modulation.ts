import { existsSync, openSync, closeSync, readSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { MovieDepth } from '../web/movie-depth';
import { modulation } from '../web/modulation';

const [work, output] = process.argv.slice(2);
const { snapshot, schedule } = JSON.parse(readFileSync(join(work, 'input.json'), 'utf8'));
const pixels = join(work, 'pixels.f32');
let image: MovieDepth | undefined;
if (existsSync(pixels)) {
  image = new MovieDepth(snapshot.width, snapshot.height, schedule);
  const fd = openSync(pixels, 'r');
  const frame = new Float32Array(snapshot.width * snapshot.height);
  try {
    for (let t = 0; t < snapshot.drift.length; t++) {
      const bytes = readSync(fd, frame, 0, frame.byteLength, null);
      if (bytes !== frame.byteLength) throw Error('Incomplete movie frame');
      image.add(frame, t);
    }
  } finally {
    closeSync(fd);
  }
}
writeFileSync(output, JSON.stringify(modulation(snapshot, schedule, image), null, 2));
