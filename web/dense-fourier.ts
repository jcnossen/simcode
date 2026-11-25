// Mixed-radix Stockham FFT. The camera and patch dimensions need not be powers of two.
export class FourierDevice {
    private pipelines = new Map<string, GPUComputePipeline>();
    constructor(readonly device: GPUDevice) { }
    buffer(data: Float32Array | number) {
        const b = this.device.createBuffer({ size: typeof data === 'number' ? data : data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        if (typeof data !== 'number')
            this.device.queue.writeBuffer(b, 0, data);
        return b;
    }
    dispatch(enc: GPUCommandEncoder, code: string, buffers: GPUBuffer[], count: number) {
        let p = this.pipelines.get(code);
        if (!p) {
            p = this.device.createComputePipeline({ layout: 'auto', compute: { module: this.device.createShaderModule({ code }), entryPoint: 'main' } });
            this.pipelines.set(code, p);
        }
        const pass = enc.beginComputePass();
        pass.setPipeline(p);
        pass.setBindGroup(0, this.device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) }));
        pass.dispatchWorkgroups(Math.ceil(count / 64));
        pass.end();
    }
    async read(buffer: GPUBuffer) {
        const out = this.device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        try {
            const enc = this.device.createCommandEncoder();
            enc.copyBufferToBuffer(buffer, 0, out, 0, buffer.size);
            this.device.queue.submit([enc.finish()]);
            await out.mapAsync(GPUMapMode.READ);
            return new Float32Array(out.getMappedRange().slice(0));
        }
        finally {
            out.destroy();
        }
    }
    async fft(data: Float32Array, w: number, h: number, inverse = false) {
        let a = this.buffer(data), b = this.buffer(data.byteLength);
        try {
            const enc = this.device.createCommandEncoder();
            for (const [n, stride] of [[w, 1], [h, w]]) {
                for (let m = 1; m < n;) {
                    let radix = 2;
                    while ((n / m) % radix)
                        radix++;
                    const code = `@group(0) @binding(0) var<storage,read> src: array<vec2f>;
@group(0) @binding(1) var<storage,read_write> dst: array<vec2f>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 let i=id.x;if(i>=arrayLength(&src)){return;}
 let pos=i/${stride}u%${n}u; let base=i-pos*${stride}u;
 let j=pos/${m * radix}u*${m}u+pos%${m}u;
 let angle=${(inverse ? 2 : -2) * Math.PI}*f32(pos%${radix * m}u)/${m * radix}.0;
 var total=vec2f(0);for(var l=0u;l<${radix}u;l++){
 let v=src[base+(j+l*${n / radix}u)*${stride}u];let phase=f32(l)*angle;
 total+=v*cos(phase)+vec2f(-v.y,v.x)*sin(phase);}
 dst[i]=total/${inverse ? radix : 1}.0;}`;
                    this.dispatch(enc, code, [a, b], data.length / 2);
                    [a, b] = [b, a];
                    m *= radix;
                }
            }
            this.device.queue.submit([enc.finish()]);
            return await this.read(a);
        }
        finally {
            a.destroy();
            b.destroy();
        }
    }
}
export function complex(data: ArrayLike<number>) {
    const z = new Float32Array(2 * data.length);
    for (let i = 0; i < data.length; i++)
        z[2 * i] = data[i];
    return z;
}
export const frequency = (j: number, n: number) => (j < Math.ceil(n / 2) ? j : j - n) / n;
export const tau = 2 * Math.PI;
