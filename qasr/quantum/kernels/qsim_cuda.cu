// =====================================================================
//  qsim_cuda.cu - batched statevector simulator with ADJOINT gradients
//  for NVIDIA GPUs. Compiled at runtime with NVRTC (through CuPy), so no
//  nvcc / Visual Studio toolchain is needed on the training machine.
//
//  Design (one CUDA block  <->  one audio sample):
//    * the whole 2^n statevector lives in on-chip SHARED memory
//      (n=10: 1024 amplitudes * 8 B = 8 KB; backward keeps psi + lambda)
//    * threads of the block split the amplitude pairs of every gate
//    * the batch dimension maps to the grid -> thousands of samples in parallel
//    * zero global-memory traffic inside the gate loop
//  Mirrors qsim_cpu.cpp exactly; see circuit.py for the conventions.
//  Gate types: 0=RX 1=RY 2=RZ 3=CNOT(q0 -> q1)
// =====================================================================

__device__ __forceinline__ int insert_zero(int i, int q) {
    return ((i >> q) << (q + 1)) | (i & ((1 << q) - 1));
}

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Im(conj(a) * b) and Re(conj(a) * b)
__device__ __forceinline__ float cim(float2 a, float2 b) { return a.x * b.y - a.y * b.x; }
__device__ __forceinline__ float cre(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }

__device__ void apply_rot(float2* st, int n, int type, int q, float theta) {
    const int half = 1 << (n - 1);
    float s, c;
    sincosf(0.5f * theta, &s, &c);
    for (int p = threadIdx.x; p < half; p += blockDim.x) {
        const int i0 = insert_zero(p, q), i1 = i0 | (1 << q);
        const float2 a = st[i0], b = st[i1];
        float2 r0, r1;
        if (type == 0) {         // RX
            r0 = make_float2(c * a.x + s * b.y, c * a.y - s * b.x);
            r1 = make_float2(s * a.y + c * b.x, -s * a.x + c * b.y);
        } else if (type == 1) {  // RY
            r0 = make_float2(c * a.x - s * b.x, c * a.y - s * b.y);
            r1 = make_float2(s * a.x + c * b.x, s * a.y + c * b.y);
        } else {                 // RZ
            r0 = cmul(a, make_float2(c, -s));
            r1 = cmul(b, make_float2(c, s));
        }
        st[i0] = r0;
        st[i1] = r1;
    }
}

__device__ void apply_cnot(float2* st, int n, int ctrl, int tgt) {
    const int D = 1 << n;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        if (((i >> ctrl) & 1) && !((i >> tgt) & 1)) {
            const int j = i | (1 << tgt);
            const float2 t = st[i];
            st[i] = st[j];
            st[j] = t;
        }
    }
}

// blockDim.x must be a power of two; red has blockDim.x floats
__device__ float block_sum(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

__device__ void run_forward(float2* st, const float* ang, int n, int G, const int* gt, const int* g0,
                            const int* g1, const int* ga) {
    const int D = 1 << n;
    for (int i = threadIdx.x; i < D; i += blockDim.x) st[i] = make_float2(i == 0 ? 1.f : 0.f, 0.f);
    __syncthreads();
    for (int k = 0; k < G; ++k) {
        const int t = gt[k];
        if (t == 3) apply_cnot(st, n, g0[k], g1[k]);
        else apply_rot(st, n, t, g0[k], ang[ga[k]]);
        __syncthreads();
    }
}

extern "C" __global__ void qsim_forward(const float* angles, int n_angles, int n, int G, const int* gt,
                                        const int* g0, const int* g1, const int* ga, float* out) {
    extern __shared__ float2 smem[];
    const int D = 1 << n;
    float2* st = smem;
    float* red = (float*)(smem + D);
    const int b = blockIdx.x;
    run_forward(st, angles + (long long)b * n_angles, n, G, gt, g0, g1, ga);
    for (int q = 0; q < n; ++q) {
        float part = 0.f;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            const float p = st[i].x * st[i].x + st[i].y * st[i].y;
            part += ((i >> q) & 1) ? -p : p;
        }
        const float z = block_sum(part, red);
        if (threadIdx.x == 0) out[(long long)b * n + q] = z;
    }
}

// grad_angles must be zero-initialised by the caller
extern "C" __global__ void qsim_backward(const float* angles, const float* grad_out, int n_angles, int n, int G,
                                         const int* gt, const int* g0, const int* g1, const int* ga,
                                         float* grad_angles) {
    extern __shared__ float2 smem[];
    const int D = 1 << n;
    float2* psi = smem;
    float2* lam = smem + D;
    float* red = (float*)(smem + 2 * D);
    const int b = blockIdx.x;
    const float* ang = angles + (long long)b * n_angles;
    const float* g = grad_out + (long long)b * n;
    float* gout = grad_angles + (long long)b * n_angles;

    run_forward(psi, ang, n, G, gt, g0, g1, ga);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float w = 0.f;
        for (int q = 0; q < n; ++q) w += ((i >> q) & 1) ? -g[q] : g[q];
        lam[i] = make_float2(psi[i].x * w, psi[i].y * w);
    }
    __syncthreads();

    const int half = 1 << (n - 1);
    for (int k = G - 1; k >= 0; --k) {
        const int t = gt[k], q = g0[k];
        if (t == 3) {
            apply_cnot(psi, n, q, g1[k]);
            apply_cnot(lam, n, q, g1[k]);
            __syncthreads();
            continue;
        }
        float part = 0.f;
        for (int p = threadIdx.x; p < half; p += blockDim.x) {
            const int i0 = insert_zero(p, q), i1 = i0 | (1 << q);
            const float2 l0 = lam[i0], l1 = lam[i1], p0 = psi[i0], p1 = psi[i1];
            if (t == 0)      part += cim(l0, p1) + cim(l1, p0);
            else if (t == 1) part += -cre(l0, p1) + cre(l1, p0);
            else             part += cim(l0, p0) - cim(l1, p1);
        }
        const float grad = block_sum(part, red);
        if (threadIdx.x == 0) gout[ga[k]] += grad;
        const float th = -ang[ga[k]];
        apply_rot(psi, n, t, q, th);
        apply_rot(lam, n, t, q, th);
        __syncthreads();
    }
}
