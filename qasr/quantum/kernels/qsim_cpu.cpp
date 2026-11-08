// =====================================================================
//  qsim_cpu.cpp - batched statevector simulator with ADJOINT gradients
//  (C++17, OpenMP over the batch, compiled at runtime, loaded via ctypes)
//
//  Same algorithm as qsim_cuda.cu; see circuit.py for conventions.
//  Gate types: 0=RX 1=RY 2=RZ 3=CNOT(q0 -> q1)
//
//  Forward : |psi> = U_G ... U_1 |0>,  out[b,q] = <psi| Z_q |psi>
//  Backward: given g[b,q] = dLoss/d<Z_q>, with O = sum_q g_q Z_q:
//            lambda = O psi_G
//            for k = G..1:   dLoss/dtheta_k = Im <lambda| P_k |psi_k>
//                            psi <- U_k^dagger psi ; lambda <- U_k^dagger lambda
//  Cost: ~3 statevector passes regardless of the number of parameters
//  (parameter-shift would need 2 * n_params full circuit runs).
// =====================================================================
#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#define QSIM_API extern "C" __declspec(dllexport)
#else
#define QSIM_API extern "C" __attribute__((visibility("default")))
#endif

typedef std::complex<float> cf;

static inline long insert_zero(long i, int q) {
    return ((i >> q) << (q + 1)) | (i & ((1L << q) - 1));
}

static void apply_rot(cf* st, int n, int type, int q, float theta) {
    const long half = 1L << (n - 1);
    const float c = std::cos(0.5f * theta), s = std::sin(0.5f * theta);
    for (long p = 0; p < half; ++p) {
        const long i0 = insert_zero(p, q), i1 = i0 | (1L << q);
        const cf a = st[i0], b = st[i1];
        if (type == 0) {         // RX = [[c, -is], [-is, c]]
            st[i0] = cf(c * a.real() + s * b.imag(), c * a.imag() - s * b.real());
            st[i1] = cf(s * a.imag() + c * b.real(), -s * a.real() + c * b.imag());
        } else if (type == 1) {  // RY = [[c, -s], [s, c]]
            st[i0] = c * a - s * b;
            st[i1] = s * a + c * b;
        } else {                 // RZ = diag(e^{-i t/2}, e^{+i t/2})
            st[i0] = a * cf(c, -s);
            st[i1] = b * cf(c, s);
        }
    }
}

static void apply_cnot(cf* st, int n, int ctrl, int tgt) {
    const long D = 1L << n;
    for (long i = 0; i < D; ++i)
        if (((i >> ctrl) & 1) && !((i >> tgt) & 1)) std::swap(st[i], st[i | (1L << tgt)]);
}

static void run_forward(cf* st, const float* ang, int n, int G, const int* gt, const int* g0, const int* g1,
                        const int* ga) {
    const long D = 1L << n;
    for (long i = 0; i < D; ++i) st[i] = cf(0.f, 0.f);
    st[0] = cf(1.f, 0.f);
    for (int k = 0; k < G; ++k) {
        if (gt[k] == 3) apply_cnot(st, n, g0[k], g1[k]);
        else apply_rot(st, n, gt[k], g0[k], ang[ga[k]]);
    }
}

// Im <lam| P_q |psi> for P in {X, Y, Z}
static float im_expect(const cf* lam, const cf* psi, int n, int type, int q) {
    const long half = 1L << (n - 1);
    double acc = 0.0;
    for (long p = 0; p < half; ++p) {
        const long i0 = insert_zero(p, q), i1 = i0 | (1L << q);
        const cf l0 = lam[i0], l1 = lam[i1], p0 = psi[i0], p1 = psi[i1];
        if (type == 0)       acc += std::imag(std::conj(l0) * p1) + std::imag(std::conj(l1) * p0);
        else if (type == 1)  acc += -std::real(std::conj(l0) * p1) + std::real(std::conj(l1) * p0);
        else                 acc += std::imag(std::conj(l0) * p0) - std::imag(std::conj(l1) * p1);
    }
    return (float)acc;
}

QSIM_API int qsim_forward(const float* angles, int B, int n_angles, int n, int G, const int* gt, const int* g0,
                          const int* g1, const int* ga, float* out) {
    const long D = 1L << n;
#pragma omp parallel
    {
        std::vector<cf> st(D);
#pragma omp for schedule(static)
        for (int b = 0; b < B; ++b) {
            run_forward(st.data(), angles + (long)b * n_angles, n, G, gt, g0, g1, ga);
            for (int q = 0; q < n; ++q) {
                double z = 0.0;
                for (long i = 0; i < D; ++i) z += (((i >> q) & 1) ? -1.0 : 1.0) * std::norm(st[i]);
                out[(long)b * n + q] = (float)z;
            }
        }
    }
    return 0;
}

QSIM_API int qsim_backward(const float* angles, const float* grad_out, int B, int n_angles, int n, int G,
                           const int* gt, const int* g0, const int* g1, const int* ga, float* grad_angles) {
    const long D = 1L << n;
    std::memset(grad_angles, 0, sizeof(float) * (size_t)B * (size_t)n_angles);
#pragma omp parallel
    {
        std::vector<cf> psi(D), lam(D);
#pragma omp for schedule(static)
        for (int b = 0; b < B; ++b) {
            const float* ang = angles + (long)b * n_angles;
            const float* g = grad_out + (long)b * n;
            float* ga_out = grad_angles + (long)b * n_angles;
            run_forward(psi.data(), ang, n, G, gt, g0, g1, ga);
            for (long i = 0; i < D; ++i) {
                float w = 0.f;
                for (int q = 0; q < n; ++q) w += ((i >> q) & 1) ? -g[q] : g[q];
                lam[i] = psi[i] * w;
            }
            for (int k = G - 1; k >= 0; --k) {
                if (gt[k] == 3) {
                    apply_cnot(psi.data(), n, g0[k], g1[k]);
                    apply_cnot(lam.data(), n, g0[k], g1[k]);
                } else {
                    ga_out[ga[k]] += im_expect(lam.data(), psi.data(), n, gt[k], g0[k]);
                    apply_rot(psi.data(), n, gt[k], g0[k], -ang[ga[k]]);
                    apply_rot(lam.data(), n, gt[k], g0[k], -ang[ga[k]]);
                }
            }
        }
    }
    return 0;
}

QSIM_API int qsim_version() { return 1; }
