import numpy as np
import cudamat as cm

n, p = int(2e3), int(40e3)
A = np.random.randn(n, p)
B = np.random.randn(p, n)
%timeit A @ B

cm.cublas_init()
cm.CUDAMatrix.init_random()
A_cm = cm.empty((n, p)).fill_with_randn()
B_cm = cm.empty((p, n)).fill_with_randn()
%timeit A_cm.dot(B_cm)
cm.cublas_shutdown()
