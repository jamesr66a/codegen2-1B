import torch
from flop_counter import FlopCounterMode
import time

D_IN, D_OUT = 2048, 6144
l = torch.nn.Linear(2048, 6144, device='mps')

B = 1
T_MAX = 128

x = torch.randn(B, T_MAX, D_IN, device='mps')

with torch.autograd.profiler.profile(with_stack=True, record_shapes=True) as prof:
  print('stable shape'.center(80, '-'))
  with torch.profiler.record_function("stable"):
    with FlopCounterMode(l) as fc:
      s = time.time()
      for _ in range(T_MAX):
        out = l(x)
      torch.mps.synchronize()
      e = time.time()
      elapsed = e - s
    print(f"elapsed: {elapsed:.3f} s")
    print('GFLOPS/s', fc.get_total_flops() / elapsed / 1e9)
  """
  elapsed: 0.204 s
  GFLOPS/s 2016.605338695304
  """

  print('growing shape'.center(80, '-'))
  with torch.profiler.record_function("unstable"):
    with FlopCounterMode(l) as fc:
      s = time.time()
      for T in range(T_MAX):
        out = l(x[:, :T, :])
      torch.mps.synchronize()
      e = time.time()
      elapsed = e - s
    print(f"elapsed: {elapsed:.3f} s")
    print('GFLOPS/s', fc.get_total_flops() / elapsed / 1e9)
  """
  elapsed: 0.302 s
  GFLOPS/s 688.9602488601146
  """
prof.export_chrome_trace("linear_bench.json")
