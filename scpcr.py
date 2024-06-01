import itertools, random
from collections import defaultdict
from tinygrad.dtype import ImageDType, dtypes
from tinygrad.device import Device, Buffer
from tinygrad.ops import MemBuffer
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import to_function_name, getenv, colored
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.engine.search import bufs_from_lin
from tinygrad.runtime.ops_cuda import PTXCompiler, PTXRenderer, CUDACompiler
from pathlib import Path
from tinygrad.codegen.kernel import Opt, OptOps, KernelOptError
from typing import Dict, List, cast, DefaultDict, Optional, Tuple, Callable
from tqdm import tqdm
import numpy as np
import os

# move to helpers?
def colorize_float(x):
  ret = f"{x:7.2f}x"
  if x < 0.75: return colored(ret, 'green')
  elif x > 1.15: return colored(ret, 'red')
  else: return colored(ret, 'yellow')

def unique(sequence):
  seen = set()
  return [x for x in sequence if not (x in seen or seen.add(x))]

if __name__ == "__main__":
  f = open(Path("sops_rn2"), 'r')
  ast_strs = unique(f.readlines())
  f = open(Path("sops_ptx2"), 'r')
  ast_strs2 = unique(f.readlines())
  dev = Device["CUDA"]
  ptx = PTXRenderer(dev.arch)

  # NUM=112 python3 test/external/speed_compare_cuda_ptx.py

  single = getenv("NUM", -1)
  # single = 1 # getenv("NUM", -1)
  if single != -1: ast_strs = ast_strs[single:single+1]
  if single != -1: ast_strs2 = ast_strs2[single:single+1]

  average_tm_cuda, average_tm_ptx = 0, 0
  slow = []
  for num, ast in (pbar:=tqdm(enumerate(ast_strs))):
    if num == 44:continue
    opts = ast.split("|")[1]
    ast = ast.split("|")[0]
    # cuda compile
    dev.compiler = CUDACompiler(dev.arch)
    lin = ast_str_to_lin(ast, opts=dev.renderer)
    lin.required_optimizations()
    for o in eval(opts):
      lin.apply_opt(o)
    cuda_prg = CompiledRunner(lin.to_program()) 

    bufs = bufs_from_lin(lin)
    bufs_2 = bufs_from_lin(lin)
    bufs_3 = bufs_from_lin(lin)
    if getenv("FILL"):
      for i in range(len(bufs)):
        data = np.random.default_rng().standard_normal(size=int(bufs[i].nbytes/bufs[i].dtype.itemsize), dtype="f").astype(bufs[i].dtype.np).data
        # data = memoryview(bytearray(os.urandom(bufs[i].nbytes)))
        # data = memoryview(bytearray([random.choice([True,False]) for _ in  range(bufs[i].nbytes)]))
        bufs[i].copyin(data)
        bufs_2[i].copyin(data)

    # ptx compile
    dev.compiler = PTXCompiler(dev.arch)
    ast_ptx = ast_strs2[num].split("|")[0]
    ops_ptx = ast_strs2[num].split("|")[1]
    assert ast_ptx == ast
    lin = ast_str_to_lin(ast_ptx, opts=ptx)
    lin.required_optimizations()
    for o in eval(ops_ptx):
      lin.apply_opt(o)
    lin.linearize()
    ptx_prg = CompiledRunner(lin.to_program())

    lin = ast_str_to_lin(ast_ptx, opts=ptx)
    lin.required_optimizations()
    for o in eval(opts):
      lin.apply_opt(o)
    lin.linearize()
    ptx_prg2 = CompiledRunner(lin.to_program())

    try:
      cuda_prg(bufs, {}, wait=True)
      ptx_prg(bufs_2, {}, wait=True)
      ptx_prg2(bufs_3, {}, wait=True)
    except:
      continue
    if getenv("FILL"):
      for b1, b2 in zip(bufs, bufs_2):
        np1 = np.frombuffer(b1.base.as_buffer(), b1.dtype.np)
        np2 = np.frombuffer(b2.base.as_buffer(), b1.dtype.np)
        # print("out buffers:")
        # print(np1)
        # print(np1.sum())
        # print(np2)
        # print(np2.sum())
        # atol=1e-5
        # rtol=1e-2
        # close = np.logical_not(np.isclose(np1,np2, atol=atol, rtol=rtol))
        # indices = np.argwhere(close)
        # print(indices)
        # print(np1[close])
        # print(np2[close])
        # np.testing.assert_allclose(np1, np2, atol=atol, rtol=rtol, err_msg=f"NOT CLOSE, {num}")
        try:
          np.testing.assert_allclose(np1, np2, atol=atol, rtol=rtol, err_msg=f"NOT CLOSE, {num}")
        except AssertionError:
          print("unmatching buffers at num", num)
            # print(np1)
            # print(np2)


    tm_cuda, tm_ptx, tm_ptx2 = [], [], []
    for i in range(5):
      tm_cuda.append(cuda_prg(bufs, {}, wait=True))
      tm_ptx.append(ptx_prg(bufs, {}, wait=True))
      tm_ptx2.append(ptx_prg2(bufs, {}, wait=True))
    average_tm_cuda += min(tm_cuda)
    average_tm_ptx += min(tm_ptx)
    ratio = min(tm_ptx)/min(tm_cuda)
    ratio2 = min(tm_ptx2)/min(tm_cuda)
    # print(f"{average_tm_ptx/average_tm_cuda:5.2f}x -- {num:4d} {colorize_float(ratio)}  {min(tm_ptx)*1e6:7.2f} us", lin.name)
    pbar.set_description(f"{average_tm_ptx/average_tm_cuda:5.2f}")
    if ratio > 1.2 or single != -1 or True:
      slow.append((num, ratio, min(tm_cuda), to_function_name(lin.name), ratio2))
      def fix(x): return x.replace('\t', ' ').strip()
      ll1, ll2 = cuda_prg.lib.decode().split('\n'), ptx_prg.lib.decode().split('\n')
      ll1 = [l1 for l1 in ll1 if "inline asm" not in l1 and l1 != "\n" and '}' != l1 and len(l1)>3]
      ll1 = ["tensor cores" if "mma.sync" in l1 else l1 for l1 in ll1]
      ll2 = ["tensor cores" if "mma.sync" in l2 else l2 for l2 in ll2]
      ii1 = set([l1.split(" ")[0].strip() for l1 in ll1 if l1.lstrip()[0] not in ["$", "@", "%", "/"]])
      ii2 = set([l2.replace("\t"," ").lstrip().split(" ")[0].strip() for l2 in  ll2 if l2.lstrip()[0] not in ["$", "@", "%", "/"]])
      if single != -1:
        for ln, (l1, l2) in enumerate(itertools.zip_longest(ll1, ll2, fillvalue='')):
          print(f"{ln:5d} | {fix(l1):80s} | {fix(l2):80s}")
      # print(len(ll1), len(ll2), "RATIO", ratio, "us", min(tm_ptx)*1e6)
  print(f"final ratio:{average_tm_ptx/average_tm_cuda:5.2f}, same_ops:{ratio2:5.2f}")
  # for i1 in ii1:
  #   if i1 not in ii2:
  #     print("unused", i1)
  # for i2 in ii2:
  #   if i2 not in ii1:
  #     print("extra", i2)
  slow.sort(key=lambda x: x[1]*x[2])
  for x in slow[-30:]:
    print(f"num {x[0]}, ratio:{x[1]}, ratio2:{x[4]}, time: {x[2]*1e6:7.2f} us, name: {x[3]}")
