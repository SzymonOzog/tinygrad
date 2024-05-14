import ctypes
from tinygrad.device import Compiled, MallocAllocator
import subprocess
from tinygrad.helpers import getenv
from tinygrad.runtime.ops_hsa import HSACompiler, RDNA3Compiler

rhip = ctypes.CDLL("/usr/local/lib/libremu.so")
class RHIPProgram:
  def __init__(self, name:str, lib:bytes):
    self.name, self.lib = name, lib
    asm = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], input=lib)
    print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))
  def __call__(self, *args, global_size, local_size, vals=(), wait=False):
    args = (*args, *vals)
    rhip.hipModuleLaunchKernel(self.lib, len(self.lib), *global_size, *local_size, 0, None, None,
                              len(args), (ctypes.c_void_p * len(args))(*[ctypes.cast(x, ctypes.c_void_p) for x in args]))

class RHIPDevice(Compiled):
  def __init__(self, device:str=""):
    self.device = int(device.split(":")[1]) if ":" in device else 0
    super().__init__(device, MallocAllocator, RDNA3Compiler("gfx1100") if getenv("RDNA") else HSACompiler("gfx1100"), RHIPProgram)
