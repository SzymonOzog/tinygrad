from typing import Callable, DefaultDict, Dict, List, Union, NamedTuple, Optional, cast
import functools, struct
from collections import defaultdict
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps, Op
from tinygrad.dtype import dtypes, DType, PtrDType, ConstType, INVERSE_DTYPES_DICT
from tinygrad.codegen.uops import UOpGraph, PatternMatcher

def render_val(x, dtype):
  if dtypes.is_float(dtype):
    if dtype == dtypes.double: return "0d%02X%02X%02X%02X%02X%02X%02X%02X" % tuple(struct.pack("d",x)[::-1])
    elif dtype == dtypes.half: return "0x%02X%02X" % tuple(struct.pack("e",x)[::-1])
    return "0f%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
  return str(int(x)) + ("U" if dtypes.is_unsigned(dtype) else "")

def ptr_ar(root, uops):
  assert root.arg in {'.shared', '.global', None}
  if root.arg is None: root.arg = '.shared' if root.vin[0].uop is UOps.DEFINE_LOCAL else '.global'  # move this to the argL
  val = uops.add(UOps.CONST, dtypes.int, tuple(), arg=root.vin[0].dtype.itemsize, insert_before=uops.uops.index(root))
  if root.vin[1].uop is UOps.ALU and root.vin[1].arg in [BinaryOps.ADD, BinaryOps.SUB] and root.vin[1].vin[1].uop is UOps.CONST:
    offset = uops.add(UOps.ALU, dtypes.int, (root.vin[1].vin[0], val), arg=BinaryOps.MUL, insert_before=uops.uops.index(root))
    offset = uops.add(UOps.CAST, dtypes.uint64, (offset,), insert_before=uops.uops.index(root))
    cache = uops.add(UOps.ALU, dtypes.uint64, (root.vin[0], offset), arg=BinaryOps.ADD, insert_before=uops.uops.index(root))
    ptr = uops.add(UOps.ALU, dtypes.int, (root.vin[1].vin[1], val), arg=BinaryOps.MUL, insert_before=uops.uops.index(root))
    if root.vin[1].arg == BinaryOps.SUB: ptr = uops.add(UOps.ALU, dtypes.int, (ptr,), arg=UnaryOps.NEG, insert_before=uops.uops.index(root))
    root.vin = (cache, ptr) + root.vin[2:]
  else:
    ptr = uops.add(UOps.ALU, dtypes.int, (root.vin[1], val), arg=BinaryOps.MUL, insert_before=uops.uops.index(root))
    if ptr.uop is UOps.CONST: root.vin = (root.vin[0], ptr) + root.vin[2:]
    else:
      zero = uops.add(UOps.CONST, dtypes.int, tuple(), arg=0, cachable=False, insert_before=uops.uops.index(root))
      bptr = uops.add(UOps.CAST, dtypes.uint64, (ptr,), insert_before=uops.uops.index(root))
      fptr = uops.add(UOps.ALU, dtypes.uint64, (root.vin[0], bptr), arg=BinaryOps.ADD, insert_before=uops.uops.index(root))
      root.vin = (fptr, zero) + root.vin[2:]

def optimize_gated_loads(uops: UOpGraph):
  @functools.lru_cache(None)
  def successors(uop): return list(filter(lambda u: uop in u.vin, uops.uops))
  for gl in list(filter(lambda u:u.uop is UOps.LOAD and len(u.vin)>3, uops.uops)):
    pred_2 = uops.add(UOps.ALU, dtypes.bool, (gl.vin[2],), arg=UnaryOps.NEG, insert_before=uops.uops.index(gl))
    gate = uops.add(UOps.IF, None, (pred_2,), insert_before=uops.uops.index(gl), cachable=False)
    end = uops.add(UOps.ENDIF, None, (gate,), arg=(gl, gl.vin[3]), insert_before=uops.uops.index(gl)+1, cachable=False)
  for gl in list(filter(lambda u:u.uop is UOps.LOAD and len(u.vin)>3, uops.uops)):
    queue = list(gl.vin)
    gi, ei = uops.uops.index(gate), uops.uops.index(end)
    while queue:
      u = queue.pop()
      scc = successors(u)
      inces = [uops.uops.index(x) for x in scc]
      if (u.uop not in [UOps.DEFINE_GLOBAL, UOps.DEFINE_VAR, UOps.DEFINE_LOCAL, UOps.PHI, UOps.STORE, UOps.LOAD, UOps.ENDIF, UOps.ENDLOOP] and
          all([s>gi and s<ei for s in inces])):
        uops.uops.insert(gi, uops.uops.pop(uops.uops.index(u)))
        gi, ei = uops.uops.index(gate), uops.uops.index(end)
        queue.extend(list(u.vin))
      

  # for end in filter(lambda x: x.uop is UOps.ENDIF, list(reversed(uops.uops))):
  #   gate = end.vin[0]
  #   gi, ei = uops.uops.index(gate), uops.uops.index(end)
  #   for u in reversed(uops.uops[:gi].copy()):
  #     scc = successors(u)
  #     inces = [uops.uops.index(x) for x in scc]
  #     if (u.uop not in [UOps.DEFINE_GLOBAL, UOps.DEFINE_VAR, UOps.DEFINE_LOCAL, UOps.PHI, UOps.STORE, UOps.LOAD, UOps.ENDIF, UOps.ENDLOOP] and
  #         all([s>gi and s<ei for s in inces])):
  #       uops.uops.insert(gi, uops.uops.pop(uops.uops.index(u)))
  #       gi, ei = uops.uops.index(gate), uops.uops.index(end)

class AssemblyLanguage(NamedTuple):
  kernel_prefix: str = ""
  barrier: str = ""
  load_global: bool = False
  label_prefix: str = ""
  gid: List[str] = []
  gdim: List[str] = []
  lid: List[str] = []
  const_requires_mov: List[DType] = [] # list of dtypes for which creating a const requires a move
  asm_for_op: Dict[Op, Callable[...,str]] = {}
  types: Dict[DType, str] = INVERSE_DTYPES_DICT
  supports_half: List[Op] = []

  def render_const(self, x:ConstType, dtype:DType, mov=None) -> Union[List[str], str]: raise NotImplementedError()
  def render_local(self, dest, name, size, dtype) -> List[str]: raise NotImplementedError()

  def render_loop(self, idx, start, label, acc=None) -> List[str]: raise NotImplementedError()
  def render_bra(self, b1, pred=None) -> List[str]: raise NotImplementedError()
  def render_gep(self, loc, base, offset, dtype, gate=None) -> List[str]: raise NotImplementedError()
  def render_load(self, loc, dest, dtype, gate=None, alt=None, ss="", offset=0) -> List[str]: raise NotImplementedError()
  def render_store(self, loc, val, dtype, gate=None, ss="", offset=0) -> List[str]: raise NotImplementedError()
  def render_cast(self, d:str, a:str, dtype:DType, atype:DType, bitcast=False, pred=False) -> List[str]: raise NotImplementedError()

  def render_kernel(self, kernel, function_name, bufs, regs) -> str: raise NotImplementedError()
  def mem_type(self, dtype) -> str: raise NotImplementedError()

def uops_to_asm(lang:AssemblyLanguage, function_name:str, _uops:UOpGraph) -> str:
  # editing the uops breaks beam search
  uops = UOpGraph(_uops.uops.copy())
  kernel:List[str] = []
  bufs = []

  matcher = PatternMatcher([
    ({"__name__": "root", "uop": UOps.ALU, "arg": BinaryOps.CMPEQ, "vin": ({"dtype": dtypes.bool},{})},
     lambda root: UOp(UOps.ALU, dtypes.bool, (UOp(root.uop, root.dtype, root.vin, BinaryOps.XOR),), UnaryOps.NEG)),
    ({"__name__": "root", "uop": UOps.ALU, "arg": BinaryOps.CMPLT, "vin": ({"__name__": "x", "dtype": dtypes.bool},{"__name__": "y"})},
     lambda root,x,y: UOp(root.uop, root.dtype, (UOp(UOps.ALU, dtypes.bool, (x,), UnaryOps.NEG), y), BinaryOps.MUL)),
    ({"__name__": "root", "uop": UOps.ALU, "arg": BinaryOps.ADD, "dtype": set([dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64]),
      "vin": [{"__name__": "non_muls"}, {"__name__": "muls", "uop": UOps.ALU, "arg": BinaryOps.MUL}]},
      lambda root, muls, non_muls: UOp(UOps.ALU, root.dtype, muls.vin + (non_muls,), TernaryOps.MULACC)),
    *[({"__name__": "x", "uop": UOps.ALU, "dtype": dtypes.half, "arg": op},
       lambda x: UOp(UOps.CAST, dtypes.half, (UOp(x.uop, dtypes.float32, tuple([UOp(UOps.CAST, dtypes.float32, (vv,)) for vv in x.vin]), x.arg),)))
      for op in lang.asm_for_op.keys() if op not in lang.supports_half],
    ({"__name__": "root", "uop": UOps.LOAD, "dtype": dtypes.bool,
      "vin": ({"__name__": "x"},{"__name__": "y"},{"__name__": "z"},{"__name__": "k"})},
     lambda root,x,y,z,k: UOp(UOps.CAST, dtypes.bool, (UOp(root.uop, dtypes.int8, (x,y,z,UOp(UOps.CAST, dtypes.uint8, (k,)))),), root.arg)),
    ({"__name__": "root", "uop": UOps.LOAD,"dtype": dtypes.bool, "vin": ({},{})},
     lambda root: UOp(UOps.CAST, dtypes.bool, (UOp(root.uop, dtypes.uint8, root.vin, root.arg),))),
    ({"__name__": "root", "uop": UOps.STORE, "vin": ({},{},{"__name__": "z","dtype": dtypes.bool}, {})},
     lambda root,z: UOp(root.uop, root.dtype, root.vin[:2] + (UOp(UOps.CAST, dtypes.uint8, (z,), None),), root.arg)),
    ({"__name__": "root", "uop": UOps.STORE, "vin": ({},{},{"__name__": "z","dtype": dtypes.bool})},
     lambda root,z: UOp(root.uop, root.dtype, root.vin[:2] + (UOp(UOps.CAST, dtypes.uint8, (z,), None),), root.arg)),
    ({"__name__": "root", "uop": UOps.STORE, "vin": ({},{},{},{"__name__": "g"})},
     lambda root,g: UOp(root.uop, root.dtype, root.vin[:3] + (UOp(UOps.CAST, dtypes.bool, (g,), root.arg),))),
  ])

  # here we do a pretransform on UOps to fix some shortcomings of PTX
  # all uops must be a register
  matcher.rewrite_graph(uops)

  for pointer_op in list(filter(lambda uop: uop.uop in [UOps.LOAD, UOps.STORE], uops.uops)): ptr_ar(pointer_op, uops)
  uops.remove_childless(set(x for x in uops if x.uop in {UOps.DEFINE_GLOBAL, UOps.PHI, UOps.ENDIF, UOps.ENDLOOP, UOps.STORE}))
  uops.optimize_loops()
  # uops.print()
  optimize_gated_loads(uops)
  # uops.uops = uops.optimize_ordering(uops.uops)

  def kk(*s: str): kernel.append("\n".join(s))

  c: DefaultDict[str, int] = defaultdict(int)
  r: Dict[UOp, Union[List[str], str]] = {}
  def ssa(prefix:str, u:Optional[UOp]=None, dtype:Optional[str]=None) -> str:
    nonlocal c, r
    prefix += f"_{dtype if dtype is not None else lang.types[cast(DType, cast(UOp, u).dtype)]}_"
    c[prefix] += 1
    if u is not None: r[u] = f"%{prefix}{c[prefix]-1}"
    return f"%{prefix}{c[prefix]-1}"

  c_label: DefaultDict[str, int] = defaultdict(int)
  r_label: Dict[UOp, str] = {}
  def ssa_label(prefix:str, u:UOp):
    nonlocal c_label, r_label
    c_label[prefix] += 1
    r_label[u] = f"{lang.label_prefix}{prefix}_{c_label[prefix]-1}"
    return r_label[u]

  def const(x:ConstType, dtype:DType, mov=False):
    if mov or dtype in lang.const_requires_mov:
      kk(*lang.render_const(x, dtype, mov=(out:=ssa('const', dtype=lang.types[dtype]))))
      return out
    return lang.render_const(x, dtype)

  def _cast(a, dtype:DType, atype:DType, bitcast=False, u=None, pred=False):
    if atype == dtype:
      if u: r[u] = a
      return a
    kk(*lang.render_cast((ret:=ssa('cast', u, lang.types[dtype])), a, dtype, atype, bitcast))
    return ret

  for u in uops:
    uop,dtype,vin,args = u.uop,u.dtype,u.vin,u.arg
    if uop is UOps.IF:
      assert vin[0].dtype is not None
      kk(*lang.render_bra(ssa_label('if', u), _cast(r[vin[0]], dtypes.bool, vin[0].dtype, u=u, pred=True)))
    elif uop is UOps.BARRIER and lang.barrier: kk(lang.barrier)
    elif uop is UOps.ENDLOOP:
      kk(lang.asm_for_op[BinaryOps.ADD](r[vin[0]], r[vin[0]], "1", dtypes.int, lang.types[dtypes.int]),
          lang.asm_for_op[BinaryOps.CMPLT](pred:=ssa("pred", dtype="pred"), r[vin[0]], r[vin[0].vin[1]], dtypes.int, lang.types[dtypes.int]))
      kk(*lang.render_bra(r_label[vin[0]], pred))
    elif uop == UOps.ENDIF:
      kk(f"@!{_cast(r[vin[0].vin[0]], dtypes.bool, vin[0].vin[0].dtype, u=u, pred=True)} bra {r_label[vin[0]]}_true;")
      kk(f"{r_label[vin[0]]}:")
      if args:
        if args[0].dtype.count > 1:
          kk(*[f"mov.b{lang.types[args[0].dtype.scalar()][1:]} {dd}, {r[args[1]][i]};" for i, dd in enumerate(r[args[0]])])
        else:
          kk(*[f"mov.b{lang.types[args[0].dtype][1:]} {r[args[0]]}, {r[args[1]]};" ])
      kk(f"{r_label[vin[0]]}_true:")
    elif uop is UOps.STORE:
      assert vin[0].dtype is not None and vin[1].dtype is not None and vin[2].dtype is not None
      if vin[2].dtype.count > 1:
        kk((f"@{r[vin[3]]} " if len(vin)>3 else "") + \
            f"st{u.arg}.v{vin[2].dtype.count}.{lang.mem_type(vin[2].dtype.scalar())} [{r[vin[0]]}+{vin[1].arg}], {{{', '.join(r[vin[2]])}}};")
      else:
        kk(*lang.render_store(r[vin[0]], r[vin[2]], vin[2].dtype, gate=r[vin[3]] if len(vin)>3 else None, ss=u.arg, offset=vin[1].arg))
    else:
      assert dtype is not None, f"None dtype for uop {uop}"
      if uop is UOps.LOOP: kk(*lang.render_loop(ssa('ridx', u), r[vin[0]], ssa_label('loop', u)))
      elif uop is UOps.ALU:
        assert vin[0].dtype is not None
        if args is BinaryOps.CMPLT or args is BinaryOps.CMPEQ:
          # pass in the other dtype here
          kk(lang.asm_for_op[args](ssa("alu", u), *[r[x] for x in vin], vin[0].dtype, lang.types[vin[0].dtype]))
        else:
          kk(lang.asm_for_op[args](ssa("alu", u), *[r[x] for x in vin], dtype, lang.types[dtype]))
      elif uop is UOps.DEFINE_ACC:
        if dtype.count > 1:
          r[u] = [ssa('acc', dtype=lang.types[dtype.scalar()]) for _ in range(dtype.count)]
          for uu in r[u]: kk(f"mov.b{lang.types[dtype.scalar()][1:]} {uu}, {const(args, dtype.scalar())};")
        else: kk(f"mov.b{lang.types[dtype][1:]} {ssa('acc', u)}, {const(args, dtype)};")
      elif uop is UOps.SPECIAL:
        assert args[1][0] != "i", "idx not supported"
        kk(f"mov.u32 %{args[1]}, {(lang.gid if args[1][0] == 'g' else lang.lid)[args[0]]};")
        r[u] = "%" + args[1]
        kernel = [f".reg .u32 %{args[1]};"] + kernel
      elif uop is UOps.CONST:
        if dtype.count > 1: r[u] = [const(args, dtype.scalar(), mov=True) for _ in range(dtype.count)]
        else: r[u] = const(args, dtype, mov=True)
      elif uop is UOps.GEP: r[u] = r[vin[0]][u.arg]
      elif uop is UOps.LOAD:
        assert vin[1].dtype is not None
        if dtype.count > 1:
          r[u] = [ssa('val', dtype=lang.types[dtype.scalar()]) for _ in range(dtype.count)]
          kk(f"ld{u.arg}.v{dtype.count}.{lang.mem_type(dtype.scalar())} {{{', '.join(r[u])}}}, [{r[vin[0]]}+{vin[1].arg}];")
        else:
          kk(*lang.render_load(r[vin[0]], ssa('val', u), dtype, ss=u.arg, offset=vin[1].arg))
      elif uop is UOps.PHI:
        kk(f"mov.b{lang.types[dtype][1:]} {r[vin[0]]}, {r[vin[1]]};")
        r[u] = r[vin[0]]
      elif uop in {UOps.CAST, UOps.BITCAST}:
        assert vin[0].dtype is not None
        if dtype.count>1: r[u] = [r[x] for x in vin] # type: ignore
        else: _cast(r[vin[0]], dtype, vin[0].dtype, bitcast=uop is UOps.BITCAST, u=u)
      elif uop is UOps.DEFINE_LOCAL:
        # TODO: we should sum these, and fetch 0xC000 from somewhere
        assert args[1]*dtype.itemsize <= 0xC000, "too large local"
        kk(*lang.render_local(ssa('local', u, lang.types[dtypes.ulong]), args[0], args[1], dtype))
      elif uop is UOps.DEFINE_VAR:
        bufs.append((args.expr, dtype))
        r[u] = f"%{args.expr}"
        if lang.load_global: kk(*lang.render_load(args.expr, ssa('dat', u, lang.types[dtype]), dtype, ss=".param"))
      elif uop is UOps.DEFINE_GLOBAL:
        bufs.append((args[1], dtype))
        r[u] = f"%{args[1]}"
        if lang.load_global:
          dt = dtypes.ulong if dtype.__class__ == PtrDType else dtype
          kk(*lang.render_load(args[1], ssa('dat', u, lang.types[dt]), dt, ss=".param"))
      elif uop is UOps.WMMA:
        wmma = []
        for vv in vin[:2]:
          for i in range(0, len(r[vv]), 2):
            wmma.append(ssa("wmma", dtype="b32"))
            kk(f'mov.b32 {wmma[-1]}, {{{", ".join(r[vv][i:i+2])}}};')
        r[u] = r[vin[2]]
        kk(f'mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\
           {{{", ".join(r[u])}}}, {{{", ".join(wmma[:4])}}}, {{{", ".join(wmma[4:])}}}, {{{", ".join(r[u])}}};')
      else: raise NotImplementedError(f"no code for {uop}")

  return lang.render_kernel(kernel, function_name, bufs, c.items())

class PTXLanguage(AssemblyLanguage):
  kernel_prefix = """.version VERSION
.target TARGET
.address_size 64
.visible .entry"""
  barrier = "bar.sync\t0;"
  has_pred = True
  load_global = True
  label_prefix = "$"
  gid = [f'%ctaid.{chr(120+i)}' for i in range(3)]
  gdim = [f'%nctaid.{chr(120+i)}' for i in range(3)]
  lid = [f'%tid.{chr(120+i)}' for i in range(3)]
  asm_for_op = {
    UnaryOps.NEG: lambda d,a,dt,name: f"not.pred {d}, {a};" if name == "pred" else f"neg.{name} {d}, {a};",
    UnaryOps.EXP2: lambda d,a,dt,name: f"ex2.approx.{name} {d}, {a};", UnaryOps.LOG2: lambda d,a,dt,name: f"lg2.approx.{name} {d}, {a};",
    UnaryOps.SIN: lambda d,a,dt,name: f"sin.approx.{name} {d}, {a};", UnaryOps.SQRT: lambda d,a,dt,name: f"sqrt.approx.{name} {d}, {a};",
    BinaryOps.ADD: lambda d,a,b,dt,name: f"{'or' if name == 'pred' else 'add'}.{name} {d}, {a}, {b};",
    BinaryOps.SUB: lambda d,a,b,dt,name: f"sub.{name} {d}, {a}, {b};",
    BinaryOps.MUL: lambda d,a,b,dt,name: ('and' if dt == dtypes.bool else 'mul') + f"{'.lo' if dtypes.is_int(dt) else ''}.{name} {d}, {a}, {b};",
    BinaryOps.XOR: lambda d,a,b,dt,name: f"xor.pred {d}, {a}, {b};" if name == "pred" else f"xor.b{name[1:]} {d}, {a}, {b};",
    BinaryOps.DIV: lambda d,a,b,dt,name: f"div{'.approx' if dtypes.is_float(dt) else ''}.{name} {d}, {a}, {b};",
    BinaryOps.MAX: lambda d,a,b,dt,name: f"max.{name} {d}, {a}, {b};", BinaryOps.MOD: lambda d,a,b,dt,name: f"rem.{name} {d}, {a}, {b};",
    BinaryOps.CMPLT: lambda d,a,b,dt,name: f"setp.lt.{name} {d}, {a}, {b};",
    BinaryOps.CMPEQ: lambda d,a,b,dt,name: f"setp.eq.{name} {d}, {a}, {b};",
    TernaryOps.MULACC: lambda d,a,b,c,dt,name: f"fma.rn.{name} {d}, {a}, {b}, {c};",
    TernaryOps.WHERE: lambda d,a,b,c,dt,name:
      f"@{a} mov.{name} {d}, {b};\n@!{a} mov.{name} {d}, {c};" if name == "pred" else f"selp.{'b16' if name == 'f16' else name} {d}, {b}, {c}, {a};"
  }
  supports_half = [UnaryOps.NEG, UnaryOps.EXP2, BinaryOps.ADD, BinaryOps.SUB, BinaryOps.MUL, BinaryOps.MAX, BinaryOps.CMPLT, TernaryOps.WHERE]
  # HACK: Use s16 and u16 for int8 and uint8 buffers. This can be wrong in cast.
  types = { dtypes.int8: "s16", dtypes.int16: "s16", dtypes.int32: "s32", dtypes.int64: "s64",
            dtypes.uint8: "u16", dtypes.uint16: "u16", dtypes.uint32: "u32", dtypes.uint64: "u64",
            dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64", dtypes.bool: "pred" }

  const_requires_mov = [dtypes.half, dtypes.bool]

  def render_const(self, x:ConstType, dtype:DType, mov=None) -> Union[List[str], str]:
    val = render_val(x, dtype)
    if dtype == dtypes.bool: return [f"setp.ne.s16 {mov}, {val}, 0;"]
    return [f"mov.b{self.types[dtype][1:]} {mov}, {val};"] if mov else val

  def render_local(self, dest, name, size, dtype) -> List[str]:
    return [f".shared .align 4 .b8 {name}[{size*dtype.itemsize}];", f"mov.u64 {dest}, {name}[0];"]

  def render_loop(self, idx, start, label, acc=None) -> List[str]: return [f"mov.u32 {idx}, {start};", f"{label}:"]

  def render_bra(self, b1, pred=None) -> List[str]: return [f"@{pred} bra {b1};"] if pred else [f"bra {b1};"]

  def mem_type(self, dtype): return 's8' if dtype.itemsize == 1 else 'b16' if dtype == dtypes.float16 else self.types[dtype]

  def render_load(self, loc, dest, dtype, gate=None, alt=None, ss="", offset=0) -> List[str]:
    assert dtype is not dtypes.bool
    return [f"ld{ss}.{self.mem_type(dtype)} {dest}, [{loc}+{offset}];"]

  def render_store(self, loc, val, dtype, gate=None, ss="", offset=0) -> List[str]:
    return [(f"@{gate} " if gate else "") + f"st{ss}.{self.mem_type(dtype)} [{loc}+{offset}], {val};"]

  def render_cast(self, d:str, a:str, dtype:DType, atype:DType, bitcast=False, pred=False) -> List[str]:
    if bitcast: return [f"mov.b{self.types[dtype][1:]} {d}, {a};"]
    if atype == dtypes.bool: return[f"selp.b{self.types[dtype][1:]} {d}, {render_val(1, dtype)}, {render_val(0, dtype)}, {a};"]
    if dtype == dtypes.bool: return [f"setp.ne.b{self.types[atype][1:]} {d}, {a}, {self.render_const(0, atype)};"]
    rnd = ('.rzi' if dtypes.is_int(dtype) and dtypes.is_float(atype) else
           '.rn' if dtypes.is_float(dtype) and (dtype.itemsize < atype.itemsize or dtypes.is_int(atype) or atype == dtypes.bool) else '')
    return [f"cvt{rnd}.{self.types[dtype]}.{self.types[atype]} {d}, {a};"]

  def render_kernel(self, kernel, function_name, bufs, regs) -> str:
    if False:
      return""".version 7.5
.target sm_86
.address_size 64
.visible .entry r_11_8_16_3_2_12966_3_3_4n1(
        .param .u64 data0,
        .param .u64 data1,
        .param .u64 data2,
        .param .u64 data3
)
{
        .reg            .u64 %dat_u64_<4>;
        .reg            .f32 %acc_f32_<9>;
        .reg            .s32 %const_s32_<29>;
        .reg            .s32 %alu_s32_<23>;
        .reg            .u64 %cast_u64_<4>;
        .reg            .u64 %alu_u64_<4>;
        .reg            .f16 %val_f16_<33>;
        .reg            .s32 %ridx_s32_<1>;
        .reg            .f16 %alu_f16_<45>;
        .reg            .f32 %cast_f32_<36>;
        .reg            .f32 %alu_f32_<36>;
        .reg            .pred %pred_pred_<1>;
        .reg            .f16 %cast_f16_<9>;
        .reg            .u32 %lidx4;
        .reg            .u32 %lidx3;
        .reg            .u32 %lidx2;
        .reg            .u32 %gidx1;
        .reg            .u32 %gidx0;
        ld.param.u64    %dat_u64_0, [data0+0];
        ld.param.u64    %dat_u64_1, [data1+0];
        ld.param.u64    %dat_u64_2, [data2+0];
        ld.param.u64    %dat_u64_3, [data3+0];
        mov.u32         %gidx0, %ctaid.y;
        mov.u32         %gidx1, %ctaid.x;
        mov.u32         %lidx2, %tid.z;
        mov.u32         %lidx3, %tid.y;
        mov.u32         %lidx4, %tid.x;
        mov.b32         %acc_f32_0, 0f00000000;
        mov.b32         %acc_f32_1, 0f00000000;
        mov.b32         %acc_f32_2, 0f00000000;
        mov.b32         %acc_f32_3, 0f00000000;
        mov.b32         %acc_f32_4, 0f00000000;
        mov.b32         %acc_f32_5, 0f00000000;
        mov.b32         %acc_f32_6, 0f00000000;
        mov.b32         %acc_f32_7, 0f00000000;
        mov.b32         %acc_f32_8, 0f00000000;
        mov.b32         %const_s32_0, 0;
        mov.b32         %const_s32_1, 12966;
        mov.b32         %const_s32_2, 466776;
        mul.lo.s32      %alu_s32_0, %gidx0, %const_s32_2;
        mov.b32         %const_s32_3, 155592;
        mul.lo.s32      %alu_s32_1, %lidx3, %const_s32_3;
        add.s32         %alu_s32_2, %alu_s32_0, %alu_s32_1;
        mov.b32         %const_s32_4, 5134536;
        mul.lo.s32      %alu_s32_3, %lidx4, %const_s32_4;
        add.s32         %alu_s32_4, %alu_s32_2, %alu_s32_3;
        mov.b32         %const_s32_5, 4;
        mov.b32         %const_s32_6, 48;
        mul.lo.s32      %alu_s32_5, %gidx1, %const_s32_6;
        mov.b32         %const_s32_7, 3;
        mul.lo.s32      %alu_s32_6, %lidx2, %const_s32_7;
        add.s32         %alu_s32_7, %alu_s32_5, %alu_s32_6;
        mov.b32         %const_s32_8, 1536;
        mov.b32         %const_s32_9, 1152;
        mov.b32         %const_s32_10, 3456;
        mul.lo.s32      %alu_s32_8, %gidx0, %const_s32_10;
        add.s32         %alu_s32_9, %alu_s32_8, %alu_s32_5;
        add.s32         %alu_s32_10, %alu_s32_9, %alu_s32_6;
        mul.lo.s32      %alu_s32_11, %lidx3, %const_s32_9;
        add.s32         %alu_s32_12, %alu_s32_10, %alu_s32_11;
        mov.b32         %const_s32_11, 38016;
        mul.lo.s32      %alu_s32_13, %lidx4, %const_s32_11;
        add.s32         %alu_s32_14, %alu_s32_12, %alu_s32_13;
        mov.b32         %const_s32_12, 2;
        mul.lo.s32      %alu_s32_15, %alu_s32_12, %const_s32_12;
        mov.b32         %const_s32_13, 0;
        cvt.u64.s32     %cast_u64_0, %alu_s32_15;
        add.u64         %alu_u64_0, %dat_u64_3, %cast_u64_0;
        ld.global.b16   %val_f16_0, [%alu_u64_0+0];
        ld.global.b16   %val_f16_1, [%alu_u64_0+2];
        mov.b32         %const_s32_14, 4;
        ld.global.b16   %val_f16_2, [%alu_u64_0+4];
        mov.b32         %const_s32_15, 768;
        ld.global.b16   %val_f16_3, [%alu_u64_0+768];
        mov.b32         %const_s32_16, 770;
        ld.global.b16   %val_f16_4, [%alu_u64_0+770];
        mov.b32         %const_s32_17, 772;
        ld.global.b16   %val_f16_5, [%alu_u64_0+772];
        mov.b32         %const_s32_18, 1536;
        ld.global.b16   %val_f16_6, [%alu_u64_0+1536];
        mov.b32         %const_s32_19, 1538;
        ld.global.b16   %val_f16_7, [%alu_u64_0+1538];
        mov.b32         %const_s32_20, 1540;
        ld.global.b16   %val_f16_8, [%alu_u64_0+1540];
        mov.b32         %const_s32_21, 0;
        mov.b32         %const_s32_22, 103728;
        mov.b32         %const_s32_23, 207456;
        mov.b32         %const_s32_24, 0;
        mov.b32         %const_s32_25, 2304;
        mov.b32         %const_s32_26, 2306;
        mov.b32         %const_s32_27, 2308;
        mul.lo.s32      %alu_s32_16, %alu_s32_14, %const_s32_12;
        mov.b32         %const_s32_28, 0;
        cvt.u64.s32     %cast_u64_1, %alu_s32_16;
        add.u64         %alu_u64_1, %dat_u64_0, %cast_u64_1;
        mov.u32         %ridx_s32_0, %const_s32_0;
$loop_0:
        mul.lo.s32      %alu_s32_17, %ridx_s32_0, %const_s32_5;
        add.s32         %alu_s32_18, %alu_s32_4, %alu_s32_17;
        mul.lo.s32      %alu_s32_19, %alu_s32_18, %const_s32_12;
        cvt.u64.s32     %cast_u64_2, %alu_s32_19;
        add.u64         %alu_u64_2, %dat_u64_1, %cast_u64_2;
        ld.global.b16        %val_f16_9, [%alu_u64_2+0];
        ld.global.b16        %val_f16_10, [%alu_u64_2+2];
        ld.global.b16        %val_f16_11, [%alu_u64_2+4];
        ld.global.b16        %val_f16_12, [%alu_u64_2+6];
        ld.global.b16        %val_f16_13, [%alu_u64_2+103728];
        ld.global.b16        %val_f16_14, [%alu_u64_2+103730];
        ld.global.b16        %val_f16_15, [%alu_u64_2+103732];
        ld.global.b16        %val_f16_16, [%alu_u64_2+103734];
        ld.global.b16        %val_f16_17, [%alu_u64_2+207456];
        ld.global.b16        %val_f16_18, [%alu_u64_2+207458];
        ld.global.b16        %val_f16_19, [%alu_u64_2+207460];
        ld.global.b16        %val_f16_20, [%alu_u64_2+207462];
        mul.lo.s32      %alu_s32_20, %ridx_s32_0, %const_s32_8;
        add.s32         %alu_s32_21, %alu_s32_7, %alu_s32_20;
        mul.lo.s32      %alu_s32_22, %alu_s32_21, %const_s32_12;
        cvt.u64.s32     %cast_u64_3, %alu_s32_22;
        add.u64         %alu_u64_3, %dat_u64_2, %cast_u64_3;
        ld.global.b16   %val_f16_21, [%alu_u64_3+0];
        ld.global.b16   %val_f16_22, [%alu_u64_3+2];
        ld.global.b16   %val_f16_23, [%alu_u64_3+4];
        ld.global.b16   %val_f16_24, [%alu_u64_3+768];
        ld.global.b16   %val_f16_25, [%alu_u64_3+770];
        ld.global.b16   %val_f16_26, [%alu_u64_3+772];
        ld.global.b16   %val_f16_27, [%alu_u64_3+1536];
        ld.global.b16   %val_f16_28, [%alu_u64_3+1538];
        ld.global.b16   %val_f16_29, [%alu_u64_3+1540];
        ld.global.b16   %val_f16_30, [%alu_u64_3+2304];
        ld.global.b16   %val_f16_31, [%alu_u64_3+2306];
        ld.global.b16   %val_f16_32, [%alu_u64_3+2308];

        mul.f16         %alu_f16_0, %val_f16_9, %val_f16_21;
        cvt.f32.f16     %cast_f32_0, %alu_f16_0;
        add.f32         %alu_f32_0, %cast_f32_9, %cast_f32_0;

        mul.f16         %alu_f16_1, %val_f16_9, %val_f16_22;
        cvt.f32.f16     %cast_f32_1, %alu_f16_1;
        add.f32         %alu_f32_1, %cast_f32_10, %cast_f32_1;

        mul.f16         %alu_f16_2, %val_f16_9, %val_f16_23;
        cvt.f32.f16     %cast_f32_2, %alu_f16_2;
        add.f32         %alu_f32_2, %cast_f32_11, %cast_f32_2;

        mul.f16         %alu_f16_3, %val_f16_13, %val_f16_21;
        cvt.f32.f16     %cast_f32_3, %alu_f16_3;
        add.f32         %alu_f32_3, %cast_f32_12, %cast_f32_3;

        mul.f16         %alu_f16_4, %val_f16_13, %val_f16_22;
        cvt.f32.f16     %cast_f32_4, %alu_f16_4;
        add.f32         %alu_f32_4, %cast_f32_13, %cast_f32_4;

        mul.f16         %alu_f16_5, %val_f16_13, %val_f16_23;
        cvt.f32.f16     %cast_f32_5, %alu_f16_5;
        add.f32         %alu_f32_5, %cast_f32_14, %cast_f32_5;

        mul.f16         %alu_f16_6, %val_f16_17, %val_f16_21;
        cvt.f32.f16     %cast_f32_6, %alu_f16_6;
        add.f32         %alu_f32_6, %cast_f32_15, %cast_f32_6;

        mul.f16         %alu_f16_7, %val_f16_17, %val_f16_22;
        cvt.f32.f16     %cast_f32_7, %alu_f16_7;
        add.f32         %alu_f32_7, %cast_f32_16, %cast_f32_7;

        mul.f16         %alu_f16_8, %val_f16_17, %val_f16_23;
        cvt.f32.f16     %cast_f32_8, %alu_f16_8;
        add.f32         %alu_f32_8, %cast_f32_17, %cast_f32_8;

        mul.f16         %alu_f16_9, %val_f16_10, %val_f16_24;
        cvt.f32.f16     %cast_f32_9, %alu_f16_9;
        add.f32         %alu_f32_9, %cast_f32_18, %alu_f32_0;

        mul.f16         %alu_f16_10, %val_f16_10, %val_f16_25;
        cvt.f32.f16     %cast_f32_10, %alu_f16_10;
        add.f32         %alu_f32_10, %cast_f32_19, %alu_f32_1;

        mul.f16         %alu_f16_11, %val_f16_10, %val_f16_26;
        cvt.f32.f16     %cast_f32_11, %alu_f16_11;
        add.f32         %alu_f32_11, %cast_f32_20, %alu_f32_2;

        mul.f16         %alu_f16_12, %val_f16_14, %val_f16_24;
        cvt.f32.f16     %cast_f32_12, %alu_f16_12;
        add.f32         %alu_f32_12, %cast_f32_21, %alu_f32_3;

        mul.f16         %alu_f16_13, %val_f16_14, %val_f16_25;
        cvt.f32.f16     %cast_f32_13, %alu_f16_13;
        add.f32         %alu_f32_13, %cast_f32_22, %alu_f32_4;

        mul.f16         %alu_f16_14, %val_f16_14, %val_f16_26;
        cvt.f32.f16     %cast_f32_14, %alu_f16_14;
        add.f32         %alu_f32_14, %cast_f32_23, %alu_f32_5;

        mul.f16         %alu_f16_15, %val_f16_18, %val_f16_24;
        cvt.f32.f16     %cast_f32_15, %alu_f16_15;
        add.f32         %alu_f32_15, %cast_f32_24, %alu_f32_6;

        mul.f16         %alu_f16_16, %val_f16_18, %val_f16_25;
        cvt.f32.f16     %cast_f32_16, %alu_f16_16;
        add.f32         %alu_f32_16, %cast_f32_25, %alu_f32_7;

        mul.f16         %alu_f16_17, %val_f16_18, %val_f16_26;
        cvt.f32.f16     %cast_f32_17, %alu_f16_17;
        add.f32         %alu_f32_17, %cast_f32_26, %alu_f32_8;

        mul.f16         %alu_f16_18, %val_f16_11, %val_f16_27;
        cvt.f32.f16     %cast_f32_18, %alu_f16_18;
        add.f32         %alu_f32_18, %cast_f32_27, %alu_f32_9;

        mul.f16         %alu_f16_19, %val_f16_11, %val_f16_28;
        cvt.f32.f16     %cast_f32_19, %alu_f16_19;
        add.f32         %alu_f32_19, %cast_f32_28, %alu_f32_10;

        mul.f16         %alu_f16_20, %val_f16_11, %val_f16_29;
        cvt.f32.f16     %cast_f32_20, %alu_f16_20;
        add.f32         %alu_f32_20, %cast_f32_29, %alu_f32_11;

        mul.f16         %alu_f16_21, %val_f16_15, %val_f16_27;
        cvt.f32.f16     %cast_f32_21, %alu_f16_21;
        add.f32         %alu_f32_21, %cast_f32_30, %alu_f32_12;

        mul.f16         %alu_f16_22, %val_f16_15, %val_f16_28;
        cvt.f32.f16     %cast_f32_22, %alu_f16_22;
        add.f32         %alu_f32_22, %cast_f32_31, %alu_f32_13;

        mul.f16         %alu_f16_23, %val_f16_15, %val_f16_29;
        cvt.f32.f16     %cast_f32_23, %alu_f16_23;
        add.f32         %alu_f32_23, %cast_f32_32, %alu_f32_14;

        mul.f16         %alu_f16_24, %val_f16_19, %val_f16_27;
        cvt.f32.f16     %cast_f32_24, %alu_f16_24;
        add.f32         %alu_f32_24, %cast_f32_33, %alu_f32_15;

        mul.f16         %alu_f16_25, %val_f16_19, %val_f16_28;
        cvt.f32.f16     %cast_f32_25, %alu_f16_25;
        add.f32         %alu_f32_25, %cast_f32_34, %alu_f32_16;

        mul.f16         %alu_f16_26, %val_f16_19, %val_f16_29;
        cvt.f32.f16     %cast_f32_26, %alu_f16_26;
        add.f32         %alu_f32_26, %cast_f32_35, %alu_f32_17;

        mul.f16         %alu_f16_27, %val_f16_12, %val_f16_30;
        cvt.f32.f16     %cast_f32_27, %alu_f16_27;
        add.f32         %alu_f32_27, %alu_f32_18, %acc_f32_0;

        mul.f16         %alu_f16_28, %val_f16_12, %val_f16_31;
        cvt.f32.f16     %cast_f32_28, %alu_f16_28;
        add.f32         %alu_f32_28, %alu_f32_19, %acc_f32_1;

        mul.f16         %alu_f16_29, %val_f16_12, %val_f16_32;
        cvt.f32.f16     %cast_f32_29, %alu_f16_29;
        add.f32         %alu_f32_29, %alu_f32_20, %acc_f32_2;

        mul.f16         %alu_f16_30, %val_f16_16, %val_f16_30;
        cvt.f32.f16     %cast_f32_30, %alu_f16_30;
        add.f32         %alu_f32_30, %alu_f32_21, %acc_f32_3;

        mul.f16         %alu_f16_31, %val_f16_16, %val_f16_31;
        cvt.f32.f16     %cast_f32_31, %alu_f16_31;
        add.f32         %alu_f32_31, %alu_f32_22, %acc_f32_4;

        mul.f16         %alu_f16_32, %val_f16_16, %val_f16_32;
        cvt.f32.f16     %cast_f32_32, %alu_f16_32;
        add.f32         %alu_f32_32, %alu_f32_23, %acc_f32_5;

        mul.f16         %alu_f16_33, %val_f16_20, %val_f16_30;
        cvt.f32.f16     %cast_f32_33, %alu_f16_33;
        add.f32         %alu_f32_33, %alu_f32_24, %acc_f32_6;

        mul.f16         %alu_f16_34, %val_f16_20, %val_f16_31;
        cvt.f32.f16     %cast_f32_34, %alu_f16_34;
        mul.f16         %alu_f16_35, %val_f16_20, %val_f16_32;
        cvt.f32.f16     %cast_f32_35, %alu_f16_35;

        mov.b32         %acc_f32_0, %alu_f32_27;
        mov.b32         %acc_f32_1, %alu_f32_28;
        mov.b32         %acc_f32_2, %alu_f32_29;
        mov.b32         %acc_f32_3, %alu_f32_30;
        mov.b32         %acc_f32_4, %alu_f32_31;
        mov.b32         %acc_f32_5, %alu_f32_32;
        mov.b32         %acc_f32_6, %alu_f32_33;
        add.f32         %alu_f32_34, %alu_f32_25, %acc_f32_7;
        mov.b32         %acc_f32_7, %alu_f32_34;
        add.f32         %alu_f32_35, %alu_f32_26, %acc_f32_8;
        mov.b32         %acc_f32_8, %alu_f32_35;
        add.s32         %ridx_s32_0, %ridx_s32_0, 1;
        setp.lt.s32     %pred_pred_0, %ridx_s32_0, %const_s32_1;
        @%pred_pred_0   bra $loop_0;
        cvt.rn.f16.f32  %cast_f16_0, %acc_f32_0;
        cvt.rn.f16.f32  %cast_f16_1, %acc_f32_1;
        cvt.rn.f16.f32  %cast_f16_2, %acc_f32_2;
        cvt.rn.f16.f32  %cast_f16_3, %acc_f32_3;
        cvt.rn.f16.f32  %cast_f16_4, %acc_f32_4;
        cvt.rn.f16.f32  %cast_f16_5, %acc_f32_5;
        cvt.rn.f16.f32  %cast_f16_6, %acc_f32_6;
        cvt.rn.f16.f32  %cast_f16_7, %acc_f32_7;
        cvt.rn.f16.f32  %cast_f16_8, %acc_f32_8;
        add.f16         %alu_f16_36, %cast_f16_0, %val_f16_0;
        add.f16         %alu_f16_37, %cast_f16_1, %val_f16_1;
        add.f16         %alu_f16_38, %cast_f16_2, %val_f16_2;
        add.f16         %alu_f16_39, %cast_f16_3, %val_f16_3;
        add.f16         %alu_f16_40, %cast_f16_4, %val_f16_4;
        add.f16         %alu_f16_41, %cast_f16_5, %val_f16_5;
        add.f16         %alu_f16_42, %cast_f16_6, %val_f16_6;
        add.f16         %alu_f16_43, %cast_f16_7, %val_f16_7;
        add.f16         %alu_f16_44, %cast_f16_8, %val_f16_8;
        st.global.b16   [%alu_u64_1+0], %alu_f16_36;
        st.global.b16   [%alu_u64_1+2], %alu_f16_37;
        st.global.b16   [%alu_u64_1+4], %alu_f16_38;
        st.global.b16   [%alu_u64_1+768], %alu_f16_39;
        st.global.b16   [%alu_u64_1+770], %alu_f16_40;
        st.global.b16   [%alu_u64_1+772], %alu_f16_41;
        st.global.b16   [%alu_u64_1+1536], %alu_f16_42;
        st.global.b16   [%alu_u64_1+1538], %alu_f16_43;
        st.global.b16   [%alu_u64_1+1540], %alu_f16_44;
        ret;
}
"""
    kernel = [f".reg .{reg.split('_')[-2]} %{reg}<{cnt}>;" for reg,cnt in regs] + kernel + ["ret;"]
    def fmt(line): return line if line[0]=="$" else "\t" + line.replace(" ", "\t" if len(line.split(" ")[0]) > 7 else "\t\t", 1)
    return (f"{self.kernel_prefix} {function_name}(\n\t" +
            ',\n\t'.join([f".param .{'u64' if dtype.__class__ == PtrDType else self.types[dtype]} {name}" for name,dtype in bufs]) + "\n)\n{\n" +
            '\n'.join([fmt(line) for op in kernel for line in op.splitlines()]) +
            "\n}")

PTXRenderer = functools.partial(uops_to_asm, PTXLanguage())
