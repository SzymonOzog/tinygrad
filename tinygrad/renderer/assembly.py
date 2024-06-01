from typing import DefaultDict, Dict, List, Union, Optional, cast, Callable
import struct, math
from collections import defaultdict
from tinygrad.helpers import DEBUG
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps, Op
from tinygrad.dtype import dtypes, DType, PtrDType, ConstType
from tinygrad.codegen.uops import UOpGraph, PatternMatcher
from tinygrad.renderer import Renderer, TensorCore

def render_val(x, dtype):
  if dtypes.is_float(dtype):
    if dtype == dtypes.double: return "0d%02X%02X%02X%02X%02X%02X%02X%02X" % tuple(struct.pack("d",x)[::-1])
    if dtype == dtypes.half: return "0x%02X%02X" % tuple(struct.pack("e",x)[::-1])
    return "0f%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
  return str(int(x)) + ("U" if dtypes.is_unsigned(dtype) else "")

def fast_remainder(d,a,b,dt,name):
  if dt == dtypes.int and b.isnumeric() and math.log2(int(b)).is_integer():
    return f"""shr.u32  %rem0, {a}, 31;                                                         
add.s32  %rem1, {a}, %rem0;                                                       
and.b32   %rem2, %rem1, -{int(b)};                                                        
sub.s32  {d}, {a}, %rem2;"""
  return f"rem.{name} {d}, {a}, {b};"


class PTXRenderer(Renderer):
  device = "CUDA"
  suffix = "PTX"
  global_max = [65535, 65535, 2147483647]
  local_max = [64, 1024, 1024]
  shared_max = 49152
  tensor_cores = [TensorCore(dims=(8,16,16), threads=[(0,2),(0,2),(1,2),(1,2),(0,2)], thread_local_sizes=[[2,2,2],[2,2],[2,2]], thread_local_aliases=[ [[0],[0],[5],[-2],[0],[-1,1,2,-3],[3,4]], [[3],[4],[0],[0],[5],[-1,1,2,-2],[0]], [[-1],[1],[5],[-2],[2],[0],[3,4]] ], dtype_in=di, dtype_out=do) for (di, do) in ([(dtypes.half, dtypes.float)])] # noqa: E501
  def __init__(self, arch:str): self.tensor_cores = PTXRenderer.tensor_cores if int(arch[3:]) >= 80 else []

  # language options
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
  asm_for_op: Dict[Op, Callable] = {
    UnaryOps.NEG: lambda d,a,dt,name: f"not.pred {d}, {a};" if name == "pred" else f"neg.{name} {d}, {a};",
    UnaryOps.EXP2: lambda d,a,dt,name: f"ex2.approx.{name} {d}, {a};", UnaryOps.LOG2: lambda d,a,dt,name: f"lg2.approx.{name} {d}, {a};",
    UnaryOps.SIN: lambda d,a,dt,name: f"sin.approx.{name} {d}, {a};", UnaryOps.SQRT: lambda d,a,dt,name: f"sqrt.approx.{name} {d}, {a};",
    BinaryOps.SHR: lambda d,a,b,dt,name: f"shr.{name} {d}, {a}, {b};", BinaryOps.SHL: lambda d,a,b,dt,name: f"shl.b{name[1:]} {d}, {a}, {b};",
    BinaryOps.ADD: lambda d,a,b,dt,name: f"{'or' if name == 'pred' else 'add'}.{name} {d}, {a}, {b};",
    BinaryOps.SUB: lambda d,a,b,dt,name: f"sub.{name} {d}, {a}, {b};",
    BinaryOps.MUL: lambda d,a,b,dt,name: ('and' if dt == dtypes.bool else 'mul') + f"{'.lo' if dtypes.is_int(dt) else ''}.{name} {d}, {a}, {b};",
    BinaryOps.XOR: lambda d,a,b,dt,name: f"xor.pred {d}, {a}, {b};" if name == "pred" else f"xor.b{name[1:]} {d}, {a}, {b};",
    BinaryOps.DIV: lambda d,a,b,dt,name: f"div{'.approx' if dtypes.is_float(dt) else ''}.{name} {d}, {a}, {b};",
    BinaryOps.MAX: lambda d,a,b,dt,name: f"max.{name} {d}, {a}, {b};", BinaryOps.MOD: fast_remainder,
    BinaryOps.CMPLT: lambda d,a,b,dt,name: f"setp.lt.{name} {d}, {a}, {b};",
    BinaryOps.CMPEQ: lambda d,a,b,dt,name: f"setp.eq.{name} {d}, {a}, {b};",
    TernaryOps.MULACC: lambda d,a,b,c,dt,name: f"{'fma.rn' if dtypes.is_float(dt) else 'mad.lo'}.{name} {d}, {a}, {b}, {c};",
    TernaryOps.WHERE: lambda d,a,b,c,dt,name:
      f"@{a} mov.{name} {d}, {b};\n@!{a} mov.{name} {d}, {c};" if name == "pred" else f"selp.{'b16' if name == 'f16' else name} {d}, {b}, {c}, {a};"
  }
  supports_half: List[Op] = [UnaryOps.NEG, UnaryOps.EXP2, BinaryOps.ADD, BinaryOps.SUB, BinaryOps.MUL, BinaryOps.MAX, BinaryOps.CMPLT,
                             TernaryOps.WHERE]
  # HACK: Use s16 and u16 for int8 and uint8 buffers. This can be wrong in cast.
  types: Dict[DType, str] = { dtypes.int8: "s16", dtypes.int16: "s16", dtypes.int32: "s32", dtypes.int64: "s64",
                              dtypes.uint8: "u16", dtypes.uint16: "u16", dtypes.uint32: "u32", dtypes.uint64: "u64",
                              dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64", dtypes.bool: "pred" }

  mem_types: Dict[DType, str] =  types.copy()
  mem_types.update({dtypes.int8: "s8", dtypes.uint8: "u8", dtypes.bool: "u8", dtypes.float16: "b16"})

  const_requires_mov: List[DType] = [dtypes.half, dtypes.bool]

  def render_const(self, x:ConstType, dtype:DType, mov=None) -> Union[List[str], str]:
    val = render_val(x, dtype)
    if dtype == dtypes.bool: return [f"setp.ne.s16 {mov}, {val}, 0;"]
    return [f"mov.b{self.types[dtype][1:]} {mov}, {val};"] if mov else val

  def render_local(self, dest, name, size, dtype) -> List[str]:
    return [f".shared .align 4 .b8 {name}[{size*dtype.itemsize}];", f"mov.u64 {dest}, {name}[0];"]

  def render_loop(self, idx, start, label, acc=None) -> List[str]: return [f"mov.u32 {idx}, {start};", f"{label}:"]

  def render_bra(self, b1, pred=None, b2=None) -> List[str]: return [f"@{pred} bra {b1};"] if pred else [f"bra {b1};"]

  def render_load(self, loc, dest, dtype, gate=None, alt=None, ss="", offset=0, b1=None, b2=None) -> List[str]:
    assert dtype != dtypes.bool
    if gate: 
      return [f"@!{gate} bra {b1};",
              f"ld{ss}.{self.mem_types[dtype]} {dest}, [{loc}+{offset}];",
              f"bra {b2};",
              f"{b1}:",
              f"mov.b{self.types[dtype][1:]} {dest}, {alt};",
              f"{b2}:"]
    return [f"ld{ss}.{self.mem_types[dtype]} {dest}, [{loc}+{offset}];"]

  def render_store(self, loc, val, dtype, gate=None, ss="", offset=0) -> List[str]:
    return [(f"@{gate} " if gate else "") + f"st{ss}.{self.mem_types[dtype]} [{loc}+{offset}], {val};"]

  def render_cast(self, d:str, a:str, dtype:DType, atype:DType, bitcast=False, pred=False) -> List[str]:
    if bitcast: return [f"mov.b{self.types[dtype][1:]} {d}, {a};"]
    if atype == dtypes.bool: return[f"selp.b{self.types[dtype][1:]} {d}, {render_val(1, dtype)}, {render_val(0, dtype)}, {a};"]
    if dtype == dtypes.bool: return [f"setp.ne.b{self.types[atype][1:]} {d}, {a}, {self.render_const(0, atype)};"]
    rnd = ('.rzi' if dtypes.is_int(dtype) and dtypes.is_float(atype) else
           '.rn' if dtypes.is_float(dtype) and (dtype.itemsize < atype.itemsize or dtypes.is_int(atype) or atype == dtypes.bool) else '')
    return [f"cvt{rnd}.{self.types[dtype]}.{self.types[atype]} {d}, {a};"]

  def render_kernel(self, kernel, function_name, bufs, regs) -> str:
    kernel = [f".reg .{reg.split('_')[-2]} %{reg}<{cnt}>;" for reg,cnt in regs] + kernel + ["ret;"]
    def fmt(line): return line if line[0]=="$" else "\t" + line.replace(" ", "\t" if len(line.split(" ")[0]) > 7 else "\t\t", 1)
    return (f"{self.kernel_prefix} {function_name}(\n\t" +
            ',\n\t'.join([f".param .{'u64' if dtype.__class__ == PtrDType else self.types[dtype]} {name}" for name,dtype in bufs]) + "\n)\n{\n" +
            '\n'.join([fmt(line) for op in kernel for line in op.splitlines()]) +
            "\n}")

  def render(self, name:str, uops:UOpGraph) -> str:
    kernel:List[str] = [".reg            .b32 %rem<3>;"]
    bufs = []

    uops.linearize(ptx_matcher)
    def optimize_ordering(block):
      def successors(uop): return list(filter(lambda x: uop in x.vin, block))
      for uu in reversed(block):
        if len(succ:=successors(uu)) and uu.uop in [UOps.ALU, UOps.CAST, UOps.SPECIAL]:
          ni = min([block.index(scc) for scc in succ])
          block.insert(ni-1, block.pop(block.index(uu)))
      return block
    uops._uops = optimize_ordering(uops.uops)
    if DEBUG >= 4: uops.print()

    def kk(*s: str): kernel.append("\n".join(s))

    c: DefaultDict[str, int] = defaultdict(int)
    r: Dict[UOp, Union[List[str], str]] = {}
    def ssa(prefix:str, u:Optional[UOp]=None, dtype:Optional[str]=None) -> str:
      nonlocal c, r
      prefix += f"_{dtype if dtype is not None else self.types[cast(DType, cast(UOp, u).dtype)]}_"
      c[prefix] += 1
      if u is not None: r[u] = f"%{prefix}{c[prefix]-1}"
      return f"%{prefix}{c[prefix]-1}"

    c_label: DefaultDict[str, int] = defaultdict(int)
    r_label: Dict[UOp, str] = {}
    def ssa_label(prefix:str, u:UOp = None):
      nonlocal c_label, r_label
      c_label[prefix] += 1
      if u is None: return f"{self.label_prefix}{prefix}_{c_label[prefix]-1}"
      r_label[u] = f"{self.label_prefix}{prefix}_{c_label[prefix]-1}"
      return r_label[u]

    def const(x:ConstType, dtype:DType, mov=False):
      if mov or dtype in self.const_requires_mov:
        kk(*self.render_const(x, dtype, mov=(out:=ssa('const', dtype=self.types[dtype]))))
        return out
      return self.render_const(x, dtype)

    def _cast(a, dtype:DType, atype:DType, bitcast=False, u=None, pred=False):
      if atype == dtype or isinstance(atype, PtrDType):
        if u: r[u] = a
        return a
      kk(*self.render_cast((ret:=ssa('cast', u, self.types[dtype])), a, dtype, atype, bitcast))
      return ret

    for u in uops:
      uop,dtype,vin,args = u.uop,u.dtype,u.vin,u.arg
      if uop is UOps.IF:
        assert vin[0].dtype is not None
        kk(*self.render_bra(lb:=ssa_label('if', u), _cast(r[vin[0]], dtypes.bool, vin[0].dtype, u=u, pred=True), f"{lb}_true"), f"{lb}_true:")
      elif uop is UOps.BARRIER and self.barrier: kk(self.barrier)
      elif uop is UOps.ENDRANGE:
        kk(self.asm_for_op[BinaryOps.ADD](r[vin[0]], r[vin[0]], "1", dtypes.int, self.types[dtypes.int]),
            self.asm_for_op[BinaryOps.CMPLT](pred:=ssa("pred", dtype="pred"), r[vin[0]], r[vin[0].vin[1]], dtypes.uint32, self.types[dtypes.uint32]))
        kk(*self.render_bra(r_label[vin[0]], pred))
      elif uop is UOps.ENDIF:
        kk(f"{r_label[vin[0]]}:")
      elif uop is UOps.STORE:
        assert vin[0].dtype is not None and vin[2].dtype is not None
        assert vin[0].dtype == dtypes.int64, "store isn't int64"
        assert vin[1].uop is UOps.CONST, f"store isn't const {u}"
        mem_type = '.shared' if vin[0].uop is UOps.DEFINE_LOCAL or any(x.uop is UOps.DEFINE_LOCAL for x in vin[0].parents) else '.global'
        if vin[2].dtype.count > 1:
          kk((f"@{r[vin[3]]} " if len(vin)>3 else "") + \
              f"st{mem_type}.v{vin[2].dtype.count}.{self.mem_types[vin[2].dtype.scalar()]} [{r[vin[0]]}+{vin[1].arg}], {{{', '.join(r[vin[2]])}}};")
        else:
          kk(*self.render_store(r[vin[0]], r[vin[2]], vin[2].dtype, gate=r[vin[3]] if len(vin)>3 else None, ss=mem_type, offset=vin[1].arg))
      else:
        assert dtype is not None, f"None dtype for uop {uop}"
        if uop is UOps.RANGE: kk(*self.render_loop(ssa('ridx', u), r[vin[0]], ssa_label('loop', u)))
        elif uop is UOps.ALU:
          assert vin[0].dtype is not None
          if args is BinaryOps.CMPLT or args is BinaryOps.CMPEQ:
            # pass in the other dtype here
            kk(self.asm_for_op[args](ssa("alu", u), *[r[x] for x in vin], vin[0].dtype, self.types[vin[0].dtype]))
          else:
            kk(self.asm_for_op[args](ssa("alu", u), *[r[x] for x in vin], dtype, self.types[dtype]))
        elif uop is UOps.DEFINE_ACC:
          if dtype.count > 1:
            r[u] = [ssa('acc', dtype=self.types[dtype.scalar()]) for _ in range(dtype.count)]
            for uu in r[u]: kk(f"mov.b{self.types[dtype.scalar()][1:]} {uu}, {const(args[0], dtype.scalar())};")
          else: kk(f"mov.b{self.types[dtype][1:]} {ssa('acc', u)}, {const(args[0], dtype)};")
        elif uop is UOps.SPECIAL:
          assert args[1][0] != "i", "idx not supported"
          kk(f"mov.u32 %{args[1]}, {(self.gid if args[1][0] == 'g' else self.lid)[args[0]]};")
          r[u] = "%" + args[1]
          kernel = [f".reg .u32 %{args[1]};"] + kernel
        elif uop is UOps.CONST:
          if dtype.count > 1: r[u] = [const(args, dtype.scalar(), mov=True) for _ in range(dtype.count)]
          else: r[u] = const(args, dtype, mov=False)
        elif uop is UOps.GEP: r[u] = r[vin[0]][u.arg]
        elif uop is UOps.LOAD:
          assert vin[0].dtype == dtypes.int64, "load isn't int64"
          assert vin[1].uop is UOps.CONST, f"load isn't const {u}"
          mem_type = '.shared' if vin[0].uop is UOps.DEFINE_LOCAL or any(x.uop is UOps.DEFINE_LOCAL for x in vin[0].parents) else '.global'
          if dtype.count > 1:
            r[u] = [ssa('val', dtype=self.types[dtype.scalar()]) for _ in range(dtype.count)]
            if(len(vin)>3):
              for v in r[u]: kk(f"mov.{self.mem_types[dtype.scalar()]} {v}, {render_val(0, dtype.scalar())};")
            kk((f"@{r[vin[2]]}"if len(vin) > 3 else "")
              + f" ld{mem_type}.v{dtype.count}.{self.mem_types[dtype.scalar()]} {{{', '.join(r[u])}}}, [{r[vin[0]]}+{vin[1].arg}];")
          else:
            kk(*self.render_load(r[vin[0]], ssa('val', u), dtype, gate=r[vin[2]] if len(vin) > 3 else None,
                                alt=r[vin[3]] if len(vin) > 3 else None, ss=mem_type, offset=vin[1].arg, b1=ssa_label("L"),b2=ssa_label("L")))
        elif uop is UOps.PHI:
          if dtype.count > 1:
            for x0, x1 in zip(r[vin[0]], r[vin[1]]): kk(f"mov.b{self.types[dtype.scalar()][1:]} {x0}, {x1};")
          else:
            kk(f"mov.b{self.types[dtype][1:]} {r[vin[0]]}, {r[vin[1]]};")
          r[u] = r[vin[0]]
        elif uop in {UOps.CAST, UOps.BITCAST}:
          assert vin[0].dtype is not None
          if dtype.count>1: r[u] = [r[x] for x in vin] # type: ignore
          else: _cast(r[vin[0]], dtype, vin[0].dtype, bitcast=uop is UOps.BITCAST, u=u)
        elif uop is UOps.DEFINE_LOCAL:
          # TODO: we should sum these, and fetch 0xC000 from somewhere
          assert args[1]*dtype.itemsize <= 0xC000, "too large local"
          kk(*self.render_local(ssa('local', u, self.types[dtypes.ulong]), args[0], args[1], dtype))
        elif uop is UOps.DEFINE_VAR:
          bufs.append((args.expr, dtype))
          r[u] = f"%{args.expr}"
          if self.load_global: kk(*self.render_load(args.expr, ssa('dat', u, self.types[dtype]), dtype, ss=".param"))
        elif uop is UOps.DEFINE_GLOBAL:
          bufs.append((nm:=f"data{args[0]}", dtype))
          r[u] = f"%{nm}"
          if self.load_global:
            dt = dtypes.ulong if dtype.__class__ == PtrDType else dtype
            kk(*self.render_load(nm, ssa('dat', u, self.types[dt]), dt, ss=".param"))
        elif uop is UOps.WMMA:
          wmma = []
          for vv in vin[:2]:
            for i in range(0, len(r[vv]), 2):
              wmma.append(ssa("wmma", dtype="b32"))
              kk(f'mov.b32 {wmma[-1]}, {{{", ".join(r[vv][i:i+2])}}};')
          r[u] = [ssa("wmma", dtype=self.types[dtype.scalar()]) for _ in range(dtype.count)]
          kk(f'mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\
            {{{", ".join(r[u])}}}, {{{", ".join(wmma[:4])}}}, {{{", ".join(wmma[4:])}}}, {{{", ".join(r[vin[2]])}}};')
        else: raise NotImplementedError(f"no code for {uop}")
    if False:
      return """ .version VERSION
.target TARGET
.address_size 64
.visible .entry r_16_4_32_2_2_2_2_2_8_28_2_16_4_4_4n1(
	.param .u64 data0,
	.param .u64 data1,
	.param .u64 data2
)
{
	.reg		.u64 %dat_u64_<3>;
	.reg		.f16 %const_f16_<1>;
	.reg		.f32 %acc_f32_<64>;
	.reg		.s32 %ridx_s32_<3>;
	.reg		.s32 %alu_s32_<46>;
	.reg		.s64 %cast_s64_<3>;
	.reg		.s64 %alu_s64_<6>;
	.reg		.f16 %val_f16_<48>;
	.reg		.pred %alu_pred_<2>;
	.reg		.b32 %wmma_b32_<96>;
	.reg		.f32 %wmma_f32_<64>;
	.reg		.pred %pred_pred_<3>;
	.reg		.u32 %gidx1;
	.reg		.u32 %gidx2;
	.reg		.u32 %lidx5;
	.reg		.u32 %lidx4;
	.reg		.u32 %lidx3;
	.reg		.u32 %gidx0;
	.reg		           .b32 %rem<3>;
	ld.param.u64	%dat_u64_0, [data0+0];
	ld.param.u64	%dat_u64_1, [data1+0];
	ld.param.u64	%dat_u64_2, [data2+0];
	mov.b16		%const_f16_0, 0x0000;
	mov.b32		%acc_f32_0, 0f00000000;
	mov.b32		%acc_f32_1, 0f00000000;
	mov.b32		%acc_f32_2, 0f00000000;
	mov.b32		%acc_f32_3, 0f00000000;
	mov.b32		%acc_f32_4, 0f00000000;
	mov.b32		%acc_f32_5, 0f00000000;
	mov.b32		%acc_f32_6, 0f00000000;
	mov.b32		%acc_f32_7, 0f00000000;
	mov.b32		%acc_f32_8, 0f00000000;
	mov.b32		%acc_f32_9, 0f00000000;
	mov.b32		%acc_f32_10, 0f00000000;
	mov.b32		%acc_f32_11, 0f00000000;
	mov.b32		%acc_f32_12, 0f00000000;
	mov.b32		%acc_f32_13, 0f00000000;
	mov.b32		%acc_f32_14, 0f00000000;
	mov.b32		%acc_f32_15, 0f00000000;
	mov.b32		%acc_f32_16, 0f00000000;
	mov.b32		%acc_f32_17, 0f00000000;
	mov.b32		%acc_f32_18, 0f00000000;
	mov.b32		%acc_f32_19, 0f00000000;
	mov.b32		%acc_f32_20, 0f00000000;
	mov.b32		%acc_f32_21, 0f00000000;
	mov.b32		%acc_f32_22, 0f00000000;
	mov.b32		%acc_f32_23, 0f00000000;
	mov.b32		%acc_f32_24, 0f00000000;
	mov.b32		%acc_f32_25, 0f00000000;
	mov.b32		%acc_f32_26, 0f00000000;
	mov.b32		%acc_f32_27, 0f00000000;
	mov.b32		%acc_f32_28, 0f00000000;
	mov.b32		%acc_f32_29, 0f00000000;
	mov.b32		%acc_f32_30, 0f00000000;
	mov.b32		%acc_f32_31, 0f00000000;
	mov.b32		%acc_f32_32, 0f00000000;
	mov.b32		%acc_f32_33, 0f00000000;
	mov.b32		%acc_f32_34, 0f00000000;
	mov.b32		%acc_f32_35, 0f00000000;
	mov.b32		%acc_f32_36, 0f00000000;
	mov.b32		%acc_f32_37, 0f00000000;
	mov.b32		%acc_f32_38, 0f00000000;
	mov.b32		%acc_f32_39, 0f00000000;
	mov.b32		%acc_f32_40, 0f00000000;
	mov.b32		%acc_f32_41, 0f00000000;
	mov.b32		%acc_f32_42, 0f00000000;
	mov.b32		%acc_f32_43, 0f00000000;
	mov.b32		%acc_f32_44, 0f00000000;
	mov.b32		%acc_f32_45, 0f00000000;
	mov.b32		%acc_f32_46, 0f00000000;
	mov.b32		%acc_f32_47, 0f00000000;
	mov.b32		%acc_f32_48, 0f00000000;
	mov.b32		%acc_f32_49, 0f00000000;
	mov.b32		%acc_f32_50, 0f00000000;
	mov.b32		%acc_f32_51, 0f00000000;
	mov.b32		%acc_f32_52, 0f00000000;
	mov.b32		%acc_f32_53, 0f00000000;
	mov.b32		%acc_f32_54, 0f00000000;
	mov.b32		%acc_f32_55, 0f00000000;
	mov.b32		%acc_f32_56, 0f00000000;
	mov.b32		%acc_f32_57, 0f00000000;
	mov.b32		%acc_f32_58, 0f00000000;
	mov.b32		%acc_f32_59, 0f00000000;
	mov.b32		%acc_f32_60, 0f00000000;
	mov.b32		%acc_f32_61, 0f00000000;
	mov.b32		%acc_f32_62, 0f00000000;
	mov.b32		%acc_f32_63, 0f00000000;
	mov.u32		%ridx_s32_0, 0;
$loop_0:
	mov.u32		%ridx_s32_1, 0;
$loop_1:
	mov.u32		%ridx_s32_2, 0;
$loop_2:
	shl.b32		%alu_s32_0, %ridx_s32_2, 4;
	mov.u32		%gidx0, %ctaid.z;
	mul.lo.s32	%alu_s32_1, %gidx0, 25088;
	mov.u32		%lidx3, %tid.x;
	shr.s32		%alu_s32_2, %lidx3, 2;
	mad.lo.s32	%alu_s32_3, %alu_s32_2, 784, %alu_s32_1;
	mov.u32		%lidx4, %tid.y;
	mad.lo.s32	%alu_s32_4, %lidx4, 1568, %alu_s32_3;
	mov.u32		%lidx5, %tid.z;
	mad.lo.s32	%alu_s32_5, %lidx5, 3136, %alu_s32_4;
	mov.u32		%gidx2, %ctaid.x;
	mad.lo.s32	%alu_s32_6, %gidx2, 3211264, %alu_s32_5;
	mad.lo.s32	%alu_s32_7, %ridx_s32_0, 401408, %alu_s32_6;
	mad.lo.s32	%alu_s32_8, %ridx_s32_1, 28, %alu_s32_7;
	add.s32		%alu_s32_9, %alu_s32_0, %alu_s32_8;
	shr.u32		 %rem0, %lidx3, 31;                                                         
	add.s32		 %rem1, %lidx3, %rem0;                                                       
	and.b32		  %rem2, %rem1, -2;                                                        
	sub.s32		 %alu_s32_10, %lidx3, %rem2;
	shl.b32		%alu_s32_11, %alu_s32_10, 1;
	add.s32		%alu_s32_12, %alu_s32_11, %alu_s32_9;
	shr.s32		%alu_s32_13, %lidx3, 1;
	shr.u32		 %rem0, %alu_s32_13, 31;                                                         
	add.s32		 %rem1, %alu_s32_13, %rem0;                                                       
	and.b32		  %rem2, %rem1, -2;                                                        
	sub.s32		 %alu_s32_14, %alu_s32_13, %rem2;
	shl.b32		%alu_s32_15, %alu_s32_14, 2;
	add.s32		%alu_s32_16, %alu_s32_15, %alu_s32_12;
	cvt.s64.s32	%cast_s64_0, %alu_s32_16;
	shl.b64		%alu_s64_0, %cast_s64_0, 1;
	add.s64		%alu_s64_1, %alu_s64_0, %dat_u64_2;
	ld.global.b16	%val_f16_0, [%alu_s64_1+0];
	ld.global.b16	%val_f16_1, [%alu_s64_1+2];
	ld.global.b16	%val_f16_2, [%alu_s64_1+12544];
	ld.global.b16	%val_f16_3, [%alu_s64_1+12546];
	ld.global.b16	%val_f16_4, [%alu_s64_1+25088];
	ld.global.b16	%val_f16_5, [%alu_s64_1+25090];
	ld.global.b16	%val_f16_6, [%alu_s64_1+37632];
	ld.global.b16	%val_f16_7, [%alu_s64_1+37634];
	add.s32		%alu_s32_17, %alu_s32_0, %alu_s32_11;
	add.s32		%alu_s32_18, %alu_s32_15, %alu_s32_17;
	setp.lt.s32	%alu_pred_0, %alu_s32_18, 19;
	@%alu_pred_0	ld.global.b16 %val_f16_8, [%alu_s64_1+18];
	@!%alu_pred_0	mov.b16 %val_f16_8, %const_f16_0;
	@%alu_pred_0	ld.global.b16 %val_f16_9, [%alu_s64_1+12562];
	@!%alu_pred_0	mov.b16 %val_f16_9, %const_f16_0;
	@%alu_pred_0	ld.global.b16 %val_f16_10, [%alu_s64_1+25106];
	@!%alu_pred_0	mov.b16 %val_f16_10, %const_f16_0;
	@%alu_pred_0	ld.global.b16 %val_f16_11, [%alu_s64_1+37650];
	@!%alu_pred_0	mov.b16 %val_f16_11, %const_f16_0;
	setp.lt.s32	%alu_pred_1, %alu_s32_18, 20;
	@%alu_pred_1	ld.global.b16 %val_f16_12, [%alu_s64_1+16];
	@!%alu_pred_1	mov.b16 %val_f16_12, %const_f16_0;
	@%alu_pred_1	ld.global.b16 %val_f16_13, [%alu_s64_1+12560];
	@!%alu_pred_1	mov.b16 %val_f16_13, %const_f16_0;
	@%alu_pred_1	ld.global.b16 %val_f16_14, [%alu_s64_1+25104];
	@!%alu_pred_1	mov.b16 %val_f16_14, %const_f16_0;
	@%alu_pred_1	ld.global.b16 %val_f16_15, [%alu_s64_1+37648];
	@!%alu_pred_1	mov.b16 %val_f16_15, %const_f16_0;
	shl.b32		%alu_s32_19, %ridx_s32_2, 5;
	mov.u32		%gidx1, %ctaid.y;
	mul.lo.s32	%alu_s32_20, %gidx1, 200704;
	mad.lo.s32	%alu_s32_21, %lidx5, 12544, %alu_s32_20;
	mad.lo.s32	%alu_s32_22, %alu_s32_2, 3136, %alu_s32_21;
	mad.lo.s32	%alu_s32_23, %lidx4, 6272, %alu_s32_22;
	mad.lo.s32	%alu_s32_24, %gidx2, 6422528, %alu_s32_23;
	mad.lo.s32	%alu_s32_25, %ridx_s32_0, 802816, %alu_s32_24;
	mad.lo.s32	%alu_s32_26, %ridx_s32_1, 112, %alu_s32_25;
	add.s32		%alu_s32_27, %alu_s32_19, %alu_s32_26;
	shl.b32		%alu_s32_28, %alu_s32_10, 2;
	add.s32		%alu_s32_29, %alu_s32_28, %alu_s32_27;
	shl.b32		%alu_s32_30, %alu_s32_14, 3;
	add.s32		%alu_s32_31, %alu_s32_30, %alu_s32_29;
	cvt.s64.s32	%cast_s64_1, %alu_s32_31;
	shl.b64		%alu_s64_2, %cast_s64_1, 1;
	add.s64		%alu_s64_3, %alu_s64_2, %dat_u64_1;
	ld.global.b16	%val_f16_16, [%alu_s64_3+0];
	ld.global.b16	%val_f16_17, [%alu_s64_3+4];
	@%alu_pred_1	ld.global.b16 %val_f16_18, [%alu_s64_3+32];
	@!%alu_pred_1	mov.b16 %val_f16_18, %const_f16_0;
	@%alu_pred_0	ld.global.b16 %val_f16_19, [%alu_s64_3+36];
	@!%alu_pred_0	mov.b16 %val_f16_19, %const_f16_0;
	ld.global.b16	%val_f16_20, [%alu_s64_3+50176];
	ld.global.b16	%val_f16_21, [%alu_s64_3+50180];
	@%alu_pred_1	ld.global.b16 %val_f16_22, [%alu_s64_3+50208];
	@!%alu_pred_1	mov.b16 %val_f16_22, %const_f16_0;
	@%alu_pred_0	ld.global.b16 %val_f16_23, [%alu_s64_3+50212];
	@!%alu_pred_0	mov.b16 %val_f16_23, %const_f16_0;
	mov.b32		%wmma_b32_0, {%val_f16_16, %val_f16_17};
	mov.b32		%wmma_b32_1, {%val_f16_20, %val_f16_21};
	mov.b32		%wmma_b32_2, {%val_f16_18, %val_f16_19};
	mov.b32		%wmma_b32_3, {%val_f16_22, %val_f16_23};
	mov.b32		%wmma_b32_4, {%val_f16_0, %val_f16_1};
	mov.b32		%wmma_b32_5, {%val_f16_12, %val_f16_8};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {%wmma_f32_0, %wmma_f32_1, %wmma_f32_2, %wmma_f32_3}, {%wmma_b32_0, %wmma_b32_1, %wmma_b32_2, %wmma_b32_3}, {%wmma_b32_4, %wmma_b32_5}, {%acc_f32_0, %acc_f32_1, %acc_f32_2, %acc_f32_3};
	mov.b32		%wmma_b32_6, {%val_f16_16, %val_f16_17};
	mov.b32		%wmma_b32_7, {%val_f16_20, %val_f16_21};
	mov.b32		%wmma_b32_8, {%val_f16_18, %val_f16_19};
	mov.b32		%wmma_b32_9, {%val_f16_22, %val_f16_23};
	mov.b32		%wmma_b32_10, {%val_f16_2, %val_f16_3};
	mov.b32		%wmma_b32_11, {%val_f16_13, %val_f16_9};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {%wmma_f32_4, %wmma_f32_5, %wmma_f32_6, %wmma_f32_7}, {%wmma_b32_6, %wmma_b32_7, %wmma_b32_8, %wmma_b32_9}, {%wmma_b32_10, %wmma_b32_11}, {%acc_f32_16, %acc_f32_17, %acc_f32_18, %acc_f32_19};
	mov.b32		%wmma_b32_12, {%val_f16_16, %val_f16_17};
	mov.b32		%wmma_b32_13, {%val_f16_20, %val_f16_21};
	mov.b32		%wmma_b32_14, {%val_f16_18, %val_f16_19};
	mov.b32		%wmma_b32_15, {%val_f16_22, %val_f16_23};
	mov.b32		%wmma_b32_16, {%val_f16_4, %val_f16_5};
	mov.b32		%wmma_b32_17, {%val_f16_14, %val_f16_10};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {%wmma_f32_8, %wmma_f32_9, %wmma_f32_10, %wmma_f32_11}, {%wmma_b32_12, %wmma_b32_13, %wmma_b32_14, %wmma_b32_15}, {%wmma_b32_16, %wmma_b32_17}, {%acc_f32_32, %acc_f32_33, %acc_f32_34, %acc_f32_35};
	mov.b32		%wmma_b32_18, {%val_f16_16, %val_f16_17};
	mov.b32		%wmma_b32_19, {%val_f16_20, %val_f16_21};
	mov.b32		%wmma_b32_20, {%val_f16_18, %val_f16_19};
	mov.b32		%wmma_b32_21, {%val_f16_22, %val_f16_23};
	mov.b32		%wmma_b32_22, {%val_f16_6, %val_f16_7};
	mov.b32		%wmma_b32_23, {%val_f16_15, %val_f16_11};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {%wmma_f32_12, %wmma_f32_13, %wmma_f32_14, %wmma_f32_15}, {%wmma_b32_18, %wmma_b32_19, %wmma_b32_20, %wmma_b32_21}, {%wmma_b32_22, %wmma_b32_23}, {%acc_f32_48, %acc_f32_49, %acc_f32_50, %acc_f32_51};
	ld.global.b16	%val_f16_24, [%alu_s64_3+100352];
	ld.global.b16	%val_f16_25, [%alu_s64_3+100356];
	@%alu_pred_1	ld.global.b16 %val_f16_26, [%alu_s64_3+100384];
	@!%alu_pred_1	mov.b16 %val_f16_26, %const_f16_0;
	@%alu_pred_0	ld.global.b16 %val_f16_27, [%alu_s64_3+100388];
	@!%alu_pred_0	mov.b16 %val_f16_27, %const_f16_0;
	ld.global.b16	%val_f16_28, [%alu_s64_3+150528];
	ld.global.b16	%val_f16_29, [%alu_s64_3+150532];
	@%alu_pred_1	ld.global.b16 %val_f16_30, [%alu_s64_3+150560];
	@!%alu_pred_1	mov.b16 %val_f16_30, %const_f16_0;
	@%alu_pred_0	ld.global.b16 %val_f16_31, [%alu_s64_3+150564];
	@!%alu_pred_0	mov.b16 %val_f16_31, %const_f16_0;
	mov.b32		%wmma_b32_24, {%val_f16_24, %val_f16_25};
	mov.b32		%wmma_b32_25, {%val_f16_28, %val_f16_29};
	mov.b32		%wmma_b32_26, {%val_f16_26, %val_f16_27};
	mov.b32		%wmma_b32_27, {%val_f16_30, %val_f16_31};
	mov.b32		%wmma_b32_28, {%val_f16_0, %val_f16_1};
	mov.b32		%wmma_b32_29, {%val_f16_12, %val_f16_8};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {%wmma_f32_16, %wmma_f32_17, %wmma_f32_18, %wmma_f32_19}, {%wmma_b32_24, %wmma_b32_25, %wmma_b32_26, %wmma_b32_27}, {%wmma_b32_28, %wmma_b32_29}, {%acc_f32_4, %acc_f32_5, %acc_f32_6, %acc_f32_7};
	mov.b32		%wmma_b32_30, {%val_f16_24, %val_f16_25};
	mov.b32		%wmma_b32_31, {%val_f16_28, %val_f16_29};
	mov.b32		%wmma_b32_32, {%val_f16_26, %val_f16_27};
	mov.b32		%wmma_b32_33, {%val_f16_30, %val_f16_31};
	mov.b32		%wmma_b32_34, {%val_f16_2, %val_f16_3};
	mov.b32		%wmma_b32_35, {%val_f16_13, %val_f16_9};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {%wmma_f32_20, %wmma_f32_21, %wmma_f32_22, %wmma_f32_23}, {%wmma_b32_30, %wmma_b32_31, %wmma_b32_32, %wmma_b32_33}, {%wmma_b32_34, %wmma_b32_35}, {%acc_f32_20, %acc_f32_21, %acc_f32_22, %acc_f32_23};
	mov.b32		%wmma_b32_36, {%val_f16_24, %val_f16_25};
	mov.b32		%wmma_b32_37, {%val_f16_28, %val_f16_29};
	mov.b32		%wmma_b32_38, {%val_f16_26, %val_f16_27};
	mov.b32		%wmma_b32_39, {%val_f16_30, %val_f16_31};
	mov.b32		%wmma_b32_40, {%val_f16_4, %val_f16_5};
	mov.b32		%wmma_b32_41, {%val_f16_14, %val_f16_10};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {%wmma_f32_24, %wmma_f32_25, %wmma_f32_26, %wmma_f32_27}, {%wmma_b32_36, %wmma_b32_37, %wmma_b32_38, %wmma_b32_39}, {%wmma_b32_40, %wmma_b32_41}, {%acc_f32_36, %acc_f32_37, %acc_f32_38, %acc_f32_39};
	mov.b32		%wmma_b32_42, {%val_f16_24, %val_f16_25};
	mov.b32		%wmma_b32_43, {%val_f16_28, %val_f16_29};
	mov.b32		%wmma_b32_44, {%val_f16_26, %val_f16_27};
	mov.b32		%wmma_b32_45, {%val_f16_30, %val_f16_31};
	mov.b32		%wmma_b32_46, {%val_f16_6, %val_f16_7};
	mov.b32		%wmma_b32_47, {%val_f16_15, %val_f16_11};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {%wmma_f32_28, %wmma_f32_29, %wmma_f32_30, %wmma_f32_31}, {%wmma_b32_42, %wmma_b32_43, %wmma_b32_44, %wmma_b32_45}, {%wmma_b32_46, %wmma_b32_47}, {%acc_f32_52, %acc_f32_53, %acc_f32_54, %acc_f32_55};
	ld.global.b16	%val_f16_32, [%alu_s64_3+200704];
	ld.global.b16	%val_f16_33, [%alu_s64_3+200708];
	@%alu_pred_1	ld.global.b16 %val_f16_34, [%alu_s64_3+200736];
	@!%alu_pred_1	mov.b16 %val_f16_34, %const_f16_0;
	@%alu_pred_0	ld.global.b16 %val_f16_35, [%alu_s64_3+200740];
	@!%alu_pred_0	mov.b16 %val_f16_35, %const_f16_0;
	ld.global.b16	%val_f16_36, [%alu_s64_3+250880];
	ld.global.b16	%val_f16_37, [%alu_s64_3+250884];
	@%alu_pred_1	ld.global.b16 %val_f16_38, [%alu_s64_3+250912];
	@!%alu_pred_1	mov.b16 %val_f16_38, %const_f16_0;
	@%alu_pred_0	ld.global.b16 %val_f16_39, [%alu_s64_3+250916];
	@!%alu_pred_0	mov.b16 %val_f16_39, %const_f16_0;
	mov.b32		%wmma_b32_48, {%val_f16_32, %val_f16_33};
	mov.b32		%wmma_b32_49, {%val_f16_36, %val_f16_37};
	mov.b32		%wmma_b32_50, {%val_f16_34, %val_f16_35};
	mov.b32		%wmma_b32_51, {%val_f16_38, %val_f16_39};
	mov.b32		%wmma_b32_52, {%val_f16_0, %val_f16_1};
	mov.b32		%wmma_b32_53, {%val_f16_12, %val_f16_8};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {%wmma_f32_32, %wmma_f32_33, %wmma_f32_34, %wmma_f32_35}, {%wmma_b32_48, %wmma_b32_49, %wmma_b32_50, %wmma_b32_51}, {%wmma_b32_52, %wmma_b32_53}, {%acc_f32_8, %acc_f32_9, %acc_f32_10, %acc_f32_11};
	mov.b32		%wmma_b32_54, {%val_f16_32, %val_f16_33};
	mov.b32		%wmma_b32_55, {%val_f16_36, %val_f16_37};
	mov.b32		%wmma_b32_56, {%val_f16_34, %val_f16_35};
	mov.b32		%wmma_b32_57, {%val_f16_38, %val_f16_39};
	mov.b32		%wmma_b32_58, {%val_f16_2, %val_f16_3};
	mov.b32		%wmma_b32_59, {%val_f16_13, %val_f16_9};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {%wmma_f32_36, %wmma_f32_37, %wmma_f32_38, %wmma_f32_39}, {%wmma_b32_54, %wmma_b32_55, %wmma_b32_56, %wmma_b32_57}, {%wmma_b32_58, %wmma_b32_59}, {%acc_f32_24, %acc_f32_25, %acc_f32_26, %acc_f32_27};
	mov.b32		%wmma_b32_60, {%val_f16_32, %val_f16_33};
	mov.b32		%wmma_b32_61, {%val_f16_36, %val_f16_37};
	mov.b32		%wmma_b32_62, {%val_f16_34, %val_f16_35};
	mov.b32		%wmma_b32_63, {%val_f16_38, %val_f16_39};
	mov.b32		%wmma_b32_64, {%val_f16_4, %val_f16_5};
	mov.b32		%wmma_b32_65, {%val_f16_14, %val_f16_10};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {%wmma_f32_40, %wmma_f32_41, %wmma_f32_42, %wmma_f32_43}, {%wmma_b32_60, %wmma_b32_61, %wmma_b32_62, %wmma_b32_63}, {%wmma_b32_64, %wmma_b32_65}, {%acc_f32_40, %acc_f32_41, %acc_f32_42, %acc_f32_43};
	mov.b32		%wmma_b32_66, {%val_f16_32, %val_f16_33};
	mov.b32		%wmma_b32_67, {%val_f16_36, %val_f16_37};
	mov.b32		%wmma_b32_68, {%val_f16_34, %val_f16_35};
	mov.b32		%wmma_b32_69, {%val_f16_38, %val_f16_39};
	mov.b32		%wmma_b32_70, {%val_f16_6, %val_f16_7};
	mov.b32		%wmma_b32_71, {%val_f16_15, %val_f16_11};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {%wmma_f32_44, %wmma_f32_45, %wmma_f32_46, %wmma_f32_47}, {%wmma_b32_66, %wmma_b32_67, %wmma_b32_68, %wmma_b32_69}, {%wmma_b32_70, %wmma_b32_71}, {%acc_f32_56, %acc_f32_57, %acc_f32_58, %acc_f32_59};
	ld.global.b16	%val_f16_40, [%alu_s64_3+301056];
	ld.global.b16	%val_f16_41, [%alu_s64_3+301060];
	@%alu_pred_1	ld.global.b16 %val_f16_42, [%alu_s64_3+301088];
	@!%alu_pred_1	mov.b16 %val_f16_42, %const_f16_0;
	@%alu_pred_0	ld.global.b16 %val_f16_43, [%alu_s64_3+301092];
	@!%alu_pred_0	mov.b16 %val_f16_43, %const_f16_0;
	ld.global.b16	%val_f16_44, [%alu_s64_3+351232];
	ld.global.b16	%val_f16_45, [%alu_s64_3+351236];
	@%alu_pred_1	ld.global.b16 %val_f16_46, [%alu_s64_3+351264];
	@!%alu_pred_1	mov.b16 %val_f16_46, %const_f16_0;
	@%alu_pred_0	ld.global.b16 %val_f16_47, [%alu_s64_3+351268];
	@!%alu_pred_0	mov.b16 %val_f16_47, %const_f16_0;
	mov.b32		%wmma_b32_72, {%val_f16_40, %val_f16_41};
	mov.b32		%wmma_b32_73, {%val_f16_44, %val_f16_45};
	mov.b32		%wmma_b32_74, {%val_f16_42, %val_f16_43};
	mov.b32		%wmma_b32_75, {%val_f16_46, %val_f16_47};
	mov.b32		%wmma_b32_76, {%val_f16_0, %val_f16_1};
	mov.b32		%wmma_b32_77, {%val_f16_12, %val_f16_8};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {%wmma_f32_48, %wmma_f32_49, %wmma_f32_50, %wmma_f32_51}, {%wmma_b32_72, %wmma_b32_73, %wmma_b32_74, %wmma_b32_75}, {%wmma_b32_76, %wmma_b32_77}, {%acc_f32_12, %acc_f32_13, %acc_f32_14, %acc_f32_15};
	mov.b32		%wmma_b32_78, {%val_f16_40, %val_f16_41};
	mov.b32		%wmma_b32_79, {%val_f16_44, %val_f16_45};
	mov.b32		%wmma_b32_80, {%val_f16_42, %val_f16_43};
	mov.b32		%wmma_b32_81, {%val_f16_46, %val_f16_47};
	mov.b32		%wmma_b32_82, {%val_f16_2, %val_f16_3};
	mov.b32		%wmma_b32_83, {%val_f16_13, %val_f16_9};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {%wmma_f32_52, %wmma_f32_53, %wmma_f32_54, %wmma_f32_55}, {%wmma_b32_78, %wmma_b32_79, %wmma_b32_80, %wmma_b32_81}, {%wmma_b32_82, %wmma_b32_83}, {%acc_f32_28, %acc_f32_29, %acc_f32_30, %acc_f32_31};
	mov.b32		%wmma_b32_84, {%val_f16_40, %val_f16_41};
	mov.b32		%wmma_b32_85, {%val_f16_44, %val_f16_45};
	mov.b32		%wmma_b32_86, {%val_f16_42, %val_f16_43};
	mov.b32		%wmma_b32_87, {%val_f16_46, %val_f16_47};
	mov.b32		%wmma_b32_88, {%val_f16_4, %val_f16_5};
	mov.b32		%wmma_b32_89, {%val_f16_14, %val_f16_10};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {%wmma_f32_56, %wmma_f32_57, %wmma_f32_58, %wmma_f32_59}, {%wmma_b32_84, %wmma_b32_85, %wmma_b32_86, %wmma_b32_87}, {%wmma_b32_88, %wmma_b32_89}, {%acc_f32_44, %acc_f32_45, %acc_f32_46, %acc_f32_47};
	mov.b32		%wmma_b32_90, {%val_f16_40, %val_f16_41};
	mov.b32		%wmma_b32_91, {%val_f16_44, %val_f16_45};
	mov.b32		%wmma_b32_92, {%val_f16_42, %val_f16_43};
	mov.b32		%wmma_b32_93, {%val_f16_46, %val_f16_47};
	mov.b32		%wmma_b32_94, {%val_f16_6, %val_f16_7};
	mov.b32		%wmma_b32_95, {%val_f16_15, %val_f16_11};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {%wmma_f32_60, %wmma_f32_61, %wmma_f32_62, %wmma_f32_63}, {%wmma_b32_90, %wmma_b32_91, %wmma_b32_92, %wmma_b32_93}, {%wmma_b32_94, %wmma_b32_95}, {%acc_f32_60, %acc_f32_61, %acc_f32_62, %acc_f32_63};
	mov.b32		%acc_f32_0, %wmma_f32_0;
	mov.b32		%acc_f32_1, %wmma_f32_1;
	mov.b32		%acc_f32_2, %wmma_f32_2;
	mov.b32		%acc_f32_3, %wmma_f32_3;
	mov.b32		%acc_f32_4, %wmma_f32_16;
	mov.b32		%acc_f32_5, %wmma_f32_17;
	mov.b32		%acc_f32_6, %wmma_f32_18;
	mov.b32		%acc_f32_7, %wmma_f32_19;
	mov.b32		%acc_f32_8, %wmma_f32_32;
	mov.b32		%acc_f32_9, %wmma_f32_33;
	mov.b32		%acc_f32_10, %wmma_f32_34;
	mov.b32		%acc_f32_11, %wmma_f32_35;
	mov.b32		%acc_f32_12, %wmma_f32_48;
	mov.b32		%acc_f32_13, %wmma_f32_49;
	mov.b32		%acc_f32_14, %wmma_f32_50;
	mov.b32		%acc_f32_15, %wmma_f32_51;
	mov.b32		%acc_f32_16, %wmma_f32_4;
	mov.b32		%acc_f32_17, %wmma_f32_5;
	mov.b32		%acc_f32_18, %wmma_f32_6;
	mov.b32		%acc_f32_19, %wmma_f32_7;
	mov.b32		%acc_f32_20, %wmma_f32_20;
	mov.b32		%acc_f32_21, %wmma_f32_21;
	mov.b32		%acc_f32_22, %wmma_f32_22;
	mov.b32		%acc_f32_23, %wmma_f32_23;
	mov.b32		%acc_f32_24, %wmma_f32_36;
	mov.b32		%acc_f32_25, %wmma_f32_37;
	mov.b32		%acc_f32_26, %wmma_f32_38;
	mov.b32		%acc_f32_27, %wmma_f32_39;
	mov.b32		%acc_f32_28, %wmma_f32_52;
	mov.b32		%acc_f32_29, %wmma_f32_53;
	mov.b32		%acc_f32_30, %wmma_f32_54;
	mov.b32		%acc_f32_31, %wmma_f32_55;
	mov.b32		%acc_f32_32, %wmma_f32_8;
	mov.b32		%acc_f32_33, %wmma_f32_9;
	mov.b32		%acc_f32_34, %wmma_f32_10;
	mov.b32		%acc_f32_35, %wmma_f32_11;
	mov.b32		%acc_f32_36, %wmma_f32_24;
	mov.b32		%acc_f32_37, %wmma_f32_25;
	mov.b32		%acc_f32_38, %wmma_f32_26;
	mov.b32		%acc_f32_39, %wmma_f32_27;
	mov.b32		%acc_f32_40, %wmma_f32_40;
	mov.b32		%acc_f32_41, %wmma_f32_41;
	mov.b32		%acc_f32_42, %wmma_f32_42;
	mov.b32		%acc_f32_43, %wmma_f32_43;
	mov.b32		%acc_f32_44, %wmma_f32_56;
	mov.b32		%acc_f32_45, %wmma_f32_57;
	mov.b32		%acc_f32_46, %wmma_f32_58;
	mov.b32		%acc_f32_47, %wmma_f32_59;
	mov.b32		%acc_f32_48, %wmma_f32_12;
	mov.b32		%acc_f32_49, %wmma_f32_13;
	mov.b32		%acc_f32_50, %wmma_f32_14;
	mov.b32		%acc_f32_51, %wmma_f32_15;
	mov.b32		%acc_f32_52, %wmma_f32_28;
	mov.b32		%acc_f32_53, %wmma_f32_29;
	mov.b32		%acc_f32_54, %wmma_f32_30;
	mov.b32		%acc_f32_55, %wmma_f32_31;
	mov.b32		%acc_f32_56, %wmma_f32_44;
	mov.b32		%acc_f32_57, %wmma_f32_45;
	mov.b32		%acc_f32_58, %wmma_f32_46;
	mov.b32		%acc_f32_59, %wmma_f32_47;
	mov.b32		%acc_f32_60, %wmma_f32_60;
	mov.b32		%acc_f32_61, %wmma_f32_61;
	mov.b32		%acc_f32_62, %wmma_f32_62;
	mov.b32		%acc_f32_63, %wmma_f32_63;
	add.s32		%ridx_s32_2, %ridx_s32_2, 1;
	setp.lt.u32	%pred_pred_0, %ridx_s32_2, 2;
	@%pred_pred_0	bra $loop_2;
	add.s32		%ridx_s32_1, %ridx_s32_1, 1;
	setp.lt.u32	%pred_pred_1, %ridx_s32_1, 28;
	@%pred_pred_1	bra $loop_1;
	add.s32		%ridx_s32_0, %ridx_s32_0, 1;
	setp.lt.u32	%pred_pred_2, %ridx_s32_0, 8;
	@%pred_pred_2	bra $loop_0;
	mov.u32		%gidx0, %ctaid.z;
	mov.u32		%lidx4, %tid.y;
	mov.u32		%lidx5, %tid.z;
	mov.u32		%gidx1, %ctaid.y;
	shl.b32		%alu_s32_32, %gidx0, 18;
	shl.b32		%alu_s32_33, %gidx1, 11;
	add.s32		%alu_s32_34, %alu_s32_32, %alu_s32_33;
	add.s32		%alu_s32_35, %alu_s32_34, %gidx2;
	shl.b32		%alu_s32_36, %alu_s32_10, 14;
	add.s32		%alu_s32_37, %alu_s32_36, %alu_s32_35;
	shl.b32		%alu_s32_38, %lidx5, 7;
	add.s32		%alu_s32_39, %alu_s32_38, %alu_s32_37;
	shl.b32		%alu_s32_40, %alu_s32_14, 15;
	add.s32		%alu_s32_41, %alu_s32_40, %alu_s32_39;
	shl.b32		%alu_s32_42, %alu_s32_2, 5;
	add.s32		%alu_s32_43, %alu_s32_42, %alu_s32_41;
	shl.b32		%alu_s32_44, %lidx4, 6;
	add.s32		%alu_s32_45, %alu_s32_44, %alu_s32_43;
	cvt.s64.s32	%cast_s64_2, %alu_s32_45;
	shl.b64		%alu_s64_4, %cast_s64_2, 2;
	add.s64		%alu_s64_5, %alu_s64_4, %dat_u64_0;
	st.global.f32	[%alu_s64_5+0], %acc_f32_0;
	st.global.f32	[%alu_s64_5+1024], %acc_f32_2;
	st.global.f32	[%alu_s64_5+2048], %acc_f32_4;
	st.global.f32	[%alu_s64_5+3072], %acc_f32_6;
	st.global.f32	[%alu_s64_5+4096], %acc_f32_8;
	st.global.f32	[%alu_s64_5+5120], %acc_f32_10;
	st.global.f32	[%alu_s64_5+6144], %acc_f32_12;
	st.global.f32	[%alu_s64_5+7168], %acc_f32_14;
	st.global.f32	[%alu_s64_5+32768], %acc_f32_1;
	st.global.f32	[%alu_s64_5+33792], %acc_f32_3;
	st.global.f32	[%alu_s64_5+34816], %acc_f32_5;
	st.global.f32	[%alu_s64_5+35840], %acc_f32_7;
	st.global.f32	[%alu_s64_5+36864], %acc_f32_9;
	st.global.f32	[%alu_s64_5+37888], %acc_f32_11;
	st.global.f32	[%alu_s64_5+38912], %acc_f32_13;
	st.global.f32	[%alu_s64_5+39936], %acc_f32_15;
	st.global.f32	[%alu_s64_5+262144], %acc_f32_16;
	st.global.f32	[%alu_s64_5+263168], %acc_f32_18;
	st.global.f32	[%alu_s64_5+264192], %acc_f32_20;
	st.global.f32	[%alu_s64_5+265216], %acc_f32_22;
	st.global.f32	[%alu_s64_5+266240], %acc_f32_24;
	st.global.f32	[%alu_s64_5+267264], %acc_f32_26;
	st.global.f32	[%alu_s64_5+268288], %acc_f32_28;
	st.global.f32	[%alu_s64_5+269312], %acc_f32_30;
	st.global.f32	[%alu_s64_5+294912], %acc_f32_17;
	st.global.f32	[%alu_s64_5+295936], %acc_f32_19;
	st.global.f32	[%alu_s64_5+296960], %acc_f32_21;
	st.global.f32	[%alu_s64_5+297984], %acc_f32_23;
	st.global.f32	[%alu_s64_5+299008], %acc_f32_25;
	st.global.f32	[%alu_s64_5+300032], %acc_f32_27;
	st.global.f32	[%alu_s64_5+301056], %acc_f32_29;
	st.global.f32	[%alu_s64_5+302080], %acc_f32_31;
	st.global.f32	[%alu_s64_5+524288], %acc_f32_32;
	st.global.f32	[%alu_s64_5+525312], %acc_f32_34;
	st.global.f32	[%alu_s64_5+526336], %acc_f32_36;
	st.global.f32	[%alu_s64_5+527360], %acc_f32_38;
	st.global.f32	[%alu_s64_5+528384], %acc_f32_40;
	st.global.f32	[%alu_s64_5+529408], %acc_f32_42;
	st.global.f32	[%alu_s64_5+530432], %acc_f32_44;
	st.global.f32	[%alu_s64_5+531456], %acc_f32_46;
	st.global.f32	[%alu_s64_5+557056], %acc_f32_33;
	st.global.f32	[%alu_s64_5+558080], %acc_f32_35;
	st.global.f32	[%alu_s64_5+559104], %acc_f32_37;
	st.global.f32	[%alu_s64_5+560128], %acc_f32_39;
	st.global.f32	[%alu_s64_5+561152], %acc_f32_41;
	st.global.f32	[%alu_s64_5+562176], %acc_f32_43;
	st.global.f32	[%alu_s64_5+563200], %acc_f32_45;
	st.global.f32	[%alu_s64_5+564224], %acc_f32_47;
	st.global.f32	[%alu_s64_5+786432], %acc_f32_48;
	st.global.f32	[%alu_s64_5+787456], %acc_f32_50;
	st.global.f32	[%alu_s64_5+788480], %acc_f32_52;
	st.global.f32	[%alu_s64_5+789504], %acc_f32_54;
	st.global.f32	[%alu_s64_5+790528], %acc_f32_56;
	st.global.f32	[%alu_s64_5+791552], %acc_f32_58;
	st.global.f32	[%alu_s64_5+792576], %acc_f32_60;
	st.global.f32	[%alu_s64_5+793600], %acc_f32_62;
	st.global.f32	[%alu_s64_5+819200], %acc_f32_49;
	st.global.f32	[%alu_s64_5+820224], %acc_f32_51;
	st.global.f32	[%alu_s64_5+821248], %acc_f32_53;
	st.global.f32	[%alu_s64_5+822272], %acc_f32_55;
	st.global.f32	[%alu_s64_5+823296], %acc_f32_57;
	st.global.f32	[%alu_s64_5+824320], %acc_f32_59;
	st.global.f32	[%alu_s64_5+825344], %acc_f32_61;
	st.global.f32	[%alu_s64_5+826368], %acc_f32_63;
	ret;
}"""
    if False:
      return f""" .version VERSION
.target TARGET
.address_size 64
.visible .entry r_16_4_32_2_2_2_2_2_8_28_2_16_4_4_4n1(
	.param .u64 data0,
	.param .u64 data1,
	.param .u64 data2
)
{{
	.reg		.u64 %dat_u64_<3>;
	.reg		.f16 %const_f16_<1>;
	.reg		.s32 %alu_s32_<46>;
	.reg		.s64 %cast_s64_<3>;
	.reg		.s64 %alu_s64_<6>;
	.reg		.f32 %acc_f32_<64>;
	.reg		.s32 %ridx_s32_<3>;
	.reg		.f16 %val_f16_<48>;
	.reg		.pred %alu_pred_<2>;
	.reg		.b32 %wmma_b32_<96>;
	.reg		.f32 %wmma_f32_<64>;
	.reg		.pred %pred_pred_<3>;
	.reg		.u32 %lidx5;
	.reg		.u32 %gidx0;
	.reg		.u32 %lidx4;
	.reg		.u32 %gidx1;
	.reg		.u32 %lidx3;
	.reg		.u32 %gidx2;
  .reg    .b32 %r0;
	.reg		           .b32 %rem<3>;
	ld.param.u64	%dat_u64_0, [data0+0];
	ld.param.u64	%dat_u64_1, [data1+0];
	ld.param.u64	%dat_u64_2, [data2+0];
	mov.u32		%gidx2, %ctaid.x;
	mov.u32		%gidx0, %ctaid.z;
	mov.u32		%gidx1, %ctaid.y;
	mov.b32		%acc_f32_0, 0f00000000;
	mov.b32		%acc_f32_1, 0f00000000;
	mov.b32		%acc_f32_2, 0f00000000;
	mov.b32		%acc_f32_3, 0f00000000;
	mov.b32		%acc_f32_4, 0f00000000;
	mov.b32		%acc_f32_5, 0f00000000;
	mov.b32		%acc_f32_6, 0f00000000;
	mov.b32		%acc_f32_7, 0f00000000;
	mov.b32		%acc_f32_8, 0f00000000;
	mov.b32		%acc_f32_9, 0f00000000;
	mov.b32		%acc_f32_10, 0f00000000;
	mov.b32		%acc_f32_11, 0f00000000;
	mov.b32		%acc_f32_12, 0f00000000;
	mov.b32		%acc_f32_13, 0f00000000;
	mov.b32		%acc_f32_14, 0f00000000;
	mov.b32		%acc_f32_15, 0f00000000;
	mov.b32		%acc_f32_16, 0f00000000;
	mov.b32		%acc_f32_17, 0f00000000;
	mov.b32		%acc_f32_18, 0f00000000;
	mov.b32		%acc_f32_19, 0f00000000;
	mov.b32		%acc_f32_20, 0f00000000;
	mov.b32		%acc_f32_21, 0f00000000;
	mov.b32		%acc_f32_22, 0f00000000;
	mov.b32		%acc_f32_23, 0f00000000;
	mov.b32		%acc_f32_24, 0f00000000;
	mov.b32		%acc_f32_25, 0f00000000;
	mov.b32		%acc_f32_26, 0f00000000;
	mov.b32		%acc_f32_27, 0f00000000;
	mov.b32		%acc_f32_28, 0f00000000;
	mov.b32		%acc_f32_29, 0f00000000;
	mov.b32		%acc_f32_30, 0f00000000;
	mov.b32		%acc_f32_31, 0f00000000;
	mov.b32		%acc_f32_32, 0f00000000;
	mov.b32		%acc_f32_33, 0f00000000;
	mov.b32		%acc_f32_34, 0f00000000;
	mov.b32		%acc_f32_35, 0f00000000;
	mov.b32		%acc_f32_36, 0f00000000;
	mov.b32		%acc_f32_37, 0f00000000;
	mov.b32		%acc_f32_38, 0f00000000;
	mov.b32		%acc_f32_39, 0f00000000;
	mov.b32		%acc_f32_40, 0f00000000;
	mov.b32		%acc_f32_41, 0f00000000;
	mov.b32		%acc_f32_42, 0f00000000;
	mov.b32		%acc_f32_43, 0f00000000;
	mov.b32		%acc_f32_44, 0f00000000;
	mov.b32		%acc_f32_45, 0f00000000;
	mov.b32		%acc_f32_46, 0f00000000;
	mov.b32		%acc_f32_47, 0f00000000;
	mov.b32		%acc_f32_48, 0f00000000;
	mov.b32		%acc_f32_49, 0f00000000;
	mov.b32		%acc_f32_50, 0f00000000;
	mov.b32		%acc_f32_51, 0f00000000;
	mov.b32		%acc_f32_52, 0f00000000;
	mov.b32		%acc_f32_53, 0f00000000;
	mov.b32		%acc_f32_54, 0f00000000;
	mov.b32		%acc_f32_55, 0f00000000;
	mov.b32		%acc_f32_56, 0f00000000;
	mov.b32		%acc_f32_57, 0f00000000;
	mov.b32		%acc_f32_58, 0f00000000;
	mov.b32		%acc_f32_59, 0f00000000;
	mov.b32		%acc_f32_60, 0f00000000;
	mov.b32		%acc_f32_61, 0f00000000;
	mov.b32		%acc_f32_62, 0f00000000;
	mov.b32		%acc_f32_63, 0f00000000;
	mov.u32		%ridx_s32_0, 0;
$loop_0:
	mov.u32		%lidx5, %tid.z;
	mov.u32		%lidx4, %tid.y;
	mov.u32		%lidx3, %tid.x;
	shr.s32		%alu_s32_0, %lidx3, 1;
	shr.s32		%alu_s32_1, %lidx3, 2;
	shr.u32		 %rem0, %lidx3, 31;                                                         
	add.s32		 %rem1, %lidx3, %rem0;                                                       
	and.b32		  %rem2, %rem1, -2;                                                        
	sub.s32		 %alu_s32_2, %lidx3, %rem2;
	shl.b32		%alu_s32_3, %gidx1, 11;
	mul.lo.s32	%alu_s32_4, %gidx1, 200704;
	shl.b32		%alu_s32_5, %lidx4, 6;
	shl.b32		%alu_s32_6, %gidx0, 18;
	mul.lo.s32	%alu_s32_7, %gidx0, 25088;
	shl.b32		%alu_s32_8, %lidx5, 7;
	mad.lo.s32	%alu_s32_9, %lidx5, 12544, %alu_s32_4;
	shr.u32		 %rem0, %alu_s32_0, 31;                                                         
	add.s32		 %rem1, %alu_s32_0, %rem0;                                                       
	and.b32		  %rem2, %rem1, -2;                                                        
	sub.s32		 %alu_s32_10, %alu_s32_0, %rem2;
	mad.lo.s32	%alu_s32_12, %alu_s32_1, 784, %alu_s32_7;
	mad.lo.s32	%alu_s32_13, %lidx4, 1568, %alu_s32_12;
	mad.lo.s32	%alu_s32_14, %lidx5, 3136, %alu_s32_13;
	mad.lo.s32	%alu_s32_15, %gidx2, 3211264, %alu_s32_14;
	shl.b32		%alu_s32_16, %alu_s32_2, 1;
	shl.b32		%alu_s32_17, %alu_s32_2, 2;
	mad.lo.s32	%alu_s32_18, %alu_s32_1, 3136, %alu_s32_9;
	mad.lo.s32	%alu_s32_19, %lidx4, 6272, %alu_s32_18;
	mad.lo.s32	%alu_s32_20, %gidx2, 6422528, %alu_s32_19;
	shl.b32		%alu_s32_23, %alu_s32_10, 2;
	shl.b32		%alu_s32_24, %alu_s32_10, 3;
	mad.lo.s32	%alu_s32_32, %ridx_s32_0, 401408, %alu_s32_15;
	mad.lo.s32	%alu_s32_33, %ridx_s32_0, 802816, %alu_s32_20;
	mov.u32		%ridx_s32_1, 0;
$loop_1:
	mad.lo.s32	%alu_s32_34, %ridx_s32_1, 28, %alu_s32_32;
	mad.lo.s32	%alu_s32_35, %ridx_s32_1, 112, %alu_s32_33;
	mov.u32		%ridx_s32_2, 0;
$loop_2:
	shl.b32		%alu_s32_36, %ridx_s32_2, 4;
	add.s32		%alu_s32_37, %alu_s32_36, %alu_s32_34;
	add.s32		%alu_s32_38, %alu_s32_16, %alu_s32_37;
	add.s32		%alu_s32_39, %alu_s32_23, %alu_s32_38;
	cvt.s64.s32	%cast_s64_1, %alu_s32_39;
	shl.b64		%alu_s64_2, %cast_s64_1, 1;
	add.s64		%alu_s64_3, %alu_s64_2, %dat_u64_2;
	ld.global.b16	%val_f16_0, [%alu_s64_3+0];
	ld.global.b16	%val_f16_1, [%alu_s64_3+2];
	ld.global.b16	%val_f16_2, [%alu_s64_3+12544];
	ld.global.b16	%val_f16_3, [%alu_s64_3+12546];
	ld.global.b16	%val_f16_4, [%alu_s64_3+25088];
	ld.global.b16	%val_f16_5, [%alu_s64_3+25090];
	ld.global.b16	%val_f16_6, [%alu_s64_3+37632];
	ld.global.b16	%val_f16_7, [%alu_s64_3+37634];
	add.s32		%alu_s32_40, %alu_s32_36, %alu_s32_16;
	add.s32		%alu_s32_41, %alu_s32_23, %alu_s32_40;
	setp.lt.s32	%alu_pred_0, %alu_s32_41, 19;
	mov.b16		%const_f16_0, 0x0000;
  @!%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  ld.global.b16 %val_f16_8, [%alu_s64_3+18];
  bra ${(b2:=ssa_label("L"))};
  ${b1}:
  mov.b16 %val_f16_8, %const_f16_0;
  ${b2}:
  @!%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  ld.global.b16 %val_f16_9, [%alu_s64_3+12562];
  bra ${(b2:=ssa_label("L"))};
  ${b1}:
  mov.b16 %val_f16_9, %const_f16_0;
  ${b2}:
  @!%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  ld.global.b16 %val_f16_10, [%alu_s64_3+25106];
  bra ${(b2:=ssa_label("L"))};
  ${b1}:
  mov.b16 %val_f16_10, %const_f16_0;
  ${b2}:
  //
  @!%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  ld.global.b16 %val_f16_11, [%alu_s64_3+37650];
  bra ${(b2:=ssa_label("L"))};
  ${b1}:
  mov.b16 %val_f16_11, %const_f16_0;
  ${b2}:
  //
  //
	setp.lt.s32	%alu_pred_1, %alu_s32_41, 20;
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
  ld.global.b16 %val_f16_12, [%alu_s64_3+16];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
  mov.b16 %val_f16_12, %const_f16_0;
  ${b3}:
  //
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
  ld.global.b16 %val_f16_13, [%alu_s64_3+12560];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
  mov.b16 %val_f16_13, %const_f16_0;
  ${b3}:
  //
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
  ld.global.b16 %val_f16_14, [%alu_s64_3+25104];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
  mov.b16 %val_f16_14, %const_f16_0;
  ${b3}:
  //
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
  ld.global.b16 %val_f16_15, [%alu_s64_3+37648];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
  mov.b16 %val_f16_15, %const_f16_0;
  ${b3}:
  //
	shl.b32		%alu_s32_42, %ridx_s32_2, 5;
	add.s32		%alu_s32_43, %alu_s32_42, %alu_s32_35;
	add.s32		%alu_s32_44, %alu_s32_17, %alu_s32_43;
	add.s32		%alu_s32_45, %alu_s32_24, %alu_s32_44;
	cvt.s64.s32	%cast_s64_2, %alu_s32_45;
	shl.b64		%alu_s64_4, %cast_s64_2, 1;
	add.s64		%alu_s64_5, %alu_s64_4, %dat_u64_1;
	ld.global.b16	%val_f16_16, [%alu_s64_5+0];
	ld.global.b16	%val_f16_17, [%alu_s64_5+4];
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
	ld.global.b16 %val_f16_18, [%alu_s64_5+32];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
	mov.b16 %val_f16_18, %const_f16_0;
  ${b3}:
  //
  @%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
  ld.global.b16 %val_f16_19, [%alu_s64_5+36];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
	mov.b16 %val_f16_19, %const_f16_0;
  ${b3}:
	ld.global.b16	%val_f16_20, [%alu_s64_5+50176];
	ld.global.b16	%val_f16_21, [%alu_s64_5+50180];
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
  ld.global.b16 %val_f16_22, [%alu_s64_5+50208];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
  mov.b16 %val_f16_22, %const_f16_0;
  ${b3}:
  @%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
	ld.global.b16 %val_f16_23, [%alu_s64_5+50212];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
	mov.b16 %val_f16_23, %const_f16_0;
  ${b3}:
	mov.b32		%wmma_b32_0, {{%val_f16_16, %val_f16_17}};
	mov.b32		%wmma_b32_1, {{%val_f16_20, %val_f16_21}};
	mov.b32		%wmma_b32_2, {{%val_f16_18, %val_f16_19}};
	mov.b32		%wmma_b32_3, {{%val_f16_22, %val_f16_23}};
	mov.b32		%wmma_b32_4, {{%val_f16_0, %val_f16_1}};
	mov.b32		%wmma_b32_5, {{%val_f16_12, %val_f16_8}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_0, %wmma_f32_1, %wmma_f32_2, %wmma_f32_3}}, {{%wmma_b32_0, %wmma_b32_1, %wmma_b32_2, %wmma_b32_3}}, {{%wmma_b32_4, %wmma_b32_5}}, {{%acc_f32_0, %acc_f32_1, %acc_f32_2, %acc_f32_3}};
	mov.b32		%wmma_b32_6, {{%val_f16_16, %val_f16_17}};
	mov.b32		%wmma_b32_7, {{%val_f16_20, %val_f16_21}};
	mov.b32		%wmma_b32_8, {{%val_f16_18, %val_f16_19}};
	mov.b32		%wmma_b32_9, {{%val_f16_22, %val_f16_23}};
	mov.b32		%wmma_b32_10, {{%val_f16_2, %val_f16_3}};
	mov.b32		%wmma_b32_11, {{%val_f16_13, %val_f16_9}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_4, %wmma_f32_5, %wmma_f32_6, %wmma_f32_7}}, {{%wmma_b32_6, %wmma_b32_7, %wmma_b32_8, %wmma_b32_9}}, {{%wmma_b32_10, %wmma_b32_11}}, {{%acc_f32_16, %acc_f32_17, %acc_f32_18, %acc_f32_19}};
	mov.b32		%wmma_b32_12, {{%val_f16_16, %val_f16_17}};
	mov.b32		%wmma_b32_13, {{%val_f16_20, %val_f16_21}};
	mov.b32		%wmma_b32_14, {{%val_f16_18, %val_f16_19}};
	mov.b32		%wmma_b32_15, {{%val_f16_22, %val_f16_23}};
	mov.b32		%wmma_b32_16, {{%val_f16_4, %val_f16_5}};
	mov.b32		%wmma_b32_17, {{%val_f16_14, %val_f16_10}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_8, %wmma_f32_9, %wmma_f32_10, %wmma_f32_11}}, {{%wmma_b32_12, %wmma_b32_13, %wmma_b32_14, %wmma_b32_15}}, {{%wmma_b32_16, %wmma_b32_17}}, {{%acc_f32_32, %acc_f32_33, %acc_f32_34, %acc_f32_35}};
	mov.b32		%wmma_b32_18, {{%val_f16_16, %val_f16_17}};
	mov.b32		%wmma_b32_19, {{%val_f16_20, %val_f16_21}};
	mov.b32		%wmma_b32_20, {{%val_f16_18, %val_f16_19}};
	mov.b32		%wmma_b32_21, {{%val_f16_22, %val_f16_23}};
	mov.b32		%wmma_b32_22, {{%val_f16_6, %val_f16_7}};
	mov.b32		%wmma_b32_23, {{%val_f16_15, %val_f16_11}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_12, %wmma_f32_13, %wmma_f32_14, %wmma_f32_15}}, {{%wmma_b32_18, %wmma_b32_19, %wmma_b32_20, %wmma_b32_21}}, {{%wmma_b32_22, %wmma_b32_23}}, {{%acc_f32_48, %acc_f32_49, %acc_f32_50, %acc_f32_51}};
	ld.global.b16	%val_f16_24, [%alu_s64_5+100352];
	ld.global.b16	%val_f16_25, [%alu_s64_5+100356];
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
	ld.global.b16 %val_f16_26, [%alu_s64_5+100384];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
	mov.b16 %val_f16_26, %const_f16_0;
  ${b3}:
  @%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
  ld.global.b16 %val_f16_27, [%alu_s64_5+100388];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
  mov.b16 %val_f16_27, %const_f16_0;
  ${b3}:
  ld.global.b16	%val_f16_29, [%alu_s64_5+150532];
	ld.global.b16	%val_f16_28, [%alu_s64_5+150528];
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
	ld.global.b16 %val_f16_30, [%alu_s64_5+150560];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
	mov.b16 %val_f16_30, %const_f16_0;
  ${b3}:
  @%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
		ld.global.b16 %val_f16_31, [%alu_s64_5+150564];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
		mov.b16 %val_f16_31, %const_f16_0;
  ${b3}:
	mov.b32		%wmma_b32_24, {{%val_f16_24, %val_f16_25}};
	mov.b32		%wmma_b32_25, {{%val_f16_28, %val_f16_29}};
	mov.b32		%wmma_b32_26, {{%val_f16_26, %val_f16_27}};
	mov.b32		%wmma_b32_27, {{%val_f16_30, %val_f16_31}};
	mov.b32		%wmma_b32_28, {{%val_f16_0, %val_f16_1}};
	mov.b32		%wmma_b32_29, {{%val_f16_12, %val_f16_8}};
  mov.b32		%wmma_b32_30, {{%val_f16_24, %val_f16_25}};
  mov.b32		%wmma_b32_31, {{%val_f16_28, %val_f16_29}};
  mov.b32		%wmma_b32_32, {{%val_f16_26, %val_f16_27}};
  mov.b32		%wmma_b32_33, {{%val_f16_30, %val_f16_31}};
  mov.b32		%wmma_b32_34, {{%val_f16_2, %val_f16_3}};
  mov.b32		%wmma_b32_35, {{%val_f16_13, %val_f16_9}};
  mov.b32		%wmma_b32_36, {{%val_f16_24, %val_f16_25}};
  mov.b32		%wmma_b32_37, {{%val_f16_28, %val_f16_29}};
  mov.b32		%wmma_b32_38, {{%val_f16_26, %val_f16_27}};
  mov.b32		%wmma_b32_39, {{%val_f16_30, %val_f16_31}};
  mov.b32		%wmma_b32_40, {{%val_f16_4, %val_f16_5}};
  mov.b32		%wmma_b32_41, {{%val_f16_14, %val_f16_10}};
  mov.b32		%wmma_b32_42, {{%val_f16_24, %val_f16_25}};
  mov.b32		%wmma_b32_43, {{%val_f16_28, %val_f16_29}};
  mov.b32		%wmma_b32_44, {{%val_f16_26, %val_f16_27}};
  mov.b32		%wmma_b32_45, {{%val_f16_30, %val_f16_31}};
  mov.b32		%wmma_b32_46, {{%val_f16_6, %val_f16_7}};
  mov.b32		%wmma_b32_47, {{%val_f16_15, %val_f16_11}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_16, %wmma_f32_17, %wmma_f32_18, %wmma_f32_19}}, {{%wmma_b32_24, %wmma_b32_25, %wmma_b32_26, %wmma_b32_27}}, {{%wmma_b32_28, %wmma_b32_29}}, {{%acc_f32_4, %acc_f32_5, %acc_f32_6, %acc_f32_7}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_20, %wmma_f32_21, %wmma_f32_22, %wmma_f32_23}}, {{%wmma_b32_30, %wmma_b32_31, %wmma_b32_32, %wmma_b32_33}}, {{%wmma_b32_34, %wmma_b32_35}}, {{%acc_f32_20, %acc_f32_21, %acc_f32_22, %acc_f32_23}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_24, %wmma_f32_25, %wmma_f32_26, %wmma_f32_27}}, {{%wmma_b32_36, %wmma_b32_37, %wmma_b32_38, %wmma_b32_39}}, {{%wmma_b32_40, %wmma_b32_41}}, {{%acc_f32_36, %acc_f32_37, %acc_f32_38, %acc_f32_39}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_28, %wmma_f32_29, %wmma_f32_30, %wmma_f32_31}}, {{%wmma_b32_42, %wmma_b32_43, %wmma_b32_44, %wmma_b32_45}}, {{%wmma_b32_46, %wmma_b32_47}}, {{%acc_f32_52, %acc_f32_53, %acc_f32_54, %acc_f32_55}};
	ld.global.b16	%val_f16_32, [%alu_s64_5+200704];
	ld.global.b16	%val_f16_33, [%alu_s64_5+200708];
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
    ld.global.b16 %val_f16_34, [%alu_s64_5+200736];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
  	mov.b16 %val_f16_34, %const_f16_0;
  ${b3}:
  @%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
   	ld.global.b16 %val_f16_35, [%alu_s64_5+200740];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
   	mov.b16 %val_f16_35, %const_f16_0;
  ${b3}:
	ld.global.b16	%val_f16_36, [%alu_s64_5+250880];
	ld.global.b16	%val_f16_37, [%alu_s64_5+250884];
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
    ld.global.b16 %val_f16_38, [%alu_s64_5+250912];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
    mov.b16 %val_f16_38, %const_f16_0;
  ${b3}:
  @%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
 	ld.global.b16 %val_f16_39, [%alu_s64_5+250916];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
    mov.b16 %val_f16_39, %const_f16_0;
  ${b3}:
	mov.b32		%wmma_b32_48, {{%val_f16_32, %val_f16_33}};
	mov.b32		%wmma_b32_49, {{%val_f16_36, %val_f16_37}};
	mov.b32		%wmma_b32_50, {{%val_f16_34, %val_f16_35}};
	mov.b32		%wmma_b32_51, {{%val_f16_38, %val_f16_39}};
	mov.b32		%wmma_b32_52, {{%val_f16_0, %val_f16_1}};
	mov.b32		%wmma_b32_53, {{%val_f16_12, %val_f16_8}};
  mov.b32		%wmma_b32_54, {{%val_f16_32, %val_f16_33}};
  mov.b32		%wmma_b32_55, {{%val_f16_36, %val_f16_37}};
  mov.b32		%wmma_b32_56, {{%val_f16_34, %val_f16_35}};
  mov.b32		%wmma_b32_57, {{%val_f16_38, %val_f16_39}};
  mov.b32		%wmma_b32_58, {{%val_f16_2, %val_f16_3}};
  mov.b32		%wmma_b32_59, {{%val_f16_13, %val_f16_9}};
  mov.b32		%wmma_b32_60, {{%val_f16_32, %val_f16_33}};
  mov.b32		%wmma_b32_61, {{%val_f16_36, %val_f16_37}};
  mov.b32		%wmma_b32_62, {{%val_f16_34, %val_f16_35}};
  mov.b32		%wmma_b32_63, {{%val_f16_38, %val_f16_39}};
  mov.b32		%wmma_b32_64, {{%val_f16_4, %val_f16_5}};
  mov.b32		%wmma_b32_65, {{%val_f16_14, %val_f16_10}};
  mov.b32		%wmma_b32_66, {{%val_f16_32, %val_f16_33}};
  mov.b32		%wmma_b32_67, {{%val_f16_36, %val_f16_37}};
  mov.b32		%wmma_b32_68, {{%val_f16_34, %val_f16_35}};
  mov.b32		%wmma_b32_69, {{%val_f16_38, %val_f16_39}};
  mov.b32		%wmma_b32_70, {{%val_f16_6, %val_f16_7}};
  mov.b32		%wmma_b32_71, {{%val_f16_15, %val_f16_11}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_32, %wmma_f32_33, %wmma_f32_34, %wmma_f32_35}}, {{%wmma_b32_48, %wmma_b32_49, %wmma_b32_50, %wmma_b32_51}}, {{%wmma_b32_52, %wmma_b32_53}}, {{%acc_f32_8, %acc_f32_9, %acc_f32_10, %acc_f32_11}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_36, %wmma_f32_37, %wmma_f32_38, %wmma_f32_39}}, {{%wmma_b32_54, %wmma_b32_55, %wmma_b32_56, %wmma_b32_57}}, {{%wmma_b32_58, %wmma_b32_59}}, {{%acc_f32_24, %acc_f32_25, %acc_f32_26, %acc_f32_27}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_40, %wmma_f32_41, %wmma_f32_42, %wmma_f32_43}}, {{%wmma_b32_60, %wmma_b32_61, %wmma_b32_62, %wmma_b32_63}}, {{%wmma_b32_64, %wmma_b32_65}}, {{%acc_f32_40, %acc_f32_41, %acc_f32_42, %acc_f32_43}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_44, %wmma_f32_45, %wmma_f32_46, %wmma_f32_47}}, {{%wmma_b32_66, %wmma_b32_67, %wmma_b32_68, %wmma_b32_69}}, {{%wmma_b32_70, %wmma_b32_71}}, {{%acc_f32_56, %acc_f32_57, %acc_f32_58, %acc_f32_59}};
	ld.global.b16	%val_f16_40, [%alu_s64_5+301056];
	ld.global.b16	%val_f16_41, [%alu_s64_5+301060];
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
	ld.global.b16 %val_f16_42, [%alu_s64_5+301088];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
	mov.b16 %val_f16_42, %const_f16_0;
  ${b3}:
  @%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
    ld.global.b16 %val_f16_43, [%alu_s64_5+301092];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
    mov.b16 %val_f16_43, %const_f16_0;
  ${b3}:
	ld.global.b16	%val_f16_44, [%alu_s64_5+351232];
	ld.global.b16	%val_f16_45, [%alu_s64_5+351236];
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
	ld.global.b16 %val_f16_46, [%alu_s64_5+351264];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
	mov.b16 %val_f16_46, %const_f16_0;
  ${b3}:
  @%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
    ld.global.b16 %val_f16_47, [%alu_s64_5+351268];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
    mov.b16 %val_f16_47, %const_f16_0;
  ${b3}:
  mov.u32  %r0, 1;
	mov.b32		%wmma_b32_72, {{%val_f16_40, %val_f16_41}};
	mov.b32		%wmma_b32_73, {{%val_f16_44, %val_f16_45}};
	mov.b32		%wmma_b32_74, {{%val_f16_42, %val_f16_43}};
	mov.b32		%wmma_b32_75, {{%val_f16_46, %val_f16_47}};
	mov.b32		%wmma_b32_76, {{%val_f16_0, %val_f16_1}};
	mov.b32		%wmma_b32_77, {{%val_f16_12, %val_f16_8}};
  mov.b32		%wmma_b32_78, {{%val_f16_40, %val_f16_41}};
  mov.b32		%wmma_b32_79, {{%val_f16_44, %val_f16_45}};
  mov.b32		%wmma_b32_80, {{%val_f16_42, %val_f16_43}};
  mov.b32		%wmma_b32_81, {{%val_f16_46, %val_f16_47}};
  mov.b32		%wmma_b32_82, {{%val_f16_2, %val_f16_3}};
  mov.b32		%wmma_b32_83, {{%val_f16_13, %val_f16_9}};
  mov.b32		%wmma_b32_84, {{%val_f16_40, %val_f16_41}};
  mov.b32		%wmma_b32_85, {{%val_f16_44, %val_f16_45}};
  mov.b32		%wmma_b32_86, {{%val_f16_42, %val_f16_43}};
  mov.b32		%wmma_b32_87, {{%val_f16_46, %val_f16_47}};
  mov.b32		%wmma_b32_88, {{%val_f16_4, %val_f16_5}};
  mov.b32		%wmma_b32_89, {{%val_f16_14, %val_f16_10}};
  mov.b32		%wmma_b32_90, {{%val_f16_40, %val_f16_41}};
  mov.b32		%wmma_b32_91, {{%val_f16_44, %val_f16_45}};
  mov.b32		%wmma_b32_92, {{%val_f16_42, %val_f16_43}};
  mov.b32		%wmma_b32_93, {{%val_f16_46, %val_f16_47}};
  mov.b32		%wmma_b32_94, {{%val_f16_6, %val_f16_7}};
  mov.b32		%wmma_b32_95, {{%val_f16_15, %val_f16_11}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_48, %wmma_f32_49, %wmma_f32_50, %wmma_f32_51}}, {{%wmma_b32_72, %wmma_b32_73, %wmma_b32_74, %wmma_b32_75}}, {{%wmma_b32_76, %wmma_b32_77}}, {{%acc_f32_12, %acc_f32_13, %acc_f32_14, %acc_f32_15}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_52, %wmma_f32_53, %wmma_f32_54, %wmma_f32_55}}, {{%wmma_b32_78, %wmma_b32_79, %wmma_b32_80, %wmma_b32_81}}, {{%wmma_b32_82, %wmma_b32_83}}, {{%acc_f32_28, %acc_f32_29, %acc_f32_30, %acc_f32_31}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_56, %wmma_f32_57, %wmma_f32_58, %wmma_f32_59}}, {{%wmma_b32_84, %wmma_b32_85, %wmma_b32_86, %wmma_b32_87}}, {{%wmma_b32_88, %wmma_b32_89}}, {{%acc_f32_44, %acc_f32_45, %acc_f32_46, %acc_f32_47}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_60, %wmma_f32_61, %wmma_f32_62, %wmma_f32_63}}, {{%wmma_b32_90, %wmma_b32_91, %wmma_b32_92, %wmma_b32_93}}, {{%wmma_b32_94, %wmma_b32_95}}, {{%acc_f32_60, %acc_f32_61, %acc_f32_62, %acc_f32_63}};
	mov.b32		%acc_f32_0, %wmma_f32_0;
	mov.b32		%acc_f32_1, %wmma_f32_1;
	mov.b32		%acc_f32_2, %wmma_f32_2;
	mov.b32		%acc_f32_3, %wmma_f32_3;
	mov.b32		%acc_f32_4, %wmma_f32_16;
	mov.b32		%acc_f32_5, %wmma_f32_17;
	mov.b32		%acc_f32_6, %wmma_f32_18;
	mov.b32		%acc_f32_7, %wmma_f32_19;
	mov.b32		%acc_f32_8, %wmma_f32_32;
	mov.b32		%acc_f32_9, %wmma_f32_33;
	mov.b32		%acc_f32_10, %wmma_f32_34;
	mov.b32		%acc_f32_11, %wmma_f32_35;
	mov.b32		%acc_f32_12, %wmma_f32_48;
	mov.b32		%acc_f32_13, %wmma_f32_49;
	mov.b32		%acc_f32_14, %wmma_f32_50;
	mov.b32		%acc_f32_15, %wmma_f32_51;
	mov.b32		%acc_f32_16, %wmma_f32_4;
	mov.b32		%acc_f32_17, %wmma_f32_5;
	mov.b32		%acc_f32_18, %wmma_f32_6;
	mov.b32		%acc_f32_19, %wmma_f32_7;
	mov.b32		%acc_f32_20, %wmma_f32_20;
	mov.b32		%acc_f32_21, %wmma_f32_21;
	mov.b32		%acc_f32_22, %wmma_f32_22;
	mov.b32		%acc_f32_23, %wmma_f32_23;
	mov.b32		%acc_f32_24, %wmma_f32_36;
	mov.b32		%acc_f32_25, %wmma_f32_37;
	mov.b32		%acc_f32_26, %wmma_f32_38;
	mov.b32		%acc_f32_27, %wmma_f32_39;
	mov.b32		%acc_f32_28, %wmma_f32_52;
	mov.b32		%acc_f32_29, %wmma_f32_53;
	mov.b32		%acc_f32_30, %wmma_f32_54;
	mov.b32		%acc_f32_31, %wmma_f32_55;
	mov.b32		%acc_f32_32, %wmma_f32_8;
	mov.b32		%acc_f32_33, %wmma_f32_9;
	mov.b32		%acc_f32_34, %wmma_f32_10;
	mov.b32		%acc_f32_35, %wmma_f32_11;
	mov.b32		%acc_f32_36, %wmma_f32_24;
	mov.b32		%acc_f32_37, %wmma_f32_25;
	mov.b32		%acc_f32_38, %wmma_f32_26;
	mov.b32		%acc_f32_39, %wmma_f32_27;
	mov.b32		%acc_f32_40, %wmma_f32_40;
	mov.b32		%acc_f32_41, %wmma_f32_41;
	mov.b32		%acc_f32_42, %wmma_f32_42;
	mov.b32		%acc_f32_43, %wmma_f32_43;
	mov.b32		%acc_f32_44, %wmma_f32_56;
	mov.b32		%acc_f32_45, %wmma_f32_57;
	mov.b32		%acc_f32_46, %wmma_f32_58;
	mov.b32		%acc_f32_47, %wmma_f32_59;
	mov.b32		%acc_f32_48, %wmma_f32_12;
	mov.b32		%acc_f32_49, %wmma_f32_13;
	mov.b32		%acc_f32_50, %wmma_f32_14;
	mov.b32		%acc_f32_51, %wmma_f32_15;
	mov.b32		%acc_f32_52, %wmma_f32_28;
	mov.b32		%acc_f32_53, %wmma_f32_29;
	mov.b32		%acc_f32_54, %wmma_f32_30;
	mov.b32		%acc_f32_55, %wmma_f32_31;
	mov.b32		%acc_f32_56, %wmma_f32_44;
	mov.b32		%acc_f32_57, %wmma_f32_45;
	mov.b32		%acc_f32_58, %wmma_f32_46;
	mov.b32		%acc_f32_59, %wmma_f32_47;
	mov.b32		%acc_f32_60, %wmma_f32_60;
	mov.b32		%acc_f32_61, %wmma_f32_61;
	mov.b32		%acc_f32_62, %wmma_f32_62;
	mov.b32		%acc_f32_63, %wmma_f32_63;
	add.s32		%ridx_s32_2, %ridx_s32_2, 1;
	setp.lt.u32	%pred_pred_0, %ridx_s32_2, 2;
  mov.u32     %ridx_s32_2, %r0;
	@%pred_pred_0	bra $loop_2;
	add.s32		%ridx_s32_1, %ridx_s32_1, 1;
	setp.lt.u32	%pred_pred_1, %ridx_s32_1, 28;
	@%pred_pred_1	bra $loop_1;
	add.s32		%ridx_s32_0, %ridx_s32_0, 1;
	setp.lt.u32	%pred_pred_2, %ridx_s32_0, 8;
	@%pred_pred_2	bra $loop_0;
	mov.u32		%lidx3, %tid.x;
	shr.s32		%alu_s32_0, %lidx3, 1;
	shr.s32		%alu_s32_1, %lidx3, 2;
	shr.u32		 %rem0, %lidx3, 31;                                                         
	add.s32		 %rem1, %lidx3, %rem0;                                                       
	and.b32		  %rem2, %rem1, -2;                                                        
	sub.s32		 %alu_s32_2, %lidx3, %rem2;
	shr.u32		 %rem0, %alu_s32_0, 31;                                                         
	add.s32		 %rem1, %alu_s32_0, %rem0;                                                       
	and.b32		  %rem2, %rem1, -2;                                                        
	sub.s32		 %alu_s32_10, %alu_s32_0, %rem2;
	shl.b32		%alu_s32_11, %alu_s32_1, 5;
	shl.b32		%alu_s32_21, %alu_s32_2, 14;
	add.s32		%alu_s32_22, %alu_s32_6, %alu_s32_3;
	shl.b32		%alu_s32_25, %alu_s32_10, 15;
	add.s32		%alu_s32_26, %alu_s32_22, %gidx2;
	add.s32		%alu_s32_27, %alu_s32_21, %alu_s32_26;
	add.s32		%alu_s32_28, %alu_s32_8, %alu_s32_27;
	add.s32		%alu_s32_29, %alu_s32_25, %alu_s32_28;
	add.s32		%alu_s32_30, %alu_s32_11, %alu_s32_29;
	add.s32		%alu_s32_31, %alu_s32_5, %alu_s32_30;
	cvt.s64.s32	%cast_s64_0, %alu_s32_31;
	shl.b64		%alu_s64_0, %cast_s64_0, 2;
	add.s64		%alu_s64_1, %alu_s64_0, %dat_u64_0;
	st.global.f32	[%alu_s64_1+0], %acc_f32_0;
	st.global.f32	[%alu_s64_1+1024], %acc_f32_2;
	st.global.f32	[%alu_s64_1+2048], %acc_f32_4;
	st.global.f32	[%alu_s64_1+3072], %acc_f32_6;
	st.global.f32	[%alu_s64_1+4096], %acc_f32_8;
	st.global.f32	[%alu_s64_1+5120], %acc_f32_10;
	st.global.f32	[%alu_s64_1+6144], %acc_f32_12;
	st.global.f32	[%alu_s64_1+7168], %acc_f32_14;
	st.global.f32	[%alu_s64_1+32768], %acc_f32_1;
	st.global.f32	[%alu_s64_1+33792], %acc_f32_3;
	st.global.f32	[%alu_s64_1+34816], %acc_f32_5;
	st.global.f32	[%alu_s64_1+35840], %acc_f32_7;
	st.global.f32	[%alu_s64_1+36864], %acc_f32_9;
	st.global.f32	[%alu_s64_1+37888], %acc_f32_11;
	st.global.f32	[%alu_s64_1+38912], %acc_f32_13;
	st.global.f32	[%alu_s64_1+39936], %acc_f32_15;
	st.global.f32	[%alu_s64_1+262144], %acc_f32_16;
	st.global.f32	[%alu_s64_1+263168], %acc_f32_18;
	st.global.f32	[%alu_s64_1+264192], %acc_f32_20;
	st.global.f32	[%alu_s64_1+265216], %acc_f32_22;
	st.global.f32	[%alu_s64_1+266240], %acc_f32_24;
	st.global.f32	[%alu_s64_1+267264], %acc_f32_26;
	st.global.f32	[%alu_s64_1+268288], %acc_f32_28;
	st.global.f32	[%alu_s64_1+269312], %acc_f32_30;
	st.global.f32	[%alu_s64_1+294912], %acc_f32_17;
	st.global.f32	[%alu_s64_1+295936], %acc_f32_19;
	st.global.f32	[%alu_s64_1+296960], %acc_f32_21;
	st.global.f32	[%alu_s64_1+297984], %acc_f32_23;
	st.global.f32	[%alu_s64_1+299008], %acc_f32_25;
	st.global.f32	[%alu_s64_1+300032], %acc_f32_27;
	st.global.f32	[%alu_s64_1+301056], %acc_f32_29;
	st.global.f32	[%alu_s64_1+302080], %acc_f32_31;
	st.global.f32	[%alu_s64_1+524288], %acc_f32_32;
	st.global.f32	[%alu_s64_1+525312], %acc_f32_34;
	st.global.f32	[%alu_s64_1+526336], %acc_f32_36;
	st.global.f32	[%alu_s64_1+527360], %acc_f32_38;
	st.global.f32	[%alu_s64_1+528384], %acc_f32_40;
	st.global.f32	[%alu_s64_1+529408], %acc_f32_42;
	st.global.f32	[%alu_s64_1+530432], %acc_f32_44;
	st.global.f32	[%alu_s64_1+531456], %acc_f32_46;
	st.global.f32	[%alu_s64_1+557056], %acc_f32_33;
	st.global.f32	[%alu_s64_1+558080], %acc_f32_35;
	st.global.f32	[%alu_s64_1+559104], %acc_f32_37;
	st.global.f32	[%alu_s64_1+560128], %acc_f32_39;
	st.global.f32	[%alu_s64_1+561152], %acc_f32_41;
	st.global.f32	[%alu_s64_1+562176], %acc_f32_43;
	st.global.f32	[%alu_s64_1+563200], %acc_f32_45;
	st.global.f32	[%alu_s64_1+564224], %acc_f32_47;
	st.global.f32	[%alu_s64_1+786432], %acc_f32_48;
	st.global.f32	[%alu_s64_1+787456], %acc_f32_50;
	st.global.f32	[%alu_s64_1+788480], %acc_f32_52;
	st.global.f32	[%alu_s64_1+789504], %acc_f32_54;
	st.global.f32	[%alu_s64_1+790528], %acc_f32_56;
	st.global.f32	[%alu_s64_1+791552], %acc_f32_58;
	st.global.f32	[%alu_s64_1+792576], %acc_f32_60;
	st.global.f32	[%alu_s64_1+793600], %acc_f32_62;
	st.global.f32	[%alu_s64_1+819200], %acc_f32_49;
	st.global.f32	[%alu_s64_1+820224], %acc_f32_51;
	st.global.f32	[%alu_s64_1+821248], %acc_f32_53;
	st.global.f32	[%alu_s64_1+822272], %acc_f32_55;
	st.global.f32	[%alu_s64_1+823296], %acc_f32_57;
	st.global.f32	[%alu_s64_1+824320], %acc_f32_59;
	st.global.f32	[%alu_s64_1+825344], %acc_f32_61;
	st.global.f32	[%alu_s64_1+826368], %acc_f32_63;
	ret;
}}"""

    if False:
      return f""" .version VERSION
.target TARGET
.address_size 64
.visible .entry r_16_4_32_2_2_2_2_2_8_28_2_16_4_4_4n1(
	.param .u64 data0,
	.param .u64 data1,
	.param .u64 data2
)
{{
	.reg		.u64 %dat_u64_<3>;
	.reg		.f16 %const_f16_<1>;
	.reg		.s32 %alu_s32_<46>;
	.reg		.s64 %cast_s64_<3>;
	.reg		.s64 %alu_s64_<6>;
	.reg		.f32 %acc_f32_<64>;
	.reg		.s32 %ridx_s32_<3>;
	.reg		.f16 %val_f16_<48>;
	.reg		.pred %alu_pred_<2>;
	.reg		.b32 %wmma_b32_<96>;
	.reg		.f32 %wmma_f32_<64>;
	.reg		.pred %pred_pred_<3>;
	.reg		.u32 %lidx5;
	.reg		.u32 %gidx0;
	.reg		.u32 %lidx4;
	.reg		.u32 %gidx1;
	.reg		.u32 %lidx3;
	.reg		.u32 %gidx2;
  .reg    .b32 %r0;
	.reg		           .b32 %rem<3>;
	ld.param.u64	%dat_u64_0, [data0+0];
	ld.param.u64	%dat_u64_1, [data1+0];
	ld.param.u64	%dat_u64_2, [data2+0];
	mov.u32		%gidx2, %ctaid.x;
	mov.u32		%gidx0, %ctaid.z;
	mov.u32		%gidx1, %ctaid.y;
	mov.b32		%acc_f32_0, 0f00000000;
	mov.b32		%acc_f32_1, 0f00000000;
	mov.b32		%acc_f32_2, 0f00000000;
	mov.b32		%acc_f32_3, 0f00000000;
	mov.b32		%acc_f32_4, 0f00000000;
	mov.b32		%acc_f32_5, 0f00000000;
	mov.b32		%acc_f32_6, 0f00000000;
	mov.b32		%acc_f32_7, 0f00000000;
	mov.b32		%acc_f32_8, 0f00000000;
	mov.b32		%acc_f32_9, 0f00000000;
	mov.b32		%acc_f32_10, 0f00000000;
	mov.b32		%acc_f32_11, 0f00000000;
	mov.b32		%acc_f32_12, 0f00000000;
	mov.b32		%acc_f32_13, 0f00000000;
	mov.b32		%acc_f32_14, 0f00000000;
	mov.b32		%acc_f32_15, 0f00000000;
	mov.b32		%acc_f32_16, 0f00000000;
	mov.b32		%acc_f32_17, 0f00000000;
	mov.b32		%acc_f32_18, 0f00000000;
	mov.b32		%acc_f32_19, 0f00000000;
	mov.b32		%acc_f32_20, 0f00000000;
	mov.b32		%acc_f32_21, 0f00000000;
	mov.b32		%acc_f32_22, 0f00000000;
	mov.b32		%acc_f32_23, 0f00000000;
	mov.b32		%acc_f32_24, 0f00000000;
	mov.b32		%acc_f32_25, 0f00000000;
	mov.b32		%acc_f32_26, 0f00000000;
	mov.b32		%acc_f32_27, 0f00000000;
	mov.b32		%acc_f32_28, 0f00000000;
	mov.b32		%acc_f32_29, 0f00000000;
	mov.b32		%acc_f32_30, 0f00000000;
	mov.b32		%acc_f32_31, 0f00000000;
	mov.b32		%acc_f32_32, 0f00000000;
	mov.b32		%acc_f32_33, 0f00000000;
	mov.b32		%acc_f32_34, 0f00000000;
	mov.b32		%acc_f32_35, 0f00000000;
	mov.b32		%acc_f32_36, 0f00000000;
	mov.b32		%acc_f32_37, 0f00000000;
	mov.b32		%acc_f32_38, 0f00000000;
	mov.b32		%acc_f32_39, 0f00000000;
	mov.b32		%acc_f32_40, 0f00000000;
	mov.b32		%acc_f32_41, 0f00000000;
	mov.b32		%acc_f32_42, 0f00000000;
	mov.b32		%acc_f32_43, 0f00000000;
	mov.b32		%acc_f32_44, 0f00000000;
	mov.b32		%acc_f32_45, 0f00000000;
	mov.b32		%acc_f32_46, 0f00000000;
	mov.b32		%acc_f32_47, 0f00000000;
	mov.b32		%acc_f32_48, 0f00000000;
	mov.b32		%acc_f32_49, 0f00000000;
	mov.b32		%acc_f32_50, 0f00000000;
	mov.b32		%acc_f32_51, 0f00000000;
	mov.b32		%acc_f32_52, 0f00000000;
	mov.b32		%acc_f32_53, 0f00000000;
	mov.b32		%acc_f32_54, 0f00000000;
	mov.b32		%acc_f32_55, 0f00000000;
	mov.b32		%acc_f32_56, 0f00000000;
	mov.b32		%acc_f32_57, 0f00000000;
	mov.b32		%acc_f32_58, 0f00000000;
	mov.b32		%acc_f32_59, 0f00000000;
	mov.b32		%acc_f32_60, 0f00000000;
	mov.b32		%acc_f32_61, 0f00000000;
	mov.b32		%acc_f32_62, 0f00000000;
	mov.b32		%acc_f32_63, 0f00000000;
	mov.u32		%ridx_s32_0, 0;
$loop_0:
	mov.u32		%lidx5, %tid.z;
	mov.u32		%lidx4, %tid.y;
	mov.u32		%lidx3, %tid.x;
	shr.s32		%alu_s32_0, %lidx3, 1;
	shr.s32		%alu_s32_1, %lidx3, 2;
	shr.u32		 %rem0, %lidx3, 31;                                                         
	add.s32		 %rem1, %lidx3, %rem0;                                                       
	and.b32		  %rem2, %rem1, -2;                                                        
	sub.s32		 %alu_s32_2, %lidx3, %rem2;
	shl.b32		%alu_s32_3, %gidx1, 11;
	mul.lo.s32	%alu_s32_4, %gidx1, 200704;
	shl.b32		%alu_s32_5, %lidx4, 6;
	shl.b32		%alu_s32_6, %gidx0, 18;
	mul.lo.s32	%alu_s32_7, %gidx0, 25088;
	shl.b32		%alu_s32_8, %lidx5, 7;
	mad.lo.s32	%alu_s32_9, %lidx5, 12544, %alu_s32_4;
	shr.u32		 %rem0, %alu_s32_0, 31;                                                         
	add.s32		 %rem1, %alu_s32_0, %rem0;                                                       
	and.b32		  %rem2, %rem1, -2;                                                        
	sub.s32		 %alu_s32_10, %alu_s32_0, %rem2;
	mad.lo.s32	%alu_s32_12, %alu_s32_1, 784, %alu_s32_7;
	mad.lo.s32	%alu_s32_13, %lidx4, 1568, %alu_s32_12;
	mad.lo.s32	%alu_s32_14, %lidx5, 3136, %alu_s32_13;
	mad.lo.s32	%alu_s32_15, %gidx2, 3211264, %alu_s32_14;
	shl.b32		%alu_s32_16, %alu_s32_2, 1;
	shl.b32		%alu_s32_17, %alu_s32_2, 2;
	mad.lo.s32	%alu_s32_18, %alu_s32_1, 3136, %alu_s32_9;
	mad.lo.s32	%alu_s32_19, %lidx4, 6272, %alu_s32_18;
	mad.lo.s32	%alu_s32_20, %gidx2, 6422528, %alu_s32_19;
	shl.b32		%alu_s32_23, %alu_s32_10, 2;
	shl.b32		%alu_s32_24, %alu_s32_10, 3;
	mad.lo.s32	%alu_s32_32, %ridx_s32_0, 401408, %alu_s32_15;
	mad.lo.s32	%alu_s32_33, %ridx_s32_0, 802816, %alu_s32_20;
	mov.u32		%ridx_s32_1, 0;
$loop_1:
	mad.lo.s32	%alu_s32_34, %ridx_s32_1, 28, %alu_s32_32;
	mad.lo.s32	%alu_s32_35, %ridx_s32_1, 112, %alu_s32_33;
	mov.u32		%ridx_s32_2, 0;
$loop_2:
	shl.b32		%alu_s32_36, %ridx_s32_2, 4;
	add.s32		%alu_s32_37, %alu_s32_36, %alu_s32_34;
	add.s32		%alu_s32_38, %alu_s32_16, %alu_s32_37;
	add.s32		%alu_s32_39, %alu_s32_23, %alu_s32_38;
	cvt.s64.s32	%cast_s64_1, %alu_s32_39;
	shl.b64		%alu_s64_2, %cast_s64_1, 1;
	add.s64		%alu_s64_3, %alu_s64_2, %dat_u64_2;
	ld.global.b16	%val_f16_0, [%alu_s64_3+0];
	ld.global.b16	%val_f16_1, [%alu_s64_3+2];
	ld.global.b16	%val_f16_2, [%alu_s64_3+12544];
	ld.global.b16	%val_f16_3, [%alu_s64_3+12546];
	ld.global.b16	%val_f16_4, [%alu_s64_3+25088];
	ld.global.b16	%val_f16_5, [%alu_s64_3+25090];
	ld.global.b16	%val_f16_6, [%alu_s64_3+37632];
	ld.global.b16	%val_f16_7, [%alu_s64_3+37634];
	add.s32		%alu_s32_40, %alu_s32_36, %alu_s32_16;
	add.s32		%alu_s32_41, %alu_s32_23, %alu_s32_40;
	setp.lt.s32	%alu_pred_0, %alu_s32_41, 19;
	mov.b16		%const_f16_0, 0x0000;
  @%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
  ld.global.b16 %val_f16_8, [%alu_s64_3+18];
  ld.global.b16 %val_f16_9, [%alu_s64_3+12562];
  ld.global.b16 %val_f16_10, [%alu_s64_3+25106];
  ld.global.b16 %val_f16_11, [%alu_s64_3+37650];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
  mov.b16 %val_f16_8, %const_f16_0;
  mov.b16 %val_f16_9, %const_f16_0;
  mov.b16 %val_f16_10, %const_f16_0;
  mov.b16 %val_f16_11, %const_f16_0;
  ${b3}:
  //
	setp.lt.s32	%alu_pred_1, %alu_s32_41, 20;
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
  ld.global.b16 %val_f16_12, [%alu_s64_3+16];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
  mov.b16 %val_f16_12, %const_f16_0;
  ${b3}:
  //
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
  ld.global.b16 %val_f16_13, [%alu_s64_3+12560];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
  mov.b16 %val_f16_13, %const_f16_0;
  ${b3}:
  //
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
  ld.global.b16 %val_f16_14, [%alu_s64_3+25104];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
  mov.b16 %val_f16_14, %const_f16_0;
  ${b3}:
  //
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
  ld.global.b16 %val_f16_15, [%alu_s64_3+37648];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
  mov.b16 %val_f16_15, %const_f16_0;
  ${b3}:
  //
	shl.b32		%alu_s32_42, %ridx_s32_2, 5;
	add.s32		%alu_s32_43, %alu_s32_42, %alu_s32_35;
	add.s32		%alu_s32_44, %alu_s32_17, %alu_s32_43;
	add.s32		%alu_s32_45, %alu_s32_24, %alu_s32_44;
	cvt.s64.s32	%cast_s64_2, %alu_s32_45;
	shl.b64		%alu_s64_4, %cast_s64_2, 1;
	add.s64		%alu_s64_5, %alu_s64_4, %dat_u64_1;
	ld.global.b16	%val_f16_16, [%alu_s64_5+0];
	ld.global.b16	%val_f16_17, [%alu_s64_5+4];
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
	ld.global.b16 %val_f16_18, [%alu_s64_5+32];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
	mov.b16 %val_f16_18, %const_f16_0;
  ${b3}:
  //
  @%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
  ld.global.b16 %val_f16_19, [%alu_s64_5+36];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
	mov.b16 %val_f16_19, %const_f16_0;
  ${b3}:
	ld.global.b16	%val_f16_20, [%alu_s64_5+50176];
	ld.global.b16	%val_f16_21, [%alu_s64_5+50180];
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
  ld.global.b16 %val_f16_22, [%alu_s64_5+50208];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
  mov.b16 %val_f16_22, %const_f16_0;
  ${b3}:
  @%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
	ld.global.b16 %val_f16_23, [%alu_s64_5+50212];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
	mov.b16 %val_f16_23, %const_f16_0;
  ${b3}:
	mov.b32		%wmma_b32_0, {{%val_f16_16, %val_f16_17}};
	mov.b32		%wmma_b32_1, {{%val_f16_20, %val_f16_21}};
	mov.b32		%wmma_b32_2, {{%val_f16_18, %val_f16_19}};
	mov.b32		%wmma_b32_3, {{%val_f16_22, %val_f16_23}};
	mov.b32		%wmma_b32_4, {{%val_f16_0, %val_f16_1}};
	mov.b32		%wmma_b32_5, {{%val_f16_12, %val_f16_8}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_0, %wmma_f32_1, %wmma_f32_2, %wmma_f32_3}}, {{%wmma_b32_0, %wmma_b32_1, %wmma_b32_2, %wmma_b32_3}}, {{%wmma_b32_4, %wmma_b32_5}}, {{%acc_f32_0, %acc_f32_1, %acc_f32_2, %acc_f32_3}};
	mov.b32		%wmma_b32_6, {{%val_f16_16, %val_f16_17}};
	mov.b32		%wmma_b32_7, {{%val_f16_20, %val_f16_21}};
	mov.b32		%wmma_b32_8, {{%val_f16_18, %val_f16_19}};
	mov.b32		%wmma_b32_9, {{%val_f16_22, %val_f16_23}};
	mov.b32		%wmma_b32_10, {{%val_f16_2, %val_f16_3}};
	mov.b32		%wmma_b32_11, {{%val_f16_13, %val_f16_9}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_4, %wmma_f32_5, %wmma_f32_6, %wmma_f32_7}}, {{%wmma_b32_6, %wmma_b32_7, %wmma_b32_8, %wmma_b32_9}}, {{%wmma_b32_10, %wmma_b32_11}}, {{%acc_f32_16, %acc_f32_17, %acc_f32_18, %acc_f32_19}};
	mov.b32		%wmma_b32_12, {{%val_f16_16, %val_f16_17}};
	mov.b32		%wmma_b32_13, {{%val_f16_20, %val_f16_21}};
	mov.b32		%wmma_b32_14, {{%val_f16_18, %val_f16_19}};
	mov.b32		%wmma_b32_15, {{%val_f16_22, %val_f16_23}};
	mov.b32		%wmma_b32_16, {{%val_f16_4, %val_f16_5}};
	mov.b32		%wmma_b32_17, {{%val_f16_14, %val_f16_10}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_8, %wmma_f32_9, %wmma_f32_10, %wmma_f32_11}}, {{%wmma_b32_12, %wmma_b32_13, %wmma_b32_14, %wmma_b32_15}}, {{%wmma_b32_16, %wmma_b32_17}}, {{%acc_f32_32, %acc_f32_33, %acc_f32_34, %acc_f32_35}};
	mov.b32		%wmma_b32_18, {{%val_f16_16, %val_f16_17}};
	mov.b32		%wmma_b32_19, {{%val_f16_20, %val_f16_21}};
	mov.b32		%wmma_b32_20, {{%val_f16_18, %val_f16_19}};
	mov.b32		%wmma_b32_21, {{%val_f16_22, %val_f16_23}};
	mov.b32		%wmma_b32_22, {{%val_f16_6, %val_f16_7}};
	mov.b32		%wmma_b32_23, {{%val_f16_15, %val_f16_11}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_12, %wmma_f32_13, %wmma_f32_14, %wmma_f32_15}}, {{%wmma_b32_18, %wmma_b32_19, %wmma_b32_20, %wmma_b32_21}}, {{%wmma_b32_22, %wmma_b32_23}}, {{%acc_f32_48, %acc_f32_49, %acc_f32_50, %acc_f32_51}};
	ld.global.b16	%val_f16_24, [%alu_s64_5+100352];
	ld.global.b16	%val_f16_25, [%alu_s64_5+100356];
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
	ld.global.b16 %val_f16_26, [%alu_s64_5+100384];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
	mov.b16 %val_f16_26, %const_f16_0;
  ${b3}:
  @%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
  ld.global.b16 %val_f16_27, [%alu_s64_5+100388];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
  mov.b16 %val_f16_27, %const_f16_0;
  ${b3}:
  ld.global.b16	%val_f16_29, [%alu_s64_5+150532];
	ld.global.b16	%val_f16_28, [%alu_s64_5+150528];
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
	ld.global.b16 %val_f16_30, [%alu_s64_5+150560];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
	mov.b16 %val_f16_30, %const_f16_0;
  ${b3}:
  @%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
		ld.global.b16 %val_f16_31, [%alu_s64_5+150564];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
		mov.b16 %val_f16_31, %const_f16_0;
  ${b3}:
	mov.b32		%wmma_b32_24, {{%val_f16_24, %val_f16_25}};
	mov.b32		%wmma_b32_25, {{%val_f16_28, %val_f16_29}};
	mov.b32		%wmma_b32_26, {{%val_f16_26, %val_f16_27}};
	mov.b32		%wmma_b32_27, {{%val_f16_30, %val_f16_31}};
	mov.b32		%wmma_b32_28, {{%val_f16_0, %val_f16_1}};
	mov.b32		%wmma_b32_29, {{%val_f16_12, %val_f16_8}};
  mov.b32		%wmma_b32_30, {{%val_f16_24, %val_f16_25}};
  mov.b32		%wmma_b32_31, {{%val_f16_28, %val_f16_29}};
  mov.b32		%wmma_b32_32, {{%val_f16_26, %val_f16_27}};
  mov.b32		%wmma_b32_33, {{%val_f16_30, %val_f16_31}};
  mov.b32		%wmma_b32_34, {{%val_f16_2, %val_f16_3}};
  mov.b32		%wmma_b32_35, {{%val_f16_13, %val_f16_9}};
  mov.b32		%wmma_b32_36, {{%val_f16_24, %val_f16_25}};
  mov.b32		%wmma_b32_37, {{%val_f16_28, %val_f16_29}};
  mov.b32		%wmma_b32_38, {{%val_f16_26, %val_f16_27}};
  mov.b32		%wmma_b32_39, {{%val_f16_30, %val_f16_31}};
  mov.b32		%wmma_b32_40, {{%val_f16_4, %val_f16_5}};
  mov.b32		%wmma_b32_41, {{%val_f16_14, %val_f16_10}};
  mov.b32		%wmma_b32_42, {{%val_f16_24, %val_f16_25}};
  mov.b32		%wmma_b32_43, {{%val_f16_28, %val_f16_29}};
  mov.b32		%wmma_b32_44, {{%val_f16_26, %val_f16_27}};
  mov.b32		%wmma_b32_45, {{%val_f16_30, %val_f16_31}};
  mov.b32		%wmma_b32_46, {{%val_f16_6, %val_f16_7}};
  mov.b32		%wmma_b32_47, {{%val_f16_15, %val_f16_11}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_16, %wmma_f32_17, %wmma_f32_18, %wmma_f32_19}}, {{%wmma_b32_24, %wmma_b32_25, %wmma_b32_26, %wmma_b32_27}}, {{%wmma_b32_28, %wmma_b32_29}}, {{%acc_f32_4, %acc_f32_5, %acc_f32_6, %acc_f32_7}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_20, %wmma_f32_21, %wmma_f32_22, %wmma_f32_23}}, {{%wmma_b32_30, %wmma_b32_31, %wmma_b32_32, %wmma_b32_33}}, {{%wmma_b32_34, %wmma_b32_35}}, {{%acc_f32_20, %acc_f32_21, %acc_f32_22, %acc_f32_23}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_24, %wmma_f32_25, %wmma_f32_26, %wmma_f32_27}}, {{%wmma_b32_36, %wmma_b32_37, %wmma_b32_38, %wmma_b32_39}}, {{%wmma_b32_40, %wmma_b32_41}}, {{%acc_f32_36, %acc_f32_37, %acc_f32_38, %acc_f32_39}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_28, %wmma_f32_29, %wmma_f32_30, %wmma_f32_31}}, {{%wmma_b32_42, %wmma_b32_43, %wmma_b32_44, %wmma_b32_45}}, {{%wmma_b32_46, %wmma_b32_47}}, {{%acc_f32_52, %acc_f32_53, %acc_f32_54, %acc_f32_55}};
	ld.global.b16	%val_f16_32, [%alu_s64_5+200704];
	ld.global.b16	%val_f16_33, [%alu_s64_5+200708];
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
    ld.global.b16 %val_f16_34, [%alu_s64_5+200736];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
  	mov.b16 %val_f16_34, %const_f16_0;
  ${b3}:
  @%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
   	ld.global.b16 %val_f16_35, [%alu_s64_5+200740];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
   	mov.b16 %val_f16_35, %const_f16_0;
  ${b3}:
	ld.global.b16	%val_f16_36, [%alu_s64_5+250880];
	ld.global.b16	%val_f16_37, [%alu_s64_5+250884];
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
    ld.global.b16 %val_f16_38, [%alu_s64_5+250912];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
    mov.b16 %val_f16_38, %const_f16_0;
  ${b3}:
  @%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
 	ld.global.b16 %val_f16_39, [%alu_s64_5+250916];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
    mov.b16 %val_f16_39, %const_f16_0;
  ${b3}:
	mov.b32		%wmma_b32_48, {{%val_f16_32, %val_f16_33}};
	mov.b32		%wmma_b32_49, {{%val_f16_36, %val_f16_37}};
	mov.b32		%wmma_b32_50, {{%val_f16_34, %val_f16_35}};
	mov.b32		%wmma_b32_51, {{%val_f16_38, %val_f16_39}};
	mov.b32		%wmma_b32_52, {{%val_f16_0, %val_f16_1}};
	mov.b32		%wmma_b32_53, {{%val_f16_12, %val_f16_8}};
  mov.b32		%wmma_b32_54, {{%val_f16_32, %val_f16_33}};
  mov.b32		%wmma_b32_55, {{%val_f16_36, %val_f16_37}};
  mov.b32		%wmma_b32_56, {{%val_f16_34, %val_f16_35}};
  mov.b32		%wmma_b32_57, {{%val_f16_38, %val_f16_39}};
  mov.b32		%wmma_b32_58, {{%val_f16_2, %val_f16_3}};
  mov.b32		%wmma_b32_59, {{%val_f16_13, %val_f16_9}};
  mov.b32		%wmma_b32_60, {{%val_f16_32, %val_f16_33}};
  mov.b32		%wmma_b32_61, {{%val_f16_36, %val_f16_37}};
  mov.b32		%wmma_b32_62, {{%val_f16_34, %val_f16_35}};
  mov.b32		%wmma_b32_63, {{%val_f16_38, %val_f16_39}};
  mov.b32		%wmma_b32_64, {{%val_f16_4, %val_f16_5}};
  mov.b32		%wmma_b32_65, {{%val_f16_14, %val_f16_10}};
  mov.b32		%wmma_b32_66, {{%val_f16_32, %val_f16_33}};
  mov.b32		%wmma_b32_67, {{%val_f16_36, %val_f16_37}};
  mov.b32		%wmma_b32_68, {{%val_f16_34, %val_f16_35}};
  mov.b32		%wmma_b32_69, {{%val_f16_38, %val_f16_39}};
  mov.b32		%wmma_b32_70, {{%val_f16_6, %val_f16_7}};
  mov.b32		%wmma_b32_71, {{%val_f16_15, %val_f16_11}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_32, %wmma_f32_33, %wmma_f32_34, %wmma_f32_35}}, {{%wmma_b32_48, %wmma_b32_49, %wmma_b32_50, %wmma_b32_51}}, {{%wmma_b32_52, %wmma_b32_53}}, {{%acc_f32_8, %acc_f32_9, %acc_f32_10, %acc_f32_11}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_36, %wmma_f32_37, %wmma_f32_38, %wmma_f32_39}}, {{%wmma_b32_54, %wmma_b32_55, %wmma_b32_56, %wmma_b32_57}}, {{%wmma_b32_58, %wmma_b32_59}}, {{%acc_f32_24, %acc_f32_25, %acc_f32_26, %acc_f32_27}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_40, %wmma_f32_41, %wmma_f32_42, %wmma_f32_43}}, {{%wmma_b32_60, %wmma_b32_61, %wmma_b32_62, %wmma_b32_63}}, {{%wmma_b32_64, %wmma_b32_65}}, {{%acc_f32_40, %acc_f32_41, %acc_f32_42, %acc_f32_43}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_44, %wmma_f32_45, %wmma_f32_46, %wmma_f32_47}}, {{%wmma_b32_66, %wmma_b32_67, %wmma_b32_68, %wmma_b32_69}}, {{%wmma_b32_70, %wmma_b32_71}}, {{%acc_f32_56, %acc_f32_57, %acc_f32_58, %acc_f32_59}};
	ld.global.b16	%val_f16_40, [%alu_s64_5+301056];
	ld.global.b16	%val_f16_41, [%alu_s64_5+301060];
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
	ld.global.b16 %val_f16_42, [%alu_s64_5+301088];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
	mov.b16 %val_f16_42, %const_f16_0;
  ${b3}:
  @%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
    ld.global.b16 %val_f16_43, [%alu_s64_5+301092];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
    mov.b16 %val_f16_43, %const_f16_0;
  ${b3}:
	ld.global.b16	%val_f16_44, [%alu_s64_5+351232];
	ld.global.b16	%val_f16_45, [%alu_s64_5+351236];
  @%alu_pred_1  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
	ld.global.b16 %val_f16_46, [%alu_s64_5+351264];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
	mov.b16 %val_f16_46, %const_f16_0;
  ${b3}:
  @%alu_pred_0  bra ${(b1:=ssa_label("L"))};
  bra.uni ${(b2:=ssa_label("L"))};
  ${b1}:
    ld.global.b16 %val_f16_47, [%alu_s64_5+351268];
  bra.uni ${(b3:=ssa_label("L"))};
  ${b2}:
    mov.b16 %val_f16_47, %const_f16_0;
  ${b3}:
  mov.u32  %r0, 1;
	mov.b32		%wmma_b32_72, {{%val_f16_40, %val_f16_41}};
	mov.b32		%wmma_b32_73, {{%val_f16_44, %val_f16_45}};
	mov.b32		%wmma_b32_74, {{%val_f16_42, %val_f16_43}};
	mov.b32		%wmma_b32_75, {{%val_f16_46, %val_f16_47}};
	mov.b32		%wmma_b32_76, {{%val_f16_0, %val_f16_1}};
	mov.b32		%wmma_b32_77, {{%val_f16_12, %val_f16_8}};
  mov.b32		%wmma_b32_78, {{%val_f16_40, %val_f16_41}};
  mov.b32		%wmma_b32_79, {{%val_f16_44, %val_f16_45}};
  mov.b32		%wmma_b32_80, {{%val_f16_42, %val_f16_43}};
  mov.b32		%wmma_b32_81, {{%val_f16_46, %val_f16_47}};
  mov.b32		%wmma_b32_82, {{%val_f16_2, %val_f16_3}};
  mov.b32		%wmma_b32_83, {{%val_f16_13, %val_f16_9}};
  mov.b32		%wmma_b32_84, {{%val_f16_40, %val_f16_41}};
  mov.b32		%wmma_b32_85, {{%val_f16_44, %val_f16_45}};
  mov.b32		%wmma_b32_86, {{%val_f16_42, %val_f16_43}};
  mov.b32		%wmma_b32_87, {{%val_f16_46, %val_f16_47}};
  mov.b32		%wmma_b32_88, {{%val_f16_4, %val_f16_5}};
  mov.b32		%wmma_b32_89, {{%val_f16_14, %val_f16_10}};
  mov.b32		%wmma_b32_90, {{%val_f16_40, %val_f16_41}};
  mov.b32		%wmma_b32_91, {{%val_f16_44, %val_f16_45}};
  mov.b32		%wmma_b32_92, {{%val_f16_42, %val_f16_43}};
  mov.b32		%wmma_b32_93, {{%val_f16_46, %val_f16_47}};
  mov.b32		%wmma_b32_94, {{%val_f16_6, %val_f16_7}};
  mov.b32		%wmma_b32_95, {{%val_f16_15, %val_f16_11}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_48, %wmma_f32_49, %wmma_f32_50, %wmma_f32_51}}, {{%wmma_b32_72, %wmma_b32_73, %wmma_b32_74, %wmma_b32_75}}, {{%wmma_b32_76, %wmma_b32_77}}, {{%acc_f32_12, %acc_f32_13, %acc_f32_14, %acc_f32_15}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_52, %wmma_f32_53, %wmma_f32_54, %wmma_f32_55}}, {{%wmma_b32_78, %wmma_b32_79, %wmma_b32_80, %wmma_b32_81}}, {{%wmma_b32_82, %wmma_b32_83}}, {{%acc_f32_28, %acc_f32_29, %acc_f32_30, %acc_f32_31}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_56, %wmma_f32_57, %wmma_f32_58, %wmma_f32_59}}, {{%wmma_b32_84, %wmma_b32_85, %wmma_b32_86, %wmma_b32_87}}, {{%wmma_b32_88, %wmma_b32_89}}, {{%acc_f32_44, %acc_f32_45, %acc_f32_46, %acc_f32_47}};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32	           {{%wmma_f32_60, %wmma_f32_61, %wmma_f32_62, %wmma_f32_63}}, {{%wmma_b32_90, %wmma_b32_91, %wmma_b32_92, %wmma_b32_93}}, {{%wmma_b32_94, %wmma_b32_95}}, {{%acc_f32_60, %acc_f32_61, %acc_f32_62, %acc_f32_63}};
	mov.b32		%acc_f32_0, %wmma_f32_0;
	mov.b32		%acc_f32_1, %wmma_f32_1;
	mov.b32		%acc_f32_2, %wmma_f32_2;
	mov.b32		%acc_f32_3, %wmma_f32_3;
	mov.b32		%acc_f32_4, %wmma_f32_16;
	mov.b32		%acc_f32_5, %wmma_f32_17;
	mov.b32		%acc_f32_6, %wmma_f32_18;
	mov.b32		%acc_f32_7, %wmma_f32_19;
	mov.b32		%acc_f32_8, %wmma_f32_32;
	mov.b32		%acc_f32_9, %wmma_f32_33;
	mov.b32		%acc_f32_10, %wmma_f32_34;
	mov.b32		%acc_f32_11, %wmma_f32_35;
	mov.b32		%acc_f32_12, %wmma_f32_48;
	mov.b32		%acc_f32_13, %wmma_f32_49;
	mov.b32		%acc_f32_14, %wmma_f32_50;
	mov.b32		%acc_f32_15, %wmma_f32_51;
	mov.b32		%acc_f32_16, %wmma_f32_4;
	mov.b32		%acc_f32_17, %wmma_f32_5;
	mov.b32		%acc_f32_18, %wmma_f32_6;
	mov.b32		%acc_f32_19, %wmma_f32_7;
	mov.b32		%acc_f32_20, %wmma_f32_20;
	mov.b32		%acc_f32_21, %wmma_f32_21;
	mov.b32		%acc_f32_22, %wmma_f32_22;
	mov.b32		%acc_f32_23, %wmma_f32_23;
	mov.b32		%acc_f32_24, %wmma_f32_36;
	mov.b32		%acc_f32_25, %wmma_f32_37;
	mov.b32		%acc_f32_26, %wmma_f32_38;
	mov.b32		%acc_f32_27, %wmma_f32_39;
	mov.b32		%acc_f32_28, %wmma_f32_52;
	mov.b32		%acc_f32_29, %wmma_f32_53;
	mov.b32		%acc_f32_30, %wmma_f32_54;
	mov.b32		%acc_f32_31, %wmma_f32_55;
	mov.b32		%acc_f32_32, %wmma_f32_8;
	mov.b32		%acc_f32_33, %wmma_f32_9;
	mov.b32		%acc_f32_34, %wmma_f32_10;
	mov.b32		%acc_f32_35, %wmma_f32_11;
	mov.b32		%acc_f32_36, %wmma_f32_24;
	mov.b32		%acc_f32_37, %wmma_f32_25;
	mov.b32		%acc_f32_38, %wmma_f32_26;
	mov.b32		%acc_f32_39, %wmma_f32_27;
	mov.b32		%acc_f32_40, %wmma_f32_40;
	mov.b32		%acc_f32_41, %wmma_f32_41;
	mov.b32		%acc_f32_42, %wmma_f32_42;
	mov.b32		%acc_f32_43, %wmma_f32_43;
	mov.b32		%acc_f32_44, %wmma_f32_56;
	mov.b32		%acc_f32_45, %wmma_f32_57;
	mov.b32		%acc_f32_46, %wmma_f32_58;
	mov.b32		%acc_f32_47, %wmma_f32_59;
	mov.b32		%acc_f32_48, %wmma_f32_12;
	mov.b32		%acc_f32_49, %wmma_f32_13;
	mov.b32		%acc_f32_50, %wmma_f32_14;
	mov.b32		%acc_f32_51, %wmma_f32_15;
	mov.b32		%acc_f32_52, %wmma_f32_28;
	mov.b32		%acc_f32_53, %wmma_f32_29;
	mov.b32		%acc_f32_54, %wmma_f32_30;
	mov.b32		%acc_f32_55, %wmma_f32_31;
	mov.b32		%acc_f32_56, %wmma_f32_44;
	mov.b32		%acc_f32_57, %wmma_f32_45;
	mov.b32		%acc_f32_58, %wmma_f32_46;
	mov.b32		%acc_f32_59, %wmma_f32_47;
	mov.b32		%acc_f32_60, %wmma_f32_60;
	mov.b32		%acc_f32_61, %wmma_f32_61;
	mov.b32		%acc_f32_62, %wmma_f32_62;
	mov.b32		%acc_f32_63, %wmma_f32_63;
	add.s32		%ridx_s32_2, %ridx_s32_2, 1;
	setp.lt.u32	%pred_pred_0, %ridx_s32_2, 2;
  mov.u32     %ridx_s32_2, %r0;
	@%pred_pred_0	bra $loop_2;
	add.s32		%ridx_s32_1, %ridx_s32_1, 1;
	setp.lt.u32	%pred_pred_1, %ridx_s32_1, 28;
	@%pred_pred_1	bra $loop_1;
	add.s32		%ridx_s32_0, %ridx_s32_0, 1;
	setp.lt.u32	%pred_pred_2, %ridx_s32_0, 8;
	@%pred_pred_2	bra $loop_0;
	mov.u32		%lidx3, %tid.x;
	shr.s32		%alu_s32_0, %lidx3, 1;
	shr.s32		%alu_s32_1, %lidx3, 2;
	shr.u32		 %rem0, %lidx3, 31;                                                         
	add.s32		 %rem1, %lidx3, %rem0;                                                       
	and.b32		  %rem2, %rem1, -2;                                                        
	sub.s32		 %alu_s32_2, %lidx3, %rem2;
	shr.u32		 %rem0, %alu_s32_0, 31;                                                         
	add.s32		 %rem1, %alu_s32_0, %rem0;                                                       
	and.b32		  %rem2, %rem1, -2;                                                        
	sub.s32		 %alu_s32_10, %alu_s32_0, %rem2;
	shl.b32		%alu_s32_11, %alu_s32_1, 5;
	shl.b32		%alu_s32_21, %alu_s32_2, 14;
	add.s32		%alu_s32_22, %alu_s32_6, %alu_s32_3;
	shl.b32		%alu_s32_25, %alu_s32_10, 15;
	add.s32		%alu_s32_26, %alu_s32_22, %gidx2;
	add.s32		%alu_s32_27, %alu_s32_21, %alu_s32_26;
	add.s32		%alu_s32_28, %alu_s32_8, %alu_s32_27;
	add.s32		%alu_s32_29, %alu_s32_25, %alu_s32_28;
	add.s32		%alu_s32_30, %alu_s32_11, %alu_s32_29;
	add.s32		%alu_s32_31, %alu_s32_5, %alu_s32_30;
	cvt.s64.s32	%cast_s64_0, %alu_s32_31;
	shl.b64		%alu_s64_0, %cast_s64_0, 2;
	add.s64		%alu_s64_1, %alu_s64_0, %dat_u64_0;
	st.global.f32	[%alu_s64_1+0], %acc_f32_0;
	st.global.f32	[%alu_s64_1+1024], %acc_f32_2;
	st.global.f32	[%alu_s64_1+2048], %acc_f32_4;
	st.global.f32	[%alu_s64_1+3072], %acc_f32_6;
	st.global.f32	[%alu_s64_1+4096], %acc_f32_8;
	st.global.f32	[%alu_s64_1+5120], %acc_f32_10;
	st.global.f32	[%alu_s64_1+6144], %acc_f32_12;
	st.global.f32	[%alu_s64_1+7168], %acc_f32_14;
	st.global.f32	[%alu_s64_1+32768], %acc_f32_1;
	st.global.f32	[%alu_s64_1+33792], %acc_f32_3;
	st.global.f32	[%alu_s64_1+34816], %acc_f32_5;
	st.global.f32	[%alu_s64_1+35840], %acc_f32_7;
	st.global.f32	[%alu_s64_1+36864], %acc_f32_9;
	st.global.f32	[%alu_s64_1+37888], %acc_f32_11;
	st.global.f32	[%alu_s64_1+38912], %acc_f32_13;
	st.global.f32	[%alu_s64_1+39936], %acc_f32_15;
	st.global.f32	[%alu_s64_1+262144], %acc_f32_16;
	st.global.f32	[%alu_s64_1+263168], %acc_f32_18;
	st.global.f32	[%alu_s64_1+264192], %acc_f32_20;
	st.global.f32	[%alu_s64_1+265216], %acc_f32_22;
	st.global.f32	[%alu_s64_1+266240], %acc_f32_24;
	st.global.f32	[%alu_s64_1+267264], %acc_f32_26;
	st.global.f32	[%alu_s64_1+268288], %acc_f32_28;
	st.global.f32	[%alu_s64_1+269312], %acc_f32_30;
	st.global.f32	[%alu_s64_1+294912], %acc_f32_17;
	st.global.f32	[%alu_s64_1+295936], %acc_f32_19;
	st.global.f32	[%alu_s64_1+296960], %acc_f32_21;
	st.global.f32	[%alu_s64_1+297984], %acc_f32_23;
	st.global.f32	[%alu_s64_1+299008], %acc_f32_25;
	st.global.f32	[%alu_s64_1+300032], %acc_f32_27;
	st.global.f32	[%alu_s64_1+301056], %acc_f32_29;
	st.global.f32	[%alu_s64_1+302080], %acc_f32_31;
	st.global.f32	[%alu_s64_1+524288], %acc_f32_32;
	st.global.f32	[%alu_s64_1+525312], %acc_f32_34;
	st.global.f32	[%alu_s64_1+526336], %acc_f32_36;
	st.global.f32	[%alu_s64_1+527360], %acc_f32_38;
	st.global.f32	[%alu_s64_1+528384], %acc_f32_40;
	st.global.f32	[%alu_s64_1+529408], %acc_f32_42;
	st.global.f32	[%alu_s64_1+530432], %acc_f32_44;
	st.global.f32	[%alu_s64_1+531456], %acc_f32_46;
	st.global.f32	[%alu_s64_1+557056], %acc_f32_33;
	st.global.f32	[%alu_s64_1+558080], %acc_f32_35;
	st.global.f32	[%alu_s64_1+559104], %acc_f32_37;
	st.global.f32	[%alu_s64_1+560128], %acc_f32_39;
	st.global.f32	[%alu_s64_1+561152], %acc_f32_41;
	st.global.f32	[%alu_s64_1+562176], %acc_f32_43;
	st.global.f32	[%alu_s64_1+563200], %acc_f32_45;
	st.global.f32	[%alu_s64_1+564224], %acc_f32_47;
	st.global.f32	[%alu_s64_1+786432], %acc_f32_48;
	st.global.f32	[%alu_s64_1+787456], %acc_f32_50;
	st.global.f32	[%alu_s64_1+788480], %acc_f32_52;
	st.global.f32	[%alu_s64_1+789504], %acc_f32_54;
	st.global.f32	[%alu_s64_1+790528], %acc_f32_56;
	st.global.f32	[%alu_s64_1+791552], %acc_f32_58;
	st.global.f32	[%alu_s64_1+792576], %acc_f32_60;
	st.global.f32	[%alu_s64_1+793600], %acc_f32_62;
	st.global.f32	[%alu_s64_1+819200], %acc_f32_49;
	st.global.f32	[%alu_s64_1+820224], %acc_f32_51;
	st.global.f32	[%alu_s64_1+821248], %acc_f32_53;
	st.global.f32	[%alu_s64_1+822272], %acc_f32_55;
	st.global.f32	[%alu_s64_1+823296], %acc_f32_57;
	st.global.f32	[%alu_s64_1+824320], %acc_f32_59;
	st.global.f32	[%alu_s64_1+825344], %acc_f32_61;
	st.global.f32	[%alu_s64_1+826368], %acc_f32_63;
	ret;
}}"""
    return self.render_kernel(kernel, name, bufs, c.items())

ptx_matcher = PatternMatcher([
  ({"__name__": "root", "uop": UOps.ALU, "arg": BinaryOps.MUL, "dtype": set([dt for dt in dtypes.fields().values() if dtypes.is_int(dt)]),
      "vin": [{"__name__": "const", "uop": UOps.CONST, "arg": set([2**i for i in range(64)])}, {"__name__": "mul"}]},
    lambda root, mul, const: UOp(UOps.ALU, root.dtype, (mul, UOp.const(root.dtype, int(math.log2(const.arg)))), BinaryOps.SHL)),
  ({"__name__": "root", "uop": UOps.ALU, "arg": TernaryOps.MULACC, "dtype": set([dt for dt in dtypes.fields().values() if dtypes.is_int(dt)]),
      "vin": ({"__name__": "mul"}, {"__name__": "const", "uop": UOps.CONST, "arg": set([2**i for i in range(64)])}, {"__name__": "add"})},
    lambda root, mul, const, add: UOp(UOps.ALU, root.dtype, (UOp(UOps.ALU, root.dtype, (mul, UOp.const(root.dtype, int(math.log2(const.arg)))), BinaryOps.SHL), add), BinaryOps.ADD)),
  ({"__name__": "root", "uop": UOps.ALU, "arg": BinaryOps.DIV, "dtype": set([dt for dt in dtypes.fields().values() if dtypes.is_int(dt)]),
      "vin": [{"__name__": "const", "uop": UOps.CONST, "arg": set([2**i for i in range(64)])}, {"__name__": "div"}]},
    lambda root, div, const: UOp(UOps.ALU, root.dtype, (div, UOp.const(root.dtype, int(math.log2(const.arg)))), BinaryOps.SHR)),
  ({"__name__": "root", "uop": UOps.ALU, "arg": BinaryOps.CMPEQ, "vin": ({"dtype": dtypes.bool},{})},
  lambda root: UOp(UOps.ALU, dtypes.bool, (UOp(root.uop, root.dtype, root.vin, BinaryOps.XOR),), UnaryOps.NEG)),
  ({"__name__": "root", "uop": UOps.ALU, "arg": BinaryOps.CMPLT, "vin": ({"__name__": "x", "dtype": dtypes.bool},{"__name__": "y"})},
  lambda root,x,y: UOp(root.uop, root.dtype, (UOp(UOps.ALU, dtypes.bool, (x,), UnaryOps.NEG), y), BinaryOps.MUL)),
  ({"__name__": "root", "uop": UOps.ALU, "arg": BinaryOps.ADD,
    "vin": [{"__name__": "non_muls"}, {"__name__": "muls", "uop": UOps.ALU, "arg": BinaryOps.MUL}]},
    lambda root, muls, non_muls: UOp(UOps.ALU, root.dtype, muls.vin + (non_muls,), TernaryOps.MULACC)),
  *[({"__name__": "x", "uop": UOps.ALU, "dtype": dtypes.half, "arg": op},
    lambda x: UOp(UOps.CAST, dtypes.half, (UOp(x.uop, dtypes.float32, tuple([UOp(UOps.CAST, dtypes.float32, (vv,)) for vv in x.vin]), x.arg),)))
    for op in PTXRenderer.asm_for_op.keys() if op not in PTXRenderer.supports_half],
  ({"__name__": "root", "uop": UOps.LOAD, "dtype": dtypes.bool,
    "vin": ({"__name__": "x"},{"__name__": "y"},{"__name__": "z"},{"__name__": "k"})},
  lambda root,x,y,z,k: UOp(UOps.CAST, dtypes.bool, (UOp(root.uop, dtypes.int8, (x,y,z,UOp(UOps.CAST, dtypes.uint8, (k,)))),), root.arg)),
  ({"__name__": "root", "uop": UOps.LOAD,"dtype": dtypes.bool, "vin": ({},{})},
  lambda root: UOp(UOps.CAST, dtypes.bool, (UOp(root.uop, dtypes.uint8, root.vin, root.arg),))),
  ({"__name__": "root", "uop": UOps.STORE, "vin": ({},{},{"__name__": "z","dtype": dtypes.bool}, {})},
  lambda root,z: UOp(root.uop, root.dtype, root.vin[:2] + (UOp(UOps.CAST, dtypes.uint8, (z,)),), root.arg)),
  ({"__name__": "root", "uop": UOps.STORE, "vin": ({},{},{"__name__": "z","dtype": dtypes.bool})},
  lambda root,z: UOp(root.uop, root.dtype, root.vin[:2] + (UOp(UOps.CAST, dtypes.uint8, (z,)),), root.arg)),
  ({"__name__": "root", "uop": UOps.STORE, "vin": ({},{},{},{"__name__": "g", "dtype": dtypes.int})},
  lambda root,g: UOp(root.uop, root.dtype, root.vin[:3] + (UOp(UOps.CAST, dtypes.bool, (g,)),), root.arg)),
  # ptr_ar (load/store)
  ({"__name__": "root", "uop": {UOps.LOAD, UOps.STORE}, "__allow_len__":[2,3,4,5], "vin": ({"uop":{UOps.DEFINE_LOCAL,UOps.DEFINE_GLOBAL}},
                               {"uop": UOps.ALU, "arg": BinaryOps.ADD,"vin":[{"__name__": "alu"}, {"__name__": "const", "uop":UOps.CONST}]})},
    lambda root, alu, const: UOp(root.uop, root.dtype,
      (alu.cast(dtypes.int64)*UOp.const(dtypes.int64, root.vin[0].dtype.itemsize)+root.vin[0].cast(dtypes.int64),
       UOp.const(const.dtype, root.vin[0].dtype.itemsize)*const)+root.vin[2:])),
  ({"__name__": "root", "uop": {UOps.LOAD, UOps.STORE}, "__allow_len__":[2,3,4,5], "vin": ({"uop":{UOps.DEFINE_LOCAL,UOps.DEFINE_GLOBAL}},
                                                                              {"__name__": "const", "uop":UOps.CONST})},
    lambda root, const: UOp(root.uop, root.dtype, (root.vin[0].cast(dtypes.int64),
                                UOp.const(dtypes.int64, const.arg * root.vin[0].dtype.itemsize),
                                                  )+root.vin[2:])),
  ({"__name__": "root", "uop": {UOps.LOAD, UOps.STORE}, "__allow_len__":[2,3,4,5], "vin": ({"uop":{UOps.DEFINE_LOCAL,UOps.DEFINE_GLOBAL}},
                                                                              {"__name__": "alu"})},  # no const here
    lambda root, alu: UOp(root.uop, root.dtype,
      (alu.cast(dtypes.int64)*UOp.const(dtypes.int64, root.vin[0].dtype.itemsize)+root.vin[0].cast(dtypes.int64),
        UOp.const(dtypes.int64, 0))+root.vin[2:])),
])
