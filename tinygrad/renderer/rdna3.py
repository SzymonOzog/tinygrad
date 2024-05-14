from typing import Callable, DefaultDict, Dict, List, Union, NamedTuple, Optional, cast
import functools, struct, copy
from collections import defaultdict
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps, Op
from tinygrad.dtype import dtypes, DType, PtrDType, ConstType, INVERSE_DTYPES_DICT
from tinygrad.codegen.uops import UOpGraph, PatternMatcher

def render_val(x, dtype):
  if dtypes.is_float(dtype):
    if dtype == dtypes.double: return "0d%02X%02X%02X%02X%02X%02X%02X%02X" % tuple(struct.pack("d",x)[::-1])
    elif dtype == dtypes.half: return "0x%02X%02X" % tuple(struct.pack("e",x)[::-1])
    return "0x%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])
  # return str(hex(x)) + dtypes.is_unsigned(dtype) else str(hex(x << 32) + =15 
  return str(hex(x)) + ("U" if dtypes.is_unsigned(dtype) else "")

def ptr_ar(root, uops):
  # pass
  assert root.arg in {'.shared', '.global', None}
  if root.arg is None: root.arg = '.shared' if root.vin[0].uop is UOps.DEFINE_LOCAL else '.global'  # move this to the argL
  # if root.vin[1].uop is UOps.ALU and root.vin[1].arg in [BinaryOps.ADD, BinaryOps.SUB] and root.vin[1].vin[1].uop is UOps.CONST:
  #   offset = uops.add(UOps.ALU, dtypes.int, (root.vin[1].vin[0], val), arg=BinaryOps.MUL, insert_before=uops.uops.index(root))
  #   offset = uops.add(UOps.CAST, dtypes.uint64, (offset,), insert_before=uops.uops.index(root))
  #   cache = uops.add(UOps.ALU, dtypes.uint64, (root.vin[0], offset), arg=BinaryOps.ADD, insert_before=uops.uops.index(root))
  #   ptr = uops.add(UOps.ALU, dtypes.int, (root.vin[1].vin[1], val), arg=BinaryOps.MUL, insert_before=uops.uops.index(root))
  #   if root.vin[1].arg == BinaryOps.SUB: ptr = uops.add(UOps.ALU, dtypes.int, (ptr,), arg=UnaryOps.NEG, insert_before=uops.uops.index(root))
  #   root.vin = (cache, ptr) + root.vin[2:]
  # else:
  val = uops.add(UOps.CONST, dtypes.int, tuple(), arg=root.vin[0].dtype.itemsize, insert_before=uops.uops.index(root))
  ptr = uops.add(UOps.ALU, dtypes.int, (root.vin[1], val), arg=BinaryOps.MUL, insert_before=uops.uops.index(root))
  root.vin = (root.vin[0], ptr) + root.vin[2:]
  # if ptr.uop is UOps.CONST:
  # else:
  #   zero = uops.add(UOps.CONST, dtypes.int, tuple(), arg=0, cachable=False, insert_before=uops.uops.index(root))
  #   bptr = uops.add(UOps.CAST, dtypes.uint64, (ptr,), insert_before=uops.uops.index(root))
  #   fptr = uops.add(UOps.ALU, dtypes.uint64, (root.vin[0], bptr), arg=BinaryOps.ADD, insert_before=uops.uops.index(root))
  #   root.vin = (fptr, zero) + root.vin[2:]

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
  def render_bra(self, b1, pred=None, b2=None) -> List[str]: raise NotImplementedError()
  def render_gep(self, loc, base, offset, dtype, gate=None) -> List[str]: raise NotImplementedError()
  def render_load(self, loc, dest, dtype, gate=None, alt=None, ss="", offset=0) -> List[str]: raise NotImplementedError()
  def render_store(self, loc, val, dtype, gate=None, ss="", offset=0) -> List[str]: raise NotImplementedError()
  def render_cast(self, d:str, a:str, dtype:DType, atype:DType, bitcast=False, pred=False) -> List[str]: raise NotImplementedError()

  def render_kernel(self, kernel, function_name, bufs, regs) -> str: raise NotImplementedError()
  def mem_type(self, dtype) -> str: raise NotImplementedError()

def uops_to_rdna(lang:AssemblyLanguage, function_name:str, _uops:UOpGraph) -> str:
  # editing the uops breaks beam search
  uops = copy.deepcopy(_uops)
  kernel:List[str] = []
  bufs = []

  matcher = PatternMatcher([
    ({"__name__": "root", "uop": UOps.ALU, "arg": UnaryOps.NEG, "vin": ({"dtype": set([dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64])})},
     lambda root: UOp(UOps.ALU, root.dtype, (UOp(UOps.CONST, root.dtype, tuple(), 0), root.vin[0]), BinaryOps.SUB)),
    ({"__name__": "root", "uop": UOps.ALU, "arg": BinaryOps.CMPEQ, "vin": ({"dtype": dtypes.bool},{})},
     lambda root: UOp(UOps.ALU, dtypes.bool, (UOp(root.uop, root.dtype, root.vin, BinaryOps.XOR),), UnaryOps.NEG)),
    ({"__name__": "root", "uop": UOps.ALU, "arg": BinaryOps.CMPLT, "vin": ({"__name__": "x", "dtype": dtypes.bool},{"__name__": "y"})},
     lambda root,x,y: UOp(root.uop, root.dtype, (UOp(UOps.ALU, dtypes.bool, (x,), UnaryOps.NEG), y), BinaryOps.MUL)),
    ({"__name__": "root", "uop": UOps.ALU, "arg": BinaryOps.ADD, "dtype": set([dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64]),
      "vin": [{"__name__": "non_muls"}, {"__name__": "muls", "uop": UOps.ALU, "arg": BinaryOps.MUL}]},
      lambda root, muls, non_muls: UOp(UOps.ALU, root.dtype, muls.vin + (non_muls,), TernaryOps.MULACC)),
    # *[({"__name__": "x", "uop": UOps.ALU, "dtype": dtypes.half, "arg": op},
    #    lambda x: UOp(UOps.CAST, dtypes.half, (UOp(x.uop, dtypes.float32, tuple([UOp(UOps.CAST, dtypes.float32, (vv,)) for vv in x.vin]), x.arg),)))
    #   for op in lang.asm_for_op.keys() if op not in lang.supports_half],
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

  uops.print()
  # here we do a pretransform on UOps to fix some shortcomings of PTX
  # all uops must be a register
  matcher.rewrite_graph(uops)

  for pointer_op in list(filter(lambda uop: uop.uop in [UOps.LOAD, UOps.STORE], uops.uops)): ptr_ar(pointer_op, uops)
  uops.remove_childless(set(x for x in uops if x.uop in {UOps.PHI, UOps.ENDIF, UOps.ENDLOOP, UOps.STORE}))
  uops.optimize_loops()
  uops.print()

  def kk(*s: str): kernel.append("\n".join(s))

  c: DefaultDict[str, int] = defaultdict(int)
  c['v']=3
  c['s']=2+len(list(filter(lambda x: x.uop is UOps.DEFINE_GLOBAL, uops.uops)))*2
  r: Dict[UOp, Union[List[str], str]] = {}
  def ssa(prefix:str, u:Optional[UOp]=None, dtype:Optional[str]=None) -> str:
    nonlocal c, r
    # prefix += f"_{dtype if dtype is not None else lang.types[cast(DType, cast(UOp, u).dtype)]}_"
    c[prefix] += 1
    if u is not None: r[u] = f"{prefix}{c[prefix]-1}"
    return f"{prefix}{c[prefix]-1}"

  def r_as(uop:UOp, reg_type:str) -> str:
    nonlocal r, kk, ssa
    if (curr:=r[uop])[0] == reg_type: return curr
    new = ssa('v') 
    kk(f"v_mov_b32 {new}, {curr}")
    r[uop] = new
    return new


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
      kk(*lang.render_bra(lb:=ssa_label('if', u), _cast(r[vin[0]], dtypes.bool, vin[0].dtype, u=u, pred=True), f"{lb}_true"), f"{lb}_true:")
    elif uop is UOps.BARRIER and lang.barrier: kk(lang.barrier)
    elif uop is UOps.ENDLOOP:
      kk(lang.asm_for_op[BinaryOps.ADD](r[vin[0]], r[vin[0]], "1", dtypes.int, lang.types[dtypes.int]),
          lang.asm_for_op[BinaryOps.CMPLT](pred:=ssa("pred", dtype="pred"), r[vin[0]], r[vin[0].vin[1]], dtypes.int, lang.types[dtypes.int]))
      kk(*lang.render_bra(r_label[vin[0]], pred, f"{r_label[vin[0]]}_exit"), f"{r_label[vin[0]]}_exit:")
    elif uop is UOps.ENDIF:
      kk(f"{r_label[vin[0]]}:")
    elif uop is UOps.STORE:
      if "shared" in u.arg:
        kk(f"flat_store_b{lang.mem_type(vin[2].dtype)[1:]} {r_as(vin[1], 'v')}, {r_as(vin[2], 'v')}, {r_as(vin[0], 's')}")
      else:
        kk(f"global_store_b{lang.mem_type(vin[2].dtype)[1:]} {r_as(vin[1], 'v')}, {r_as(vin[2], 'v')}, {r_as(vin[0], 's')}")
    else:
      assert dtype is not None, f"None dtype for uop {uop}"
      if uop is UOps.LOOP: kk(*lang.render_loop(ssa('ridx', u), r[vin[0]], ssa_label('loop', u)))
      elif uop is UOps.ALU:
        assert vin[0].dtype is not None
        if args is BinaryOps.CMPLT or args is BinaryOps.CMPEQ:
          # pass in the other dtype here
          kk(lang.asm_for_op[args](ssa("s", u), *[r[x] for x in vin], vin[0].dtype, lang.types[vin[0].dtype]))
        elif args is BinaryOps.DIV:
          #Amd does not have division op it needs to use https://en.wikipedia.org/wiki/Division_algorithm#Newton%E2%80%93Raphson_division
          if dtypes.is_float(dtype):
            name = lang.types[dtype]
            kk(f"v_div_scale_{name} {(result:=ssa('v', u))}, null, {r[vin[1]]}, {r[vin[1]]}, {r[vin[0]]}")
            kk("s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)")
            kk(f"v_rcp_{name} {(est:=ssa('v'))}, {result}")
            kk("s_waitcnt_depctr 0xfff")
            kk(f"v_fma_{name} {(err:=ssa('v'))}, -{result}, {est}, 1.0")
            kk(f"v_fmac_{name} {est}, {err}, {est}")
            kk(f"v_div_scale_{name} {(numerator:=ssa('v'))}, vcc_lo, {r[vin[0]]}, {r[vin[1]]}, {r[vin[0]]}")
            kk("s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)")
            kk(f"v_mul_{name} {(est2:=ssa('v'))}, {numerator}, {est}")
            kk(f"v_fma_{name} {err}, -{result}, {est2}, {numerator}")
            kk("s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)")
            kk(f"v_fmac_{name} {est2}, {err}, {est}")
            kk(f"v_fma_{name} {result}, -{result}, {est2}, {numerator}")
            kk("s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)")
            kk(f"v_div_fmas_{name} {result}, {result}, {est}, {est2}")
            kk(f"v_div_fixup_{name} {result}, {result}, {r[vin[1]]}, {r[vin[0]]}")
          else:
            kk(f"s_ashr_i32 {(divisor_sign:=ssa('s'))}, {(two_compl:=r[vin[1]])}, 31")
            kk(f"s_add_i32 {(one_compl:=ssa('s'))}, {two_compl}, {divisor_sign}")
            kk(f"s_xor_b32 {(one_compl)}, {divisor_sign}, {one_compl}")
            kk(f"s_sub_i32 {(unsigned:=ssa('s'))}, 0, {one_compl}")
            kk(f"v_cvt_f32_u32 {(rcp:=ssa('v'))}, {one_compl}")
            kk(f"v_rcp_iflag_f32 {rcp}, {rcp}")
            kk("s_waitcnt_depctr 0xfff")
            kk(f"v_mul_f32 {rcp}, 0x4f7ffffe, {rcp}")
            kk(f"s_ashr_i32 {(num_sign:=ssa('s'))}, {(two_compl_num:=r[vin[0]])}, 31")
            kk(f"s_add_i32 {(one_comp_num:=ssa('s'))}, {two_compl_num}, {num_sign}")
            kk(f"s_xor_b32 {one_comp_num}, {one_comp_num}, {num_sign}")
            kk(f"s_xor_b32 {(res_sign:=ssa('s'))}, {divisor_sign}, {num_sign}")
            kk(f"v_cvt_u32_f32 {(rcp)}, {rcp}")
            kk(f"v_readfirstlane_b32 {(rcp_scalar:=ssa('s'))}, {rcp}")
            kk(f"s_mul_i32 {(idk:=ssa('s'))}, {unsigned}, {rcp_scalar}")
            kk(f"s_mul_hi_u32 {idk}, {rcp_scalar}, {idk}")
            kk(f"s_add_i32 {rcp_scalar}, {rcp_scalar}, {idk}")
            kk(f"s_mul_hi_u32 {rcp_scalar}, {one_comp_num}, {rcp_scalar}")
            kk(f"s_mul_i32 {idk}, {rcp_scalar}, {one_compl}")
            kk(f"s_sub_i32 {(result:=ssa('s'))}, {one_comp_num}, {idk}")
            kk(f"s_add_i32 {idk}, {rcp_scalar}, 1")
            kk(f"s_sub_i32 {(idk3:=ssa('s'))}, {result}, {one_compl}")
            kk(f"s_cmp_ge_u32 {result}, {one_compl}")
            kk(f"s_cselect_b32 {rcp_scalar}, {idk}, {rcp_scalar}")
            kk(f"s_cselect_b32 {(result)}, {idk3}, {result}")
            kk(f"s_add_i32 {(idk)}, {rcp_scalar}, 1")
            kk(f"s_cmp_ge_u32 {result}, {one_compl}")
            kk(f"s_cselect_b32 {(result)}, {idk}, {rcp_scalar}")
            kk(f"s_xor_b32 {(result)}, {result}, {res_sign}")
            kk(f"s_sub_i32 {result}, {result}, {res_sign}")
            r[u] = result
        elif args is BinaryOps.MOD:
          kk(f"s_ashr_i32 {(divisor_sign:=ssa('s'))}, {(two_compl:=r[vin[1]])}, 31")
          kk(f"s_add_i32 {(one_compl:=ssa('s'))}, {two_compl}, {divisor_sign}")
          kk(f"s_xor_b32 {(one_compl)}, {divisor_sign}, {one_compl}")
          kk(f"s_sub_i32 {(unsigned:=ssa('s'))}, 0, {one_compl}")
          kk(f"v_cvt_f32_u32 {(rcp:=ssa('v'))}, {one_compl}")
          kk(f"v_rcp_iflag_f32 {rcp}, {rcp}")
          kk("s_waitcnt_depctr 0xfff")
          kk(f"v_mul_f32 {rcp}, 0x4f7ffffe, {rcp}")
          kk(f"s_ashr_i32 {(num_sign:=ssa('s'))}, {(two_compl_num:=r[vin[0]])}, 31")
          kk(f"s_add_i32 {(one_comp_num:=ssa('s'))}, {two_compl_num}, {num_sign}")
          kk(f"s_xor_b32 {one_comp_num}, {one_comp_num}, {num_sign}")
          kk(f"v_cvt_u32_f32 {(rcp)}, {rcp}")
          kk(f"v_readfirstlane_b32 {(rcp_scalar:=ssa('s'))}, {rcp}")
          kk(f"s_mul_i32 {(idk:=ssa('s'))}, {unsigned}, {rcp_scalar}")
          kk(f"s_mul_hi_u32 {idk}, {rcp_scalar}, {idk}")
          kk(f"s_add_i32 {rcp_scalar}, {rcp_scalar}, {idk}")
          kk(f"s_mul_hi_u32 {rcp_scalar}, {one_comp_num}, {rcp_scalar}")
          kk(f"s_mul_i32 {idk}, {rcp_scalar}, {one_compl}")
          kk(f"s_sub_i32 {(result:=ssa('s'))}, {one_comp_num}, {idk}")
          kk(f"s_sub_i32 {(idk3:=ssa('s'))}, {result}, {one_compl}")
          kk(f"s_cmp_ge_u32 {result}, {one_compl}")
          kk(f"s_cselect_b32 {(result)}, {idk3}, {result}")
          kk(f"s_sub_i32 {(idk3)}, {result}, {one_compl}")
          kk(f"s_cmp_ge_u32 {result}, {one_compl}")
          kk(f"s_cselect_b32 {(result)}, {idk3}, {result}")
          kk(f"s_xor_b32 {(result)}, {result}, {num_sign}")
          kk(f"s_sub_i32 {result}, {result}, {num_sign}")
          r[u] = result
        else:
          pref = 's' if dtype == dtypes.bool or dtypes.is_int(dtype) else 'v'
          kk(lang.asm_for_op[args](ssa(pref, u), *[r[x] for x in vin], dtype, lang.types[dtype]))
      elif uop is UOps.DEFINE_ACC:
        if dtype.count > 1:
          r[u] = [ssa('acc', dtype=lang.types[dtype.scalar()]) for _ in range(dtype.count)]
          for uu in r[u]: kk(f"mov.b{lang.types[dtype.scalar()][1:]} {uu}, {const(args, dtype.scalar())};")
        else: kk(f"mov.b{lang.types[dtype][1:]} {ssa('acc', u)}, {const(args, dtype)};")
      elif uop is UOps.SPECIAL:
        if u.arg[1].startswith("lidx"):
          r[u] = f'v{u.arg[0]}'
        elif u.arg[1].startswith("gidx"):
          r[u] = f's{2+u.arg[0]}'
      elif uop is UOps.CONST:
        pref = 's' if dtype == dtypes.bool else 'v'
        pref = 's' if dtype == dtypes.bool or dtypes.is_int(dtype) else 'v'
        kk(f"{pref}_mov_b{lang.types[dtype][1:]} {ssa(pref, u)}, {render_val(args, dtype)}")
      elif uop is UOps.GEP: r[u] = r[vin[0]][u.arg]
      elif uop is UOps.LOAD:
        assert vin[1].dtype is not None
        if "shared" in u.arg:
          kk(f"flat_load_{lang.mem_type(dtype)} {ssa('v', u)}, {r_as(vin[1], 'v')}, {r[vin[0]]}")
        elif dtypes.is_int(dtype) and not dtypes.is_unsigned(dtype):
          if vin[1].uop is UOps.CONST:
            r[vin[1]] = render_val(vin[1].arg, vin[1].dtype)
          kk(f"s_load_{lang.mem_type(dtype)} {ssa('s', u)}, {r[vin[0]]}, {r[vin[1]]}")
        else:
          kk(f"global_load_{lang.mem_type(dtype)} {ssa('v', u)}, {r_as(vin[1], 'v')}, {r[vin[0]]}")
      elif uop is UOps.DEFINE_GLOBAL:
        idx = u.arg[0]
        r[u] = f"s[{(idx+1)*2}:{(idx+1)*2+1}]"
        kk(f"s_load_b64 {r[u]}, s[0:1], {idx*8}")
        kk("s_waitcnt lgkmcnt(0)")
      elif uop is UOps.DEFINE_LOCAL:
        kk(f"v_mov_b32 {ssa('v', u)}, 0")
      elif uop is UOps.CAST: 
        if vin[0].dtype == dtypes.bool:
          kk(lang.asm_for_op[TernaryOps.WHERE](ssa('s' if dtype == dtypes.bool else 'v', u), r[vin[0]], 1, 0, dtype, lang.types[dtype]))
        elif dtype  == dtypes.bool:
          kk(lang.asm_for_op[BinaryOps.CMPEQ](ssa('s', u), 1, r[vin[0]], dtype, lang.types[vin[0].dtype]))
        else:
          kk(f"v_cvt_{lang.types[dtype]}_{lang.types[vin[0].dtype]} {r[vin[0]]}, {r[vin[0]]}")
          r[u] = r[vin[0]]
      else: raise NotImplementedError(f"no code for {uop}")

  # return lang.render_kernel(kernel, function_name, bufs, c.items())
  kern ="\n".join(kernel)
  return lang.kernel_prefix + kern + """
s_nop 0                                                    // 000000001628: BF800000
s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000162C: BFB60003
s_endpgm                                                   // 000000001630: BFB00000
.amdgpu_metadata
amdhsa.kernels:
  - .args:
      ; - .address_space:  global
      ;   .name:           buf0
      ;   .offset:         0
      ;   .size:           8
      ;   .type_name:      'float*'
      ;   .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .language:       OpenCL C
    .language_version:
      - 1
      - 2
    .max_flat_workgroup_size: 256
    .name:           code
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .sgpr_spill_count: 0
    .symbol:         code.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1100
amdhsa.version:
  - 1
  - 2
.end_amdgpu_metadata
"""
class RDNA3Language(AssemblyLanguage):
  kernel_prefix=""".global _start
_start:
.rodata
.align 0x10
.global code.kd
.type code.kd,STT_OBJECT
code.kd:
.long 0,0,0,0
.long 0x00000bc0,0x00000000,0x00000000,0x00000000
.long 0,0,0,0
.long 0x60af0000,0x0000009e,0x00000408,0x00000000
.text
.global code
.type code,STT_FUNC
code:
"""
  asm_for_op = {
    UnaryOps.NEG: lambda d,a,dt,name: f"{d[0]}_xor_b{name[1:]} {d}, {a}, {'1' if dt == dtypes.bool else '0x8' + ('0' * (2*dt.itemsize-1))}",
    UnaryOps.EXP2: lambda d,a,dt,name: f"v_exp_{name} {d}, {a};", UnaryOps.LOG2: lambda d,a,dt,name: f"v_log_{name} {d}, {a};",
    UnaryOps.SIN: lambda d,a,dt,name: f"v_sin_{name} {d}, {a};", UnaryOps.SQRT: lambda d,a,dt,name: f"v_sqrt_{name} {d}, {a};",
    BinaryOps.ADD: lambda d,a,b,dt,name: f"{'s_or_' if dt == dtypes.bool else f'{d[0]}_add_'}{name} {d}, {a}, {b};",
    BinaryOps.SUB: lambda d,a,b,dt,name: f"{d[0]}_sub_{name} {d}, {a}, {b};",
    BinaryOps.MUL: lambda d,a,b,dt,name: f"{'s_and_' if dt == dtypes.bool else f'{d[0]}_mul_'}{'lo_' if 'u' in name else ''}{name} {d}, {a}, {b};",
    BinaryOps.XOR: lambda d,a,b,dt,name: f"{d[0]}_xor_b{name[1:]} {d}, {a}, {b};",
    BinaryOps.MAX: lambda d,a,b,dt,name: f"v_max_{name} {d}, {a}, {b};", BinaryOps.MOD: lambda d,a,b,dt,name: f"rem.{name} {d}, {a}, {b};",
    BinaryOps.CMPLT: lambda d,a,b,dt,name: f"{'s' if dt == dtypes.bool else 'v'}_cmp_lt_{name} {d}, {a}, {b};",
    BinaryOps.CMPEQ: lambda d,a,b,dt,name: f"v_cmp_eq_{name} {d}, {a}, {b};",
    TernaryOps.MULACC: lambda d,a,b,c,dt,name: f"fma.rn.{name} {d}, {a}, {b}, {c};",
    TernaryOps.WHERE: lambda d,a,b,c,dt,name: f"v_cndmask_b{name[1:]} {d}, {c}, {b}, {a}"
  }

  types = { dtypes.int8: "i8", dtypes.int16: "i16", dtypes.int32: "i32", dtypes.int64: "i64",
            dtypes.uint8: "u32", dtypes.uint16: "u16", dtypes.uint32: "u32", dtypes.uint64: "u64",
            dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64", dtypes.bool: "b32" }
  def mem_type(self, dtype): return self.mem_types[dtype]
  mem_types = { dtypes.int8: "b8", dtypes.int16: "b16", dtypes.int32: "b32", dtypes.int64: "b64",
            dtypes.uint8: "u8", dtypes.uint16: "u16", dtypes.uint32: "u32", dtypes.uint64: "u64",
            dtypes.float16: "u16", dtypes.float32: "b32", dtypes.float64: "b64", dtypes.bool: "u8" }

RDNA3Renderer = functools.partial(uops_to_rdna, RDNA3Language())
