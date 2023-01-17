from sympy import (
    symbols,
    factor_list,
    Eq,
    Matrix,
    diff,
    Symbol,
    Poly,
    Add,
)


def get_sym_chart_str(sym):
    sym = str(sym)
    if "(" not in sym:
        return ""
    else:
        return sym.split("(")[1].split(")")[0]


def get_sym_name(sym):
    sym = str(sym)
    if "(" not in sym:
        return sym
    else:
        return sym.split("^")[0]


def get_rlct(f_pullback, det_of_jacobian):
    result = []
    for fac, exponent in factor_list(f_pullback)[1]:
        if (
            Eq(fac.subs([(v, 0) for v in fac.free_symbols]), 0)
            and len(fac.atoms()) == 1
        ):
            for fac_j, exponent_j in factor_list(det_of_jacobian)[1]:
                if fac_j.atoms() == fac.atoms():
                    rlct = (exponent_j + 1) / exponent
                    result.append((fac, rlct))
                    break
    return result


def get_jacobian_matrix(mapping, var_list):
    return Matrix([[diff(e, v) for v in var_list] for _, e in mapping])


def _eval_at_val(expr, val=0):
    return expr.subs([(v, val) for v in expr.free_symbols]).expand()

def _is_exceptional_divisor(expr):  
    return (len(expr.atoms(Symbol)) == 1) and (len(Add.make_args(expr)) == 1)

def find_any_nonexceptional_factor(expr): 
    # might not be the entire strict transform, 
    # e.g. x**2 * y * (1 + y) * (1 + z) would return (1 + y)
    for t, m in factor_list(expr)[1]:
        if not _is_exceptional_divisor(t):
            return t
    return None

def is_normal_crossing(expr):
    factors = factor_list(expr.factor())[1]
    for t, m in factors:
        if not _is_exceptional_divisor(t):
            if Eq(_eval_at_val(t), 0):
                return False
    return True

# def is_normal_crossing(expr):
#     factors = factor_list(expr.factor())[1]
#     for t, m in factors:B
#         # if term t is not an exceptional divisor
#         if len(t.atoms(Symbol)) != 1:
#             if Eq(_eval_at_val(t), 0):
#                 return False
#         else:
#             deg_list = Poly(t).degree_list()
#             if len(deg_list) != 1:  # therefore not an exceptional divisor
#                 if 0 not in deg_list:
#                     return False
#     return True


def embedded_blowup(v_old, var_indices=None):
    if var_indices is None:
        var_indices = list(range(len(v_old)))
    else:
        var_indices = sorted(var_indices)
    excluded_indices = sorted(set(range(len(v_old))) - set(var_indices))

    sub_list = []
    for i, vidx in enumerate(var_indices):
        sym_strings = [
            f"{get_sym_name(s)}^({get_sym_chart_str(s)}{i + 1})" for s in v_old
        ]
        v = symbols(" ".join(sym_strings))
        subs = [(v_old[vidx], v[vidx])]
        subs += [(v_old[k], v[k]) for k in excluded_indices]
        subs += [(v_old[k], v[vidx] * v[k]) for k in var_indices if k != vidx]

        sub_list.append((v, subs))
    return sub_list
