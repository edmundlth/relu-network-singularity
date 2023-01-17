from sympy import symbols, factor, latex, Add, Symbol, Eq
from IPython.display import display, Math
from utils import (
    get_jacobian_matrix,
    get_sym_chart_str,
    get_rlct,
    is_normal_crossing,
    embedded_blowup,
    find_any_nonexceptional_factor,
)
import networkx as nx
import matplotlib.pyplot as plt


class BlowupTree(object):
    def __init__(self, H, verbose=False):
        self.H = H
        self.verbose = verbose

        self.tree = nx.DiGraph()
        vs = symbols(" ".join([f"{s}_{i + 1}" for i in range(H) for s in ["a", "b"]]))
        self.f = sum([vs[2 * i] * vs[2 * i + 1] for i in range(0, H)])
        self.K = self.f**2
        subs = [(v, v) for v in vs]
        attr = {"vars": vs, "expr": self.f, "map": subs, "subs": subs}
        self.tree.add_node("", **attr)

        self.task_stack = [""]

    def next_blowup(self, blowup_coord_indices=None):
        if not self.task_stack:
            print("No more blow-up task.")
            return

        chart_old = self.task_stack.pop()
        rec = self.tree.nodes[chart_old]
        vars_old = rec["vars"]
        expr_old = rec["expr"]
        map_old = rec["map"]
        
        if self.verbose:
            print("\n---------------\nCurrent expression: ")
            display(Math(latex(expr_old)))
        
        if blowup_coord_indices is None:    
            blowup_coord_indices = [
                int(num) for num in input("Enter coordinate indices: ").split()
            ]
        
        if self.verbose: 
            print(f"Blowup coords    : {blowup_coord_indices}")
            display(Math(f"Blowup variables : {latex([vars_old[i] for i in blowup_coord_indices])}"))
        rec["blowup-coords"] = blowup_coord_indices

        blow_up_subs = embedded_blowup(vars_old, var_indices=blowup_coord_indices)
        while blow_up_subs:
            vars_new, subs = blow_up_subs.pop()
            chart_new = get_sym_chart_str(vars_new[0])
            map_new = [(v, e.subs(subs)) for v, e in map_old]
            expr_new = factor(expr_old.subs(subs))

            f_pullback = self.f.subs(map_new).factor()
            is_nc = is_normal_crossing(f_pullback)
            J = get_jacobian_matrix(map_new, vars_new)
            det = J.det()
            attr = {
                "vars": vars_new,
                "subs": subs,
                "expr": expr_new,
                "map": map_new,
                "is_normal_crossing": is_nc,
                "J": J,
                "det(J)": det,
                "f_pullback": f_pullback,
            }
            self.tree.add_node(chart_new, **attr)
            self.tree.add_edge(chart_old, chart_new)

            if self.verbose:
                display(Math(latex(f_pullback)))

            # If new expression is not normal crossing, add it to the blow-up task stack.
            if not is_nc:
                self.task_stack.append(chart_new)
            else:  # if it is normal crossing, calculate the RLCTs
                rlcts = get_rlct(f_pullback, det)
                rec["rlcts"] = rlcts
                self.tree.nodes[chart_new]["rlcts"] = rlcts

                rlct_vals = [r for _, r in rlcts]
                min_rlct = min(rlct_vals)
                multiplicity = rlcts.count(rlct_vals)
                self.tree.nodes[chart_new]["rlct_min"] = min_rlct
                self.tree.nodes[chart_new]["multiplicity"] = multiplicity
                if self.verbose:
                    display(Math(f"RLCTS = {latex(rlcts)}"))
                    display(Math(f"(\lambda, m) = ({min_rlct}, {multiplicity})"))

        return

    def blowup_game(self):
        while self.task_stack:
            self.next_blowup()
        print("Tasks stack is empty!")
        return

    def auto_blowup(self):
        blowup_coords = self.find_next_blowup_coord()
        while blowup_coords:
            self.next_blowup(blowup_coords)
            blowup_coords = self.find_next_blowup_coord()
        return

    def find_next_blowup_coord(self):
        if not self.task_stack:
            print("Task stack empty")
            return None
        rec = self.tree.nodes[self.task_stack[-1]]
        var_list = rec["vars"]
        expr = rec["expr"]
        if is_normal_crossing(expr):
            print("Already normal crossing")
            return None

        nonexceptional_factor = find_any_nonexceptional_factor(expr)
        if nonexceptional_factor is None:
            print("No non-exceptional factor")
            return None

        blowup_coords = []
        for term in Add.make_args(nonexceptional_factor):
            any_var = term.atoms(Symbol).pop()
            blowup_coords.append(var_list.index(any_var))
        return blowup_coords

    def check_resolved(self):
        for chart in nx.dfs_preorder_nodes(self.tree, source=""):
            if len(list(self.tree.successors(chart))) == 0:  # is a leaf node
                if not is_normal_crossing(self.tree.nodes[chart]["expr"]):
                    return False
        return True

    def display_tree(self, with_labels=True, node_size=10, font_size=8, figsize=(5, 5)):
        assert nx.is_tree(self.tree)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        nx.draw(
            self.tree,
            with_labels=with_labels,
            ax=ax,
            pos=nx.planar_layout(self.tree),
            node_size=node_size,
            font_size=font_size,
        )
        return fig, ax


    def display_leaves_data(
            self, 
            display_expr=True, 
            display_det=True, 
            display_rlcts=True, 
            display_j=False, 
            display_mapping=True, 
            display_k=True
        ):
        for chart in nx.dfs_preorder_nodes(self.tree, source=''):
            if len(list(self.tree.successors(chart))) != 0: # if it is a leaf node. 
                continue

            rec = self.tree.nodes[chart]
            m = rec["map"]
            expr = rec["expr"]
            var_list = rec["vars"]
            J = rec["J"]
            det = rec["det(J)"]
            rlcts = rec["rlcts"]
            
            
            f_pullback = self.f.subs(m).factor()
            K_pullback = self.K.subs(m).factor()
            is_nc = is_normal_crossing(f_pullback)
            assert is_nc, "Error: Not normal crossing at a leaf node."
            assert Eq((expr - f_pullback).expand(), 0), f"Error: {expr - f_pullback}"

            if display_expr:
                display(Math(f"f^* = {latex(expr)}"))
            if display_det:
                display(Math(f"det(J) = {latex(det)}"))
            if display_rlcts:
                for s, lmbda_val in rlcts:
                    lmbda_s = symbols(f"lambda_{s}")
                    display(Math(f"{latex(Eq(lmbda_s, lmbda_val))}"))
            if display_j:
                display(Math(latex(J)))
            if display_mapping:
                for x in m:
                    display(Math(latex(Eq(x[0], x[1]))))
            if display_k:
                display(Math(f"K^* = {latex(K_pullback)}"))
            print("------------\n")