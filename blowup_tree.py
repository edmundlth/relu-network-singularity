from sympy import symbols, factor, latex
from IPython.display import display, Math
from utils import (
    get_jacobian_matrix,
    get_sym_chart_str,
    get_rlct,
    is_normal_crossing,
    embedded_blowup,
)
import networkx as nx
import matplotlib.pyplot as plt


class BlowupTree(object):
    def __init__(self, H):
        self.tree = nx.DiGraph()
        self.H = H
        vs = symbols(" ".join([f"{s}_{i + 1}" for i in range(H) for s in ["a", "b"]]))
        self.f = sum([vs[2 * i] * vs[2 * i + 1] for i in range(0, H)])
        self.K = self.f**2
        subs = [(v, v) for v in vs]
        attr = {"vars": vs, "expr": self.f, "map": subs, "subs": subs}
        self.tree.add_node("", **attr)

        self.task_stack = [""]

    def next_blowup(self):
        if not self.task_stack:
            print("No more blow-up task.")
            return

        chart_old = self.task_stack.pop()
        rec = self.tree.nodes[chart_old]
        vars_old = rec["vars"]
        expr_old = rec["expr"]
        map_old = rec["map"]
        print("\n---------------\nCurrent expression: ")
        display(Math(latex(expr_old)))
        blowup_coord_indices = [
            int(num) for num in input("Enter coordinate indices: ").split()
        ]
        print(f"Blowup coords: {blowup_coord_indices}")
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
            display(Math(latex(f_pullback)))

            # If new expression is not normal crossing, add it to the blow-up task stack.
            if not is_nc:
                self.task_stack.append(chart_new)
            else:  # if it is normal crossing, calculate the RLCTs
                rlcts = get_rlct(f_pullback, det)
                rec["rlcts"] = rlcts
                display(Math(f"RLCTS = {latex(rlcts)}"))
                self.tree.nodes[chart_new]["rlcts"] = rlcts
        return

    def blowup_game(self):
        while self.task_stack:
            self.next_blowup()
        print("Tasks stack is empty!")
        return

    def check_resolved(self):
        for chart in nx.dfs_preorder_nodes(self.tree, source=""):
            if len(list(self.tree.successors(chart))) == 0:  # is a leaf node
                if not is_normal_crossing(self.tree.nodes[chart]["expr"]):
                    return False
        return True

    def display_tree(self):
        assert nx.is_tree(self.tree)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        nx.draw(
            self.tree,
            with_labels=True,
            ax=ax,
            pos=nx.planar_layout(self.tree),
            node_size=10,
            font_size=8,
        )
        return fig, ax
