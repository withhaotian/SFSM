
import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph, graph

plt.rcParams['font.family'] = 'sans-serif'  
plt.rcParams['font.size'] = 10  
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['mathtext.fontset'] = 'cm'

class DAGFigure():
    def visual(self, G: DiGraph, job_name=None, disp_computations=False):
        # G: DiGraph
        plt.title(job_name, y=-0.1, fontsize=20)

        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        nx.draw(G, pos, font_color='whitesmoke', with_labels=True)
        edge_labels = nx.get_edge_attributes(G, "data")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        if disp_computations:
            node_labels = nx.get_node_attributes(G, 'computations')
            nx.draw_networkx_labels(G, pos, labels=node_labels)

        plt.axis("off")
        plt.show()