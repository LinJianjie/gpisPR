import networkx as nx
class Graph:
    def __init__(self) -> None:
        self._nodes=None
        self._edges=None
    def full_connected_grasph(self):
        pass
    @property
    def nodes(self):
        return self._nodes
    @property
    def edges(self):
        return self._edges    
    def subgraph_matching(self, B,A):
        GM = nx.algorithms.isomorphism.GraphMatcher(B,A)
        for subgraph in GM.subgraph_isomorphisms_iter():
            print(subgraph)

if __name__=="__main__":
    #G = nx.Graph()
    import numpy as np
    G1=nx.complete_graph(5)
    for i in np.arange(1,6):
        for j in np.arange(i,6):
            G1.add_edge(i, j, weight=np.random.random())
    print(G1)
    G2=nx.complete_graph(5)
    for i in np.arange(1,6):
        for j in np.arange(i,6):
            G2.add_edge(i, j, weight=np.random.random())
    GM = nx.algorithms.isomorphism.GraphMatcher(G1,G2)
    for index,subgraph in enumerate(GM.subgraph_isomorphisms_iter()):
            print(index,subgraph)