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
    G=nx.complete_graph(5)
    print(G.nodes)
    print(G.edges)
    G.add_weighted_edges_from([(1, 2, 0.5), (1, 2, 0.75), (2, 3, 0.5)])
    print(G.edges.data())
    print(G.edges(0))     