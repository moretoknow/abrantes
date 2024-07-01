import networkx as nx
import numpy as np
import heapq

DEFAULT_FV_METHOD = 'lobpcg'

def spectral_split(graph, fv_method=DEFAULT_FV_METHOD, fv_normalized=True):
    if len(graph) == 1:
        yield from [graph]
        return

    if not nx.is_connected(graph):
        yield from map(lambda cc: graph.subgraph(cc), nx.connected_components(graph))
        return

    fv_pos = nx.fiedler_vector(graph, normalized=fv_normalized, method=fv_method) > 0
    nodes = np.array(graph)

    for i in [nodes[fv_pos], nodes[~fv_pos]]:
        yield from map(lambda cc: graph.subgraph(cc), nx.connected_components(graph.subgraph(i)))

def abrantes(graph, stopping_fx, priority_fx, fv_method=DEFAULT_FV_METHOD, fv_normalized=True):
    make_pq_item = lambda subgraph: [[priority_fx(subgraph, graph), id(subgraph)], subgraph]

    subgraph_pq = []

    heapq.heappush(subgraph_pq, make_pq_item(graph))

    while not stopping_fx(subgraph_pq):
        _, subgraph = heapq.heappop(subgraph_pq)
        for subgraph_split in spectral_split(subgraph, fv_method, fv_normalized):
            heapq.heappush(subgraph_pq, make_pq_item(subgraph_split))
            
    return [item[1] for item in subgraph_pq]
