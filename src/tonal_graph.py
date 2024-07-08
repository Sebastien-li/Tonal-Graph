"""
Module for the TonalGraph class.
The TonalGraph class is used to create a graph of roman numerals
It builds upon the RhythmTreeAnalyzed class and the tonality_distance function.
"""
import numpy as np

from src.rhythm_tree import RhythmTreeAnalyzed
from src.music_theory_classes import Pitch, Interval
from src.tonality_distance import get_tonality_distance
from src.utils import cartesian_product

class TonalGraph:
    """ Class for the tonal graph"""
    node_dtype = [('id',np.int32),
                  ('rt_id',np.int32),
                  ('onset',np.int32),
                  ('duration',np.int32),
                  ('tonic_diatonic',np.int32),
                  ('tonic_chromatic',np.int32),
                  ('mode','U1'), # M or m
                  ('label','U7'), # IIImaj7 is longest with 7 characters
                  ('inversion',np.int32),
                  ('weight','float'),
                  ('selected','bool')]

    edge_attr_dtype = [('weight','float'),
                       ('selected','bool')]

    def __init__(self,
                 rhythm_tree_analyzed: RhythmTreeAnalyzed,
                 mode_list:list,
                 qualities=None,
                 chord_depth = 5,
                 neighbor_weight = 1,
                 relative_weight = 0.7,
                 parallel_weight = 1.3,
                 enharmonic_weight = 0.5,
                 dominant_weight = 1.2):

        self.rt = rhythm_tree_analyzed
        self.duration_divisor = rhythm_tree_analyzed.duration_divisor
        self.qualities = rhythm_tree_analyzed.qualities if qualities is None else qualities
        self.mode_list = mode_list
        self.quality_to_rn = self.get_quality_to_rn()
        self.chord_depth = chord_depth
        self.tonality_distances, self.keys_graph = get_tonality_distance(neighbor_weight,
                                                                         relative_weight,
                                                                         parallel_weight,
                                                                         enharmonic_weight,
                                                                         dominant_weight)
        self.nodes = self.create_nodes()
        self.onsets = np.unique(self.nodes['onset'])
        self.edge_index, self.edge_attr = self.create_edges()

        self.nodes_extended, self.edge_index_extended, self.edge_attr_extended = self.extend_graph()
        self.shortest_path = self.find_shortest_path()

    def __len__(self):
        return len(self.nodes)

    def get_quality_to_rn(self):
        """Returns a dictionary that maps a quality to a list of roman numerals with that quality"""
        quality_to_rn = {} # quality -> [(roman_numeral,mode)]
        for mode in self.mode_list :
            for roman_numeral in mode:
                if roman_numeral.quality not in quality_to_rn:
                    quality_to_rn[roman_numeral.quality] = []
                quality_to_rn[roman_numeral.quality].append((roman_numeral,mode))
        return quality_to_rn

    def create_nodes(self):
        """ Creates the nodes of the graph. Each node represents a roman numeral with a mode."""
        nodes = []
        i=0
        for rt_i,rt_node in enumerate(self.rt.depth_first_search()):
            if not rt_node.selected:
                continue
            n_best = rt_node.get_n_best(self.chord_depth)
            for diatonic, chromatic, quality_idx in n_best:
                possible_rn = self.quality_to_rn[self.qualities.idx_to_name[quality_idx]]
                for roman_numeral, mode in possible_rn:
                    diatonic_root = int(roman_numeral.diatonic_root)
                    chromatic_root = int(roman_numeral.chromatic_root)
                    to_tonic = Interval(Pitch(diatonic_root, chromatic_root),Pitch(0,0))
                    tonic = Pitch(int(diatonic), int(chromatic)) + to_tonic
                    nodes.append((i,
                                  rt_i,
                                  rt_node.onset,
                                  rt_node.duration,
                                  tonic.diatonic,
                                  tonic.chromatic,
                                  mode.name,
                                  roman_numeral.label,
                                  rt_node.inversion[diatonic, chromatic, quality_idx],
                                  1-rt_node.root_score[diatonic,chromatic,quality_idx]*\
                                  roman_numeral.score,
                                  False))
                    i+=1

        nodes_array = np.array(nodes, dtype = self.node_dtype)
        nodes_array['id'] = np.arange(len(nodes_array))
        return nodes_array

    def create_edges(self):
        """
        Creates the edges of the graph.
        Each edge links two nodes that are consecutive in time.
        """
        edge_index = np.zeros((0,2),dtype=np.int32)
        edge_attr = np.zeros(0, dtype=self.edge_attr_dtype)
        onsets = sorted(set(self.nodes['onset']))
        for i in range(len(onsets)-1):
            u_onset = onsets[i]
            v_onset = onsets[i+1]
            u_nodes = self.nodes[self.nodes['onset'] == u_onset]
            v_nodes = self.nodes[self.nodes['onset'] == v_onset]
            uv = cartesian_product(u_nodes,v_nodes)
            u, v = uv[:,0], uv[:,1]
            edge_index = np.concatenate((edge_index, uv['id']))
            edge_attr = np.concatenate((edge_attr, self.get_edge_attr(u,v)))
        return edge_index, edge_attr

    def get_edge_attr(self,u,v):
        """ Computes the distance between two nodes"""
        diatonic_interval = (v['tonic_diatonic'] - u['tonic_diatonic'])%7
        chromatic_interval = (v['tonic_chromatic'] - u['tonic_chromatic'])%12
        mode_distance = (u['mode']=='m') + 2*(v['mode']=='m')
        key_distance = self.tonality_distances[diatonic_interval,chromatic_interval,mode_distance]
        key_distance[key_distance!=0]+=1
        edge_weight = (u['weight'] + v['weight'])/2 + key_distance * 0.05
        edge_attr = edge_weight.astype(self.edge_attr_dtype)
        edge_attr['selected'] = False
        return edge_attr

    def extend_graph(self):
        """ Adds a start and end node to the graph for the shortest path algorithm."""
        start = np.array([(-1, -1, -1, 1, 0, 0, 'M', 'Start', 0, 0, True)], dtype=self.node_dtype)
        end = np.array([(len(self), -1,
                         self.nodes[-1]['onset'],
                         1, 0, 0, 'M', 'End', 0, 0, True)], dtype=self.node_dtype)
        nodes_extended = np.concatenate((start, self.nodes, end))
        starting_nodes = self.nodes[self.nodes['onset'] == 0]
        ending_nodes = self.nodes[self.nodes['onset'] == self.nodes[-1]['onset']]
        starting_edges = np.array([(-1, node['id']) for node in starting_nodes])
        ending_edges = np.array([(node['id'], len(self)) for node in ending_nodes])

        starting_edge_attr = np.array([(0,False) for _ in starting_nodes],
                                      dtype=self.edge_attr_dtype)
        ending_edge_attr = np.array([(0,False) for _ in ending_nodes],
                                    dtype=self.edge_attr_dtype)

        edge_index_extended = np.concatenate((starting_edges, self.edge_index, ending_edges)) + 1
        edge_attr_extended = np.concatenate(( starting_edge_attr, self.edge_attr, ending_edge_attr))
        sort_idx = np.lexsort((edge_index_extended[:,1], edge_index_extended[:,0]))
        edge_index_extended = edge_index_extended[sort_idx]
        edge_attr_extended = edge_attr_extended[sort_idx]
        return nodes_extended, edge_index_extended, edge_attr_extended


    def find_shortest_path(self):
        """Shortest path algorithm for a DAG. Returns the nodes of the shortest path."""
        n = self.nodes_extended.shape[0]
        d = np.ones(n, dtype=np.float32) * np.inf
        d[0] = 0.
        predecessors = np.zeros(n, dtype=np.int64)
        e = 0
        for u in range(n):
            if e >= self.edge_index_extended.shape[0]-1:
                break
            while self.edge_index_extended[e,0] == u:
                v = self.edge_index_extended[e,1]
                w = self.edge_attr_extended[e]['weight']
                if d[v] > d[u] + w:
                    d[v] = d[u] + w
                    predecessors[v] = u

                e = e + 1
        end = n - 1
        u = end
        shortest_path = [u]
        while u != 0:
            shortest_path.append(predecessors[u])
            u = predecessors[u]

        for i in shortest_path[::-1][1:-1]:
            self.nodes[i-1]['selected'] = True

        return self.nodes[([i-1 for i in shortest_path[::-1]][1:-1])]
