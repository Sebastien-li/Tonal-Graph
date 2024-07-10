""" This module contains the classes Music21Score and NoteGraph."""

from pathlib import Path
import pickle
from fractions import Fraction
from math import lcm

import numpy as np
import music21

from src.utils import find_le

class Music21Score:
    """Class to represent a score in music21 format."""
    def __init__(self, load_path, composer = '', title = ''):
        self.composer = composer
        self.title = title
        self.load_path = Path(load_path)
        self.score_m21 = music21.converter.parse(self.load_path, forceSource=True)
        self.recursed = self.score_m21.recurse().stream()

        self.notes = sorted(list(self.recursed.notes), key=lambda x: x.offset)
        self.notes_and_rests = self.collapse_rest()

        self.measure_list = self._get_measures()
        self.ts_list = self._get_timesig()

        self._measure_idx_to_ts = self._get_measure_idx_to_ts()

        self.duration_divisor = lcm(*([Fraction(x.offset).denominator
                                       for x in self.notes_and_rests] +
                                      [Fraction(x.duration.quarterLength).denominator
                                      for x in self.notes_and_rests]))


    def _get_measures(self):
        measure_list = []
        for meas in sorted(self.recursed.getElementsByClass('Measure'), key=lambda x: x.offset):
            if measure_list and measure_list[-1][0] == meas.offset:
                continue
            measure_list.append((Fraction(meas.offset),meas.measureNumberWithSuffix(),
                                 Fraction(meas.duration.quarterLength)))
        return measure_list

    def _get_timesig(self):
        list_timesig = []
        for ts in sorted(self.recursed.getElementsByClass('TimeSignature'), key=lambda x: x.offset):
            try:
                ts.beatDuration.quarterLength
            except AttributeError as e:
                raise AttributeError(f"Irregular time signature at onset {ts.offset}") from e
            if list_timesig and list_timesig[-1].offset == ts.offset:
                continue
            list_timesig.append(ts)

        return list_timesig

    def _get_measure_idx_to_ts(self):
        list_timesig_by_measure = []
        i_timesig = 0
        for meas in self.measure_list:
            if i_timesig < len(self.ts_list)-1 and self.ts_list[i_timesig+1].offset <= meas[0]:
                i_timesig += 1
            list_timesig_by_measure.append(self.ts_list[i_timesig])
        return list_timesig_by_measure

    def measure_idx_to_ts(self, idx):
        """ Return the time signature of the measure at index idx."""
        return self._measure_idx_to_ts[idx]

    def onset_to_measure(self, onset):
        """ Return the measure at onset (fraction)."""
        return find_le(self.measure_list, onset, key=lambda x: x[0])

    def onset_to_ts(self, onset):
        """ Return the time signature at onset (fraction) """
        return find_le(self.ts_list, onset, key=lambda x: x.offset)

    def onset_to_measure_and_beat(self, onset):
        """ Return the measure and beat at onset (fraction) """
        measure = self.onset_to_measure(onset)
        ts = self.onset_to_ts(onset)
        onset_in_measure = Fraction(onset) - Fraction(measure[0])
        # To count anacruses ...
        beat = 1 + (onset_in_measure ) / Fraction(ts.beatDuration.quarterLength)
        return measure, beat

    def collapse_rest(self):
        """ Collapse rests that are consecutive."""
        notes_and_rests = []
        maxi_offset = 0
        for note in self.notes:
            if note.isRest:
                continue
            onset = Fraction(note.offset)
            offset = Fraction(note.offset) + Fraction(note.duration.quarterLength)
            if onset > maxi_offset:
                rest = music21.note.Rest(quarterLength=onset - maxi_offset)
                rest.offset = maxi_offset
                notes_and_rests.append(rest)
            if offset > maxi_offset:
                maxi_offset = offset
            notes_and_rests.append(note)
        return notes_and_rests

    def save(self, save_path):
        """ Save the score in a pickle file."""
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, load_path):
        """ Load a score from a pickle file."""
        with open(load_path, 'rb') as f:
            return pickle.load(f)

class Graph:
    """
    Class to represent a graph with numpy arrays.
    """
    def __init__(self, nodes=None, edge_index=None, edge_attr=None):
        self.nodes = nodes
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def __getitem__(self, i):
        return self.nodes[i]

    def __len__(self):
        return len(self.nodes)

    def order(self):
        """ Return the number of nodes and edges."""
        return len(self.nodes), len(self.edge_index)

    def get_edge_source(self, src):
        """ Return the index of the edges that have src as source."""
        return np.where(self.edge_index['src'] == src)[0]

    def get_edge_dest(self, dst):
        """ Return the index of the edges that have dst as destination."""
        return np.where(self.edge_index['dst'] == dst)[0]

    def edge_index_to_adj(self):
        """ Return the adjacency matrix of the graph."""
        n = len(self.nodes)
        adj = np.zeros((n,n))
        for edge in self.edge_index:
            adj[edge[0], edge[1]] = 1
        return adj

class NoteGraph(Graph):
    """ Class to represent a graph of notes."""
    dtype_nodes = [ ('id',int),
                    ('pitch_chromatic', int),
                    ('pitch_name', 'U10'),
                    ('pitch_diatonic', int),
                    ('pitch_space', np.float32),
                    ('pitch_octave', int),
                    ('onset', int),
                    ('duration', int),
                    ('offset', int),
                    ('measure', 'U10'),
                    ('beat', Fraction),
                    ('isRest', bool)]

    dtype_edges = [('type', 'U10')]

    def __init__(self, score_path):
        super().__init__()
        self.score_path = Path(score_path)
        self.score = None
        self.score : Music21Score
        self.vertical_dict = None
        self.duration_divisor = None

    @classmethod
    def from_xml(cls, score_path):
        """ Create a NoteGraph from an xml or mxl path."""
        note_graph = cls(score_path)
        note_graph.score = Music21Score(note_graph.score_path)
        note_graph.duration_divisor = note_graph.score.duration_divisor
        note_graph.create_graph()
        note_graph.vertical_dict = note_graph.get_vertical_dict()

        return note_graph

    @classmethod
    def from_m21score(cls, score: Music21Score):
        """ Create a NoteGraph from a Music21Score."""
        note_graph = cls(score.load_path)
        note_graph.score = score
        note_graph.duration_divisor = note_graph.score.duration_divisor
        note_graph.create_graph()
        note_graph.vertical_dict = note_graph.get_vertical_dict()
        return note_graph

    def create_nodes(self):
        """ Create the nodes of the graph."""
        note_name_to_class_dict = {'C': 0,'D': 1,'E': 2,'F': 3,'G': 4,'A': 5,'B': 6}
        nodes = []
        onset_to_measure_beat = {}
        for i, note in enumerate(self.score.notes_and_rests):
            if note.quarterLength == 0:
                # Skip grace notes
                continue
            onset_frac = Fraction(note.offset)
            quarter_length_frac = Fraction(note.quarterLength)
            onset = int(onset_frac * self.duration_divisor)
            quarter_length = int(quarter_length_frac * self.duration_divisor)
            if onset_frac not in onset_to_measure_beat:
                onset_to_measure_beat[onset_frac] = self.score.onset_to_measure_and_beat(onset_frac)
            measure, beat = onset_to_measure_beat[onset_frac]
            if note.isRest:
                nodes.append((-1, -1, 'R', -1 , -1, -1,
                              onset, quarter_length, onset+quarter_length,
                              measure[1], beat, True))
                continue
            for pitch in note.pitches:
                pitch_space = pitch.ps
                if note.tie is not None and note.tie.type == 'stop':
                    for i in range(len(nodes)-1, -1, -1):
                        n = nodes[i]
                        if n[4] == pitch_space and n[8] == onset:
                            new_quarter_length = n[7] + quarter_length
                            nodes[i] = (n[0], n[1], n[2], n[3], n[4], n[5], n[6],
                                        new_quarter_length, n[8], n[9], n[10], n[11])
                            break
                else:
                    nodes.append((-1, pitch.pitchClass, pitch.nameWithOctave,
                                    note_name_to_class_dict[pitch.nameWithOctave[0]],
                                    pitch.ps, pitch.octave,
                                    onset, quarter_length, onset+quarter_length,
                                    measure[1], beat, False))

        nodes = np.sort(np.array(nodes, dtype = self.dtype_nodes), order = ['onset', 'pitch_space'])
        nodes['id'] = np.arange(len(nodes))
        return nodes

    def create_edges(self):
        """ Create the edges of the graph."""
        nodes = self.nodes
        edge_index = []
        edge_attr = []

        def append_edge(u, v , edge_index, edge_attr, edge_type = 'onset'):
            if u['id'] < v['id']:
                edge_index.append((u['id'],v['id']))
                edge_attr.append((edge_type))
        for u in nodes:#tqdm(nodes, desc='Creating edges'):
            v_onset = nodes[nodes['onset'] == u['offset']]
            for v in v_onset:
                append_edge(u, v, edge_index, edge_attr, 'onset')

            v_during = np.logical_and(nodes['onset'] > u['onset'], nodes['onset'] < u['offset'])
            for v in nodes[v_during]:
                append_edge(u, v, edge_index, edge_attr, 'during')

            v_follow = nodes[nodes['onset'] == u['offset']]
            for v in v_follow:
                append_edge(u, v, edge_index, edge_attr, 'follow')


        edge_index = np.array(edge_index, dtype = [('src', np.int32), ('dst', np.int32)])
        edge_attr = np.array(edge_attr, dtype = self.dtype_edges)
        #Silence edges
        edges_follow = edge_index[edge_attr['type'] == 'follow']

        for edge in edges_follow:
            if nodes[edge['dst']]['isRest'] and not nodes[edge['src']]['isRest']:
                src_node = nodes[edge['src']]
                node_to_explore = edge['dst']

                while True:
                    following_nodes =  edges_follow[edges_follow['src'] == node_to_explore]['dst']
                    if len(following_nodes) == 0:
                        break
                    if not nodes[following_nodes[0]]['isRest'] :
                        for node in following_nodes:
                            edge_index = np.concatenate((edge_index,
                                            [np.array((src_node['id'], node),
                                            dtype=[('src', np.int32),
                                                   ('dst', np.int32)])]))
                            edge_attr = np.concatenate((edge_attr,
                                            [np.array(('silence'),
                                                dtype=self.dtype_edges)]))
                        break
                    node_to_explore = following_nodes[0]

        return edge_index, edge_attr

    def create_graph(self):
        """ Create the nodes and the edges of the graph. Also find the leap nodes."""
        self.nodes = self.create_nodes()
        self.edge_index, self.edge_attr = self.create_edges()
        self.find_leap()

    def get_vertical_dict(self):
        """Vertical dict : onset -> nodes at onset sorted by pitch_space."""
        nodes = self.nodes
        edge_index = self.edge_index
        edge_attr = self.edge_attr

        vertical_dict = {}
        for node in nodes:
            onset = node['onset']
            if onset not in vertical_dict:
                vertical_dict[onset] = []
                for inc_edge_idx in self.get_edge_dest(node['id']):
                    inc_node = edge_index[inc_edge_idx]['src']
                    inc_edge_attr = edge_attr[inc_edge_idx]
                    if inc_edge_attr['type'] in ['during']:
                        vertical_dict[onset].append(nodes[nodes['id']==inc_node][0])

            insert_index = np.searchsorted([x['pitch_space'] for x in vertical_dict[onset]],
                                           node['pitch_space'])
            vertical_dict[onset].insert(insert_index, node)

        for onset, nodes in vertical_dict.items():
            vertical_dict[onset] = np.array(nodes, dtype = nodes[0].dtype)

        return vertical_dict

    def find_leap(self):
        """ Find the leap nodes."""
        is_leap = []
        for node in self.nodes:
            if node['isRest']:
                is_leap.append(False)
                continue

            inc_edg_idx = self.get_edge_dest(node['id'])
            out_edg_idx = self.get_edge_source(node['id'])
            inc_edg_idx = inc_edg_idx[self.edge_attr[inc_edg_idx]['type'] != 'onset']
            out_edg_idx = out_edg_idx[self.edge_attr[out_edg_idx]['type'] != 'onset']
            inc_nodes = self[self.edge_index[inc_edg_idx]['src']]
            out_nodes = self[self.edge_index[out_edg_idx]['dst']]
            inc_edges_attr = self.edge_attr[inc_edg_idx]
            out_edges_attr = self.edge_attr[out_edg_idx]

            inc_nodes = inc_nodes[np.logical_and(inc_edges_attr['type'] != 'onset',
                                                 ~inc_nodes['isRest'])]

            out_nodes = out_nodes[np.logical_and(out_edges_attr['type'] != 'onset',
                                                 ~out_nodes['isRest'])]

            if inc_nodes.size==0:
                is_leap.append(False)
                continue

            if out_edg_idx.size==0:
                is_leap.append(False)
                continue

            closest_inc_index = np.argmin([abs(x['pitch_space']-node['pitch_space'])
                                            for x in inc_nodes])
            inc_node = inc_nodes[closest_inc_index]
            prev_interval = abs(7*node['pitch_octave']+node['pitch_diatonic'] - 7*inc_node['pitch_octave']-inc_node['pitch_diatonic'])

            closest_out_index = np.argmin([abs(x['pitch_space']-node['pitch_space'])
                                            for x in out_nodes])
            out_node = out_nodes[closest_out_index]
            next_interval = abs(7*out_node['pitch_octave']+out_node['pitch_diatonic'] - 7*node['pitch_octave']-node['pitch_diatonic'])

            is_leap.append(prev_interval > 1 and next_interval > 1)

        new_dtype = self.dtype_nodes + [('isLeap', bool)]
        nodes = np.zeros_like(self.nodes, dtype = new_dtype)
        for name,_ in self.dtype_nodes:
            nodes[name] = self.nodes[name]
        nodes['isLeap'] = is_leap
        self.nodes = nodes
