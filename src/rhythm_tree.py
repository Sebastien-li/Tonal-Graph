""" RhthmTree class with functions to construct and analyze it """

from fractions import Fraction
from pathlib import Path
import pickle
from functools import lru_cache, reduce
import numpy as np
from src.music_theory_classes import Pitch
from src.utils import display_float, interval_collision, interval_in

def get_subdivision(duration:Fraction, beat_ql:Fraction, minimum_subdivision=0.5):
    """ Returns the subdivision of a duration given a beat quarter length"""
    duration = Fraction(duration)
    beat_ql = Fraction(beat_ql)
    if duration <= minimum_subdivision:
        return None
    if duration%3 == 0:
        duration = beat_ql * duration/3
    elif duration%2 == 0 or duration == 1:
        duration /= 2
    elif duration%1.5 == 0:
        duration /= 3
    else:
        duration = Fraction(1,1)
    return duration

class RhythmTree:
    """ A tree structure that represents the possible rhythm segmentations of a piece of music"""
    def __init__(self, note_graph, onset, subdivision, duration, parent, measure_idx, depth = 0,
                 minimum_subdivision = 0.5):
        self.note_graph = note_graph
        self.onset = int(onset)
        self.duration = int(duration)
        self.offset = self.onset + self.duration
        self.duration_divisor = note_graph.duration_divisor
        self.subdivision = subdivision
        self.minimum_subdivision = minimum_subdivision
        self.parent = parent
        self.children = []
        self.measure_idx = measure_idx
        self.ts = self.note_graph.score.measure_idx_to_ts(self.measure_idx)
        self.depth = depth

        all_onsets = np.array(list(self.note_graph.vertical_dict.keys()))
        self.onsets = all_onsets[(self.onset<=all_onsets) & (all_onsets<self.offset)]
        if subdivision is not None:# and len(self.onsets) > 1:
            self.subdivide()
        else:
            self.subdivision = Fraction(self.duration, self.duration_divisor)

        self.remove_duplicates()

    def __str__(self):
        duration_display = [display_float(Fraction(x,self.duration_divisor)) for x in self.onsets]
        return f"onset = {Fraction(self.onset,self.duration_divisor)}, " \
               f"subdivision={self.subdivision}, " \
               f"onsets=[{', '.join(duration_display)}]"

    def __repr__(self):
        return f"RhythmTreeNode(onset={Fraction(self.onset,self.duration_divisor)}, " \
               f"subdivision={self.subdivision}, " \
               f"duration={Fraction(self.duration,self.duration_divisor)})"

    def __getitem__(self,idx):
        return self.children[idx]

    def size(self):
        """ Returns the number of nodes in the tree"""
        return 1 + sum([child.size() for child in self.children])

    def print(self, depth = 0):
        """ Prints the tree structure"""
        print("\t" * depth + str(self))
        for child in self.children:
            child.print(depth+1)

    def add_child(self, child):
        """ Adds a child to the tree"""
        self.children.append(child)

    def subdivide(self):
        """ Subdivides the tree into smaller segments according to the subdivision"""
        new_subdivision = get_subdivision(self.subdivision,
                                          Fraction(self.ts.beatDuration.quarterLength),
                                          self.minimum_subdivision)
        if new_subdivision is None:
            return

        where = (self.onsets-self.onset)%(new_subdivision*self.duration_divisor) == 0
        checkpoints = self.onsets[where]

        for i, checkpoint in enumerate(checkpoints):
            checkpoint_next = checkpoints[i+1] if i+1 < len(checkpoints) else self.offset

            child = RhythmTree(note_graph=self.note_graph,
                               onset=checkpoint,
                               subdivision=new_subdivision,
                               duration=checkpoint_next-checkpoint ,
                               parent=self,
                               measure_idx=self.measure_idx,
                               depth=self.depth+1,
                               minimum_subdivision=self.minimum_subdivision)
            self.add_child(child)

    def depth_first_search(self):
        """ Depth first search generator for the tree"""
        for child in self.children:
            yield from child.depth_first_search()
        yield self

    def save(self, filename):
        """ Saves the tree to a pickle"""
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """ Loads a tree from a pickle"""
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def construct_tree(cls, note_graph, **kwargs):
        """Constructs a rhythm tree from a note graph"""
        last_measure = note_graph.score.measure_list[-1]
        onset_max = (last_measure.onset + last_measure.duration)*note_graph.duration_divisor
        root = RhythmTree(note_graph, 0, None, onset_max, None, 0)
        for i,(measure_onset, _, measure_duration) in enumerate(note_graph.score.measure_list):
            child = RhythmTree(note_graph=note_graph,
                               onset = measure_onset * root.duration_divisor,
                               subdivision = Fraction(measure_duration),
                               duration = measure_duration*root.duration_divisor,
                               parent = root,
                               measure_idx=i,
                               depth=1,
                               **kwargs)
            root.add_child(child)
        return root

    def remove_duplicates(self):
        """ Contracts nodes with only one child"""
        if len(self.children) == 1:
            self.children = self.children[0].children

class RhythmTreeAnalyzed(RhythmTree):
    """ Rhythm tree with analyzed nodes"""
    def __init__(self, rhythm_tree, qualities, parent=None):
        self.__dict__.update(rhythm_tree.__dict__)
        self.parent = parent
        self.qualities = qualities
        self.root_score_onset = np.zeros((7,12,len(qualities)))
        self.children = [RhythmTreeAnalyzed(child, qualities, parent=self)
                         for child in rhythm_tree.children]
        self.note_graph_selected_nodes = self.get_note_graph_selected_nodes()
        self.root_score_before_leap, self.inversion = self.analyze()
        self.root_score_leap = self.analyze_leap()
        self.root_score = self.root_score_before_leap * self.root_score_leap
        self.selected = False
        if self.parent is None:
            self.analyze_onset()
            self.select_nodes()

    def get_note_graph_selected_nodes(self):
        """ Returns the nodes of the note graph that are analyzed by the rhythm node"""
        if self.parent is None:
            return self.note_graph.nodes

        if len(self.children) == 1 :
            return self.children[0].note_graph_selected_nodes
        where = np.logical_and(self.onset < self.note_graph.nodes['offset'],
                            self.offset > self.note_graph.nodes['onset'])
        selected_nodes = self.note_graph.nodes[where]
        return selected_nodes[~selected_nodes['isRest']]

    def analyze(self):
        """ Analyzes the tree and returns the root score for each node (before leap and onset)"""
        root_score = np.zeros((7,12,self.qualities.len))
        inversion = np.zeros((7,12,self.qualities.len), dtype=np.int32)
        selected_nodes = np.sort(self.note_graph_selected_nodes,order='pitch_space')
        if self.parent is None or selected_nodes.size == 0:
            return root_score, inversion
        if len(self.children) == 1 :
            return self.children[0].root_score

        def octave_weight(octave):
            # 1 / (1 + e^(octave-6))
            return [0.997, 0.993, 0.982, 0.952, 0.880, 0.731, 0.5, 0.268, 0.119, 0.047][octave]

        def duration_weight(duration):
            return duration ** 0.5

        def doubling_weight(nb_double):
            # sqrt(nb_double/3) or 1
            if nb_double == 1:
                return 0.577
            if nb_double == 2:
                return 0.816
            return 1

        def get_relative_duration(node):
            intersected_duration = min(self.offset, node['offset']) - max(self.onset, node['onset'])
            return intersected_duration / self.duration

        vertices = {} # {pitch : [(octave, duration), ...] }

        for node in selected_nodes:
            diatonic = int(node['pitch_diatonic'])
            chromatic = int(node['pitch_chromatic'])
            pitch = Pitch(diatonic, chromatic)
            if pitch not in vertices:
                vertices[pitch] = []
            vertices[pitch].append((node['pitch_octave'], get_relative_duration(node)))
        for pitch, otave_duration_list in vertices.items():
            min_octave = min(octave for octave, _ in otave_duration_list)
            sum_duration = min(1,sum(duration for _, duration in otave_duration_list))
            pitch_weight =  octave_weight(min_octave) * \
                            duration_weight(sum_duration) * \
                            doubling_weight(len(otave_duration_list))
            chords = self.qualities.pitch_beam[pitch]
            for (root,quality,chord_score) in chords:
                diatonic, chromatic = root.diatonic, root.chromatic
                quality_idx = self.qualities.name_to_idx[quality]
                notes = self.qualities.chord_array[root.diatonic, root.chromatic, quality_idx]
                chord_score *= len(notes)

                root_score[diatonic, chromatic, quality_idx] += pitch_weight*chord_score

        for diatonic_root, chromatic_root, quality_idx in np.argwhere(root_score):
            notes = self.qualities.chord_array[diatonic_root, chromatic_root, quality_idx]
            union = set(vertices).union(set(notes))
            list_notes = [(x.diatonic,x.chromatic) for x in notes.keys()]
            for node in selected_nodes:
                node_pc = (node['pitch_diatonic'], node['pitch_chromatic'])
                if node_pc in list_notes:
                    inversion[diatonic_root, chromatic_root, quality_idx] = list_notes.index(node_pc)
                    break
            root_score[diatonic_root, chromatic_root, quality_idx] /= len(union)

        return root_score, inversion

    def get_n_best(self,n=6):
        """ Returns the n best chords for the node"""
        if n == -1:
            return np.argwhere(self.root_score)
        best_n_idx_flat = np.argpartition(self.root_score.flatten(), -n)[-n:]
        best_n = zip(*np.unravel_index(best_n_idx_flat, self.root_score.shape))
        best_n = sorted(best_n, key=lambda x: -self.root_score[x])
        return [x for x in best_n if self.root_score[x] > 0]

    def selected_nodes(self):
        """ Returns the selected nodes of the tree"""
        def geo_mean(*args):
            return np.prod(args) ** (1/len(args))

        def children_score(node, mean):
            if len(node.children) == 0:
                return node.root_score.max()
            return max(mean(*[child.root_score.max() for child in node.children]),
                       mean(*[children_score(child, mean) for child in node.children]))

        score = self.root_score.max()
        if score >= children_score(self, geo_mean):
            return [self]
        else:
            return reduce(list.__add__, [child.selected_nodes() for child in self.children])

    def select_nodes(self):
        """ Applies the selection to the tree"""
        for child in self.children:
            for node in child.selected_nodes():
                node.selected = True

    def analyze_leap(self):
        """ Creates the root filter of the leap analysis"""
        root_score_leap = np.zeros((7,12,self.qualities.len))
        list_chords = []
        for note_node in self.note_graph_selected_nodes[self.note_graph_selected_nodes['isLeap']]:
            diatonic = int(note_node['pitch_diatonic'])
            chromatic = int(note_node['pitch_chromatic'])
            pitch = Pitch(diatonic, chromatic)
            chords = self.qualities.pitch_beam[pitch]
            list_chords.append(set((root,quality) for root,quality,_ in chords))

        if not list_chords:
            return np.ones((7,12,self.qualities.len))

        chord_intersection = set.intersection(*list_chords)

        for (root, quality) in chord_intersection:
            diatonic, chromatic = root.diatonic, root.chromatic
            quality_idx = self.qualities.name_to_idx[quality]
            root_score_leap[diatonic, chromatic, quality_idx] = 1

        return root_score_leap


    def analyze_onset(self):
        """ Creates the root filter of the onset analysis"""
        onset_roots = []
        for onset, nodes in self.note_graph.vertical_dict.items():
            if nodes[0]['isRest']:
                continue
            pitch_set = {}
            for node in nodes:
                diatonic = int(node['pitch_diatonic'])
                chromatic = int(node['pitch_chromatic'])
                pitch = Pitch(diatonic, chromatic)
                if pitch not in pitch_set:
                    pitch_set[pitch] = node['offset']
                else:
                    pitch_set[pitch] = max(pitch_set[pitch], node['offset'])
            min_offset = min(pitch_set.values())
            possible_roots = find_roots_onset(self.qualities, frozenset(pitch_set))
            if possible_roots:
                onset_roots.append((onset, min_offset, possible_roots))

        for onset, offset, chords in onset_roots:
            self.update_onset_score(onset, offset, chords)

    def update_onset_score(self, chord_onset, chord_offset, chord_root_qualities):
        """ Recursively updates the onset score of the node and its children"""
        for child in self.children:
            if interval_collision(chord_onset, chord_offset, child.onset, child.offset):
                if interval_in(child.onset, child.offset, chord_onset, chord_offset):
                    for child_child in child.depth_first_search():
                        for chord_root, chord_quality in chord_root_qualities:
                            diatonic, chromatic = chord_root.diatonic, chord_root.chromatic
                            quality_idx = self.qualities.name_to_idx[chord_quality]
                            child_child.root_score_onset[diatonic, chromatic, quality_idx] = 1
                        child_child.root_score *= child_child.root_score_onset
                    continue
                child.update_onset_score(chord_onset, chord_offset, chord_root_qualities)

@lru_cache(maxsize=None)
def find_roots_onset(qualities, pitch_set):
    """
    Returns the possible roots of the onset chord given the pitch set
    Output : a list of (root, quality) tuples
    """
    if len(pitch_set) < 3:
        return []
    chord_list = []
    for pitch in pitch_set:
        root, quality, _ = zip(*qualities.pitch_beam[pitch])
        chord_list.append(set(zip(root,quality)))
    chord_set = list(set.intersection(*chord_list))
    return chord_set
