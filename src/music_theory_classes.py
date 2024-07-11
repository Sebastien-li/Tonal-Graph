"""
Contains the classes for the music theory objects
"""

import numpy as np
from multimethod import multimethod

class Pitch:
    """
    Class that represents a pitch in the diatonic / chromatic space
    """
    note_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    diatonic_dict = {note: i for i, note in enumerate(note_names)}
    chromatic_dict = {'C':0, 'D':2, 'E':4, 'F':5, 'G':7, 'A':9, 'B':11}

    @multimethod
    def __init__(self, name:str):
        name = ''.join([x for x in name if not x.isdigit()])
        self.name = name
        self.name_without_accidental = name[0]
        self.accidental = name[1:].replace('b','-')
        assert self.name_without_accidental in self.diatonic_dict, f'Invalid note name: {name}'
        assert all('#' == x for x in self.accidental) or all('-' == x for x in self.accidental)\
            , f'Invalid accidental: {self.accidental}'
        self.diatonic = self.diatonic_dict[self.name_without_accidental]
        if self.accidental:
            accidental_number = len(self.accidental) * (1 if self.accidental[0] == '#' else -1)
        else:
            accidental_number = 0
        self.chromatic = (self.chromatic_dict[self.name_without_accidental] +
                          accidental_number ) %12

    @multimethod
    def __init__(self, diatonic:int, chromatic:int):
        self.diatonic = diatonic%7
        self.chromatic = chromatic%12
        self.name_without_accidental = self.note_names[self.diatonic]
        accidental_number = (chromatic - self.chromatic_dict[self.name_without_accidental])%12
        if accidental_number <= 6:
            self.accidental = '#' * accidental_number
        else:
            self.accidental = '-' * (12-accidental_number)
        self.name = self.name_without_accidental + self.accidental

    def __repr__(self):
        return f'{self.name}'

    def __eq__(self, other):
        return self.chromatic == other.chromatic and self.diatonic == other.diatonic

    def __add__(self, interval):
        diatonic = (self.diatonic + interval.diatonic)%7
        chromatic = (self.chromatic + interval.chromatic)%12
        return Pitch(diatonic, chromatic)

    def __hash__(self):
        return hash((self.diatonic, self.chromatic))

class Interval:
    """
    Class that represents an interval in the diatonic / chromatic space
    """
    @multimethod
    def __init__(self, pitch_start:Pitch, pitch_end:Pitch ):
        self.diatonic = (pitch_end.diatonic - pitch_start.diatonic)%7
        self.chromatic = (pitch_end.chromatic - pitch_start.chromatic)%12
        self.interval_number = self.diatonic + 1
    @multimethod
    def __init__(self, diatonic:int, chromatic:int):
        self.diatonic = diatonic
        self.chromatic = chromatic
        self.interval_number = self.diatonic + 1

    def __repr__(self):
        return f'({self.diatonic}, {self.chromatic})'

    def __eq__(self, other):
        return self.diatonic == other.diatonic and self.chromatic == other.chromatic

    def __hash__(self):
        return hash((self.diatonic, self.chromatic))

class Quality:
    """
    Class that represents a chord quality and the score of each note in the chord
    """
    def __init__(self, label:str='NO', name:str='NO', score_dict:dict={}):
        self.label = label
        self.score_dict = score_dict
        self.name = name
        self.cardinality = len(score_dict)

    def __repr__(self):
        return f'{self.label}'

    def label_with_inversion(self, inversion):
        """ Returns the label of the chord with the inversion"""
        if self.cardinality == 3:
            inversion_name = ['','6','64'][inversion]
            full_name = self.label + inversion_name
        elif self.cardinality  == 4:
            inversion_name = ['7','65','43','2'][inversion]
            full_name = self.label.replace('7',inversion_name)
        else:
            full_name = 'NO'
        return full_name

class Qualities:
    """
    Class that represents a collection of qualities
    """
    def __init__(self, *quality_list):
        self.quality_list = [Quality(*qu) for qu in quality_list]
        self.quality_dict = {quality.label:quality for quality in self.quality_list}
        self.idx_to_name = {i: quality.label for i, quality in enumerate(self.quality_list)}
        self.name_to_idx = {quality.label: i for i, quality in enumerate(self.quality_list)}
        self.pitch_beam = self.__compute_pitch_beat() #pitch -> [(root, quality) of the chord]
        self.chord_array = self.__compute_chord_array() #diatonic, chromatic, quality_idx -> chord
        self.len = len(self.quality_list)

    def __getitem__(self,label):
        if isinstance(label,int):
            return self.quality_list[label]
        if isinstance(label,str):
            return self.quality_dict[label]
        raise ValueError('Invalid label for qualities')

    def __iter__(self):
        return iter(self.quality_dict.items())

    def __repr__(self):
        return f'{list(self.quality_dict.keys())}'

    def __len__(self):
        return len(self.quality_list)

    def __compute_pitch_beat(self):
        pitch_beam = {}
        for quality_label, quality in self:
            for root_diatonic in range(7):
                for root_chromatic in range(12):
                    root = Pitch(root_diatonic, root_chromatic)
                    for interval_note_name, score in quality.score_dict.items():
                        interval = Interval(Pitch(interval_note_name),Pitch('C'))
                        if root not in pitch_beam:
                            pitch_beam[root] = []
                        pitch_beam[root].append((root+interval,quality_label,score))
        return pitch_beam

    def __compute_chord_array(self):
        chord_array = np.zeros((7,12,len(self)), dtype=object)
        for root_diatonic in range(7):
            for root_chromatic in range(12):
                for quality_idx, (_, quality) in enumerate(self):
                    chord = {Pitch(note)+Interval(root_diatonic,root_chromatic): score
                             for note,score in quality.score_dict.items()}
                    chord_array[root_diatonic, root_chromatic, quality_idx] = chord
        return chord_array

class RomanNumeralFigure:
    """" Class that represents a roman numeral """
    def __init__(self, figure:str, diatonic_root:int, chromatic_root:int, quality:str, score:float):
        self.figure = figure
        self.diatonic_root = diatonic_root
        self.chromatic_root = chromatic_root
        self.quality = quality
        self.score = score
        self.label = self.get_label()
    def __repr__(self):
        return f'{self.label}'

    def get_label(self):
        ''' Converts the figure into a label '''
        label = self.figure
        if self.quality in ['m','o', 'm7', 'o7', 'ø7']:
            label = label.lower()
        if self.quality in ['M','m','It','Ger','Fr']:
            quality = ''
        elif self.quality == 'm7':
            quality = '7'
        else:
            quality = self.quality
        return label+quality

class Mode:
    """ Class that represents a mode (major or minor) """
    def __init__(self, name: str, roman_numeral_list: list):
        self.name = name
        self.roman_numeral_list = [RomanNumeralFigure(*rn) for rn in roman_numeral_list]

    def __iter__(self):
        return iter(self.roman_numeral_list)

    def __repr__(self):
        return f'{self.name}'