""" Main script for the project. Will analyze all the musicXML files in the data folder."""

from pathlib import Path
from time import time

from src.music_theory_objects import qualities, major_mode, minor_mode
from src.note_graph import NoteGraph, Music21Score
from src.rhythm_tree import RhythmTree, RhythmTreeAnalyzed
from src.tonal_graph import TonalGraph
from src.roman_text import RomanText

data_path = Path('data')
mxl_list = list(data_path.glob('*/*/score.mxl'))[1:2]
overall_accuracy = 0
len_mxl_list_with_accuracy = 0
for file in mxl_list:
    composer, title = file.parts[-3:-1]
    print(f'Analyzing {composer} - {title} ...')
    t0 = time()

    # Load the music21 score
    score = Music21Score(file, composer, title)
    #print(f'File loaded with music21 in {time()-t0:.2f} seconds')
    t1 = time()

    # Create the note graph
    note_graph = NoteGraph.from_m21score(score)
    t2 = time()
    #print(f'Note graph created in {t2-t1:.2f} seconds')

    # Create the rhythm tree
    rhythm_tree = RhythmTree.construct_tree(note_graph)
    t3 = time()
    #print(f'Rhythm tree created in {t3-t2:.2f} seconds')

    # Analyze the rhythm tree
    rhythm_tree_analyzed = RhythmTreeAnalyzed(rhythm_tree, qualities)
    t4 = time()
    #print(f'Rhythm tree analyzed in {t4-t3:.2f} seconds')

    # Create the tonal graph
    tonal_graph = TonalGraph(rhythm_tree_analyzed, [major_mode, minor_mode])
    t5 = time()
    #print(f'Tonal graph created in {t5-t4:.2f} seconds')

    # Create the roman text
    roman_text = RomanText.from_tonal_graph(tonal_graph)
    roman_text.save(file.parent / 'analysis_generated.txt')
    if (file.parent / 'analysis.txt').exists():
        accuracy = roman_text.compare(file.parent / 'analysis.txt')
        overall_accuracy += accuracy
        len_mxl_list_with_accuracy += 1
        print(f'Accuracy: {100*accuracy:.2f}%')
    else:
        print('Ground truth roman text not provided, skipping accuracy calculation')
    t6 = time()
    #print(f'Roman text created and saved in {t6-t5:.2f} seconds')

    print(f'Analysis of {composer} - {title} completed in {t6-t0:.2f} seconds')
    print()

if len_mxl_list_with_accuracy > 0:
    print(f'Overall accuracy: {100*overall_accuracy/len_mxl_list_with_accuracy:.2f}%')

