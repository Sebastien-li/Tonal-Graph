""" Main script for the project. Will analyze all the musicXML files in the data folder."""

from pathlib import Path
from time import time

from src.music_theory_objects import qualities, major_mode, minor_mode, transitions
from src.note_graph import NoteGraph, Music21Score
from src.rhythm_tree import RhythmTree, RhythmTreeAnalyzed
from src.tonal_graph import TonalGraph
from src.roman_text import RomanText
from src.utils import get_multilogger

def main():
    """ Analyze all the musicXML files in the data folder."""
    logger = get_multilogger()
    mxl_list = list(Path('data').glob('*/*/score.mxl'))
    mxl_list = mxl_list
    overall_accuracy = 0
    overall_degree_accuracy = 0
    overall_key_accuracy = 0
    overall_len = 0
    for file in mxl_list:
        composer, title = file.parts[-3:-1]
        logger.info('Analyzing %s - %s ...', composer, title)

        # Load the music21 score
        t0 = time()
        score = Music21Score(file, composer, title)
        logger.debug('File loaded with music21 in %.2f seconds', time()-t0)

        # Create the note graph
        t = time()
        note_graph = NoteGraph.from_m21score(score)
        logger.debug('Note graph created in %.2f seconds', time()-t)

        # Create the rhythm tree
        t = time()
        rhythm_tree = RhythmTree.construct_tree(note_graph)
        logger.debug('Rhythm tree created in %.2f seconds', time()-t)

        # Analyze the rhythm tree
        t = time()
        rhythm_tree_analyzed = RhythmTreeAnalyzed(rhythm_tree, qualities)
        logger.debug('Rhythm tree analyzed in %.2f seconds', time()-t)

        # Create the tonal graph
        t = time()
        tonal_graph = TonalGraph(rhythm_tree_analyzed,
                                [major_mode, minor_mode],
                                transitions=transitions)
        logger.debug('Tonal graph created in %.2f seconds', time()-t)

        # Create the roman text
        t = time()
        roman_text = RomanText.from_tonal_graph(tonal_graph)
        roman_text.save(file.parent / 'analysis_generated.txt')
        if (file.parent / 'analysis.txt').exists():
            m21_roman_text = RomanText.from_rntxt(file.parent/'analysis.txt')
            accuracy, key_accuracy, key_degree_accuracy = roman_text.compare(m21_roman_text)
            overall_accuracy += accuracy
            overall_degree_accuracy += key_degree_accuracy
            overall_key_accuracy += key_accuracy
            overall_len += 1
            logger.info('Key accuracy: %.2f%%', 100*key_accuracy)
            logger.info('Degree accuracy: %.2f%%', 100*key_degree_accuracy)
            logger.info('Quality accuracy: %.2f%%', 100*accuracy)
        else:
            logger.info('Ground truth roman text not provided, skipping accuracy calculation')
        logger.debug('Roman Numeral created and saved in %.2f seconds', time()-t)

        logger.info('Completed in %.2f seconds', time()-t0)

    if overall_len > 0:
        logger.info('Key accuracy: %.2f%%', 100*overall_key_accuracy/overall_len)
        logger.info('Degree accuracy: %.2f%%', 100*overall_degree_accuracy/overall_len)
        logger.info('Overall quality accuracy: %.2f%%', 100*overall_accuracy/overall_len)

if __name__ == '__main__':
    main()
