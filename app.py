""" Dash app for harmony analysis tool. """
from pathlib import Path

from dash import Dash
from tqdm import tqdm

from src.music_theory_objects import qualities, major_mode, minor_mode, transitions
from src.note_graph import NoteGraph, Music21Score
from src.rhythm_tree import RhythmTree, RhythmTreeAnalyzed
from src.tonal_graph import TonalGraph
from src.roman_text import RomanText
from src.app.layout import get_layout
from src.app.callbacks.choose_piece import choose_piece_callback
from src.app.callbacks.click_rhythm import click_rhythm_callback

def analyze(dir_path):
    """ Analyze a musicXML file."""
    composer, title = dir_path.parts[-2:]
    mxl_path = dir_path / 'score.mxl'
    score = Music21Score(mxl_path, composer, title)
    note_graph = NoteGraph.from_m21score(score)
    rhythm_tree = RhythmTree.construct_tree(note_graph)
    rhythm_tree_analyzed = RhythmTreeAnalyzed(rhythm_tree, qualities)
    tonal_graph = TonalGraph(rhythm_tree_analyzed,
                            [major_mode, minor_mode],
                            transitions=transitions)
    roman_text = RomanText.from_tonal_graph(tonal_graph)
    m21_roman_text = None
    augnet_roman_text = None
    if (dir_path/'analysis.txt').exists():
        m21_roman_text = RomanText.from_rntxt(dir_path/'analysis.txt')
    if (dir_path/'score_annotated.csv').exists():
        augnet_roman_text = RomanText.from_csv(dir_path/'score_annotated.csv')
    return note_graph, rhythm_tree_analyzed, tonal_graph, roman_text, m21_roman_text, augnet_roman_text

def main():
    """ Main function for the app."""
    dir_path_list = list(Path('assets').glob('scores/*/*'))
    analysis_dict = {}
    for dir_path in tqdm(dir_path_list, desc=f'Analyzing {len(dir_path_list)} examples'):
        composer, title = dir_path.parts[-2:]
        composer_and_title = f'{composer} : {title}'
        note_graph, rhythm_tree, tonal_graph, roman_text, m21_roman_text, augnet_roman_text = analyze(dir_path)
        analysis_dict[composer_and_title] = {
            'dir_path': dir_path,
            'descr' : composer_and_title,
            'note_graph': note_graph,
            'rhythm_tree': rhythm_tree,
            'tonal_graph': tonal_graph,
            'roman_text': roman_text,
            'm21_roman_text': m21_roman_text,
            'augnet_roman_text': augnet_roman_text}

    app = Dash(__name__)
    app.title = 'Tonal Graph Analyzer'
    app.layout = get_layout(analysis_dict)
    choose_piece_callback(app, analysis_dict)
    click_rhythm_callback(app, analysis_dict)
    app.run(debug=True, port=8050)

if __name__ == '__main__':
    main()
