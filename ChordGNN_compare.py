from pathlib import Path
from itertools import chain
import pandas as pd
import json
from src.roman_text import RomanText

augnet_overall_quality_accuracy = 0
augnet_overall_key_accuracy = 0
augnet_overall_degree_accuracy = 0
tograph_overall_quality_accuracy = 0
tograph_overall_key_accuracy = 0
tograph_overall_degree_accuracy = 0
cgnn_overall_quality_accuracy = 0
cgnn_overall_key_accuracy = 0
cgnn_overall_degree_accuracy = 0


dataset_path = Path(r"dataset")
globber = list(chain(*[list(dataset_path.glob(f'**/score.{ext}')) for ext in ['mxl', 'xml']]))

df = {'composer': [], 'piece': [],
      'augnet_key_accuracy': [], 'augnet_degree_accuracy': [], 'augnet_quality_accuracy': [],
      'tonal_graph_key_accuracy': [], 'tonal_graph_degree_accuracy': [], 'tonal_graph_quality_accuracy': [],
      'cgnn_key_accuracy': [], 'cgnn_degree_accuracy': [], 'cgnn_quality_accuracy': []}

for path in globber:
    directory = path.parent
    composer = directory.parent.name
    piece = directory.name
    print(f'Accuracy of {composer} - {piece}:')
    with open(directory / 'score-analysis-chordgnn-divs-pq.json', 'r') as f:
        divs_pq = json.load(f)
    chord_gnn_rntxt = RomanText.from_chord_gnn(directory / 'score-analysis-chordgnn.json',
                                                     divs_pq = divs_pq)
    augnet_rntxt = RomanText.from_csv(directory / 'score_annotated.csv')
    gt_rntxt = RomanText.from_rntxt(directory / 'analysis.txt')
    tograph_rntxt = RomanText.from_pickle(directory / 'analysis_TonalGraph.pkl')

    cgnn_accuracy, cgnn_key_accuracy, cgnn_degree_accuracy = chord_gnn_rntxt.compare(gt_rntxt)
    augnet_accuracy, augnet_key_accuracy, augnet_degree_accuracy = augnet_rntxt.compare(gt_rntxt)
    tograph_accuracy, tograph_key_accuracy, tograph_degree_accuracy = tograph_rntxt.compare(gt_rntxt)

    augnet_overall_quality_accuracy += augnet_accuracy
    augnet_overall_key_accuracy += augnet_key_accuracy
    augnet_overall_degree_accuracy += augnet_degree_accuracy
    tograph_overall_quality_accuracy += tograph_accuracy
    tograph_overall_key_accuracy += tograph_key_accuracy
    tograph_overall_degree_accuracy += tograph_degree_accuracy
    cgnn_overall_key_accuracy += cgnn_key_accuracy
    cgnn_overall_degree_accuracy += cgnn_degree_accuracy
    cgnn_overall_quality_accuracy += cgnn_accuracy


    df['composer'].append(composer)
    df['piece'].append(piece)
    df['augnet_key_accuracy'].append(augnet_key_accuracy)
    df['augnet_degree_accuracy'].append(augnet_degree_accuracy)
    df['augnet_quality_accuracy'].append(augnet_accuracy)
    df['tonal_graph_key_accuracy'].append(tograph_key_accuracy)
    df['tonal_graph_degree_accuracy'].append(tograph_degree_accuracy)
    df['tonal_graph_quality_accuracy'].append(tograph_accuracy)
    df['cgnn_key_accuracy'].append(cgnn_key_accuracy)
    df['cgnn_degree_accuracy'].append(cgnn_degree_accuracy)
    df['cgnn_quality_accuracy'].append(cgnn_accuracy)

    print(f'Key accuracy: \t\tAugmentedNet: {augnet_key_accuracy*100:.2f}%\tChordGNN: {cgnn_key_accuracy*100:.2f}%\tTonalGraph: {tograph_key_accuracy*100:.2f}%')
    print(f'Degree accuracy: \tAugmentedNet: {augnet_degree_accuracy*100:.2f}%\tChordGNN: {cgnn_degree_accuracy*100:.2f}%\tTonalGraph: {tograph_degree_accuracy*100:.2f}%')
    print(f'Quality accuracy: \tAugmentedNet: {augnet_accuracy*100:.2f}%\tChordGNN: {cgnn_accuracy*100:.2f}%\tTonalGraph: {tograph_accuracy*100:.2f}%')
    print()

augnet_overall_degree_accuracy /= len(globber)
augnet_overall_key_accuracy /= len(globber)
augnet_overall_quality_accuracy /= len(globber)
tograph_overall_degree_accuracy /= len(globber)
tograph_overall_key_accuracy /= len(globber)
tograph_overall_quality_accuracy /= len(globber)
cgnn_overall_degree_accuracy /= len(globber)
cgnn_overall_key_accuracy /= len(globber)
cgnn_overall_quality_accuracy /= len(globber)

print('\nOverall results:')
print(f'Overall key accuracy: \t\tAugmentedNet: '
      f'{augnet_overall_key_accuracy*100:.2f}%\t'
      f'ChordGNN: {cgnn_overall_key_accuracy*100:.2f}%\t'
      f'TonalGraph: {tograph_overall_key_accuracy*100:.2f}%\t')



print(f'Overall degree accuracy: \tAugmentedNet: '
      f'{augnet_overall_degree_accuracy*100:.2f}%\t'
      f'ChordGNN: {cgnn_overall_degree_accuracy*100:.2f}%\t'
      f'TonalGraph: {tograph_overall_degree_accuracy*100:.2f}%')

print(f'Overall quality accuracy: \tAugmentedNet: '
      f'{augnet_overall_quality_accuracy*100:.2f}%\t'
      f'ChordGNN: {cgnn_overall_quality_accuracy*100:.2f}%\t'
      f'TonalGraph: {tograph_overall_quality_accuracy*100:.2f}%')


df = pd.DataFrame(df)
df.to_csv('algorithm_comparison.csv', index=False)
