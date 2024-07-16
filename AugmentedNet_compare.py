from pathlib import Path
from src.roman_text import RomanText

augnet_overall_quality_accuracy = 0
augnet_overall_key_accuracy = 0
augnet_overall_degree_accuracy = 0
tograph_overall_quality_accuracy = 0
tograph_overall_key_accuracy = 0
tograph_overall_degree_accuracy = 0
globber = list(Path(r"C:\Users\lxqse\Documents\Seb\ATIAM\Stage\AugmentedNet\data").glob('**/score.mxl'))
for path in globber:
    directory = path.parent
    augnet_rntxt = RomanText.from_csv(directory / 'score_annotated.csv')
    gt_rntxt = RomanText.from_rntxt(directory / 'analysis.txt')
    tograph_rntxt = RomanText.from_rntxt(directory / 'analysis_generated.txt')
    augnet_accuracy, augnet_key_accuracy, augnet_degree_accuracy = augnet_rntxt.compare(gt_rntxt)
    tograph_accuracy, tograph_key_accuracy, tograph_degree_accuracy = tograph_rntxt.compare(gt_rntxt)
    augnet_overall_quality_accuracy += augnet_accuracy
    augnet_overall_key_accuracy += augnet_key_accuracy
    augnet_overall_degree_accuracy += augnet_degree_accuracy
    tograph_overall_quality_accuracy += tograph_accuracy
    tograph_overall_key_accuracy += tograph_key_accuracy
    tograph_overall_degree_accuracy += tograph_degree_accuracy
    print(f'Accuracy of {directory.parent.name} - {directory.name}:')
    print(f'Key accuracy: \t\tAugmentedNet: {augnet_key_accuracy*100:.2f}%\tTonalGraph: {tograph_key_accuracy*100:.2f}%')
    print(f'Degree accuracy: \tAugmentedNet: {augnet_degree_accuracy*100:.2f}%\tTonalGraph: {tograph_degree_accuracy*100:.2f}%')
    print(f'Quality accuracy: \tAugmentedNet: {augnet_accuracy*100:.2f}%\tTonalGraph: {tograph_accuracy*100:.2f}%')
    print()
augnet_overall_degree_accuracy /= len(globber)
augnet_overall_key_accuracy /= len(globber)
augnet_overall_quality_accuracy /= len(globber)
tograph_overall_degree_accuracy /= len(globber)
tograph_overall_key_accuracy /= len(globber)
tograph_overall_quality_accuracy /= len(globber)
print('\nOverall results:')
print(f'Overall key accuracy: \tAugmentedNet: '
      f'{augnet_overall_key_accuracy*100:.2f}%\t'
      f'TonalGraph: {tograph_overall_key_accuracy*100:.2f}%')

print(f'Overall degree accuracy: \tAugmentedNet: '
      f'{augnet_overall_degree_accuracy*100:.2f}%\t'
      f'TonalGraph: {tograph_overall_degree_accuracy*100:.2f}%')
print(f'Overall quality accuracy: \tAugmentedNet: '
      f'{augnet_overall_quality_accuracy*100:.2f}%\t'
      f'TonalGraph: {tograph_overall_quality_accuracy*100:.2f}%')
