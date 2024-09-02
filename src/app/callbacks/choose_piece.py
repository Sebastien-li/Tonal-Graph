"""Module for the callback when choosing a piece from the dropdown"""
from dash import Output, Input, html
import pandas as pd
from ..plotter import plot_time_graph, plot_image, plot_roman_graph
from ..layout import invisible_style
from ...utils import to_html_text


def choose_piece_callback(app, analysis_dict):
    """ Function for the callback that choosing a piece from the dropdown"""

    @app.callback(
        Output('image-content', 'figure'),
        Output('time_graph_content', 'figure'),
        Output('root_cloud_container','style'),
        Output('romantext_generated','children'),
        Output('romantext_m21','children'),
        Output('romantext_augnet','children'),
        Output('accuracy_content','children'),
        Output('roman_graph_content','figure'),
        Input('dropdown-selection', 'value'),
    )
    def choose_piece(composer_and_title):
        """Callback function for the piece selection dropdown"""
        dir_path, composer_and_title, note_graph, rhythm_tree, tonal_graph, roman_text, m21_roman_text, augnet_roman_text = analysis_dict[composer_and_title].values()
        accuracy_text = []
        image_content = dir_path / 'image.png'
        image_figure = plot_image(image_content)
        html_romantext = to_html_text(roman_text.text)
        html_m21_roman_text = to_html_text(m21_roman_text.text)
        df = pd.read_csv(dir_path / 'score_annotated.csv')
        df = df[['LocalKey38','RomanNumeral31','TonicizedKey38','offset','measure']]
        with open(dir_path / 'score_annotated.rntxt', 'r') as f:
            html_augnet_roman_text = to_html_text(f.read())

        accuracy, key_accuracy, key_degree_accuracy  = roman_text.compare(m21_roman_text)
        aug_accuracy, aug_key_accuracy, aug_degree_accuracy = augnet_roman_text.compare(m21_roman_text)
        accuracy_text.append(f'Key accuracy: Tonal Graph: {key_accuracy*100:.2f}%  AugmentedNet: {aug_key_accuracy*100:.2f}%')
        accuracy_text.append(html.Br())
        accuracy_text.append(f'Degree accuracy: Tonal Graph: {key_degree_accuracy*100:.2f}%  AugmentedNet: {aug_degree_accuracy*100:.2f}%')
        accuracy_text.append(html.Br())
        accuracy_text.append(f'Quality accuracy: Tonal Graph: {accuracy*100:.2f}%  AugmentedNet: {aug_accuracy*100:.2f}%')
        accuracy_text.append(html.Br())

        time_figure = plot_time_graph(note_graph, rhythm_tree, tonal_graph)
        roman_figure = plot_roman_graph(note_graph.score, roman_text, m21_roman_text, augnet_roman_text)
        return image_figure, time_figure, invisible_style, html_romantext, html_m21_roman_text, html_augnet_roman_text, accuracy_text, roman_figure
