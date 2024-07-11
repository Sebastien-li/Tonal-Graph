"""Module for the callback when choosing a piece from the dropdown"""
from dash import Output, Input, html
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
        Output('accuracy_content','children'),
        Output('roman_graph_content','figure'),
        Input('dropdown-selection', 'value'),
    )
    def choose_piece(composer_and_title):
        """Callback function for the piece selection dropdown"""
        dir_path, composer_and_title, note_graph, rhythm_tree, tonal_graph, roman_text, m21_roman_text = analysis_dict[composer_and_title].values()
        accuracy_text = []
        image_content = dir_path / 'image.png'
        image_figure = plot_image(image_content)
        html_romantext = to_html_text(roman_text.text)
        if m21_roman_text is None:
            html_m21_roman_text = ''
        else:
            html_m21_roman_text = to_html_text(m21_roman_text.text)
            accuracy, key_accuracy, key_degree_accuracy  = roman_text.compare(m21_roman_text)
            accuracy_text.append(f'Key accuracy: {key_accuracy*100:.2f}%')
            accuracy_text.append(html.Br())
            accuracy_text.append(f'Degree accuracy: {key_degree_accuracy*100:.2f}%')
            accuracy_text.append(html.Br())
            accuracy_text.append(f'Quality accuracy: {accuracy*100:.2f}%')
            accuracy_text.append(html.Br())

        time_figure = plot_time_graph(note_graph, rhythm_tree, tonal_graph)
        roman_figure = plot_roman_graph(note_graph.score, roman_text, m21_roman_text)
        return image_figure, time_figure, invisible_style, html_romantext, html_m21_roman_text, accuracy_text, roman_figure
