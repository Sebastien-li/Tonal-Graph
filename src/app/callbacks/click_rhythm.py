""" Module for the callback when clicking on a rhythm"""
from dash import Output, Input, State
from ..plotter import plot_time_graph, plot_root_cloud
from ..layout import invisible_style, visible_style


def click_rhythm_callback(app, analysis_dict):
    """ Function for the callback when clicking on a rhythm"""

    @app.callback(
        Output('time_graph_content', 'figure', allow_duplicate=True),
        Output('root_cloud_content', 'figure'),
        Output('root_cloud_container','style', allow_duplicate=True),
        Input('time_graph_content','clickData'),
        Input('collapse_by_radio', 'value'),
        Input('marker_size_slider', 'value'),
        State('dropdown-selection', 'value'),
        State('time_graph_content', 'figure'),
        prevent_initial_call=True
    )
    def select_rhythm(click_data, collapse_by, marker_size_mult, composer_and_title, figure):
        xmin, xmax = figure['layout']['xaxis']['range']
        if click_data is None:
            return figure, {}, '', invisible_style
        _, composer_and_title, note_graph, rhythm_tree, tonal_graph, _, _, _ = analysis_dict[composer_and_title].values()
        selected_node_idx = -1 if 'customdata' not in click_data['points'][0] \
            else click_data['points'][0]['customdata']
        #berk
        if isinstance(selected_node_idx, list):
            node = list(rhythm_tree.depth_first_search())[selected_node_idx[0]]
        else:
            node = list(rhythm_tree.depth_first_search())[selected_node_idx]
        time_figure = plot_time_graph(note_graph, rhythm_tree, tonal_graph,
                                      selected_idx=node.note_graph_selected_nodes['id'],
                                      xmin=xmin, xmax=xmax)
        root_cloud_figure = plot_root_cloud(node, collapse_by=collapse_by,
                                            marker_size=40*2**marker_size_mult)

        return time_figure, root_cloud_figure, visible_style
