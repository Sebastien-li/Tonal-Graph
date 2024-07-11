from dash import dcc, html, dash_table
from src.music_theory_objects import qualities

invisible_style = {'display':'none'}
visible_style = {'display':'block'}

def get_layout(analysis_dict):
    """ Returns the layout of the app."""
    layout = [
        html.H1('Harmony Analysis Tool'),
        html.H2('Select an example:'),
        dcc.Dropdown(list(analysis_dict.keys()),
                     list(analysis_dict.keys())[0], id='dropdown-selection'),

        html.H2('Click on an element of the Rhythm Tree, the Segmentation Graph or the Root Cloud to see more details!'),
        html.Div([
            dcc.Graph(id='image-content',figure={}),
            html.H4(id='accuracy_content'),
            dcc.Graph(id='roman_graph_content',figure={}),
            html.Div(id='romantext_container', children = [
                html.Div(id='romantext_generated',
                         style = {'display':'inline-block', 'width':'50%'}),
                html.Div(id='romantext_m21',
                         style = {'display':'inline-block', 'width':'50%'}),
            ]),
            dcc.Graph(id='time_graph_content',figure={}),
        ], id = 'score_information_container', style = {'width':'100%'}),

        html.Div([
            html.Div('Marker size multiplier:'),
            dcc.Slider(id='marker_size_slider', min=-2,max=3,step=0.01,
                    marks={i: f'{2 ** i}' for i in range(-2,3)},
                    value = 0,
                    updatemode='mouseup'),
            html.Div('Collapse root cloud by:'),
            dcc.RadioItems(id = 'collapse_by_radio', options=['Quality','Diatonic','Chromatic'],
                           value = 'Diatonic', inline=True),
            dcc.Graph(id='root_cloud_content',figure={}),
        ], id = 'root_cloud_container', style = invisible_style),
    ]

    return layout
