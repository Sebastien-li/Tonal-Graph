""" Contains the functions to create the plotly figures for the dash app """
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from src.music_theory_classes import Pitch
from src.roman_text import RomanNumeral

color_palette = {'red': 'orangered', 'orange' : 'darkorange',
                 'yellow' : 'gold',
                 'blue': 'royalblue', 'light_blue':'skyblue',
                 'green': 'forestgreen', 'light_green': 'limegreen',
                 'white': 'white', 'black': 'black', 'transparent_white' : 'rgba(255,255,255,0.2)',
                  'gray': 'gray' }

color_dict = {'onset': color_palette['red'],
              'during': color_palette['blue'],
              'follow': color_palette['green'],
              'silence': color_palette['light_green']}

def plot_image(url):
    """ Create a plotly figure with an image as background"""
    url = str(url)
    fig = go.Figure()
    image_width = 1900
    image_height = 200
    fig.add_layout_image(x=0, sizex=image_width, y=0, sizey=image_height, xref="x", yref="y",
                         layer="below",
                         source = url)
    fig.update_xaxes(showgrid=False, visible=False, range=[0,image_width])
    fig.update_yaxes(showgrid=False, visible=False, range=[image_height,0])
    fig.update_layout(height=image_height, width=image_width, margin={'l':0, 'r':0, 'b':0, 't':0},
                      plot_bgcolor='white')
    return fig


def make_note_graph_trace(note_graph, selected_idx=None):
    """ Create the traces for the note graph"""

    # Note graph
    if selected_idx is None:
        selected_idx = note_graph.nodes['id']

    edge_traces = []
    edge_show_legend = {'onset':True, 'during':True, 'follow':True, 'silence':True}
    dur_div = note_graph.duration_divisor
    for i,edge in enumerate(note_graph.edge_index):
        u = note_graph.nodes[edge[0]]
        v = note_graph.nodes[edge[1]]
        selected = edge[0] in selected_idx and edge[1] in selected_idx
        edge_attr = note_graph.edge_attr[i]['type']
        edge_traces.append(go.Scatter(
            x = [u['onset']/dur_div, v['onset']/dur_div, None],
            y = [u['pitch_space'], v['pitch_space'], None],
            line={'color':color_dict[edge_attr]},
            hoverinfo='none',
            mode='lines',
            opacity = 1 if selected else 0.2,
            showlegend=edge_show_legend[edge_attr],
            name = edge_attr,
            legendgroup='note_graph',
        ))
        edge_show_legend[edge_attr] = False


    leap_nodes = note_graph.nodes[note_graph.nodes['isLeap']]
    not_leap_nodes = note_graph.nodes[~note_graph.nodes['isLeap']]

    leap_nodes_trace = go.Scatter(
        x = leap_nodes['onset']/dur_div,
        y = leap_nodes['pitch_space'],
        hovertext = [f'{name}\tOnset: {onset/dur_div:.2f}'
                     for name,onset in zip (leap_nodes['pitch_name'], leap_nodes['onset'])],
        mode='markers',
        hoverinfo='text',
        name='Leap Notes',
        marker={
            'color':color_palette['light_blue'],
            'size':10,
            'line_width':2,
            'opacity' : [1 if (node['id'] in selected_idx) else 0.2 for node in leap_nodes],},
        showlegend=True,
        legendgroup='note_graph',
    )

    not_leap_nodes_trace = go.Scatter(
        x = not_leap_nodes['onset']/dur_div,
        y = not_leap_nodes['pitch_space'],
        hovertext = [f'{name}\tOnset: {onset/dur_div:.2f}'
                     for name,onset in zip (not_leap_nodes['pitch_name'], not_leap_nodes['onset'])],
        mode='markers',
        hoverinfo='text',
        name='Notes',
        marker=dict(
            color=color_palette['white'],
            size=10,
            line_width=2,
            opacity = [1 if (node['id'] in selected_idx) else 0.2 for node in not_leap_nodes]),
        showlegend=True,
        legendgroup='note_graph',
    )

    return edge_traces, leap_nodes_trace, not_leap_nodes_trace

def make_rhythm_tree_trace(rhythm_tree):
    """ Create the traces for the rhythm tree"""
    text_x = []
    text_y = []
    text = []
    text_annotations = []
    rectangle_fill_traces_best = []
    rectangle_fill_traces = []
    rectangle_border_traces = []

    selected_show_legend = True
    unselected_show_legend = True
    dur_div = rhythm_tree.duration_divisor
    for i,node in enumerate(rhythm_tree.depth_first_search()):
        x0 = node.onset/dur_div
        x1 = node.offset/dur_div - 0.1
        y0 = np.log2(float(node.subdivision)) if node.depth != 0 else 3
        y1 = np.log2(float(node.subdivision))+0.5 if node.depth != 0 else 3.5

        idx = node.root_score.argmax()
        root,pitch_class,quality_index = np.unravel_index(idx, node.root_score.shape)
        pitch = Pitch(int(root), int(pitch_class))
        quality = rhythm_tree.qualities[int(quality_index)]
        inversion = node.inversion[root,pitch_class,quality_index]
        quality_label =quality.label_with_inversion(inversion)
        opacity = min(1,node.root_score[root,pitch_class,quality_index])
        if opacity == 0:
            pitch, quality_label = '', ''

        # Rectangle fill
        if node.selected:
            rectangle_fill_traces_best.append(go.Scatter(
                x = [x0,x1,x1,x0,x0,None],
                y = [y0,y0,y1,y1,y0,None],
                fill='toself',
                mode='lines',
                line={'color':color_palette['orange']},
                hoverinfo='text' ,
                text = f'{node.root_score[root,pitch_class,quality_index]:.3f}',
                fillcolor=color_palette['orange'],
                customdata=[i],
                opacity=opacity if node.depth != 0 else 1,
                showlegend=selected_show_legend,
                name = "Best analysis",
                legendgroup='rhythm_tree',
                ))
            selected_show_legend = False
        else:
            rectangle_fill_traces.append(go.Scatter(
                x = [x0,x1,x1,x0,x0,None],
                y = [y0,y0,y1,y1,y0,None],
                fill='toself',
                mode='lines',
                line={'color':color_palette['light_blue']},
                hoverinfo='text',
                text = f'{node.root_score[root,pitch_class,quality_index]:.3f}',
                fillcolor=color_palette['light_blue'],
                customdata=[i],
                opacity=opacity if node.depth != 0 else 1,
                showlegend=unselected_show_legend,
                name = "Analysis",
                legendgroup='rhythm_tree',
                ))
            unselected_show_legend = False
        # Rectangle border
        root_score_onset = node.root_score_onset[root,pitch_class,quality_index]
        rectangle_border_traces.append(go.Scatter(
            x = [x0,x1,x1,x0,x0,None],
            y = [y0,y0,y1,y1,y0,None],
            mode='lines',
            line={'color':color_palette['gray'] if root_score_onset < 1 \
                  else color_palette['green']},
            hoverinfo='skip' ,
            name = f'{node.root_score[root,pitch_class,quality_index]:.3f}',
            showlegend=False,
            ))

        text_x.append((x0+x1)/2)
        text_y.append((y0+y1)/2)
        text.append(f"{pitch}{quality_label}" if node.depth != 0 else "Entire score")
        text_annotations.append(f"{node.subdivision}")

    text_trace = go.Scatter(
        x=text_x,
        y=text_y,
        text=text,
        mode='text',
        textposition='middle center',
        hoverinfo='skip',
        textfont={
            'size':14,
            'color':'black'
        },
        showlegend=False
        )

    return rectangle_fill_traces, rectangle_fill_traces_best, rectangle_border_traces, text_trace

def make_tonal_graph_trace(tonal_graph):
    """ Create the traces for the tonal graph"""
    row_height = np.zeros(12)
    for onset in tonal_graph.onsets:
        nodes = tonal_graph.nodes[tonal_graph.nodes['onset'] == onset]
        chromatics, counts = np.unique(nodes['tonic_chromatic'], return_counts=True)
        for chromatic,count in zip(chromatics,counts):
            row_height[chromatic] = max(row_height[chromatic],count)
    row_height += 1
    row_height = np.cumsum(np.concatenate(([0],row_height)))

    node_x = []
    node_y = []
    rn_name = []
    annotations = []
    color = []
    rt_id = []
    for onset in tonal_graph.onsets:
        current_y = np.copy(row_height)
        nodes = tonal_graph.nodes[tonal_graph.nodes['onset'] == onset]
        for node in nodes:
            roman_numeral = RomanNumeral.from_tonal_graph_node(tonal_graph, node['id'])
            x = node['onset']/tonal_graph.duration_divisor
            y = current_y[node['tonic_chromatic']]
            current_y[node['tonic_chromatic']] += 1
            node_x.append(x)
            node_y.append(y)
            rn_name.append(f"{roman_numeral.full_name_with_key}")
            annotations.append(f"{node['weight']:.3f}")
            color.append(color_palette['red'] if node['selected'] else color_palette['black'])
            rt_id.append(node['rt_id'])

    node_trace = go.Scatter(
        x = node_x,
        y = node_y,
        mode='markers',
        text=annotations,
        hoverinfo='text',
        marker = {
            'color' : color,
            'size' : 10,
            'line_width' : 2
        },
        opacity=0,
        showlegend=False,
        customdata=rt_id
        )

    chord_text_trace = go.Scatter(
        x = node_x,
        y = node_y,
        mode='text',
        text=rn_name,
        textfont = {'color':color, 'size':14},
        hoverinfo='skip',
        showlegend=False,
    )

    return node_trace, chord_text_trace, row_height

def plot_time_graph(note_graph, rhythm_tree, tonal_graph, selected_idx=None,xmin=None,xmax=None):
    """ Create a plotly figure with the temporal graphs of the analysis"""
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=('Note Graph', 'Tonal Graph', 'Rhythm Tree'),
                        specs=[[{}, {"rowspan":2}],
                               [{}, None]],)
    fig.update_layout(
        showlegend=True, dragmode = 'pan',
        height=900, width=1900,
        title_text="Temporal graphs",
        legend_tracegroupgap=110
    )

    #X Axis layout
    score = note_graph.score
    if xmin is None:
        xmin = float(score.measure_list[0].onset)-0.2
        last_measure = score.measure_list[min(1,len(score.measure_list)-1)]
        xmax = float(last_measure.onset + last_measure.duration) + 0.2
    fig.update_xaxes(range = [xmin,xmax],
                     tickvals = [float(x.onset) for x in score.measure_list],
                     ticktext = [x.number for x in score.measure_list], matches='x')
    fig.update_layout(xaxis_showticklabels=False,
                      xaxis2_showticklabels=True,xaxis2_title = 'measure number',
                      xaxis3_showticklabels=True)

    #Note graph
    edge_trace, leap_trace, not_leap_trace = make_note_graph_trace(note_graph, selected_idx)
    for trace in edge_trace:
        fig.add_trace(trace, row=1, col=1)
    fig.add_trace(leap_trace, row=1, col=1)
    fig.add_trace(not_leap_trace, row=1, col=1)

    # Rhythm tree
    rhythm_tree_fill_traces, rhythm_tree_fill_traces_best, rhythm_tree_border_traces, rhythm_tree_text_trace = make_rhythm_tree_trace(rhythm_tree)
    for trace in rhythm_tree_fill_traces:
        fig.add_trace(trace, row=2, col=1)
    for trace in rhythm_tree_fill_traces_best:
        fig.add_trace(trace, row=2, col=1)
    for trace in rhythm_tree_border_traces:
        fig.add_trace(trace, row=2, col=1)
    fig.add_trace(rhythm_tree_text_trace, row=2, col=1)

    # Tonal graph
    node_trace, chord_text_trace, row_height = make_tonal_graph_trace(tonal_graph)
    fig.add_trace(node_trace, row=1, col=2)
    fig.add_trace(chord_text_trace, row=1, col=2)


    # Y axis layout
    fig.update_yaxes(showgrid = False, row=1, col=1, title='pitch space')
    fig.update_yaxes(showgrid = True,
                     zeroline = False,
                     tickvals = [-0.75,0.25,1.25,2.25],
                     ticktext = ['1/2','1','2','4'],
                     row=2, col=1,
                     title = 'subdivision')
    fig.update_yaxes(zeroline = False,
                tickvals = row_height-0.5,
                ticktext = ['C','C#', 'D','E-', 'E', 'F', 'F#', 'G', 'A-', 'A','B-', 'B', ''],
                row=1, col=2)

    return fig

def plot_roman_graph(score, roman_text, m21_roman_text=None, augnet_roman_text=None):
    """ Create a plotly figure with the roman text """
    fig = go.Figure()
    fig.update_layout(
        dragmode = 'pan',
        title_text="Analysis graph")
    # xmin = float(score.measure_list[0].onset)-0.2
    # last_measure = score.measure_list[min(3,len(score.measure_list)-1)]
    # xmax = float(last_measure.onset + last_measure.duration) + 0.2
    fig.update_xaxes(tickvals = [float(x.onset) for x in score.measure_list],
                     ticktext = [x.number for x in score.measure_list])
    fig.update_yaxes(range = [-1,5],
                     tickvals = [0,1,2,3,4],
                     ticktext = ['Tonal Graph analysis',
                                 'Tonal Graph comparison',
                                 'Ground truth analysis',
                                 'AugmentedNet comparison',
                                 'AugmentedNet analysis'],
                     showgrid = False, zeroline = False,)
    text_x = []
    text_y = []
    text = []
    for rn in roman_text.rn_list:
        x0 = float(rn.onset)
        x1 = float(rn.onset + rn.duration)-0.2
        y0 = -0.5
        y1 = 0.5
        rn_text = '<br>'.join(rn.full_name_with_key.split(': '))
        text.append(rn_text)
        text_x.append((x0+x1)/2)
        text_y.append(0)
        fig.add_trace(go.Scatter(
            x = [x0,x1,x1,x0,x0,None],
            y = [y0,y0,y1,y1,y0,None],
            fill='toself',
            fillcolor=color_palette['orange'],
            mode='lines',
            hoverinfo='text',
            text = rn.full_name_with_key,
            line={'color':color_palette['orange']},
            opacity=0.5,
            showlegend=False))

    togra_offset = float(roman_text.rn_list[-1].onset + roman_text.rn_list[-1].duration)
    m21_offset = float(m21_roman_text.rn_list[-1].onset + m21_roman_text.rn_list[-1].duration)
    augnet_offset = float(augnet_roman_text.rn_list[-1].onset + augnet_roman_text.rn_list[-1].duration)

    fig.add_trace(go.Scatter(
        x = (0, min(togra_offset,m21_offset,augnet_offset)),
        y = (1,1),
        mode='lines',
        line={'color':color_palette['light_green'], "width":10},
        hoverinfo='skip',
        showlegend=True,
        name = "Correct analysis"
        ))
    fig.add_trace(go.Scatter(
        x = (0, min(augnet_offset,m21_offset)),
        y = (3,3),
        mode='lines',
        line={'color':color_palette['light_green'], "width":10},
        hoverinfo='skip',
        showlegend=False,
        name = "Correct analysis"
        ))

    false_x = []
    false_y = []
    for onset,duration in roman_text.where_wrong_key:
        false_x.append(float(onset))
        false_x.append(float(onset+duration))
        false_x.append(None)
        false_y.append(1)
        false_y.append(1)
        false_y.append(None)
    fig.add_trace(go.Scatter(
        x = false_x,
        y = false_y,
        mode='lines',
        line={'color':color_palette['red'], "width":10},
        hoverinfo='skip',
        showlegend=True,
        name = "Wrong key"
        ))

    false_x = []
    false_y = []
    for onset,duration in roman_text.where_wrong_degree:
        false_x.append(float(onset))
        false_x.append(float(onset+duration))
        false_x.append(None)
        false_y.append(1)
        false_y.append(1)
        false_y.append(None)
    fig.add_trace(go.Scatter(
        x = false_x,
        y = false_y,
        mode='lines',
        line={'color':color_palette['orange'], "width":10},
        hoverinfo='skip',
        showlegend=True,
        name = "Wrong degree"
        ))

    false_x = []
    false_y = []
    for onset,duration in roman_text.where_wrong_quality:
        false_x.append(float(onset))
        false_x.append(float(onset+duration))
        false_x.append(None)
        false_y.append(1)
        false_y.append(1)
        false_y.append(None)
    fig.add_trace(go.Scatter(
        x = false_x,
        y = false_y,
        mode='lines',
        line={'color':color_palette['yellow'], "width":10},
        hoverinfo='skip',
        showlegend=True,
        name = "Wrong quality"
        ))


    false_x = []
    false_y = []
    for onset,duration in augnet_roman_text.where_wrong_key:
        false_x.append(float(onset))
        false_x.append(float(onset+duration))
        false_x.append(None)
        false_y.append(3)
        false_y.append(3)
        false_y.append(None)
    fig.add_trace(go.Scatter(
        x = false_x,
        y = false_y,
        mode='lines',
        line={'color':color_palette['red'], "width":10},
        hoverinfo='skip',
        showlegend=True,
        name = "Wrong key"
        ))

    false_x = []
    false_y = []
    for onset,duration in augnet_roman_text.where_wrong_degree:
        false_x.append(float(onset))
        false_x.append(float(onset+duration))
        false_x.append(None)
        false_y.append(3)
        false_y.append(3)
        false_y.append(None)
    fig.add_trace(go.Scatter(
        x = false_x,
        y = false_y,
        mode='lines',
        line={'color':color_palette['orange'], "width":10},
        hoverinfo='skip',
        showlegend=False,
        name = "Wrong degree"
        ))

    false_x = []
    false_y = []
    for onset,duration in augnet_roman_text.where_wrong_quality:
        false_x.append(float(onset))
        false_x.append(float(onset+duration))
        false_x.append(None)
        false_y.append(3)
        false_y.append(3)
        false_y.append(None)
    fig.add_trace(go.Scatter(
        x = false_x,
        y = false_y,
        mode='lines',
        line={'color':color_palette['yellow'], "width":10},
        hoverinfo='skip',
        showlegend=False,
        name = "Wrong quality"
        ))

    for rn in m21_roman_text.rn_list:
        x0 = float(rn.onset)
        x1 = float(rn.onset + rn.duration)-0.2
        y0 = 1.5
        y1 = 2.5
        rn_text = '<br>'.join(rn.full_name_with_key.split(': '))
        text.append(rn_text)
        text_x.append((x0+x1)/2)
        text_y.append(2)
        fig.add_trace(go.Scatter(
            x = [x0,x1,x1,x0,x0,None],
            y = [y0,y0,y1,y1,y0,None],
            fill='toself',
            fillcolor=color_palette['light_blue'],
            hoverinfo='text',
            text = rn.full_name_with_key,
            mode='lines',
            line={'color':color_palette['light_blue']},
            opacity=0.5,
            showlegend=False))

    for rn in augnet_roman_text.rn_list:
        x0 = float(rn.onset)
        x1 = float(rn.onset + rn.duration)-0.2
        y0 = 3.5
        y1 = 4.5
        rn_text = '<br>'.join(rn.full_name_with_key.split(': '))
        text.append(rn_text)
        text_x.append((x0+x1)/2)
        text_y.append(4)
        fig.add_trace(go.Scatter(
            x = [x0,x1,x1,x0,x0,None],
            y = [y0,y0,y1,y1,y0,None],
            fill='toself',
            fillcolor=color_palette['orange'],
            hoverinfo='text',
            text = rn.full_name_with_key,
            mode='lines',
            line={'color':color_palette['orange']},
            opacity=0.5,
            showlegend=False))

    fig.add_trace(go.Scatter(
        x=text_x,
        y=text_y,
        text=text,
        mode='text',
        textposition='middle center',
        hoverinfo='skip',
        textfont={
            'size':14,
            'color':'black'
        },
        showlegend=True,
        name = 'Click here to hide/show the roman numerals',
        ))


    return fig

def plot_root_cloud(rhythm_tree_node, collapse_by = 'Quality', marker_size=100):
    """ Create a plotly figure with the root cloud of a rhythm tree node"""
    assert collapse_by in ['Quality', 'Diatonic', 'Chromatic']
    root_score_before_leap = rhythm_tree_node.root_score_before_leap
    leap_score = rhythm_tree_node.root_score_leap
    onset_score = rhythm_tree_node.root_score_onset
    qualities = rhythm_tree_node.qualities

    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=f'Root Cloud (Collapsed by {collapse_by.lower()})')
    where = np.argwhere(root_score_before_leap)
    point_dict = {}
    for diatonic, chromatic, quality_idx in where:
        if collapse_by == 'Quality':
            x,y,z = diatonic, chromatic, quality_idx
        elif collapse_by == 'Diatonic':
            x,y,z = chromatic, quality_idx, diatonic
        elif collapse_by == 'Chromatic':
            x,y,z = diatonic, quality_idx, chromatic

        if (x, y) not in point_dict:
            point_dict[(x, y)] = []
        point_dict[(x, y)].append((z, root_score_before_leap[diatonic, chromatic, quality_idx]))

    x = []
    y = []
    text = []
    scores = []
    colors = []

    for (x_axis, y_axis), points in point_dict.items():
        if collapse_by == 'Quality':
            leap_compatible = leap_score[x_axis, y_axis, :].sum() > 0
            onset_compatible = onset_score[x_axis, y_axis, :].sum() > 0
        elif collapse_by == 'Diatonic':
            leap_compatible = leap_score[:, x_axis, y_axis].sum() > 0
            onset_compatible = onset_score[:, x_axis, y_axis].sum() > 0
        elif collapse_by == 'Chromatic':
            leap_compatible = leap_score[x_axis, :, y_axis].sum() > 0
            onset_compatible = onset_score[x_axis, :, y_axis].sum() > 0

        x.append(x_axis)
        y.append(y_axis)
        max_score = 0
        t = ''
        for (z_axis, score) in points:
            if collapse_by == 'Quality':
                diatonic, chromatic, quality_idx = x_axis, y_axis, z_axis
            elif collapse_by == 'Diatonic':
                chromatic, quality_idx, diatonic = x_axis, y_axis, z_axis
            elif collapse_by == 'Chromatic':
                diatonic, quality_idx, chromatic = x_axis, y_axis, z_axis
            quality = qualities[int(quality_idx)]
            pitch = Pitch(int(diatonic), int(chromatic))
            inversion = rhythm_tree_node.inversion[diatonic, chromatic, quality_idx]
            t += f"{pitch}{quality.label}: {100*score:.1f}%<br>Inversion: {inversion}<br><br>"
            max_score = max(max_score, score)
        scores.append(marker_size*max_score)
        text.append(t)
        if onset_compatible:
            colors.append(color_palette['green'])
        elif leap_compatible:
            colors.append(color_palette['blue'])
        else:
            colors.append(color_palette['red'])

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        hoverinfo='text',
        text=text,
        marker={
            'size':scores,
            'color':colors,
            'opacity':0.7,
        }
    ))

    diatonic_axis = {
            'range' : [-0.5, 6.5],
            'tickvals' : list(range(7)),
            'ticktext' : ['C','D','E','F','G','A','B'],
    }
    chromatic_axis =  {
            'range' : [-0.5, 11.5],
            'tickvals' : list(range(12)),
            'ticktext' : ['C','C#', 'D','E-', 'E', 'F', 'F#', 'G', 'A-', 'A','B-', 'B'],
    }
    quality_axis = {
            'range' : [-0.5, qualities.len-0.5],
            'tickvals' : list(range(qualities.len)),
            'ticktext' : list(qualities.quality_dict.keys()),
    }

    if collapse_by == 'Quality':
        fig.update_xaxes(diatonic_axis)
        fig.update_yaxes(chromatic_axis)
    elif collapse_by == 'Diatonic':
        fig.update_xaxes(chromatic_axis)
        fig.update_yaxes(quality_axis)
    elif collapse_by == 'Chromatic':
        fig.update_xaxes(diatonic_axis)
        fig.update_yaxes(quality_axis)

    return fig
