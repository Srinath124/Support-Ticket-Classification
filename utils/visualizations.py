"""
Enhanced visualization utilities for black/white theme dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


# Color scheme for black/white theme
COLORS = {
    'primary': '#FFFFFF',
    'secondary': '#B0B0B0',
    'accent': '#808080',
    'success': '#4ADE80',
    'warning': '#FBBF24',
    'danger': '#EF4444',
    'info': '#60A5FA',
    'background': '#000000',
    'card': '#1A1A1A',
    'text': '#FFFFFF',
}

CATEGORY_COLORS = {
    'Bug Report': '#EF4444',
    'Feature Request': '#8B5CF6',
    'Billing Issue': '#F59E0B',
    'Account Help': '#3B82F6',
    'Performance': '#10B981',
    'Other': '#6B7280',
}

PRIORITY_COLORS = {
    'High': '#EF4444',
    'Medium': '#F59E0B',
    'Low': '#10B981',
}

SENTIMENT_COLORS = {
    'Positive': '#4ADE80',
    'Neutral': '#8B5CF6',
    'Negative': '#EF4444',
}


def get_base_layout():
    """Get base layout for all charts with black theme."""
    return {
        'plot_bgcolor': '#000000',
        'paper_bgcolor': '#1A1A1A',
        'font': {
            'family': 'Inter, sans-serif',
            'color': '#FFFFFF',
            'size': 12
        },
        'title': {
            'font': {'size': 18, 'color': '#FFFFFF', 'family': 'Inter'},
            'x': 0.5,
            'xanchor': 'center'
        },
        'xaxis': {
            'gridcolor': '#2A2A2A',
            'linecolor': '#3A3A3A',
            'tickfont': {'color': '#B0B0B0'},
            'title': {'font': {'color': '#FFFFFF'}}
        },
        'yaxis': {
            'gridcolor': '#2A2A2A',
            'linecolor': '#3A3A3A',
            'tickfont': {'color': '#B0B0B0'},
            'title': {'font': {'color': '#FFFFFF'}}
        },
        'hoverlabel': {
            'bgcolor': '#1A1A1A',
            'font': {'family': 'Inter', 'color': '#FFFFFF'},
            'bordercolor': '#3A3A3A'
        },
        'margin': {'l': 40, 'r': 40, 't': 60, 'b': 40}
    }


def create_bar_chart(data, x_col, y_col, title="", color=None):
    """Create a modern bar chart with gradient effect."""
    if isinstance(data, dict):
        df = pd.DataFrame(list(data.items()), columns=[x_col, y_col])
    else:
        df = data
    
    fig = go.Figure()
    
    # Add bars with gradient effect
    fig.add_trace(go.Bar(
        x=df[x_col],
        y=df[y_col],
        marker=dict(
            color=df[y_col],
            colorscale=[[0, '#3A3A3A'], [0.5, '#FFFFFF'], [1, '#E0E0E0']],
            line=dict(color='#FFFFFF', width=1),
            showscale=False
        ),
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
        text=df[y_col],
        textposition='outside',
        textfont=dict(color='#FFFFFF', size=14, family='Inter')
    ))
    
    layout = get_base_layout()
    layout['title'] = {'text': title}
    layout['showlegend'] = False
    layout['xaxis']['title'] = ''
    layout['yaxis']['title'] = ''
    layout['yaxis']['showgrid'] = True
    
    fig.update_layout(**layout)
    fig.update_layout(height=400)
    
    return fig


def create_category_bar_chart(data):
    """Create category-specific bar chart with custom colors."""
    if isinstance(data, dict):
        categories = list(data.keys())
        values = list(data.values())
    else:
        categories = data.index.tolist()
        values = data.values.tolist()
    
    colors = [CATEGORY_COLORS.get(cat, '#FFFFFF') for cat in categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker=dict(
            color=colors,
            line=dict(color='#FFFFFF', width=1.5),
            opacity=0.9
        ),
        hovertemplate='<b>%{x}</b><br>Tickets: %{y}<extra></extra>',
        text=values,
        textposition='outside',
        textfont=dict(color='#FFFFFF', size=14, family='Inter', weight='bold')
    ))
    
    layout = get_base_layout()
    layout['showlegend'] = False
    layout['xaxis']['tickangle'] = -45
    layout['yaxis']['showgrid'] = True
    layout['height'] = 450
    
    fig.update_layout(**layout)
    
    return fig


def create_line_chart(data, x_col, y_cols, title="", colors=None):
    """Create an interactive line chart with multiple series."""
    fig = go.Figure()
    
    if colors is None:
        colors = ['#FFFFFF', '#B0B0B0', '#808080']
    
    for i, col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[col],
            name=col.capitalize(),
            mode='lines+markers',
            line=dict(
                color=colors[i % len(colors)],
                width=3,
                shape='spline'
            ),
            marker=dict(
                size=8,
                color=colors[i % len(colors)],
                line=dict(color='#000000', width=2)
            ),
            hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>',
            fill='tonexty' if i > 0 else None,
            fillcolor=f'rgba({int(colors[i % len(colors)][1:3], 16)}, {int(colors[i % len(colors)][3:5], 16)}, {int(colors[i % len(colors)][5:7], 16)}, 0.1)'
        ))
    
    layout = get_base_layout()
    layout['title'] = {'text': title}
    layout['showlegend'] = True
    layout['legend'] = {
        'orientation': 'h',
        'yanchor': 'bottom',
        'y': 1.02,
        'xanchor': 'right',
        'x': 1,
        'font': {'color': '#FFFFFF'}
    }
    layout['yaxis']['showgrid'] = True
    layout['height'] = 400
    
    fig.update_layout(**layout)
    
    return fig


def create_pie_chart(labels, values, title=""):
    """Create a modern donut chart."""
    colors = [SENTIMENT_COLORS.get(label, '#FFFFFF') for label in labels]
    
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(
            colors=colors,
            line=dict(color='#000000', width=2)
        ),
        textfont=dict(size=14, color='#FFFFFF', family='Inter', weight='bold'),
        hovertemplate='<b>%{label}</b><br>%{value} (%{percent})<extra></extra>',
        textposition='outside',
        textinfo='label+percent'
    ))
    
    layout = get_base_layout()
    layout['title'] = {'text': title}
    layout['showlegend'] = False
    layout['height'] = 350
    
    # Add center text
    fig.add_annotation(
        text=f"<b>{sum(values)}</b><br>Total",
        x=0.5, y=0.5,
        font=dict(size=24, color='#FFFFFF', family='Inter'),
        showarrow=False
    )
    
    fig.update_layout(**layout)
    
    return fig


def create_metric_bars(metrics, title=""):
    """Create horizontal bar chart for metrics."""
    metric_names = list(metrics.keys())
    metric_values = [v * 100 if v < 1 else v for v in metrics.values()]
    
    colors = ['#FFFFFF', '#E0E0E0', '#C0C0C0', '#B0B0B0']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=metric_names,
        x=metric_values,
        orientation='h',
        marker=dict(
            color=colors[:len(metric_names)],
            line=dict(color='#FFFFFF', width=1.5)
        ),
        text=[f'{v:.1f}%' for v in metric_values],
        textposition='outside',
        textfont=dict(color='#FFFFFF', size=14, family='Inter', weight='bold'),
        hovertemplate='<b>%{y}</b><br>%{x:.1f}%<extra></extra>'
    ))
    
    layout = get_base_layout()
    layout['title'] = {'text': title}
    layout['showlegend'] = False
    layout['xaxis']['range'] = [0, 100]
    layout['xaxis']['showgrid'] = True
    layout['yaxis']['showgrid'] = False
    layout['height'] = 300
    
    fig.update_layout(**layout)
    
    return fig


def create_gauge_chart(value, title="", max_value=100):
    """Create a gauge chart for single metrics."""
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title, 'font': {'color': '#FFFFFF', 'size': 18}},
        delta={'reference': max_value * 0.8, 'increasing': {'color': '#4ADE80'}},
        gauge={
            'axis': {'range': [0, max_value], 'tickcolor': '#FFFFFF'},
            'bar': {'color': '#FFFFFF'},
            'bgcolor': '#1A1A1A',
            'borderwidth': 2,
            'bordercolor': '#3A3A3A',
            'steps': [
                {'range': [0, max_value * 0.5], 'color': '#2A2A2A'},
                {'range': [max_value * 0.5, max_value * 0.75], 'color': '#3A3A3A'},
                {'range': [max_value * 0.75, max_value], 'color': '#4A4A4A'}
            ],
            'threshold': {
                'line': {'color': '#4ADE80', 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        },
        number={'font': {'color': '#FFFFFF', 'size': 40}}
    ))
    
    layout = get_base_layout()
    layout['height'] = 250
    
    fig.update_layout(**layout)
    
    return fig


# Helper functions for colors
def get_category_color(category):
    """Get color for a category."""
    return CATEGORY_COLORS.get(category, '#FFFFFF')


def get_priority_color(priority):
    """Get color for a priority level."""
    return PRIORITY_COLORS.get(priority, '#FFFFFF')


def get_sentiment_color(sentiment):
    """Get color for a sentiment."""
    return SENTIMENT_COLORS.get(sentiment, '#FFFFFF')


__all__ = [
    'create_bar_chart',
    'create_category_bar_chart',
    'create_line_chart',
    'create_pie_chart',
    'create_metric_bars',
    'create_gauge_chart',
    'get_category_color',
    'get_priority_color',
    'get_sentiment_color',
    'COLORS',
    'CATEGORY_COLORS',
    'PRIORITY_COLORS',
    'SENTIMENT_COLORS',
]
