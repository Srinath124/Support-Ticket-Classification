"""Utility package initialization."""

from .model_loader import load_models, predict_ticket, predict_batch
from .data_processor import (
    parse_uploaded_file,
    parse_pasted_data,
    validate_ticket_data,
    normalize_ticket_data,
    add_predictions_to_dataframe,
    suggest_columns
)
from .visualizations import (
    create_bar_chart,
    create_category_bar_chart,
    create_line_chart,
    create_pie_chart,
    create_metric_bars,
    create_gauge_chart,
    get_category_color,
    get_priority_color,
    get_sentiment_color,
    COLORS
)
from .mock_data import (
    get_real_dashboard_metrics as get_dashboard_metrics,
    get_real_category_distribution as get_category_distribution,
    get_real_model_metrics as get_model_metrics,
    get_weekly_trends,
    get_accuracy_trend,
    get_sentiment_distribution,
    get_recent_tickets,
    get_model_versions,
    get_model_status,
    get_model_comparison,
)

__all__ = [
    'load_models',
    'predict_ticket',
    'predict_batch',
    'parse_uploaded_file',
    'parse_pasted_data',
    'validate_ticket_data',
    'normalize_ticket_data',
    'add_predictions_to_dataframe',
    'suggest_columns',
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
    'get_dashboard_metrics',
    'get_category_distribution',
    'get_weekly_trends',
    'get_accuracy_trend',
    'get_sentiment_distribution',
    'get_model_metrics',
    'get_recent_tickets',
    'get_model_versions',
    'get_model_status',
    'get_model_comparison',
]
