"""
Training progress dashboard with real-time metrics visualization.
"""
import numpy as np
import torch
from typing import Dict, List, Optional
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from pathlib import Path
import logging
from threading import Thread
from queue import Queue
import json

logger = logging.getLogger(__name__)

class TrainingDashboard:
    """Interactive dashboard for monitoring training progress."""
    
    def __init__(
        self,
        config: Dict,
        log_dir: str,
        port: int = 8050,
        dark_mode: bool = True
    ):
        """
        Initialize dashboard.
        
        Args:
            config: Dashboard configuration
            log_dir: Directory containing training logs
            port: Port for dashboard server
            dark_mode: Use dark theme
        """
        self.config = config
        self.log_dir = Path(log_dir)
        self.port = port
        self.dark_mode = dark_mode
        
        # Initialize data storage
        self.metrics_data = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'gpu_utilization': [],
            'memory_usage': [],
            'batch_time': []
        }
        
        # Setup update queue
        self.update_queue = Queue()
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.H1('Shuttlecock Detection Training Dashboard',
                   style={'textAlign': 'center'}),
            
            # Training Metrics Section
            html.Div([
                html.H2('Training Metrics'),
                dcc.Graph(id='loss-plot'),
                dcc.Graph(id='lr-plot'),
                dcc.Interval(id='interval-component',
                           interval=2*1000,  # 2 seconds
                           n_intervals=0)
            ]),
            
            # Resource Utilization Section
            html.Div([
                html.H2('Resource Utilization'),
                dcc.Graph(id='gpu-plot'),
                dcc.Graph(id='memory-plot')
            ]),
            
            # Model Performance Section
            html.Div([
                html.H2('Model Performance'),
                dcc.Graph(id='precision-recall-plot'),
                dcc.Graph(id='confusion-matrix')
            ]),
            
            # Batch Statistics Section
            html.Div([
                html.H2('Batch Statistics'),
                dcc.Graph(id='batch-time-plot'),
                dcc.Graph(id='batch-size-plot')
            ])
        ])
        
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        @self.app.callback(
            [Output('loss-plot', 'figure'),
             Output('lr-plot', 'figure'),
             Output('gpu-plot', 'figure'),
             Output('memory-plot', 'figure'),
             Output('batch-time-plot', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_graphs(n):
            # Process any new data in queue
            while not self.update_queue.empty():
                update = self.update_queue.get()
                self._update_metrics(update)
                
            return (
                self._create_loss_plot(),
                self._create_lr_plot(),
                self._create_gpu_plot(),
                self._create_memory_plot(),
                self._create_batch_time_plot()
            )
            
    def _create_loss_plot(self) -> go.Figure:
        """Create loss plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=self.metrics_data['train_loss'],
            name='Training Loss',
            mode='lines+markers'
        ))
        
        fig.add_trace(go.Scatter(
            y=self.metrics_data['val_loss'],
            name='Validation Loss',
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title='Training and Validation Loss',
            xaxis_title='Iteration',
            yaxis_title='Loss',
            template='plotly_dark' if self.dark_mode else 'plotly_white'
        )
        
        return fig
        
    def _create_lr_plot(self) -> go.Figure:
        """Create learning rate plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=self.metrics_data['learning_rate'],
            name='Learning Rate',
            mode='lines'
        ))
        
        fig.update_layout(
            title='Learning Rate Schedule',
            xaxis_title='Iteration',
            yaxis_title='Learning Rate',
            template='plotly_dark' if self.dark_mode else 'plotly_white'
        )
        
        return fig
        
    def _create_gpu_plot(self) -> go.Figure:
        """Create GPU utilization plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=self.metrics_data['gpu_utilization'],
            name='GPU Utilization',
            mode='lines',
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title='GPU Utilization',
            xaxis_title='Time',
            yaxis_title='Utilization (%)',
            template='plotly_dark' if self.dark_mode else 'plotly_white'
        )
        
        return fig
        
    def _create_memory_plot(self) -> go.Figure:
        """Create memory usage plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=self.metrics_data['memory_usage'],
            name='Memory Usage',
            mode='lines',
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title='GPU Memory Usage',
            xaxis_title='Time',
            yaxis_title='Memory (GB)',
            template='plotly_dark' if self.dark_mode else 'plotly_white'
        )
        
        return fig
        
    def _create_batch_time_plot(self) -> go.Figure:
        """Create batch processing time plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=self.metrics_data['batch_time'],
            name='Batch Time',
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title='Batch Processing Time',
            xaxis_title='Batch',
            yaxis_title='Time (ms)',
            template='plotly_dark' if self.dark_mode else 'plotly_white'
        )
        
        return fig
        
    def _update_metrics(self, update: Dict):
        """Update stored metrics with new data."""
        for key, value in update.items():
            if key in self.metrics_data:
                self.metrics_data[key].append(value)
                
    def update(self, metrics: Dict):
        """
        Update dashboard with new metrics.
        
        Args:
            metrics: Dictionary of new metric values
        """
        self.update_queue.put(metrics)
        
    def start(self):
        """Start dashboard server."""
        self.app.run_server(debug=False, port=self.port)
        
    def start_async(self):
        """Start dashboard server in separate thread."""
        thread = Thread(target=self.start, daemon=True)
        thread.start()
        
    def save_metrics(self, filename: str):
        """Save metrics history to file."""
        save_path = self.log_dir / filename
        with open(save_path, 'w') as f:
            json.dump(self.metrics_data, f)
            
    def load_metrics(self, filename: str):
        """Load metrics history from file."""
        load_path = self.log_dir / filename
        with open(load_path, 'r') as f:
            self.metrics_data = json.load(f)
