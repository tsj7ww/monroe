"""
Visualization module for loss aversion analysis.

This module provides functions to create publication-quality plots and
interactive visualizations for loss aversion analysis results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LossAversionPlotter:
    """Class to create visualizations for loss aversion analysis."""
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        style: str = 'seaborn-whitegrid',
        context: str = 'paper',
        palette: str = 'colorblind'
    ):
        """
        Initialize the plotter with style settings.

        Args:
            output_dir: Directory to save plots
            style: Matplotlib/Seaborn style
            context: Seaborn context
            palette: Color palette
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use(style)
        sns.set_context(context)
        sns.set_palette(palette)
        
    def plot_volume_response(
        self,
        event_data: pd.DataFrame,
        save: bool = True,
        interactive: bool = False
    ) -> Union[plt.Figure, go.Figure]:
        """
        Plot volume response around events.
        
        Args:
            event_data: Processed event window data
            save: Whether to save the plot
            interactive: Whether to create interactive Plotly plot
            
        Returns:
            Figure object
        """
        if interactive:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Volume Response', 'Cumulative Response')
            )
            
            # Volume response
            for event_type in ['Gain_Event', 'Loss_Event']:
                data = event_data[event_data['Event_Type'] == event_type]
                mean_vol = data.groupby('Days_From_Event')['Abnormal_Volume'].mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=mean_vol.index,
                        y=mean_vol.values,
                        name=event_type,
                        mode='lines+markers'
                    ),
                    row=1, col=1
                )
                
            # Cumulative response
            for event_type in ['Gain_Event', 'Loss_Event']:
                data = event_data[event_data['Event_Type'] == event_type]
                cum_vol = data.groupby('Days_From_Event')['Abnormal_Volume'].mean().cumsum()
                
                fig.add_trace(
                    go.Scatter(
                        x=cum_vol.index,
                        y=cum_vol.values,
                        name=f'{event_type} (Cumulative)',
                        mode='lines'
                    ),
                    row=2, col=1
                )
                
            fig.update_layout(
                height=800,
                title_text='Volume Response to Gains and Losses',
                showlegend=True
            )
            
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            # Volume response
            for event_type in ['Gain_Event', 'Loss_Event']:
                data = event_data[event_data['Event_Type'] == event_type]
                mean_vol = data.groupby('Days_From_Event')['Abnormal_Volume'].mean()
                std_vol = data.groupby('Days_From_Event')['Abnormal_Volume'].std()
                
                ax1.plot(mean_vol.index, mean_vol.values, 'o-', label=event_type)
                ax1.fill_between(
                    mean_vol.index,
                    mean_vol - 1.96 * std_vol / np.sqrt(len(data)),
                    mean_vol + 1.96 * std_vol / np.sqrt(len(data)),
                    alpha=0.2
                )
                
            ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
            ax1.set_xlabel('Days from Event')
            ax1.set_ylabel('Abnormal Volume')
            ax1.set_title('Volume Response')
            ax1.legend()
            
            # Cumulative response
            for event_type in ['Gain_Event', 'Loss_Event']:
                data = event_data[event_data['Event_Type'] == event_type]
                cum_vol = data.groupby('Days_From_Event')['Abnormal_Volume'].mean().cumsum()
                
                ax2.plot(cum_vol.index, cum_vol.values, label=f'{event_type} (Cumulative)')
                
            ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Days from Event')
            ax2.set_ylabel('Cumulative Abnormal Volume')
            ax2.set_title('Cumulative Response')
            ax2.legend()
            
            plt.tight_layout()
            
        if save:
            output_path = self.output_dir / 'volume_response'
            if interactive:
                fig.write_html(output_path.with_suffix('.html'))
            else:
                fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
                
        return fig
        
    def plot_magnitude_effects(
        self,
        event_data: pd.DataFrame,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot relationship between event magnitude and response.
        
        Args:
            event_data: Processed event window data
            save: Whether to save the plot
            
        Returns:
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot with regression
        for event_type, ax in zip(['Gain_Event', 'Loss_Event'], [ax1, ax2]):
            data = event_data[event_data['Event_Type'] == event_type]
            
            sns.regplot(
                data=data,
                x='Event_Size',
                y='Abnormal_Volume',
                scatter_kws={'alpha': 0.5},
                ax=ax
            )
            
            ax.set_title(f'{event_type} Magnitude Effect')
            ax.set_xlabel('Event Size (Std Dev)')
            ax.set_ylabel('Abnormal Volume')
            
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / 'magnitude_effects.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_sector_heatmap(
        self,
        sector_metrics: pd.DataFrame,
        save: bool = True,
        interactive: bool = False
    ) -> Union[plt.Figure, go.Figure]:
        """
        Create heatmap of sector-specific loss aversion patterns.
        
        Args:
            sector_metrics: DataFrame with sector-level metrics
            save: Whether to save the plot
            interactive: Whether to create interactive Plotly plot
            
        Returns:
            Figure object
        """
        if interactive:
            fig = go.Figure(data=go.Heatmap(
                z=sector_metrics.values,
                x=sector_metrics.columns,
                y=sector_metrics.index,
                colorscale='RdBu',
                zmid=1.0
            ))
            
            fig.update_layout(
                title='Sector-Specific Loss Aversion Patterns',
                xaxis_title='Metric',
                yaxis_title='Sector'
            )
            
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            sns.heatmap(
                sector_metrics,
                center=1.0,
                cmap='RdBu',
                annot=True,
                fmt='.2f',
                ax=ax
            )
            
            ax.set_title('Sector-Specific Loss Aversion Patterns')
            plt.tight_layout()
            
        if save:
            output_path = self.output_dir / 'sector_heatmap'
            if interactive:
                fig.write_html(output_path.with_suffix('.html'))
            else:
                fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
                
        return fig
        
    def plot_temporal_stability(
        self,
        rolling_metrics: pd.DataFrame,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot temporal stability of loss aversion patterns.
        
        Args:
            rolling_metrics: DataFrame with rolling window metrics
            save: Whether to save the plot
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot rolling metrics
        ax.plot(
            rolling_metrics.index,
            rolling_metrics['volume_ratio'],
            label='Volume Response Ratio'
        )
        
        # Add confidence bands if available
        if 'ci_lower' in rolling_metrics.columns:
            ax.fill_between(
                rolling_metrics.index,
                rolling_metrics['ci_lower'],
                rolling_metrics['ci_upper'],
                alpha=0.2
            )
            
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Loss/Gain Response Ratio')
        ax.set_title('Temporal Stability of Loss Aversion')
        ax.legend()
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / 'temporal_stability.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_summary_dashboard(
        self,
        results: Dict,
        save: bool = True
    ) -> go.Figure:
        """
        Create interactive dashboard summarizing key results.
        
        Args:
            results: Dictionary containing analysis results
            save: Whether to save the dashboard
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Volume Response',
                'Magnitude Effects',
                'Sector Patterns',
                'Temporal Stability'
            )
        )
        
        # Add volume response plot
        event_data = results['volume_response']['event_data']
        for event_type in ['Gain_Event', 'Loss_Event']:
            data = event_data[event_data['Event_Type'] == event_type]
            mean_vol = data.groupby('Days_From_Event')['Abnormal_Volume'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=mean_vol.index,
                    y=mean_vol.values,
                    name=event_type
                ),
                row=1, col=1
            )
            
        # Add magnitude effects
        fig.add_trace(
            go.Scatter(
                x=results['magnitude_effects']['event_size'],
                y=results['magnitude_effects']['response'],
                mode='markers',
                name='Magnitude Effect'
            ),
            row=1, col=2
        )
        
        # Add sector heatmap
        fig.add_trace(
            go.Heatmap(
                z=results['sector_patterns']['heatmap_data'],
                x=results['sector_patterns']['columns'],
                y=results['sector_patterns']['index']
            ),
            row=2, col=1
        )
        
        # Add temporal stability
        fig.add_trace(
            go.Scatter(
                x=results['temporal_stability']['dates'],
                y=results['temporal_stability']['values'],
                name='Stability'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=1000,
            title_text='Loss Aversion Analysis Summary',
            showlegend=True
        )
        
        if save:
            output_path = self.output_dir / 'summary_dashboard.html'
            fig.write_html(output_path)
            
        return fig
        
def main():
    """Example usage of plotting functions."""
    try:
        # Initialize plotter
        plotter = LossAversionPlotter(output_dir='results/figures')
        
        # Load example data
        event_data = pd.read_parquet('data/processed/event_windows.parquet')
        sector_metrics = pd.read_parquet('data/processed/sector_metrics.parquet')
        
        # Create plots
        plotter.plot_volume_response(event_data)
        plotter.plot_magnitude_effects(event_data)
        plotter.plot_sector_heatmap(sector_metrics)
        
        logger.info("Plotting completed successfully")
        
    except Exception as e:
        logger.error(f"Error in plotting: {str(e)}")
        raise

if __name__ == '__main__':
    main()