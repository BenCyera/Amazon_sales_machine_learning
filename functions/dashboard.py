import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import widgets
from IPython.display import display, HTML
import numpy as np

class MappingDashboard:
    def __init__(self, json_path):
        """Initialize dashboard with path to mapping suggestions JSON"""
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.setup_dashboard()
    
    def setup_dashboard(self):
        """Create and display the interactive dashboard"""
        # Create dropdown for dataset selection
        self.dataset_dropdown = widgets.Dropdown(
            options=list(self.data['mapping_suggestions'].keys()),
            description='Dataset Comparison:',
            style={'description_width': 'initial'},
            layout={'width': '50%'}
        )
        
        # Create output widgets for plots
        self.similarity_plot = widgets.Output()
        self.mapping_details = widgets.Output()
        
        # Set up event handling
        self.dataset_dropdown.observe(self.update_dashboard, names='value')
        
        # Display all components
        display(HTML("<h2>Category Mapping Analysis Dashboard</h2>"))
        display(self.dataset_dropdown)
        display(self.similarity_plot)
        display(self.mapping_details)
        
        # Show initial visualization
        if self.dataset_dropdown.options:
            self.update_dashboard({'new': self.dataset_dropdown.options[0]})
    
    def create_similarity_heatmap(self, comparison_key):
        """Create heatmap of similarity scores"""
        mappings = self.data['mapping_suggestions'][comparison_key]
        
        # Prepare data for heatmap
        source_cats = list(mappings.keys())
        target_cats = list(set(s['category'] for m in mappings.values() for s in m))
        
        scores = np.zeros((len(source_cats), len(target_cats)))
        for i, source in enumerate(source_cats):
            for suggestion in mappings[source]:
                j = target_cats.index(suggestion['category'])
                scores[i, j] = suggestion['similarity_score']
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=scores,
            x=target_cats,
            y=source_cats,
            colorscale='Blues',
            text=np.round(scores, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Similarity Scores Heatmap',
            xaxis_title='Target Categories',
            yaxis_title='Source Categories',
            height=max(400, len(source_cats) * 30),
            width=max(600, len(target_cats) * 50)
        )
        
        return fig
    
    def create_bar_chart(self, comparison_key):
        """Create bar chart of best matches"""
        mappings = self.data['mapping_suggestions'][comparison_key]
        
        # Prepare data for bar chart
        data = []
        for source, suggestions in mappings.items():
            best_match = max(suggestions, key=lambda x: x['similarity_score'])
            data.append({
                'Source': source,
                'Target': best_match['category'],
                'Score': best_match['similarity_score']
            })
        
        df = pd.DataFrame(data)
        
        fig = px.bar(
            df,
            x='Score',
            y='Source',
            orientation='h',
            title='Best Matches by Category',
            text=df['Target'],
            color='Score',
            color_continuous_scale='Blues'
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=max(400, len(data) * 30),
            width=800,
            showlegend=False
        )
        
        return fig
    
    def display_mapping_details(self, comparison_key):
        """Display detailed mapping information"""
        mappings = self.data['mapping_suggestions'][comparison_key]
        
        html = "<div style='margin-top: 20px'>"
        html += "<h3>Detailed Mapping Information</h3>"
        html += "<table style='width: 100%; border-collapse: collapse;'>"
        html += "<tr><th>Source Category</th><th>Top Matches</th></tr>"
        
        for source, suggestions in mappings.items():
            html += f"<tr style='border-bottom: 1px solid #ddd'>"
            html += f"<td style='padding: 8px'>{source}</td>"
            html += "<td style='padding: 8px'>"
            for s in sorted(suggestions, key=lambda x: x['similarity_score'], reverse=True)[:3]:
                html += f"{s['category']} ({s['similarity_score']:.2f})<br>"
            html += "</td></tr>"
        
        html += "</table></div>"
        return HTML(html)
    
    def update_dashboard(self, change):
        """Update all dashboard components"""
        comparison_key = change['new']
        
        with self.similarity_plot:
            self.similarity_plot.clear_output(wait=True)
            heatmap = self.create_similarity_heatmap(comparison_key)
            bar_chart = self.create_bar_chart(comparison_key)
            display(heatmap)
            display(bar_chart)
        
        with self.mapping_details:
            self.mapping_details.clear_output(wait=True)
            details = self.display_mapping_details(comparison_key)
            display(details)

# Example usage
if __name__ == "__main__":
    # For .py file
    dashboard = MappingDashboard('category_mapping_suggestions.json')