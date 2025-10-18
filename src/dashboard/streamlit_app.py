import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime, timedelta
import logging

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.log_parser import SparkLogParser
from models.anomaly_detector import SparkLogAnomalyDetector
from models.root_cause_analyzer import RootCauseAnalyzer

# Configure Streamlit page
st.set_page_config(
    page_title="Spark Cluster Anomaly Detection Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SparkAnomalyDashboard:
    """Interactive dashboard for Spark cluster anomaly detection and monitoring."""
    
    def __init__(self):
        self.output_dir = "output"
        self.data_loaded = False
        self.df_anomalies = None
        self.root_cause_report = None
        
    def load_data(self):
        """Load processed data from pipeline outputs"""
        try:
            # Load anomaly results
            anomaly_file = os.path.join(self.output_dir, 'anomaly_results.csv')
            if os.path.exists(anomaly_file):
                self.df_anomalies = pd.read_csv(anomaly_file)
                if 'timestamp' in self.df_anomalies.columns:
                    self.df_anomalies['timestamp'] = pd.to_datetime(self.df_anomalies['timestamp'])
                self.data_loaded = True
            
            # Load root cause report
            report_file = os.path.join(self.output_dir, 'root_cause_report.json')
            if os.path.exists(report_file):
                with open(report_file, 'r') as f:
                    self.root_cause_report = json.load(f)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🔥 Spark Cluster Anomaly Detection Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        **Real-time monitoring and analysis of Spark cluster logs for proactive failure detection**
                """)
        
        st.divider()
    
    def render_sidebar(self):
        """Render sidebar with controls and filters"""
        st.sidebar.header("🎛️ Dashboard Controls")
        
        # Data refresh button
        if st.sidebar.button("🔄 Refresh Data", type="primary"):
            self.load_data()
            st.rerun()
        
        st.sidebar.divider()
        
        # Filters
        st.sidebar.header("🔍 Filters")
        
        if self.data_loaded and self.df_anomalies is not None:
            # Time range filter
            if 'timestamp' in self.df_anomalies.columns:
                min_date = self.df_anomalies['timestamp'].min().date()
                max_date = self.df_anomalies['timestamp'].max().date()
                
                date_range = st.sidebar.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            
            # Component filter
            components = ['All'] + list(self.df_anomalies['component'].unique())
            selected_component = st.sidebar.selectbox("Component", components)
            
            # Log level filter
            log_levels = ['All'] + list(self.df_anomalies['log_level'].unique())
            selected_level = st.sidebar.selectbox("Log Level", log_levels)
            
            # Anomaly threshold
            anomaly_threshold = st.sidebar.slider(
                "Anomaly Score Threshold", 
                0.0, 1.0, 0.5, 0.1
            )
            
            return {
                'date_range': date_range if 'timestamp' in self.df_anomalies.columns else None,
                'component': selected_component,
                'log_level': selected_level,
                'anomaly_threshold': anomaly_threshold
            }
        
        return {}
    
    def apply_filters(self, filters):
        """Apply filters to the data"""
        if not self.data_loaded or self.df_anomalies is None:
            return self.df_anomalies
        
        filtered_df = self.df_anomalies.copy()
        
        # Apply date filter
        if filters.get('date_range') and 'timestamp' in filtered_df.columns:
            start_date, end_date = filters['date_range']
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= start_date) &
                (filtered_df['timestamp'].dt.date <= end_date)
            ]
        
        # Apply component filter
        if filters.get('component') and filters['component'] != 'All':
            filtered_df = filtered_df[filtered_df['component'] == filters['component']]
        
        # Apply log level filter
        if filters.get('log_level') and filters['log_level'] != 'All':
            filtered_df = filtered_df[filtered_df['log_level'] == filters['log_level']]
        
        # Apply anomaly threshold
        if filters.get('anomaly_threshold'):
            threshold = filters['anomaly_threshold']
            filtered_df = filtered_df[
                (~filtered_df['is_anomaly']) | 
                (filtered_df['anomaly_score'] >= threshold)
            ]
        
        return filtered_df
    
    def render_overview_metrics(self, df):
        """Render key metrics overview"""
        st.header("📊 System Overview")
        
        if df is None or df.empty:
            st.warning("No data available")
            return
        
        # Calculate metrics
        total_logs = len(df)
        anomalous_logs = len(df[df['is_anomaly']]) if 'is_anomaly' in df.columns else 0
        anomaly_rate = (anomalous_logs / total_logs * 100) if total_logs > 0 else 0
        
        critical_anomalies = len(df[(df.get('is_anomaly', False)) & (df.get('anomaly_score', 0) > 0.8)])
        error_logs = len(df[df['log_level'] == 'ERROR']) if 'log_level' in df.columns else 0
        
        # Display metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Logs", f"{total_logs:,}")
        
        with col2:
            st.metric("Anomalies", f"{anomalous_logs:,}", 
                     delta=f"{anomaly_rate:.1f}% rate")
        
        with col3:
            st.metric("Critical Issues", critical_anomalies,
                     delta="High Priority" if critical_anomalies > 0 else "None")
        
        with col4:
            st.metric("Error Logs", error_logs)
        
        with col5:
            unique_apps = df['application_id'].nunique() if 'application_id' in df.columns else 0
            st.metric("Applications", unique_apps)
    
    def render_anomaly_timeline(self, df):
        """Render anomaly timeline chart"""
        st.header("📈 Anomaly Timeline")
        
        if df is None or df.empty or 'timestamp' not in df.columns:
            st.warning("No timestamp data available for timeline")
            return
        
        # Prepare timeline data
        df_timeline = df.copy()
        df_timeline['hour'] = df_timeline['timestamp'].dt.floor('h')
        
        # Aggregate by hour
        timeline_data = df_timeline.groupby(['hour', 'is_anomaly']).size().reset_index(name='count')
        timeline_pivot = timeline_data.pivot(index='hour', columns='is_anomaly', values='count').fillna(0)
        
        # Create timeline chart
        fig = go.Figure()
        
        if False in timeline_pivot.columns:
            fig.add_trace(go.Scatter(
                x=timeline_pivot.index,
                y=timeline_pivot[False],
                mode='lines+markers',
                name='Normal Logs',
                line=dict(color='green', width=2),
                fill='tonexty'
            ))
        
        if True in timeline_pivot.columns:
            fig.add_trace(go.Scatter(
                x=timeline_pivot.index,
                y=timeline_pivot[True],
                mode='lines+markers',
                name='Anomalous Logs',
                line=dict(color='red', width=3),
                fill='tozeroy'
            ))
        
        fig.update_layout(
            title="Log Volume and Anomalies Over Time",
            xaxis_title="Time",
            yaxis_title="Log Count",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_component_analysis(self, df):
        """Render component-wise anomaly analysis"""
        st.header("🔧 Component Analysis")
        
        if df is None or df.empty:
            st.warning("No data available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Anomalies by component
            if 'is_anomaly' in df.columns:
                component_anomalies = df[df['is_anomaly']]['component'].value_counts()
                
                fig = px.bar(
                    x=component_anomalies.values,
                    y=component_anomalies.index,
                    orientation='h',
                    title="Anomalies by Component",
                    labels={'x': 'Anomaly Count', 'y': 'Component'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Log level distribution
            if 'log_level' in df.columns:
                level_counts = df['log_level'].value_counts()
                
                fig = px.pie(
                    values=level_counts.values,
                    names=level_counts.index,
                    title="Log Level Distribution"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    def render_anomaly_heatmap(self, df):
        """Render anomaly heatmap"""
        st.header("🔥 Anomaly Heatmap")
        
        if df is None or df.empty or 'timestamp' not in df.columns:
            st.warning("No timestamp data available for heatmap")
            return
        
        # Prepare heatmap data
        df_heatmap = df[df.get('is_anomaly', False)].copy()
        
        if df_heatmap.empty:
            st.info("No anomalies to display in heatmap")
            return
        
        df_heatmap['hour'] = df_heatmap['timestamp'].dt.hour
        df_heatmap['day'] = df_heatmap['timestamp'].dt.day_name()
        
        # Create heatmap data
        heatmap_data = df_heatmap.groupby(['day', 'hour']).size().reset_index(name='anomaly_count')
        heatmap_pivot = heatmap_data.pivot(index='day', columns='hour', values='anomaly_count').fillna(0)
        
        # Create heatmap
        fig = px.imshow(
            heatmap_pivot,
            title="Anomaly Heatmap (Day vs Hour)",
            labels=dict(x="Hour of Day", y="Day of Week", color="Anomaly Count"),
            color_continuous_scale="Reds"
        )
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)

    def render_anomaly_distribution(self, df):
        """Render anomaly distribution analysis section.
        - By Log Level
        - Top 10 Affected Components
        """
        st.header("📊 Anomaly Distribution Analysis")

        if df is None or df.empty:
            st.warning("No data available")
            return

        anomalies = df[df.get('is_anomaly', False)].copy()

        if anomalies.empty:
            st.info("No anomalies detected to analyze distribution")
            return

        # Textual summary like the requested dashboard format
        total_anomalies = len(anomalies)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Log Level")
            if 'log_level' in anomalies.columns:
                level_counts = anomalies['log_level'].value_counts()
                if not level_counts.empty:
                    fig = px.pie(
                        values=level_counts.values,
                        names=level_counts.index,
                    )
                    fig.update_layout(
                        height=400,
                        legend=dict(
                            orientation='v',
                            y=0.5,
                            yanchor='middle',
                            x=0.0,
                            xanchor='left'
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No log level information available for anomalies")
            else:
                st.info("Column 'log_level' not found in data")

        with col2:
            st.subheader("Top 5 Affected Components")
            if 'component' in anomalies.columns and total_anomalies > 0:
                comp_counts = (
                    anomalies['component']
                    .fillna('UNKNOWN')
                    .replace('', 'UNKNOWN')
                    .value_counts()
                )
                top5 = comp_counts.head(5)
                if not top5.empty:
                    lines = []
                    for i, (comp, count) in enumerate(top5.items(), start=1):
                        pct = (count / total_anomalies) * 100
                        lines.append(f"{i}. **{comp}:** {count:,} anomalies ({pct:.1f}%)")
                    st.markdown("\n".join(lines))
                else:
                    st.info("No component information available for anomalies")
            else:
                st.info("Column 'component' not found in data")

    def render_root_cause_analysis(self):
        """Render root cause analysis results"""
        st.header("🔍 Root Cause Analysis")
        
        if not self.root_cause_report:
            st.warning("No root cause analysis data available")
            return
        
        # Failure patterns summary
        patterns = self.root_cause_report.get('failure_patterns', {})
        recommendations = self.root_cause_report.get('recommendations', [])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Failure Pattern Types (WHAT went wrong)")
            
            # Count failure patterns
            cascade_count = len(patterns.get('cascade_failures', []))
            resource_count = len(patterns.get('resource_exhaustion', []))
            network_count = len(patterns.get('network_issues', []))
            config_count = len(patterns.get('configuration_errors', []))
            
            st.markdown(f"🌐 **Network Issues**: {network_count} nodes")
            st.markdown(f"💾 **Resource Exhaustion**: {resource_count} applications")
            st.markdown(f"🔗 **Cascade Failures**: {cascade_count}")
            st.markdown(f"⚙️ **Configuration Errors**: {config_count}")
            
            # Temporal clusters (separate section)
            st.divider()
            st.subheader("Temporal Clustering (WHEN it happened)")
            
            temporal_clusters = patterns.get('temporal_clusters', [])
            if temporal_clusters:
                st.markdown(f"⏰ **Incident Periods Identified**: {len(temporal_clusters)}")
                
                # Show cluster details
                for i, cluster in enumerate(temporal_clusters[:3], 1):
                    cluster_size = cluster.get('size', 0)
                    start_time = cluster.get('start_time', 'N/A')
                    end_time = cluster.get('end_time', 'N/A')
                    st.markdown(f"  - **Cluster {i}**: {cluster_size} anomalies ({start_time} to {end_time})")
            else:
                st.markdown("⏰ **Incident Periods**: None detected")
        
        with col2:
            st.subheader("🎯 Recommendations")
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {rec}")
            else:
                st.info("No specific recommendations available")

    def render_detailed_view(self, df):
        """Render detailed anomaly log view"""
        st.header("📋 Detailed Anomaly Log View")
        
        if df is None or df.empty:
            st.warning("No data available")
            return
        
        # Show anomalies table
        anomalies = df[df.get('is_anomaly', False)].copy()
        
        if anomalies.empty:
            st.info("No anomalies detected")
            return
        
        # Select columns to display
        display_columns = ['timestamp', 'log_level', 'component', 'container_id', 
                          'anomaly_score', 'raw_message']
        available_columns = [col for col in display_columns if col in anomalies.columns]
        
        # Sort by anomaly score
        if 'anomaly_score' in anomalies.columns:
            anomalies = anomalies.sort_values('anomaly_score', ascending=False)
        
        st.dataframe(
            anomalies[available_columns].head(100),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = anomalies.to_csv(index=False)
        st.download_button(
            label="📥 Download Anomaly Data",
            data=csv,
            file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def run(self):
        """Main dashboard execution"""
        self.render_header()
        
        # Load data
        if not self.data_loaded:
            with st.spinner("Loading data..."):
                self.load_data()
        
        # Render sidebar and get filters
        filters = self.render_sidebar()
        
        # Apply filters to data
        filtered_df = self.apply_filters(filters)
        
        # Main dashboard content
        if self.data_loaded:
            # Overview metrics
            self.render_overview_metrics(filtered_df)
            
            st.divider()
            
            # Anomaly Timeline
            self.render_anomaly_timeline(filtered_df)
            
            st.divider()
            
            # Anomaly Distribution Analysis
            self.render_anomaly_distribution(filtered_df)

            st.divider()
            
            # Anomaly Heatmap
            self.render_anomaly_heatmap(filtered_df)

            st.divider()

            # Root Cause Analysis
            self.render_root_cause_analysis()
            
            st.divider()
            
            # Detailed view
            self.render_detailed_view(filtered_df)
            
        else:
            st.error("""
            No data found. Please run the main pipeline first:
            
            ```bash
            python src/main_pipeline.py
            ```
            
            This will generate the required data files for the dashboard.
            """)
        
        # Footer
        st.divider()
        st.markdown("""
        ---
        **Spark Cluster Anomaly Detection System**  
        *Capstone Project - IIT Roorkee Applied Data Science & AI Program*  
        *Developed by: Srinjoy Roy*
        """)

def main():
    """Main function to run the Streamlit dashboard"""
    dashboard = SparkAnomalyDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
