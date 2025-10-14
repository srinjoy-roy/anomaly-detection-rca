import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

class RootCauseAnalyzer:
    """
    Advanced Root Cause Analysis system for Spark cluster failures.
    Uses graph analysis and correlation techniques to identify failure patterns.
    """
    
    def __init__(self, time_window_minutes: int = 5):
        self.time_window_minutes = time_window_minutes
        self.failure_graph = nx.DiGraph()
        self.correlation_matrix = None
        self.failure_patterns = {}
        
    def build_failure_graph(self, df_anomalies: pd.DataFrame) -> nx.DiGraph:
        """Build a directed graph of failure propagation"""
        G = nx.DiGraph()
        
        # Group anomalies by time windows
        anomalies = df_anomalies[df_anomalies['is_anomaly']].copy()
        
        if 'timestamp' not in anomalies.columns or anomalies.empty:
            return G
        
        anomalies = anomalies.sort_values('timestamp')
        
        # Create nodes for each anomaly
        for idx, row in anomalies.iterrows():
            node_id = f"{row['component']}_{row['container_id']}_{idx}"
            G.add_node(node_id, 
                      component=row['component'],
                      container_id=row['container_id'],
                      timestamp=row['timestamp'],
                      log_level=row['log_level'],
                      template_id=row['template_id'],
                      reconstruction_error=row['reconstruction_error'])
        
        # Create edges based on temporal and component relationships
        nodes = list(G.nodes(data=True))
        for i, (node1, data1) in enumerate(nodes):
            for j, (node2, data2) in enumerate(nodes[i+1:], i+1):
                time_diff = (data2['timestamp'] - data1['timestamp']).total_seconds() / 60
                
                # Add edge if within time window
                if 0 < time_diff <= self.time_window_minutes:
                    # Calculate edge weight based on relationship strength
                    weight = self._calculate_relationship_strength(data1, data2, time_diff)
                    if weight > 0.5:  # Threshold for significant relationship
                        G.add_edge(node1, node2, weight=weight, time_diff=time_diff)
        
        self.failure_graph = G
        return G
    
    def _calculate_relationship_strength(self, data1: Dict, data2: Dict, time_diff: float) -> float:
        """Calculate the strength of relationship between two anomalies"""
        weight = 0.0
        
        # Same component increases weight
        if data1['component'] == data2['component']:
            weight += 0.4
        
        # Same container increases weight significantly
        if data1['container_id'] == data2['container_id']:
            weight += 0.6
        
        # Error severity increases weight
        if data1['log_level'] == 'ERROR' or data2['log_level'] == 'ERROR':
            weight += 0.3
        
        # Template similarity
        if data1['template_id'] == data2['template_id']:
            weight += 0.2
        
        # Time proximity (closer in time = higher weight)
        time_weight = max(0, 1 - (time_diff / self.time_window_minutes))
        weight += time_weight * 0.3
        
        # Reconstruction error correlation
        error_diff = abs(data1['reconstruction_error'] - data2['reconstruction_error'])
        if error_diff < 0.1:  # Similar reconstruction errors
            weight += 0.2
        
        return min(weight, 1.0)  # Cap at 1.0
    
    def identify_failure_patterns(self, df_anomalies: pd.DataFrame) -> Dict:
        """Identify common failure patterns and root causes"""
        patterns = {
            'cascade_failures': [],
            'resource_exhaustion': [],
            'network_issues': [],
            'configuration_errors': [],
            'temporal_clusters': []
        }
        
        anomalies = df_anomalies[df_anomalies['is_anomaly']].copy()
        
        # 1. Cascade Failures - Find strongly connected components
        if len(self.failure_graph.nodes()) > 0:
            strongly_connected = list(nx.strongly_connected_components(self.failure_graph))
            for component in strongly_connected:
                if len(component) > 2:  # Significant cascade
                    patterns['cascade_failures'].append({
                        'nodes': list(component),
                        'size': len(component),
                        'components_involved': [self.failure_graph.nodes[node]['component'] for node in component]
                    })
        
        # 2. Resource Exhaustion Patterns
        resource_anomalies = anomalies[
            (anomalies['high_memory'] == True) | 
            (anomalies['high_vcores'] == True) |
            (anomalies['raw_message'].str.contains('OutOfMemory|memory|resource', case=False, na=False))
        ]
        
        if not resource_anomalies.empty:
            patterns['resource_exhaustion'] = self._analyze_resource_patterns(resource_anomalies)
        
        # 3. Network Issues
        network_anomalies = anomalies[
            anomalies['raw_message'].str.contains('connection|network|timeout|unreachable', case=False, na=False)
        ]
        
        if not network_anomalies.empty:
            patterns['network_issues'] = self._analyze_network_patterns(network_anomalies)
        
        # 4. Configuration Errors
        config_anomalies = anomalies[
            anomalies['raw_message'].str.contains('config|property|setting|FileNotFoundException', case=False, na=False)
        ]
        
        if not config_anomalies.empty:
            patterns['configuration_errors'] = self._analyze_config_patterns(config_anomalies)
        
        # 5. Temporal Clustering
        if 'timestamp' in anomalies.columns:
            patterns['temporal_clusters'] = self._find_temporal_clusters(anomalies)
        
        self.failure_patterns = patterns
        return patterns
    
    def _analyze_resource_patterns(self, resource_anomalies: pd.DataFrame) -> List[Dict]:
        """Analyze resource exhaustion patterns"""
        patterns = []
        
        # Group by application and analyze resource trends
        for app_id in resource_anomalies['application_id'].unique():
            if pd.isna(app_id):
                continue
                
            app_anomalies = resource_anomalies[resource_anomalies['application_id'] == app_id]
            
            pattern = {
                'application_id': app_id,
                'anomaly_count': len(app_anomalies),
                'high_memory_count': app_anomalies['high_memory'].sum(),
                'high_vcores_count': app_anomalies['high_vcores'].sum(),
                'affected_containers': app_anomalies['container_id'].nunique(),
                'time_span': None
            }
            
            if 'timestamp' in app_anomalies.columns:
                time_span = app_anomalies['timestamp'].max() - app_anomalies['timestamp'].min()
                pattern['time_span'] = time_span.total_seconds() / 60  # minutes
            
            patterns.append(pattern)
        
        return patterns
    
    def _analyze_network_patterns(self, network_anomalies: pd.DataFrame) -> List[Dict]:
        """Analyze network-related failure patterns"""
        patterns = []
        
        # Group by IP address if available
        if 'ip_address' in network_anomalies.columns:
            for ip in network_anomalies['ip_address'].dropna().unique():
                ip_anomalies = network_anomalies[network_anomalies['ip_address'] == ip]
                
                pattern = {
                    'ip_address': ip,
                    'anomaly_count': len(ip_anomalies),
                    'affected_components': ip_anomalies['component'].unique().tolist(),
                    'error_types': ip_anomalies['raw_message'].str.extract(r'(connection|timeout|unreachable)')[0].value_counts().to_dict()
                }
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_config_patterns(self, config_anomalies: pd.DataFrame) -> List[Dict]:
        """Analyze configuration error patterns"""
        patterns = []
        
        # Group by error type
        error_patterns = config_anomalies['raw_message'].str.extract(r'(FileNotFoundException|config|property)')
        
        for error_type in error_patterns[0].dropna().unique():
            type_anomalies = config_anomalies[config_anomalies['raw_message'].str.contains(error_type, na=False)]
            
            pattern = {
                'error_type': error_type,
                'anomaly_count': len(type_anomalies),
                'affected_components': type_anomalies['component'].unique().tolist(),
                'common_templates': type_anomalies['template_id'].value_counts().head(3).to_dict()
            }
            patterns.append(pattern)
        
        return patterns
    
    def _find_temporal_clusters(self, anomalies: pd.DataFrame) -> List[Dict]:
        """Find temporal clusters of anomalies"""
        if anomalies.empty or 'timestamp' not in anomalies.columns:
            return []
        
        # Drop rows without timestamp to avoid NaNs in clustering
        anomalies = anomalies.dropna(subset=['timestamp']).copy()
        if anomalies.empty:
            return []
        
        # Convert timestamps to minutes since start
        start_time = anomalies['timestamp'].min()
        anomalies_copy = anomalies.copy()
        anomalies_copy['minutes_since_start'] = (anomalies_copy['timestamp'] - start_time).dt.total_seconds() / 60
        anomalies_copy = anomalies_copy.dropna(subset=['minutes_since_start'])
        if anomalies_copy.empty:
            return []
        
        # Use DBSCAN for temporal clustering
        X = anomalies_copy[['minutes_since_start']].values
        clustering = DBSCAN(eps=5, min_samples=3).fit(X)  # 5-minute windows, min 3 anomalies
        
        clusters = []
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Skip noise
                continue
                
            cluster_mask = clustering.labels_ == cluster_id
            cluster_anomalies = anomalies_copy[cluster_mask]
            
            cluster_info = {
                'cluster_id': cluster_id,
                'size': len(cluster_anomalies),
                'time_span': cluster_anomalies['minutes_since_start'].max() - cluster_anomalies['minutes_since_start'].min(),
                'start_time': cluster_anomalies['timestamp'].min(),
                'end_time': cluster_anomalies['timestamp'].max(),
                'components_involved': cluster_anomalies['component'].unique().tolist(),
                'log_levels': cluster_anomalies['log_level'].value_counts().to_dict(),
                'avg_reconstruction_error': cluster_anomalies['reconstruction_error'].mean()
            }
            clusters.append(cluster_info)
        
        return sorted(clusters, key=lambda x: x['size'], reverse=True)
    
    def generate_root_cause_report(self, df_anomalies: pd.DataFrame) -> Dict:
        """Generate comprehensive root cause analysis report"""
        # Build failure graph
        self.build_failure_graph(df_anomalies)
        
        # Identify patterns
        patterns = self.identify_failure_patterns(df_anomalies)
        
        # Calculate graph metrics
        graph_metrics = {}
        if len(self.failure_graph.nodes()) > 0:
            graph_metrics = {
                'total_nodes': len(self.failure_graph.nodes()),
                'total_edges': len(self.failure_graph.edges()),
                'density': nx.density(self.failure_graph),
                'strongly_connected_components': len(list(nx.strongly_connected_components(self.failure_graph))),
                'weakly_connected_components': len(list(nx.weakly_connected_components(self.failure_graph)))
            }
        
        # Identify critical nodes (high centrality)
        critical_nodes = []
        if len(self.failure_graph.nodes()) > 0:
            centrality = nx.degree_centrality(self.failure_graph)
            critical_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        report = {
            'analysis_timestamp': datetime.now(),
            'total_anomalies': len(df_anomalies[df_anomalies['is_anomaly']]),
            'failure_patterns': patterns,
            'graph_metrics': graph_metrics,
            'critical_nodes': critical_nodes,
            'recommendations': self._generate_recommendations(patterns)
        }
        
        return report
    
    def _generate_recommendations(self, patterns: Dict) -> List[str]:
        """Generate actionable recommendations based on failure patterns"""
        recommendations = []
        
        # Cascade failure recommendations
        if patterns['cascade_failures']:
            recommendations.append("Implement circuit breakers to prevent cascade failures")
            recommendations.append("Add health checks and graceful degradation mechanisms")
        
        # Resource exhaustion recommendations
        if patterns['resource_exhaustion']:
            recommendations.append("Implement dynamic resource allocation and monitoring")
            recommendations.append("Set up memory and CPU usage alerts")
            recommendations.append("Consider container resource limits optimization")
        
        # Network issue recommendations
        if patterns['network_issues']:
            recommendations.append("Implement network retry mechanisms with exponential backoff")
            recommendations.append("Add network connectivity monitoring")
            recommendations.append("Consider network topology optimization")
        
        # Configuration error recommendations
        if patterns['configuration_errors']:
            recommendations.append("Implement configuration validation at startup")
            recommendations.append("Add configuration file existence checks")
            recommendations.append("Consider centralized configuration management")
        
        # Temporal clustering recommendations
        if patterns['temporal_clusters']:
            recommendations.append("Investigate periodic failure patterns")
            recommendations.append("Consider load balancing and traffic distribution")
            recommendations.append("Implement predictive scaling based on temporal patterns")
        
        return recommendations
    
    def visualize_failure_graph(self, max_nodes: int = 50):
        """Visualize the failure propagation graph"""
        if len(self.failure_graph.nodes()) == 0:
            print("No failure graph to visualize")
            return
        
        # Limit nodes for visualization
        if len(self.failure_graph.nodes()) > max_nodes:
            # Select top nodes by degree centrality
            centrality = nx.degree_centrality(self.failure_graph)
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            subgraph = self.failure_graph.subgraph([node for node, _ in top_nodes])
        else:
            subgraph = self.failure_graph
        
        fig = plt.figure(figsize=(14, 9), constrained_layout=True)
        
        # Create layout
        pos = nx.spring_layout(subgraph, k=1, iterations=50, seed=42)
        
        # Draw nodes with different colors for different components
        components = set(nx.get_node_attributes(subgraph, 'component').values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
        component_colors = dict(zip(components, colors))
        
        node_colors = [component_colors.get(subgraph.nodes[node].get('component', 'Unknown'), 'gray') 
                      for node in subgraph.nodes()]
        
        # Draw the graph
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=400, alpha=0.8, 
                              edgecolors='black', linewidths=1.5)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.4, arrows=True, arrowsize=15, 
                              edge_color='gray', width=1.5)
        
        # Add labels for critical nodes
        centrality = nx.degree_centrality(subgraph)
        critical_nodes = {node: f"{subgraph.nodes[node].get('component', 'Unknown')}" 
                         for node, cent in centrality.items() if cent > 0.1}
        nx.draw_networkx_labels(subgraph, pos, critical_nodes, font_size=7, 
                               font_weight='bold', font_color='black')
        
        plt.title("Failure Propagation Graph", fontsize=14, fontweight='bold', pad=20)
        plt.axis('off')
        
        # Add legend with limited entries
        legend_components = list(component_colors.items())[:10]  # Limit to 10 for readability
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=8, label=component)
                          for component, color in legend_components]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=8, 
                  framealpha=0.9, title='Components', title_fontsize=9)
        
        plt.show()
    
    def plot_failure_timeline(self, df_anomalies: pd.DataFrame):
        """Plot failure timeline analysis"""
        anomalies = df_anomalies[df_anomalies['is_anomaly']].copy()
        
        if anomalies.empty or 'timestamp' not in anomalies.columns:
            print("No timestamp data available for timeline analysis")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)
        
        # 1. Anomalies over time
        anomalies_hourly = anomalies.set_index('timestamp').resample('H').size()
        axes[0].plot(anomalies_hourly.index, anomalies_hourly.values, 
                    marker='o', linewidth=2, markersize=6, color='steelblue')
        axes[0].set_title('Anomalies Over Time (Hourly)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Anomaly Count', fontsize=10)
        axes[0].tick_params(axis='both', labelsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(anomalies_hourly.index.min(), anomalies_hourly.index.max())
        
        # 2. Anomalies by component over time (top 5 components only)
        component_timeline = anomalies.pivot_table(
            index='timestamp', columns='component', 
            values='is_anomaly', aggfunc='count', fill_value=0
        ).resample('H').sum()
        
        # Plot only top 5 components by total anomalies
        top_components = component_timeline.sum().nlargest(5).index
        for component in top_components:
            axes[1].plot(component_timeline.index, component_timeline[component], 
                        label=component, marker='o', markersize=4, linewidth=1.5)
        axes[1].set_title('Anomalies by Top 5 Components Over Time', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Anomaly Count', fontsize=10)
        axes[1].legend(loc='upper left', fontsize=8, framealpha=0.9)
        axes[1].tick_params(axis='both', labelsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(component_timeline.index.min(), component_timeline.index.max())
        
        # 3. Reconstruction error over time
        error_timeline = anomalies.set_index('timestamp')['reconstruction_error'].resample('H').mean()
        axes[2].plot(error_timeline.index, error_timeline.values, 
                    color='red', marker='o', linewidth=2, markersize=6)
        axes[2].set_title('Average Reconstruction Error Over Time', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Reconstruction Error', fontsize=10)
        axes[2].set_xlabel('Time', fontsize=10)
        axes[2].tick_params(axis='both', labelsize=9)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(error_timeline.index.min(), error_timeline.index.max())
        
        plt.show()

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    analyzer = RootCauseAnalyzer()
    print("Root Cause Analyzer initialized")
    print("Use with anomaly detection results from anomaly_detector.py")
