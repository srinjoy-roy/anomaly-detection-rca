#!/usr/bin/env python3
"""
Main Pipeline for Spark Cluster Log Anomaly Detection & Root Cause Analysis
Capstone Project - IIT Roorkee Applied Data Science & AI

This script orchestrates the complete pipeline from log parsing to anomaly detection
and root cause analysis for Spark cluster logs.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data.log_parser import SparkLogParser
from models.anomaly_detector import SparkLogAnomalyDetector
from models.root_cause_analyzer import RootCauseAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spark_anomaly_detection.log'),
        logging.StreamHandler()
    ]
)

class SparkAnomalyDetectionPipeline:
    """
    Complete pipeline for Spark log anomaly detection and root cause analysis.
    """
    
    def __init__(self, log_directory: str, output_directory: str = "output", config: dict = None):
        self.log_directory = log_directory
        self.output_directory = output_directory
        self.config = config or {}
        
        # Initialize components with config parameters
        self.parser = SparkLogParser()
        
        # Get model config parameters
        model_config = self.config.get('model', {})
        sequence_length = model_config.get('sequence_length', 50)
        self.detector = SparkLogAnomalyDetector(sequence_length=sequence_length)
        
        # Get RCA config parameters  
        rca_config = self.config.get('root_cause', {})
        time_window = rca_config.get('time_window_minutes', 5)
        self.analyzer = RootCauseAnalyzer(time_window_minutes=time_window)
        
        # Create output directory
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Pipeline initialized with log directory: {log_directory}")
    
    def run_complete_pipeline(self, sample_size: int = None):
        """
        Run the complete anomaly detection pipeline.
        
        Args:
            sample_size: Limit number of log files for testing (None for all)
        """
        logging.info("=" * 60)
        logging.info("SPARK CLUSTER LOG ANOMALY DETECTION PIPELINE")
        logging.info("=" * 60)
        
        try:
            # Step 1: Parse logs
            logging.info("Step 1: Parsing Spark cluster logs...")
            df_logs = self._parse_logs(sample_size)
            
            if df_logs.empty:
                logging.error("No logs parsed. Exiting pipeline.")
                return None
            
            # Step 2: Train or load model
            logging.info("Step 2: Training anomaly detection model...")
            self._train_model(df_logs)
            
            # Step 3: Detect anomalies
            logging.info("Step 3: Detecting anomalies...")
            df_with_anomalies = self._detect_anomalies(df_logs)
            # logging.info(df_with_anomalies)
            
            # Step 4: Root cause analysis
            logging.info("Step 4: Performing root cause analysis...")
            root_cause_report = self._perform_root_cause_analysis(df_with_anomalies)
            
            # Step 5: Generate reports
            logging.info("Step 5: Generating comprehensive reports...")
            self._generate_reports(df_with_anomalies, root_cause_report)
            
            # Step 6: Create visualizations
            logging.info("Step 6: Creating visualizations...")
            self._create_visualizations(df_with_anomalies)
            
            logging.info("Pipeline completed successfully!")
            logging.info(f"Results saved in: {self.output_directory}")
            
            return {
                'parsed_logs': df_logs,
                'anomaly_results': df_with_anomalies,
                'root_cause_report': root_cause_report
            }
            
        except Exception as e:
            logging.error(f"Pipeline failed with error: {str(e)}")
            raise
    
    def _parse_logs(self, sample_size: int = None) -> pd.DataFrame:
        """Parse Spark logs and extract structured data"""
        log_path = Path(self.log_directory)
        
        if sample_size:
            # Get limited number of log files for testing
            log_files = list(log_path.rglob('*.log'))[:sample_size]
            logging.info(f"Processing {len(log_files)} log files (sample)")
            
            all_logs = []
            for log_file in log_files:
                file_logs = self.parser.parse_log_file(str(log_file))
                all_logs.extend(file_logs)
            
            df = pd.DataFrame(all_logs)
            if not df.empty:
                df = self.parser._add_derived_features(df)
        else:
            # Process all logs
            df = self.parser.parse_directory(self.log_directory)
        
        # Save parsed logs
        output_file = os.path.join(self.output_directory, 'parsed_logs.csv')
        df.to_csv(output_file, index=False)
        
        # Save discovered templates
        templates_file = os.path.join(self.output_directory, 'log_templates.json')
        self.parser.save_templates(templates_file)
        
        logging.info(f"Parsed {len(df)} log entries")
        logging.info(f"Discovered {len(self.parser.get_templates())} unique log templates")
        
        return df
    
    def _train_model(self, df_logs: pd.DataFrame):
        """Train the anomaly detection model"""
        # Filter out logs with missing critical features for training
        training_data = df_logs.dropna(subset=['log_level', 'component'])
        
        logging.info(f"Training on {len(training_data)} log entries")
        
        # Get config parameters
        model_config = self.config.get('model', {})
        epochs = model_config.get('epochs', 100)
        validation_split = model_config.get('validation_split', 0.2)
        
        # Train the model
        self.detector.train(training_data, validation_split=validation_split, epochs=epochs)
        
        # Save the trained model
        model_path = os.path.join(self.output_directory, 'trained_model')
        self.detector.save_model(model_path)
        
        logging.info("Model training completed and saved")
    
    def _detect_anomalies(self, df_logs: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in the log data"""
        df_with_anomalies = self.detector.detect_anomalies(df_logs)
        
        # Save anomaly results
        output_file = os.path.join(self.output_directory, 'anomaly_results.csv')
        df_with_anomalies.to_csv(output_file, index=False)
        
        # Log summary statistics
        total_logs = len(df_with_anomalies)
        anomalous_logs = len(df_with_anomalies[df_with_anomalies['is_anomaly']])
        anomaly_rate = (anomalous_logs / total_logs) * 100
        
        logging.info(f"Detected {anomalous_logs} anomalies out of {total_logs} logs ({anomaly_rate:.2f}%)")
        
        return df_with_anomalies
    
    def _perform_root_cause_analysis(self, df_with_anomalies: pd.DataFrame) -> dict:
        """Perform root cause analysis on detected anomalies"""
        report = self.analyzer.generate_root_cause_report(df_with_anomalies)
        
        # Save root cause report
        import json
        report_file = os.path.join(self.output_directory, 'root_cause_report.json')
        
        # Convert datetime objects to strings for JSON serialization
        report_copy = report.copy()
        report_copy['analysis_timestamp'] = report_copy['analysis_timestamp'].isoformat()
        
        with open(report_file, 'w') as f:
            json.dump(report_copy, f, indent=2, default=str)
        
        logging.info("Root cause analysis completed")
        
        return report
    
    def _generate_reports(self, df_with_anomalies: pd.DataFrame, root_cause_report: dict):
        """Generate comprehensive analysis reports"""
        
        # 1. Executive Summary Report
        self._generate_executive_summary(df_with_anomalies, root_cause_report)
        
        # 2. Technical Analysis Report
        self._generate_technical_report(df_with_anomalies, root_cause_report)
        
        # 3. Anomaly Details Report
        self._generate_anomaly_details(df_with_anomalies)
    
    def _generate_executive_summary(self, df_with_anomalies: pd.DataFrame, root_cause_report: dict):
        """Generate executive summary report"""
        anomalies = df_with_anomalies[df_with_anomalies['is_anomaly']]
        
        summary = f"""
# SPARK CLUSTER ANOMALY DETECTION - EXECUTIVE SUMMARY

## Overview
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Log Entries Analyzed**: {len(df_with_anomalies):,}
- **Anomalies Detected**: {len(anomalies):,}
- **Anomaly Rate**: {(len(anomalies)/len(df_with_anomalies)*100):.2f}%

## Key Findings
- **Critical Issues**: {len(anomalies[anomalies['log_level'] == 'ERROR'])} ERROR-level anomalies
- **Resource Issues**: {len(anomalies[(anomalies['high_memory']) | (anomalies['high_vcores'])])} resource-related anomalies
- **Affected Applications**: {anomalies['application_id'].nunique()} unique applications
- **Affected Components**: {', '.join(anomalies['component'].unique())}

## Failure Patterns Identified
"""
        
        patterns = root_cause_report.get('failure_patterns', {})
        
        if patterns.get('cascade_failures'):
            summary += f"- **Cascade Failures**: {len(patterns['cascade_failures'])} detected\n"
        
        if patterns.get('resource_exhaustion'):
            summary += f"- **Resource Exhaustion**: {len(patterns['resource_exhaustion'])} incidents\n"
        
        if patterns.get('network_issues'):
            summary += f"- **Network Issues**: {len(patterns['network_issues'])} incidents\n"
        
        if patterns.get('configuration_errors'):
            summary += f"- **Configuration Errors**: {len(patterns['configuration_errors'])} incidents\n"
        
        if patterns.get('temporal_clusters'):
            summary += f"- **Temporal Clusters**: {len(patterns['temporal_clusters'])} distinct incident periods\n"
        
        summary += f"""
## Recommendations
"""
        for rec in root_cause_report.get('recommendations', []):
            summary += f"- {rec}\n"
        
        summary += f"""
## Impact Assessment
- **High Priority**: {len(anomalies[anomalies['anomaly_score'] > 0.8])} critical anomalies
- **Medium Priority**: {len(anomalies[(anomalies['anomaly_score'] > 0.5) & (anomalies['anomaly_score'] <= 0.8)])} moderate anomalies
- **Low Priority**: {len(anomalies[anomalies['anomaly_score'] <= 0.5])} minor anomalies

---
*Generated by Spark Cluster Anomaly Detection System*
*Capstone Project - IIT Roorkee Applied Data Science & AI*
"""
        
        with open(os.path.join(self.output_directory, 'executive_summary.md'), 'w') as f:
            f.write(summary)
    
    def _generate_technical_report(self, df_with_anomalies: pd.DataFrame, root_cause_report: dict):
        """Generate detailed technical analysis report"""
        analysis = self.detector.analyze_anomalies(df_with_anomalies)
        
        report = f"""
# TECHNICAL ANALYSIS REPORT

## Model Performance
- **Anomaly Detection Model**: LSTM-Autoencoder
- **Sequence Length**: {self.detector.sequence_length}
- **Feature Dimensions**: {len(self.detector.feature_columns)}
- **Reconstruction Threshold**: {self.detector.lstm_autoencoder.threshold:.4f}

## Statistical Analysis
- **Total Logs**: {analysis['total_logs']:,}
- **Anomalous Logs**: {analysis['anomalous_logs']:,}
- **Anomaly Rate**: {analysis['anomaly_rate']:.2f}%
- **Average Reconstruction Error**: {analysis['avg_reconstruction_error']:.4f}
- **Maximum Reconstruction Error**: {analysis['max_reconstruction_error']:.4f}

## Anomaly Distribution by Log Level
"""
        for level, count in analysis['anomaly_by_level'].items():
            report += f"- **{level}**: {count} anomalies\n"
        
        report += "\n## Anomaly Distribution by Component\n"
        for component, count in analysis['anomaly_by_component'].items():
            report += f"- **{component}**: {count} anomalies\n"
        
        report += f"""
## Graph Analysis Metrics
- **Total Failure Nodes**: {root_cause_report['graph_metrics'].get('total_nodes', 0)}
- **Failure Connections**: {root_cause_report['graph_metrics'].get('total_edges', 0)}
- **Graph Density**: {root_cause_report['graph_metrics'].get('density', 0):.4f}
- **Connected Components**: {root_cause_report['graph_metrics'].get('weakly_connected_components', 0)}

## Feature Importance
The following features were used for anomaly detection:
"""
        for feature in self.detector.feature_columns:
            report += f"- {feature}\n"
        
        with open(os.path.join(self.output_directory, 'technical_report.md'), 'w') as f:
            f.write(report)
    
    def _generate_anomaly_details(self, df_with_anomalies: pd.DataFrame):
        """Generate detailed anomaly report"""
        anomalies = df_with_anomalies[df_with_anomalies['is_anomaly']].copy()
        
        # Sort by anomaly score (highest first)
        anomalies = anomalies.sort_values('anomaly_score', ascending=False)
        
        # Select key columns for the report
        columns = ['timestamp', 'log_level', 'component', 'container_id', 
                  'anomaly_score', 'reconstruction_error', 'raw_message']
        
        # Filter columns that exist
        available_columns = [col for col in columns if col in anomalies.columns]
        anomaly_details = anomalies[available_columns]
        
        # Save detailed anomaly report
        anomaly_details.to_csv(os.path.join(self.output_directory, 'anomaly_details.csv'), index=False)
        
        logging.info(f"Generated detailed report for {len(anomalies)} anomalies")
    
    def _create_visualizations(self, df_with_anomalies: pd.DataFrame):
        """Create and save visualizations"""
        try:
            # Anomaly analysis plots
            self.detector.plot_anomaly_analysis(df_with_anomalies)
            
            # Root cause analysis plots
            self.analyzer.visualize_failure_graph()
            self.analyzer.plot_failure_timeline(df_with_anomalies)
            
            # Training history (if available)
            if hasattr(self.detector.lstm_autoencoder, 'history') and self.detector.lstm_autoencoder.history:
                self.detector.lstm_autoencoder.plot_training_history()
            
            logging.info("Visualizations created successfully")
            
        except Exception as e:
            logging.warning(f"Some visualizations could not be created: {str(e)}")

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file"""
    if config_path is None:
        # Default to config/config.yaml relative to project root
        project_root = Path(__file__).resolve().parents[1]
        config_path = project_root / "config" / "config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        logging.warning(f"Could not load config from {config_path}: {e}")
        logging.info("Using default configuration values")
        return {
            'data': {
                'log_directory': 'logs',
                'output_directory': 'output',
                'sample_size': None
            },
            'model': {
                'sequence_length': 50,
                'encoding_dim': 32,
                'epochs': 100
            },
            'root_cause': {
                'time_window_minutes': 5,
                'relationship_threshold': 0.5
            }
        }

def main():
    """Main execution function"""
    # Load configuration from YAML
    config = load_config()
    
    # Extract configuration values
    project_root = Path(__file__).resolve().parents[1]
    LOG_DIRECTORY = str(project_root / config['data']['log_directory'])
    OUTPUT_DIRECTORY = config['data']['output_directory']
    SAMPLE_SIZE = config['data']['sample_size']
    
    logging.info(f"Using configuration: LOG_DIRECTORY={LOG_DIRECTORY}, SAMPLE_SIZE={SAMPLE_SIZE}")
    
    # Initialize and run pipeline with config
    pipeline = SparkAnomalyDetectionPipeline(LOG_DIRECTORY, OUTPUT_DIRECTORY, config)
    
    try:
        results = pipeline.run_complete_pipeline(
            sample_size=SAMPLE_SIZE
        )
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved in: {OUTPUT_DIRECTORY}/")
        print("Key outputs:")
        print("- parsed_logs.csv: Structured log data")
        print("- anomaly_results.csv: Anomaly detection results")
        print("- root_cause_report.json: Root cause analysis")
        print("- executive_summary.md: Executive summary")
        print("- technical_report.md: Technical analysis")
        print("- anomaly_details.csv: Detailed anomaly information")
        
    except Exception as e:
        print(f"\nPipeline execution failed: {str(e)}")
        logging.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
