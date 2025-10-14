import re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import logging
from pathlib import Path

class SparkLogParser:
    """
    Advanced Spark log parser using Drain algorithm for template extraction
    and structured log processing for anomaly detection.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.template_miner = self._initialize_drain(config_path)
        self.log_patterns = self._compile_patterns()
        self.parsed_logs = []
        
    def _initialize_drain(self, config_path: Optional[str]) -> TemplateMiner:
        """Initialize Drain algorithm for log template mining"""
        if config_path:
            config = TemplateMinerConfig()
            config.load(config_path)
        else:
            config = TemplateMinerConfig()
            config.profiling_enabled = False
            config.snapshot_interval_minutes = 5
            
        return TemplateMiner(config=config)
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for Spark log parsing"""
        patterns = {
            'timestamp': re.compile(r'(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})'),
            'log_level': re.compile(r'\b(DEBUG|INFO|WARN|ERROR|FATAL)\b'),
            # General component extractor: timestamp + level + Component:
            'component_general': re.compile(r'^\s*\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s+(?:DEBUG|INFO|WARN|ERROR|FATAL)\s+([A-Za-z0-9._$\-]+):'),
            # Fallback: any token like ClassName: later in the line (used if general fails)
            'component_fallback': re.compile(r'\b([A-Za-z0-9._$\-]+):'),
            # Legacy shortlist (kept as hint only)
            'component_legacy': re.compile(r'\b(ApplicationMaster|YarnAllocator|SecurityManager|RMProxy|YarnRMClient)\b'),
            'container_id': re.compile(r'container_\d+_\d+_\d+_\d+'),
            'application_id': re.compile(r'application_\d+_\d+'),
            'memory_allocation': re.compile(r'<memory:(\d+), vCores:(\d+)>'),
            'exception': re.compile(r'(Exception|Error): (.+)'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'port': re.compile(r':(\d{4,5})\b')
        }
        return patterns
    
    def parse_log_line(self, log_line: str) -> Dict:
        """Parse a single log line into structured format"""
        parsed = {
            'raw_message': log_line.strip(),
            'timestamp': None,
            'log_level': 'UNKNOWN',
            'component': 'UNKNOWN',
            'template_id': None,
            'parameters': [],
            'container_id': None,
            'application_id': None,
            'memory': None,
            'vcores': None,
            'ip_address': None,
            'port': None,
            'is_exception': False,
            'exception_type': None,
            'message_length': len(log_line)
        }
        
        # Extract timestamp
        timestamp_match = self.log_patterns['timestamp'].search(log_line)
        if timestamp_match:
            try:
                parsed['timestamp'] = datetime.strptime(timestamp_match.group(1), '%y/%m/%d %H:%M:%S')
            except ValueError:
                parsed['timestamp'] = None
        
        # Extract log level
        level_match = self.log_patterns['log_level'].search(log_line)
        if level_match:
            parsed['log_level'] = level_match.group(1)
        
        # Extract component (robust)
        component = None
        # 1) Try general "timestamp level Component:" pattern
        m = self.log_patterns['component_general'].search(log_line)
        if m:
            component = m.group(1)
        else:
            # 2) Try legacy shortlist as hint
            m2 = self.log_patterns['component_legacy'].search(log_line)
            if m2:
                component = m2.group(1)
            else:
                # 3) Fallback: first token ending with ':' after level token if present
                level_m = self.log_patterns['log_level'].search(log_line)
                search_str = log_line[level_m.end():] if level_m else log_line
                m3 = self.log_patterns['component_fallback'].search(search_str)
                if m3:
                    token = m3.group(1)
                    # Avoid capturing the level token itself or numeric-only tokens
                    if token not in {"DEBUG","INFO","WARN","ERROR","FATAL"} and not token.isdigit():
                        component = token
        parsed['component'] = component or 'UNKNOWN'
        
        # Extract container and application IDs
        container_match = self.log_patterns['container_id'].search(log_line)
        if container_match:
            parsed['container_id'] = container_match.group(0)
            
        app_match = self.log_patterns['application_id'].search(log_line)
        if app_match:
            parsed['application_id'] = app_match.group(0)
        
        # Extract resource allocation
        memory_match = self.log_patterns['memory_allocation'].search(log_line)
        if memory_match:
            parsed['memory'] = int(memory_match.group(1))
            parsed['vcores'] = int(memory_match.group(2))
        
        # Extract network information
        ip_match = self.log_patterns['ip_address'].search(log_line)
        if ip_match:
            parsed['ip_address'] = ip_match.group(0)
            
        port_match = self.log_patterns['port'].search(log_line)
        if port_match:
            parsed['port'] = int(port_match.group(1))
        
        # Check for exceptions
        exception_match = self.log_patterns['exception'].search(log_line)
        if exception_match or 'ERROR' in log_line or 'Exception' in log_line:
            parsed['is_exception'] = True
            if exception_match:
                parsed['exception_type'] = exception_match.group(1)
        
        # Use Drain for template extraction
        result = self.template_miner.add_log_message(log_line)
        if result:
            # Drain3 versions differ in keys. Prefer cluster_id, fallback to cluster object
            template_id = result.get('cluster_id')
            if template_id is None and 'cluster' in result and hasattr(result['cluster'], 'cluster_id'):
                template_id = getattr(result['cluster'], 'cluster_id')
            parsed['template_id'] = template_id

            # Parameters key can vary: 'parameter_list' or 'parameters'
            params = result.get('parameter_list', result.get('parameters', []))
            parsed['parameters'] = params if isinstance(params, list) else []
        
        return parsed
    
    def parse_log_file(self, file_path: str) -> List[Dict]:
        """Parse entire log file"""
        parsed_logs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                for line_num, line in enumerate(file, 1):
                    if line.strip():  # Skip empty lines
                        parsed_log = self.parse_log_line(line)
                        parsed_log['file_path'] = file_path
                        parsed_log['line_number'] = line_num
                        parsed_logs.append(parsed_log)
        except Exception as e:
            logging.error(f"Error parsing file {file_path}: {str(e)}")
            
        return parsed_logs
    
    def parse_directory(self, directory_path: str) -> pd.DataFrame:
        """Parse all log files in a directory"""
        all_logs = []
        directory = Path(directory_path)
        
        # Find all .log files recursively
        log_files = list(directory.rglob('*.log'))
        
        logging.info(f"Found {len(log_files)} log files to process")
        
        for log_file in log_files:
            logging.info(f"Processing: {log_file}")
            file_logs = self.parse_log_file(str(log_file))
            all_logs.extend(file_logs)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_logs)
        
        # Add derived features
        if not df.empty:
            df = self._add_derived_features(df)
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for anomaly detection"""
        # Time-based features
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Log level encoding
        level_mapping = {'DEBUG': 0, 'INFO': 1, 'WARN': 2, 'ERROR': 3, 'FATAL': 4, 'UNKNOWN': -1}
        df['log_level_numeric'] = df['log_level'].map(level_mapping)
        
        # Component encoding
        df['component_encoded'] = pd.Categorical(df['component']).codes
        
        # Resource utilization flags
        df['high_memory'] = (df['memory'] > 20000).fillna(False)
        df['high_vcores'] = (df['vcores'] > 4).fillna(False)
        
        # Template frequency (anomaly indicator)
        template_counts = df['template_id'].value_counts()
        df['template_frequency'] = df['template_id'].map(template_counts)
        df['is_rare_template'] = df['template_frequency'] < 5
        
        return df
    
    def get_templates(self) -> Dict:
        """Get discovered log templates"""
        templates = {}
        clusters_obj = getattr(self.template_miner.drain, 'clusters', None)
        if clusters_obj is None:
            return templates

        # clusters may be a dict or an iterable of cluster objects depending on drain3 version
        try:
            iterator = clusters_obj.items()  # dict-like
        except AttributeError:
            iterator = ((getattr(c, 'cluster_id', None), c) for c in list(clusters_obj))

        for cluster_id, cluster in iterator:
            try:
                templates[cluster_id] = {
                    'template': cluster.get_template() if hasattr(cluster, 'get_template') else None,
                    'size': getattr(cluster, 'size', None),
                    'log_template_tokens': getattr(cluster, 'log_template_tokens', None)
                }
            except Exception:
                continue
        return templates
    
    def save_templates(self, output_path: str):
        """Save discovered templates to file"""
        import json
        templates = self.get_templates()
        with open(output_path, 'w') as f:
            json.dump(templates, f, indent=2, default=str)

if __name__ == "__main__":
    # Example usage
    parser = SparkLogParser()
    
    # Parse sample log directory
    from pathlib import Path as _Path
    log_dir = str(_Path(__file__).resolve().parents[2] / "logs" / "Spark")
    df = parser.parse_directory(log_dir)
    
    print(f"Parsed {len(df)} log entries")
    print(f"Discovered {len(parser.get_templates())} log templates")
    
    # Save results
    df.to_csv('parsed_spark_logs.csv', index=False)
    parser.save_templates('log_templates.json')
