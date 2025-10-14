# Spark Cluster Log Anomaly Detection & Root Cause Analysis

## 🎯 Project Overview
A sophisticated machine learning system for detecting anomalies in Spark cluster logs and performing automated root cause analysis to prevent system failures. This capstone project demonstrates advanced big data engineering and deep learning techniques.

## 🏗️ System Architecture

### Core Components
1. **Log Parser & Preprocessor** - Structured log extraction using Drain3 algorithm
2. **Feature Engineering Pipeline** - 11 statistical and temporal features
3. **Anomaly Detection Engine** - LSTM-Autoencoder (50-step sequences, 32-dim encoding)
4. **Root Cause Analysis** - Graph-based failure correlation with DBSCAN clustering
5. **Failure Pattern Detection** - 5 pattern types (cascade, resource, network, config, temporal)
6. **Interactive Dashboard** - Streamlit-based visualization with 6 analysis sections

## 📊 Dataset
- **Source**: LogHub Spark cluster logs
- **Size**: 150+ Spark applications with container logs
- **Log Types**: ApplicationMaster, Container, YARN, Resource Manager
- **Time Range**: Production cluster logs with various failure scenarios

## 🔧 Technical Stack
- **Deep Learning**: TensorFlow/Keras (LSTM-Autoencoder)
- **Log Processing**: Drain3 template mining, regex parsing
- **Machine Learning**: scikit-learn (DBSCAN, StandardScaler, LabelEncoder)
- **Graph Analysis**: NetworkX (failure propagation graphs)
- **Visualization**: Streamlit, Matplotlib, Seaborn, Plotly
- **Data Processing**: Pandas, NumPy

## 🚀 Key Features
- **LSTM-Autoencoder**: Sequence-based anomaly detection with reconstruction error analysis
- **4 Failure Pattern Types**: Cascade failures, resource exhaustion, network issues, configuration errors
- **Graph-Based Analysis**: Failure propagation graphs with centrality metrics
- **Temporal Clustering**: DBSCAN-based time-window grouping (identifies WHEN incidents occurred)
- **Interactive Dashboard**: 6 analysis sections with real-time metrics
- **Comprehensive Reports**: Executive summary, technical report, anomaly details

## 📈 Results Achieved
- **Total Logs Analyzed**: 68,927 log entries
- **Anomalies Detected**: 4,960 (7.2% anomaly rate)
- **Temporal Clusters**: 3 distinct incident periods identified
- **Network Issues**: 10 nodes with connection problems detected
- **Resource Issues**: 2 applications with memory pressure
- **Model Performance**: Threshold 2.3230, converged in 17 epochs

## 📁 Project Structure
```
AnomalyDetection/
├── src/
│   ├── data/
│   │   ├── log_parser.py           # Drain3-based log parsing
│   ├── models/
│   │   ├── anomaly_detector.py     # LSTM-Autoencoder implementation
│   │   └── root_cause_analyzer.py  # Graph analysis & pattern detection
│   ├── dashboard/
│   │   └── streamlit_app.py        # Interactive dashboard (6 sections)
│   └── main_pipeline.py            # End-to-end pipeline orchestration
├── config/
│   └── config.yaml                 # Configuration parameters
├── logs/                           # Input: Spark cluster logs
├── output/
│   ├── trained_model               # Saved model files (.h5, .pkl)
│   ├── anomaly_results.csv         # All logs with anomaly flags
│   ├── anomaly_details.csv         # Top anomalies with details
│   ├── root_cause_report.json      # Failure patterns & analysis
│   ├── executive_summary.md        # Executive summary report
│   ├── technical_report.md         # Technical analysis report
│   └── log_templates.json          # Discovered log templates
├── activate_env.sh                 # Virtual environment activation
└── requirements.txt                # Python dependencies
```

## 🏃‍♂️ Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
source activate_env.sh

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
# Execute end-to-end pipeline (parsing → training → detection → analysis)
python src/main_pipeline.py
```

**Pipeline Steps:**
1. Parse logs using Drain3 (extracts templates)
2. Engineer 11 features (temporal, resource, categorical)
3. Train LSTM-Autoencoder (50-step sequences, early stopping)
4. Detect anomalies (reconstruction error > threshold)
5. Perform root cause analysis (5 pattern types)
6. Generate reports (executive summary, technical report, CSV files)

**Expected Runtime:** ~35 minutes (68K logs, 17 epochs)

### 3. Launch Interactive Dashboard
```bash
streamlit run src/dashboard/streamlit_app.py
```

**Dashboard Sections:**
- 📊 System Overview (key metrics)
- 📈 Anomaly Timeline (time series)
- 🔍 Root Cause Analysis (failure patterns + recommendations)
- 🔧 Component Analysis (distribution charts)
- 🔥 Anomaly Heatmap (temporal patterns)
- 📋 Detailed Anomaly Log View (data table)

### 4. View Generated Reports
```bash
# Executive summary
cat output/executive_summary.md

# Technical report
cat output/technical_report.md

# Root cause analysis
cat output/root_cause_report.json

# Anomaly details
head output/anomaly_details.csv
```

## 📊 Output Files

| File | Description | Format |
|------|-------------|--------|
| `anomaly_results.csv` | All 68K logs with anomaly flags | CSV |
| `anomaly_details.csv` | Top 100 anomalies sorted by score | CSV |
| `root_cause_report.json` | 5 failure patterns + graph metrics | JSON |
| `executive_summary.md` | High-level findings & recommendations | Markdown |
| `technical_report.md` | Model performance & statistics | Markdown |
| `log_templates.json` | Drain3 discovered templates | JSON |
| `*.h5` | LSTM-Autoencoder weights | Keras H5 |
| `*.pkl` | Scaler & encoders | Pickle |

## 🔧 Configuration

Edit `config/config.yaml` to customize:
```yaml
data:
  log_directory: "logs"           # Input log directory
  sample_size: null               # Process all logs (or set limit)

model:
  sequence_length: 50             # LSTM sequence length
  encoding_dim: 32                # Autoencoder bottleneck size
  epochs: 100                     # Max epochs (early stopping enabled)
  batch_size: 64                  # Training batch size

root_cause:
  time_window_minutes: 5          # Temporal clustering window
  min_cluster_size: 3             # Min anomalies per cluster
```

## 🎯 Model Architecture

### LSTM-Autoencoder
```
Input: [batch, 50, 11]  ← 50 time steps, 11 features
   ↓
Encoder LSTM: 64 units → 32 units
   ↓
Bottleneck: [batch, 32]  ← Compressed representation
   ↓
Decoder LSTM: 32 units → 64 units
   ↓
Output: [batch, 50, 11]  ← Reconstructed sequence

Loss: Mean Squared Error
Threshold: 95th percentile (2.3230)
```

### Features (11 total)
1. `log_level_encoded` - ERROR=2, WARN=1, INFO=0
2. `component_encoded` - Spark component ID
3. `template_id` - Drain3 template ID
4. `hour` - Hour of day (0-23)
5. `minute` - Minute (0-59)
6. `day_of_week` - Day (0-6)
7. `memory_mb` - Memory allocation
8. `vcores` - CPU cores
9. `container_id_encoded` - Container ID
10. `application_id_encoded` - Application ID
11. `ip_address_encoded` - IP address

---
**Author**: Srinjoy Roy  
**Course**: Post Graduate Certification in Applied Data Science and AI (IIT Roorkee)  
**Institution**: IIT Roorkee (CloudxLab)  
**Project Type**: Capstone Project - Spark Log Anomaly Detection & Root Cause Analysis  
**Dataset**: LogHub Spark Cluster Logs 
