import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import joblib
import logging

class LSTMAutoencoder:
    """
    LSTM-based Autoencoder for sequence anomaly detection in Spark logs.
    Detects anomalies by learning normal log sequence patterns.
    """
    
    def __init__(self, sequence_length: int = 50, encoding_dim: int = 32):
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.history = None
        
    def build_model(self, input_dim: int) -> Model:
        """Build LSTM Autoencoder architecture"""
        # Encoder
        encoder_inputs = Input(shape=(self.sequence_length, input_dim))
        encoder = LSTM(128, return_sequences=True)(encoder_inputs)
        encoder = Dropout(0.2)(encoder)
        encoder = LSTM(64, return_sequences=True)(encoder)
        encoder = Dropout(0.2)(encoder)
        encoder = LSTM(self.encoding_dim, return_sequences=False)(encoder)
        
        # Decoder
        decoder = RepeatVector(self.sequence_length)(encoder)
        decoder = LSTM(self.encoding_dim, return_sequences=True)(decoder)
        decoder = Dropout(0.2)(decoder)
        decoder = LSTM(64, return_sequences=True)(decoder)
        decoder = Dropout(0.2)(decoder)
        decoder = LSTM(128, return_sequences=True)(decoder)
        decoder = TimeDistributed(Dense(input_dim))(decoder)
        
        # Autoencoder model
        autoencoder = Model(encoder_inputs, decoder)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return autoencoder
    
    def prepare_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM training"""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)
    
    def fit(self, X: np.ndarray, validation_split: float = 0.2, epochs: int = 100, batch_size: int = 32):
        """Train the LSTM Autoencoder"""
        # Normalize data
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_sequences = self.prepare_sequences(X_scaled)
        
        # Build model
        self.model = self.build_model(X.shape[1])
        
        # LR tracker callback to record learning rate each epoch
        class _LearningRateTracker(Callback):
            def __init__(self):
                super().__init__()
                self.lrs = []
            def on_epoch_end(self, epoch, logs=None):
                try:
                    # TF2: optimizer.learning_rate may be a schedule or tensor
                    lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
                except Exception:
                    try:
                        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
                    except Exception:
                        lr = None
                self.lrs.append(lr)

        lr_tracker = _LearningRateTracker()

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            ModelCheckpoint('best_autoencoder.h5', save_best_only=True, monitor='val_loss'),
            lr_tracker
        ]
        
        # Train model
        self.history = self.model.fit(
            X_sequences, X_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Attach recorded learning rates to history for plotting
        try:
            self.history.history['lr'] = lr_tracker.lrs
        except Exception:
            pass

        # Calculate reconstruction threshold
        train_predictions = self.model.predict(X_sequences)
        train_mse = np.mean(np.power(X_sequences - train_predictions, 2), axis=(1, 2))
        self.threshold = np.percentile(train_mse, 95)  # 95th percentile as threshold
        
        logging.info(f"Training completed. Anomaly threshold: {self.threshold:.4f}")
        
    def predict_anomalies(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies in new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Normalize and create sequences
        X_scaled = self.scaler.transform(X)
        X_sequences = self.prepare_sequences(X_scaled)
        
        # Get reconstructions
        reconstructions = self.model.predict(X_sequences)
        
        # Calculate reconstruction errors
        mse = np.mean(np.power(X_sequences - reconstructions, 2), axis=(1, 2))
        
        # Determine anomalies
        anomalies = mse > self.threshold
        
        return anomalies, mse
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        
        # Plot 1: Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=10)
        axes[0].set_ylabel('Loss', fontsize=10)
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Learning Rate
        axes[1].plot(self.history.history['lr'] if 'lr' in self.history.history else [], 
                     linewidth=2, color='orange')
        axes[1].set_title('Learning Rate', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=10)
        axes[1].set_ylabel('Learning Rate', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.show()

class SparkLogAnomalyDetector:
    """
    Complete anomaly detection system for Spark logs combining multiple techniques.
    """
    
    def __init__(self, sequence_length: int = 50):
        self.sequence_length = sequence_length
        self.lstm_autoencoder = LSTMAutoencoder(sequence_length)
        self.feature_encoders = {}
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for anomaly detection"""
        # Select relevant features
        feature_columns = [
            'log_level_numeric', 'component_encoded', 'message_length',
            'hour', 'day_of_week', 'is_weekend', 'is_exception',
            'high_memory', 'high_vcores', 'template_frequency', 'is_rare_template'
        ]
        
        # Handle missing values
        df_features = df[feature_columns].fillna(0)
        
        # Convert boolean columns to int
        bool_columns = ['is_weekend', 'is_exception', 'high_memory', 'high_vcores', 'is_rare_template']
        for col in bool_columns:
            if col in df_features.columns:
                df_features[col] = df_features[col].astype(int)
        
        self.feature_columns = feature_columns
        return df_features.values
    
    def train(self, df: pd.DataFrame, validation_split: float = 0.2, epochs: int = 100):
        """Train the anomaly detection system"""
        logging.info("Preparing features for training...")
        X = self.prepare_features(df)
        
        logging.info(f"Training on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Train LSTM Autoencoder
        self.lstm_autoencoder.fit(X, validation_split=validation_split, epochs=epochs)
        
        logging.info("Training completed successfully!")
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in log data"""
        X = self.prepare_features(df)
        
        # Get anomaly predictions
        anomalies, reconstruction_errors = self.lstm_autoencoder.predict_anomalies(X)
        
        # Add results to dataframe
        result_df = df.copy()
        
        # Handle sequence length offset
        anomaly_results = np.zeros(len(df), dtype=bool)
        error_results = np.zeros(len(df))
        
        if len(anomalies) > 0:
            # Map sequence-level anomalies back to individual log entries
            for i, is_anomaly in enumerate(anomalies):
                start_idx = i
                end_idx = min(i + self.sequence_length, len(df))
                if is_anomaly:
                    anomaly_results[start_idx:end_idx] = True
                error_results[start_idx:end_idx] = reconstruction_errors[i]
        
        result_df['is_anomaly'] = anomaly_results
        result_df['reconstruction_error'] = error_results
        result_df['anomaly_score'] = (error_results - error_results.min()) / (error_results.max() - error_results.min())
        
        return result_df
    
    def analyze_anomalies(self, df_with_anomalies: pd.DataFrame) -> Dict:
        """Analyze detected anomalies"""
        anomalies = df_with_anomalies[df_with_anomalies['is_anomaly']]
        
        analysis = {
            'total_logs': len(df_with_anomalies),
            'anomalous_logs': len(anomalies),
            'anomaly_rate': len(anomalies) / len(df_with_anomalies) * 100,
            'anomaly_by_level': anomalies['log_level'].value_counts().to_dict(),
            'anomaly_by_component': anomalies['component'].value_counts().to_dict(),
            'anomaly_by_hour': anomalies['hour'].value_counts().to_dict() if 'hour' in anomalies.columns else {},
            'top_anomalous_templates': anomalies['template_id'].value_counts().head(10).to_dict(),
            'avg_reconstruction_error': anomalies['reconstruction_error'].mean(),
            'max_reconstruction_error': anomalies['reconstruction_error'].max()
        }
        
        return analysis
    
    def plot_anomaly_analysis(self, df_with_anomalies: pd.DataFrame):
        """Plot anomaly analysis visualizations"""
        anomalies = df_with_anomalies[df_with_anomalies['is_anomaly']]
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
        
        # Anomaly rate by log level
        anomaly_by_level = anomalies['log_level'].value_counts()
        axes[0, 0].bar(anomaly_by_level.index, anomaly_by_level.values, color='steelblue', alpha=0.8)
        axes[0, 0].set_title('Anomalies by Log Level', fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel('Log Level', fontsize=9)
        axes[0, 0].set_ylabel('Count', fontsize=9)
        axes[0, 0].tick_params(axis='both', labelsize=8)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Anomaly rate by component
        anomaly_by_component = anomalies['component'].value_counts().head(10)
        axes[0, 1].barh(range(len(anomaly_by_component)), anomaly_by_component.values, color='coral', alpha=0.8)
        axes[0, 1].set_yticks(range(len(anomaly_by_component)))
        axes[0, 1].set_yticklabels(anomaly_by_component.index, fontsize=8)
        axes[0, 1].set_title('Top 10 Components with Anomalies', fontsize=11, fontweight='bold')
        axes[0, 1].set_xlabel('Count', fontsize=9)
        axes[0, 1].tick_params(axis='x', labelsize=8)
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        axes[0, 1].invert_yaxis()
        
        # Reconstruction error distribution
        axes[0, 2].hist(df_with_anomalies['reconstruction_error'], bins=50, alpha=0.6, 
                       label='All Logs', color='lightblue', edgecolor='black')
        axes[0, 2].hist(anomalies['reconstruction_error'], bins=50, alpha=0.7, 
                       label='Anomalies', color='red', edgecolor='black')
        axes[0, 2].set_title('Reconstruction Error Distribution', fontsize=11, fontweight='bold')
        axes[0, 2].set_xlabel('Reconstruction Error', fontsize=9)
        axes[0, 2].set_ylabel('Frequency', fontsize=9)
        axes[0, 2].legend(fontsize=8)
        axes[0, 2].tick_params(axis='both', labelsize=8)
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # Anomalies over time (if timestamp available)
        if 'timestamp' in df_with_anomalies.columns:
            hourly_anomalies = anomalies.groupby('hour').size()
            axes[1, 0].plot(hourly_anomalies.index, hourly_anomalies.values, 
                          marker='o', linewidth=2, markersize=6, color='green')
            axes[1, 0].set_title('Anomalies by Hour of Day', fontsize=11, fontweight='bold')
            axes[1, 0].set_xlabel('Hour', fontsize=9)
            axes[1, 0].set_ylabel('Anomaly Count', fontsize=9)
            axes[1, 0].tick_params(axis='both', labelsize=8)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Template frequency vs anomalies
        template_anomaly_rate = df_with_anomalies.groupby('template_frequency')['is_anomaly'].mean()
        axes[1, 1].scatter(template_anomaly_rate.index, template_anomaly_rate.values, 
                          alpha=0.6, s=50, color='purple')
        axes[1, 1].set_title('Anomaly Rate vs Template Frequency', fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('Template Frequency', fontsize=9)
        axes[1, 1].set_ylabel('Anomaly Rate', fontsize=9)
        axes[1, 1].tick_params(axis='both', labelsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Anomaly score distribution
        box_data = [df_with_anomalies[~df_with_anomalies['is_anomaly']]['anomaly_score'],
                    df_with_anomalies[df_with_anomalies['is_anomaly']]['anomaly_score']]
        bp = axes[1, 2].boxplot(box_data, labels=['Normal', 'Anomaly'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
            patch.set_facecolor(color)
        axes[1, 2].set_title('Anomaly Score Distribution', fontsize=11, fontweight='bold')
        axes[1, 2].set_ylabel('Anomaly Score', fontsize=9)
        axes[1, 2].tick_params(axis='both', labelsize=8)
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.show()
    
    def save_model(self, model_path: str):
        """Save the trained model"""
        if self.lstm_autoencoder.model is None:
            raise ValueError("No trained model to save")
        
        # Save the Keras model
        self.lstm_autoencoder.model.save(f"{model_path}_autoencoder.h5")
        
        # Save the scaler and other components
        joblib.dump({
            'scaler': self.lstm_autoencoder.scaler,
            'threshold': self.lstm_autoencoder.threshold,
            'sequence_length': self.sequence_length,
            'feature_columns': self.feature_columns
        }, f"{model_path}_components.pkl")
        
        logging.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        # Load Keras model
        self.lstm_autoencoder.model = tf.keras.models.load_model(f"{model_path}_autoencoder.h5")
        
        # Load other components
        components = joblib.load(f"{model_path}_components.pkl")
        self.lstm_autoencoder.scaler = components['scaler']
        self.lstm_autoencoder.threshold = components['threshold']
        self.sequence_length = components['sequence_length']
        self.feature_columns = components['feature_columns']
        
        logging.info(f"Model loaded from {model_path}")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would be used with parsed log data
    print("Spark Log Anomaly Detector initialized")
    print("Use with parsed log DataFrame from log_parser.py")
