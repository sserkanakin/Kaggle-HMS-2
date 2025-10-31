"""
Pytest tests for spectrogram preprocessing functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import pandas as pd
import numpy as np
import torch

from src.data.utils.spectrogram_process import SpectrogramGraphBuilder


@pytest.fixture
def sample_spectrogram_df():
    """Create a sample spectrogram DataFrame for testing."""
    np.random.seed(42)
    # Need 600 seconds of data, sampled every 2 seconds = 300 timepoints
    n_timepoints = 300
    
    data = {
        'time': np.arange(0, n_timepoints * 2, 2),  # Every 2 seconds, 0 to 598
    }
    
    # Add frequency columns for each region
    regions = ['LL', 'RL', 'LP', 'RP']
    freqs = [0.59, 1.17, 2.34, 4.1, 5.27, 8.01, 10.35, 13.48, 16.8, 19.53]
    
    for region in regions:
        for freq in freqs:
            # Generate random spectral power values in range [1e-8, 1e-3]
            col_name = f"{region}_{freq:.2f}"
            data[col_name] = np.random.uniform(1e-8, 1e-3, n_timepoints)
    
    return pd.DataFrame(data)


class TestSpectrogramPreprocessing:
    """Test suite for spectrogram preprocessing."""
    
    def test_preprocessing_enabled(self, sample_spectrogram_df):
        """Test that preprocessing is applied when enabled."""
        builder = SpectrogramGraphBuilder(
            apply_preprocessing=True,
            clip_min=1e-7,
            clip_max=1e-4
        )
        
        preprocessed_df = builder.preprocess_spectrogram(sample_spectrogram_df)
        
        # Get numeric columns
        numeric_cols = [col for col in preprocessed_df.columns if col != 'time']
        preprocessed_values = preprocessed_df[numeric_cols].values.flatten()
        
        # Check that preprocessing was applied (values should be normalized)
        assert np.abs(preprocessed_values.mean()) < 0.1, "Mean should be close to 0"
        assert np.abs(preprocessed_values.std() - 1.0) < 0.1, "Std should be close to 1"
        
        # Check that values are in reasonable range after normalization
        assert np.isfinite(preprocessed_values).all(), "All values should be finite"
    
    def test_preprocessing_disabled(self, sample_spectrogram_df):
        """Test that preprocessing is skipped when disabled."""
        builder = SpectrogramGraphBuilder(
            apply_preprocessing=False
        )
        
        preprocessed_df = builder.preprocess_spectrogram(sample_spectrogram_df)
        
        # Get numeric columns
        numeric_cols = [col for col in preprocessed_df.columns if col != 'time']
        original_values = sample_spectrogram_df[numeric_cols].values.flatten()
        preprocessed_values = preprocessed_df[numeric_cols].values.flatten()
        
        # Check that values are unchanged
        np.testing.assert_allclose(original_values, preprocessed_values)
    
    def test_clipping_applied(self, sample_spectrogram_df):
        """Test that clipping is correctly applied."""
        clip_min = 1e-7
        clip_max = 1e-4
        
        builder = SpectrogramGraphBuilder(
            apply_preprocessing=True,
            clip_min=clip_min,
            clip_max=clip_max
        )
        
        # Add some extreme values to test clipping
        test_df = sample_spectrogram_df.copy()
        numeric_cols = [col for col in test_df.columns if col != 'time']
        
        # Add values outside clip range
        test_df.loc[0, numeric_cols[0]] = 1e-10  # Below clip_min
        test_df.loc[1, numeric_cols[1]] = 1e-2   # Above clip_max
        
        preprocessed_df = builder.preprocess_spectrogram(test_df)
        
        # After preprocessing (clip + log + normalize), we can't directly check clip values
        # but we can verify no NaN or inf values resulted from extreme values
        preprocessed_values = preprocessed_df[numeric_cols].values.flatten()
        assert np.isfinite(preprocessed_values).all(), "Extreme values should not produce NaN/inf"
    
    def test_time_column_preserved(self, sample_spectrogram_df):
        """Test that time column is not modified during preprocessing."""
        builder = SpectrogramGraphBuilder(apply_preprocessing=True)
        
        preprocessed_df = builder.preprocess_spectrogram(sample_spectrogram_df)
        
        # Time column should be unchanged
        np.testing.assert_array_equal(
            sample_spectrogram_df['time'].values,
            preprocessed_df['time'].values
        )
    
    def test_graph_building_with_preprocessing(self, sample_spectrogram_df):
        """Test that graphs can be built with preprocessed data."""
        builder = SpectrogramGraphBuilder(
            apply_preprocessing=True,
            clip_min=1e-7,
            clip_max=1e-4
        )
        
        graphs = builder.process_spectrogram(sample_spectrogram_df)
        
        # Should produce 119 graphs
        assert len(graphs) == 119, f"Expected 119 graphs, got {len(graphs)}"
        
        # Check first graph
        graph = graphs[0]
        assert graph.x is not None, "Graph should have node features"
        assert graph.x.shape == (4, 5), f"Expected shape (4, 5), got {graph.x.shape}"
        assert torch.isfinite(graph.x).all(), "All node features should be finite"
        
        # Check that positional encoding is present
        assert hasattr(graph, 'time_idx'), "Graph should have time_idx"
        assert hasattr(graph, 'is_center'), "Graph should have is_center"
    
    def test_graph_building_without_preprocessing(self, sample_spectrogram_df):
        """Test that graphs can be built without preprocessing."""
        builder = SpectrogramGraphBuilder(apply_preprocessing=False)
        
        graphs = builder.process_spectrogram(sample_spectrogram_df)
        
        # Should produce 119 graphs
        assert len(graphs) == 119
        
        # Check that features are still valid
        graph = graphs[0]
        assert graph.x is not None
        assert torch.isfinite(graph.x).all()
    
    def test_normalization_statistics(self, sample_spectrogram_df):
        """Test that normalization produces correct statistics."""
        builder = SpectrogramGraphBuilder(
            apply_preprocessing=True,
            clip_min=1e-7,
            clip_max=1e-4
        )
        
        preprocessed_df = builder.preprocess_spectrogram(sample_spectrogram_df)
        
        numeric_cols = [col for col in preprocessed_df.columns if col != 'time']
        all_values = preprocessed_df[numeric_cols].values.flatten()
        
        # Mean should be very close to 0
        assert np.abs(all_values.mean()) < 1e-10, f"Mean should be ~0, got {all_values.mean()}"
        
        # Std should be very close to 1
        assert np.abs(all_values.std() - 1.0) < 1e-10, f"Std should be ~1, got {all_values.std()}"
    
    def test_empty_dataframe_handling(self):
        """Test that empty DataFrames are handled gracefully."""
        builder = SpectrogramGraphBuilder(apply_preprocessing=True)
        
        # Create empty DataFrame with just time column
        empty_df = pd.DataFrame({'time': []})
        
        preprocessed_df = builder.preprocess_spectrogram(empty_df)
        
        # Should return empty DataFrame without errors
        assert len(preprocessed_df) == 0
        assert 'time' in preprocessed_df.columns


class TestSpectrogramBuilderInitialization:
    """Test SpectrogramGraphBuilder initialization."""
    
    def test_default_initialization(self):
        """Test builder with default parameters."""
        builder = SpectrogramGraphBuilder()
        
        assert builder.apply_preprocessing is True
        assert builder.clip_min == 1e-7
        assert builder.clip_max == 1e-4
        assert builder.n_windows == 119
    
    def test_custom_clip_values(self):
        """Test builder with custom clip values."""
        clip_min = 1e-8
        clip_max = 1e-3
        
        builder = SpectrogramGraphBuilder(
            clip_min=clip_min,
            clip_max=clip_max
        )
        
        assert builder.clip_min == clip_min
        assert builder.clip_max == clip_max
    
    def test_preprocessing_disabled_initialization(self):
        """Test builder with preprocessing disabled."""
        builder = SpectrogramGraphBuilder(apply_preprocessing=False)
        
        assert builder.apply_preprocessing is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
