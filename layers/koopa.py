import torch
import torch.nn as nn
import torch.nn.functional as F

class KPLayer(nn.Module):
    """
    Koopman Operator Layer to model time-series dynamics using Koopman operator.
    """
    def __init__(self, dynamic_dim):
        super(KPLayer, self).__init__()
        self.dynamic_dim = dynamic_dim
        self.K = None  # Koopman operator (linear transition matrix)
        
    def forward(self, z):
        """
        Apply Koopman operator to predict future states of the time series.
        :param z: Time-series data, shape (B, L, E), where B is batch size, L is length, E is the embedding dimension.
        :return: Reconstructed time-series (backcast) and forecasted time-series (forecast).
        """
        B, L, E = z.shape
        assert L > 1, "Input sequence length should be greater than 1"
        
        # Split the input into two sets of snapshots: x (past) and y (future)
        x, y = z[:, :-1], z[:, 1:]
        
        # Compute Koopman operator (K) using least squares solution
        self.K = torch.linalg.lstsq(x, y).solution  # K is shape (B, E, E)
        
        # Predict future state using Koopman operator
        z_pred = torch.bmm(z[:, -1:], self.K)  # B x 1 x E
        
        # Reconstruct the full trajectory using Koopman operator
        z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
        
        return z_rec, z_pred


class KoopmanDecomposition(nn.Module):
    """
    Time series decomposition using Koopman Operator: Trend and Variance Components.
    """
    def __init__(self, dynamic_dim=128):
        super(KoopmanDecomposition, self).__init__()
        self.dynamic_dim = dynamic_dim
        self.kp_layer = KPLayer(dynamic_dim=self.dynamic_dim)

    def forward(self, x):
        """
        Decompose the time series into trend and dynamic components using Koopman operator.
        :param x: Time series data, shape (B, L, E).
        :return: Trend component (time-invariant) and dynamic component (time-variant).
        """
        # Apply Koopman operator layer to the time series
        x_rec, x_pred = self.kp_layer(x)
        
        # Trend component (time-invariant): We take the predicted backcast as the trend
        trend_component = x_rec - x_pred  # Trend is the part that doesn't change over time
        
        # Dynamic component (time-variant): The prediction error is the dynamic part
        dynamic_component = x_pred - trend_component
        
        return trend_component, dynamic_component


# # Example usage of the KoopmanDecomposition class

# # Define random example time series data (batch_size=1, sequence_length=100, embedding_dim=5)
# B, L, E = 1, 100, 5
# time_series_data = torch.randn(B, L, E)

# # Instantiate the model
# model = KoopmanDecomposition(dynamic_dim=128)

# # Apply the decomposition
# trend, dynamic = model(time_series_data)

# print(f"Trend Component: {trend.shape}")
# print(f"Dynamic Component: {dynamic.shape}")
