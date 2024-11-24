"""
Trajectory prediction module combining LSTM, Transformer attention, and Kalman filtering.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal self-attention to the sequence."""
        attended, _ = self.attention(x, x, x)
        return self.norm(x + attended)

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.hidden_dim = hidden_dim
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Process sequence through LSTM."""
        return self.lstm(x, hidden)

class BadmintonMotionModel:
    """Physics-based motion model for badminton shuttlecock."""
    
    def __init__(self, dt: float = 1/30):  # Assuming 30 fps
        self.dt = dt
        self.g = 9.81  # Gravity
        self.drag_coeff = 0.6  # Drag coefficient for shuttlecock
        
    def predict_state(self, state: np.ndarray) -> np.ndarray:
        """
        Predict next state using physics model.
        state: [x, y, vx, vy, ax, ay]
        """
        x, y, vx, vy, ax, ay = state
        
        # Update velocities with acceleration
        vx_new = vx + ax * self.dt
        vy_new = vy + (ay - self.g) * self.dt
        
        # Apply drag force
        v = np.sqrt(vx_new**2 + vy_new**2)
        drag = self.drag_coeff * v**2
        
        # Update positions
        x_new = x + vx_new * self.dt
        y_new = y + vy_new * self.dt
        
        return np.array([x_new, y_new, vx_new, vy_new, ax, ay])

class BadmintonKalmanFilter:
    def __init__(self, dt: float = 1/30):
        self.motion_model = BadmintonMotionModel(dt)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Process noise
        self.Q = np.eye(6) * 0.1
        
        # Measurement matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        
        # Measurement noise
        self.R = np.eye(2) * 1.0
        
        self.P = np.eye(6) * 100  # Initial state covariance
        self.state = None
        
    def initialize(self, measurement: np.ndarray):
        """Initialize state from first measurement."""
        self.state = np.zeros(6)
        self.state[:2] = measurement
        
    def predict(self) -> np.ndarray:
        """Predict next state using physics-based motion model."""
        if self.state is None:
            raise ValueError("Filter not initialized")
            
        # Physics-based prediction
        self.state = self.motion_model.predict_state(self.state)
        
        # Update covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state[:2]  # Return predicted position
        
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update state with new measurement."""
        if self.state is None:
            self.initialize(measurement)
            return measurement
            
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        y = measurement - self.H @ self.state
        self.state = self.state + K @ y
        
        # Update covariance
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        return self.state[:2]  # Return filtered position

class TrajectoryPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8
    ):
        super().__init__()
        
        # LSTM for sequence processing
        self.lstm = TrajectoryLSTM(input_dim, hidden_dim, num_layers)
        
        # Temporal attention
        self.attention = TemporalAttention(hidden_dim * 2, num_heads)  # *2 for bidirectional
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Kalman filter for each track
        self.kalman_filters = {}
        
    def forward(
        self,
        sequence: torch.Tensor,
        track_ids: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict future trajectory points.
        
        Args:
            sequence: [batch_size, seq_len, 2] tensor of positions
            track_ids: Optional list of track IDs for Kalman filtering
            
        Returns:
            predictions: Next position predictions
            confidence: Prediction confidence scores
        """
        batch_size, seq_len, _ = sequence.shape
        
        # 1. Process through LSTM
        lstm_out, _ = self.lstm(sequence)
        
        # 2. Apply temporal attention
        attended = self.attention(lstm_out)
        
        # 3. Generate predictions
        predictions = self.predictor(attended[:, -1])  # Use last timestep
        
        # 4. Apply Kalman filtering if track_ids provided
        if track_ids is not None:
            filtered_predictions = []
            for i, track_id in enumerate(track_ids):
                if track_id not in self.kalman_filters:
                    self.kalman_filters[track_id] = BadmintonKalmanFilter()
                
                # Convert to numpy for Kalman filter
                track_sequence = sequence[i].detach().cpu().numpy()
                kf = self.kalman_filters[track_id]
                
                # Initialize or update Kalman filter
                if len(track_sequence) == 1:
                    kf.initialize(track_sequence[0])
                    filtered_pos = track_sequence[0]
                else:
                    kf.update(track_sequence[-1])
                    filtered_pos = kf.predict()
                
                filtered_predictions.append(filtered_pos)
            
            # Combine ML predictions with Kalman filter
            filtered_predictions = torch.tensor(
                filtered_predictions,
                device=predictions.device,
                dtype=predictions.dtype
            )
            
            # Weighted average based on prediction confidence
            confidence = torch.sigmoid(self.predictor(attended[:, -1]))
            predictions = confidence * predictions + (1 - confidence) * filtered_predictions
        else:
            confidence = torch.sigmoid(self.predictor(attended[:, -1]))
        
        return predictions, confidence
