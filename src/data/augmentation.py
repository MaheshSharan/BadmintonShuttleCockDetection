"""
Advanced augmentation pipeline for shuttlecock detection.
"""
import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from scipy.ndimage import gaussian_filter

class ShuttlecockAugmentor:
    """Advanced augmentation pipeline with physics-based augmentations."""
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (512, 512),
        sequence_length: int = 16,
        physics_enabled: bool = True,
        noise_std: float = 0.02,
        blur_limit: Tuple[int, int] = (3, 7),
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        motion_blur_p: float = 0.5
    ):
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.physics_enabled = physics_enabled
        self.noise_std = noise_std
        self.blur_limit = blur_limit
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.motion_blur_p = motion_blur_p
        
        # Initialize augmentation pipelines
        self._init_augmentation_pipelines()
        
    def _init_augmentation_pipelines(self):
        """Initialize different augmentation pipelines."""
        # Training augmentations
        self.train_aug = A.Compose([
            A.RandomResizedCrop(
                height=self.img_size[0],
                width=self.img_size[1],
                scale=(0.8, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=self.brightness_limit,
                    contrast_limit=self.contrast_limit
                ),
                A.RandomGamma(),
                A.CLAHE()
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.ISONoise(),
                A.MultiplicativeNoise()
            ], p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=self.blur_limit),
                A.GaussianBlur(blur_limit=self.blur_limit),
                A.MedianBlur(blur_limit=self.blur_limit)
            ], p=self.motion_blur_p),
            A.Normalize(),
            ToTensorV2()
        ])
        
        # Validation augmentations
        self.val_aug = A.Compose([
            A.Resize(height=self.img_size[0], width=self.img_size[1]),
            A.Normalize(),
            ToTensorV2()
        ])
        
        # Physics-based augmentations
        if self.physics_enabled:
            self.physics_aug = PhysicsAugmentor(
                img_size=self.img_size,
                sequence_length=self.sequence_length
            )
        
    def __call__(
        self,
        frames: np.ndarray,
        labels: np.ndarray,
        is_train: bool = True
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Apply augmentation pipeline to sequence."""
        # Apply physics-based augmentations first
        if self.physics_enabled and is_train:
            frames, labels = self.physics_aug(frames, labels)
            
        # Apply frame-wise augmentations
        augmented_frames = []
        for frame in frames:
            aug = self.train_aug if is_train else self.val_aug
            augmented = aug(image=frame)['image']
            augmented_frames.append(augmented)
            
        return torch.stack(augmented_frames), labels
        
class PhysicsAugmentor:
    """Physics-based augmentations for realistic motion effects."""
    
    def __init__(
        self,
        img_size: Tuple[int, int],
        sequence_length: int,
        gravity: float = 9.81,
        air_resistance: float = 0.1,
        max_initial_velocity: float = 20.0
    ):
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.gravity = gravity
        self.air_resistance = air_resistance
        self.max_initial_velocity = max_initial_velocity
        
    def __call__(
        self,
        frames: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply physics-based augmentations."""
        # Generate realistic trajectory
        trajectory = self._generate_trajectory(labels)
        
        # Apply motion blur based on velocity
        frames = self._apply_motion_blur(frames, trajectory)
        
        # Update labels with new trajectory
        labels = self._update_labels(labels, trajectory)
        
        return frames, labels
        
    def _generate_trajectory(self, labels: np.ndarray) -> np.ndarray:
        """Generate physically plausible trajectory."""
        # Extract initial position and velocity
        pos = labels[0, :2]  # [x, y]
        vel = np.random.uniform(
            -self.max_initial_velocity,
            self.max_initial_velocity,
            size=2
        )
        
        trajectory = np.zeros((self.sequence_length, 2))
        trajectory[0] = pos
        
        # Simulate motion with physics
        dt = 1/30  # Assuming 30 fps
        for i in range(1, self.sequence_length):
            # Update velocity
            vel[1] -= self.gravity * dt
            vel *= (1 - self.air_resistance * dt)
            
            # Update position
            pos = pos + vel * dt
            
            # Bounce off boundaries
            for j in range(2):
                if pos[j] < 0:
                    pos[j] = 0
                    vel[j] *= -0.8  # Energy loss
                elif pos[j] >= self.img_size[j]:
                    pos[j] = self.img_size[j] - 1
                    vel[j] *= -0.8
                    
            trajectory[i] = pos
            
        return trajectory
        
    def _apply_motion_blur(
        self,
        frames: np.ndarray,
        trajectory: np.ndarray
    ) -> np.ndarray:
        """Apply motion blur based on velocity."""
        blurred_frames = frames.copy()
        
        for i in range(self.sequence_length):
            # Calculate velocity magnitude
            if i > 0:
                velocity = trajectory[i] - trajectory[i-1]
                speed = np.linalg.norm(velocity)
                
                # Apply directional blur
                if speed > 1:
                    angle = np.arctan2(velocity[1], velocity[0])
                    kernel_size = int(min(speed * 2, 15))
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                        
                    kernel = self._get_motion_blur_kernel(
                        kernel_size,
                        angle
                    )
                    blurred_frames[i] = cv2.filter2D(
                        frames[i],
                        -1,
                        kernel
                    )
                    
        return blurred_frames
        
    def _get_motion_blur_kernel(
        self,
        kernel_size: int,
        angle: float
    ) -> np.ndarray:
        """Generate directional motion blur kernel."""
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        # Create line at specified angle
        for i in range(kernel_size):
            offset = i - center
            x = int(center + offset * np.cos(angle))
            y = int(center + offset * np.sin(angle))
            
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
                
        # Normalize kernel
        return kernel / kernel.sum()
        
    def _update_labels(
        self,
        labels: np.ndarray,
        trajectory: np.ndarray
    ) -> np.ndarray:
        """Update labels with new trajectory."""
        new_labels = labels.copy()
        new_labels[:, :2] = trajectory
        return new_labels
        
class EnvironmentalAugmentor:
    """Environmental condition augmentations."""
    
    def __init__(
        self,
        shadow_intensity: float = 0.5,
        light_intensity: float = 0.3,
        noise_std: float = 0.1
    ):
        self.shadow_intensity = shadow_intensity
        self.light_intensity = light_intensity
        self.noise_std = noise_std
        
    def apply_shadow(self, image: np.ndarray) -> np.ndarray:
        """Apply random shadow effects."""
        h, w = image.shape[:2]
        
        # Generate random shadow polygon
        points = np.random.randint(0, max(h, w), size=(4, 2))
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(mask, [points], 1)
        
        # Blur shadow edges
        mask = gaussian_filter(mask, sigma=10)
        
        # Apply shadow
        image = image.astype(np.float32)
        image *= (1 - self.shadow_intensity * mask[..., np.newaxis])
        
        return np.clip(image, 0, 255).astype(np.uint8)
        
    def apply_lighting(self, image: np.ndarray) -> np.ndarray:
        """Apply random lighting effects."""
        h, w = image.shape[:2]
        
        # Generate random light source
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        
        # Create radial gradient
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r = r / np.max(r)
        
        # Apply lighting
        image = image.astype(np.float32)
        image *= (1 + self.light_intensity * (1 - r)[..., np.newaxis])
        
        return np.clip(image, 0, 255).astype(np.uint8)
        
    def apply_weather(self, image: np.ndarray) -> np.ndarray:
        """Apply weather effects (rain, snow, etc.)."""
        h, w = image.shape[:2]
        
        # Add noise for rain/snow effect
        noise = np.random.normal(0, self.noise_std, image.shape)
        image = image.astype(np.float32)
        image += noise * 255
        
        return np.clip(image, 0, 255).astype(np.uint8)
