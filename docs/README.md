# Badminton Shuttlecock Detection Documentation

This directory contains comprehensive documentation for the Badminton Shuttlecock Detection System.

## Documentation Structure

- `architecture.md`: Detailed system architecture with component diagrams
  - System pipeline flows
  - Component interactions
  - Key parameters
  - Optimization strategies

## Viewing Diagrams

The architecture documentation uses Mermaid diagrams. To view them properly:
1. Use a Markdown viewer that supports Mermaid (like GitHub, VS Code with Mermaid extension)
2. Or visit [Mermaid Live Editor](https://mermaid.live/) and paste the diagram code

## Research Paper References

The documentation in this folder will be particularly useful for writing research papers, focusing on:
1. Novel architecture design
2. Physics-informed deep learning
3. Performance optimization techniques
4. Real-time processing strategies


The Training Flow:

Data Flow:

Raw Video → VideoProcessor → DataGenerator → Training Pipeline
                ↓
        Augmentation Pipeline


Forward Pass:

Input Frames → CSPBackbone → FPN → Detection Head
                                    ↓
                            Physics Validation
                                    ↓
                            Tracking System


Loss Computation:

Detection Loss + Tracking Loss + Physics-Based Loss

Core Detection Model:
A highly optimized single model that can:
Detect shuttlecocks in real-time (60+ FPS)
Process multiple frames simultaneously
Output precise 2D/3D positions
Predict trajectories
Model Components & Capabilities:
Final Model Structure:
├── Detection Engine
│   ├── CSPDarknet Backbone (Feature Extraction)
│   ├── FPN (Multi-scale Features)
│   └── Custom Detection Head
│
├── Physics Engine
│   ├── Trajectory Validator
│   ├── PINN (Physics-Informed Neural Network)
│   └── Motion Models
│
├── Tracking System
│   ├── Modified DeepSORT
│   ├── Trajectory Predictor
│   └── Kalman Filter
│
└── Optimization Layer
    ├── TensorRT Optimizations
    ├── Quantization
    └── Model Pruning



Preprocessed Data Structure -> metadata.json -> Frame Sequences -> Training
└── frames/                 └── sequences     └── JPG files
└── annotations/            └── frame_count   └── CSV annotations