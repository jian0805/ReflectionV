EMNLP 2025] Official Code for the Paper Look Again, Think Slowly: Enhancing Visual Reflection in Vision-Language Models

Models: [Reflection-V-7B]() on HuggingFace

## 📁 Structure

```
.
├── evaluate/                     # Evaluation & analysis tools
│   ├── t2v_attn_weight_process.py   # Extract text-to-vision attention weights
│   ├── visual_cutoff_reasoning.py   # Test robustness to visual occlusion
│   └── visual_dependency_measure.py # Quantify visual vs. language reliance
│
└── verl/                         # Training code (built on verl)
    ├── models/                   
    ├── trainer/                  # PPO/GRPO trainers
    ├── workers/                  # Rollout & data collection
    └── protocol.py               # Controller-worker communication
```

### Evaluate

```
# Extract attention weights
python evaluate/t2v_attn_weight_process.py --model_path /path/to/ckpt --image demo.jpg

# Visual cutoff reasoning
python evaluate/visual_cutoff_reasoning.py --model_path /path/to/ckpt

# Dependency measurement
python evaluate/visual_dependency_measure.py --model_path /path/to/ckpt
```

