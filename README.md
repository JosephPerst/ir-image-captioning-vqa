# IR Image Captioning & Visual Question Answering

Computer vision project for generating captions and answering questions about infrared/thermal imagery.

## Projects

| File | Description |
|------|-------------|
| `LLM and Image Encorperated (3).ipynb` | LLM integration with image processing |
| `Model Training FLIR.ipynb` | FLIR thermal image model training |
| `ir_action_caption_generator.py` | Verb-aware action caption generator for IR images |
| `visual_question_answering.ipynb` | VQA system for image understanding |

## Features
- **Action Caption Generation**: Generates human-like captions describing actions in IR imagery
- **YOLO Integration**: Parses YOLO format labels for object detection
- **Pattern-Based Captioning**: Uses extracted patterns from human-written captions
- **Visual Question Answering**: Answer questions about image content

## Detected Objects
- People: walking, standing, crossing, riding, pushing
- Vehicles: driving, parked, turning
- Environments: urban, residential, highway, intersection

## Tech Stack
- Python 3.8+
- PyTorch
- YOLO
- PIL / NumPy
- Hugging Face Transformers

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Generate captions for IR images:
```python
from ir_action_caption_generator import YOLOLabelParser

parser = YOLOLabelParser('path/to/data.yaml')
detections = parser.parse_yolo_label('path/to/label.txt')
```
