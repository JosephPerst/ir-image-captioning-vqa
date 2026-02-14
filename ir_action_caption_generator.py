"""
IR Image Action Caption Generator - Verb-Aware (Hybrid Approach)
Based on human-written caption patterns from FLIR dataset

Extracted Patterns:
- Person actions: walking, standing, crossing, riding, pushing
- Vehicle actions: driving, parked, turning
- Environments: urban, residential, highway, intersection, downtown, etc.
- Locations: sidewalk, road, street, curb, shoulder, etc.
"""

import numpy as np
import random
from pathlib import Path
import json
import yaml
from PIL import Image
from collections import Counter

# ============================================================================
# PART 1: YOLO Label Parser
# ============================================================================

class YOLOLabelParser:
    """Parse YOLO format labels into detections"""
    
    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config['names']
        self.train_path = Path(self.config['train'])
        self.val_path = Path(self.config['val'])
        self.test_path = Path(self.config.get('test', ''))
        
        self.train_label_path = self.train_path.parent.parent / 'train' / 'labels'
        self.val_label_path = self.val_path.parent.parent / 'valid' / 'labels'
        
        print(f"Classes: {self.class_names}")
    
    def parse_yolo_label(self, label_path, img_width=640, img_height=640):
        detections = []
        
        if not Path(label_path).exists():
            return detections
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            x_center_abs = x_center * img_width
            y_center_abs = y_center * img_height
            width_abs = width * img_width
            height_abs = height * img_height
            
            x1 = x_center_abs - width_abs / 2
            y1 = y_center_abs - height_abs / 2
            x2 = x_center_abs + width_abs / 2
            y2 = y_center_abs + height_abs / 2
            
            detections.append({
                'class': self.class_names[class_id],
                'class_id': class_id,
                'bbox': [x1, y1, x2, y2],
                'x_norm': x_center,
                'y_norm': y_center,
                'width_norm': width,
                'height_norm': height,
                'conf': 1.0
            })
        
        return detections
    
    def get_image_list(self, split='train'):
        if split == 'train':
            img_path = self.train_path
        elif split == 'val':
            img_path = self.val_path
        else:
            img_path = self.test_path
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(Path(img_path).glob(f'*{ext}')))
            image_files.extend(list(Path(img_path).glob(f'*{ext.upper()}')))
        
        return sorted(image_files)
    
    def get_label_path(self, image_path, split='train'):
        if split == 'train':
            label_base = self.train_label_path
        elif split == 'val':
            label_base = self.val_label_path
        else:
            label_base = self.test_path.parent.parent / 'test' / 'labels'
        
        label_name = Path(image_path).stem + '.txt'
        return label_base / label_name


# ============================================================================
# PART 2: Action Inference Engine
# ============================================================================

class ActionInferenceEngine:
    """Infer actions/verbs from spatial relationships"""
    
    def __init__(self):
        # Person actions with probabilities based on location
        self.person_actions = {
            'sidewalk': ['walking', 'standing', 'walking'],  # Walking more common
            'road': ['crossing', 'walking', 'crossing'],     # Crossing if on road
            'center': ['crossing', 'standing', 'walking'],
            'edge': ['walking', 'standing', 'walking'],
            'near_vehicle': ['standing', 'walking', 'approaching'],
            'group': ['standing', 'walking together', 'gathering'],
            'default': ['walking', 'standing', 'walking']
        }
        
        # Vehicle actions based on position/context
        self.vehicle_actions = {
            'edge': ['parked', 'parked', 'parked'],           # Edge = likely parked
            'center': ['driving', 'driving', 'moving'],       # Center = likely driving
            'intersection': ['driving', 'turning', 'stopping'],
            'multiple_center': ['driving', 'driving'],        # Multiple in center = traffic
            'single': ['driving', 'parked'],
            'default': ['driving', 'parked']
        }
        
        # Locations based on position
        self.location_terms = {
            'left_sidewalk': 'on the left sidewalk',
            'right_sidewalk': 'on the right sidewalk',
            'sidewalk': 'on the sidewalk',
            'left_road': 'on the left side of the road',
            'right_road': 'on the right side of the road',
            'center_road': 'in the middle of the road',
            'crossing': 'crossing the street',
            'near_curb': 'near the curb',
            'along_street': 'along the street',
        }
    
    def get_position_zone(self, x_norm, y_norm):
        """Determine which zone an object is in"""
        # Horizontal zones
        if x_norm < 0.25:
            h_zone = 'far_left'
        elif x_norm < 0.4:
            h_zone = 'left'
        elif x_norm < 0.6:
            h_zone = 'center'
        elif x_norm < 0.75:
            h_zone = 'right'
        else:
            h_zone = 'far_right'
        
        # Vertical zones (in driving view, bottom = closer)
        if y_norm < 0.3:
            v_zone = 'far'
        elif y_norm < 0.6:
            v_zone = 'mid'
        else:
            v_zone = 'near'
        
        return h_zone, v_zone
    
    def is_on_sidewalk(self, x_norm):
        """Estimate if position is sidewalk (edges) vs road (center)"""
        return x_norm < 0.2 or x_norm > 0.8
    
    def is_on_road(self, x_norm):
        """Estimate if position is on the road"""
        return 0.2 <= x_norm <= 0.8
    
    def infer_person_action(self, detection, all_detections):
        """Infer what a person is doing"""
        x_norm = detection['x_norm']
        y_norm = detection['y_norm']
        
        # Check if near a vehicle
        near_vehicle = False
        for other in all_detections:
            if 'vehicle' in other['class']:
                dist = ((x_norm - other['x_norm'])**2 + (y_norm - other['y_norm'])**2)**0.5
                if dist < 0.15:
                    near_vehicle = True
                    break
        
        # Determine action based on position
        if near_vehicle:
            action = random.choice(self.person_actions['near_vehicle'])
        elif self.is_on_sidewalk(x_norm):
            action = random.choice(self.person_actions['sidewalk'])
        elif self.is_on_road(x_norm):
            action = random.choice(self.person_actions['road'])
        else:
            action = random.choice(self.person_actions['default'])
        
        return action
    
    def infer_vehicle_action(self, detection, all_detections):
        """Infer what a vehicle is doing"""
        x_norm = detection['x_norm']
        y_norm = detection['y_norm']
        
        # Count vehicles in center (likely driving)
        vehicles_in_center = sum(1 for d in all_detections 
                                  if 'vehicle' in d['class'] and 0.25 < d['x_norm'] < 0.75)
        
        # Determine action based on position
        if x_norm < 0.15 or x_norm > 0.85:
            # Far edge = parked
            action = 'parked'
        elif 0.3 < x_norm < 0.7:
            # Center = driving
            if vehicles_in_center > 3:
                action = 'driving'
            else:
                action = random.choice(['driving', 'driving', 'parked'])
        else:
            # Middle zones
            action = random.choice(self.vehicle_actions['default'])
        
        return action
    
    def get_person_location(self, x_norm, y_norm):
        """Get location description for a person"""
        h_zone, v_zone = self.get_position_zone(x_norm, y_norm)
        
        if self.is_on_sidewalk(x_norm):
            if x_norm < 0.3:
                return 'on the left sidewalk'
            else:
                return 'on the right sidewalk'
        elif self.is_on_road(x_norm):
            return 'in the street'
        else:
            if x_norm < 0.5:
                return 'on the left side'
            else:
                return 'on the right side'
    
    def get_vehicle_location(self, x_norm, y_norm, action):
        """Get location description for a vehicle"""
        if action == 'parked':
            if x_norm < 0.3:
                return 'on the left side of the road'
            elif x_norm > 0.7:
                return 'on the right side of the road'
            else:
                return 'along the street'
        else:  # driving
            if x_norm < 0.4:
                return 'on the left side of the road'
            elif x_norm > 0.6:
                return 'on the right side of the road'
            else:
                return 'down the street'


# ============================================================================
# PART 3: Verb-Aware Caption Generator
# ============================================================================

class VerbAwareCaptionGenerator:
    """Generate action-rich captions from YOLO detections"""
    
    def __init__(self):
        self.action_engine = ActionInferenceEngine()
        
        # Environment types (randomly assigned but could be inferred)
        self.environments = [
            'urban street', 'residential street', 'downtown area',
            'urban intersection', 'suburban road', 'parking area',
            'busy road', 'neighborhood', 'commercial street'
        ]
        
        # Caption structure templates matching your style
        self.templates = [
            "FLIR image {environment}, {object_descriptions}.",
            "FLIR image {environment}, {count_summary}. {action_details}",
            "FLIR image {environment}, {action_details}",
            "{time}FLIR image {environment}, {object_descriptions}.",
        ]
        
        self.time_prefixes = ['', '', '', 'Night-time ']  # Occasionally add night-time
    
    def count_by_class(self, detections):
        """Count objects by class"""
        counts = Counter([d['class'] for d in detections])
        return counts
    
    def format_count(self, count, obj_type):
        """Format count naturally"""
        if count == 0:
            return f"0 {obj_type}s"
        elif count == 1:
            return f"1 {obj_type}"
        elif count >= 10:
            return f"10+ {obj_type}s"
        else:
            return f"{count} {obj_type}s"
    
    def generate_count_summary(self, detections):
        """Generate count summary like '6 cars and 4 people'"""
        counts = self.count_by_class(detections)
        
        # Map class names to simpler terms
        class_map = {
            'large vehicle': 'cars',
            'small vehicle': 'cars', 
            'person': 'people'
        }
        
        # Aggregate vehicles
        vehicle_count = counts.get('large vehicle', 0) + counts.get('small vehicle', 0)
        person_count = counts.get('person', 0)
        
        parts = []
        if vehicle_count > 0 or person_count == 0:
            parts.append(self.format_count(vehicle_count, 'car').replace('1 car', '1 car'))
        if person_count > 0 or vehicle_count == 0:
            ppl = self.format_count(person_count, 'person')
            ppl = ppl.replace('persons', 'people').replace('1 person', '1 person')
            if person_count > 1:
                ppl = ppl.replace('person', 'people')
            parts.append(ppl)
        
        return ' and '.join(parts)
    
    def generate_action_descriptions(self, detections):
        """Generate action-based descriptions for each object group"""
        
        persons = [d for d in detections if d['class'] == 'person']
        vehicles = [d for d in detections if 'vehicle' in d['class']]
        
        descriptions = []
        
        # Process persons
        if persons:
            # Group persons by action
            person_actions = {}
            for p in persons:
                action = self.action_engine.infer_person_action(p, detections)
                location = self.action_engine.get_person_location(p['x_norm'], p['y_norm'])
                key = (action, location)
                if key not in person_actions:
                    person_actions[key] = 0
                person_actions[key] += 1
            
            for (action, location), count in person_actions.items():
                if count == 1:
                    descriptions.append(f"1 person {action} {location}")
                elif count >= 10:
                    descriptions.append(f"10+ people {action} {location}")
                else:
                    descriptions.append(f"{count} people {action} {location}")
        
        # Process vehicles
        if vehicles:
            # Group vehicles by action
            vehicle_actions = {'driving': 0, 'parked': 0, 'turning': 0}
            for v in vehicles:
                action = self.action_engine.infer_vehicle_action(v, detections)
                vehicle_actions[action] = vehicle_actions.get(action, 0) + 1
            
            for action, count in vehicle_actions.items():
                if count == 0:
                    continue
                if count == 1:
                    descriptions.append(f"1 car {action}")
                elif count >= 10:
                    descriptions.append(f"10+ cars {action}")
                else:
                    descriptions.append(f"{count} cars {action}")
        
        return descriptions
    
    def generate_caption(self, detections, img_width=640, img_height=640):
        """Generate a verb-aware caption"""
        
        if len(detections) == 0:
            env = random.choice(self.environments)
            return f"FLIR image in a {env}, 0 cars, 0 people."
        
        # Choose template style
        style = random.choice(['detailed', 'count_first', 'actions_only'])
        
        env = random.choice(self.environments)
        time_prefix = random.choice(self.time_prefixes)
        
        count_summary = self.generate_count_summary(detections)
        action_descs = self.generate_action_descriptions(detections)
        
        if style == 'detailed':
            # Style: "FLIR image in X, containing Y cars and Z people. Actions..."
            action_text = ', '.join(action_descs) if action_descs else 'no activity'
            caption = f"{time_prefix}FLIR image in a {env}, containing {count_summary}. {action_text}."
            
        elif style == 'count_first':
            # Style: "FLIR image in X, Y cars driving, Z people walking..."
            action_text = ', '.join(action_descs) if action_descs else f"{count_summary}"
            caption = f"{time_prefix}FLIR image in a {env}, {action_text}."
            
        else:  # actions_only
            # Style: "FLIR image on X, actions..."
            action_text = ', '.join(action_descs) if action_descs else f"{count_summary}"
            preposition = random.choice(['in a', 'on a', 'at a'])
            caption = f"{time_prefix}FLIR image {preposition} {env}, {action_text}."
        
        # Clean up caption
        caption = caption.replace('..', '.').replace(',,', ',')
        caption = caption.replace(' ,', ',').replace('  ', ' ')
        
        return caption


# ============================================================================
# PART 4: Caption Dataset Builder
# ============================================================================

class ActionCaptionDatasetBuilder:
    """Build caption dataset with action-aware captions"""
    
    def __init__(self, yaml_path):
        self.parser = YOLOLabelParser(yaml_path)
        self.caption_gen = VerbAwareCaptionGenerator()
    
    def generate_captions_for_split(self, split='train', num_variations=1):
        image_files = self.parser.get_image_list(split)
        captions_data = []
        
        print(f"\nGenerating action captions for {len(image_files)} images in {split} split...")
        
        for idx, image_path in enumerate(image_files):
            label_path = self.parser.get_label_path(image_path, split)
            
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except:
                width, height = 640, 480
            
            detections = self.parser.parse_yolo_label(label_path, width, height)
            
            captions = []
            for _ in range(num_variations):
                caption = self.caption_gen.generate_caption(detections, width, height)
                captions.append(caption)
            
            captions_data.append({
                'image_path': str(image_path),
                'image_name': image_path.name,
                'captions': captions,
                'num_objects': len(detections),
                'detections': [
                    {
                        'class': d['class'],
                        'x_norm': d['x_norm'],
                        'y_norm': d['y_norm']
                    } for d in detections
                ]
            })
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(image_files)} images")
        
        print(f"✓ Generated captions for {len(captions_data)} images")
        return captions_data
    
    def export_to_json(self, output_file='ir_action_captions.json', splits=['train', 'val'], 
                      num_variations=3):
        all_data = {}
        
        for split in splits:
            print(f"\n{'='*60}")
            print(f"Processing {split} split")
            print(f"{'='*60}")
            all_data[split] = self.generate_captions_for_split(split, num_variations)
        
        with open(output_file, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"\n✓ Captions exported to: {output_file}")
        return all_data
    
    def show_samples(self, split='train', num_samples=10):
        captions_data = self.generate_captions_for_split(split, num_variations=3)
        
        print(f"\n{'='*80}")
        print(f"SAMPLE ACTION CAPTIONS FROM {split.upper()} SPLIT")
        print(f"{'='*80}")
        
        for i, sample in enumerate(captions_data[:num_samples]):
            print(f"\n{i+1}. Image: {sample['image_name']}")
            print(f"   Objects: {sample['num_objects']}")
            classes = [d['class'] for d in sample['detections']]
            print(f"   Classes: {Counter(classes)}")
            print(f"   Captions:")
            for j, caption in enumerate(sample['captions']):
                print(f"     {j+1}. {caption}")


# ============================================================================
# PART 5: Main
# ============================================================================

def main():
    YAML_PATH = 'data (1).yaml'
    
    print("="*80)
    print("IR IMAGE ACTION CAPTION GENERATOR (Verb-Aware)")
    print("="*80)
    print("\nExtracted action patterns from your captions:")
    print("  Person actions: walking, standing, crossing, riding, pushing")
    print("  Vehicle actions: driving, parked, turning")
    
    builder = ActionCaptionDatasetBuilder(YAML_PATH)
    
    print("\n[1/2] Generating sample action captions...")
    builder.show_samples(split='train', num_samples=10)
    
    print("\n[2/2] Exporting full dataset...")
    builder.export_to_json(
        output_file='ir_action_captions.json',
        splits=['train', 'val'],
        num_variations=3
    )
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
