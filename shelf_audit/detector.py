"""Small, reproducible single-class detector baseline on approved video boxes."""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.transforms.functional import to_tensor, adjust_brightness

from .recognizer import ROOT, file_hash

CHECKPOINT = ROOT / 'artifacts/product_detector/baseline.pt'
VALIDATION_FRAMES = {60, 270}


def make_model(pretrained=False):
    torch.hub.set_dir(str(ROOT / '.cache/torch'))
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights='DEFAULT' if pretrained else None, weights_backbone=None,
        min_size=540, max_size=960, box_detections_per_img=30,
        box_score_thresh=0.05, box_nms_thresh=0.4)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, 2)
    # Freeze the backbone for a small-data baseline; train proposal and box heads.
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    return model


def load_detector(checkpoint=CHECKPOINT, device='cuda'):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = make_model().to(device)
    model.load_state_dict(state['model'])
    model.eval()
    return model


def approved_frames(extra_boxes=None):
    source = ROOT / 'data/video_queries/annotations.json'
    annotations = json.loads(source.read_text())
    grouped = {}
    for a in annotations['annotations']:
        if a['label_status'] != 'user_approved':
            raise ValueError('Training requires approved annotations.')
        grouped.setdefault(a['frame_index'], {'file': a['frame_file'], 'boxes': []})['boxes'].append(a['bbox_xyxy'])
    if extra_boxes:
        extra = json.loads(Path(extra_boxes).read_text())
        manifest = json.loads((ROOT/'data/video_frames/pass_original_1/manifest.json').read_text())
        if extra['source'] != manifest['source'] or extra['coordinate_space'] != '540x960 upright preview pixels':
            raise ValueError('Extra annotation source or coordinates do not match.')
        lookup = {r['frame_index']: r for r in manifest['frames']}
        for key, boxes in extra['frames'].items():
            index = int(key)
            if index in VALIDATION_FRAMES or index in grouped:
                raise ValueError('Extra annotations cannot replace existing or validation frames.')
            for x1,y1,x2,y2 in boxes:
                if not 0 <= x1 < x2 <= 540 or not 0 <= y1 < y2 <= 960:
                    raise ValueError('Invalid extra box')
            grouped[index] = {'file':lookup[index]['file'], 'boxes':[[v*2 for v in box] for box in boxes]}
    return grouped


def sample(record, augment=False):
    path = ROOT / record['image_path'] if 'image_path' in record else ROOT / 'data/video_frames/pass_original_1' / record['file']
    with Image.open(path) as image:
        image = image.convert('RGB').resize((540, 960))
    boxes = torch.tensor(record['boxes'], dtype=torch.float32).reshape(-1,4) / 2
    if augment:
        image = adjust_brightness(image, random.uniform(.8, 1.2))
        if random.random() < .5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            boxes[:, [0, 2]] = 540 - boxes[:, [2, 0]]
    return to_tensor(image), {'boxes': boxes, 'labels': torch.ones(len(boxes), dtype=torch.int64)}


def match_boxes(predictions, truth, threshold=.5):
    """One-to-one score-ordered matches; duplicate boxes count as false positives."""
    matched = set()
    tp = 0
    overlaps = box_iou(predictions, truth)
    for row in overlaps:
        available = [(float(value), j) for j, value in enumerate(row) if j not in matched]
        value, j = max(available, default=(0., -1))
        if value >= threshold:
            tp += 1
            matched.add(j)
    return tp, len(predictions) - tp, len(truth) - tp


@torch.inference_mode()
def evaluate(model, frames, device):
    model.eval()
    tp = fp = fn = 0
    rows = []
    for index, record in frames.items():
        image, target = sample(record)
        prediction = model([image.to(device)])[0]
        boxes = prediction['boxes'][prediction['scores'] >= .5].cpu()
        a, b, c = match_boxes(boxes, target['boxes'])
        tp += a; fp += b; fn += c
        rows.append(dict(frame_index=index, tp=a, fp=b, fn=c, boxes=boxes.tolist()))
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn) if tp+fn else 0, score_threshold=.5, iou_threshold=.5, frames=rows)


def train(epochs=40, checkpoint=CHECKPOINT, extra_boxes=None, corrections=None):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    random.seed(42); torch.manual_seed(42); torch.set_num_threads(8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        raise ValueError('Choose a new checkpoint path to preserve earlier experiments.')
    frames = approved_frames(extra_boxes)
    training = {i:r for i,r in frames.items() if i not in VALIDATION_FRAMES}
    validation = {i:r for i,r in frames.items() if i in VALIDATION_FRAMES}
    if corrections:
        correction_data=json.loads(Path(corrections).read_text())
        if correction_data['coordinate_space']!='540x960 upright preview pixels':raise ValueError('Wrong correction coordinate space')
        for key,boxes in correction_data['frames'].items():
            if key in training:raise ValueError('Duplicate training key')
            for x1,y1,x2,y2 in boxes:
                if not 0<=x1<x2<=540 or not 0<=y1<y2<=960:raise ValueError('Invalid corrected box')
            image_path=f'data/corrected_detector/{key}.jpg'
            if not (ROOT/image_path).is_file():raise ValueError('Missing corrected frame')
            training[key]={'image_path':image_path,'boxes':[[v*2 for v in b] for b in boxes]}
    if not training or len(validation) != 2:
        raise ValueError('Expected fixed development partition.')
    model = make_model(pretrained=True).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.005, momentum=.9, weight_decay=.0005)
    history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train(); model.backbone.eval()
        keys = list(training); random.shuffle(keys)
        total = 0.
        for offset in range(0, len(keys), 2):
            batch = [sample(training[k], augment=True) for k in keys[offset:offset+2]]
            images = [im.to(device) for im,_ in batch]
            targets = [{k:v.to(device) for k,v in target.items()} for _,target in batch]
            loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step(); total += float(loss.detach())
        history.append(total / ((len(keys)+1)//2))
        print(f'Epoch {epoch+1}/{epochs}: loss={history[-1]:.4f}', flush=True)
        if epoch+1 == int(epochs*.75):
            for group in optimizer.param_groups: group['lr'] *= .1
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(architecture='fasterrcnn_mobilenet_v3_large_fpn', classes=['background','product'],
                    seed=42, epochs=epochs, training_frames=list(training), validation_frames=list(validation),
                    annotation_sha256=file_hash(ROOT/'data/video_queries/annotations.json'),
                    limitation='Development diagnostic, not independent accuracy. Reviewed rearranged frames are now training data.' if corrections else 'Same-video development split, correlated frames; not independent accuracy. No rearranged videos used.',
                    correction_sha256=file_hash(corrections) if corrections else None,
                    corrected_image_hashes={k:file_hash(ROOT/r['image_path']) for k,r in training.items() if 'image_path' in r},
                    extra_annotation_sha256=file_hash(extra_boxes) if extra_boxes else None,
                    extra_annotation_status=json.loads(Path(extra_boxes).read_text())['status'] if extra_boxes else None,
                    training_seconds=time.perf_counter()-start, loss_history=history)
    torch.save({'model': model.state_dict(), 'metadata': metadata}, checkpoint)
    metadata['validation'] = evaluate(model, validation, device)
    metadata['training_fit'] = evaluate(model, training, device)
    checkpoint.with_suffix('.report.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata['validation'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--extra-boxes', type=Path)
    parser.add_argument('--corrections', type=Path)
    args=parser.parse_args()
    train(args.epochs, args.checkpoint, args.extra_boxes, args.corrections)
