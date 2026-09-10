"""Run automatic detection and catalog recognition on a prerecorded video."""
from .report_style import apply_file
import argparse
import html
import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor

from .detector import CHECKPOINT, load_detector
from .recognizer import ROOT, Recognizer, file_hash, rank_matches


def process(source, out, threshold=.5, checkpoint=CHECKPOINT):
    source = Path(source).resolve(); out = Path(out).resolve()
    if not 0 <= threshold <= 1:
        raise ValueError('Detection threshold must be in [0,1].')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose an empty output directory to preserve previous results.')
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {source}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release(); raise ValueError('Video has no usable frame rate.')
    torch.set_num_threads(8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = load_detector(checkpoint=checkpoint, device=device)
    recognizer = Recognizer(ROOT/'artifacts/catalog_index', device=device)
    out.mkdir(parents=True, exist_ok=True); (out/'frames').mkdir(exist_ok=True)
    records = []; cards = []; encoder = None; start = time.perf_counter()
    try:
        with torch.inference_mode():
            while True:
                ok, bgr = cap.read()
                if not ok: break
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                full = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                size = (540, round(full.height * 540 / full.width / 2)*2)
                im = full.resize(size)
                if encoder is None:
                    encoder = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
                        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
                        '-s', f'{size[0]}x{size[1]}', '-r', str(fps), '-i', '-', '-an',
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(out/'annotated.mp4')], stdin=subprocess.PIPE)
                result = detector([to_tensor(im).to(device)])[0]
                boxes = result['boxes'][result['scores'] >= threshold].cpu().tolist()
                scores = result['scores'][result['scores'] >= threshold].cpu().tolist()
                crops = []; valid = []
                for box, score in zip(boxes, scores):
                    x1,y1,x2,y2 = box
                    box = [max(0,math.floor(x1)), max(0,math.floor(y1)), min(im.width,math.ceil(x2)), min(im.height,math.ceil(y2))]
                    if box[2] <= box[0] or box[3] <= box[1]: continue
                    # Recognize from original-resolution pixels, not the preview.
                    sx, sy = full.width/im.width, full.height/im.height
                    original_box = [round(box[0]*sx),round(box[1]*sy),round(box[2]*sx),round(box[3]*sy)]
                    crops.append(full.crop(original_box)); valid.append((box, score, original_box))
                vectors = recognizer.encoder.encode(crops) if crops else []
                detections = []; draw = ImageDraw.Draw(im)
                for (box,score,original_box), vector in zip(valid,vectors):
                    match = rank_matches(vector, recognizer.vectors, recognizer.records)
                    detections.append(dict(bbox_xyxy=original_box, detector_score=score, recognition=match))
                    accepted = match['status'] == 'candidate_match'
                    color = '#65efa9' if accepted else '#ffce63'
                    name = match['matches'][0]['class_id'].removeprefix('great_value_').replace('_',' ')
                    # Keep uncertain top guesses explicitly marked on the video.
                    label = ('' if accepted else 'REVIEW: ') + name
                    draw.rectangle(box, outline=color, width=3)
                    y = max(25, box[1]-35)
                    for line, offset in ((label[:37],0),(label[37:],15)):
                        if line:
                            x = min(box[0], max(0,im.width-300))
                            draw.rectangle((x,y+offset,x+min(320,len(line)*8),y+offset+16),fill='#14222c')
                            draw.text((x+2,y+offset),line,fill=color,font_size=13)
                draw.rectangle((0,0,im.width,23),fill='#14222c')
                draw.text((6,3),f'AUTOMATIC | {timestamp:.2f}s | development baseline',fill='white',font_size=14)
                index = len(records)
                records.append(dict(frame_index=index,time_seconds=timestamp,width=full.width,height=full.height,detections=detections))
                encoder.stdin.write(np.asarray(im).tobytes())
                if index % 30 == 0:
                    filename = f'frame_{index:06d}.jpg'; im.save(out/'frames'/filename,quality=90)
                    cards.append(f'<article><img src="frames/{filename}"><p>{timestamp:.2f}s · {len(detections)} detections</p></article>')
                    print(f'Processed frame {index}',flush=True)
    finally:
        cap.release()
        if encoder is not None:
            encoder.stdin.close()
            code = encoder.wait()
            if code: raise RuntimeError(f'Video encoder exited with {code}')
    if not records: raise ValueError('No decodable video frames.')
    summary = dict(source=source.name,source_sha256=file_hash(source),checkpoint_sha256=file_hash(checkpoint),
        frames=len(records),nominal_fps=fps,processing_seconds=time.perf_counter()-start,device=device,
        detector_threshold=threshold,coordinate_space='upright source image pixels',
        limitations='Development footage, including training frames. Automatic per-frame boxes and recognition only; no tracking, unique counts, or global shelf order. Preview uses nominal frame rate and omits audio.',
        catalog_embeddings_sha256=recognizer.meta['embeddings_sha256'])
    (out/'results.json').write_text(json.dumps(dict(summary=summary,frames=records),indent=2))
    (out/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Automatic video detection</title><style>body{font:16px system-ui;background:#10212b;color:#edf5fa;margin:28px}a{color:#7be5bf}video{max-height:75vh;max-width:100%;background:black}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px}img{width:100%}article{max-width:420px}p{max-width:900px;line-height:1.5}</style><h1>Automatic product detection + recognition</h1><p>These boxes are model predictions. Green: candidate product match. Amber: uncertain identity requiring review. Missed packages and incorrect boxes are possible.</p><video controls playsinline src="annotated.mp4"></video><p>'+html.escape(summary['limitations'])+'</p><p><a href="results.json">Per-frame predictions</a></p><h2>Sample frames</h2><div class="grid">'+''.join(cards)+'</div></html>',encoding='utf-8')
    apply_file((out/'review.html'))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--threshold',type=float,default=.5)
    parser.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    args=parser.parse_args()
    process(args.video,args.out,args.threshold,args.checkpoint)
