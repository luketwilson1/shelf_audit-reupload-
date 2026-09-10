"""Frozen CNN embeddings with nearest-reference product retrieval."""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'torchvision/resnet50/IMAGENET1K_V2'
PREPROCESS_ID = 'exif_rgb_meanpad_square_bilinear224_imagenetnorm_v1'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank_matches(query, vectors, records, *, top_k=3, min_score=.65, min_margin=.03, exclude=()):
    """Max reference cosine per distinct product. Scores are not probabilities."""
    if not -1 <= min_score <= 1 or not 0 <= min_margin <= 2:
        raise ValueError('Score must be in [-1,1] and margin in [0,2].')
    if top_k < 1:
        raise ValueError('top_k must be positive.')
    q = np.asarray(query, dtype=np.float32)
    if q.ndim != 1 or vectors.ndim != 2 or vectors.shape[1] != q.shape[0] or len(records) != len(vectors):
        raise ValueError('Embedding dimensions or record count do not match.')
    if not np.isfinite(q).all() or not np.isfinite(vectors).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError('Invalid embedding.')
    scores = vectors @ (q / np.linalg.norm(q))
    groups = {}
    excluded = set(exclude)
    for i, (score, record) in enumerate(zip(scores, records)):
        if i in excluded:
            continue
        key = record['class_id']
        if key not in groups or score > groups[key]['similarity']:
            groups[key] = dict(class_id=key, product=record['product'], similarity=float(np.clip(score, -1, 1)), reference_index=i, reference_file=record['crop_file'])
    ranked = sorted(groups.values(), key=lambda x: (-x['similarity'], x['class_id']))
    if not ranked:
        raise ValueError('No reference candidates remain.')
    margin = ranked[0]['similarity'] - ranked[1]['similarity'] if len(ranked) > 1 else None
    reasons = []
    if ranked[0]['similarity'] < min_score:
        reasons.append('low_similarity')
    if margin is None or margin < min_margin:
        reasons.append('ambiguous_match')
    return dict(predicted_class=None if reasons else ranked[0]['class_id'], status='review_needed' if reasons else 'candidate_match', reasons=reasons, margin=margin, matches=ranked[:top_k], thresholds=dict(min_score=min_score, min_margin=min_margin, calibrated=False))


class Encoder:
    def __init__(self, device='auto'):
        import torch
        from torchvision.models import resnet50, ResNet50_Weights
        from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
        self.torch = torch
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        torch.hub.set_dir(str(ROOT / '.cache/torch'))
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable. Try --device cpu.')
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)
        self.transform = Compose([Resize((224,224), interpolation=InterpolationMode.BILINEAR, antialias=True), ToTensor(), Normalize([.485,.456,.406],[.229,.224,.225])])

    def tensor(self, image):
        image = ImageOps.exif_transpose(image).convert('RGB')
        side = max(image.size)
        square = Image.new('RGB',(side,side),(124,116,104))
        square.paste(image,((side-image.width)//2,(side-image.height)//2))
        return self.transform(square)

    def encode(self, images, batch_size=16):
        if not images or batch_size < 1:
            raise ValueError('Supply images and a positive batch size.')
        result=[]
        with self.torch.inference_mode():
            for start in range(0,len(images),batch_size):
                tensors=[]
                for source in images[start:start+batch_size]:
                    if isinstance(source, Image.Image):
                        tensors.append(self.tensor(source))
                    else:
                        with Image.open(source) as im:
                            tensors.append(self.tensor(im))
                x=self.torch.stack(tensors).to(self.device)
                features=self.torch.nn.functional.normalize(self.model(x),dim=1)
                result.append(features.cpu().numpy())
        return np.concatenate(result).astype(np.float32)


def build_index(catalog, destination, device='auto'):
    catalog=Path(catalog).resolve();destination=Path(destination).resolve()
    records=json.loads((catalog/'manifest.json').read_text(encoding='utf-8'))
    if len({r['class_id'] for r in records}) < 2:
        raise ValueError('The reference catalog must contain at least two products.')
    paths=[]
    for r in records:
        path=(catalog/r['crop_file']).resolve()
        if not path.is_relative_to(catalog):
            raise ValueError('Catalog path escapes its directory.')
        r['sha256']=file_hash(path);paths.append(path)
    encoder=Encoder(device)
    start=time.perf_counter();vectors=encoder.encode(paths);elapsed=time.perf_counter()-start
    metadata=dict(schema_version=1,model=MODEL_ID,preprocess=PREPROCESS_ID,catalog_root=str(catalog),records=records,embedding_dimension=int(vectors.shape[1]),device=encoder.device,build_seconds=elapsed)
    destination.mkdir(parents=True,exist_ok=True)
    np.save(destination/'embeddings.npy',vectors,allow_pickle=False)
    metadata['embeddings_sha256']=file_hash(destination/'embeddings.npy')
    (destination/'index.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return dict(references=len(records),products=len({r['class_id'] for r in records}),dimension=vectors.shape[1],device=encoder.device,embedding_seconds=round(elapsed,3),index=str(destination))


class Recognizer:
    def __init__(self, index, device='auto', load_encoder=True):
        self.index=Path(index).resolve()
        self.meta=json.loads((self.index/'index.json').read_text(encoding='utf-8'))
        if self.meta['model'] != MODEL_ID or self.meta['preprocess'] != PREPROCESS_ID:
            raise ValueError('Index model/preprocessing mismatch. Rebuild the index.')
        if file_hash(self.index/'embeddings.npy') != self.meta['embeddings_sha256']:
            raise ValueError('Index integrity check failed. Rebuild the index.')
        self.vectors=np.load(self.index/'embeddings.npy',allow_pickle=False)
        self.records=self.meta['records'];self.catalog=Path(self.meta['catalog_root'])
        if self.vectors.shape != (len(self.records),self.meta['embedding_dimension']) or not np.isfinite(self.vectors).all() or not np.allclose(np.linalg.norm(self.vectors,axis=1),1,atol=1e-4):
            raise ValueError('Invalid index vectors.')
        for r in self.records:
            path=(self.catalog/r['crop_file']).resolve()
            if not path.is_relative_to(self.catalog.resolve()) or file_hash(path) != r['sha256']:
                raise ValueError('Catalog changed since indexing. Rebuild the index.')
        self.encoder=Encoder(device) if load_encoder else None

    def predict(self, image, **kwargs):
        if self.encoder is None:
            raise RuntimeError('Encoder is not loaded.')
        start=time.perf_counter()
        embedding=self.encoder.encode([image])[0]
        result=rank_matches(embedding,self.vectors,self.records,**kwargs)
        result['elapsed_ms']=round((time.perf_counter()-start)*1000,2)
        result['device']=self.encoder.device
        return result

    def evaluate(self):
        """Exclude each query and byte-identical copies. Same-session diagnostic only."""
        results=[]
        for i,r in enumerate(self.records):
            excluded=[j for j,s in enumerate(self.records) if s['sha256']==r['sha256']]
            result=rank_matches(self.vectors[i],self.vectors,self.records,exclude=excluded)
            truth=r['class_id']
            results.append(dict(query=r['crop_file'],actual=truth,correct=result['matches'][0]['class_id']==truth,top3_correct=truth in [x['class_id'] for x in result['matches']],**result))
        return dict(protocol='leave-one-reference-image-out, excluding identical file hashes',limitation='Same products, background, physical packages and capture session. Not independent video accuracy or calibrated unknown-product performance.',queries=len(results),top1_correct=sum(r['correct'] for r in results),top3_correct=sum(r['top3_correct'] for r in results),review_needed=sum(r['status']=='review_needed' for r in results),results=results)
