"""Offline spatial tracking and evidence-based ordering for a single shelf row.

Heuristic baseline: short gaps only, no long-range reidentification. Identities
never merge tracks, so two packages of the same SKU can remain separate.
"""
from .report_style import apply_file
import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path


def iou(a, b):
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection/union if union > 0 else 0.


def track_frames(frames, max_gap=.25, reconnect=False):
    tracks = []; assignments = []
    for frame in frames:
        now = frame['time_seconds']; detections = frame['detections']; candidates = []
        for t in tracks:
            last = t['observations'][-1]
            gap = now-last['time_seconds']
            if not 0 <= gap <= (.75 if reconnect else max_gap): continue
            predicted = list(last['bbox_xyxy'])
            if len(t['observations']) > 1:
                previous = t['observations'][-2]
                dt = last['time_seconds']-previous['time_seconds']
                if dt > 0:
                    # Estimate translation only; clipping can distort package width.
                    dx = ((last['bbox_xyxy'][0]+last['bbox_xyxy'][2])-(previous['bbox_xyxy'][0]+previous['bbox_xyxy'][2]))/2
                    dy = ((last['bbox_xyxy'][1]+last['bbox_xyxy'][3])-(previous['bbox_xyxy'][1]+previous['bbox_xyxy'][3]))/2
                    factor = min(gap/dt, 3.)
                    predicted = [v + (dx if j%2==0 else dy)*factor for j,v in enumerate(predicted)]
            for j,d in enumerate(detections):
                if gap>max_gap:
                    recent=[o['recognition']['predicted_class'] for o in t['observations'][-10:] if o['recognition']['status']=='candidate_match']
                    votes=Counter(recent)
                    winner,count=votes.most_common(1)[0] if votes else (None,0)
                    if count<3 or count/len(recent)<.8 or d['recognition']['status']!='candidate_match' or d['recognition']['predicted_class']!=winner:continue
                overlap = max(iou(predicted,d['bbox_xyxy']), iou(last['bbox_xyxy'],d['bbox_xyxy']))
                if overlap >= (.5 if gap>max_gap else .25):
                    candidates.append((overlap,t['id'],j))
        used_tracks = set(); used_detections = set(); links = {}
        for score,tid,j in sorted(candidates,reverse=True):
            if tid in used_tracks or j in used_detections: continue
            used_tracks.add(tid); used_detections.add(j); links[j] = tid
        frame_links = []
        for j,d in enumerate(detections):
            if j not in links:
                links[j] = len(tracks)+1
                tracks.append(dict(id=links[j],observations=[]))
            tid = links[j]
            tracks[tid-1]['observations'].append(dict(frame_index=frame['frame_index'],time_seconds=now,**d))
            frame_links.append(dict(track_id=tid,detection_index=j))
        assignments.append(dict(frame_index=frame['frame_index'],time_seconds=now,assignments=frame_links))
    for t in tracks:
        obs = t['observations']; votes = Counter()
        for o in obs:
            r = o['recognition']
            if r['status']=='candidate_match' and r['predicted_class']:
                votes[r['predicted_class']] += 1
        ranked = votes.most_common(); winner,count = ranked[0] if ranked else (None,0)
        agreement = count/sum(votes.values()) if votes else 0.
        persistent = len(obs)>=5 and obs[-1]['time_seconds']-obs[0]['time_seconds']>=.15
        stable = persistent and count>=5 and agreement>=.7
        t.update(class_id=winner if stable else None, candidate_class=winner,
                 accepted_votes=dict(votes),vote_agreement=agreement,
                 status='stable_candidate' if stable else 'review_needed',persistent=persistent,
                 first_seconds=obs[0]['time_seconds'],last_seconds=obs[-1]['time_seconds'],
                 detection_count=len(obs))
    return tracks, assignments


def topological_order(nodes, edges):
    outgoing = defaultdict(set); indegree = dict.fromkeys(nodes,0)
    for a,b in edges:
        if b not in outgoing[a]:
            outgoing[a].add(b); indegree[b]+=1
    order=[]; ambiguous=False
    while len(order)<len(nodes):
        available=sorted(n for n in nodes if indegree[n]==0 and n not in order)
        if not available: return None, True
        if len(available)>1: ambiguous=True
        n=available[0];order.append(n)
        for b in outgoing[n]:indegree[b]-=1
    return order, ambiguous


def infer_order(tracks, assignments, frames):
    nodes=[t['id'] for t in tracks if t['persistent']]
    # Temporal bins reduce the influence of nearly identical consecutive frames.
    evidence=defaultdict(set)
    for f,links in zip(frames,assignments):
        visible=[(x['track_id'],f['detections'][x['detection_index']]['bbox_xyxy']) for x in links['assignments'] if x['track_id'] in nodes]
        for a,boxa in visible:
            for b,boxb in visible:
                if a>=b:continue
                # Require vertical overlap: this implementation targets one row.
                overlap=max(0,min(boxa[3],boxb[3])-max(boxa[1],boxb[1]))
                if overlap < .3*min(boxa[3]-boxa[1],boxb[3]-boxb[1]):continue
                xa=(boxa[0]+boxa[2])/2; xb=(boxb[0]+boxb[2])/2
                if abs(xa-xb)<.02*f['width']:continue
                evidence[(a,b) if xa<xb else (b,a)].add(int(f['time_seconds']/.2))
    edges=[]; details=[]; conflicts=[]
    for a,b in sorted({tuple(sorted(pair)) for pair in evidence}):
        forward=len(evidence[(a,b)]);reverse=len(evidence[(b,a)])
        left,right=(a,b) if forward>=reverse else (b,a)
        support=max(forward,reverse); opposition=min(forward,reverse)
        reliable=support>=2 and support/(support+opposition)>=.8
        details.append(dict(left=left,right=right,support_bins=support,opposing_bins=opposition,used=reliable))
        if reliable:edges.append((left,right))
        elif support:conflicts.append([a,b])
    order,ambiguous=topological_order(nodes,edges)
    reasons=[]
    if order is None:reasons.append('Contradictory pairwise order creates a cycle.')
    elif ambiguous:reasons.append('Evidence does not establish a unique order for all persistent tracks.')
    if conflicts:reasons.append('Some pairwise relationships have insufficient or conflicting evidence.')
    if any(t['persistent'] and t['status']!='stable_candidate' for t in tracks):reasons.append('Some persistent tracks have uncertain identities.')
    counts=Counter(t['class_id'] for t in tracks if t['persistent'] and t['class_id'])
    if any(v>1 for v in counts.values()):reasons.append('Repeated identities may represent duplicate packages or fragmented tracks; review required.')
    if not nodes:reasons.append('No persistent tracks.')
    return dict(track_order=order,unique_order=bool(nodes) and order is not None and not ambiguous,
                status='review_needed' if reasons else 'candidate_order',reasons=reasons,pairwise_evidence=details,
                persistent_tracks=len(nodes),short_tracks=sum(not t['persistent'] for t in tracks))


def run(predictions, out, reconnect=False):
    predictions=Path(predictions);out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Choose an empty output directory.')
    data=json.loads(predictions.read_text());frames=data['frames']
    if any(b['time_seconds']<a['time_seconds'] for a,b in zip(frames,frames[1:])):raise ValueError('Frames must be timestamp ordered.')
    tracks,assignments=track_frames(frames,reconnect=reconnect);order=infer_order(tracks,assignments,frames)
    identified=[dict(t,persistent=t['persistent'] and t['status']=='stable_candidate') for t in tracks]
    identified_order=infer_order(identified,assignments,frames)
    out.mkdir(parents=True,exist_ok=True)
    result=dict(source=data['summary'],protocol='Offline greedy spatial association with translation prediction, 0.25s maximum gap; accepted-identity votes; co-visible left/right evidence in 0.2s bins.',
                limitations='Single-row development baseline. Votes are correlated, not probabilities. No long-gap reidentification or validated unique inventory counts. Input includes training frames. Rearranged videos unused.',
                reconnect_enabled=reconnect,order=order,identified_order=identified_order,tracks=tracks,frames=assignments)
    if reconnect:result['protocol']+=' Gaps up to 0.75s require consistent accepted identity and overlap >=0.5.'
    (out/'results.json').write_text(json.dumps(result,indent=2))
    rows=[]
    by_id={t['id']:t for t in tracks}
    for tid in identified_order['track_order'] or [t['id'] for t in identified if t['persistent']]:
        t=by_id[tid];name=(t['class_id'] or 'Unknown / review').removeprefix('great_value_').replace('_',' ')
        rows.append(f'<li><b>Track {tid}: {html.escape(name)}</b><br>{t["detection_count"]} detections · {t["first_seconds"]:.2f}–{t["last_seconds"]:.2f}s · {t["vote_agreement"]:.0%} agreement among accepted votes</li>')
    title='Candidate left-to-right order of identified packages' if identified_order['unique_order'] else 'Identified track list — shelf order uncertain'
    (out/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shelf tracking review</title><style>body{font:17px system-ui;background:#10212b;color:#e8f3f8;margin:35px;max-width:1000px}li{padding:12px;background:#1a3541;margin:8px 0;border-radius:8px}p{line-height:1.5}a{color:#82e5b6}video{max-height:70vh;max-width:100%}</style><h1>'+title+'</h1><p>'+html.escape('; '.join(order['reasons']) or 'Order supported by repeated co-visible positions. This is a development candidate, not a validated inventory audit.')+'</p><ol>'+''.join(rows)+'</ol><p>'+html.escape(result['limitations'])+'</p><p>'+str(order['short_tracks'])+' short tracks retained in JSON for review; omitted from this order.</p><p><a href="results.json">Tracks and pairwise evidence</a></p></html>',encoding='utf-8')
    apply_file((out/'review.html'))
    print(json.dumps(order,indent=2))
    print('Identified order:', identified_order['track_order'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('predictions',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reconnect',action='store_true')
    a=p.parse_args();run(a.predictions,a.out,a.reconnect)
