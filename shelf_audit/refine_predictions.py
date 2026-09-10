"""Conservative overlap cleanup. Preserve excluded predictions for audit."""
import argparse
import copy
import json
from pathlib import Path


def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])


def intersection(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def clean_frame(frame):
    result=copy.deepcopy(frame);keep=[];excluded=[]
    ds=frame['detections']
    for i,d in enumerate(ds):
        reason=None;support=[]
        if d['recognition']['status']=='review_needed':
            b=d['bbox_xyxy'];a=area(b)
            for j,other in enumerate(ds):
                if i==j or other['recognition']['status']!='candidate_match':continue
                ob=other['bbox_xyxy'];oa=area(ob);overlap=intersection(b,ob)
                if a and overlap/a>=.85 and a<oa*.65:
                    reason='uncertain fragment contained in an identified package';support=[j];break
            if reason is None:
                contained=[j for j,o in enumerate(ds) if j!=i and o['recognition']['status']=='candidate_match' and area(o['bbox_xyxy'])>0 and intersection(b,o['bbox_xyxy'])/area(o['bbox_xyxy'])>=.8]
                # Only remove a merged proposal when separate, non-overlapping
                # accepted proposals already explain most of its area.
                for j in contained:
                    for k in contained:
                        if j>=k:continue
                        x,y=ds[j]['bbox_xyxy'],ds[k]['bbox_xyxy']
                        if intersection(x,y)<.1*min(area(x),area(y)) and (intersection(b,x)+intersection(b,y))/a>=.7:
                            reason='merged proposal covered by two separate identified packages';support=[j,k]
        if reason:excluded.append(dict(original_detection_index=i,reason=reason,supporting_indices=support,detection=d))
        else:keep.append(d)
    result['detections']=keep;result['excluded_detections']=excluded
    return result


def run(source,out):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new predictions file.')
    data=json.loads(Path(source).read_text());data['frames']=[clean_frame(f) for f in data['frames']]
    data['summary']['cleanup']='Conservative uncertain-overlap removal; every exclusion retained in each frame. No boxes invented or identities changed.'
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(data,indent=2))
    print('Excluded',sum(len(f['excluded_detections']) for f in data['frames']),'uncertain overlapping proposals')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.source,a.out)
