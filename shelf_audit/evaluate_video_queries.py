"""Recognition diagnostic on manually localized development-video products."""
from .report_style import apply_file
import json,html,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
from .recognizer import ROOT,Recognizer,rank_matches,file_hash

def main():
    frame_root=ROOT/'data/video_frames/pass_original_1'
    out=ROOT/'data/video_queries';out.mkdir(exist_ok=True)
    for d in ['crops','overlays','labels']:(out/d).mkdir(exist_ok=True)
    specs=json.loads(Path(__file__).with_name('video_query_boxes.json').read_text())
    source=json.loads((frame_root/'manifest.json').read_text())
    lookup={r['frame_index']:r for r in source['frames']}
    classes=(ROOT/'data/catalog_annotations/classes.txt').read_text().splitlines()
    catalog=json.loads((ROOT/'data/catalog_crops/manifest.json').read_text())
    names={r['class_id']:r['product'] for r in catalog}
    annotations=[];images=[];overlay_cards=[]
    for frame_number,boxes in specs['frames'].items():
        f=lookup[int(frame_number)];im=Image.open(frame_root/f['file']).convert('RGB');assert im.size==(1080,1920)
        overlay=im.copy();draw=ImageDraw.Draw(overlay);lines=[]
        for ci,*coords in boxes:
            box=[v*2 for v in coords];x1,y1,x2,y2=box
            assert 0<=x1<x2<=im.width and 0<=y1<y2<=im.height
            key=f'q{len(annotations)+1:03d}';crop=im.crop(box);crop_path=out/'crops'/f'{key}.jpg';crop.save(crop_path,quality=95)
            partial=x1==0 or x2==im.width
            # Narrow border fragments are reported separately; identities use temporal context.
            fragment=partial and crop.width<140
            annotations.append(dict(id=key,frame_file=f['file'],frame_index=f['frame_index'],time_seconds=f['time_seconds'],bbox_xyxy=box,class_id=classes[ci],product=names[classes[ci]],crop_file='crops/'+key+'.jpg',truncated=partial,border_fragment=fragment,label_status=specs['status'],identity_evidence='visible packaging plus neighboring frames' if fragment else 'visible packaging',sha256=file_hash(crop_path)))
            draw.rectangle(box,outline='#00ff88',width=6);draw.text((x1+5,y1+5),key,fill='black',stroke_fill='white',stroke_width=2,font_size=32)
            lines.append(f'{ci} {(x1+x2)/2160:.8f} {(y1+y2)/3840:.8f} {(x2-x1)/1080:.8f} {(y2-y1)/1920:.8f}')
            images.append(crop_path)
        overlay.thumbnail((540,960));overlay.save(out/'overlays'/f['file'],quality=90)
        (out/'labels'/(Path(f['file']).stem+'.txt')).write_text('\n'.join(lines)+'\n')
        overlay_cards.append(f'<article><img src="overlays/{f["file"]}" alt="Manual boxes frame {frame_number}"><p>Frame {frame_number} · {f["time_seconds"]:.2f}s</p></article>')
    (out/'annotations.json').write_text(json.dumps(dict(source_sha256=source['source_sha256'],scope='visible instances of the 12 catalog targets in nine selected development frames',coordinate_space='1080x1920 upright frames',annotations=annotations),indent=2))
    recognizer=Recognizer(ROOT/'artifacts/catalog_index')
    reference_hashes={r['sha256'] for r in recognizer.records}
    assert all(r['sha256'] not in reference_hashes for r in annotations)
    vectors=recognizer.encoder.encode(images);results=[]
    for a,v in zip(annotations,vectors):
        p=rank_matches(v,recognizer.vectors,recognizer.records)
        results.append(a|p|dict(correct=p['matches'][0]['class_id']==a['class_id'],top3_correct=a['class_id'] in [m['class_id'] for m in p['matches']]))
    def metrics(rows):
        accepted=[r for r in rows if r['status']=='candidate_match']
        return dict(queries=len(rows),top1_correct=sum(r['correct'] for r in rows),top3_correct=sum(r['top3_correct'] for r in rows),flagged_for_review=sum(r['status']=='review_needed' for r in rows),accepted=len(accepted),accepted_correct=sum(r['correct'] for r in accepted))
    summary=dict(protocol='New development-video crops; fixed reference catalog and unchanged thresholds; manual boxes and identities approved by user',model=recognizer.meta['model'],frame_count=len(specs['frames']),products=len({r['class_id'] for r in results}),all=metrics(results),fully_visible=metrics([r for r in results if not r['truncated']]),partial=metrics([r for r in results if r['truncated']]),border_fragments=metrics([r for r in results if r['border_fragment']]),limitations='Recognition given manual localization, not detector/tracker accuracy. Same physical packages and household setting. Selected development frames are correlated. No held-out rearranged video used. No unknown-product test.',catalog_embeddings_sha256=recognizer.meta['embeddings_sha256'])
    (out/'results.json').write_text(json.dumps(dict(summary=summary,results=results),indent=2))
    cards=[]
    for r in sorted(results,key=lambda r:(r['correct'],r['id'])):
        match=r['matches'][0];label='Correct' if r['correct'] else 'Incorrect';flag='Needs review' if r['status']=='review_needed' else 'Accepted candidate'
        cards.append(f'<article><img src="{r["crop_file"]}" alt="{r["id"]}"><h3>{r["id"]} · {label} · {flag}</h3><p>Manual: {html.escape(r["product"])}</p><p>Predicted: {html.escape(match["product"])}</p><p>Similarity {match["similarity"]:.3f} · margin {r["margin"]:.3f}</p><small>{r["time_seconds"]:.2f}s · '+('border fragment' if r['border_fragment'] else 'partial package' if r['truncated'] else 'full package')+'</small></article>')
    css='<style>body{font:16px system-ui;background:#f1f5f7;color:#18313e;margin:28px}p{line-height:1.5}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}article{background:white;padding:16px;border:1px solid #d1dde3;border-radius:12px}img{width:100%;height:320px;object-fit:contain}small{color:#567}h3{font-size:17px}</style>'
    (out/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Video recognition review</title>'+css+f'<h1>Development video recognition</h1><p>{len(results)} manually boxed crops · {len(specs["frames"])} frames · {summary["products"]} products</p><p>Top-1 correct: {summary["all"]["top1_correct"]}/{len(results)}. Flagged for review: {summary["all"]["flagged_for_review"]}. Incorrect results appear first.</p><p>Manual labels and boxes have been reviewed and approved by the user. No detector was used. Reference catalog and thresholds were unchanged; rearranged videos remain held out.</p><p><a href="boxes.html">Review frame boxes</a> · <a href="results.json">Detailed results</a></p><div class="grid">'+''.join(cards)+'</div></html>',encoding='utf-8')
    apply_file((out/'review.html'))
    (out/'boxes.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Video frame boxes</title>'+css+'<h1>Manual product boxes</h1><p>User-approved labels for all visible catalog targets in these selected frames. Green boxes and query IDs are manual annotations, not detector predictions.</p><div class="grid">'+''.join(overlay_cards)+'</div></html>',encoding='utf-8')
    apply_file((out/'boxes.html'))
    # Compact crop preview for visual QA.
    for start in range(0,len(results),12):
        batch=results[start:start+12];sheet=Image.new('RGB',(1200,((len(batch)+3)//4)*300),'white');d=ImageDraw.Draw(sheet)
        for j,r in enumerate(batch):
            crop=Image.open(out/r['crop_file']);crop.thumbnail((280,245));x=j%4*300;y=j//4*300;sheet.paste(crop,(x+(300-crop.width)//2,y));d.text((x+8,y+255),r['id']+(' OK' if r['correct'] else ' WRONG'),fill='black',font_size=20)
        sheet.save(out/f'query_contact_{start//12+1}.jpg')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
