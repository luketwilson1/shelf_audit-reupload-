"""Extract timestamped development frames without reading held-out videos."""
from .report_style import apply_file
import argparse,json,hashlib,html
from pathlib import Path
import cv2
from PIL import Image,ImageDraw

def sample(video,out,interval=.5):
    video=Path(video);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if interval<=0:raise ValueError('Interval must be positive')
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError('Cannot open video')
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,1)
    fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rotation=cap.get(cv2.CAP_PROP_ORIENTATION_META)
    rows=[];next_time=0.;n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        timestamp=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if timestamp+1e-6>=next_time:
            name=f'frame_{n:06d}_{timestamp:07.3f}s.jpg'
            if not cv2.imwrite(str(out/name),frame,[cv2.IMWRITE_JPEG_QUALITY,95]):raise OSError(name)
            rows.append(dict(file=name,frame_index=n,time_seconds=timestamp,width=frame.shape[1],height=frame.shape[0],laplacian_variance=float(cv2.Laplacian(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())))
            next_time=timestamp+interval
        n+=1
    cap.release()
    if not rows:raise ValueError('No decoded frames')
    meta=dict(source=video.name,source_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),split='development',nominal_fps=fps,reported_frame_count=count,decoded_frames=n,rotation_metadata=rotation,duration_estimate_seconds=n/fps if fps else None,sample_interval_seconds=interval,frames=rows)
    (out/'manifest.json').write_text(json.dumps(meta,indent=2))
    cards=[]
    for start in range(0,len(rows),12):
        batch=rows[start:start+12];sheet=Image.new('RGB',(1440,((len(batch)+3)//4)*330),'white');d=ImageDraw.Draw(sheet)
        for j,r in enumerate(batch):
            with Image.open(out/r['file']) as im:
                im.thumbnail((350,280));x=j%4*360;y=j//4*330;sheet.paste(im,(x+(360-im.width)//2,y+(280-im.height)//2))
            d.text((x+10,y+285),f"{r['time_seconds']:.2f}s / frame {r['frame_index']}",fill='black',font_size=20)
        sheet.save(out/f'contact_{start//12+1}.jpg',quality=90)
    for r in rows:
        cards.append(f'<article><a href="{r["file"]}"><img loading="lazy" src="{r["file"]}" alt="Frame at {r["time_seconds"]:.2f} seconds"></a><p>{r["time_seconds"]:.2f}s · frame {r["frame_index"]} · {r["width"]} × {r["height"]}</p></article>')
    (out/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Original video frame review</title><style>body{font:16px system-ui;background:#edf2f4;margin:28px;color:#162e38}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}article{background:white;padding:12px;border-radius:10px}img{width:100%;height:300px;object-fit:contain}</style><h1>'+html.escape(video.name)+' · Development frames</h1><p>Sampled about every '+str(interval)+' seconds. Click a frame for full resolution. These frames have no detector annotations yet.</p><div class="grid">'+''.join(cards)+'</div></html>',encoding='utf-8')
    apply_file((out/'review.html'))
    print(json.dumps({k:v for k,v in meta.items() if k!='frames'}|{'samples':len(rows),'frame_size':[rows[0]['width'],rows[0]['height']]}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--interval',type=float,default=.5);a=p.parse_args();sample(a.video,a.out,a.interval)
