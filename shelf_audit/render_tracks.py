"""Render offline track IDs, stabilized identities, and representative crops."""
from .report_style import apply_file
import argparse
import html
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw
from .recognizer import file_hash


def render(source, directory):
    directory=Path(directory)
    data=json.loads((directory/'results.json').read_text())
    if file_hash(source)!=data['source']['source_sha256']:raise ValueError('Wrong source video')
    tracks={t['id']:t for t in data['tracks']}
    best={}
    for tid,t in tracks.items():
        if not t['persistent']:continue
        eligible=[o for o in t['observations'] if o['recognition']['predicted_class']==t['class_id']] or t['observations']
        best[tid]=max(eligible,key=lambda o:o['detector_score']*max(0,o['recognition']['matches'][0]['similarity']))
    (directory/'crops').mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,1)
    encoder=None;count=0
    try:
        for frame in data['frames']:
            ok,bgr=cap.read()
            if not ok:raise ValueError('Source ended before predictions')
            full=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im=full.copy();im.thumbnail((540,960))
            for tid,o in best.items():
                if o['frame_index']==frame['frame_index']:full.crop(o['bbox_xyxy']).save(directory/'crops'/f'track_{tid}.jpg',quality=92)
            draw=ImageDraw.Draw(im)
            for link in frame['assignments']:
                t=tracks[link['track_id']]
                o=next(o for o in t['observations'] if o['frame_index']==frame['frame_index'])
                box=[v*(im.width/full.width if j%2==0 else im.height/full.height) for j,v in enumerate(o['bbox_xyxy'])]
                stable=t['status']=='stable_candidate';color='#70ebb4' if stable else '#ffcb73'
                name=(t['class_id'] or 'unknown — review').removeprefix('great_value_').replace('_',' ')
                label=f'T{t["id"]}: '+name
                draw.rectangle(box,outline=color,width=3)
                x=min(box[0],220);y=max(25,box[1]-32)
                for j,line in enumerate([label[:38],label[38:]]):
                    if line:
                        draw.rectangle((x,y+j*16,x+min(320,len(line)*8),y+j*16+16),fill='#10212b')
                        draw.text((x+2,y+j*16),line,fill=color,font_size=13)
            draw.rectangle((0,0,im.width,23),fill='#10212b')
            draw.text((5,3),f'OFFLINE TRACKS | {frame["time_seconds"]:.2f}s',fill='white',font_size=14)
            if encoder is None:
                encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{im.width}x{im.height}','-r',str(data['source']['nominal_fps']),'-i','-','-an','-c:v','libx264','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(directory/'tracked.mp4')],stdin=subprocess.PIPE)
            encoder.stdin.write(np.asarray(im).tobytes());count+=1
    finally:
        cap.release()
        if encoder:
            encoder.stdin.close()
            if encoder.wait():raise RuntimeError('Encoding failed')
    page=directory/'review.html'
    text=page.read_text(encoding='utf-8')
    cards=''
    for tid in data['identified_order']['track_order'] or []:
        cards+=f'<figure style="display:inline-block;width:145px;vertical-align:top;margin:8px"><img style="width:145px;height:180px;object-fit:contain" src="crops/track_{tid}.jpg"><figcaption>T{tid}: '+html.escape(tracks[tid]['class_id'].removeprefix('great_value_').replace('_',' '))+'</figcaption></figure>'
    unknown=''.join(f'<p>Track {tid}: {t["detection_count"]} detections, {t["first_seconds"]:.2f}–{t["last_seconds"]:.2f}s; unknown identity.</p><img style="height:180px;max-width:300px;object-fit:contain" src="crops/track_{tid}.jpg">' for tid,t in tracks.items() if t['persistent'] and t['status']!='stable_candidate')
    text=text.replace('</h1>','</h1><video controls playsinline src="tracked.mp4"></video><p>Track identities use the whole saved video, including later observations. This is offline smoothing, not live recognition.</p><div>'+cards+'</div>',1)
    text=text.replace('</html>','<h2>Unresolved persistent tracks</h2>'+unknown+'</html>')
    page.write_text(text,encoding='utf-8')
    apply_file(page)
    print(f'Rendered {count} frames and {len(best)} track crops.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('directory',type=Path)
    a=p.parse_args();render(a.source,a.directory)
