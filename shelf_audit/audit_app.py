"""Local video-upload demo with one GPU job at a time."""
import argparse
import json
import mimetypes
import re
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
from .recognizer import ROOT

RUNS=ROOT/'outputs/app_runs'
RESULT_ROOT=ROOT/'outputs'
LOCK=threading.Lock()
JOBS={}


def safe_result(path):
    target=(RESULT_ROOT/unquote(path).replace('\\','/')).resolve()
    if not target.is_relative_to(RESULT_ROOT.resolve()):raise ValueError('Invalid path')
    return target


def worker(job, directory):
    try:
        commands=[('Detecting and recognizing packages',['shelf_audit.process_video',str(directory/'upload.mov'),'--out',str(directory/'detection'),'--checkpoint',str(ROOT/'artifacts/product_detector/corrected.pt')]),
                  ('Cleaning overlapping boxes',['shelf_audit.refine_predictions',str(directory/'detection/results.json'),str(directory/'refined.json')]),
                  ('Tracking and reconstructing order',['shelf_audit.track_video',str(directory/'refined.json'),'--out',str(directory/'tracking'),'--reconnect']),
                  ('Rendering tracked playback',['shelf_audit.render_tracks',str(directory/'upload.mov'),str(directory/'tracking')])]
        with (directory/'run.log').open('w',encoding='utf-8') as log:
            for stage,args in commands:
                job.update(stage=stage)
                subprocess.run([sys.executable,'-m',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        # The general-purpose renderers contain historical dataset wording.
        # An arbitrary upload has unknown training provenance: do not call it a test.
        for relative in ['detection/results.json','refined.json','tracking/results.json']:
            path=directory/relative;result=json.loads(path.read_text())
            summary=result.get('summary',result.get('source'))
            summary['limitations']='User-uploaded video; training provenance unknown. Candidate predictions, not verified inventory counts.'
            if 'limitations' in result:result['limitations']='Offline single-row heuristic tracking. Upload training provenance unknown. Review candidate identities, order, and duplicates; votes are not calibrated probabilities.'
            path.write_text(json.dumps(result,indent=2))
        for relative in ['detection/review.html','tracking/review.html']:
            path=directory/relative;text=path.read_text(encoding='utf-8')
            text=text.replace('Development footage, including training frames.','Uploaded video; training provenance unknown.').replace('Input includes training frames. Rearranged videos unused.','Uploaded video; training provenance unknown.')
            path.write_text(text,encoding='utf-8')
        job.update(status='complete',stage='Ready for review',url=f'/results/app_runs/{directory.name}/tracking/review.html')
    except Exception:
        job.update(status='failed',stage='Processing failed. Check that the upload is a readable video.',log=f'/results/app_runs/{directory.name}/run.log')
    finally:LOCK.release()


def serve(port=8766):
    RUNS.mkdir(parents=True,exist_ok=True)
    class Handler(BaseHTTPRequestHandler):
        def respond(self,status,body,kind='application/json'):
            if not isinstance(body,bytes):body=json.dumps(body).encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)

        def do_GET(self):
            path=urlsplit(self.path).path
            if path=='/':return self.respond(200,Path(__file__).with_name('audit_app.html').read_bytes(),'text/html; charset=utf-8')
            if path=='/api/info':return self.respond(200,{'ready':(ROOT/'artifacts/product_detector/corrected.pt').exists() and (ROOT/'artifacts/catalog_index/index.json').exists(),'busy':LOCK.locked()})
            if path.startswith('/api/jobs/'):
                job=JOBS.get(path.rsplit('/',1)[-1]);return self.respond(200,job) if job else self.respond(404,{'error':'Job not found'})
            if path.startswith('/results/'):
                try:target=safe_result(path[len('/results/'):])
                except ValueError:return self.respond(403,{'error':'Invalid path'})
                if not target.is_file():return self.respond(404,{'error':'Not found'})
                size=target.stat().st_size;start=0;end=size-1;status=200
                requested=self.headers.get('Range')
                if requested:
                    match=re.fullmatch(r'bytes=(\d+)-(\d*)',requested)
                    if not match:return self.respond(416,{'error':'Unsupported range'})
                    start=int(match[1]);end=min(int(match[2]),size-1) if match[2] else size-1
                    if start>end or start>=size:return self.respond(416,{'error':'Invalid range'})
                    status=206
                self.send_response(status);self.send_header('Content-Type',mimetypes.guess_type(target.name)[0] or 'application/octet-stream');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
                if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
                self.end_headers()
                try:
                    with target.open('rb') as f:
                        f.seek(start);remaining=end-start+1
                        while remaining:
                            chunk=f.read(min(1024*1024,remaining))
                            if not chunk:break
                            self.wfile.write(chunk);remaining-=len(chunk)
                except (BrokenPipeError,ConnectionResetError):pass
                return
            self.respond(404,{'error':'Not found'})

        def do_POST(self):
            if self.path!='/api/jobs':return self.respond(404,{'error':'Not found'})
            if self.headers.get('Origin') not in (None,f'http://127.0.0.1:{port}',f'http://localhost:{port}'):return self.respond(403,{'error':'Origin not allowed'})
            try:length=int(self.headers.get('Content-Length','0'))
            except ValueError:return self.respond(400,{'error':'Invalid upload length'})
            if not 0<length<=256*1024*1024:return self.respond(413,{'error':'Choose a video under 256 MB.'})
            if not LOCK.acquire(blocking=False):return self.respond(409,{'error':'A video is already processing. Please wait.'})
            try:
                self.connection.settimeout(120)
                key=uuid.uuid4().hex;directory=RUNS/key;directory.mkdir()
                with (directory/'upload.mov').open('wb') as f:
                    remaining=length
                    while remaining:
                        chunk=self.rfile.read(min(1024*1024,remaining))
                        if not chunk:raise ValueError('Incomplete upload')
                        f.write(chunk);remaining-=len(chunk)
                job={'id':key,'status':'running','stage':'Starting'};JOBS[key]=job
                threading.Thread(target=worker,args=(job,directory),daemon=True).start()
            except Exception:
                LOCK.release();return self.respond(400,{'error':'Upload failed'})
            self.respond(202,job)
    print(f'ShelfAudit video demo: http://127.0.0.1:{port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8766);serve(p.parse_args().port)
