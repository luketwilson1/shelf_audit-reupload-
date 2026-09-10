"""Local-only upload demo. Query images stay in memory."""
import base64,io,json,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from PIL import Image,ImageOps,UnidentifiedImageError
from .recognizer import Recognizer

def serve(index,device,port):
    recognizer=Recognizer(index,device)
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,body,kind='application/json'):
            if isinstance(body,dict):body=json.dumps(body).encode()
            self.send_response(status);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            path=urlsplit(self.path).path
            if path=='/':return self.send(200,Path(__file__).with_name('demo.html').read_bytes(),'text/html; charset=utf-8')
            if path=='/api/info':
                return self.send(200,dict(references=len(recognizer.records),products=len({r['class_id'] for r in recognizer.records}),device=recognizer.encoder.device,model='ResNet-50',ready=True))
            if path.startswith('/reference/'):
                try:
                    n=int(path.rsplit('/',1)[-1])
                    if not 0<=n<len(recognizer.records):raise ValueError()
                    with Image.open(recognizer.catalog/recognizer.records[n]['crop_file']) as im:
                        im.thumbnail((450,450));buf=io.BytesIO();im.convert('RGB').save(buf,format='JPEG',quality=85)
                    return self.send(200,buf.getvalue(),'image/jpeg')
                except (ValueError,OSError):pass
            return self.send(404,{'error':'Not found'})
        def do_POST(self):
            if self.path!='/api/recognize':return self.send(404,{'error':'Not found'})
            # No cross-origin uploads to the local service.
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{port}',f'http://localhost:{port}'):
                return self.send(403,{'error':'Origin not allowed'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length<1 or length>20*1024*1024:return self.send(413,{'error':'Use a JPEG or PNG under 14 MB.'})
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict) or not isinstance(payload.get('image'),str):
                    raise ValueError('Supply a JSON object with a base64 image string.')
                raw=base64.b64decode(payload['image'],validate=True)
                with Image.open(io.BytesIO(raw)) as im:
                    if im.width*im.height>24_000_000:raise ValueError('Image is too large. Use at most 24 megapixels.')
                    query=ImageOps.exif_transpose(im).convert('RGB')
                with lock:
                    result=recognizer.predict(query,min_score=float(payload.get('min_score',.65)),min_margin=float(payload.get('min_margin',.03)))
                return self.send(200,result)
            except (ValueError,TypeError,KeyError,UnidentifiedImageError,OSError,Image.DecompressionBombError) as error:
                return self.send(400,{'error':str(error)})
    print(f'Recognizer ready: http://127.0.0.1:{port} | {recognizer.encoder.device}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()
