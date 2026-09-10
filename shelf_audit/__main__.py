import argparse,json
from pathlib import Path
from .recognizer import ROOT,Recognizer,build_index

def main():
    parser=argparse.ArgumentParser(description='ShelfAudit product crop recognizer')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['index','predict','evaluate','serve']:
        p=sub.add_parser(name)
        p.add_argument('--index',type=Path,default=ROOT/'artifacts/catalog_index')
        p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
        if name=='index':p.add_argument('--catalog',type=Path,default=ROOT/'data/catalog_crops')
        if name=='predict':
            p.add_argument('image',type=Path)
            p.add_argument('--min-score',type=float,default=.65)
            p.add_argument('--min-margin',type=float,default=.03)
        if name=='evaluate':p.add_argument('--output',type=Path,default=ROOT/'outputs/reference_evaluation.json')
        if name=='serve':p.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    if args.command=='index':result=build_index(args.catalog,args.index,args.device)
    elif args.command=='predict':result=Recognizer(args.index,args.device).predict(args.image,min_score=args.min_score,min_margin=args.min_margin)
    elif args.command=='evaluate':
        result=Recognizer(args.index,load_encoder=False).evaluate()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        result={k:v for k,v in result.items() if k!='results'}|{'report':str(args.output)}
    else:
        from .server import serve
        serve(args.index,args.device,args.port);return
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
