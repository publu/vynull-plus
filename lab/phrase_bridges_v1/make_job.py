"""On-demand loop bridge, with no silent fallback to an earlier straight fade."""
from pathlib import Path
import sys,json,argparse
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from engine import generate,Separator,r,joblib,TRAIN,write
from analyze import analyze
from publish import publish

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=int,required=True);p.add_argument('--target',type=int,required=True);p.add_argument('--bars',type=int,choices=[8,16],default=8);p.add_argument('--entry-loop',type=int,choices=[0,2,4],default=0);p.add_argument('--job-id',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();errors=[]
 for length in ([a.entry_loop] if a.entry_loop else [4,2]):
  try:result=generate(a.source,a.target,(a.bars,),(length,),a.job_id);break
  except ValueError as e:errors.append(str(e))
 else:raise ValueError(errors[-1] if errors else 'No usable incoming instrumental loop found')
 separator=Separator();tracks={t['id']:t for t in r.api('tracks')};gate=joblib.load(TRAIN/'part_alignment_v1/artifacts/beat_repair_gate.joblib');entry=analyze(result['previews'][0],separator,tracks,gate);published=publish([entry])[0];write(a.output,published);print('LOOP_BRIDGE_READY',published['audio']['sha256'],flush=True)
if __name__=='__main__':main()
