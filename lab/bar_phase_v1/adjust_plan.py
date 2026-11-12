"""Opt-in preview-plan transformation; no device or library grid commands."""
from pathlib import Path
import argparse,json
from inference import PreviewPhaseAligner
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=PreviewPhaseAligner().plan(json.loads(args.plan.read_text()));temp=args.output.with_suffix('.tmp.json')
    temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(args.output)
    print(json.dumps(result['selection']['bar_phase_alignment']),flush=True)
