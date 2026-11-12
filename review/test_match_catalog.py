import tempfile,json,io
from pathlib import Path
import match_catalog as m
class Handler:
 def __init__(self):self.status=None;self.payload=None;self.wfile=io.BytesIO()
 def send(self,status,payload):self.status=status;self.payload=payload
 def send_response(self,status):self.status=status
 def send_header(self,*args):pass
 def end_headers(self):pass
with tempfile.TemporaryDirectory() as temp:
 m.ROOT=Path(temp);h=Handler();m.serve(h,'match-catalog',{});assert h.status==503
 (m.ROOT/'catalog.json').write_text(json.dumps(dict(total_tracks=2,tracks=[{'id':1},{'id':2}],ordered_pairs=2,pairs=[dict(source=1,target=2),dict(source=2,target=1)])))
 h=Handler();m.serve(h,'match-catalog',{'summary':['1']});assert h.status==200 and 'pairs' not in h.payload and h.payload['total_tracks']==2
 h=Handler();m.serve(h,'match-catalog',{'source':['1']});assert h.payload['pairs']==[dict(source=1,target=2)]
 h=Handler();m.serve(h,'match-catalog',{'source':['invalid']});assert h.status==400
 h=Handler();m.serve(h,'matches',{});assert h.status==200 and b'Vocal + tempo candidates' in h.wfile.getvalue()
print('PASS: catalog summary, source filtering, invalid query, pending catalog and page routes.')
