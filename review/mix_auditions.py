"""Local model-selected mix auditions and versioned listening feedback."""
import json,hashlib,re,sqlite3,time,mimetypes
from pathlib import Path
import os
ROOT=Path(os.environ.get('VYNULL_LAB_DIR',Path(__file__).resolve().parent.parent/'lab'))/'section_pair_v1/renders'

def previews():
 manifest=ROOT/'manifest.json'
 if not manifest.exists():return []
 raw=manifest.read_bytes();version=hashlib.sha256(raw).hexdigest();data=json.loads(raw);out=[]
 for index,item in enumerate(data.get('renders',[])):
  path=Path(item['audio']['path']).resolve()
  if not path.is_relative_to(ROOT.resolve()) or not path.is_file():continue
  variant='Loop-assisted' if item['engine'].get('loop') else 'EQ blend' if item['eq_enabled'] else 'Equal-power baseline'
  # Content identity survives appending other auditions; plans and immutable
  # audio metadata still change the ID, so a stale vote cannot hit a new mix.
  identity=hashlib.sha256(json.dumps(item,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]
  alignment=item.get('engine',{}).get('source_alignment')
  superseded=bool(alignment and not alignment.get('origin_anchor_omitted'))
  out.append(dict(item,preview_id=identity,manifest_sha256=version,variant=item.get('comparison_label',variant),superseded=superseded))
 return out

def review_key(preview):
 # Manifest growth changes public IDs; recover ratings only for the identical plan/audio.
 return json.dumps({k:preview.get(k) for k in ('selection','engine','audio','eq_enabled','comparison')},sort_keys=True)

def verified_analyses(items,filename,version,metric_version=None,folder='library_lab_v1'):
 # Sidecar data keeps existing preview identities and saved ratings intact.
 path=ROOT.parent.parent/folder/filename
 if not path.exists():return {}
 try:report=json.loads(path.read_text())
 except (OSError,ValueError):return {}
 if (report.get('version')!=version or metric_version and report.get('metric_version')!=metric_version or not report.get('all_reconstructions_bit_exact')):return {}
 result={}
 for item in items:
  stored=report.get('previews',{}).get(item.get('audio',{}).get('sha256'))
  for entry in stored if isinstance(stored,list) else [stored]:
   if (entry and entry.get('reconstruction_pcm_max_difference')==0
       and entry.get('preview_snapshot_sha256')==hashlib.sha256(review_key(item).encode()).hexdigest()):
    result[item['preview_id']]=entry;break
 return result

def overlap_analyses(items):
 result=verified_analyses(items,'overlap-analysis.json','exact_overlap_librosa_v1','simultaneous_frame_chroma_and_rendered_onsets_v2')
 result.update(verified_analyses(items,'final-overlap-analysis.json','exact_overlap_librosa_v1','simultaneous_frame_chroma_and_rendered_onsets_v2',folder='discovery_rounds_v1'))
 result.update(verified_analyses(items,'public-overlap-analysis.json','exact_overlap_librosa_v1','simultaneous_frame_chroma_and_rendered_onsets_v2',folder='library_search_v2'))
 result.update(verified_analyses(items,'overlap.json','exact_overlap_librosa_v1','simultaneous_frame_chroma_and_rendered_onsets_v2',folder='phrase_bridges_v1'))
 return result

def deep_merge_analyses(items):
 result=verified_analyses(items,'deep-merge-analysis-v2.json','deep_merge_librosa_v2')
 result.update(verified_analyses(items,'final-deep-analysis.json','deep_merge_librosa_v2',folder='discovery_rounds_v1'))
 result.update(verified_analyses(items,'public-deep-analysis.json','deep_merge_librosa_v2',folder='library_search_v2'))
 result.update(verified_analyses(items,'deep.json','deep_merge_librosa_v2',folder='phrase_bridges_v1'))
 return result

def vocal_analyses(items):
 result=verified_analyses(items,'public-vocal-audit.json','vocal_overlap_v1',folder='library_search_v2')
 result.update(verified_analyses(items,'vocals.json','vocal_overlap_v1',folder='phrase_bridges_v1'))
 return result

def phrase_analyses(items):
 return verified_analyses(items,'structure.json','phrase_bridge_structure_v1',folder='phrase_bridges_v1')

def latest_reviews(database,items):
 with sqlite3.connect(database) as c:
  rows=c.execute('SELECT id,vote,tags,note,preview_snapshot,created FROM transition_reviews ORDER BY id DESC').fetchall()
 latest={};by_audio={}
 for ident,vote,tags,note,snapshot,created in rows:
  key=review_key(json.loads(snapshot))
  value={'review_id':ident,'vote':vote,'tags':json.loads(tags),'note':note,'created':created}
  if key not in latest:latest[key]=value
  digest=json.loads(snapshot).get('audio',{}).get('sha256','')
  if re.fullmatch('[0-9a-f]{64}',digest) and digest not in by_audio:by_audio[digest]=value
 result={}
 for p in items:
  exact=latest.get(review_key(p));audio=by_audio.get(p.get('audio',{}).get('sha256'));choices=[v for v in [exact,audio] if v]
  if choices:
   chosen=max(choices,key=lambda v:v['review_id']);result[p['preview_id']]=dict(chosen,match='exact_render' if chosen is exact else 'identical_audio')
 return result

def install_database(database):
 with sqlite3.connect(database) as c:c.execute('CREATE TABLE IF NOT EXISTS transition_reviews(id INTEGER PRIMARY KEY, preview_id TEXT NOT NULL, vote INTEGER NOT NULL, tags TEXT NOT NULL, note TEXT NOT NULL, preview_snapshot TEXT NOT NULL, created REAL NOT NULL)')

def page():
 return r'''<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Vynull · Mix auditions</title><style>
body{margin:0;background:#0d0e11;color:#eee;font:15px/1.5 system-ui}main{max-width:1080px;margin:auto;padding:40px 24px}a{color:#ff922d}h1{font-size:34px;margin-bottom:8px}.intro{color:#aaa;max-width:780px}article{border:1px solid #333740;background:#14161b;padding:24px;margin:20px 0;border-radius:8px}h2{font-size:19px;margin:0 0 8px}.muted{color:#a2a6b0;font-size:13px}audio{width:100%;margin:16px 0}button{background:#222631;border:1px solid #545b6a;color:#eee;border-radius:5px;padding:10px 16px;cursor:pointer;margin:4px 6px 4px 0}button:hover,button.selected{border-color:#ff881c;color:#ff9a39}select,textarea{background:#0d0f13;color:#ddd;border:1px solid #444b59;border-radius:4px;padding:10px;box-sizing:border-box}textarea{display:block;width:100%;margin:10px 0;resize:vertical}.status{color:#8ce0aa;font-size:13px}.badge{color:#ffad61}small{display:block;color:#888;margin-top:12px}
.bridge-plan{border:1px solid #cc873f;padding:18px;margin:16px 0}.bridge-timeline{display:flex;gap:2px;margin:14px 0;font-size:11px;min-height:40px}.bridge-timeline span{background:#31414c;padding:8px 4px}.bridge-timeline span:nth-child(2){background:#735326}.bridge-timeline span:nth-child(3){background:#29605a}.bridge-timeline span:nth-child(4){background:#385232}.vocal-check{border:1px solid #355146;padding:14px 16px;margin:16px 0}.vocal-watch{border-color:#9b683e;background:#241b15}.vocal-check p{margin:6px 0}.merge-check{border-left:3px solid #9b754d;padding:0 16px;margin:16px 0}h3{font-size:16px;margin:8px 0}.overlap-details{margin:12px 0 20px}summary{cursor:pointer;color:#ffad61}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:8px;border-bottom:1px solid #333740;white-space:nowrap}.preview-builder{border:1px solid #333740;background:#14161b;padding:24px;margin:24px 0}.builder-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.preview-builder label{display:flex;flex-direction:column;gap:6px;color:#c5c7cf}.preview-builder select{width:100%}.preview-builder .phase-option{flex-direction:row;align-items:center;margin:16px 0}.preview-builder .phase-option input{margin-right:6px}.builder-status{margin-top:12px}
</style></head><body><main><a href="/">← Library</a><h1>Mix auditions</h1><p class="intro">Choose a pair and compare loop bridges. An instrumental loop carries the handoff; the new vocal enters after the old track is out. Playback uses your browser.</p><p class="muted">Candidate transitions with loop, timing and vocal checks. Listening feedback is saved against the exact version.</p><p><a href="/api/review/matches">Find matches · vocals and timing</a></p><div id="collections"></div><p class="muted"><a href="?earlier=1">Include earlier renderer experiments</a></p><form id="make-preview" class="preview-builder"><h2>Make a mix preview</h2><div class="builder-grid"><label for="preview-source">Outgoing song<select id="preview-source" required><option value="">Loading songs…</option></select></label><label for="preview-target">Incoming song<select id="preview-target" required><option value="">Loading songs…</option></select></label><label for="preview-style">Transition<select id="preview-style"><option value="bridge">Instrumental loop bridge</option><option value="blend">Earlier straight blend</option></select></label><label for="preview-entry-loop">Incoming instrumental loop<select id="preview-entry-loop"><option value="0">Find a clean loop</option><option value="4">4 bars</option><option value="2">2 bars</option></select></label><label for="preview-method" class="legacy-control">Part selection<select id="preview-method"><option value="semantic">Music model</option><option value="measured">Sound similarity</option></select></label><label for="preview-bars">Blend length<select id="preview-bars"><option value="8">8 bars</option><option value="4" id="four-bar-overlap" disabled>4 bars (straight blend)</option><option value="16">16 bars</option></select></label><label for="preview-loop" class="legacy-control">Outgoing loop<select id="preview-loop"><option value="0">None</option><option value="4">4 bars</option><option value="8">8 bars</option></select></label></div><label class="phase-option legacy-control"><input id="preview-feedback" type="checkbox" checked>Use my listening feedback</label><p class="muted legacy-control">Uses saved opinions about these song sections. Turn it off to try a fresh model choice.</p><label class="phase-option legacy-control"><input id="preview-phase" type="checkbox">Check bar alignment for this preview</label><p class="muted legacy-control">The bar check adjusts preview cues only. Library beat grids stay as saved.</p><button id="preview-submit" type="submit" disabled>MAKE PREVIEW</button><div id="preview-status" class="builder-status" role="status" aria-live="polite"></div></form><div id="items">Loading auditions…</div></main><script>
const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const stamp=s=>`${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;
const cueStamp=s=>{const t=Math.round(s*10);return `${Math.floor(t/600)}:${(t%600/10).toFixed(1).padStart(4,'0')}`;};
let items=[];
const query=new URLSearchParams(location.search);
const comparisonId=query.get('comparison'),collectionId=query.get('collection');
const number=(x,d=2)=>x==null?'—':Number(x).toFixed(d);
function bridgeHtml(p,d,index){
 if(!d)return '';
 const s=p.selection,out=d.source_out_seconds,release=d.loop_release_seconds,voice=d.incoming_vocal_seconds,total=p.audio.seconds;
 const stages=[['Old song',8*240/s.mix_bpm],['Loop underneath',out-8*240/s.mix_bpm],['Instrumental alone',release-out],['New phrase',total-release]];
 return `<div class="bridge-plan"><strong>INSTRUMENTAL LOOP BRIDGE</strong><p>Loop the incoming track at <b>${cueStamp(d.incoming_native_loop_start)}–${cueStamp(d.incoming_native_loop_end)}</b> (${d.loop_bars} bars). Finish the outgoing phrase, hold the loop alone, then release into the original next phrase.</p><div class="bridge-timeline">${stages.map(([name,seconds])=>`<span style="flex:${Math.max(.1,seconds)}">${name}</span>`).join('')}</div><p class="muted">Old track out at ${stamp(out)} · loop release at ${stamp(release)} · new vocal at ${stamp(voice)} (${Number(d.incoming_vocal_gap_bars).toFixed(1)} bars after the old track is gone).</p><button type="button" data-seek-index="${index}" data-seek-seconds="${Math.max(0,d.source_fade_begins_seconds-2)}">HEAR THE HANDOFF</button><button type="button" data-seek-index="${index}" data-seek-seconds="${out+.1}">HEAR THE LOOP ALONE</button><button type="button" data-seek-index="${index}" data-seek-seconds="${Math.max(0,voice-2)}">HEAR THE VOCAL ENTRY</button></div>`;
}
function vocalHtml(p,v,index){
 if(!v)return '<p class="muted">Vocal overlap has not been checked for this rendering.</p>';
 const watch=v.status==='vocal_overlap_watch';
 return `<div class="vocal-check ${watch?'vocal-watch':''}"><strong>${watch?'VOCAL OVERLAP WARNING':'VOCAL CHECK'}</strong><p>${escape(v.text)}</p>${watch?v.intervals.slice(0,3).map(point=>`<button type="button" data-seek-index="${index}" data-seek-seconds="${Number(point.preview_seconds)}">Hear merge bar ${point.merge_bar}</button>`).join(''):''}<small>Estimated from separated vocals after the actual mix gains and EQ. Backing vocals or instrument leakage can trigger a warning; this is a prompt to listen.</small></div>`;
}
function mergeHtml(p,d){
 if(!d)return '';
 const a=d.source_analysis,b=d.target_analysis,r=d.rhythm,h=d.harmony;
 const candidates=k=>k.top.map(v=>escape(v.key)+' ('+number(v.correlation)+')').join(', ');
 return `<div class="merge-check"><h3>Beat and key check</h3><p><strong>Timing:</strong> ${escape(r.text)}</p><p class="muted">Renderer target: ${number(r.renderer_target_bpm,1)} BPM for both. Low-drum pulse estimates near that target: ${number(a.near_renderer_bpm.bpm)} → ${number(b.near_renderer_bpm.bpm)} BPM. These can be uncertain when the rhythm changes or the drums are sparse.</p><p><strong>Keys:</strong> ${escape(h.text)}</p><p class="muted">Saved section keys: ${escape(p.selection.measured_source_key||'Unknown')} → ${escape(p.selection.measured_target_key||'Unknown')}. The independent estimate above can disagree when the local passage is ambiguous.</p><details><summary>Drum timing and local pitch evidence</summary><p class="muted">Independent broad tempo estimates (may follow half or double time): ${number(a.unconstrained_tempo.full,1)} → ${number(b.unconstrained_tempo.full,1)} BPM. Estimated relative tuning: ${number(h.relative_tuning_cents,0)} cents. Source key candidates: ${candidates(h.source)}. Incoming: ${candidates(h.target)}. Key values in parentheses are profile correlations.</p><div style="overflow-x:auto"><table><thead><tr><th>Merge bar</th><th>Low-drum offset</th><th>Outgoing tonal candidate</th><th>Incoming tonal candidate</th></tr></thead><tbody>${h.bars.map(bar=>{const phase=r.bar_offsets.find(v=>v.bar===bar.bar);return `<tr><td>${bar.bar}</td><td>${phase?number(phase.offset_beats*60/r.renderer_target_bpm*1000,0)+' ms':'—'}</td><td>${escape(bar.source_key.key||'Unclear')}${bar.source_key.confident?'':' ?'}</td><td>${escape(bar.target_key.key||'Unclear')}${bar.target_key.confident?'':' ?'}</td></tr>`;}).join('')}</tbody></table></div><p class="muted">Timing flags are prompts to listen: syncopation can create a deliberate offset. A related key does not guarantee that the overlapping chords or vocals work.</p></details></div>`;
}
function overlapHtml(p,analysis,index){
 if(!analysis)return '';
 const summary=analysis.summary;
 return `<details class="overlap-details"><summary>What overlaps, bar by bar · ${analysis.overlap_bars} bars</summary><p class="muted">Measured on the actual aligned audio, including loop repeats, EQ, fades and echo. Harmony compares simultaneous pitch classes frame by frame; bass overlap measures simultaneous bass energy. These describe the sound and do not decide whether the mix works. “Both present” means the two streams are within 12 dB of each other.</p><p class="muted">Whole overlap: harmony ${number(summary.raw.frame_harmonic_chroma_cosine)} before EQ/fades → ${number(summary.rendered.frame_harmonic_chroma_cosine)} with the rendered gains · bass overlap ${number(summary.rendered.simultaneous_bass_ratio)} · texture distance ${number(summary.raw.mfcc_distance)} before EQ/fades.</p><div style="overflow-x:auto"><table><thead><tr><th>Jump to bar</th><th>Harmony<br>raw / rendered</th><th>Both present</th><th>Bass overlap</th><th>Attack correlation</th><th>Mix level</th></tr></thead><tbody>${analysis.bars.map(b=>`<tr><td><button type="button" data-seek-index="${index}" data-seek-seconds="${Number(b.preview_seconds)}">${b.bar}${b.source_loop_cycle?' · out loop '+b.source_loop_cycle:b.target_loop_cycle?' · in loop '+b.target_loop_cycle:''}</button></td><td>${number(b.raw.frame_harmonic_chroma_cosine)} / ${number(b.rendered.frame_harmonic_chroma_cosine)}</td><td>${number(b.rendered.coaudible_fraction*100,0)}%</td><td>${number(b.rendered.simultaneous_bass_ratio)}</td><td>${number(b.rendered.onset_correlation)}</td><td>${number(b.mix_rms_dbfs,1)} dBFS</td></tr>`).join('')}</tbody></table></div><p class="muted">Harmony and bass values run from 0 to 1; attack correlation runs from −1 to 1. Higher values are not a musical-quality rating. Click a bar to seek the player, then press play.</p></details>`;
}
function collectionOverview(data){
 const fresh=data.previews.filter(p=>p.comparison?.collection_id);
 const collections=[...new Set(fresh.map(p=>p.comparison.collection_id))];
 document.getElementById('collections').innerHTML=collections.length?'<p>'+collections.map(id=>`<a href="?collection=${encodeURIComponent(id)}">${escape(id==='wide-library-v1'?'Around the library · eight mixing ideas':id==='discovery-rounds-v1'?'Earlier search rounds · all candidates':id==='phrase-bridges-v1'?'Loop bridges · vocals after the handoff':id)}</a>`).join(' · ')+'</p>':'';
 if(!collectionId||comparisonId)return false;
 const chosen=fresh.filter(p=>p.comparison.collection_id===collectionId),groups=new Map();
 for(const p of chosen){const id=p.comparison.group_id;if(!groups.has(id))groups.set(id,[]);groups.get(id).push(p);}
 const ordered=[...groups.values()].sort((a,b)=>a[0].comparison.experiment_order-b[0].comparison.experiment_order);
 const songs=new Set(chosen.flatMap(p=>[p.selection.source_track,p.selection.target_track]));
 document.querySelector('h1').textContent=collectionId==='phrase-bridges-v1'?'Loop bridges':collectionId==='discovery-rounds-v1'?'Found in three rounds':'Around the library';
 document.querySelector('.intro').textContent=`${songs.size} songs · ${ordered.length} mixing ideas · ${chosen.length} listening candidates. Open an idea to compare versions and save what you hear.`;
 document.getElementById('items').innerHTML=ordered.length?ordered.map(group=>{const p=group[0],c=p.comparison,s=p.selection;const rated=group.filter(p=>data.reviews?.[p.preview_id]).length;return `<article><div class="badge">${escape(c.style)} · ${group.length} versions · ${rated}/${group.length} reviewed</div><h2><a href="?collection=${encodeURIComponent(collectionId)}&comparison=${encodeURIComponent(c.group_id)}">${escape(c.idea)}</a></h2><p>${escape(c.source_artist||'')} — ${escape(s.source_title)} → ${escape(c.target_artist||'')} — ${escape(s.target_title)}</p><p class="muted">${escape(c.description)}</p></article>`;}).join(''):'This collection has no completed comparisons yet.';
 return true;
}
async function start(){const r=await fetch('/api/review/audition-data');if(!r.ok)throw Error('Could not load auditions');const data=await r.json();if(collectionOverview(data))return;items=data.previews.filter(p=>(!collectionId||p.comparison?.collection_id===collectionId)&&(!comparisonId||p.comparison?.group_id===comparisonId)&&(new URLSearchParams(location.search).has('earlier')||!p.superseded||p.preview_id===location.hash.slice(1))).slice().reverse();if(comparisonId)items.sort((a,b)=>String(a.comparison.variant).localeCompare(String(b.comparison.variant)));document.getElementById('items').innerHTML=items.length?items.map((p,i)=>{const s=p.selection;return `<article id="${escape(p.preview_id)}"><div class="badge">${escape(p.variant)} · ${p.engine.loop?`${p.engine.loop.bars}-bar outgoing loop · `:p.engine.target_loop?`${p.engine.target_loop.bars}-bar incoming loop · `:''}${s.overlap_bars}-bar overlap · ${s.mix_bpm.toFixed(1)} BPM</div><h2>${escape(s.source_title)} → ${escape(s.target_title)}</h2><div class="muted">${escape(s.source_section.name)} at ${stamp(s.source_start_seconds)} → ${escape(s.target_section.name)} at ${stamp(s.target_start_seconds)} · incoming tempo ${((s.target_tempo_factor-1)*100).toFixed(1)}%</div>${s.listening_preference?'<p class="muted">Previously liked section choice · '+(data.reviews?.[p.preview_id]?'see the saved review for this rendering':'this rendering is unrated')+'</p>':''}<audio controls preload="none" src="/api/review/audition-audio?id=${encodeURIComponent(p.preview_id)}"></audio>${bridgeHtml(p,data.phrase_analysis?.[p.preview_id],i)}${vocalHtml(p,data.vocal_analysis?.[p.preview_id],i)}${mergeHtml(p,data.merge_diagnostics?.[p.preview_id])}${overlapHtml(p,data.overlap_analysis?.[p.preview_id],i)}<div><button data-index="${i}" data-vote="1">WORKS</button><button data-index="${i}" data-vote="-1">DOESN’T WORK</button><select id="reason-${i}" aria-label="Reason"><option value="">Optional reason</option><option>Good energy transition</option><option>Good musical fit</option><option>Clashing vocals or melodies</option><option>Beat alignment problem</option><option>Fade too noticeable</option><option>Level jump</option><option>Energy mismatch</option><option>Unnatural loop</option><option>Too abrupt</option><option>Too long or boring</option><option>Wrong sections</option></select></div><textarea id="note-${i}" rows="2" maxlength="500" placeholder="What worked, or what should change?" aria-label="Feedback note"></textarea><div class="status" id="status-${i}"></div><small>${p.engine.recipe==='phrase_bridge_v1'?'':`Rank score: ${Number(s.model_logit).toFixed(2)} (uncalibrated) · `}${data.reviews?.[p.preview_id]?'listening review saved':'musical listening review pending'}</small></article>`}).join(''):'No rendered auditions are available yet.';if(comparisonId){if(collectionId){const back=document.createElement('p'),a=document.createElement('a');a.href='?collection='+encodeURIComponent(collectionId);a.textContent='← All mixing ideas';back.append(a);document.querySelector('.intro').before(back);}document.querySelector('.intro').textContent=items.find(p=>p.comparison?.description)?.comparison.description|| (items.some(p=>p.comparison?.kind==='fade_exit')?'Same song parts, loop, tempo and beat alignment. Compare the original fade with a later one- or two-bar handoff.':items.some(p=>p.comparison?.kind==='beat_alignment')?'Same song sections and fade. Compare the original timing with an incoming cue adjustment within the same section.':'Compare two section choices for the same songs, with the same tempo, overlap, beat alignment and fade. Rate how each sounds.');document.querySelectorAll('article small').forEach(n=>n.textContent=n.textContent.replace(/^Rank score:.*? · /,''));const comment=items.find(p=>p.comparison?.source_feedback)?.comparison.source_feedback;if(comment){const note=document.createElement('p');note.className='muted';note.textContent='Based on your comment: “'+comment+'”';document.querySelector('.intro').after(note);}}items.forEach((p,i)=>{const saved=data.reviews?.[p.preview_id];if(!saved)return;document.getElementById('note-'+i).value=saved.note;document.getElementById('reason-'+i).value=saved.tags[0]||'';document.getElementById('status-'+i).textContent='Saved: '+(saved.vote>0?'WORKS':'DOESN’T WORK');document.querySelectorAll(`button[data-index="${i}"]`).forEach(b=>b.classList.toggle('selected',Number(b.dataset.vote)===saved.vote));});}
document.getElementById('items').addEventListener('click',async event=>{const b=event.target.closest('button[data-vote]');if(!b)return;const i=Number(b.dataset.index);const payload={preview_id:items[i].preview_id,vote:Number(b.dataset.vote),tags:document.getElementById('reason-'+i).value?[document.getElementById('reason-'+i).value]:[],note:document.getElementById('note-'+i).value};const status=document.getElementById('status-'+i);b.disabled=true;try{const r=await fetch('/api/review/audition-vote',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const result=await r.json();if(!r.ok)throw Error(result.error||'Save failed');status.textContent='Saved with the exact transition plan. '+result.total_reviews+' listening reviews recorded.';b.parentElement.querySelectorAll('button').forEach(x=>x.classList.toggle('selected',x===b));}catch(e){status.textContent=e.message;}finally{b.disabled=false;}});
document.getElementById('items').addEventListener('click',event=>{const b=event.target.closest('button[data-seek-index]');if(!b)return;const p=items[Number(b.dataset.seekIndex)];if(!p)return;const audio=document.getElementById(p.preview_id)?.querySelector('audio');if(audio)audio.currentTime=Number(b.dataset.seekSeconds);});
document.addEventListener('play',e=>{if(e.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause();});},true);

async function setupBuilder(){try{const r=await fetch('/api/tracks');if(!r.ok)throw Error('Could not load songs');const tracks=await r.json();const options='<option value="">Choose a song</option>'+tracks.slice().sort((a,b)=>(a.artist+' '+a.title).localeCompare(b.artist+' '+b.title)).map(t=>`<option value="${t.id}">${escape(t.artist||'Unknown artist')} — ${escape(t.title)}</option>`).join('');document.getElementById('preview-source').innerHTML=options;document.getElementById('preview-target').innerHTML=options;if(items[0]){document.getElementById('preview-source').value=items[0].selection.source_track;document.getElementById('preview-target').value=items[0].selection.target_track;}for(const side of ['source','target']){const requested=Number(query.get(side));if(requested&&tracks.some(t=>t.id===requested))document.getElementById('preview-'+side).value=requested;}document.getElementById('preview-submit').disabled=tracks.length<2;}catch(e){document.getElementById('preview-status').textContent=e.message;}}
async function submitPreview(event){event.preventDefault();const button=document.getElementById('preview-submit'),status=document.getElementById('preview-status');button.disabled=true;try{const payload={source:Number(document.getElementById('preview-source').value),target:Number(document.getElementById('preview-target').value),method:document.getElementById('preview-method').value,bars:Number(document.getElementById('preview-bars').value),loop_bars:document.getElementById('preview-style').value==='blend'?Number(document.getElementById('preview-loop').value):0,phase_align:document.getElementById('preview-style').value==='blend'&&document.getElementById('preview-phase').checked,use_feedback:document.getElementById('preview-style').value==='blend'&&document.getElementById('preview-feedback').checked,style:document.getElementById('preview-style').value,entry_loop_bars:document.getElementById('preview-style').value==='bridge'?Number(document.getElementById('preview-entry-loop').value):0};if(!payload.source||!payload.target||payload.source===payload.target)throw Error('Choose two different songs');let r=await fetch('/api/review/mix-jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});let job=await r.json();if(!r.ok)throw Error(job.error||'Could not create preview');for(let count=0;count<480;count++){status.textContent=job.phase||job.state;if(job.state==='failed')throw Error(job.error);if(job.state==='complete'){const s=job.selection;const bridge=job.bridge?'<p>Old track out at '+stamp(job.bridge.source_out_seconds)+' · new vocal at '+stamp(job.bridge.incoming_vocal_seconds)+' · '+Number(job.bridge.incoming_vocal_gap_bars).toFixed(1)+' bars of space between them.</p>':'';const preference=job.listening_preference?'<p>Using parts from an earlier liked preview. This rendering is not rated yet.</p>':'';const check=job.phase_alignment?'<p>Bar check: '+['source','target'].map(side=>{const d=job.phase_alignment[side];return (side==='source'?'outgoing':'incoming')+(d.applied?' shifted '+(d.shift_beats>0?'+':'')+d.shift_beats+' beat(s)':' kept')+(d.reason?' · '+escape(d.reason):'');}).join('; ')+'.</p>':'';status.innerHTML=`<p>${escape(s.source_title)} → ${escape(s.target_title)} · ${job.loop_bars?job.loop_bars+'-bar source loop · ':''}${s.overlap_bars}-bar blend</p>${bridge}${preference}${check}<audio controls preload="none" src="${escape(job.audio_url)}"></audio><p><a href="${escape(job.preview_url)}" target="_blank">Listen and rate this preview</a></p>`;return;}await new Promise(resolve=>setTimeout(resolve,2000));r=await fetch('/api/review/mix-jobs?id='+encodeURIComponent(job.id));job=await r.json();if(!r.ok)throw Error(job.error||'Could not check preview');}status.textContent='The preview is still queued or running. Submit the same choices again to check it.';}catch(e){status.textContent=e.message;}finally{button.disabled=false;}}
document.getElementById('make-preview').addEventListener('submit',submitPreview);
function updateTransitionControls(){const bridge=document.getElementById('preview-style').value==='bridge';document.querySelectorAll('.legacy-control').forEach(n=>n.hidden=bridge);document.getElementById('preview-entry-loop').disabled=!bridge;document.getElementById('four-bar-overlap').disabled=bridge;if(bridge&&document.getElementById('preview-bars').value==='4')document.getElementById('preview-bars').value='8';}
document.getElementById('preview-style').addEventListener('change',updateTransitionControls);updateTransitionControls();
start().then(async()=>{await setupBuilder();if(location.hash)document.getElementById(location.hash.slice(1))?.scrollIntoView?.({block:'start'});}).catch(e=>document.getElementById('items').textContent=e.message);

</script></body></html>'''.encode()

def serve(handler,route,query,database,body=None):
 if route in ('matches','match-catalog') and body is None:
  import match_catalog
  return match_catalog.serve(handler,route,query)
 if route=='auditions' and body is None:
  payload=page();handler.send_response(200);handler.send_header('Content-Type','text/html; charset=utf-8');handler.send_header('Content-Length',str(len(payload)));handler.end_headers();handler.wfile.write(payload);return True
 if route=='audition-data' and body is None:
  items=previews();handler.send(200,{'previews':items,'reviews':latest_reviews(database,items),'overlap_analysis':overlap_analyses(items),'merge_diagnostics':deep_merge_analyses(items),'vocal_analysis':vocal_analyses(items),'phrase_analysis':phrase_analyses(items),'control_mode':'browser audition only'});return True
 if route=='audition-audio' and body is None:
  ident=query.get('id',[''])[0];item=next((p for p in previews() if p['preview_id']==ident),None)
  if not item:handler.send(404,{'error':'Audition not found; refresh this page'});return True
  path=Path(item['audio']['path']);size=path.stat().st_size;start,end=0,size-1;partial=False
  header=handler.headers.get('Range')
  if header:
   match=re.fullmatch(r'bytes=(\d+)-(\d*)',header)
   if not match:handler.send(416,{'error':'Unsupported byte range'});return True
   start=int(match[1]);end=min(size-1,int(match[2]) if match[2] else size-1);partial=True
   if start> end or start>=size:handler.send(416,{'error':'Range out of bounds'});return True
  handler.send_response(206 if partial else 200);handler.send_header('Content-Type','audio/wav');handler.send_header('Accept-Ranges','bytes');handler.send_header('Content-Length',str(end-start+1))
  if partial:handler.send_header('Content-Range',f'bytes {start}-{end}/{size}')
  handler.end_headers()
  with path.open('rb') as f:
   f.seek(start);remaining=end-start+1
   while remaining:
    chunk=f.read(min(65536,remaining))
    if not chunk:break
    handler.wfile.write(chunk);remaining-=len(chunk)
  return True
 if route=='audition-vote' and body is not None:
  preview=next((p for p in previews() if p['preview_id']==body.get('preview_id')),None)
  if not preview:handler.send(409,{'error':'The audition changed; refresh before rating it'});return True
  vote=body.get('vote');tags=body.get('tags',[]);note=body.get('note','')
  if type(vote)!=int or vote not in (-1,1) or not isinstance(tags,list) or len(tags)>5 or any(not isinstance(t,str) or len(t)>100 for t in tags) or not isinstance(note,str) or len(note)>500:handler.send(400,{'error':'Invalid review'});return True
  with sqlite3.connect(database) as c:
   c.execute('INSERT INTO transition_reviews(preview_id,vote,tags,note,preview_snapshot,created) VALUES(?,?,?,?,?,?)',(preview['preview_id'],vote,json.dumps(tags),note,json.dumps(preview),time.time()));total=c.execute('SELECT count(*) FROM transition_reviews').fetchone()[0]
  handler.send(200,{'ok':True,'total_reviews':total});return True
 return False
