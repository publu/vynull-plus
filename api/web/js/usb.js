// SPDX-License-Identifier: GPL-3.0-or-later
// USB tab: search the rekordbox export on a USB/SD plugged into a CDJ and load
// its tracks onto any deck. Data comes from /api/media (the stick's export.pdb,
// downloaded once over NFS and cached server-side), so searching is local.
(() => {
'use strict';
const panel = document.getElementById('usb');
const sourcesEl = document.getElementById('usb-sources');
const searchEl = document.getElementById('usb-search');
const contentEl = document.getElementById('usb-content');
const statsEl = document.getElementById('usb-stats');
const sidebarEl = document.getElementById('usb-sidebar');
const MAX_ROWS = 400;

let sources = [];            // [{player, player_name, slot, slot_name, state, track_count}]
let selected = null;         // "player:slot"
let tracks = null;           // tracks of the selected media (null = not loaded)
let tracksKey = '';          // which media `tracks` belongs to
let loadingTracks = false;
let sortCol = 'artist', sortDir = 1;
let lastDecks = '';
let playlists = [];          // the stick's playlist tree, flattened
let playlistID = 0;          // 0 = all tracks
const openFolders = new Set();
let byID = new Map();

const esc = s => escape(s ?? '');
const keyOf = s => `${s.player}:${s.slot}`;
const fold = s => String(s ?? '').normalize('NFKD').replace(/[̀-ͯ]/g, '').toLowerCase();
const clock = s => s ? `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}` : '';
const isActive = () => panel.classList.contains('active');

function decks() {
  return (currentPlayers || []).filter(p => p.device_number >= 1 && p.device_number <= 6)
    .sort((a, b) => a.device_number - b.device_number);
}

async function refreshSources(rescan) {
  try {
    sources = await fetchJSON('/api/media' + (rescan ? '?refresh=1' : ''));
  } catch (e) {
    contentEl.innerHTML = `<div class="empty-state">USB LIST UNAVAILABLE · ${esc(e.message)}</div>`;
    return;
  }
  if (rescan) tracksKey = '';
  if (!sources.some(s => keyOf(s) === selected)) {
    const ready = sources.find(s => s.state === 'ready') || sources[0];
    selected = ready ? keyOf(ready) : null;
  }
  renderSources();
  const src = sources.find(s => keyOf(s) === selected);
  if (!src) {
    tracks = null;
    renderSidebar();
    statsEl.textContent = '';
    contentEl.innerHTML = `<div class="empty-state">${decks().length ? 'NO REKORDBOX USB/SD FOUND IN THE DECKS' : 'AWAITING DEVICE LINK'}</div>`;
    return;
  }
  if (src.state === 'ready' && tracksKey !== selected) loadTracks(src);
  else if (src.state !== 'ready' && tracksKey !== selected) {
    contentEl.innerHTML = '<div class="empty-state">READING THE STICK…</div>';
  }
}

function renderSources() {
  sourcesEl.innerHTML = sources.map(s => {
    const k = keyOf(s);
    const count = s.state === 'ready' ? s.track_count : '…';
    return `<button type="button" class="usb-src ${k === selected ? 'active' : ''}" data-key="${k}" title="${esc(s.player_name)} · player ${s.player}">
      <span class="name">${esc(s.slot_name)} · P${s.player}</span><span class="count">${count}</span></button>`;
  }).join('');
}

async function loadTracks(src) {
  if (loadingTracks) return;
  loadingTracks = true;
  const k = keyOf(src);
  try {
    const r = await fetch(`/api/media/tracks?player=${src.player}&slot=${src.slot}`);
    if (r.status === 202) return;            // still downloading; next poll retries
    if (!r.ok) throw new Error(await errorText(r));
    const d = await r.json();
    if (selected !== k) return;
    tracks = d.tracks.map(t => ({ ...t, _hay: fold([t.title, t.artist, t.album, t.genre, t.label, t.key, typeof keyBoth === 'function' ? keyBoth(t.key) : '',
      t.bpm ? t.bpm.toFixed(0) : '', t.bpm ? t.bpm.toFixed(1) : '', t.year || '', t.file_name, t.comment].join(' ')) }));
    byID = new Map(tracks.map(t => [t.id, t]));
    playlists = d.playlists || [];
    if (!playlists.some(p => p.id === playlistID && !p.is_folder)) playlistID = 0;
    tracksKey = k;
    renderSidebar();
    render();
  } catch (e) {
    contentEl.innerHTML = `<div class="empty-state">COULDN'T READ THE STICK · ${esc(e.message)}</div>`;
  } finally {
    loadingTracks = false;
  }
}

function filtered() {
  const tokens = fold(searchEl.value).split(/\s+/).filter(Boolean);
  const pl = playlistID ? playlists.find(p => p.id === playlistID) : null;
  let base = pl ? pl.track_ids.map((id, i) => byID.get(id) && { ...byID.get(id), _pos: i + 1 }).filter(Boolean) : tracks;
  let list = tokens.length ? base.filter(t => tokens.every(tok => t._hay.includes(tok))) : base.slice();
  if (sortCol === 'pos' && !pl) sortCol = 'artist';
  const val = t => sortCol === 'bpm' || sortCol === 'duration' || sortCol === 'pos' ? (t[sortCol === 'pos' ? '_pos' : sortCol] || 0) : fold(t[sortCol]);
  list.sort((a, b) => {
    const x = val(a), y = val(b);
    if (x < y) return -sortDir;
    if (x > y) return sortDir;
    return fold(a.title) < fold(b.title) ? -1 : 1;
  });
  return list;
}

function render() {
  if (!tracks) return;
  const src = sources.find(s => keyOf(s) === selected);
  const list = filtered();
  const ds = decks();
  lastDecks = ds.map(d => d.device_number + ':' + d.track_id + ':' + d.is_playing).join(',');
  const sourceLabel = src ? `${src.slot_name} · player ${src.player}` : '';
  const onDeck = new Map(ds.filter(d => d.external && d.source === sourceLabel).map(d => [d.track_id, d.device_number]));
  const pl = playlistID ? playlists.find(p => p.id === playlistID) : null;
  const total = pl ? pl.track_ids.length : tracks.length;
  statsEl.textContent = (pl ? pl.name.toUpperCase() + ' · ' : '') + (list.length === total ? `${total} TRACKS` : `${list.length} / ${total} TRACKS`);
  if (!list.length) {
    contentEl.innerHTML = `<div class="empty-state">${total ? 'NO MATCHES' : (pl ? 'THIS PLAYLIST IS EMPTY' : 'THE STICK HAS NO TRACKS')}</div>`;
    return;
  }
  const cols = [...(pl ? [['pos', '#']] : []), ['title', 'TITLE'], ['artist', 'ARTIST'], ['album', 'ALBUM'], ['genre', 'GENRE'], ['bpm', 'BPM'], ['key', 'KEY'], ['duration', 'TIME']];
  const head = cols.map(([c, l]) => `<th data-sort="${c}" class="${c === sortCol ? 'sorted' : ''}">${l}${c === sortCol ? (sortDir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('');
  const rows = list.slice(0, MAX_ROWS).map(t => {
    const deck = onDeck.get(t.id);
    const btns = ds.map(d => `<button type="button" class="usb-load" data-track="${t.id}" data-deck="${d.device_number}" title="Load onto player ${d.device_number}${d.is_playing ? ' (playing)' : ''}">▶${d.device_number}</button>`).join('');
    return `<tr class="${deck ? 'on-deck' : ''}">
      <td class="usb-actions">${btns || '<span class="usb-muted">—</span>'}</td>
      ${pl ? `<td class="num usb-muted">${t._pos}</td>` : ''}<td class="t-title">${esc(t.title)}${deck ? ` <span class="usb-deck-chip">ON ${deck}</span>` : ''}</td>
      <td>${esc(t.artist)}</td><td class="usb-muted">${esc(t.album)}</td><td class="usb-muted">${esc(t.genre)}</td>
      <td class="num">${t.bpm ? t.bpm.toFixed(2) : ''}</td><td>${esc(typeof fmtKey === 'function' ? fmtKey(t.key) : t.key)}</td>
      <td class="num">${clock(t.duration)}</td></tr>`;
  }).join('');
  const more = list.length > MAX_ROWS ? `<div class="usb-more">SHOWING ${MAX_ROWS} OF ${list.length} — TYPE TO NARROW DOWN</div>` : '';
  contentEl.innerHTML = `<div class="usb-table-wrap"><table class="usb-table"><thead><tr><th>LOAD</th>${head}</tr></thead><tbody>${rows}</tbody></table></div>${more}`;
}

function renderSidebar() {
  if (!tracks) { sidebarEl.innerHTML = ''; return; }
  const kids = new Map();
  for (const p of playlists) {
    if (!kids.has(p.parent_id)) kids.set(p.parent_id, []);
    kids.get(p.parent_id).push(p);
  }
  const node = (p, depth) => {
    const pad = `style="padding-left:${10 + depth * 14}px"`;
    if (p.is_folder) {
      const open = openFolders.has(p.id);
      return `<button type="button" class="usb-pl folder" data-folder="${p.id}" ${pad}><span class="icon">${open ? '▾' : '▸'}</span><span class="name">${esc(p.name)}</span></button>`
        + (open ? (kids.get(p.id) || []).map(c => node(c, depth + 1)).join('') : '');
    }
    return `<button type="button" class="usb-pl ${p.id === playlistID ? 'active' : ''}" data-pl="${p.id}" ${pad}><span class="icon">♫</span><span class="name">${esc(p.name)}</span><span class="count">${p.track_ids.length}</span></button>`;
  };
  const roots = kids.get(0) || [];
  sidebarEl.innerHTML = `<button type="button" class="usb-pl ${playlistID === 0 ? 'active' : ''}" data-pl="0"><span class="icon">▦</span><span class="name">ALL TRACKS</span><span class="count">${tracks.length}</span></button>
    <div class="usb-side-head">PLAYLISTS</div>${roots.length ? roots.map(p => node(p, 0)).join('') : '<div class="usb-muted usb-side-empty">— NONE ON THIS STICK —</div>'}`;
}

sidebarEl.addEventListener('click', e => {
  const f = e.target.closest('[data-folder]');
  if (f) {
    const id = Number(f.dataset.folder);
    openFolders.has(id) ? openFolders.delete(id) : openFolders.add(id);
    renderSidebar();
    return;
  }
  const b = e.target.closest('[data-pl]');
  if (!b) return;
  playlistID = Number(b.dataset.pl);
  sortCol = playlistID ? 'pos' : 'artist'; sortDir = 1;
  renderSidebar();
  render();
});

async function loadOnto(trackID, deckNum) {
  const src = sources.find(s => keyOf(s) === selected);
  if (!src) return;
  const t = tracks.find(x => x.id === trackID);
  const target = decks().find(p => p.device_number === deckNum);
  if (target && target.is_playing) {
    const now = target.track_title ? `"${target.track_title}"${target.artist ? ' — ' + target.artist : ''}` : 'a track';
    const ok = await openConfirmModal({
      title: 'DECK IS PLAYING',
      message: `PLAYER ${String(deckNum).padStart(2, '0')} is currently playing ${now}.\n\nLoading a new track will interrupt it. Continue?`,
      okLabel: 'LOAD ANYWAY', danger: true,
    });
    if (!ok) return;
  }
  const r = await fetch('/api/media/load', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ player: src.player, slot: src.slot, track_id: trackID, device_number: deckNum }),
  });
  if (r.ok) toast(`▶ ${t ? t.title : 'TRACK ' + trackID} → PLAYER ${String(deckNum).padStart(2, '0')}`, 'success');
  else toast('LOAD FAILED · ' + (await errorText(r)), 'error');
}

sourcesEl.addEventListener('click', e => {
  const b = e.target.closest('.usb-src');
  if (!b || b.dataset.key === selected) return;
  selected = b.dataset.key;
  tracks = null; tracksKey = ''; playlistID = 0; openFolders.clear();
  renderSidebar();
  contentEl.innerHTML = '<div class="empty-state">READING THE STICK…</div>';
  refreshSources(false);
});
contentEl.addEventListener('click', e => {
  const th = e.target.closest('th[data-sort]');
  if (th) {
    if (sortCol === th.dataset.sort) sortDir = -sortDir; else { sortCol = th.dataset.sort; sortDir = 1; }
    render();
    return;
  }
  const b = e.target.closest('.usb-load');
  if (b) loadOnto(Number(b.dataset.track), Number(b.dataset.deck));
});
let searchTimer = 0;
searchEl.addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(render, 80); });
searchEl.addEventListener('keydown', e => { if (e.key === 'Escape') { searchEl.value = ''; render(); } });
document.getElementById('usb-search-clear').addEventListener('click', () => { searchEl.value = ''; render(); searchEl.focus(); });
document.getElementById('usb-refresh').addEventListener('click', () => {
  tracks = null;
  contentEl.innerHTML = '<div class="empty-state">RE-READING THE STICK…</div>';
  refreshSources(true);
});
document.querySelector('.tab[data-tab="usb"]').addEventListener('click', () => {
  refreshSources(false);
  setTimeout(() => searchEl.focus(), 0);
});
// While the tab is open: pick up sticks being inserted/finishing download, and
// redraw when the decks change (load buttons, ON-DECK markers).
setInterval(() => {
  if (!isActive() || document.hidden) return;
  const d = decks().map(d => d.device_number + ':' + d.track_id + ':' + d.is_playing).join(',');
  if (tracks && d !== lastDecks) render();
}, 1000);
setInterval(() => { if (isActive() && !document.hidden) refreshSources(false); }, 4000);
})();
