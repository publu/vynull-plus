# vynull-plus

A modified build of **[Vynull](https://github.com/vynulldev/vynull)**, a Linux DJ library that serves your music
straight to Pioneer CDJ/XDJ decks over Pro DJ Link, with no rekordbox and no USB export. The repo adds:

- **USB tab.** Browse and search the rekordbox USB/SD that's plugged into any CDJ on the link: title, artist,
  album, genre, label, key, BPM, year, filename and comment. It shows the stick's **playlists and folders** in the
  stick's own order, and has **▶N** buttons that load a track from that stick onto any deck. The stick's
  `export.pdb` is downloaded once over NFS and cached, so searching never touches the decks.
- **Cue labels.** Named hot/memory cues that survive the deck saving and moving them.
- **NEXT UP panel** (optional, needs `review/`). Library suggestions for the loaded track (tempo, key, genre,
  measured sound), a request queue, and listening feedback.
- **Mix lab** (optional, experimental; needs `review/` + `lab/`). A librosa/scikit-learn/CLAP pipeline that picks
  song parts, aligns beats and bars, and renders short **auto-mix previews** you can rate.

Upstream Vynull's own documentation is kept in **[README.upstream.md](README.upstream.md)**. Read its
"Before you start" section. This is hobby software that speaks an undocumented protocol. Don't use it at a gig,
and back up your library first.

> Not affiliated with Pioneer DJ / AlphaTheta, or with the upstream Vynull project.

---

## 1. Vynull (required)

You need Linux, Go 1.25.7+, `ffmpeg`, and a wired network shared with your CDJs.

```bash
git clone https://github.com/publu/vynull-plus && cd vynull-plus
go build -o vynull .
./vynull --interface eth0 --music-dir ~/Music --web --listen 0.0.0.0:9443
```

Open `http://<this-machine>:9443/`. The decks see the machine as a rekordbox source. The default `--mode rekordbox` needs no root; `--mode cdj` needs port 111, see README.upstream.md. To use the **USB tab**, plug
a rekordbox-exported stick into any linked CDJ and open the **USB** tab. Prebuilt linux-amd64 binaries are on the
[releases page](https://github.com/publu/vynull-plus/releases).

There's a systemd user unit you can adapt in `deploy/systemd/vynull.service`.

## 2. Review service (optional): NEXT UP panel

This is plain Python 3.11+ with the standard library only. It keeps its data in `review/review.sqlite`.

```bash
python3 review/dj_review.py          # listens on 127.0.0.1:9444
```

| Variable | Default | |
|---|---|---|
| `VYNULL_API` | `http://127.0.0.1:9443/api/` | where Vynull's API is |
| `DJ_REVIEW_HOST` / `DJ_REVIEW_PORT` | `127.0.0.1` / `9444` | listen address |
| `DJ_REVIEW_DIR` | `review/` | sqlite DB, mix-job folders, `measured-audio.json` |
| `VYNULL_REVIEW_URL` (set on **vynull**) | `http://127.0.0.1:9444` | where Vynull proxies `/api/review/` |

If the service isn't running, the NEXT UP panel just doesn't appear. With only Vynull's own analysis, you get
suggestions based on tempo, key and genre, plus the request queue and feedback.

**`measured-audio.json`** (per-track sound descriptors and measured sections) unlocks the "similar sound"
scoring, the **PARTS** view and the mix lab below. It was produced by a separate analyzer that is **not included
here**. Its schema is whatever `review/section_matches.py` and `lab/semantic_pair_v1/halo_planner.py` read:
`{track_id: {features: {...}, sections: [{start_seconds, end_seconds, label, key, tempo_bpm, energy_change_db,
core_vector, texture_vector, ...}]}}`.

## 3. Mix lab (optional, experimental)

`lab/` holds the analysis and rendering code the review service runs for **AUTO MIX PREVIEW / LOOP PREVIEW**, plus
the batch library analyses:

| Path | What |
|---|---|
| `semantic_pair_v1/` | CLAP (`laion/larger_clap_music`) music-semantic features + the part/pair planner |
| `part_alignment_v1/` | beat refinement, loop finding, beat-warped live rendering (rubberband + ffmpeg) |
| `bar_phase_v1/` | bar-phase (downbeat) correction gate |
| `phrase_bridges_v1/` | instrumental-loop to vocal-phrase "bridge" transitions (Demucs vocals) |
| `library_lab_v1/`, `library_search_v2/`, `discovery_rounds_v1/` | librosa batch analyses: overlap chroma/onset checks, deep-merge percussion analysis, library pulse screening, match catalog, vocal-overlap audit |
| `tempo_alias_v1/`, `tempo_grid_v1/`, `djmix_grid_v2/` | tempo-alias and grid feature helpers |

The lab expects the decks' audio files to be readable on the same machine (it reads `file_path` from Vynull's
API).

```bash
sudo apt install ffmpeg rubberband-cli
python3 -m venv lab/.venv
lab/.venv/bin/pip install -r lab/requirements.txt   # add --extra-index-url https://download.pytorch.org/whl/cpu for CPU torch
LAB_PYTHON=$PWD/lab/.venv/bin/python lab/.venv/bin/python review/dj_review.py
```

`LAB_PYTHON` picks the interpreter the review service uses for lab jobs (default `lab/.venv/bin/python`).
`VYNULL_LAB_DIR` moves the lab (default `lab/`). The first planner run downloads the CLAP model from Hugging Face.
Set `HF_HUB_OFFLINE=1` to forbid that, or `VYNULL_HF_CACHE` to pick the cache folder.

**Bundled models** (`*/artifacts/*.joblib`, `bar_phase_v1/bar_phase.joblib`) are small scikit-learn models
trained on the **[Raveform](https://huggingface.co/datasets/taejunkim/raveform)** dataset (CC BY 4.0, Taejun Kim)
and contain no personal data. They're pickles, so load them only if you trust this repo, and use scikit-learn
1.9.x. `bar_phase_v1/*.json` hold that model's protocol and integrity locks. `inference.py` refuses to run if the
model or feature code changes.

Everything here is **experimental**. Previews are listening candidates, not claims of mix quality.

## License and credits

GPL-3.0-or-later, same as upstream (see [LICENSE](LICENSE)). This is a modified version of Vynull by vynulldev
(upstream commit `282e80bc91d58445e85a2077412953b970140b38`). The first commit in this repo is that upstream
tree unchanged, and later commits are the modifications. Protocol and format work credited in
[README.upstream.md](README.upstream.md#acknowledgements). The bundled models are derived from Raveform (CC BY 4.0). The CLAP model is
`laion/larger_clap_music`, Demucs is by Meta (MIT), and librosa is ISC.
