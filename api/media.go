// SPDX-License-Identifier: GPL-3.0-or-later

// USB tab: browse and search the rekordbox export on a USB/SD plugged into a
// CDJ on the link, and load its tracks onto any deck. The export.pdb is
// downloaded over NFS by package mediadb (cached; see mediadb.Fetcher.Media),
// so searching never touches the deck after the first download.
package api

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"sort"
	"strconv"

	"github.com/vynulldev/vynull/mediadb"
)

// mediaSlots are the CDJ media slots that carry a rekordbox export.
var mediaSlots = []struct {
	slot uint8
	name string
}{{3, "USB"}, {2, "SD"}}

type mediaSource struct {
	Player     uint8  `json:"player"`
	PlayerName string `json:"player_name"`
	Slot       uint8  `json:"slot"`
	SlotName   string `json:"slot_name"`
	State      string `json:"state"`
	TrackCount int    `json:"track_count"`
}

// mediaPlayers returns the CDJs (device numbers 1-6, not us) on the link.
func (s *Server) mediaPlayers() []struct {
	num  uint8
	name string
	ip   net.IP
} {
	var out []struct {
		num  uint8
		name string
		ip   net.IP
	}
	if s.Device == nil || s.Device.Peers == nil {
		return out
	}
	for _, p := range s.Device.Peers.Peers() {
		if p.DeviceNumber < 1 || p.DeviceNumber > 6 || p.DeviceNumber == s.Device.DeviceNumber {
			continue
		}
		out = append(out, struct {
			num  uint8
			name string
			ip   net.IP
		}{p.DeviceNumber, p.Name, p.IP})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].num < out[j].num })
	return out
}

// GET /api/media[?refresh=1] — every CDJ's USB/SD with its download state.
// The first call starts the downloads; poll until states settle.
func (s *Server) handleMedia(w http.ResponseWriter, r *http.Request) {
	if !allow(w, r, http.MethodGet) {
		return
	}
	refresh := r.URL.Query().Get("refresh") == "1"
	out := []mediaSource{}
	if s.ExtMedia == nil {
		writeJSON(w, out)
		return
	}
	for _, p := range s.mediaPlayers() {
		usbExport := ""
		for _, ms := range mediaSlots {
			db, export, state := s.ExtMedia(p.num, ms.slot, refresh)
			src := mediaSource{Player: p.num, PlayerName: p.name, Slot: ms.slot, SlotName: ms.name, State: string(state)}
			if ms.slot == 2 && (db == nil || export != "/B/" || export == usbExport) {
				// An empty SD slot's fetch falls back to whatever export the deck
				// advertises — usually the USB at /C/. Only list a real SD card
				// (its own /B/ export), and never while it's still loading.
				continue
			}
			if db != nil {
				if ms.slot == 3 {
					usbExport = export
				}
				src.TrackCount = len(db.Tracks)
			}
			if state == mediadb.MediaUnavailable {
				continue // empty slot or unreachable deck
			}
			out = append(out, src)
		}
	}
	writeJSON(w, out)
}

type mediaTrack struct {
	ID       uint32  `json:"id"`
	Title    string  `json:"title"`
	Artist   string  `json:"artist"`
	Album    string  `json:"album"`
	Genre    string  `json:"genre"`
	Label    string  `json:"label"`
	Key      string  `json:"key"`
	BPM      float64 `json:"bpm"`
	Duration int     `json:"duration"`
	Year     int     `json:"year"`
	Rating   int     `json:"rating"`
	FileName string  `json:"file_name"`
	Comment  string  `json:"comment"`
	Added    string  `json:"date_added"`
}

// mediaPlaylist is one node of the stick's playlist tree (folders included);
// TrackIDs are in the playlist's own order.
type mediaPlaylist struct {
	ID        uint32   `json:"id"`
	ParentID  uint32   `json:"parent_id"`
	Name      string   `json:"name"`
	IsFolder  bool     `json:"is_folder"`
	SortOrder uint32   `json:"sort_order"`
	TrackIDs  []uint32 `json:"track_ids"`
}

// GET /api/media/tracks?player=N&slot=S — every track on that media.
// 202 {"state":"loading"} while the download is in flight.
func (s *Server) handleMediaTracks(w http.ResponseWriter, r *http.Request) {
	if !allow(w, r, http.MethodGet) {
		return
	}
	player, slot, ok := mediaParams(w, r.URL.Query().Get("player"), r.URL.Query().Get("slot"))
	if !ok {
		return
	}
	if s.ExtMedia == nil {
		jsonError(w, "media browsing unavailable", http.StatusServiceUnavailable)
		return
	}
	db, _, state := s.ExtMedia(player, slot, false)
	if db == nil {
		if state == mediadb.MediaLoading {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusAccepted)
			json.NewEncoder(w).Encode(map[string]string{"state": string(state)})
			return
		}
		jsonError(w, fmt.Sprintf("no rekordbox media in player %d slot %d", player, slot), http.StatusNotFound)
		return
	}
	tracks := make([]mediaTrack, 0, len(db.Tracks))
	for _, t := range db.Tracks {
		if t == nil {
			continue
		}
		tracks = append(tracks, mediaTrack{
			ID: t.ID, Title: t.Title, Artist: t.Artist, Album: t.Album, Genre: t.Genre,
			Label: t.Label, Key: t.Key, BPM: float64(t.Tempo) / 100, Duration: int(t.Duration),
			Year: int(t.Year), Rating: int(t.Rating), FileName: t.FileName, Comment: t.Comment,
			Added: t.DateAdded,
		})
	}
	playlists := make([]mediaPlaylist, 0, len(db.PlaylistTree))
	for _, n := range db.PlaylistTree {
		pl := mediaPlaylist{ID: n.ID, ParentID: n.ParentID, Name: n.Name, IsFolder: n.IsFolder, SortOrder: n.SortOrder, TrackIDs: []uint32{}}
		if !n.IsFolder && n.TrackIDs != nil {
			pl.TrackIDs = n.TrackIDs
		}
		playlists = append(playlists, pl)
	}
	sort.SliceStable(playlists, func(i, j int) bool { return playlists[i].SortOrder < playlists[j].SortOrder })
	writeJSON(w, map[string]interface{}{"state": string(state), "tracks": tracks, "playlists": playlists})
}

// POST /api/media/load {"player":1,"slot":3,"track_id":42,"device_number":2}
// loads a track from player's media onto deck device_number.
func (s *Server) handleMediaLoad(w http.ResponseWriter, r *http.Request) {
	if !allow(w, r, http.MethodPost) {
		return
	}
	var req struct {
		Player       uint8  `json:"player"`
		Slot         uint8  `json:"slot"`
		TrackID      uint32 `json:"track_id"`
		DeviceNumber uint8  `json:"device_number"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Player == 0 || (req.Slot != 2 && req.Slot != 3) || req.TrackID == 0 || req.DeviceNumber == 0 {
		jsonError(w, "player, slot (2 or 3), track_id and device_number required", http.StatusBadRequest)
		return
	}
	name := ""
	if s.ExtMedia != nil {
		if db, _, _ := s.ExtMedia(req.Player, req.Slot, false); db != nil {
			t := db.TrackByID(req.TrackID)
			if t == nil {
				jsonError(w, "track not on that media", http.StatusNotFound)
				return
			}
			name = t.Title
		}
	}
	var targetIP net.IP
	for _, p := range s.mediaPlayers() {
		if p.num == req.DeviceNumber {
			targetIP = p.ip
		}
	}
	if targetIP == nil {
		jsonError(w, fmt.Sprintf("CDJ device %d not found", req.DeviceNumber), http.StatusNotFound)
		return
	}
	if err := s.Device.LoadTrackFromOnCDJ(req.Player, req.Slot, req.TrackID, name, req.DeviceNumber, targetIP); err != nil {
		jsonError(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]interface{}{"ok": true})
}

func mediaParams(w http.ResponseWriter, ps, ss string) (uint8, uint8, bool) {
	player, err1 := strconv.ParseUint(ps, 10, 8)
	slot, err2 := strconv.ParseUint(ss, 10, 8)
	if err1 != nil || err2 != nil || player == 0 || (slot != 2 && slot != 3) {
		jsonError(w, "player and slot (2=SD, 3=USB) required", http.StatusBadRequest)
		return 0, 0, false
	}
	return uint8(player), uint8(slot), true
}
