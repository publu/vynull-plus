package api

import (
	"github.com/vynulldev/vynull/dbserver"
	"testing"
)

func TestCueLabelPersistsAndSurvivesDeckSave(t *testing.T) {
	dir := t.TempDir()
	store := dbserver.NewCueStore(dir)
	adapter := &CueStoreAdapter{Store: store}
	original := CueInfo{Number: 4, Type: "cue", TimeMs: 31000, LoopMs: -1, ColorID: 33, Label: "UP"}
	adapter.SaveCue(10, original)
	store.Close()
	reopened := dbserver.NewCueStore(dir)
	defer reopened.Close()
	adapter.Store = reopened
	if got := adapter.GetCues(10); len(got) != 1 || got[0] != original {
		t.Fatalf("cue lost after reopening: %+v", got)
	}
	deck := dbserver.CuePoint{Number: 4, Type: 1, TimeMs: 32000, LoopMs: -1, Status: 1, ColorID: 33}
	reopened.SaveCue(10, &deck, dbserver.MarshalCueBlob(&deck))
	if got := adapter.GetCues(10); len(got) != 1 || got[0].Label != "UP" || got[0].TimeMs != 32000 {
		t.Fatalf("deck save lost name or movement: %+v", got)
	}
	original.Label = ""
	adapter.SaveCue(10, original)
	if got := adapter.GetCues(10); got[0].Label != "" {
		t.Fatalf("API could not clear label: %+v", got)
	}
}
