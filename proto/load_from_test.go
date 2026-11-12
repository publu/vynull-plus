// SPDX-License-Identifier: GPL-3.0-or-later
package proto

import (
	"bytes"
	"testing"
)

// A load from another player's USB names that player and slot as the source
// but is still sent by us; loading from ourselves is byte-identical to before.
func TestMarshalLoadTrackCommandFrom(t *testing.T) {
	own := MarshalLoadTrackCommand("Halo", 7, 4, 2, 99)
	same := MarshalLoadTrackCommandFrom("Halo", 7, 7, 4, 2, 99)
	if !bytes.Equal(own, same) {
		t.Fatal("own-source load changed")
	}
	pkt := MarshalLoadTrackCommandFrom("Halo", 7, 1, 3, 3, 3961)
	if pkt[0x21] != 7 || pkt[0x24] != 7 {
		t.Fatalf("sender = %d/%d, want 7", pkt[0x21], pkt[0x24])
	}
	if pkt[0x28] != 1 || pkt[0x29] != 3 {
		t.Fatalf("source = player %d slot %d, want player 1 slot 3", pkt[0x28], pkt[0x29])
	}
	if pkt[0x40] != 2 {
		t.Fatalf("target index = %d, want 2", pkt[0x40])
	}
}
