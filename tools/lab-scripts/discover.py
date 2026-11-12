#!/usr/bin/env python3
"""Listen for Pro DJ Link announcements without sending anything to decks."""
import argparse
import json
import selectors
import socket
import time


def discover(seconds=15, bind='0.0.0.0'):
    selector = selectors.DefaultSelector()
    devices = {}
    errors = []
    for port in (50000, 50001, 50002):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((bind, port))
            sock.setblocking(False)
            selector.register(sock, selectors.EVENT_READ, port)
        except OSError as exc:
            errors.append({'port': port, 'error': str(exc)})
            sock.close()
    deadline = time.monotonic() + seconds
    try:
        while selector.get_map() and time.monotonic() < deadline:
            for key, _ in selector.select(max(0, deadline - time.monotonic())):
                data, addr = key.fileobj.recvfrom(65535)
                if not data.startswith(b'Qspt1WmJOL'):
                    continue
                name = data[12:32].split(b'\0', 1)[0].decode('ascii', errors='replace')
                identity = (addr[0], name)
                entry = devices.setdefault(identity, {'ip': addr[0], 'name': name, 'packets': 0, 'ports': []})
                entry['packets'] += 1
                if key.data not in entry['ports']:
                    entry['ports'].append(key.data)
    finally:
        for key in list(selector.get_map().values()):
            key.fileobj.close()
        selector.close()
    return {'devices': list(devices.values()), 'bind_errors': errors, 'duration_seconds': seconds}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--bind', default='0.0.0.0')
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    print(json.dumps(discover(args.seconds, args.bind), indent=2))
