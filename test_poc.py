#!/usr/bin/env python3
"""Run from your laptop after provisioning. Requires: pip install requests"""
import argparse
import base64
from pathlib import Path
import requests


def main():
    p = argparse.ArgumentParser(description='Rosely Voice + MuseTalk clip smoke test')
    p.add_argument('--url', required=True, help='Vast mapped endpoint, e.g. http://HOST:EXTERNAL_PORT')
    p.add_argument('--api-key', required=True)
    p.add_argument('--avatar-file', required=True, type=Path, help='PNG/JPG/MP4 file with a visible adult character face')
    p.add_argument('--avatar-id', default='rosely_demo_01')
    p.add_argument('--voice', default=None, help='Existing voice ID or omit for Chatterbox default')
    p.add_argument('--text', default='Hello! It is really nice to see you today.')
    p.add_argument('--out', default='rosely_demo.mp4', type=Path)
    args = p.parse_args()
    base = args.url.rstrip('/')
    headers = {'X-API-Key': args.api_key}

    def report(response):
        if response.status_code >= 400:
            raise RuntimeError(f'HTTP {response.status_code}: {response.text[:1500]}')
        return response

    print('Checking health...')
    print(report(requests.get(base + '/health', timeout=30)).json())

    print('Preparing character (first run may be slow)...')
    with args.avatar_file.open('rb') as f:
        r = requests.post(base + '/v1/avatar/prepare', headers=headers,
                          data={'avatar_id': args.avatar_id},
                          files={'file': (args.avatar_file.name, f)}, timeout=1200)
    if r.status_code == 409:
        print('Already prepared; reusing existing avatar ID.')
    else:
        print(report(r).json())

    print('Generating speech through existing Chatterbox endpoint...')
    payload = {'input': args.text}
    if args.voice:
        payload['voice'] = args.voice
    tts = report(requests.post(base + '/v1/tts', headers=headers, json=payload, timeout=300)).json()
    wav = base64.b64decode(tts['audio_base64'])
    print(f'Voice output: {len(wav)} bytes; sample rate: {tts.get("sample_rate")}')

    print('Rendering lip-synced clip...')
    result = report(requests.post(base + '/v1/avatar/render', headers=headers,
                                  data={'avatar_id': args.avatar_id},
                                  files={'audio': ('reply.wav', wav, 'audio/wav')},
                                  timeout=1200))
    args.out.write_bytes(result.content)
    print(f'Success: {args.out.resolve()} ({len(result.content)} bytes)')


if __name__ == '__main__':
    main()
