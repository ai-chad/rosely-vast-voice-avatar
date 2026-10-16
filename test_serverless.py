"""Smoke-test the queued JSON Vast endpoint. Requires `pip install vastai-sdk`."""
import argparse
import asyncio
import base64
from pathlib import Path
from vastai import Serverless

async def main():
    p = argparse.ArgumentParser()
    p.add_argument('--endpoint', required=True)
    p.add_argument('--image', help='Optional avatar image for slow full-PoC test')
    p.add_argument('--text', default='Hello from Rosely, this is a test call.')
    p.add_argument('--out', default='rosely_serverless_poc.mp4')
    args = p.parse_args()
    client = Serverless()
    endpoint = await client.get_endpoint(args.endpoint)
    if args.image:
        image = base64.b64encode(Path(args.image).read_bytes()).decode('ascii')
        payload = {'image_base64': image, 'image_filename': Path(args.image).name, 'input': args.text}
        result = await endpoint.request('/v1/avatar/generate', payload)
        data = result['response']
        Path(args.out).write_bytes(base64.b64decode(data['video_base64']))
        print('Video saved:', args.out)
    else:
        result = await endpoint.request('/v1/tts', {'input': args.text})
        data = result['response']
        print('TTS bytes:', data.get('size_bytes'), 'content type:', data.get('content_type'))

if __name__ == '__main__':
    asyncio.run(main())
