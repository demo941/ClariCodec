"""Download the official ClariCodec 300 bps inference checkpoints."""
import argparse
from pathlib import Path
import shutil
import tarfile
import urllib.request

MODELS = ('nonstreaming-stage1', 'nonstreaming-rl', 'streaming-stage1', 'streaming-rl')
RELEASE_URL = 'https://github.com/demo941/ClariCodec/releases/download/v1.0.0'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=(*MODELS, 'all'), default='nonstreaming-rl')
    args = parser.parse_args()
    destination = Path('checkpoints')
    destination.mkdir(exist_ok=True)
    for model in MODELS if args.model == 'all' else (args.model,):
        filename = f'claricodec-300bps-{model}.tar'
        archive = destination / filename
        print(f'Downloading {model} from {RELEASE_URL}/{filename}', flush=True)
        with urllib.request.urlopen(f'{RELEASE_URL}/{filename}') as response, archive.open('wb') as output:
            shutil.copyfileobj(response, output)
        with tarfile.open(archive) as package:
            package.extractall(destination, filter='data')
        archive.unlink()
        print(f'Ready: {destination / model}', flush=True)


if __name__ == '__main__':
    main()
