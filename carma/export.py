"""Export completed studies for Git; classifier checkpoints remain in the full local bundle."""
import argparse
import json
from pathlib import Path
import shutil
from .run import write_json
from .verify import verify


def export(source,destination):
    source,destination=Path(source),Path(destination)
    verify(source,require_checkpoints=True)
    destination.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text())
    omitted={}
    for file in source.iterdir():
        if not file.is_file() or file.name=='manifest.json':continue
        if file.name.startswith('classifier_') and file.suffix=='.pt':
            omitted[file.name]=manifest['artifact_sha256'][file.name]
        else:shutil.copyfile(file,destination/file.name)
    manifest['omitted_checkpoints']=omitted
    manifest['checkpoint_storage']='Classifier states remain in the original local run and full evidence bundle; hashes preserved here. Policy checkpoints, source snapshot and all raw metrics are included.'
    write_json(destination/'manifest.json',manifest)
    return verify(destination)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('source'); p.add_argument('destination'); a=p.parse_args()
    print(json.dumps(export(a.source,a.destination)))


if __name__=='__main__':main()
