#!/usr/bin/env python3
"""Persist health issue state and emit only new/escalated/recovered notifications."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
ROOT=str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path: sys.path.insert(0,ROOT)
from scripts.health_check import classify_issue_transitions


def process(result: dict, state_path: Path) -> dict:
    previous_doc={}
    if state_path.exists():
        try: previous_doc=json.loads(state_path.read_text())
        except Exception: previous_doc={}
    previous=previous_doc.get('issues') or {}
    counts=previous_doc.get('counts') or {}
    transitions,current=classify_issue_transitions(previous,result.get('issues') or [])
    next_counts={}
    notify=[]
    for item in transitions:
        key=item['issue_key']; kind=item['transition']
        if kind=='continuing':
            next_counts[key]=int(counts.get(key,1))+1
            if next_counts[key] in {3,10,30}: notify.append(dict(item,occurrences=next_counts[key]))
        elif kind=='new':
            next_counts[key]=1;notify.append(dict(item,occurrences=1))
        elif kind=='recovered':
            notify.append(item)
    state_path.parent.mkdir(parents=True,exist_ok=True)
    tmp=state_path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'issues':current,'counts':next_counts},ensure_ascii=False,indent=2))
    os.chmod(tmp,0o600);tmp.replace(state_path);os.chmod(state_path,0o600)
    return {'notify':notify,'state':current}


def main() -> int:
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--state',required=True)
    a=p.parse_args();result=json.loads(Path(a.input).read_text());out=process(result,Path(a.state))
    if out['notify']: print(json.dumps(out['notify'],ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
