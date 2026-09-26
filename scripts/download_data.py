#!/usr/bin/env python3
"""Download and prepare the benchmark data used by ExpGym.

    python scripts/download_data.py                  # everything
    python scripts/download_data.py --only search,audit
    python scripts/download_data.py --check          # verify only

Datasets (all placed under $EXPGYM_DATA_ROOT, default ./data):

  search       PhantomWiki v1 question/answer and corpus parquet files (~3 MB)
  audit        ContractNLI test split (~2 MB)
  paramnet     pinned HPOBench checkout + ParamNet surrogate forests (~200 MB)
  nasbench101  NAS-Bench-101 full tfrecord (2 GB download), reduced to a ~25 MB table
  nasbench201  NATS-Bench topology archive (1.1 GB download), reduced to three small tables

Large downloads are verified against pinned checksums and deleted after
conversion. Offline copies can be supplied with the environment variables
NASBENCH101_TFRECORD, NATS_TSS_ARCHIVE and PARAMNET_SURROGATES_DIR.
The paramnet step needs git.
"""
from __future__ import annotations

import argparse
import base64
import bz2
import hashlib
import io
import json
import os
import pickle
import shutil
import struct
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from typing import Callable, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expgym.envs.nas import NAS101_SCHEMA, NAS201_OPERATIONS, NAS201_SCHEMA  # noqa: E402
from expgym.envs.search import DEFAULT_SEEDS  # noqa: E402
from expgym.paths import (PHANTOM_WIKI_REPO, PHANTOM_WIKI_REVISION, contract_nli_file,  # noqa: E402
                          hpo_dir, phantom_wiki_dir)

CONTRACT_NLI_ZIP = "https://raw.githubusercontent.com/stanfordnlp/contract-nli/gh-pages/resources/contract-nli.zip"

HPOBENCH_GIT = "https://github.com/automl/HPOBench.git"
HPOBENCH_REF = "47bf141f79e6bdfb26d1f1218b5d5aac09d7d2ce"
# ParamNet surrogates from the HPOlib2 "surrogates" branch, pinned by commit and git blob hash.
PARAMNET_SOURCE = "https://raw.githubusercontent.com/LoneKnightz/HPOlib2/de88ab3aa2a39a86ccf8c85e9069f3441c1cfc61/surrogates/"
PARAMNET_FILES = {
    "rf_surrogate_paramnet_adult.pkl": (57408198, "c4e0814a71ce795119a12f1c14e831a306d53540"),
    "rf_cost_surrogate_paramnet_adult.pkl": (8095179, "f6df8e698158408286e16dd2961f3ead53b46d75"),
    "rf_surrogate_paramnet_higgs.pkl": (57436470, "a8e991e95b6aea95920355de57c25063a7c3d768"),
    "rf_cost_surrogate_paramnet_higgs.pkl": (8096715, "4026854c55c578a68b566f10b0a5acf338d3edfd"),
    "rf_surrogate_paramnet_letter.pkl": (57575097, "06e6ff8b5d3ab01e692ca8b54d7dcd70ae499f19"),
    "rf_cost_surrogate_paramnet_letter.pkl": (8073678, "33ac25783749a3bc4fd1cfb5db84fdeb2d2dd1eb"),
}

NAS101_URL = "https://storage.googleapis.com/nasbench/nasbench_full.tfrecord"
NAS101_MD5 = "7bff458f43238c7a5f08e9074c903f83"
NAS101_ARCHITECTURES = 423624

NATS_URL = "https://drive.usercontent.google.com/download?id=17_saCsj_krKjlCBLOJEpNtzPXArMCqxU&export=download&confirm=t"
NATS_SHA256 = "580fd8f3425fed9f495640b0f12ccb7744f89dec7aff263bd75aed37ab0b8bb6"
# dataset name in ExpGym -> (name in NATS-Bench, validation split)
NATS_DATASETS = {"cifar10-valid": ("cifar10-valid", "x-valid"),
                 "cifar100": ("cifar100", "ori-test"),
                 "imagenet16-120": ("ImageNet16-120", "ori-test")}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def file_hash(path: str, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: str) -> str:
    digest = hashlib.sha1(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: str) -> None:
    """Download to ``target`` atomically (via a temporary file)."""
    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = target + ".partial"
    request = urllib.request.Request(url, headers={"User-Agent": "expgym-data"})
    print("  downloading %s" % url)
    with urllib.request.urlopen(request, timeout=300) as response, open(partial, "wb") as handle:
        shutil.copyfileobj(response, handle, 1 << 22)
    os.replace(partial, target)


def fetch_verified(url: str, target: str, algorithm: str, checksum: str, offline_env: str) -> Tuple[str, bool]:
    """Return ``(path, downloaded)`` for a checksum-verified file."""
    offline = os.environ.get(offline_env)
    path = offline or target
    if not os.path.exists(path):
        if offline:
            raise RuntimeError("%s points to a missing file: %s" % (offline_env, offline))
        download(url, path)
    if file_hash(path, algorithm) != checksum:
        if not offline:
            os.remove(path)  # let the next run download it again
        raise RuntimeError("checksum mismatch for %s" % path)
    return path, not offline


def write_pickle(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".partial", "wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    os.replace(path + ".partial", path)


# ----------------------------------------------------------------------------
# Datasets. Each step returns True when the data is ready.
# ----------------------------------------------------------------------------

def search(check: bool) -> bool:
    root = phantom_wiki_dir()
    files = ["%s/depth_20_size_5000_seed_%d-00000-of-00001.parquet" % (kind, seed)
             for kind in ("question-answer", "text-corpus") for seed in DEFAULT_SEEDS]
    ready = all(os.path.exists(os.path.join(root, f)) for f in files)
    if ready or check:
        return ready
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=PHANTOM_WIKI_REPO, repo_type="dataset", revision=PHANTOM_WIKI_REVISION,
                      local_dir=root, allow_patterns=files)
    return search(check=True)


def audit(check: bool) -> bool:
    target = contract_nli_file()
    if os.path.exists(target) or check:
        return os.path.exists(target)
    request = urllib.request.Request(CONTRACT_NLI_ZIP, headers={"User-Agent": "expgym-data"})
    print("  downloading %s" % CONTRACT_NLI_ZIP)
    with urllib.request.urlopen(request, timeout=300) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    member = next(n for n in archive.namelist() if n.endswith("/test.json") or n == "test.json")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as handle:
        handle.write(archive.read(member))
    return True


def paramnet(check: bool) -> bool:
    checkout = os.path.join(hpo_dir(), "HPOBench")
    surrogates = os.path.join(hpo_dir(), "hpobench_data", "Surrogates")

    def checkout_ready() -> bool:
        try:
            head = subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"],
                                           stderr=subprocess.DEVNULL, universal_newlines=True)
            return head.strip() == HPOBENCH_REF
        except (OSError, subprocess.CalledProcessError):
            return False

    def surrogate_ready(name: str) -> bool:
        path = os.path.join(surrogates, name)
        size, blob = PARAMNET_FILES[name]
        return os.path.exists(path) and os.path.getsize(path) == size and git_blob_sha1(path) == blob

    if check:
        return checkout_ready() and all(surrogate_ready(n) for n in PARAMNET_FILES)
    if not checkout_ready():
        shutil.rmtree(checkout, ignore_errors=True)
        for command in (["git", "init", "-q", checkout],
                        ["git", "-C", checkout, "remote", "add", "origin", HPOBENCH_GIT],
                        ["git", "-C", checkout, "fetch", "-q", "--depth", "1", "origin", HPOBENCH_REF],
                        ["git", "-C", checkout, "checkout", "-q", "--detach", "FETCH_HEAD"]):
            subprocess.check_call(command)
    offline = os.environ.get("PARAMNET_SURROGATES_DIR")
    for name in PARAMNET_FILES:
        if surrogate_ready(name):
            continue
        target = os.path.join(surrogates, name)
        if offline:
            os.makedirs(surrogates, exist_ok=True)
            shutil.copyfile(os.path.join(offline, name), target)
        else:
            download(PARAMNET_SOURCE + name, target)
        if not surrogate_ready(name):
            os.remove(target)
            raise RuntimeError("checksum mismatch for %s" % name)
    return True


def _varint(data: bytes, position: int) -> Tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, position
        shift += 7


def _protobuf(data: bytes) -> Dict[int, list]:
    """Minimal protobuf decoder (enough for NAS-Bench-101 metrics)."""
    fields: Dict[int, list] = {}
    position = 0
    while position < len(data):
        tag, position = _varint(data, position)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, position = _varint(data, position)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value, position = data[position:position + size], position + size
        elif wire == 2:
            size, position = _varint(data, position)
            value, position = data[position:position + size], position + size
        else:
            raise ValueError("unsupported protobuf wire type %d" % wire)
        fields.setdefault(number, []).append(value)
    return fields


def nasbench101(check: bool) -> bool:
    """Keep the mean validation error and total training cost at 108 epochs."""
    target = os.path.join(hpo_dir(), "nasbench101.pkl")
    if os.path.exists(target) or check:
        return os.path.exists(target)
    source, downloaded = fetch_verified(NAS101_URL, os.path.join(hpo_dir(), "nasbench_full.tfrecord"),
                                        "md5", NAS101_MD5, "NASBENCH101_TFRECORD")
    sums: Dict[str, list] = {}
    with open(source, "rb") as handle:
        while True:
            header = handle.read(12)
            if not header:
                break
            payload = handle.read(struct.unpack("<Q", header[:8])[0])
            handle.read(4)
            module_hash, epochs, _, _, metrics = json.loads(payload.decode("utf-8"))
            if int(epochs) != 108:
                continue
            final = _protobuf(_protobuf(base64.b64decode(metrics))[1][2])
            validation_accuracy = struct.unpack("<d", final[4][-1])[0]
            training_time = struct.unpack("<d", final[2][-1])[0]
            entry = sums.setdefault(module_hash, [0, 0.0, 0.0])
            entry[0] += 1
            entry[1] += validation_accuracy
            entry[2] += training_time
    if len(sums) != NAS101_ARCHITECTURES or any(e[0] != 3 for e in sums.values()):
        raise RuntimeError("unexpected NAS-Bench-101 content (%d architectures)" % len(sums))
    table = {h: (1.0 - acc / 3.0, cost) for h, (_, acc, cost) in sums.items()}
    write_pickle(target, {"schema": NAS101_SCHEMA, "table": table})
    if downloaded:
        os.remove(source)
    return True


def nasbench201(check: bool) -> bool:
    """Keep the 200-epoch validation error (mean of 3 seeds) and total cost."""
    directory = os.path.join(hpo_dir(), "nasbench201")
    targets = {name: os.path.join(directory, name + ".pkl") for name in NATS_DATASETS}
    if all(os.path.exists(p) for p in targets.values()) or check:
        return all(os.path.exists(p) for p in targets.values())
    source, downloaded = fetch_verified(NATS_URL, os.path.join(hpo_dir(), "NATS-tss-v1_0-3ffb9-simple.tar"),
                                        "sha256", NATS_SHA256, "NATS_TSS_ARCHIVE")
    tables: Dict[str, Dict[str, tuple]] = {name: {} for name in NATS_DATASETS}

    def lookup(mapping: Dict, key):  # NATS-Bench mixes integer and string keys
        return mapping[key] if key in mapping else mapping[str(key)]

    with tarfile.open(source, mode="r|") as archive:
        for member in archive:
            base = os.path.basename(member.name)
            if not member.isfile() or not base.endswith(".pickle.pbz2") or not base.split(".")[0].isdigit():
                continue
            record = pickle.loads(bz2.decompress(archive.extractfile(member).read()))["200"]
            operations = [part.split("|")[-1] for part in record["arch_str"].split("~")[:-1]]
            arch_id = "".join(str(NAS201_OPERATIONS.index(op)) for op in operations)
            for name, (source_name, split) in NATS_DATASETS.items():
                rows = []
                for seed in (777, 888, 999):
                    result = record["all_results"].get((source_name, seed))
                    if result is None:
                        continue
                    # Final (epoch 200, zero-indexed 199) accuracy and the summed cost, as in HPOBench.
                    accuracy = float(lookup(result["eval_acc1es"], "%s@199" % split))
                    train = sum(float(lookup(result["train_times"], e)) for e in range(1, 200))
                    evaluation = sum(float(lookup(result["eval_times"], "%s@%d" % (split, e))) for e in range(1, 200))
                    cost = train + evaluation
                    rows.append((accuracy, cost))
                while len(rows) < 3:  # as in HPOBench, a missing seed repeats the last available run
                    rows.append(rows[-1])
                tables[name][arch_id] = (100.0 - sum(r[0] for r in rows) / 3.0, sum(r[1] for r in rows))
    for name, table in tables.items():
        if len(table) != 5 ** 6:
            raise RuntimeError("unexpected NATS-Bench content for %s (%d architectures)" % (name, len(table)))
        write_pickle(targets[name], {"schema": NAS201_SCHEMA, "table": table})
    if downloaded:
        os.remove(source)
    return True


STEPS: Dict[str, Callable[[bool], bool]] = {
    "search": search, "audit": audit, "paramnet": paramnet,
    "nasbench101": nasbench101, "nasbench201": nasbench201,
}
GROUPS = {"all": list(STEPS), "tuning": ["paramnet", "nasbench101", "nasbench201"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="all",
                        help="Comma-separated subset of: %s, tuning, all." % ", ".join(STEPS))
    parser.add_argument("--check", action="store_true", help="Only report what is present.")
    args = parser.parse_args()
    names = []
    for part in args.only.split(","):
        names += GROUPS.get(part, [part])
    failed = []
    for name in dict.fromkeys(names):
        if name not in STEPS:
            parser.error("unknown dataset %r" % name)
        print("[%s]" % name)
        try:
            ready = STEPS[name](args.check)
        except Exception as exc:
            print("  error: %s" % exc)
            ready = False
        print("  %s" % ("ready" if ready else "missing"))
        if not ready:
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
