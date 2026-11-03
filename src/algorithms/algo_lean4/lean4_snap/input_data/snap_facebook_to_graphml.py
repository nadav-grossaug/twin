#!/usr/bin/env python3
"""Convert SNAP Facebook ego networks and combined network into attributed GraphML.

Features are intentionally scoped by ego ID: the same feature column position in
separate ego networks is not assumed to carry the same meaning. A JSON manifest
preserves exact raw feature names and the generated GraphML keys.
"""
import argparse
import gzip
import json
import re
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import networkx as nx

BASE = 'https://snap.stanford.edu/data/'


def download_if_missing(dest, filename):
    target = dest / filename
    if not target.is_file():
        print(f'Downloading {filename}...')
        urllib.request.urlretrieve(BASE + filename, target)
    return target


def extract_ego_archive(dest, archive):
    ego_dir = dest / 'facebook'
    if not list(dest.rglob('*.featnames')):
        with tarfile.open(archive, 'r:gz') as tar:
            # Only extract expected simple dataset files; no arbitrary paths.
            for member in tar.getmembers():
                p = Path(member.name)
                if (not member.isfile() or p.name == '' or
                    p.suffix not in {'.edges', '.circles', '.feat', '.egofeat', '.featnames'} or
                    len(p.parts) > 2 or '..' in p.parts):
                    continue
                output = dest / p
                output.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, open(output, 'wb') as dst:
                    dst.write(src.read())
    dirs = sorted({p.parent for p in dest.rglob('*.featnames')})
    if len(dirs) != 1:
        raise RuntimeError(f'Expected one ego network folder, found {dirs}')
    return dirs[0]


def read_feature_names(path):
    names = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        index, raw_name = line.split(maxsplit=1)
        names[int(index)] = raw_name.strip()
    return names


def safe_key(ego, index, raw_name):
    # Namespace every feature by originating ego network and index.
    slug = re.sub(r'[^A-Za-z0-9]+', '_', raw_name).strip('_')[:75]
    return f'ego_{ego}__f{index}_{slug}' if slug else f'ego_{ego}__f{index}'


GRAPHML_NS = 'http://graphml.graphdrawing.org/xmlns'


def descriptive_key_ids(path):
    """Rename auto-generated GraphML key ids (d0, d1, ...) to carry a short
    description of the attribute, e.g. d2291 -> d2291_education_year."""
    ET.register_namespace('', GRAPHML_NS)
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    tree = ET.parse(path)
    root = tree.getroot()
    rename = {}
    for key in root.findall(f'{{{GRAPHML_NS}}}key'):
        short = key.get('attr.name', '')
        short = re.sub(r'^ego_\d+__(f\d+_)?', '', short)
        short = re.sub(r'(_id)?_anonymized_feature(_\d+)?$', '', short)
        short = re.sub(r'[^A-Za-z0-9]+', '_', short).strip('_')[:40]
        old = key.get('id')
        rename[old] = f'{old}_{short}' if short else old
        key.set('id', rename[old])
    for data in root.iter(f'{{{GRAPHML_NS}}}data'):
        data.set('key', rename.get(data.get('key'), data.get('key')))
    tree.write(path, xml_declaration=True, encoding='utf-8')


def read_circles(path):
    membership = defaultdict(list)
    if path.is_file():
        for line in path.read_text().splitlines():
            parts = line.split()
            if parts:
                for node in parts[1:]:
                    membership[int(node)].append(parts[0])
    return membership


def feature_rows(path, is_ego=False, ego=None):
    if not path.is_file():
        return
    with path.open() as f:
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            if is_ego:
                yield ego, [int(x) for x in tokens]
            else:
                yield int(tokens[0]), [int(x) for x in tokens[1:]]


def convert(ego_dir, combined_path, output):
    output.mkdir(parents=True, exist_ok=True)
    ego_files = sorted(ego_dir.glob('*.edges'), key=lambda p: int(p.stem))
    if not ego_files:
        raise RuntimeError(f'No .edges files found in {ego_dir}')

    # Combined file is authoritative for the exported full graph's topology.
    with gzip.open(combined_path, 'rt') as f:
        full = nx.parse_edgelist(f, nodetype=int, create_using=nx.Graph())

    full.graph['dataset'] = 'SNAP Facebook social circles'
    full.graph['feature_merge_policy'] = 'ego-scoped; sparse: absent key means 0 or unobserved'
    manifest = {'feature_policy': 'ego-scoped; same column index across egos is NOT merged; sparse: absent key means 0 or unobserved',
                'features': {}, 'ego_networks': []}

    for edge_path in ego_files:
        ego = int(edge_path.stem)
        prefix = edge_path.with_suffix('')
        # Load the ego-specific friendship subgraph.
        ego_graph = nx.read_edgelist(edge_path, nodetype=int, create_using=nx.Graph())
        feat_rows = list(feature_rows(prefix.with_suffix('.feat')))
        # Alters with no friendships among themselves appear only in .feat,
        # never in .edges; every alter is still a friend of the ego.
        alters = set(ego_graph.nodes) | {node for node, _ in feat_rows}
        ego_graph.add_node(ego)
        ego_graph.add_edges_from((ego, friend) for friend in alters)
        ego_graph.graph['ego_id'] = str(ego)
        ego_graph.graph['dataset'] = 'SNAP Facebook ego network'
        for node in ego_graph:
            ego_graph.nodes[node]['is_ego'] = (node == ego)

        full.nodes[ego]['is_dataset_ego'] = True
        full.nodes[ego]['ego_network_id'] = str(ego)
        names = read_feature_names(prefix.with_suffix('.featnames'))
        key_map = {i: safe_key(ego, i, raw) for i, raw in names.items()}
        for index, raw in names.items():
            manifest['features'][key_map[index]] = {
                'ego_id': ego, 'local_feature_index': index, 'raw_feature_name': raw
            }

        # Read friend and central user's features. Stored sparsely: only
        # nonzero values are written; an absent key means 0 or unobserved.
        rows = feat_rows + list(feature_rows(prefix.with_suffix('.egofeat'), is_ego=True, ego=ego))
        for node, values in rows:
            if node not in ego_graph:
                raise ValueError(f'Node {node} has features but is absent from ego {ego}')
            if node not in full:
                raise ValueError(f'Node {node} is absent from combined graph')
            if len(values) != len(names):
                raise ValueError(f'Feature count mismatch for ego {ego}, node {node}: {len(values)} vs {len(names)}')
            for i, v in enumerate(values):
                if v == 0:
                    continue
                key = key_map[i]
                ego_graph.nodes[node][key] = v
                full.nodes[node][key] = v

        circles = read_circles(prefix.with_suffix('.circles'))
        for node in ego_graph:
            memberships = circles.get(node, [])
            ego_graph.nodes[node]['circles'] = '|'.join(memberships)
            # Encode circle memberships with ego-network namespace in combined graph.
            full.nodes[node][f'ego_{ego}__circles'] = '|'.join(memberships)

        ego_out = output / f'facebook_ego_{ego}.graphml'
        nx.write_graphml(ego_graph, ego_out)
        descriptive_key_ids(ego_out)
        manifest['ego_networks'].append({'ego_id': ego, 'file': ego_out.name,
                                         'nodes': ego_graph.number_of_nodes(),
                                         'edges': ego_graph.number_of_edges(),
                                         'features': len(names)})
        print(f'{ego_out.name}: {ego_graph.number_of_nodes()} nodes, {ego_graph.number_of_edges()} edges, {len(names)} features')

    full_out = output / 'facebook_full_features.graphml'
    nx.write_graphml(full, full_out)
    descriptive_key_ids(full_out)
    (output / 'facebook_feature_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{full_out.name}: {full.number_of_nodes()} nodes, {full.number_of_edges()} edges')
    print('Feature key mapping: facebook_feature_manifest.json')
    if (full.number_of_nodes(), full.number_of_edges()) != (4039, 88234):
        print('WARNING: Combined graph counts do not match published SNAP dataset statistics')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('snap_facebook_data'))
    parser.add_argument('--output-dir', type=Path, default=Path('graphml'))
    parser.add_argument('--no-download', action='store_true')
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.no_download:
        archive = args.data_dir / 'facebook.tar.gz'
        combined = args.data_dir / 'facebook_combined.txt.gz'
    else:
        archive = download_if_missing(args.data_dir, 'facebook.tar.gz')
        combined = download_if_missing(args.data_dir, 'facebook_combined.txt.gz')
    ego_data = extract_ego_archive(args.data_dir, archive)
    convert(ego_data, combined, args.output_dir)
