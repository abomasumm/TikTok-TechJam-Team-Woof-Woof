"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""
import csv
import os

import numpy as np

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
SPLIT_FILES = {
    'train': ('log_standard_4_08_to_4_21_pure.csv',),
    'valid': ('log_standard_4_22_to_5_08_pure.csv',),
    'test':  ('log_standard_4_22_to_5_08_pure.csv',),
}
# 5 个特征域。想加特征就往这里加 —— 这是学生最该动的地方之一。
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
RESEARCH_COLUMNS = (
    'user_id', 'video_id', 'date', 'hourmin', 'time_ms',
    'is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
    'long_view', 'play_time_ms', 'duration_ms', 'tab',
)


def _requested_splits(split_names):
    if split_names is None:
        return tuple(SPLITS)
    if isinstance(split_names, str):
        split_names = (split_names,)
    requested = tuple(dict.fromkeys(split_names))
    if not requested:
        raise ValueError('split_names must contain at least one split')
    unknown = [name for name in requested if name not in SPLITS]
    if unknown:
        raise ValueError(f"unknown split name(s): {unknown}; expected one of {list(SPLITS)}")
    return requested


def load(data_dir, split_names=None):
    """读日志 + 视频侧特征，返回按划分切好的 dict。

    ``split_names`` 可用于只加载指定划分。默认为 ``None``，保留原有
    行为（train/valid/test 全部返回）。开发阶段应显式传入
    ``('train', 'valid')``，避免物化 test 行及其标签。
    """
    requested = _requested_splits(split_names)
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv'), newline='') as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    out = {name: [] for name in requested}
    files = tuple(dict.fromkeys(
        filename
        for name in requested
        for filename in SPLIT_FILES[name]
    ))
    for filename in files:
        with open(os.path.join(data_dir, filename), newline='') as fh:
            for r in csv.DictReader(fh):
                date = int(r['date'])
                for name in requested:
                    lo, hi = SPLITS[name]
                    if lo <= date <= hi:
                        # Only convert/materialize fields (including the label) for
                        # a requested split. This keeps held-out payloads out of a
                        # normal train+validation development run.
                        out[name].append((
                            date,
                            r['user_id'],
                            r['video_id'],
                            vid2author.get(r['video_id'], 'UNK'),
                            r['tab'],
                            float(r['duration_ms']),
                            1 if r[LABEL] != '0' else 0,
                        ))
                        break
    return out


def iter_interactions(data_dir, split_names=('train', 'valid'), columns=RESEARCH_COLUMNS):
    """Yield selected raw signals as ``(split_name, values)`` pairs.

    This streaming interface exposes timestamps and training-only auxiliary
    outcomes needed by causal-history and multi-task experiments without
    changing the compact tuple returned by :func:`load`. Values intentionally
    remain CSV strings so each experiment must choose and document its own
    normalization. Development callers should keep the default train/valid
    boundary; the orchestrator supplies generated candidates a view containing
    no held-out rows.
    """
    requested = _requested_splits(split_names)
    columns = tuple(columns)
    if not columns:
        raise ValueError('columns must contain at least one field')
    files = tuple(dict.fromkeys(
        filename
        for name in requested
        for filename in SPLIT_FILES[name]
    ))
    for filename in files:
        with open(os.path.join(data_dir, filename), newline='') as fh:
            reader = csv.DictReader(fh)
            fieldnames = set(reader.fieldnames or ())
            missing = [column for column in columns if column not in fieldnames]
            if missing:
                raise ValueError(f'{filename} is missing requested columns: {missing}')
            for row in reader:
                date = int(row['date'])
                for name in requested:
                    lo, hi = SPLITS[name]
                    if lo <= date <= hi:
                        yield name, tuple(row[column] for column in columns)
                        break

def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])

def encode(splits):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    返回 (X, y, users) per split，X 为 int32 (N, len(FIELDS))，以及 field_dims。"""
    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]                 # 每个域末尾留一个 UNK 槽
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))
