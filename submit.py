"""生成与校验提交文件。

提交格式（CSV，含表头）：
    row_id,user_id,video_id,score

  row_id   : 0 起的行号，对应 data.load()[split] 的行序（确定性：先读
             log_standard_4_08_to_4_21_pure.csv 再读 log_standard_4_22_to_5_08_pure.csv，
             按 date 过滤后保持原文件顺序）
  user_id  : 该行的 user_id（冗余字段，仅用于校验对齐）
  video_id : 该行的 video_id（冗余字段，仅用于校验对齐）
  score    : 你的模型给该行打的分，任意实数，只用相对大小

为什么带 row_id：(user_id, video_id) 在评测集里**不唯一**
（test 集有 3.06% 的重复对，最多重复 12 次），所以无法作为主键。

用法：
    python3 submit.py --make   submission.csv     # 用官方 FM baseline 生成一份示例提交
    python3 submit.py --check  submission.csv     # 校验格式与对齐
    python3 submit.py --score --split valid submission_valid.csv
                                                    # 校验并打分（仅 valid）
"""
import argparse
import csv
import math

from data import load, encode
from evaluate import evaluate

HEADER = ['row_id', 'user_id', 'video_id', 'score']

def write_submission(path, rows, scores):
    """写出提交文件，并在打开目标文件前校验分数数量和有效性。"""
    if not hasattr(rows, '__len__'):
        rows = list(rows)
    if not hasattr(scores, '__len__'):
        scores = list(scores)
    if len(scores) != len(rows):
        raise ValueError(
            f"score 数量 {len(scores)} 与评测集行数 {len(rows)} 不一致"
        )
    numeric_scores = []
    for i, score in enumerate(scores):
        try:
            value = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"第 {i} 个 score 无法解析为数字：{score!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"第 {i} 个 score 是 NaN/Inf，不允许")
        numeric_scores.append(value)

    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, (x, score) in enumerate(zip(rows, numeric_scores)):
            w.writerow([i, x[1], x[2], f"{score:.17g}"])

def read_submission(path, rows):
    """读取并逐行校验对齐，返回 scores。任何不一致都抛出可读错误。"""
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        head = next(r, None)
        if head != HEADER:
            raise ValueError(f"表头必须是 {','.join(HEADER)}，实际是 {head}")
        scores, n = [], 0
        for ln, rec in enumerate(r, start=2):
            if len(rec) != 4:
                raise ValueError(f"第 {ln} 行有 {len(rec)} 个字段，应为 4 个")
            rid, uid, vid, sc = rec
            if int(rid) != n:
                raise ValueError(f"第 {ln} 行 row_id={rid}，应为 {n}（必须 0 起连续递增）")
            if n >= len(rows):
                raise ValueError(f"提交行数超过评测集（评测集 {len(rows)} 行）")
            if uid != rows[n][1] or vid != rows[n][2]:
                raise ValueError(f"第 {ln} 行对齐错误：提交 ({uid},{vid})，"
                                 f"评测集第 {n} 行是 ({rows[n][1]},{rows[n][2]})")
            try:
                v = float(sc)
            except ValueError:
                raise ValueError(f"第 {ln} 行 score 无法解析为数字：{sc!r}")
            if v != v or v in (float('inf'), float('-inf')):
                raise ValueError(f"第 {ln} 行 score 是 NaN/Inf，不允许")
            scores.append(v); n += 1
    if n != len(rows):
        raise ValueError(f"提交 {n} 行，评测集 {len(rows)} 行，数量不符")
    return scores

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='test', choices=['valid', 'test'])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--make',  action='store_true', help='用官方 FM baseline 生成示例提交')
    g.add_argument('--check', action='store_true', help='只校验格式与对齐')
    g.add_argument('--score', action='store_true', help='校验并打分（仅 valid 划分）')
    a = ap.parse_args(argv)

    if a.score and a.split != 'valid':
        ap.error('--score is restricted to --split valid; held-out test scoring is disabled')

    if a.make:
        load_splits = ('train', 'valid', a.split)
    else:
        load_splits = (a.split,)
    splits = load(a.data_dir, split_names=load_splits)
    rows = splits[a.split]

    if a.make:
        import baseline as B, numpy as np
        enc, dim = encode(splits)
        Xtr, ytr, _ = enc['train']
        Xva, yva, uva = enc['valid']
        X, y, u = enc[a.split]
        m = B.FM(dim, k=16, lr=0.001, seed=0)
        rng = np.random.default_rng(0)
        best, state, bad = -1, None, 0
        for ep in range(40):
            idx = rng.permutation(len(ytr))
            for i in range(0, len(idx), 8192):
                m.step(Xtr[idx[i:i+8192]], ytr[idx[i:i+8192]])
            p = evaluate(uva, yva, m.predict(Xva))['primary']
            if p > best + 1e-5: best, bad, state = p, 0, (m.V.copy(), m.W.copy(), m.b)
            else:
                bad += 1
                if bad >= 4: break
        m.V, m.W, m.b = state
        write_submission(a.path, rows, m.predict(X))
        print(f"已写出 {a.path}：{len(rows):,d} 行（split={a.split}，官方 FM baseline）")
    else:
        scores = read_submission(a.path, rows)
        print(f"✓ 格式与对齐校验通过：{len(scores):,d} 行，split={a.split}")
        if a.score:
            r = evaluate([x[1] for x in rows], [x[6] for x in rows], scores)
            print(f"  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")


if __name__ == '__main__':
    main()
