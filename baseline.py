"""KuaiRand-Pure baselines.
  --model pop   : item popularity (official statistical baseline; no training)
  --model fm    : Factorization Machine (starter model to improve)
  --model random: random scores (lower bound and evaluator sanity check)
Requires only NumPy. See README.md for usage.
"""
import argparse, collections, time
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ---------------- Item popularity (official baseline) ----------------
def run_pop(splits, prior=20.0):
    pos, imp = collections.Counter(), collections.Counter()
    for x in splits['train']:
        imp[x[2]] += 1; pos[x[2]] += x[6]
    gmean = sum(pos.values()) / sum(imp.values())
    score = lambda v: (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             [score(x[2]) for x in rws])
    return out

def run_random(splits, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             rng.random(len(rws)))
    return out

# ---------------- Factorization Machine ----------------
class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)    # (B,)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


def _fit_pointwise_encoded(
    enc, dim, k=16, lr=0.001, epochs=40, bs=8192, patience=4,
    seed=0, verbose=True,
):
    """Fit one FM on an existing encoding and restore its best checkpoint."""
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    model = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        indices = rng.permutation(len(ytr))
        losses = [
            model.step(Xtr[indices[i:i + bs]], ytr[indices[i:i + bs]])
            for i in range(0, len(indices), bs)
        ]
        valid = evaluate(uva, yva, model.predict(Xva))
        if verbose:
            print(
                f"  seed {seed} epoch {epoch:2d} | loss {np.mean(losses):.4f} | "
                f"valid primary {valid['primary']:.6f}"
            )
        if valid['primary'] > best + 1e-5:
            best, bad = valid['primary'], 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                break
    model.V, model.W, model.b = best_state
    return model


def run_fm_ensemble(
    splits, seeds=(0, 1, 2, 3, 4), k=16, lr=0.001, epochs=40,
    bs=8192, patience=4, verbose=True, evaluate_test=True,
    return_models=False,
):
    """Average independently seeded FM scores and evaluate the ensemble once."""
    enc, dim = encode(splits)
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']
    valid_scores = np.zeros(len(yva), dtype=np.float64)
    test_scores = np.zeros(len(yte), dtype=np.float64) if evaluate_test else None
    models = []
    for count, seed in enumerate(seeds, start=1):
        model = _fit_pointwise_encoded(
            enc, dim, k=k, lr=lr, epochs=epochs, bs=bs,
            patience=patience, seed=seed, verbose=verbose,
        )
        valid_scores += model.predict(Xva)
        models.append(model)
        if evaluate_test:
            test_scores += model.predict(Xte)
        if verbose:
            partial = evaluate(uva, yva, valid_scores / count)
            print(f"  ensemble {count} model(s) | valid primary {partial['primary']:.6f}")
    size = len(tuple(seeds))
    metrics = {'valid': evaluate(uva, yva, valid_scores / size)}
    if evaluate_test:
        metrics['test'] = evaluate(ute, yte, test_scores / size)
    if return_models:
        return metrics, models, enc
    return metrics


# ---------------- Ranking-aware FM continuation ----------------
def _within_user_pairs(labels, users, rng, pairs_per_positive=1):
    """Create training-only positive/negative pairs from logged impressions."""
    grouped = collections.defaultdict(lambda: ([], []))
    for index, (user, label) in enumerate(zip(users, labels)):
        grouped[user][0 if label > 0.5 else 1].append(index)

    positive, negative = [], []
    repeats = max(1, int(pairs_per_positive))
    for pos, neg in grouped.values():
        if not pos or not neg:
            continue
        pos_array = np.asarray(pos, dtype=np.int32)
        neg_array = np.asarray(neg, dtype=np.int32)
        for _ in range(repeats):
            positive.append(pos_array)
            negative.append(rng.choice(neg_array, size=len(pos_array), replace=True))
    if not positive:
        raise ValueError("No users with both positive and negative training rows")
    return np.concatenate(positive), np.concatenate(negative)


def _encode_with_crosses(splits, crosses):
    """Encode baseline fields plus explicit categorical crosses from train only."""
    if not crosses:
        return encode(splits)
    allowed = {'video_tab', 'author_tab', 'video_duration', 'author_duration'}
    unknown = set(crosses) - allowed
    if unknown:
        raise ValueError(f"Unknown crosses: {sorted(unknown)}")

    edges = np.quantile(
        np.asarray([x[5] for x in splits['train']]),
        np.linspace(0, 1, 11)[1:-1],
    )

    def raw(row):
        duration = str(int(np.searchsorted(edges, row[5])))
        base = [row[1], row[2], row[3], row[4], duration]
        values = {
            'video_tab': f'{row[2]}|{row[4]}',
            'author_tab': f'{row[3]}|{row[4]}',
            'video_duration': f'{row[2]}|{duration}',
            'author_duration': f'{row[3]}|{duration}',
        }
        return base + [values[name] for name in crosses]

    field_count = 5 + len(crosses)
    vocabs = [dict() for _ in range(field_count)]
    for row in splits['train']:
        for field, value in enumerate(raw(row)):
            if value not in vocabs[field]:
                vocabs[field][value] = len(vocabs[field])
    unk = [len(vocab) for vocab in vocabs]
    dimensions = [len(vocab) + 1 for vocab in vocabs]
    offsets = np.cumsum([0] + dimensions[:-1]).astype(np.int32)

    encoded = {}
    for split, rows in splits.items():
        X = np.empty((len(rows), field_count), dtype=np.int32)
        y = np.empty(len(rows), dtype=np.float32)
        users = []
        for index, row in enumerate(rows):
            for field, value in enumerate(raw(row)):
                X[index, field] = vocabs[field].get(value, unk[field]) + offsets[field]
            y[index] = row[6]
            users.append(row[1])
        encoded[split] = (X, y, users)
    return encoded, int(sum(dimensions))


def _bpr_step(model, Xpos, Xneg):
    """One Adam update for pairwise Bayesian Personalized Ranking loss."""
    batch = len(Xpos)
    zpos, Epos, Spos = model.logits(Xpos)
    zneg, Eneg, Sneg = model.logits(Xneg)
    diff = zpos - zneg
    grad = ((sigmoid(diff) - 1.0) / batch).astype(np.float32)

    gV = np.zeros_like(model.V)
    gW = np.zeros_like(model.W)
    np.add.at(gW, Xpos, grad[:, None])
    np.add.at(gW, Xneg, -grad[:, None])
    np.add.at(gV, Xpos, grad[:, None, None] * (Spos[:, None, :] - Epos))
    np.add.at(gV, Xneg, -grad[:, None, None] * (Sneg[:, None, :] - Eneg))
    gV += model.l2 * model.V
    gW += model.l2 * model.W

    model.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for parameter, gradient, mean, variance in (
        (model.V, gV, model.mV, model.vV),
        (model.W, gW, model.mW, model.vW),
    ):
        mean *= b1
        mean += (1 - b1) * gradient
        variance *= b2
        variance += (1 - b2) * (gradient * gradient)
        parameter -= model.lr * (mean / (1 - b1 ** model.t)) / (
            np.sqrt(variance / (1 - b2 ** model.t)) + eps
        )
    return float(-np.mean(np.log(sigmoid(diff) + 1e-9)))


def run_hybrid_fm(
    splits,
    k=16,
    lr=0.001,
    pointwise_epochs=40,
    bpr_epochs=6,
    bpr_lr=0.0002,
    bs=8192,
    patience=4,
    pairs_per_positive=1,
    crosses=(),
    seed=0,
    verbose=True,
    return_model=False,
):
    """Train the official FM, then continue with within-user BPR updates."""
    enc, dim = _encode_with_crosses(splits, tuple(crosses))
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']
    model = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)

    best_score, best_state, bad = -1.0, None, 0
    for epoch in range(1, pointwise_epochs + 1):
        indices = rng.permutation(len(ytr))
        losses = [
            model.step(Xtr[indices[i:i + bs]], ytr[indices[i:i + bs]])
            for i in range(0, len(indices), bs)
        ]
        valid = evaluate(uva, yva, model.predict(Xva))
        if verbose:
            print(
                f"  pointwise {epoch:2d} | loss {np.mean(losses):.4f} | "
                f"valid primary {valid['primary']:.6f}"
            )
        if valid['primary'] > best_score + 1e-5:
            best_score, bad = valid['primary'], 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                break

    model.V, model.W, model.b = best_state
    model.lr = bpr_lr
    # Reset Adam moments so continuation is governed by the ranking loss rather
    # than stale pointwise-loss momentum.
    model.mV.fill(0); model.vV.fill(0)
    model.mW.fill(0); model.vW.fill(0); model.t = 0
    pair_pos, pair_neg = _within_user_pairs(
        ytr, utr, rng, pairs_per_positive=pairs_per_positive
    )
    bad = 0
    for epoch in range(1, bpr_epochs + 1):
        order = rng.permutation(len(pair_pos))
        losses = []
        for i in range(0, len(order), bs):
            batch_ids = order[i:i + bs]
            losses.append(_bpr_step(model, Xtr[pair_pos[batch_ids]], Xtr[pair_neg[batch_ids]]))
        valid = evaluate(uva, yva, model.predict(Xva))
        if verbose:
            print(
                f"  bpr       {epoch:2d} | loss {np.mean(losses):.4f} | "
                f"valid primary {valid['primary']:.6f}"
            )
        if valid['primary'] > best_score + 1e-5:
            best_score, bad = valid['primary'], 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                break

    model.V, model.W, model.b = best_state
    if return_model:
        return model, enc
    return {
        'valid': evaluate(uva, yva, model.predict(Xva)),
        'test': evaluate(ute, yte, model.predict(Xte)),
    }


def run_hybrid_ensemble(
    splits, seeds=(0, 1, 2, 3), k=16, lr=0.001,
    pointwise_epochs=40, bpr_epochs=6, bpr_lr=0.0002,
    bs=8192, patience=4, pairs_per_positive=1, verbose=True,
    evaluate_test=True, return_models=False,
):
    """Average independently seeded ranking-aware FM continuations."""
    valid_scores = None
    test_scores = None
    models = []
    yva = uva = yte = ute = None
    for count, seed in enumerate(seeds, start=1):
        model, enc = run_hybrid_fm(
            splits, k=k, lr=lr, pointwise_epochs=pointwise_epochs,
            bpr_epochs=bpr_epochs, bpr_lr=bpr_lr, bs=bs,
            patience=patience, pairs_per_positive=pairs_per_positive,
            seed=seed, verbose=verbose, return_model=True,
        )
        Xva, yva, uva = enc['valid']
        Xte, yte, ute = enc['test']
        if valid_scores is None:
            valid_scores = np.zeros(len(yva), dtype=np.float64)
            test_scores = np.zeros(len(yte), dtype=np.float64) if evaluate_test else None
        valid_scores += model.predict(Xva)
        models.append(model)
        if evaluate_test:
            test_scores += model.predict(Xte)
        if verbose:
            partial = evaluate(uva, yva, valid_scores / count)
            print(f"  hybrid ensemble {count} model(s) | valid primary {partial['primary']:.6f}")
    size = len(tuple(seeds))
    metrics = {'valid': evaluate(uva, yva, valid_scores / size)}
    if evaluate_test:
        metrics['test'] = evaluate(ute, yte, test_scores / size)
    if return_models:
        return metrics, models, enc
    return metrics

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='Data directory of the extracted KuaiRand-Pure dataset')
    ap.add_argument(
        '--model', default='fm',
        choices=['pop', 'fm', 'ensemble', 'hybrid', 'hybrid-ensemble', 'random'],
    )
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bpr-epochs', type=int, default=6)
    ap.add_argument('--bpr-lr', type=float, default=0.0002)
    ap.add_argument('--pairs-per-positive', type=int, default=1)
    ap.add_argument(
        '--crosses', default='',
        help='Comma-separated explicit crosses: video_tab, author_tab, '
             'video_duration, author_duration',
    )
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ensemble-seeds', default='0,1,2,3,4')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed),
           'ensemble': lambda s: run_fm_ensemble(
               s, seeds=tuple(int(x) for x in a.ensemble_seeds.split(',')),
               k=a.k, lr=a.lr, epochs=a.epochs,
           ),
           'hybrid': lambda s: run_hybrid_fm(
               s, k=a.k, lr=a.lr, pointwise_epochs=a.epochs,
               bpr_epochs=a.bpr_epochs, bpr_lr=a.bpr_lr,
               pairs_per_positive=a.pairs_per_positive,
               crosses=tuple(x for x in a.crosses.split(',') if x), seed=a.seed,
           ),
           'hybrid-ensemble': lambda s: run_hybrid_ensemble(
               s, seeds=tuple(int(x) for x in a.ensemble_seeds.split(',')),
               k=a.k, lr=a.lr, pointwise_epochs=a.epochs,
               bpr_epochs=a.bpr_epochs, bpr_lr=a.bpr_lr,
               pairs_per_positive=a.pairs_per_positive,
           )}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
