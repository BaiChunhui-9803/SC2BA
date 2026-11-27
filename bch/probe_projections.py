"""Same-frame projections, quantitative neighborhood agreement, and RL clustering validity."""

from collections import defaultdict
import inspect

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from .lower_action.reward import short_reward
from .upper_state.bktree_distance import distance_matrices
from .upper_state.codebook import hash_features
from .upper_state.influence import HistogramHash, influence_window

VIEW_NAMES = ("relation-refined", "im-medoid", "unit-set", "cnn", "bk-position", "bk-health", "bk-joint")
CLUSTER_BUDGETS = (8, 16, 32, 64)
NEIGHBORS = 5
REFINED_QUANT_WEIGHT = 0.25
DISCOUNT_GAMMA = 0.9


def _tsne(distances, perplexity, seed):
    kwargs = dict(
        n_components=2, metric="precomputed", perplexity=perplexity, random_state=seed,
        init="random", learning_rate="auto", method="exact",
    )
    iter_param = "max_iter" if "max_iter" in inspect.signature(TSNE.__init__).parameters else "n_iter"
    kwargs[iter_param] = 1500
    return TSNE(**kwargs).fit_transform(distances.copy()).astype(float)


def quant_geometry_features(observation):
    """Continuous within-relation geometry: pairwise distances, engagement margin, spread, and HP profile."""
    own = np.array([unit.xy for unit in observation.friendly], dtype=float).reshape(-1, 2)
    enemy = np.array([unit.xy for unit in observation.enemy], dtype=float).reshape(-1, 2)
    own_hp = np.array([unit.health / max(unit.health_max, 1.0) for unit in observation.friendly], dtype=float)
    enemy_hp = np.array([unit.health / max(unit.health_max, 1.0) for unit in observation.enemy], dtype=float)

    if len(own) and len(enemy):
        dmat = cdist(own, enemy)
        own_min = dmat.min(axis=1)
        enemy_min = dmat.min(axis=0)
        d_sorted = np.sort(dmat.ravel())
        quantiles = np.quantile(d_sorted, [0.0, 0.25, 0.5, 0.75, 1.0]) / 20.0
        centroid_dist = float(np.linalg.norm(own.mean(axis=0) - enemy.mean(axis=0))) / 20.0
        own_min_mean = float(own_min.mean()) / 20.0
        enemy_min_mean = float(enemy_min.mean()) / 20.0
        range_margin = float(np.clip((5.75 - dmat) / 5.75, -1.0, 1.0).mean())
        summary = {
            "min_dist": round(float(d_sorted[0]), 3),
            "median_dist": round(float(np.median(d_sorted)), 3),
            "max_dist": round(float(d_sorted[-1]), 3),
            "centroid_dist": round(float(centroid_dist * 20.0), 3),
        }
    else:
        quantiles = np.ones(5)
        centroid_dist = 1.0
        own_min_mean = 1.0
        enemy_min_mean = 1.0
        range_margin = -1.0
        summary = {"min_dist": None, "median_dist": None, "max_dist": None, "centroid_dist": None}

    own_spread = float(cdist(own, own).mean() / 10.0) if len(own) > 1 else 0.0
    enemy_spread = float(cdist(enemy, enemy).mean() / 10.0) if len(enemy) > 1 else 0.0
    own_hp_mean = float(own_hp.mean()) if len(own_hp) else 0.0
    enemy_hp_mean = float(enemy_hp.mean()) if len(enemy_hp) else 0.0
    own_hp_min = float(own_hp.min()) if len(own_hp) else 0.0
    enemy_hp_min = float(enemy_hp.min()) if len(enemy_hp) else 0.0

    vector = np.array([
        *quantiles, centroid_dist, own_min_mean, enemy_min_mean, range_margin,
        own_spread, enemy_spread, own_hp_mean, enemy_hp_mean, own_hp_min, enemy_hp_min,
    ], dtype=float)
    return vector, summary


def compute_rl_targets(rows, observations, episodes):
    """Derive per-window RL outcomes, discounted returns, short rewards, and HP margins."""
    sorted_rows = sorted(rows, key=lambda r: (r["episode"], r["state"]["step"]))
    by_episode = defaultdict(list)
    for index, row in enumerate(sorted_rows):
        by_episode[row["episode"]].append(index)

    n = len(sorted_rows)
    ep_ids = np.array([row["episode"] for row in sorted_rows])
    macros = np.array([row["macro"] for row in sorted_rows])
    behaviors = np.array([row["behavior"] for row in sorted_rows])
    hp_delta = np.array([float(row["hp_change"]) for row in sorted_rows])
    win = np.zeros(n, dtype=float)
    final_margin = np.zeros(n, dtype=float)
    curr_margin = np.zeros(n, dtype=float)
    rem_hp_return = np.zeros(n, dtype=float)
    disc_hp_return = np.zeros(n, dtype=float)
    short_rew = np.zeros(n, dtype=float)
    disc_short_rew = np.zeros(n, dtype=float)
    outcomes = [""] * n
    final_hp_pairs = [[0.0, 0.0]] * n

    for ep_name, indices in by_episode.items():
        ep = episodes.get(ep_name)
        if ep is not None:
            outcome = ep["outcome"]
            z = 1.0 if outcome == "win" else (0.5 if outcome == "draw" else 0.0)
            fm = float(ep["final_hp"][0] - ep["final_hp"][1])
            fhp = [round(float(ep["final_hp"][0]), 1), round(float(ep["final_hp"][1]), 1)]
        else:
            outcome = "unknown"
            z = np.nan
            fm = np.nan
            fhp = [0.0, 0.0]

        sr_list = []
        for pos, idx in enumerate(indices):
            obs = observations[idx]
            own_hp = sum(unit.health for unit in obs.friendly)
            enemy_hp = sum(unit.health for unit in obs.enemy)
            cm = own_hp - enemy_hp
            outcomes[idx] = outcome
            final_hp_pairs[idx] = fhp
            win[idx] = z
            final_margin[idx] = fm
            curr_margin[idx] = cm
            rem_hp_return[idx] = fm - cm if np.isfinite(fm) else np.nan
            if pos + 1 < len(indices):
                r_s = float(short_reward(obs, observations[indices[pos + 1]]))
            else:
                r_s = float(hp_delta[idx] * (255.0 / 45.0) / 10.0)
            short_rew[idx] = r_s
            sr_list.append(r_s)

        g_hp = 0.0
        g_sr = 0.0
        for pos in range(len(indices) - 1, -1, -1):
            idx = indices[pos]
            g_hp = hp_delta[idx] + DISCOUNT_GAMMA * g_hp
            g_sr = sr_list[pos] + DISCOUNT_GAMMA * g_sr
            disc_hp_return[idx] = g_hp
            disc_short_rew[idx] = g_sr

    return {
        "episode": ep_ids,
        "macro": macros,
        "behavior": behaviors,
        "outcomes": outcomes,
        "final_hp_pairs": final_hp_pairs,
        "win": win,
        "final_margin": final_margin,
        "curr_margin": curr_margin,
        "hp_delta": hp_delta,
        "rem_hp_return": rem_hp_return,
        "disc_hp_return": disc_hp_return,
        "short_rew": short_rew,
        "disc_short_rew": disc_short_rew,
    }


def _neighbors(distances, candidates, k):
    """Marginal inclusion probability under uniform tie breaking at the k-th neighbor."""
    eligible = candidates.sum(axis=1) >= k
    weights = np.zeros_like(distances)
    if not np.any(eligible):
        return weights, eligible
    choices = candidates[eligible]
    values = distances[eligible]
    kth = np.partition(np.where(choices, values, np.inf), k - 1, axis=1)[:, [k - 1]]
    tied = choices & np.isclose(values, kth, atol=1e-10, rtol=1e-9)
    closer = choices & (values < kth) & ~tied
    tie_weight = (k - closer.sum(axis=1)) / tied.sum(axis=1)
    weights[eligible] = closer + tied * tie_weight[:, None]
    return weights, eligible


def _overlap(left, right, anchors, k):
    if not np.any(anchors):
        return None
    return round(float((left[anchors] * right[anchors]).sum(axis=1).mean() / k), 6)


def _rank_agreement(left, right):
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    return round(float(spearmanr(left, right).statistic), 6)


def _agreement(matrices, projected, roster):
    n = len(roster)
    if n < 2:
        return None
    k = min(NEIGHBORS, n - 1)
    same = (roster[:, None, :] == roster[None, :, :]).all(axis=2)
    other = ~np.eye(n, dtype=bool)
    within = same & other
    pairs = np.triu(other, k=1)
    same_pairs = pairs & same
    weights = {name: _neighbors(matrix, other, k)[0] for name, matrix in matrices.items()}
    projected_weights = {name: _neighbors(matrix, other, k)[0] for name, matrix in projected.items()}
    within_weights = {name: _neighbors(matrix, within, k)[0] for name, matrix in matrices.items()}
    within_anchors = within.sum(axis=1) >= k
    compare_targets = [name for name in ("im-medoid", "unit-set", "bk-position", "bk-health", "bk-joint")
                       if name in matrices and name in projected_weights]
    return {
        "frames": n, "pairs": int(pairs.sum()), "neighbors": k,
        "same_roster_pairs": int(same_pairs.sum()), "same_roster_anchors": int(within_anchors.sum()),
        "random_overlap": round(k / (n - 1), 6),
        "within_roster_random_overlap": (
            round(float((k / within.sum(axis=1)[within_anchors]).mean()), 6)
            if np.any(within_anchors) else None
        ),
        "roster_chance": round(float(within.mean() * n / (n - 1)), 6),
        "roster_purity": {name: round(float((values * within).sum() / (n * k)), 6)
                          for name, values in weights.items()},
        "comparisons": {
            name: {
                "spearman": _rank_agreement(matrices["cnn"][pairs], matrices[name][pairs]),
                "same_roster_spearman": _rank_agreement(matrices["cnn"][same_pairs], matrices[name][same_pairs]),
                "neighbor_overlap": _overlap(weights["cnn"], weights[name], other.any(axis=1), k),
                "projected_overlap": _overlap(projected_weights["cnn"], projected_weights[name], other.any(axis=1), k),
                "same_roster_neighbor_overlap": _overlap(within_weights["cnn"], within_weights[name], within_anchors, k),
            }
            for name in compare_targets
        },
    }


def _knn_predict(dist, y, candidate_mask, k=5):
    """Cross-episode (or action-matched cross-episode) k-NN prediction metrics."""
    preds, trues = [], []
    finite = np.isfinite(y)
    for i in range(len(y)):
        if not finite[i]:
            continue
        valid = np.flatnonzero(candidate_mask[i] & finite)
        if not len(valid):
            continue
        kk = min(k, len(valid))
        d = dist[i, valid]
        kth = np.partition(d, kk - 1)[kk - 1]
        closer = d < kth - 1e-10
        tied = np.isclose(d, kth, atol=1e-10, rtol=1e-9)
        w = closer.astype(float) + tied.astype(float) * ((kk - closer.sum()) / max(int(tied.sum()), 1))
        preds.append(float((w * y[valid]).sum() / kk))
        trues.append(float(y[i]))
    if len(trues) < 5:
        return {"n": len(trues), "r2": None, "mae": None, "acc": None}
    preds = np.asarray(preds, dtype=float)
    trues = np.asarray(trues, dtype=float)
    var = float(np.var(trues))
    mse = float(np.mean((preds - trues) ** 2))
    r2 = 1.0 - mse / var if var > 1e-12 else None
    mae = float(np.mean(np.abs(preds - trues)))
    acc = float(np.mean((preds >= 0.5) == (trues >= 0.5)))
    return {
        "n": len(trues),
        "r2": round(r2, 4) if r2 is not None else None,
        "mae": round(mae, 4),
        "acc": round(acc, 4),
    }


def _kmedoids_from_dist(dist, k, seed=1201, restarts=8):
    """Deterministic PAM-style k-medoids on a precomputed distance matrix."""
    n = len(dist)
    k = max(1, min(k, n))
    best_cost = np.inf
    best_labels = np.zeros(n, dtype=int)
    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        medoids = [int(rng.integers(n))]
        nearest = dist[:, medoids[0]].copy()
        while len(medoids) < k:
            probs = nearest ** 2
            total = probs.sum()
            if total <= 1e-12:
                remaining = [i for i in range(n) if i not in medoids]
                medoids.append(int(rng.choice(remaining)))
            else:
                medoids.append(int(rng.choice(n, p=probs / total)))
            np.minimum(nearest, dist[:, medoids[-1]], out=nearest)
        medoids = np.asarray(medoids, dtype=int)
        for _it in range(25):
            labels = dist[:, medoids].argmin(axis=1)
            updated = []
            for cid in range(k):
                members = np.flatnonzero(labels == cid)
                if not len(members):
                    updated.append(medoids[cid])
                else:
                    within = dist[np.ix_(members, members)].sum(axis=1)
                    updated.append(members[within.argmin()])
            updated = np.asarray(updated, dtype=int)
            if np.array_equal(updated, medoids):
                break
            medoids = updated
        labels = dist[:, medoids].argmin(axis=1)
        cost = float(dist[np.arange(n), medoids[labels]].sum())
        if cost < best_cost:
            best_cost = cost
            best_labels = labels
    return best_labels


def _bktree_two_level_clusters(pos_dist, hp_dist, tau_pos=1.0, tau_hp=0.5):
    """Exact historical HRL-IM/CBS greedy two-level threshold clustering (Level-1 pos -> Level-2 HP)."""
    n = len(pos_dist)
    l1_centers, l1_clusters = [], []
    for i in range(n):
        best_c, best_d = None, np.inf
        for ci, center in enumerate(l1_centers):
            d = pos_dist[i, center]
            if d <= tau_pos and d < best_d:
                best_d, best_c = d, ci
        if best_c is None:
            l1_centers.append(i)
            l1_clusters.append([i])
        else:
            l1_clusters[best_c].append(i)

    labels = np.zeros(n, dtype=int)
    cid = 0
    for members in l1_clusters:
        l2_centers = []
        for idx in members:
            best_c, best_d = None, np.inf
            for ci, center in enumerate(l2_centers):
                d = hp_dist[idx, center]
                if d <= tau_hp and d < best_d:
                    best_d, best_c = d, ci
            if best_c is None:
                l2_centers.append(idx)
                labels[idx] = cid
                cid += 1
            else:
                labels[idx] = labels[l2_centers[best_c]]
    return labels, len(l1_centers), cid


def _evaluate_clusters(labels, dist, sub_targets, roster_keys):
    """Evaluate a discrete clustering on cross-episode reuse, value/outcome/reward eta^2, LOEO, and geometry."""
    n = len(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    k_eff = int(len(uniq))
    probs = counts / max(n, 1)
    norm_entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(max(k_eff, 2))) if k_eff > 1 else 0.0
    sil = float(silhouette_score(dist, labels, metric="precomputed")) if 1 < k_eff < n else None
    roster_nmi = float(normalized_mutual_info_score(roster_keys, labels))

    eps = sub_targets["episode"]
    macros = sub_targets["macro"]
    cross_ep_hits = sum(1 for i in range(n) if np.any((labels == labels[i]) & (eps != eps[i])))

    metric_out = {}
    for key in ("win", "final_margin", "disc_hp_return", "curr_margin", "hp_delta", "short_rew"):
        y = sub_targets[key]
        finite = np.isfinite(y)
        if finite.sum() < 5:
            metric_out[key] = {"eta2": None, "loeo_n": 0, "loeo_r2": None, "loeo_mae": None, "loeo_acc": None}
            continue
        yf = y[finite]
        lf = labels[finite]
        grand = yf.mean()
        ss_tot = float(((yf - grand) ** 2).sum())
        ss_within = sum(float(((yf[lf == c] - yf[lf == c].mean()) ** 2).sum()) for c in np.unique(lf))
        eta2 = 1.0 - ss_within / ss_tot if ss_tot > 1e-12 else None

        preds, trues = [], []
        for i in range(n):
            if not finite[i]:
                continue
            mask = (labels == labels[i]) & (eps != eps[i]) & finite
            if key in ("hp_delta", "short_rew"):
                mask = mask & (macros == macros[i])
            if np.any(mask):
                preds.append(float(y[mask].mean()))
                trues.append(float(y[i]))
        if len(trues) >= 5:
            p = np.asarray(preds, dtype=float)
            t = np.asarray(trues, dtype=float)
            var = float(np.var(t))
            loeo_r2 = 1.0 - float(np.mean((p - t) ** 2)) / var if var > 1e-12 else None
            loeo_mae = float(np.mean(np.abs(p - t)))
            loeo_acc = float(np.mean((p >= 0.5) == (t >= 0.5))) if key == "win" else None
        else:
            loeo_r2 = loeo_mae = loeo_acc = None

        metric_out[key] = {
            "eta2": round(eta2, 4) if eta2 is not None else None,
            "loeo_n": len(trues),
            "loeo_r2": round(loeo_r2, 4) if loeo_r2 is not None else None,
            "loeo_mae": round(loeo_mae, 4) if loeo_mae is not None else None,
            "loeo_acc": round(loeo_acc, 4) if loeo_acc is not None else None,
        }

    return {
        "k_eff": k_eff,
        "cross_ep_hit_rate": round(cross_ep_hits / max(n, 1), 4),
        "norm_entropy": round(norm_entropy, 4),
        "silhouette": round(sil, 4) if sil is not None else None,
        "roster_nmi": round(roster_nmi, 4),
        "metrics": metric_out,
    }


def _evaluate_validity_scope(matrices, targets, roster_keys, centroid_dist, im_hashes, indices, seed):
    n = len(indices)
    if n < 6:
        return None
    sub_idx = np.ix_(indices, indices)
    sub_mats = {name: mat[sub_idx] for name, mat in matrices.items()}
    sub_targets = {
        key: targets[key][indices]
        for key in ("episode", "macro", "behavior", "win", "final_margin", "curr_margin",
                    "hp_delta", "disc_hp_return", "short_rew", "disc_short_rew")
    }
    sub_roster = roster_keys[indices]
    sub_geom = centroid_dist[indices]

    eps = sub_targets["episode"]
    macros = sub_targets["macro"]
    diff_ep = eps[:, None] != eps[None, :]
    same_macro_diff_ep = diff_ep & (macros[:, None] == macros[None, :])
    n_episodes = int(len(np.unique(eps)))
    if n_episodes < 2:
        return None

    knn_order = ("relation", "relation-refined", "im-medoid", "unit-set", "cnn", "bk-position", "bk-health", "bk-joint")
    knn_methods = [m for m in knn_order if m in sub_mats]
    knn_stats = {}
    for method in knn_methods:
        dmat = sub_mats[method]
        knn_stats[method] = {
            "win": _knn_predict(dmat, sub_targets["win"], diff_ep, NEIGHBORS),
            "final_margin": _knn_predict(dmat, sub_targets["final_margin"], diff_ep, NEIGHBORS),
            "disc_hp_return": _knn_predict(dmat, sub_targets["disc_hp_return"], diff_ep, NEIGHBORS),
            "curr_margin": _knn_predict(dmat, sub_targets["curr_margin"], diff_ep, NEIGHBORS),
            "same_action_hp_delta": _knn_predict(dmat, sub_targets["hp_delta"], same_macro_diff_ep, NEIGHBORS),
            "uncond_hp_delta": _knn_predict(dmat, sub_targets["hp_delta"], diff_ep, NEIGHBORS),
            "same_action_short_rew": _knn_predict(dmat, sub_targets["short_rew"], same_macro_diff_ep, NEIGHBORS),
            "geom_centroid": _knn_predict(dmat, sub_geom, diff_ep, NEIGHBORS),
        }

    # Historical two-level fixed-threshold BK-tree (1.0 / 0.5)
    bk2_labels, l1_count, l2_count = _bktree_two_level_clusters(
        sub_mats["bk-position"], sub_mats["bk-health"], tau_pos=1.0, tau_hp=0.5,
    )
    historical_bktree = {
        "tau_pos": 1.0,
        "tau_hp": 0.5,
        "l1_clusters": int(l1_count),
        "l2_clusters": int(l2_count),
        **_evaluate_clusters(bk2_labels, sub_mats["bk-joint"], sub_targets, sub_roster),
    }

    # Original IM hash exact string baseline
    original_im_hash = None
    if im_hashes is not None and "im-medoid" in sub_mats:
        sub_hashes = [im_hashes[i] for i in indices]
        uniq_h = {}
        hash_labels = np.array([uniq_h.setdefault(h, len(uniq_h)) for h in sub_hashes], dtype=int)
        original_im_hash = _evaluate_clusters(hash_labels, sub_mats["im-medoid"], sub_targets, sub_roster)

    # Matched cluster budget K comparison across methods
    cluster_order = ("im-medoid", "unit-set", "cnn", "bk-position", "bk-health", "bk-joint", "relation-refined")
    cluster_methods = [m for m in cluster_order if m in sub_mats]
    matched_k = {}
    for k_budget in CLUSTER_BUDGETS:
        if k_budget >= n - 1:
            continue
        matched_k[str(k_budget)] = {}
        for method in cluster_methods:
            labels = _kmedoids_from_dist(sub_mats[method], k_budget, seed=seed)
            matched_k[str(k_budget)][method] = _evaluate_clusters(labels, sub_mats[method], sub_targets, sub_roster)

    # Identical-pair dispersion statistics (pure qualitative vs refined qualitative+quantitative)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    zero_pure = int((sub_mats["relation"][upper] < 1e-9).sum())
    zero_refined = int((sub_mats["relation-refined"][upper] < 1e-9).sum())
    total_pairs = int(upper.sum())

    # Split-diagnosis: among qualitatively-close cross-episode pairs (same roster & low d_qual),
    # compare reward/value divergence of pairs kept NEAR (bottom 25%) vs pushed FAR (top 25%) in each space.
    same_roster_mat = sub_roster[:, None] == sub_roster[None, :]
    qual_close = upper & diff_ep & same_roster_mat & (sub_mats["relation"] < 0.15)
    if qual_close.sum() < 20:
        qual_close = upper & diff_ep & same_roster_mat
    qual_close_same_act = qual_close & (macros[:, None] == macros[None, :])

    def _pair_gap(arr, mask):
        if not np.any(mask):
            return None
        ii, jj = np.where(mask)
        vals = np.abs(arr[ii] - arr[jj])
        finite = np.isfinite(vals)
        return round(float(vals[finite].mean()), 3) if np.any(finite) else None

    split_bars = {}
    for method in [m for m in ("im-medoid", "unit-set", "cnn", "bk-joint", "relation-refined") if m in sub_mats]:
        dm = sub_mats[method]
        entry = {}
        for tag, mask in (("all", qual_close), ("same_action", qual_close_same_act)):
            if mask.sum() < 16:
                entry[tag] = None
                continue
            q25, q75 = np.percentile(dm[mask], [25, 75])
            near_m = mask & (dm <= q25)
            far_m = mask & (dm >= q75)
            entry[tag] = {
                "pairs": int(mask.sum()),
                "dhp_near": _pair_gap(sub_targets["hp_delta"], near_m),
                "dhp_far": _pair_gap(sub_targets["hp_delta"], far_m),
                "sr_near": _pair_gap(sub_targets["short_rew"], near_m),
                "sr_far": _pair_gap(sub_targets["short_rew"], far_m),
                "hp_margin_near": _pair_gap(sub_targets["curr_margin"], near_m),
                "hp_margin_far": _pair_gap(sub_targets["curr_margin"], far_m),
                "win_near": _pair_gap(sub_targets["win"], near_m),
                "win_far": _pair_gap(sub_targets["win"], far_m),
            }
        split_bars[method] = entry

    # Mine top 4 concrete "Qual-close but IM-CNN-far with divergent ΔHP/reward" cross-episode case pairs
    split_cases = []
    if "cnn" in sub_mats and qual_close.sum() >= 8:
        ii, jj = np.where(qual_close)
        cnn_d = sub_mats["cnn"][ii, jj]
        qual_d = sub_mats["relation"][ii, jj]
        dhp_gap = np.abs(sub_targets["hp_delta"][ii] - sub_targets["hp_delta"][jj])
        sr_gap = np.abs(sub_targets["short_rew"][ii] - sub_targets["short_rew"][jj])
        cnn_p75 = float(np.percentile(cnn_d, 75))
        score = (cnn_d / max(float(cnn_d.max()), 1e-6)) * (0.6 * (dhp_gap / max(float(dhp_gap.max()), 1e-6))
                                                          + 0.4 * (sr_gap / max(float(sr_gap.max()), 1e-6)))
        order = np.argsort(score)[::-1]
        used_frames = set()
        for pos in order:
            if cnn_d[pos] < cnn_p75:
                continue
            a_local, b_local = int(ii[pos]), int(jj[pos])
            a_orig, b_orig = int(indices[a_local]), int(indices[b_local])
            if a_orig in used_frames or b_orig in used_frames:
                continue
            used_frames.add(a_orig)
            used_frames.add(b_orig)
            split_cases.append({
                "a": a_orig,
                "b": b_orig,
                "roster": str(sub_roster[a_local]),
                "same_action": bool(macros[a_local] == macros[b_local]),
                "qual_dist": round(float(qual_d[pos]), 3),
                "cnn_dist": round(float(cnn_d[pos]), 3),
                "dhp_gap": round(float(dhp_gap[pos]), 2),
                "sr_gap": round(float(sr_gap[pos]), 2),
                "hp_margin_gap": round(float(abs(sub_targets["curr_margin"][a_local] - sub_targets["curr_margin"][b_local])), 1),
            })
            if len(split_cases) >= 4:
                break

    return {
        "frames": n,
        "episodes": n_episodes,
        "pairs": total_pairs,
        "collapsed_pairs": {
            "pure_qualitative": zero_pure,
            "refined_qualitative": zero_refined,
            "reduction_ratio": round(1.0 - zero_refined / max(zero_pure, 1), 4) if zero_pure else 0.0,
        },
        "knn": knn_stats,
        "matched_k": matched_k,
        "original_im_hash": original_im_hash,
        "historical_bktree": historical_bktree,
        "split_diagnosis": {
            "qual_close_pairs": int(qual_close.sum()),
            "qual_close_same_action_pairs": int(qual_close_same_act.sum()),
            "by_method": split_bars,
            "cases": split_cases,
        },
    }


def build_refined_relation_distance(relation_features, observations):
    """Combine macro qualitative relation distance with normalized continuous unit-distance micro-differences."""
    quant_pairs = [quant_geometry_features(obs) for obs in observations]
    quant_matrix = np.vstack([pair[0] for pair in quant_pairs])
    geom_summaries = [pair[1] for pair in quant_pairs]
    std = np.where(quant_matrix.std(axis=0) > 1e-8, quant_matrix.std(axis=0), 1.0)
    quant_z = (quant_matrix - quant_matrix.mean(axis=0)) / std
    d_qual = cdist(relation_features, relation_features)
    d_quant = cdist(quant_z, quant_z)
    scale = float(np.percentile(d_quant[d_quant > 0], 95)) if np.any(d_quant > 0) else 1.0
    d_quant_norm = d_quant / max(scale, 1e-8)
    d_refined = d_qual + REFINED_QUANT_WEIGHT * d_quant_norm
    centroid_dist = quant_matrix[:, 5] * 20.0
    return d_qual, d_refined, centroid_dist, geom_summaries


def _im_hash_distance(observations):
    hasher = HistogramHash()
    hashes = [hasher(influence_window(obs), obs.influence_scale) for obs in observations]
    feats = np.stack([hash_features(h) for h in hashes])
    dist = cdist(feats, feats, metric="hamming") * float(feats.shape[1])
    return hashes, dist


def compare_projections(
    encoder, observations, relation_features, roster, scopes, seed,
    targets=None, unit_set_encoder=None, im_medoid_encoder=None,
):
    latents = encoder.embed_many(observations)
    position, hp = distance_matrices(observations)
    d_qual, d_refined, centroid_dist, geom_summaries = build_refined_relation_distance(relation_features, observations)
    im_hashes, d_im_hamming = _im_hash_distance(observations)

    pos_scale = float(np.percentile(position[position > 0], 50)) if np.any(position > 0) else 1.0
    hp_scale = float(np.percentile(hp[hp > 0], 50)) if np.any(hp > 0) else 1.0
    bk_joint = (position / max(pos_scale, 1e-8)) + 0.5 * (hp / max(hp_scale, 1e-8))

    matrices = {
        "relation": d_qual,
        "relation-refined": d_refined,
        "im-medoid": d_im_hamming,
        "cnn": cdist(latents, latents),
        "bk-position": position,
        "bk-health": hp,
        "bk-joint": bk_joint,
    }
    if unit_set_encoder is not None:
        unit_latents = unit_set_encoder.embed_many(observations)
        matrices["unit-set"] = cdist(unit_latents, unit_latents)

    active_views = [name for name in VIEW_NAMES if name in matrices]
    perplexity = min(20.0, len(observations) / 4)
    xy = {name: _tsne(matrices[name], perplexity, seed) for name in active_views}
    projected = {name: cdist(coords, coords) for name, coords in xy.items()}
    results = {}
    validity = {}
    roster_arr = np.asarray(roster)
    roster_keys = np.array([f"{row[0]}v{row[1]}" for row in roster_arr])
    for name, indices in scopes.items():
        subset = np.ix_(indices, indices)
        results[name] = _agreement(
            {method: distance[subset] for method, distance in matrices.items()},
            {method: distance[subset] for method, distance in projected.items()},
            roster_arr[indices],
        )
        if targets is not None:
            validity[name] = _evaluate_validity_scope(
                matrices, targets, roster_keys, centroid_dist, im_hashes, indices, seed,
            )

    def _meta_block(enc):
        if enc is None:
            return None
        m = enc.fit_manifest
        return {
            "sha256": enc.fingerprint,
            **{key: m[key] for key in ("seed", "latent_dim", "epochs", "best_epoch", "sample_counts") if key in m},
        }

    return {
        "coordinates": {name: coords.round(5).tolist() for name, coords in xy.items()},
        "perplexity": perplexity,
        "refined_weight": REFINED_QUANT_WEIGHT,
        "agreement": results,
        "validity": validity,
        "geom_summaries": geom_summaries,
        "im_hashes": im_hashes,
        "cnn": _meta_block(encoder),
        "unit_set": _meta_block(unit_set_encoder),
        "im_medoid": _meta_block(im_medoid_encoder),
    }, matrices
