#!/usr/bin/env python3
"""
Tarefa 5, passo 2 — train a WGAN-GP on the harvested input corpus and generate
synthetic TEST INPUTS (plus a matched random baseline).

Reuses the project's Generator/Critic (python/model.py) and the gradient
penalty / feature matching losses (python/utils.py), so the architecture under
evaluation is the same one selected in Tarefa 1.

The corpus (see extract_test_input_corpus.py) pools all method families into a
single 12-dim vector [arr_len, v0..v7, s0, s1, s2]. One generator is trained
over the pooled corpus and each generated vector is decoded per method family
downstream. Pooling is deliberate: with only ~121 real inputs, a per-family
generator (ARRAY_INT has 6, INT2 has 7) would have far too little data.

Four generators are emitted, forming a ladder of increasingly demanding
baselines. The point of the ladder is that beating only the weakest one would
say very little:

    RANDOM  uniform over the per-dimension [min, max] range of the real corpus.
            Matching the range matters: a baseline drawing from an arbitrary
            range would be trivially worse and the comparison would say nothing
            about what the WGAN-GP learnt. This is the *floor*.

    CORPUS  resampling the real corpus with replacement. Asks the sceptical
            question directly: the WGAN-GP trained on 121 human-written inputs, so
            would simply reusing those inputs do just as well? Note its ceiling
            is 121 distinct inputs no matter the budget, which is precisely the
            limitation a generator is supposed to lift.

    KDE     a Gaussian kernel density estimate fitted to the same corpus, with
            the bandwidth chosen by cross-validation. This is the "do you need
            a WGAN-GP at all?" control: a classical density model with the same
            training data and the same sampling budget. Mirrors the model
            ladder already used for the testability predictor, where the
            RandomForest turned out to add only 0.007 AUC over logistic
            regression.

    WGAN    the WGAN-GP under evaluation.

Usage:
    python python/scripts/generate_test_inputs_wgan.py \
        --corpus outputs/task5/input_corpus.csv \
        --n-generate 200 --epochs 400 --seed 42

    # ablacao do feature matching (o termo que foi decisivo em E2)
    python python/scripts/generate_test_inputs_wgan.py --fm-weight 0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.optim as optim

from model import Critic, Generator
from utils import (
    compute_energy_distance,
    compute_feature_matching_loss,
    compute_gradient_penalty,
    compute_wasserstein_distance,
)

SEP = "=" * 72
Z_DIM = 16
MAX_ARR = 8
VEC_COLS = ["arr_len"] + [f"v{i}" for i in range(MAX_ARR)] + ["s0", "s1", "s2"]
N_DIMS = len(VEC_COLS)


class MinMaxTanh:
    """Scale each dimension to [-1, 1] (Generator's Tanh range) and back."""

    def __init__(self, X: np.ndarray):
        self.lo = X.min(axis=0)
        self.hi = X.max(axis=0)
        self.rng = np.where(self.hi - self.lo == 0, 1.0, self.hi - self.lo)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.clip(2 * (X - self.lo) / self.rng - 1, -1, 1)

    def inverse(self, Xs: np.ndarray) -> np.ndarray:
        return (Xs + 1) / 2 * self.rng + self.lo


def _load_corpus(path: Path) -> tuple[np.ndarray, list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[c]) for c in VEC_COLS] for r in rows], dtype=np.float64)
    return X, rows


def _train_wgan(X_scaled: np.ndarray, epochs: int, device: torch.device,
                seed: int, fm_weight: float, verbose: bool,
                z_dim: int = Z_DIM, conditional: bool = False) -> Generator:
    """Train the WGAN-GP, optionally conditioned on the declared array length.

    Unconditioned, the network must model all twelve dimensions jointly, and
    that is where the encoding fights it. `arr_len` governs which of `v0..v7`
    the oracle will ever read -- OracleRunner slices with
    Arrays.copyOfRange(v, 0, arrLen) -- so the value slots carry a hard
    conditional dependency on a dimension the generator is also having to
    invent. Worse, 63% of the corpus declares an empty or null array, so the
    slots are zero most of the time and the easy way to lower the loss is to
    emit near-zero values always. The rare long-array inputs then come out with
    plausible marginals and meaningless contents.

    Conditioning states the dependency instead of hoping it is inferred:
    `arr_len` is drawn from the corpus's own empirical distribution and handed
    to both networks, which then model the eleven remaining dimensions *given*
    it. The critic receives the same condition, without which it could not
    penalise a sample that is plausible on its own but wrong for its length.

    conditional=False reproduces the previous network exactly, so the
    unconditional results already reported stay valid without re-running.
    """
    torch.manual_seed(seed)
    cond_dim = 1 if conditional else 0
    out_dim = N_DIMS - cond_dim
    g = Generator(z_dim=z_dim, output_dim=out_dim, hidden_dims=[64, 128],
                  cond_dim=cond_dim).to(device)
    c = Critic(input_dim=out_dim, hidden_dims=[128, 64], use_minibatch_std=False,
               cond_dim=cond_dim).to(device)
    g_opt = optim.Adam(g.parameters(), lr=1e-4, betas=(0.5, 0.9))
    c_opt = optim.Adam(c.parameters(), lr=1e-4, betas=(0.5, 0.9))

    data = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    n = data.size(0)
    batch_size = min(32, n)

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            batch = data[perm[i:i + batch_size]]
            bs = batch.size(0)
            if bs < 2:
                continue
            # Column 0 is arr_len: the condition when conditioning is on, just
            # another modelled dimension when it is off.
            cond = batch[:, :1] if conditional else None
            real = batch[:, 1:] if conditional else batch
            # compute_gradient_penalty only ever calls its first argument, so a
            # closure binding the condition is enough to keep it working here.
            gp_critic = (lambda t: c(t, cond)) if conditional else c

            for _ in range(5):
                c_opt.zero_grad()
                z = torch.randn(bs, z_dim, device=device)
                fake = g(z, cond).detach()
                loss = c(fake, cond).mean() - c(real, cond).mean()
                gp = compute_gradient_penalty(gp_critic, real, fake, device, bs, 10.0)
                (loss + gp).backward()
                c_opt.step()

            g_opt.zero_grad()
            z = torch.randn(bs, z_dim, device=device)
            fake = g(z, cond)
            g_loss = -c(fake, cond).mean()
            if fm_weight > 0:
                g_loss = g_loss + fm_weight * compute_feature_matching_loss(real, fake)
            g_loss.backward()
            g_opt.step()

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            with torch.no_grad():
                m = min(200, n * 4)
                z = torch.randn(m, z_dim, device=device)
                if conditional:
                    idx = torch.randint(0, n, (m,), device=device)
                    cond_s = data[idx, :1]
                    sample = torch.cat([cond_s, g(z, cond_s)], dim=1).cpu().numpy()
                else:
                    sample = g(z).cpu().numpy()
                w1 = compute_wasserstein_distance(sample, X_scaled)
            print(f"    epoch {epoch + 1:>4}/{epochs}  W1={w1:.4f}", flush=True)

    g.eval()
    return g


@torch.no_grad()
def _generate(g: Generator, n: int, device: torch.device, seed: int,
              z_dim: int = Z_DIM, cond: torch.Tensor | None = None) -> np.ndarray:
    torch.manual_seed(seed + 999)
    z = torch.randn(n, z_dim, device=device)
    if cond is None:
        return g(z).cpu().numpy()
    # The condition IS the arr_len column, so it is prepended rather than
    # generated: the emitted length is exactly a value the corpus contains.
    return np.concatenate([cond.cpu().numpy(), g(z, cond).cpu().numpy()], axis=1)


def _decode(vectors: np.ndarray) -> list[dict]:
    """Round to integers, clamp arr_len into [-1, MAX_ARR], and canonicalise.

    Canonicalisation zeroes the value slots at or beyond arr_len. Those slots
    are already ignored downstream -- OracleRunner builds the argument with
    Arrays.copyOfRange(v, 0, arrLen) -- so this changes no test behaviour
    whatsoever. What it changes is the honesty of the fidelity metrics.

    Left un-canonicalised, 92% of generated inputs carried values in slots that
    can never be read, against 0% in the real corpus, and those dead slots were
    averaged into the reported W1 alongside the ones that matter. Since 63% of
    corpus inputs declare an empty or null array, most of the twelve dimensions
    were dead most of the time: 5.21 of 12 carried behaviour on average. The
    metric was therefore dominated by noise the generator is not even asked to
    get right.

    Applied to every arm, so the comparison still isolates where the inputs came
    from rather than which arm happened to emit structurally valid vectors.
    """
    out = []
    for vec in vectors:
        row = {c: int(round(float(v))) for c, v in zip(VEC_COLS, vec)}
        row["arr_len"] = max(-1, min(MAX_ARR, row["arr_len"]))
        for i in range(max(0, row["arr_len"]), MAX_ARR):
            row[f"v{i}"] = 0
        out.append(row)
    return out


def _as_matrix(rows: list[dict]) -> np.ndarray:
    return np.array([[r[c] for c in VEC_COLS] for r in rows], dtype=np.float64)


def _write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VEC_COLS)
        writer.writeheader()
        writer.writerows(rows)


def _fit_kde(X_scaled: np.ndarray, seed: int) -> "KernelDensity":
    """Gaussian KDE with the bandwidth picked by cross-validation.

    Giving the KDE a tuned bandwidth is what makes it a fair control: the WGAN
    gets hundreds of epochs of fitting, so handing its rival an arbitrary
    bandwidth would rig the comparison. The grid is log-spaced over the range
    that is sensible for data already scaled to [-1, 1].
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KernelDensity

    grid = GridSearchCV(
        KernelDensity(kernel="gaussian"),
        {"bandwidth": np.logspace(-2, 0, 20)},
        cv=min(5, len(X_scaled)),
    )
    grid.fit(X_scaled)
    return grid.best_estimator_


def _structural_violations(rows: list[dict]) -> int:
    """Inputs carrying values in slots the oracle can never read.

    Should be 0 after _decode canonicalises. Kept as an assertion in metric
    form: if it ever climbs, the encoding and the oracle have drifted apart.
    """
    bad = 0
    for r in rows:
        used = max(0, r["arr_len"])
        if any(r[f"v{i}"] != 0 for i in range(used, MAX_ARR)):
            bad += 1
    return bad


def _live_values(rows: list[dict]) -> np.ndarray:
    """Pool every value slot the oracle will actually read.

    This is the view the aggregate W1 cannot give. That metric averages over
    all twelve dimensions, so the value slots of an input declaring arr_len=0
    -- dead weight the oracle never touches -- count exactly as much as the
    slots of one declaring arr_len=5. A generator can therefore look faithful
    while every array it emits is filled with nonsense, because most arrays are
    empty and the empty ones dominate the average.

    Restricting to live slots asks the question directly: when this generator
    does commit to a non-empty array, are the contents drawn from anything like
    the real distribution? It is the metric the conditioning is meant to move.
    """
    vals: list[float] = []
    for r in rows:
        for i in range(max(0, r["arr_len"])):
            vals.append(float(r[f"v{i}"]))
    return np.asarray(vals, dtype=np.float64)


def _novelty(gen: np.ndarray, real: np.ndarray) -> dict:
    """How far the generated inputs sit from the real ones they were learnt from.

    Distances are euclidean in the scaled space. A generator that memorises the
    corpus scores ~0 here and its apparent gain would be the corpus's, not its
    own; one that ignores the corpus scores high and is effectively noise.
    """
    d = np.sqrt(((gen[:, None, :] - real[None, :, :]) ** 2).sum(-1))
    dmin = d.min(axis=1)
    return {
        "nn_dist_median": round(float(np.median(dmin)), 4),
        "nn_dist_p10": round(float(np.percentile(dmin, 10)), 4),
        "n_exact_copies": int((dmin < 1e-9).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train WGAN-GP on the test-input corpus and generate synthetic inputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--corpus", type=str, default="outputs/task5/input_corpus.csv")
    parser.add_argument("--out-dir", type=str, default="outputs/task5")
    parser.add_argument("--n-generate", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--fm-weight", type=float, default=0.3,
                        help="Feature matching weight. A sweep over {0, 0.3, 1.0} at 800 "
                             "epochs moved W1 by under 0.001, so this knob is inert here.")
    parser.add_argument("--z-dim", type=int, default=Z_DIM,
                        help="Latent width. The default exceeds the 12 output dimensions "
                             "(of which ~5.2 carry behaviour), so smaller values are worth testing.")
    parser.add_argument("--corpus-frac", type=float, default=1.0,
                        help="Train on this fraction of the corpus. Below 1.0 this becomes the "
                             "low-data regime where synthesis is supposed to pay off; the CORPUS "
                             "arm is then restricted to the same subset, so both see equal data.")
    parser.add_argument("--conditional", action="store_true",
                        help="Condition the generator and critic on arr_len, drawn from the "
                             "corpus's empirical distribution, instead of asking the network "
                             "to invent it alongside the value slots it governs.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"[ERRO] Corpus not found: {corpus_path} — run extract_test_input_corpus.py first")
        sys.exit(1)

    X_full, rows = _load_corpus(corpus_path)
    device = torch.device("cpu")
    rng = np.random.default_rng(args.seed)

    # Low-data regime: every generator, including the CORPUS baseline, is
    # restricted to the same subset. Subsetting only the WGAN's training data
    # would hand CORPUS an information advantage and make the comparison
    # meaningless.
    if args.corpus_frac < 1.0:
        k = max(4, int(round(len(X_full) * args.corpus_frac)))
        X = X_full[rng.choice(len(X_full), size=k, replace=False)]
    else:
        X = X_full

    print(SEP)
    print("  WGAN-GP — geracao de INPUTS de teste (Tarefa 5)")
    print(SEP)
    print(f"\n  Corpus real : {len(X)} inputs de {len(X_full)}, {N_DIMS} dimensoes"
          f"{'' if args.corpus_frac >= 1.0 else f'  (frac={args.corpus_frac})'}")
    print(f"  WGAN        : {args.epochs} epochs, z_dim={args.z_dim}, "
          f"fm_weight={args.fm_weight}, seed={args.seed}"
          f"{', condicionada em arr_len' if args.conditional else ''}")
    print(f"  A gerar     : {args.n_generate} inputs por braco (wgan, random, corpus, kde)\n")

    scaler = MinMaxTanh(X)
    X_scaled = scaler.forward(X)
    n = args.n_generate

    # --- WGAN-GP ----------------------------------------------------------
    g = _train_wgan(X_scaled, args.epochs, device, args.seed, args.fm_weight,
                    verbose=True, z_dim=args.z_dim, conditional=args.conditional)
    cond = None
    if args.conditional:
        # Drawn from a private stream so that turning conditioning on does not
        # shift the draws of the RANDOM, CORPUS and KDE arms. Those three are
        # the regression control: if they move between the two modes, the
        # change touched the shared pipeline rather than the generator.
        cond_rng = np.random.default_rng(args.seed + 7)
        idx = cond_rng.integers(0, len(X_scaled), size=n)
        cond = torch.tensor(X_scaled[idx, :1], dtype=torch.float32, device=device)
    gan_scaled = _generate(g, n, device, args.seed, z_dim=args.z_dim, cond=cond)

    # --- RANDOM: uniform over the corpus's per-dimension range ------------
    rand_raw = rng.uniform(X.min(axis=0), X.max(axis=0), size=(n, N_DIMS))
    rand_scaled = scaler.forward(rand_raw)

    # --- CORPUS: resample the real inputs with replacement ----------------
    # Deliberately capped at the corpus's own diversity: with 121 distinct rows
    # it cannot yield more than 121 distinct inputs however large the budget.
    corpus_raw = X[rng.integers(0, len(X), size=n)]
    corpus_scaled = scaler.forward(corpus_raw)

    # --- KDE: classical density model on the same data --------------------
    kde = _fit_kde(X_scaled, args.seed)
    kde_scaled = np.clip(kde.sample(n, random_state=args.seed), -1, 1)

    arms = {
        "wgan": gan_scaled,
        "random": rand_scaled,
        "corpus": corpus_scaled,
        "kde": kde_scaled,
    }

    out_dir = Path(args.out_dir)
    metrics = {
        "seed": args.seed,
        "fm_weight": args.fm_weight,
        "epochs": args.epochs,
        "z_dim": args.z_dim,
        "conditional": args.conditional,
        "corpus_frac": args.corpus_frac,
        "n_generate": n,
        "corpus_size": len(X),
        "corpus_size_full": len(X_full),
        "kde_bandwidth": round(float(kde.bandwidth), 4),
        "arms": {},
    }

    print(f"\n{SEP}")
    print("  RESULTADO")
    print(SEP)
    print(f"\n  KDE bandwidth escolhida por CV: {kde.bandwidth:.4f}")
    print(f"\n  {'Braco':<9}{'W1':>9}{'W1emit':>9}{'energy':>9}{'W1viv':>8}{'nviv':>7}"
          f"{'unicos':>8}{'copias':>8}{'d(viz)':>9}{'viol.':>8}")
    print(f"  {'-' * 84}")

    real_live = _live_values(_decode(X))

    for name, scaled in arms.items():
        rows = _decode(scaler.inverse(scaled))
        _write(rows, out_dir / f"inputs_{name}.csv")

        # Two fidelity readings, deliberately kept side by side.
        #   w1        continuous output before rounding, as reported so far, so
        #             the series in the dissertation stays comparable
        #   w1_emit   the integer vectors actually written out and executed;
        #             rounding is a large distortion and the first number does
        #             not see it
        # Plus the energy distance, which unlike either W1 responds to how the
        # dimensions are coupled -- the property that separates a structurally
        # valid input from one that merely has the right marginals.
        emitted_scaled = scaler.forward(_as_matrix(rows))
        w1 = float(compute_wasserstein_distance(scaled, X_scaled))
        w1_emit = float(compute_wasserstein_distance(emitted_scaled, X_scaled))
        energy = float(compute_energy_distance(emitted_scaled, X_scaled))

        uniq = len({tuple(r.values()) for r in rows})
        nov = _novelty(emitted_scaled, X_scaled)
        viol = _structural_violations(rows)

        gen_live = _live_values(rows)
        if len(gen_live) and len(real_live):
            from scipy.stats import wasserstein_distance
            # Normalised by the corpus's own value range so this is on the same
            # scale as the other W1 readings, which are computed in [-1, 1].
            span = max(float(real_live.max() - real_live.min()), 1.0)
            w1_live = float(wasserstein_distance(real_live, gen_live)) * 2 / span
        else:
            w1_live = float("nan")

        metrics["arms"][name] = {
            "w1": round(w1, 4), "w1_emitted": round(w1_emit, 4),
            "energy": round(energy, 4), "w1_live": round(w1_live, 4),
            "n_live_slots": int(len(gen_live)), "n_unique": uniq,
            "n_generated": len(rows), "structural_violations": viol, **nov,
        }
        print(f"  {name:<9}{w1:>9.4f}{w1_emit:>9.4f}{energy:>9.4f}{w1_live:>8.4f}"
              f"{len(gen_live):>7}{uniq:>8}{nov['n_exact_copies']:>8}"
              f"{nov['nn_dist_median']:>9.3f}{viol:>8}")

    with open(out_dir / "generation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  {'Dimensao':<10}{'real min':>10}{'real max':>10}{'WGAN min':>10}{'WGAN max':>10}")
    print(f"  {'-' * 50}")
    gan_arr = scaler.inverse(gan_scaled)
    for i, col in enumerate(VEC_COLS):
        print(f"  {col:<10}{X[:, i].min():>10.0f}{X[:, i].max():>10.0f}"
              f"{gan_arr[:, i].min():>10.0f}{gan_arr[:, i].max():>10.0f}")

    print(f"\n  Inputs   : {out_dir}/inputs_{{wgan,random,corpus,kde}}.csv")
    print(f"  Metricas : {out_dir / 'generation_metrics.json'}")


if __name__ == "__main__":
    main()
