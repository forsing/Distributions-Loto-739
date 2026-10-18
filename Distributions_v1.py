#!/usr/bin/env python3


"""
Sadrzi: 
1. Proveru i opis podataka.
2. Devet analiza raspodela.
3. Grafikone stvarnih podataka.
4. Podesavanje referentnih raspodela.
5. Beta-Bernoulli azuriranje.
6. Poisson regresiju i OLS poredjenje.
7. Ridge i kernel regresiju za jednu NEXT kombinaciju.
8. Hronolosku validaciju i test.

Nema:
- random generisanja ili uzorkovanja;
- klasifikatora;
- frekvencijskog rangiranja pojedinacnih brojeva;
- nametanja normalne ili uniformne raspodele prediktoru;
- rucnih filtera oblika kombinacije;
- zaokruzivanja predvidjenih pozicija brojeva;
- istorijskih kombinacija kao pocetnih kandidata.

Svaki CSV obradjuje se potpuno zasebno.
Ulazni CSV fajlovi se ne menjaju.
Program cuva nove izvestaje u zasebnom direktorijumu.

Zavisnosti:
numpy scipy matplotlib scikit-learn
"""


import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import csv
import hashlib
import math
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler


DEFAULTS = [
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83_loto_2974.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83_loto_plus_1730.csv",
]

LAGS = (1, 2, 4, 8, 16, 32)
MAX_LAG = max(LAGS)

VALIDATION = 48
TEST = 64
REFIT = 16

BLOCK = 20
EVENT_THRESHOLD = 2

CONFIGS = (
    ("ridge", 0.1),
    ("ridge", 1.0),
    ("kernel", 0.1),
    ("kernel", 1.0),
)

POSITION_PAIRS = np.asarray(
    list(combinations(range(7), 2)),
    dtype=np.int32,
)

NUMBER_PAIRS = np.asarray(
    list(combinations(range(39), 2)),
    dtype=np.int32,
)

PAIRS_BY_SIZE = {
    size: np.asarray(
        list(combinations(range(size), 2)),
        dtype=np.int32,
    )
    for size in range(2, 8)
}

ALL_START_PAIRS = tuple(
    combinations(range(1, 40), 2)
)

TOTAL_COMBINATIONS = math.comb(39, 7)


def read_csv(path):
    rows = []

    with Path(path).open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for line, cells in enumerate(csv.reader(handle), 1):
            if not cells or all(not c.strip() for c in cells):
                continue

            try:
                row = tuple(
                    sorted(int(c.strip()) for c in cells)
                )
            except ValueError as exc:
                raise ValueError(
                    f"{path}, red {line}: neispravan ceo broj."
                ) from exc

            if (
                len(row) != 7
                or len(set(row)) != 7
                or not 1 <= row[0] <= row[-1] <= 39
            ):
                raise ValueError(
                    f"{path}, red {line}: potrebno je "
                    "7 razlicitih brojeva 1..39."
                )

            rows.append(row)

    minimum = MAX_LAG + 96 + VALIDATION + TEST

    if len(rows) < minimum:
        raise ValueError(
            f"{path}: potrebno najmanje {minimum} kola; "
            f"pronadjeno {len(rows)}."
        )

    return np.asarray(rows, dtype=np.int32)


def save_csv(path, header, rows):
    with Path(path).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def profile(rows):
    """Opis cele kombinacije i svih parnih rastojanja."""
    a = np.atleast_2d(rows).astype(np.float64)

    distances = (
        a[:, POSITION_PAIRS[:, 1]]
        - a[:, POSITION_PAIRS[:, 0]]
    )

    distance_cdf = np.stack(
        [
            (distances <= boundary).mean(axis=1)
            for boundary in (2, 5, 10, 20)
        ],
        axis=1,
    )

    span = (a[:, -1:] - a[:, :1]) / 38.0

    return np.concatenate(
        (a / 39.0, distance_cdf, span),
        axis=1,
    )


def combination_rank(row):
    """Tacni leksikografski rang kombinacije, pocinje od nule."""
    result = 0
    previous = 0

    for i, value in enumerate(map(int, row)):
        remaining = 7 - i

        result += (
            math.comb(39 - previous, remaining)
            - math.comb(40 - value, remaining)
        )

        previous = value

    return result


class Dataset:
    def __init__(self, rows):
        self.rows = rows
        self.n = len(rows)

        self.sums = rows.sum(axis=1).astype(float)
        self.odd = (rows % 2).sum(axis=1).astype(float)
        self.spans = (rows[:, -1] - rows[:, 0]).astype(float)

        distances = (
            rows[:, POSITION_PAIRS[:, 1]]
            - rows[:, POSITION_PAIRS[:, 0]]
        )

        self.mean_distance = distances.mean(axis=1)

        self.overlap = np.full(self.n, np.nan)

        for t in range(1, self.n):
            self.overlap[t] = len(
                set(map(int, rows[t - 1]))
                & set(map(int, rows[t]))
            )

        # Celokombinacijski dogadjaj za analizu raspodela:
        # najmanje dva ponovljena broja iz prethodnog kola.
        self.event = np.full(self.n, np.nan)

        self.event[1:] = (
            self.overlap[1:] >= EVENT_THRESHOLD
        ).astype(float)

        self.age = np.zeros(self.n, dtype=float)

        for t in range(1, self.n):
            self.age[t] = (
                0.0
                if self.event[t] == 1.0
                else self.age[t - 1] + 1.0
            )

        self.ranks = np.asarray(
            [combination_rank(row) for row in rows],
            dtype=np.int64,
        )

        self.features = profile(rows)

        x = []

        for t in range(MAX_LAG, self.n + 1):
            states = {
                lag: self.features[t - lag]
                for lag in LAGS
            }

            parts = [states[lag] for lag in LAGS]

            parts.extend([
                states[1] - states[2],
                states[2] - states[4],
                states[4] - states[8],
            ])

            # Log-transformacije poznatih pozitivnih velicina.
            # Ne namecu normalnu raspodelu.
            parts.append(np.asarray([
                np.log1p(self.spans[t - 1]),
                np.log1p(self.mean_distance[t - 1]),
                np.log1p(self.age[t - 1]),
                self.overlap[t - 1] / 7.0,
                self.event[t - 1],
            ]))

            x.append(np.concatenate(parts))

        self.x = np.asarray(x, dtype=np.float64)

        inc = np.zeros(
            (self.n - MAX_LAG, 39),
            dtype=np.float64,
        )

        targets = rows[MAX_LAG:]

        inc[
            np.arange(len(targets))[:, None],
            targets - 1,
        ] = 1.0

        # Regresioni ciljevi: zajednicko prisustvo parova
        # u stvarnom narednom kolu.
        self.y_pairs = (
            inc[:, NUMBER_PAIRS[:, 0]]
            * inc[:, NUMBER_PAIRS[:, 1]]
        )

        self.y_count = self.overlap[MAX_LAG:].copy()

        if not np.isfinite(self.x).all():
            raise ValueError("Ulazne osobine nisu konacne.")

        if not np.all(self.y_pairs.sum(axis=1) == 21):
            raise ValueError("Neispravni ciljevi parova.")

        self.verify_alignment()

    def verify_alignment(self):
        t = MAX_LAG

        assert np.array_equal(
            self.x[0, :12],
            self.features[t - 1],
        )

        assert np.array_equal(
            self.x[0, 60:72],
            self.features[t - 32],
        )

        assert len(self.x) == len(self.y_pairs) + 1

        assert combination_rank(
            (1, 2, 3, 4, 5, 6, 7)
        ) == 0

        assert combination_rank(
            (33, 34, 35, 36, 37, 38, 39)
        ) == TOTAL_COMBINATIONS - 1


def make_series(data):
    times = np.arange(data.n)
    hits = np.flatnonzero(data.event == 1.0)

    block_values = []
    block_times = []

    observed_events = data.event[1:]

    for end in range(BLOCK, len(observed_events) + 1, BLOCK):
        block_values.append(
            observed_events[end - BLOCK:end].sum()
        )
        block_times.append(end)

    gaps = np.diff(hits)
    gap_times = hits[1:]

    third_wait = (
        hits[3:] - hits[:-3]
        if len(hits) >= 4
        else np.asarray([], dtype=float)
    )

    third_times = (
        hits[3:]
        if len(hits) >= 4
        else np.asarray([], dtype=int)
    )

    return [
        {
            "family": "normal",
            "title": "Normalna: zbir (aproksimacija)",
            "values": data.sums,
            "times": times,
        },
        {
            "family": "bernoulli",
            "title": "Bernoulli: ponavljanje >= 2",
            "values": data.event[1:],
            "times": times[1:],
        },
        {
            "family": "binomial",
            "title": "Binomna: dogadjaji u 20 kola",
            "values": np.asarray(block_values, dtype=float),
            "times": np.asarray(block_times, dtype=int),
        },
        {
            "family": "geometric",
            "title": "Geometrijska: intervali dogadjaja",
            "values": gaps.astype(float),
            "times": gap_times,
        },
        {
            "family": "poisson",
            "title": "Poisson: broj ponovljenih brojeva",
            "values": data.overlap[1:],
            "times": times[1:],
        },
        {
            "family": "exponential",
            "title": "Eksponencijalna: intervali (aproks.)",
            "values": gaps.astype(float),
            "times": gap_times,
        },
        {
            "family": "gamma",
            "title": "Gamma: cekanje 3 dogadjaja (aproks.)",
            "values": third_wait.astype(float),
            "times": third_times,
        },
        {
            "family": "beta",
            "title": "Beta: relativna parna udaljenost",
            "values": data.mean_distance / 38.0,
            "times": times,
        },
        {
            "family": "uniform",
            "title": "Uniformna referenca: rang kombinacije",
            "values": data.ranks.astype(float),
            "times": times,
        },
    ]


def fit_reference(family, train, event_probability):
    if len(train) == 0:
        return None, "Nema dovoljno posmatranja."

    if family == "normal":
        mu = float(train.mean())
        sd = float(train.std())

        if sd <= 1e-12:
            return None, "Konstantan zbir."

        return stats.norm(loc=mu, scale=sd), (
            f"mu={mu:.6g}; sd={sd:.6g}"
        )

    if family == "bernoulli":
        p = float(train.mean())
        return stats.bernoulli(p), f"p={p:.6g}"

    if family == "binomial":
        return stats.binom(
            BLOCK,
            event_probability,
        ), f"n={BLOCK}; p={event_probability:.6g}"

    if family == "geometric":
        if event_probability <= 0:
            return None, "Nema dogadjaja za geometrijsku referencu."

        return stats.geom(
            event_probability
        ), f"p={event_probability:.6g}"

    if family == "poisson":
        mu = float(train.mean())
        return stats.poisson(mu), f"lambda={mu:.6g}"

    if family == "exponential":
        scale = float(train.mean())

        if scale <= 0:
            return None, "Neispravni intervali."

        return stats.expon(
            loc=0,
            scale=scale,
        ), f"scale={scale:.6g}; kontinuirana aproksimacija"

    if family == "gamma":
        if len(train) < 3 or float(train.std()) <= 1e-12:
            return None, "Premalo varijacije za Gamma fit."

        try:
            shape, loc, scale = stats.gamma.fit(
                train,
                floc=0,
            )
        except (ValueError, RuntimeError, FloatingPointError):
            return None, "Gamma fit nije uspeo za ove podatke."

        return stats.gamma(
            shape,
            loc=loc,
            scale=scale,
        ), (
            f"shape={shape:.6g}; scale={scale:.6g}; "
            "kontinuirana aproksimacija"
        )

    if family == "beta":
        mean = float(train.mean())
        variance = float(train.var())

        if (
            not 0 < mean < 1
            or variance <= 1e-14
        ):
            return None, "Premalo varijacije za Beta fit."

        concentration = (
            mean * (1.0 - mean) / variance - 1.0
        )

        if concentration <= 0:
            return None, "Momenti nisu saglasni sa Beta referencom."

        alpha = mean * concentration
        beta = (1.0 - mean) * concentration

        return stats.beta(
            alpha,
            beta,
        ), f"alpha={alpha:.6g}; beta={beta:.6g}; metod momenata"

    if family == "uniform":
        return stats.randint(
            0,
            TOTAL_COMBINATIONS,
        ), (
            f"diskretna referenca 0..{TOTAL_COMBINATIONS - 1}; "
            "ne koristi se za NEXT"
        )

    raise ValueError(f"Nepoznata porodica: {family}")


def cdf_distance(values, distribution):
    """
    Deskriptivna ECDF distanca.
    Ne prijavljuje p-vrednost niti prihvatanje raspodele.
    """
    if len(values) == 0:
        return np.nan

    unique, counts = np.unique(
        np.asarray(values, dtype=float),
        return_counts=True,
    )

    right = np.cumsum(counts) / len(values)
    left = right - counts / len(values)

    theoretical_right = distribution.cdf(unique)

    theoretical_left = distribution.cdf(
        np.nextafter(unique, -np.inf)
    )

    return float(max(
        np.max(np.abs(right - theoretical_right)),
        np.max(np.abs(left - theoretical_left)),
    ))


def distribution_analysis(data, cutoff, outdir):
    series = make_series(data)

    observed = data.event[1:cutoff]
    event_probability = float(observed.mean())

    print("\nDEVET ANALIZA RASPODELA")
    print(
        "Parametri se podesavaju pre validacionog/test perioda."
    )
    print(
        "D je deskriptivna ECDF distanca, nije p-vrednost."
    )

    records = []

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(16, 12),
    )

    for item, ax in zip(series, axes.ravel()):
        values = np.asarray(item["values"], dtype=float)
        times = np.asarray(item["times"])

        train = values[times < cutoff]
        holdout = values[times >= cutoff]

        distribution, parameters = fit_reference(
            item["family"],
            train,
            event_probability,
        )

        distance = (
            cdf_distance(holdout, distribution)
            if distribution is not None
            else np.nan
        )

        records.append([
            item["family"],
            item["title"],
            len(train),
            len(holdout),
            parameters,
            distance,
        ])

        print(
            f"{item['family']:12s} | "
            f"obuka={len(train):4d} | "
            f"izdvojeno={len(holdout):3d} | "
            f"D={distance:.5f}"
        )
        print(f"  {parameters}")

        display_values = values.copy()

        if item["family"] == "uniform":
            display_values /= TOTAL_COMBINATIONS - 1

        if len(display_values):
            unique_count = len(np.unique(display_values))
            bins = min(40, max(2, unique_count))

            ax.hist(
                display_values,
                bins=bins,
                color="steelblue",
                edgecolor="white",
                alpha=0.85,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "Nema zavrsenih intervala",
                ha="center",
                va="center",
            )

        ax.set_title(item["title"], fontsize=10)
        ax.set_ylabel("Broj posmatranja")

    fig.suptitle(
        "Stvarni podaci — nazivi su referentne porodice,\n"
        "ne tvrdnja da podaci prate te raspodele",
        fontsize=13,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    fig.savefig(
        outdir / "01_devet_raspodela.png",
        dpi=150,
    )

    plt.close(fig)

    save_csv(
        outdir / "02_reference_raspodela.csv",
        [
            "porodica",
            "velicina",
            "n_obuka",
            "n_izdvojeno",
            "parametri",
            "ecdf_distanca",
        ],
        records,
    )

    print(
        "Napomena: intervali su diskretni; eksponencijalna "
        "i Gamma ovde su samo kontinuirane aproksimacije."
    )

    print(
        "Binomna/geometrijska referenca pretpostavljaju "
        "konstantnu stopu i odgovarajucu nezavisnost."
    )

    print(
        "Poisson nije tacan opis ogranicenog broja 0..7; "
        "analiza prikazuje koliko se referenca razlikuje."
    )

    hits = np.flatnonzero(data.event == 1)

    if len(hits):
        print(
            "Nezavrseno cekanje na kraju CSV-a:",
            data.n - 1 - int(hits[-1]),
            "kola; nije dodato medju zavrsene intervale.",
        )


def beta_update(data, outdir):
    """
    Beta-Bernoulli analiza celokombinacijskog dogadjaja.
    Ne koristi se za bodovanje NEXT kombinacije.
    """
    observed = data.event[1:]

    successes = int(observed.sum())
    failures = len(observed) - successes

    # Eksplicitna Jeffreys referenca za verovatnocu dogadjaja.
    alpha_prior = 0.5
    beta_prior = 0.5

    alpha_post = alpha_prior + successes
    beta_post = beta_prior + failures

    mean = alpha_post / (alpha_post + beta_post)

    low, high = stats.beta.ppf(
        [0.025, 0.975],
        alpha_post,
        beta_post,
    )

    grid = np.linspace(0.001, 0.999, 600)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        grid,
        stats.beta.pdf(
            grid,
            alpha_prior,
            beta_prior,
        ),
        linestyle="--",
        label="Beta(0.5, 0.5) referenca",
    )

    ax.plot(
        grid,
        stats.beta.pdf(
            grid,
            alpha_post,
            beta_post,
        ),
        label="Azurirano stvarnim dogadjajima",
    )

    ax.axvline(
        mean,
        color="black",
        linestyle=":",
        label=f"Sredina={mean:.4f}",
    )

    ax.set_xlabel(
        "Verovatnoca najmanje dva ponovljena broja"
    )
    ax.set_ylabel("Gustina")
    ax.set_title("Beta-Bernoulli analiza")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        outdir / "03_beta_azuriranje.png",
        dpi=150,
    )

    plt.close(fig)

    save_csv(
        outdir / "04_beta_azuriranje.csv",
        [
            "uspeha",
            "neuspeha",
            "alpha_prior",
            "beta_prior",
            "alpha_post",
            "beta_post",
            "sredina",
            "donja_025",
            "gornja_975",
        ],
        [[
            successes,
            failures,
            alpha_prior,
            beta_prior,
            alpha_post,
            beta_post,
            mean,
            low,
            high,
        ]],
    )

    print(
        "\nBETA AZURIRANJE: "
        f"sredina={mean:.6f}; "
        f"95% interval=[{low:.6f}, {high:.6f}]"
    )

    print(
        "Interval vazi u Beta-Bernoulli modelu sa konstantnim p; "
        "nije interval verovatnoce dobitka NEXT kombinacije."
    )


def rational_kernel(a, b, bandwidth):
    squared = np.maximum(
        (a * a).sum(axis=1)[:, None]
        + (b * b).sum(axis=1)[None, :]
        - 2.0 * a @ b.T,
        0.0,
    ) / a.shape[1]

    return 1.0 / (
        1.0 + squared / bandwidth
    )


class RegressorHead:
    def __init__(self, bank, config):
        family, penalty = config

        self.bank = bank
        self.family = family

        z = bank.z

        if family == "kernel":
            count = min(64, len(z))

            indexes = np.linspace(
                0,
                len(z) - 1,
                count,
            ).astype(int)

            self.centers = z[indexes].copy()

            distances = (
                (
                    self.centers[:, None]
                    - self.centers[None, :]
                ) ** 2
            ).mean(axis=2)

            positive = distances[
                np.triu_indices(count, 1)
            ]
            positive = positive[positive > 1e-12]

            self.bandwidth = (
                float(np.median(positive))
                if len(positive)
                else 1.0
            )

            phi = rational_kernel(
                z,
                self.centers,
                self.bandwidth,
            )
        elif family == "ridge":
            phi = z
        else:
            raise ValueError(
                f"Nepoznat regresor: {family}"
            )

        self.phi_mean = phi.mean(axis=0)

        self.phi_scale = np.maximum(
            phi.std(axis=0),
            1e-6,
        )

        phi = (
            phi - self.phi_mean
        ) / self.phi_scale

        self.y_mean = bank.y.mean(axis=0)

        gram = (
            phi.T @ phi
        ) / len(phi)

        rhs = (
            phi.T @ (bank.y - self.y_mean)
        ) / len(phi)

        self.coef = np.linalg.solve(
            gram + penalty * np.eye(phi.shape[1]),
            rhs,
        )

    def predict(self, raw):
        z = (
            np.atleast_2d(raw)
            - self.bank.center
        ) / self.bank.scale

        if self.family == "kernel":
            phi = rational_kernel(
                z,
                self.centers,
                self.bandwidth,
            )
        else:
            phi = z

        phi = (
            phi - self.phi_mean
        ) / self.phi_scale

        result = self.y_mean + phi @ self.coef

        if not np.isfinite(result).all():
            raise RuntimeError(
                "Regresor je proizveo neispravne vrednosti."
            )

        return result[0]


class RegressorBank:
    def __init__(self, data, end):
        stop = end - MAX_LAG

        if stop < 2:
            raise ValueError("Premalo prelaza za obuku.")

        # Svi dostupni prelazi; nema poslednjih 192/384.
        x = data.x[:stop]

        self.y = data.y_pairs[:stop]

        self.center = x.mean(axis=0)

        self.scale = np.maximum(
            x.std(axis=0),
            1e-6,
        )

        self.z = (
            x - self.center
        ) / self.scale

        self.count = stop
        self.models = {}

    def get(self, config):
        if config not in self.models:
            self.models[config] = RegressorHead(
                self,
                config,
            )

        return self.models[config]


def pair_matrix(values):
    matrix = np.zeros((39, 39), dtype=float)

    matrix[
        NUMBER_PAIRS[:, 0],
        NUMBER_PAIRS[:, 1],
    ] = values

    matrix[
        NUMBER_PAIRS[:, 1],
        NUMBER_PAIRS[:, 0],
    ] = values

    # Regresioni skorovi ostaju kakvi jesu.
    # Ne pretvaraju se vestacki u verovatnoce.
    return matrix


def candidate_scores(candidates, matrix):
    a = np.asarray(
        candidates,
        dtype=np.int32,
    ) - 1

    pairs = PAIRS_BY_SIZE[a.shape[1]]

    return matrix[
        a[:, pairs[:, 0]],
        a[:, pairs[:, 1]],
    ].sum(axis=1)


def rank_candidates(candidates, matrix, width):
    candidates = sorted(candidates)
    values = candidate_scores(candidates, matrix)

    order = sorted(
        range(len(candidates)),
        key=lambda i: (
            -float(values[i]),
            candidates[i],
        ),
    )

    return [
        candidates[i]
        for i in order[:width]
    ]


def next_combination(values):
    if not np.isfinite(values).all():
        raise RuntimeError("Neispravni skorovi parova.")

    if np.ptp(values) <= 1e-12:
        raise RuntimeError(
            "Regresor ne razlikuje parove; "
            "proizvoljan NEXT nece biti ispisan."
        )

    matrix = pair_matrix(values)

    # Pocetne tacke su svi parovi iz pravila igre,
    # a ne istorijske sedmoclane kombinacije.
    beam = rank_candidates(
        combinations(range(1, 40), 2),
        matrix,
        width=32,
    )

    for _ in range(3, 8):
        candidates = set()

        for row in beam:
            used = set(row)

            for number in range(1, 40):
                if number not in used:
                    candidates.add(
                        tuple(sorted(used | {number}))
                    )

        beam = rank_candidates(
            candidates,
            matrix,
            width=32,
        )

    beam = beam[:8]

    best = float(
        candidate_scores(
            beam[:1],
            matrix,
        )[0]
    )

    for _ in range(4):
        candidates = set(beam)

        for row in beam:
            used = set(row)

            for removed in row:
                remaining = used - {removed}

                for added in range(1, 40):
                    if added not in used:
                        candidates.add(
                            tuple(
                                sorted(
                                    remaining | {added}
                                )
                            )
                        )

        new_beam = rank_candidates(
            candidates,
            matrix,
            width=8,
        )

        value = float(
            candidate_scores(
                new_beam[:1],
                matrix,
            )[0]
        )

        if value <= best + 1e-12:
            break

        beam = new_beam
        best = value

    result = beam[0]

    assert len(result) == len(set(result)) == 7
    assert 1 <= result[0] <= result[-1] <= 39

    return result


def evaluate(data, start, stop, configs, label):
    records = {
        config: {
            "hits": [],
            "mse": [],
            "predictions": [],
        }
        for config in configs
    }

    for t in range(start, stop):
        if (t - start) % REFIT == 0:
            bank = RegressorBank(data, t)

            print(
                f"{label}: {t-start+1}/{stop-start}; "
                f"obuka na {bank.count} prelaza",
                flush=True,
            )

        raw = data.x[t - MAX_LAG]
        actual_pairs = data.y_pairs[t - MAX_LAG]

        for config in configs:
            values = bank.get(config).predict(raw)
            prediction = next_combination(values)

            matches = len(
                set(prediction)
                & set(map(int, data.rows[t]))
            )

            records[config]["hits"].append(matches)

            records[config]["mse"].append(
                float(
                    np.mean(
                        (values - actual_pairs) ** 2
                    )
                )
            )

            records[config]["predictions"].append(
                prediction
            )

    for record in records.values():
        record["hits"] = np.asarray(
            record["hits"],
            dtype=np.int32,
        )

        record["mse"] = np.asarray(
            record["mse"],
            dtype=float,
        )

    return records


def count_regression(data, outdir):
    """
    Clanak: Poisson regresija naspram OLS.
    Cilj: broj ponovljenih brojeva u narednom kolu.
    Ovaj deo ne prepravlja NEXT kombinaciju.
    """
    end = data.n - TEST
    stop = end - MAX_LAG

    x_train = data.x[:stop]
    y_train = data.y_count[:stop]

    x_test = data.x[
        stop:data.n - MAX_LAG
    ]

    y_test = data.y_count[stop:]

    scaler = StandardScaler()

    z_train = scaler.fit_transform(x_train)
    z_test = scaler.transform(x_test)

    if float(y_train.mean()) == 0.0:
        poisson_prediction = np.zeros(len(y_test))
        poisson_next = 0.0
    else:
        poisson_model = PoissonRegressor(
            alpha=1.0,
            max_iter=2000,
            tol=1e-7,
        )

        poisson_model.fit(
            z_train,
            y_train,
        )

        poisson_prediction = poisson_model.predict(
            z_test
        )

        # Finalna count regresija koristi sva poznata kola.
        final_scaler = StandardScaler()

        final_z = final_scaler.fit_transform(
            data.x[:-1]
        )

        final_poisson = PoissonRegressor(
            alpha=1.0,
            max_iter=2000,
            tol=1e-7,
        )

        final_poisson.fit(
            final_z,
            data.y_count,
        )

        poisson_next = float(
            final_poisson.predict(
                final_scaler.transform(
                    data.x[-1:]
                )
            )[0]
        )

    design_train = np.column_stack(
        (np.ones(len(z_train)), z_train)
    )

    design_test = np.column_stack(
        (np.ones(len(z_test)), z_test)
    )

    coefficients = np.linalg.lstsq(
        design_train,
        y_train,
        rcond=None,
    )[0]

    ols_prediction = design_test @ coefficients

    poisson_mae = float(
        np.mean(
            np.abs(poisson_prediction - y_test)
        )
    )

    ols_mae = float(
        np.mean(
            np.abs(ols_prediction - y_test)
        )
    )

    print("\nREGRESIJA BROJA PONAVLJANJA")

    print(
        f"Poisson MAE: {poisson_mae:.6f}"
    )

    print(
        f"OLS MAE: {ols_mae:.6f}"
    )

    print(
        "Poisson prognoza broja ponavljanja za NEXT: "
        f"{poisson_next:.6f}"
    )

    print(
        "Cilj je ogranicen na 0..7; Poisson je ovde "
        "uporedni regresioni model, ne tacan zakon igre."
    )

    save_csv(
        outdir / "05_poisson_ols_test.csv",
        [
            "red",
            "stvarni_broj_ponavljanja",
            "poisson",
            "ols",
        ],
        [
            [
                end + i + 1,
                y_test[i],
                poisson_prediction[i],
                ols_prediction[i],
            ]
            for i in range(len(y_test))
        ],
    )


def longest_run(row):
    best = current = 1

    for a, b in zip(row, row[1:]):
        current = (
            current + 1
            if b == a + 1
            else 1
        )

        best = max(best, current)

    return best


def export_observations(data, outdir):
    save_csv(
        outdir / "00_stvarni_podaci_i_opisi.csv",
        [
            "red",
            "brojevi",
            "zbir",
            "neparnih",
            "raspon",
            "prosecna_parna_udaljenost",
            "ponovljenih_iz_prethodnog",
            "dogadjaj_ponavljanje_najmanje_2",
            "cekano_kola",
            "rang_kombinacije",
        ],
        [
            [
                i + 1,
                " ".join(map(str, data.rows[i])),
                data.sums[i],
                data.odd[i],
                data.spans[i],
                data.mean_distance[i],
                data.overlap[i],
                data.event[i],
                data.age[i],
                data.ranks[i],
            ]
            for i in range(data.n)
        ],
    )


def run(path, outdir):
    rows = read_csv(path)
    data = Dataset(rows)

    fingerprint = hashlib.sha256(
        rows.astype("<i4").tobytes()
    ).hexdigest()[:16]

    print(
        f"\n{'=' * 70}\n"
        f"{Path(path).name}\n"
        f"Kola: {data.n}; "
        f"SHA256 podataka: {fingerprint}",
        flush=True,
    )

    test_start = data.n - TEST
    validation_start = test_start - VALIDATION

    export_observations(data, outdir)

    distribution_analysis(
        data,
        validation_start,
        outdir,
    )

    beta_update(data, outdir)

    count_regression(data, outdir)

    validation = evaluate(
        data,
        validation_start,
        test_start,
        CONFIGS,
        "Validacija NEXT",
    )

    for config in CONFIGS:
        record = validation[config]

        print(
            f"Validacija {config}: "
            f"pogodaka={record['hits'].mean():.4f}; "
            f"regresioni MSE={record['mse'].mean():.6f}"
        )

    # Prvo broj pogodaka; zatim regresiona greska.
    # Zavrsni test ne utice na izbor.
    selected = max(
        CONFIGS,
        key=lambda config: (
            float(
                validation[config]["hits"].mean()
            ),
            -float(
                validation[config]["mse"].mean()
            ),
            config[1],
        ),
    )

    tested = evaluate(
        data,
        test_start,
        data.n,
        (selected,),
        "TEST NEXT",
    )[selected]

    hits = tested["hits"]
    predictions = tested["predictions"]

    previous = np.asarray([
        len(
            set(map(int, rows[t - 1]))
            & set(map(int, rows[t]))
        )
        for t in range(test_start, data.n)
    ])

    history_set = {
        tuple(map(int, row))
        for row in rows[:test_start]
    }

    historical = 0
    previous_copies = 0

    for t, prediction in zip(
        range(test_start, data.n),
        predictions,
    ):
        historical += prediction in history_set

        previous_copies += (
            prediction
            == tuple(map(int, rows[t - 1]))
        )

        history_set.add(
            tuple(map(int, rows[t]))
        )

    print("\nREZULTAT NEXT REGRESORA")
    print("Izabrano:", selected)

    print(
        f"TEST: {hits.mean():.4f} pogodaka"
    )

    print(
        "Kontrola: prethodno izvuceno kolo: "
        f"{previous.mean():.4f}"
    )

    print(
        "Pogodaka 0..7:",
        np.bincount(hits, minlength=8).tolist(),
    )

    print(
        f"Razlicitih predloga: "
        f"{len(set(predictions))}/{TEST}"
    )

    print(
        f"Istorijskih kombinacija: "
        f"{historical}/{TEST}"
    )

    print(
        f"Kopija poslednjeg kola: "
        f"{previous_copies}/{TEST}"
    )

    print(
        "Predloga sa nizom 5+:",
        sum(
            longest_run(p) >= 5
            for p in predictions
        ),
    )

    print(
        "Predloga potpuno u jednoj polovini:",
        sum(
            p[-1] <= 19 or p[0] >= 20
            for p in predictions
        ),
    )

    save_csv(
        outdir / "06_next_regresor_test.csv",
        [
            "red",
            "stvarna_kombinacija",
            "predikcija",
            "pogodaka",
            "regresioni_mse",
        ],
        [
            [
                test_start + i + 1,
                " ".join(
                    map(
                        str,
                        rows[test_start + i],
                    )
                ),
                " ".join(map(str, predictions[i])),
                int(hits[i]),
                float(tested["mse"][i]),
            ]
            for i in range(TEST)
        ],
    )

    # Finalno ucenje koristi sve poznate prelaze.
    bank = RegressorBank(data, data.n)
    model = bank.get(selected)

    pair_values = model.predict(
        data.x[-1]
    )

    prediction = next_combination(pair_values)

    repeated_prediction = next_combination(
        model.predict(data.x[-1])
    )

    if prediction != repeated_prediction:
        raise RuntimeError(
            "Predikcija nije ponovljiva."
        )

    print(
        f"\nNEXT red {data.n + 1}: "
        + ",".join(map(str, prediction))
    )

    print(
        "Prelaza koriscenih za finalno ucenje:",
        bank.count,
    )

    print(
        "Postoji u istoriji:",
        prediction in history_set,
    )

    print(
        f"Najduzi uzastopni niz: "
        f"{longest_run(prediction)}"
    )

    print(
        "Brojeva 1..19:",
        sum(number <= 19 for number in prediction),
    )

    if hits.mean() <= previous.mean():
        print(
            "Na ovom testu regresor nije nadmasio "
            "kontrolu prethodnog kola."
        )

    save_csv(
        outdir / "07_next.csv",
        [
            "csv",
            "kola",
            "regresor",
            "regularizacija",
            "next",
            "test_srednji_broj_pogodaka",
            "kontrola_prethodno_kolo",
        ],
        [[
            str(path),
            data.n,
            selected[0],
            selected[1],
            " ".join(map(str, prediction)),
            float(hits.mean()),
            float(previous.mean()),
        ]],
    )

    return prediction


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Devet analiza raspodela, regresori "
            "i jedna NEXT kombinacija po CSV-u."
        )
    )

    parser.add_argument(
        "csv",
        nargs="*",
        help="CSV bez zaglavlja: 7 brojeva po redu.",
    )

    parser.add_argument(
        "--out",
        default="Distributions_Loto_V1_reports",
        help="Osnovni direktorijum za nove izvestaje.",
    )

    args = parser.parse_args()

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    root = Path(args.out) / stamp

    root.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        "Distributions Loto V1 — "
        "stvarni CSV podaci i deterministicki regresori."
    )

    print(
        "Normalna/uniformna referenca i Beta analiza "
        "ne odredjuju NEXT kombinaciju."
    )

    results = []

    for index, path in enumerate(
        args.csv or DEFAULTS,
        start=1,
    ):
        outdir = root / (
            f"{index:02d}_{Path(path).stem}"
        )

        outdir.mkdir(exist_ok=False)

        prediction = run(path, outdir)
        results.append((path, prediction))

    for i, (path, prediction) in enumerate(results):
        for other_path, other in results[:i]:
            if prediction == other:
                print(
                    f"Isti NEXT: {Path(path).name} / "
                    f"{Path(other_path).name}: "
                    f"{prediction}"
                )

    print(
        "\nIzvestaji sacuvani u:",
        root.resolve(),
    )


if __name__ == "__main__":
    main()



"""
Distributions Loto V1 — stvarni CSV podaci i deterministicki regresori.
Normalna/uniformna referenca i Beta analiza ne odredjuju NEXT kombinaciju.


======================================================================
loto7_4704_k83.csv
Kola: 4704; SHA256 podataka: 195fcf4ffd3d6abb

DEVET ANALIZA RASPODELA
Parametri se podesavaju pre validacionog/test perioda.
D je deskriptivna ECDF distanca, nije p-vrednost.
normal       | obuka=4592 | izdvojeno=112 | D=0.12238
  mu=140.512; sd=27.6712
bernoulli    | obuka=4591 | izdvojeno=112 | D=0.08849
  p=0.356349
binomial     | obuka= 229 | izdvojeno=  6 | D=0.43990
  n=20; p=0.356349
geometric    | obuka=1635 | izdvojeno= 30 | D=0.09556
  p=0.356349
poisson      | obuka=4591 | izdvojeno=112 | D=0.08553
  lambda=1.23481
exponential  | obuka=1635 | izdvojeno= 30 | D=0.30005
  scale=2.80306; kontinuirana aproksimacija
gamma        | obuka=1633 | izdvojeno= 30 | D=0.15289
  shape=5.05068; scale=1.66409; kontinuirana aproksimacija
beta         | obuka=4592 | izdvojeno=112 | D=0.06653
  alpha=16.5828; beta=30.6975; metod momenata
uniform      | obuka=4592 | izdvojeno=112 | D=0.10837
  diskretna referenca 0..15380936; ne koristi se za NEXT
Napomena: intervali su diskretni; eksponencijalna i Gamma ovde su samo kontinuirane aproksimacije.
Binomna/geometrijska referenca pretpostavljaju konstantnu stopu i odgovarajucu nezavisnost.
Poisson nije tacan opis ogranicenog broja 0..7; analiza prikazuje koliko se referenca razlikuje.
Nezavrseno cekanje na kraju CSV-a: 5 kola; nije dodato medju zavrsene intervale.

BETA AZURIRANJE: sredina=0.354273; 95% interval=[0.340666, 0.367998]
Interval vazi u Beta-Bernoulli modelu sa konstantnim p; nije interval verovatnoce dobitka NEXT kombinacije.

REGRESIJA BROJA PONAVLJANJA
Poisson MAE: 0.715799
OLS MAE: 0.719039
Poisson prognoza broja ponavljanja za NEXT: 1.391898
Cilj je ogranicen na 0..7; Poisson je ovde uporedni regresioni model, ne tacan zakon igre.
Validacija NEXT: 1/48; obuka na 4560 prelaza
Validacija NEXT: 17/48; obuka na 4576 prelaza
Validacija NEXT: 33/48; obuka na 4592 prelaza
Validacija ('ridge', 0.1): pogodaka=1.3125; regresioni MSE=0.027831
Validacija ('ridge', 1.0): pogodaka=1.1875; regresioni MSE=0.027655
Validacija ('kernel', 0.1): pogodaka=1.2708; regresioni MSE=0.027667
Validacija ('kernel', 1.0): pogodaka=1.0625; regresioni MSE=0.027608
TEST NEXT: 1/64; obuka na 4608 prelaza
TEST NEXT: 17/64; obuka na 4624 prelaza
TEST NEXT: 33/64; obuka na 4640 prelaza
TEST NEXT: 49/64; obuka na 4656 prelaza

REZULTAT NEXT REGRESORA
Izabrano: ('ridge', 0.1)
TEST: 1.3594 pogodaka
Kontrola: prethodno izvuceno kolo: 1.1406
Pogodaka 0..7: [9, 30, 18, 7, 0, 0, 0, 0]
Razlicitih predloga: 64/64
Istorijskih kombinacija: 0/64
Kopija poslednjeg kola: 0/64
Predloga sa nizom 5+: 0
Predloga potpuno u jednoj polovini: 0

NEXT red 4705: 9,10,12,19,24,25,34
Prelaza koriscenih za finalno ucenje: 4672
Postoji u istoriji: False
Najduzi uzastopni niz: 2
Brojeva 1..19: 4



======================================================================
loto7_4704_k83_loto_2974.csv
Kola: 2974; SHA256 podataka: 291014418cd0db4a

DEVET ANALIZA RASPODELA
Parametri se podesavaju pre validacionog/test perioda.
D je deskriptivna ECDF distanca, nije p-vrednost.
normal       | obuka=2862 | izdvojeno=112 | D=0.05700
  mu=140.742; sd=27.7243
bernoulli    | obuka=2861 | izdvojeno=112 | D=0.00373
  p=0.369801
binomial     | obuka= 143 | izdvojeno=  5 | D=0.32729
  n=20; p=0.369801
geometric    | obuka=1057 | izdvojeno= 41 | D=0.09065
  p=0.369801
poisson      | obuka=2861 | izdvojeno=112 | D=0.08057
  lambda=1.25201
exponential  | obuka=1057 | izdvojeno= 41 | D=0.30951
  scale=2.70009; kontinuirana aproksimacija
gamma        | obuka=1055 | izdvojeno= 41 | D=0.14908
  shape=4.85516; scale=1.66842; kontinuirana aproksimacija
beta         | obuka=2862 | izdvojeno=112 | D=0.09126
  alpha=16.3939; beta=30.4927; metod momenata
uniform      | obuka=2862 | izdvojeno=112 | D=0.06285
  diskretna referenca 0..15380936; ne koristi se za NEXT
Napomena: intervali su diskretni; eksponencijalna i Gamma ovde su samo kontinuirane aproksimacije.
Binomna/geometrijska referenca pretpostavljaju konstantnu stopu i odgovarajucu nezavisnost.
Poisson nije tacan opis ogranicenog broja 0..7; analiza prikazuje koliko se referenca razlikuje.
Nezavrseno cekanje na kraju CSV-a: 2 kola; nije dodato medju zavrsene intervale.

BETA AZURIRANJE: sredina=0.369704; 95% interval=[0.352442, 0.387132]
Interval vazi u Beta-Bernoulli modelu sa konstantnim p; nije interval verovatnoce dobitka NEXT kombinacije.

REGRESIJA BROJA PONAVLJANJA
Poisson MAE: 0.682444
OLS MAE: 0.661218
Poisson prognoza broja ponavljanja za NEXT: 1.192625
Cilj je ogranicen na 0..7; Poisson je ovde uporedni regresioni model, ne tacan zakon igre.
Validacija NEXT: 1/48; obuka na 2830 prelaza
Validacija NEXT: 17/48; obuka na 2846 prelaza
Validacija NEXT: 33/48; obuka na 2862 prelaza
Validacija ('ridge', 0.1): pogodaka=1.1667; regresioni MSE=0.027994
Validacija ('ridge', 1.0): pogodaka=1.1667; regresioni MSE=0.027690
Validacija ('kernel', 0.1): pogodaka=1.2292; regresioni MSE=0.027698
Validacija ('kernel', 1.0): pogodaka=1.3333; regresioni MSE=0.027602
TEST NEXT: 1/64; obuka na 2878 prelaza
TEST NEXT: 17/64; obuka na 2894 prelaza
TEST NEXT: 33/64; obuka na 2910 prelaza
TEST NEXT: 49/64; obuka na 2926 prelaza

REZULTAT NEXT REGRESORA
Izabrano: ('kernel', 1.0)
TEST: 1.1562 pogodaka
Kontrola: prethodno izvuceno kolo: 1.3125
Pogodaka 0..7: [16, 28, 14, 6, 0, 0, 0, 0]
Razlicitih predloga: 64/64
Istorijskih kombinacija: 0/64
Kopija poslednjeg kola: 0/64
Predloga sa nizom 5+: 1
Predloga potpuno u jednoj polovini: 2

NEXT red 2975: 4,11,25,29,31,38,39
Prelaza koriscenih za finalno ucenje: 2942
Postoji u istoriji: False
Najduzi uzastopni niz: 2
Brojeva 1..19: 2
Na ovom testu regresor nije nadmasio kontrolu prethodnog kola.



======================================================================
loto7_4704_k83_loto_plus_1730.csv
Kola: 1730; SHA256 podataka: 86af7b8c14c04be7

DEVET ANALIZA RASPODELA
Parametri se podesavaju pre validacionog/test perioda.
D je deskriptivna ECDF distanca, nije p-vrednost.
normal       | obuka=1618 | izdvojeno=112 | D=0.07996
  mu=140.008; sd=27.5626
bernoulli    | obuka=1617 | izdvojeno=112 | D=0.01824
  p=0.375387
binomial     | obuka=  80 | izdvojeno=  6 | D=0.17807
  n=20; p=0.375387
geometric    | obuka= 606 | izdvojeno= 40 | D=0.03486
  p=0.375387
poisson      | obuka=1617 | izdvojeno=112 | D=0.03948
  lambda=1.26283
exponential  | obuka= 606 | izdvojeno= 40 | D=0.31287
  scale=2.66502; kontinuirana aproksimacija
gamma        | obuka= 604 | izdvojeno= 40 | D=0.18617
  shape=5.45013; scale=1.46755; kontinuirana aproksimacija
beta         | obuka=1618 | izdvojeno=112 | D=0.06566
  alpha=16.6634; beta=30.6192; metod momenata
uniform      | obuka=1618 | izdvojeno=112 | D=0.09772
  diskretna referenca 0..15380936; ne koristi se za NEXT
Napomena: intervali su diskretni; eksponencijalna i Gamma ovde su samo kontinuirane aproksimacije.
Binomna/geometrijska referenca pretpostavljaju konstantnu stopu i odgovarajucu nezavisnost.
Poisson nije tacan opis ogranicenog broja 0..7; analiza prikazuje koliko se referenca razlikuje.
Nezavrseno cekanje na kraju CSV-a: 0 kola; nije dodato medju zavrsene intervale.

BETA AZURIRANJE: sredina=0.374277; 95% interval=[0.351621, 0.397210]
Interval vazi u Beta-Bernoulli modelu sa konstantnim p; nije interval verovatnoce dobitka NEXT kombinacije.

REGRESIJA BROJA PONAVLJANJA
Poisson MAE: 0.807922
OLS MAE: 0.819722
Poisson prognoza broja ponavljanja za NEXT: 1.164783
Cilj je ogranicen na 0..7; Poisson je ovde uporedni regresioni model, ne tacan zakon igre.
Validacija NEXT: 1/48; obuka na 1586 prelaza
Validacija NEXT: 17/48; obuka na 1602 prelaza
Validacija NEXT: 33/48; obuka na 1618 prelaza
Validacija ('ridge', 0.1): pogodaka=1.0417; regresioni MSE=0.028446
Validacija ('ridge', 1.0): pogodaka=1.2500; regresioni MSE=0.027845
Validacija ('kernel', 0.1): pogodaka=1.2083; regresioni MSE=0.027845
Validacija ('kernel', 1.0): pogodaka=1.3958; regresioni MSE=0.027668
TEST NEXT: 1/64; obuka na 1634 prelaza
TEST NEXT: 17/64; obuka na 1650 prelaza
TEST NEXT: 33/64; obuka na 1666 prelaza
TEST NEXT: 49/64; obuka na 1682 prelaza

REZULTAT NEXT REGRESORA
Izabrano: ('kernel', 1.0)
TEST: 1.3750 pogodaka
Kontrola: prethodno izvuceno kolo: 1.2812
Pogodaka 0..7: [10, 28, 20, 4, 2, 0, 0, 0]
Razlicitih predloga: 63/64
Istorijskih kombinacija: 0/64
Kopija poslednjeg kola: 0/64
Predloga sa nizom 5+: 0
Predloga potpuno u jednoj polovini: 3

NEXT red 1731: 6,18,26,30,32,36,39
Prelaza koriscenih za finalno ucenje: 1698
Postoji u istoriji: False
Najduzi uzastopni niz: 1
Brojeva 1..19: 2



Izvestaji sacuvani u: /Users/4c/Desktop/GHQ/Astra/Distributions/Distributions_Loto_V1_reports/20261018_103038_589035
"""



"""
Za NEXT koristim determinističku višeciljnu Ridge i kernel regresiju. 
Kod uključuje svih devet analiza raspodela, grafikone, Beta ažuriranje i poređenje Poisson regresije sa OLS. 
Normalna i uniformna raspodela ne učestvuju u izboru kombinacije.
"""
