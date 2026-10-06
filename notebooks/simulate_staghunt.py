import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2-player stag hunt game simulations

    - **GOAL**: simulate choice probabilities of actions (choose stag or choose hare) for each player within the dyad
       - start with 0-level thinkers and work our way up as k-reasoning levels increase.

    - **NEXT**:
        - [x] Simulate a dataset: each player gets a latent level $k$ drawn from Poisson ($\tau$) and choose hare/stag depending on their level's policy
        - [x] Evaluate CH model: score data under the model (compute log likelihoods)
        - [ ] Fit model: find best params that explain the data -- see if we can recover the same parameters that were used to simulate the data. Approaches: MLE, MAP, EM
        - [ ] bonus could be to expand to the case where there are more than 2 people playing
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from scipy.stats import poisson
    from scipy.special import expit, logsumexp
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="ticks", font_scale=0.9)
    return expit, logsumexp, mo, np, pl, plt, poisson, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## the payoff matrix

    row player's payoffs (row = my action, column = opponent's action):

    |              | opp C (stag) | opp D (hare) |
    |--------------|--------------|--------------|
    | **C (stag)** | $a$          | $b$          |
    | **D (hare)** | $c$          | $d$          |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    payoff_sliders = mo.ui.dictionary({
        "a": mo.ui.slider(0, 10, step=0.5, value=4, label="$a$ (C, C)", show_value=True),
        "b": mo.ui.slider(0, 10, step=0.5, value=0, label="$b$ (C, D)", show_value=True),
        "c": mo.ui.slider(0, 10, step=0.5, value=3, label="$c$ (D, C)", show_value=True),
        "d": mo.ui.slider(0, 10, step=0.5, value=2, label="$d$ (D, D)", show_value=True),
    })
    payoff_sliders
    return (payoff_sliders,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    a stag hunt requires $a > c \ge d > b$, where mutual stag (C,C) is the best outcome, but hare is the safe choice (small reward, independent of partner's choice)

    if instead $c > a$, hare strictly dominates stag and the game becomes a
    prisoner's dilemma
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, payoff_sliders):
    ACTIONS = ["C", "D"]  # index 0 = C (stag), 1 = D (hare)

    # (my action, opp action): my payoff
    payoffs = {
        ("C", "C"): payoff_sliders.value["a"],
        ("C", "D"): payoff_sliders.value["b"],
        ("D", "C"): payoff_sliders.value["c"],
        ("D", "D"): payoff_sliders.value["d"],
    }

    # R[my_action, opp_action] = my payoff
    # where action is: C = 0, D = 1
    R = np.array([[payoffs[(me, opp)] for opp in ACTIONS] for me in ACTIONS])

    # threshold belief: stag beats hare iff P(opp stag) > q_star
    (_a, _b), (_c, _d) = R
    _denom = (_a - _c) + (_d - _b)

    # indifference point: the belief P(partner plays stag) at which my stag and my hare have equal expected payoff
    q_star = (_d - _b) / _denom if _denom != 0 else float("nan")

    # ensure that the CC is highest reward
    # followed by when I defect, partner cooperates
    # then both defect and get small reward
    # last: I cooperate and partner defects (I get nothing)
    _is_stag_hunt = _a > _c >= _d > _b

    mo.vstack([
        mo.md(f"$R =$ `{R.tolist()}`, $\\quad q^\\ast = {q_star:.2f}$"),
        mo.callout(
            mo.md("Valid stag hunt ($a > c \\ge d > b$)." if _is_stag_hunt
                  else "**Not a stag hunt**: need $a > c \\ge d > b$."),
            kind="success" if _is_stag_hunt else "warn",
        ),
    ])
    return R, q_star


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Solving for q* (the indifference point)

    - indifference point = the point where the expected payoff is equal for either action

    we first set the $Q$-values to each other:

    $$Q(\text{stag}) = Q(\text{hare})$$

    where each side expands to:

    $$Q(\text{stag}) = a \cdot q + b \cdot (1-q)$$
    $$Q(\text{hare}) = c \cdot q + d \cdot (1-q)$$

    so setting them equal gives:

    $$a \cdot q + b \cdot (1-q) = c \cdot q + d \cdot (1-q)$$

    then we plug in the default reward values from the payoff matrix, to get $q* = 2/3$.
    that is, when our belief of our opponent choosing stag is $2/3 = .667$, we expected the same payoff for either action we take.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q-values as a function of an agent's belief in their partner cooperating

    $$Q(a_i) = \sum_{a_j} R(a_i, a_j)\,\bar\pi_j(a_j)$$

    - with two actions, the opponent model $\bar\pi_j$ reduces to one number,
    $q =$ the believed probability the opponent plays stag. Each action's
    Q-value is then a straight line in $q$, and the lines cross at $q^\ast$, which is the indifference point (see previous section).

    - we treat $q^\ast$ as a threshold: it is optimal (according to expected payoffs) to play hare when $q < q^\ast$ and stag when $q > q^\ast$

    - triangles mark where each CH level's belief $q_k$ lands, taken from the
    ladder in *Climbing the hierarchy* below (so they follow the belief model, parameterized by $\tau$, $K$, and $\beta$).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    q_slider = mo.ui.slider(
        0, 1, step=0.01, value=0.5,
        label="$q$ = believed P(opponent plays stag)", show_value=True,
    )
    q_slider
    return (q_slider,)


@app.cell(hide_code=True)
def _(np, plt):
    def plot_q_lines(R: np.ndarray, q: float, q_star: float, level_qs: dict[int, float]):
        q_grid = np.linspace(0, 1, 101)
        opp = np.stack([q_grid, 1 - q_grid])  # (2, 101): each column is an opponent policy
        Q_grid = R @ opp                      # (2, 101): Q(stag), Q(hare) at every q
        Q_now = R @ np.array([q, 1 - q])      # Eq 5 at the slider value

        STAG, HARE, INK, MUTED = "#2a78d6", "#eb6834", "#333333", "#8a8a85"
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(q_grid, Q_grid[0], color=STAG, lw=2, label="Q(stag)")
        ax.plot(q_grid, Q_grid[1], color=HARE, lw=2, label="Q(hare)")
        if 0 <= q_star <= 1:
            ax.axvline(q_star, color=MUTED, ls=":", lw=1)
            ax.text(q_star, 0.15, f" q* = {q_star:.2f}", color=MUTED, fontsize=9)

        # current slider value
        ax.axvline(q, color=INK, lw=1, alpha=0.5)
        for Qv, col in zip(Q_now, [STAG, HARE]):
            ax.plot(q, Qv, "o", ms=8, color=col, mec="white", mew=2, zorder=3)

        # where each CH level's belief lands; levels with the same belief
        # (common under argmax) share one triangle so labels don't overlap
        shared: dict[float, list[int]] = {}
        for k, qk in level_qs.items():
            shared.setdefault(round(qk, 3), []).append(k)
        for qk, ks in shared.items():
            ax.plot(qk, -0.15, "^", ms=8, color=INK, clip_on=False)
            ax.text(qk, -0.45, "k=" + ",".join(map(str, ks)), ha="center", fontsize=8, color=INK)

        ax.set_xlim(0, 1)
        ax.set_ylim(min(R.min(), 0) - 0.6, R.max() + 0.2)
        ax.set_xlabel("q = believed P(opponent plays stag)")
        ax.set_ylabel("expected payoff Q(a)")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
        ax.set_title(
            f"q = {q:.2f}:  Q(stag) = {Q_now[0]:.2f},  Q(hare) = {Q_now[1]:.2f}",
            fontsize=10, loc="left",
        )
        plt.close(fig)
        return fig

    return (plot_q_lines,)


@app.cell(hide_code=True)
def _(R, ladder, pl, plot_q_lines, q_slider, q_star):
    # level 0 has no belief (it's uniform by fiat), so skip its null q
    _qs = ladder.filter(pl.col("q").is_not_null())
    plot_q_lines(
        R,
        q_slider.value,
        q_star,
        level_qs=dict(zip(_qs["k"], _qs["q"])),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Softmax: turning a value gap into a choice probability

    $$\pi(\text{stag}) = \frac{e^{\beta U(\text{stag})}}{e^{\beta U(\text{stag})} + e^{\beta U(\text{hare})}} = \frac{1}{1 + e^{-\beta\,(U(\text{stag}) - U(\text{hare}))}}$$

    - with two actions, only the **gap** between the two values matters, scaled by $\beta$ (only relative value diff matters, not magnitudes)
    - $\beta = 0$ ignores the values (uniform, like level 0)
    - $\beta \to \infty$ is argmax (original CH). faint curves are reference $\beta$ values
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    softmax_sliders = mo.ui.dictionary(
        {
            "beta": mo.ui.slider(
                0,
                10,
                step=0.1,
                value=1.0,
                label=r"$\beta$ (inverse temperature)",
                show_value=True,
            ),
            "gap": mo.ui.slider(
                -3,
                3,
                step=0.05,
                value=-0.5,
                label=r"gap $U(\text{stag}) - U(\text{hare})$",
                show_value=True,
            ),
        }
    )
    softmax_sliders
    return (softmax_sliders,)


@app.cell(hide_code=True)
def _(np, plt, sns):
    def plot_softmax_stag(
        beta: float, gap: float, ref_betas: tuple[float, ...] = (0.5, 1, 2, 5)
    ):
        gap_grid = np.linspace(-3, 3, 241)
        p_stag = lambda g, b: (
            1 / (1 + np.exp(-b * g))
        )  # 2-action softmax = sigmoid of the gap

        STAG, INK, MUTED = "#f7aef8", "#333333", "#8a8a85"
        fig, ax = plt.subplots(figsize=(6.5, 3.8))

        # faint reference curves at fixed betas, for comparison
        for b in ref_betas:
            ax.plot(
                gap_grid, p_stag(gap_grid, b), color=MUTED, lw=1, alpha=0.35
            )
            ax.text(
                0.6,
                p_stag(0.6, b),
                f" β={b:g}",
                color=MUTED,
                fontsize=7,
                va="center",
            )

        # current beta
        sns.lineplot(
            x=gap_grid,
            y=p_stag(gap_grid, beta),
            color=STAG,
            linewidth=3.5,
            label=f"β = {beta:g}",
            ax=ax,
        )

        # current gap
        p_now = p_stag(gap, beta)
        ax.axvline(gap, color=INK, lw=1, alpha=0.5)
        ax.axhline(0.5, color=MUTED, ls=":", lw=1)
        ax.plot(gap, p_now, "o", ms=9, color=STAG, mec=INK, mew=1, zorder=3)

        ax.set(
            xlim=(-3, 3),
            ylim=(-0.02, 1.02),
            xlabel="value gap  U(stag) − U(hare)",
            ylabel="P(stag)",
        )
        ax.set_title(
            f"β = {beta:g}, gap = {gap:+.2f}:  P(stag) = {p_now:.3f}",
            fontsize=10,
            loc="left",
        )
        ax.legend(frameon=False, loc="upper left")
        sns.despine(ax=ax)
        fig.tight_layout()
        plt.close(fig)
        return fig

    return (plot_softmax_stag,)


@app.cell(hide_code=True)
def _(plot_softmax_stag, softmax_sliders):
    plot_softmax_stag(
        softmax_sliders.value["beta"], softmax_sliders.value["gap"]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Climbing the hierarchy

    Each level $k$ is built from the levels below it, bottom-up:

    1. **Beliefs** (Eq 4): $b_k(h) = w(h) / \sum_{l<k} w(l)$, where $w = f$ (Poisson, Eq 3) or $w = 1$ (uniform)
    2. **Opponent model** (Eq 5 braces): $q_k = \sum_{h<k} b_k(h)\, \pi_h(\text{stag})$
    3. **Expected payoffs** (Eq 5): $U_k(a) = \sum_{a_j} R(a, a_j)\, \bar\pi_j(a_j)$
    4. **Choice**: argmax (Eq 6, original CH) or softmax with $\beta$ (PI's version)

    Level $k$'s policy then joins the pool that level $k+1$ believes about.
    """)
    return


@app.cell
def _(mo):
    ch_controls = mo.ui.dictionary(
        {
            "model": mo.ui.dropdown(
                {"Poisson (τ)": "poisson", "uniform (PI)": "uniform"},
                value="Poisson (τ)",
                label="belief model",
            ),
            "tau": mo.ui.slider(
                0.1,
                5,
                step=0.1,
                value=1.5,
                label=r"$\tau$ (Poisson only)",
                show_value=True,
            ),
            "K": mo.ui.slider(
                1, 10, step=1, value=5, label=r"max level $K$", show_value=True
            ),
            "argmax": mo.ui.checkbox(
                value=True, label="argmax (original CH; ignores β)"
            ),
            "beta": mo.ui.slider(
                0,
                10,
                step=0.1,
                value=1.0,
                label=r"$\beta$ (softmax)",
                show_value=True,
            ),
        }
    )
    ch_controls
    return (ch_controls,)


@app.cell
def _(np, poisson):
    def softmax_policy(U: np.ndarray, beta: float) -> np.ndarray:
        z = beta * (U - U.max())  # subtract max for numerical stability
        p = np.exp(z)
        return p / p.sum()

    def argmax_policy(U: np.ndarray) -> np.ndarray:
        best = np.isclose(U, U.max())  # ties split evenly (Eq 6)
        return best / best.sum()

    def level_weights(model: str, tau: float, K: int) -> np.ndarray:
        """Unnormalized weight on each level 0..K, which beliefs truncate + renormalize."""
        if model == "poisson":
            return poisson.pmf(np.arange(K + 1), tau)  # Eq 3
        return np.ones(K + 1)  # uniform: every lower level equally likely

    def ch_levels(
        R: np.ndarray, K: int, weights: np.ndarray, beta: float | None
    ) -> dict[str, list]:
        """Climb the hierarchy from level 0 to K. beta=None means argmax (original CH)."""
        policies = np.zeros((K + 1, 2))
        policies[0] = [0.5, 0.5]  # Eq 1: level 0 is uniform
        out = {
            "k": [0],
            "beliefs": [None],
            "q": [None],
            "U_stag": [None],
            "U_hare": [None],
            "gap": [None],
            "p_stag": [0.5],
        }

        for k in range(1, K + 1):
            b = (
                weights[:k] / weights[:k].sum()
            )  # Eq 4: truncate below k, renormalize
            opp = b @ policies[:k]  # Eq 5 braces: believed opponent policy
            U = R @ opp  # Eq 5: expected payoff of each action
            policies[k] = (
                argmax_policy(U) if beta is None else softmax_policy(U, beta)
            )

            out["k"].append(k)
            out["beliefs"].append([round(float(x), 3) for x in b])
            out["q"].append(float(opp[0]))
            out["U_stag"].append(float(U[0]))
            out["U_hare"].append(float(U[1]))
            out["gap"].append(float(U[0] - U[1]))
            out["p_stag"].append(float(policies[k][0]))

        return out

    return ch_levels, level_weights


@app.cell
def _(R, ch_controls, ch_levels, level_weights, pl):
    _c = ch_controls.value
    ladder = pl.DataFrame(
        ch_levels(
            R,
            K=_c["K"],
            weights=level_weights(_c["model"], _c["tau"], _c["K"]),
            beta=None if _c["argmax"] else _c["beta"],
        )
    ).with_columns(pl.col(pl.Float64).round(3))
    ladder
    return (ladder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulating data and scoring it under the model

    **Generative story.** Each participant $i$ has a latent level $k_i \sim \text{Poisson}(\tau)$ (truncated at $K$), and makes $T$ independent choices from level $k_i$'s softmax policy $\pi_{k_i}(\text{stag})$. Softmax is required here: under argmax, any choice a level wouldn't make has probability 0 and $\log L = -\infty$.

    **Log likelihood.** We don't observe $k_i$, so we marginalize over it. With $s_i$ = number of stag choices out of $T$:

    $$\log L(\tau, \beta) = \sum_i \log \sum_{k=0}^{K} f_\tau(k)\; \pi_k^{\,s_i} (1 - \pi_k)^{\,T - s_i}$$

    **One-shot vs repeated.** With $T = 1$ the inner sum collapses to the population-wide $P(\text{stag}) = \sum_k f_\tau(k)\,\pi_k$: the data pin down *one number*, so any $(\tau, \beta)$ pair that produces the same overall stag rate fits equally well, no matter how many people we collect. With $T > 1$ each person's choices cluster around their own level's policy, which separates the mixture weights ($\tau$) from the policies ($\beta$).

    The middle panel below is the control: one-shot with $N \times T$ people, i.e., the **same number of choices** as the repeated design, so differences from the right panel come from the repeated structure, not from more data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sim_controls = mo.ui.dictionary(
        {
            "tau": mo.ui.slider(
                0.1, 5, step=0.1, value=1.5, label=r"true $\tau$", show_value=True
            ),
            "beta": mo.ui.slider(
                0, 10, step=0.1, value=2.0, label=r"true $\beta$", show_value=True
            ),
            "N": mo.ui.slider(
                10, 500, step=10, value=100, label=r"$N$ participants", show_value=True
            ),
            "T": mo.ui.slider(
                2, 50, step=1, value=20, label=r"$T$ rounds (repeated design)", show_value=True
            ),
            "seed": mo.ui.number(value=0, label="seed"),
        }
    )
    mo.vstack([sim_controls, mo.md("_Uses max level $K$ from the hierarchy controls above._")])
    return (sim_controls,)


@app.cell
def _(expit, level_weights, logsumexp, np):
    def level_policies(
        R: np.ndarray, tau: float, beta: float | np.ndarray, K: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Population weights f_tau(k) over levels 0..K, and each level's P(stag).

        Same recursion as ch_levels' softmax branch, specialized to 2 actions and
        vectorized over beta: an array of betas gives p_stag of shape (*beta.shape, K+1),
        so a whole beta grid is one pass up the hierarchy instead of one per beta.
        """
        w = level_weights("poisson", tau, K)
        w = w / w.sum()  # truncate at K, renormalize
        beta = np.asarray(beta, dtype=float)
        stag_minus_hare = R[0] - R[1]  # payoff advantage of stag vs opp stag, opp hare
        p_stag = np.empty(beta.shape + (K + 1,))
        p_stag[..., 0] = 0.5  # level 0 is uniform
        for k in range(1, K + 1):
            q = p_stag[..., :k] @ (w[:k] / w[:k].sum())  # believed P(opp stag)
            gap = stag_minus_hare[0] * q + stag_minus_hare[1] * (1 - q)  # U(stag) - U(hare)
            p_stag[..., k] = expit(beta * gap)  # 2-action softmax = sigmoid of the gap
        return w, p_stag

    def simulate_choices(
        R: np.ndarray, tau: float, beta: float, K: int, N: int, T: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (levels, choices): levels is (N,), choices is (N, T) with 1 = stag."""
        w, p_stag = level_policies(R, tau, beta, K)
        levels = rng.choice(K + 1, size=N, p=w)
        choices = (rng.random((N, T)) < p_stag[levels, None]).astype(int)
        return levels, choices

    def log_lik_given_policies(
        choices: np.ndarray, w: np.ndarray, p_stag: np.ndarray
    ) -> np.ndarray:
        """Marginal log likelihood, summing over each participant's unknown level.

        p_stag can carry leading batch dims (e.g. a beta grid); the result has those dims.
        """
        p = np.clip(p_stag, 1e-12, 1 - 1e-12)  # large beta can push p to exactly 0 or 1
        T = choices.shape[1]
        # stag count is a sufficient statistic per participant, so participants with the
        # same count contribute identically: score each count 0..T once, weight by how many
        n_with_s = np.bincount(choices.sum(axis=1), minlength=T + 1)
        s = np.arange(T + 1).reshape((-1,) + (1,) * p.ndim)  # broadcast against p
        # (T+1, ..., K+1): log P(T choices with s stags | level k)
        log_p_given_k = s * np.log(p) + (T - s) * np.log(1 - p)
        log_p_s = logsumexp(log_p_given_k + np.log(w), axis=-1)  # (T+1, ...)
        return np.tensordot(n_with_s, log_p_s, axes=1)

    def log_lik(
        choices: np.ndarray, R: np.ndarray, tau: float, beta: float, K: int
    ) -> float:
        """Convenience wrapper: log likelihood of the data at a single (tau, beta)."""
        w, p_stag = level_policies(R, tau, beta, K)
        return float(log_lik_given_policies(choices, w, p_stag))

    return level_policies, log_lik_given_policies, simulate_choices


@app.cell
def _(R, ch_controls, mo, np, pl, sim_controls, simulate_choices):
    _s = sim_controls.value
    sim_K = ch_controls.value["K"]
    _args = dict(R=R, tau=_s["tau"], beta=_s["beta"], K=sim_K)

    # repeated design; its first round doubles as the one-shot design with the same people
    sim_levels, choices_repeated = simulate_choices(
        **_args, N=_s["N"], T=_s["T"], rng=np.random.default_rng(_s["seed"])
    )
    choices_oneshot = choices_repeated[:, :1]
    # one-shot with N*T people: same number of choices as the repeated design
    _, choices_oneshot_big = simulate_choices(
        **_args, N=_s["N"] * _s["T"], T=1, rng=np.random.default_rng(_s["seed"] + 1)
    )

    _by_level = (
        pl.DataFrame({"level": sim_levels, "p_stag_obs": choices_repeated.mean(axis=1)})
        .group_by("level")
        .agg(pl.len().alias("n_participants"), pl.col("p_stag_obs").mean().round(3))
        .sort("level")
    )
    mo.vstack([mo.md("**Simulated repeated-design data, by true level**"), _by_level])
    return choices_oneshot, choices_oneshot_big, choices_repeated, sim_K


@app.cell
def _(
    R,
    choices_oneshot,
    choices_oneshot_big,
    choices_repeated,
    level_policies,
    log_lik_given_policies,
    np,
    sim_K,
):
    # log likelihood over a (tau, beta) grid for each design
    tau_grid = np.linspace(0.1, 5, 50)
    beta_grid = np.linspace(0, 10, 51)
    designs = {
        f"one-shot, N = {choices_oneshot.shape[0]}": choices_oneshot,
        f"one-shot, N = {choices_oneshot_big.shape[0]}": choices_oneshot_big,
        f"repeated, N = {choices_repeated.shape[0]} × T = {choices_repeated.shape[1]}": choices_repeated,
    }
    ll_surfaces = {name: np.empty((len(beta_grid), len(tau_grid))) for name in designs}
    for i_t, t in enumerate(tau_grid):
        # one pass up the hierarchy covers the whole beta grid, shared by all designs
        _w, _p = level_policies(R, t, beta_grid, sim_K)
        for name, ch in designs.items():
            ll_surfaces[name][:, i_t] = log_lik_given_policies(ch, _w, _p)
    return beta_grid, ll_surfaces, tau_grid


@app.cell(hide_code=True)
def _(beta_grid, ll_surfaces, np, plt, sim_controls, tau_grid):
    def plot_ll_surfaces(
        surfaces: dict[str, np.ndarray], tau_grid: np.ndarray, beta_grid: np.ndarray,
        true_tau: float, true_beta: float, floor: float = -20,
    ):
        INK, MUTED = "#333333", "#8a8a85"
        fig, axes = plt.subplots(1, len(surfaces), figsize=(11, 3.8), sharey=True)
        for ax, (name, ll) in zip(axes, surfaces.items()):
            d_ll = np.maximum(ll - ll.max(), floor)  # ΔLL from this design's best grid point
            mesh = ax.pcolormesh(
                tau_grid, beta_grid, d_ll, cmap="Blues", vmin=floor, vmax=0, shading="auto"
            )
            # ΔLL = -3 ≈ 95% joint confidence region for 2 params (χ²₂ / 2)
            ax.contour(tau_grid, beta_grid, d_ll, levels=[-3], colors=INK, linewidths=1.5, linestyles="solid")
            i_b, i_t = np.unravel_index(ll.argmax(), ll.shape)
            ax.plot(true_tau, true_beta, "*", ms=14, color="white", mec=INK, mew=1.2,
                    label="true")
            ax.plot(tau_grid[i_t], beta_grid[i_b], "X", ms=9, color=INK, mec="white",
                    mew=1, label="grid max")
            ax.set_title(name, fontsize=10, loc="left")
            ax.set_xlabel(r"$\tau$")
        axes[0].set_ylabel(r"$\beta$")
        axes[0].legend(frameon=False, loc="upper right", fontsize=8, labelcolor=MUTED)
        cbar = fig.colorbar(mesh, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("ΔLL from max (black contour = −3)")
        plt.close(fig)
        return fig

    plot_ll_surfaces(
        ll_surfaces, tau_grid, beta_grid,
        sim_controls.value["tau"], sim_controls.value["beta"],
    )
    return


if __name__ == "__main__":
    app.run()
