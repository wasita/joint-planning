import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2-player stag hunt game simulations

    - GOAL: to simulate choice probabilities of actions (choose stag or choose hare) for each player within the dyad
       - start with 0-level thinkers and work our way up as k-reasoning levels increase.
       - bonus could be to expand to the case where there's more than 2 people playing.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from scipy.stats import poisson
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="ticks", font_scale=0.9)
    return mo, np, pl, plt, poisson, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## the payoff matrix

    row player's payoffs (row = my action, column = opponent's action):

    |              | opp C (stag) | opp D (hare) |
    |--------------|--------------|--------------|
    | **C (stag)** | $a$          | $b$          |
    | **D (hare)** | $c$          | $d$          |

    a stag hunt requires $a > c \ge d > b$, where mutual stag is the best outcome, but hare is the safe choice (small reward, independent of partner's choice)

    if instead $c > a$, hare strictly dominates stag and the game becomes a
    prisoner's dilemma
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
    R = np.array([[payoffs[(me, opp)] for opp in ACTIONS] for me in ACTIONS])

    # threshold belief: stag beats hare iff P(opp stag) > q_star
    (_a, _b), (_c, _d) = R
    _denom = (_a - _c) + (_d - _b)
    q_star = (_d - _b) / _denom if _denom != 0 else float("nan")

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
    ## Q-values as a function of belief

    $$Q(a_i) = \sum_{a_j} R(a_i, a_j)\,\bar\pi_j(a_j)$$

    - with two actions, the opponent model $\bar\pi_j$ reduces to one number,
    $q =$ the believed probability the opponent plays stag. Each action's
    Q-value is then a straight line in $q$, and the lines cross at $q^\ast$, which is the indifference point.

    - triangles mark where each CH level's belief $q_k$ lands (e.g., currently
    $\tau = 1.5$; placeholder values until we compute them).
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

        # where each CH level's belief lands
        for k, qk in level_qs.items():
            ax.plot(qk, -0.15, "^", ms=8, color=INK, clip_on=False)
            ax.text(qk, -0.45, f"k={k}", ha="center", fontsize=8, color=INK)

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
def _(R, plot_q_lines, q_slider, q_star):
    plot_q_lines(
        R,
        q_slider.value,
        q_star,
        level_qs={
            1: 0.50, 
            2: 0.20,
            3: 0.14
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Softmax: how $\beta$ turns a value gap into a choice probability

    $$\pi(\text{stag}) = \frac{e^{\beta U(\text{stag})}}{e^{\beta U(\text{stag})} + e^{\beta U(\text{hare})}} = \frac{1}{1 + e^{-\beta\,(U(\text{stag}) - U(\text{hare}))}}$$

    - with two actions, only the **gap** between the two values matters, scaled by $\beta$ (only relative value diff matters, not magnitudes)
    - $\beta = 0$ ignores the values (uniform, like step 0)
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


@app.cell
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
    ## Step 3: climbing the hierarchy

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
    return


if __name__ == "__main__":
    app.run()
