import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RL Rewrites of Poisson-Cognitive Hierarchy Equations

    - reference sheet for every equation from Camerer, Ho & Chong (2004),
    *A Cognitive Hierarchy Model of Games*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (1) - the step-0 decision rule

    $$P_0\left(s_i^j\right) = \frac{1}{m_i} \quad \forall j$$

    **Terms**

    - $P_0(\cdot)$ -- the step-0 choice-probability function (general form)
    - $P_0(s_i^j)$  -- probability player $i$ chooses $j$-th strategy
    - $s_i^j$ -- player $i$'s $j$-th strategy
    - $m_i$ -- the total number of strategies available to player $i$
    - $\forall j$ -- holds for *every* strategy, which is what makes te
      distribution uniform

    **What it means for cognitive hierarchy theory**

    - step 0 is a player who isn't considering the other players' actions at all; they are effectively choosing actions uniformly at random
    - levels above are defined *relative to* this non-strategic baseline level
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Re-written in RL terms

    $$\pi_0(a_i) = \frac{1}{|\mathcal{A}_i|} \quad \forall a_i \in \mathcal{A}_i$$

    - $\pi_0$ -- step-0 policy. Note: policies are "the mapping between states and actions", but here there is no state; this is because this is a one-shot situation, so there is only one state and we don't need to specify it
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A variation with $\pi_i$ so $i$ can be the agent as well as the level of the agent as superscript: $\pi_i^{(0)}$

    $$\pi_i^{(0)}(a_i) = \frac{1}{|\mathcal{A}_i|} \quad \forall a_i \in \mathcal{A}_i$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (2) the belief-bound assumption on recursion

    $$g_k(h) = 0,\ \forall h \ge k+1 \qquad \text{and} \qquad g_k(k) = 0$$

    **Terms**

    - $g_k(h)$ -- a step-$k$ player's belief about the proportion of the
      population doing exactly $h$ steps of thinking.
    - $k$ -- the believer's own thinking level.
    - $h$ -- the level being believed about.
    - the first clause -- a step-$k$ player assigns zero probability to anyone
      reasoning *more* than they do.
    - the second clause -- a step-$k$ player also assigns zero probability to
      anyone reasoning at *exactly* their own level (overconfidence).

    **What it means for cognitive hierarchy theory**

    Together these two conditions bound the recursion: a step-$k$ player's
    beliefs only ever put weight on levels $0$ through $k-1$. This is what makes
    the model solvable by simple bottom-up substitution rather than by finding a
    fixed point -- each level's problem only ever depends on levels that have
    already been solved. It's also the model's formalization of a bounded
    working-memory constraint: nobody can conceive of a mind more sophisticated
    than (or exactly as sophisticated as) their own.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (3) agent's belief about oopponent's $k$-level

    $$b_i^{(k_i)}(k_j) = \begin{cases} \dfrac{1}{k_i} & \text{if } 0 \le k_j < k_i \\ 0 & \text{otherwise} \end{cases}$$

    - $b_i^{(k)}$ is the belief held by agent $i$, who is a level-$k$ thinker.
    - $k_j$ is the level of the opponent $j$.

    Note: Poisson-CH would entail putting truncated Poisson weights on the lower levels (at $\tau = 1.5$, a step-2 thinker believes [.4, 0.6])
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | believer level | about level 0 | about 1 | about 2 | about 3 |
    |---|---|---|---|---|
    | $b^{(1)}$ | 1 | 0 | | |
    | $b^{(2)}$ | 1/2 | 1/2 | 0 | |
    | $b^{(3)}$ | 1/3 | 1/3 | 1/3 | 0 |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### written out in the CH-Camerer way:

    $$g_k(h) = \frac{f(h)}{\sum_{l=0}^{k-1} f(l)}, \qquad 0 \le h < k$$

    - $g_k(h)$ -- a step-$k$ thinker's *belief* about the proportion of opponents at level $h$
    - $k$ -- the believer's own thinking level
    - $\sum_{l=0}^{k-1} f(l)$ -- the total true proportion at every level below $k$. Dividing by it renormalizes the truncated distribution so the beliefs sum to 1.
    - $l$ -- a dummy index for the sum; it iterates over the levels $0, \ldots, k-1$ the believer can conceive of
    - $0 \le h < k$ -- the truncation: a step-$k$ thinker only puts weight on strictly lower levels (and $g_k(h) = 0$ for $h \ge k$).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RL rewrite:

    $$b_i^{(k_i)}(k_j) = \begin{cases} \dfrac{p(k_j)}{\sum_{l=0}^{k_i - 1} p(l)} & \text{if } 0 \le k_j < k_i \\ 0 & \text{otherwise} \end{cases}$$

    - $b_i^{(k_i)}(k_j)$ -- agent $i$'s belief, as a level-$k_i$ agent, that opponent $j$ is at level $k_j$. The superscript is the believer's level; the subscript is the agent
    - $\sum_{l=0}^{k_i - 1} p(l)$ -- the total population proportion across every level below $k_i$; dividing by it renormalizes the truncated prior.
    - $0 \le k_j < k_i$ -- agent $i$ only models opponents at strictly lower levels, whose policies are already computed and fixed. This keeps the opponent's policy stationary from $i$'s point of view.
    - "otherwise $0$" -- zero belief in opponents at the same or higher levels.
    - the belief is a fixed prior, not a Bayesian posterior: in a one-shot game, there are no observations to update on.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (4) true population distribution of k-level thinkers

    Note: truncated Poisson - that is, we take values of $h$ up to the $k$-level of the agent, then renormalize so that the probabilities sum up to 1.

    $$f(h) = \frac{e^{-\tau}\,\tau^{h}}{h!}$$

    - $f(h)$ -- the *true* proportion of the population that thinks exactly $h$ steps. Defined for every $h = 0, 1, 2, \ldots$, independent of any particular thinker
    - $h$ -- a thinking level (the level whose proportion, or believed proportion, we're asking about)
    - $\tau$ -- the single free parameter: the mean (and variance) of the Poisson distribution, i.e., the population's average number of thinking steps
    - $e^{-\tau}$ -- the normalizing constant that makes $f$ sum to 1. Also equals $f(0)$, the proportion of step-0 players
    - $\tau^{h} / h!$ -- sets the shape. Neighbouring levels satisfy $f(h)/f(h-1) = \tau/h$, so each extra step of thinking is relatively rarer than the last

    ### RL rewrite

    $$p(k_j) = \frac{e^{-\tau}\,\tau^{k_j}}{k_j!}$$

    - $p(k_j)$ -- the population distribution over agents' levels of iterated best response (the same numbers as $f$). Computed once, shared by every agent
    - $i$, $j$ -- agent indices: $i$ is the agent doing the reasoning, $j$ is the opponent
    - $k_i$ -- agent $i$'s own level: how many rounds of best response it performs
    - $k_j$ -- opponent $j$'s level, which agent $i$ doesn't know; this is the variable the belief is *over*
    - $\tau$ -- the population's average depth of strategic recursion. (note: not a planning horizon: the game is one-shot, so there's no lookahead through time.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (5) step-$k$'s expected payoff for each of my strategies

    $$E_k\big(\pi_i(s_i^j)\big) = \sum_{j'=1}^{m_{-i}} \pi_i\big(s_i^j, s_{-i}^{j'}\big) \left\{ \sum_{h=0}^{k-1} g_k(h)\, P_h\big(s_{-i}^{j'}\big) \right\}$$

    ### breakdown:

    **left-hand side**:

    $$E_k\big(\pi_i(s_i^j)\big)$$

    - $s_i^j$: one particular strategy $j$ of player $i$. In the stag hunt, $j$ = Stag or $j$ = Hare
    - $\pi_i(\cdot)$: player $i$'s payoff. In econ notation, $\pi$ is payoff, not policy!
    - $E_k$: the expected value, as computed by a step-k thinker

    **translation**: how much does a step-$k$ thinker _expect_ to earn from playing strategy $j$? compute once for choosing stag, compute once for choosing hare, then compare the expected values

    **right-hand side -- innermost**: belief probability that opponent plays each $j'$

    $$\left\{ \sum_{h=0}^{k-1} g_k(h), P_h\big(s_{-i}^{j'}\big) \right\}$$

    - $s_{-i}^{j'}$: the opponent's strategy j′ ($-i$ means "not i").
    - $P_h(s_{-i}^{j'})$: the probability that a step-h opponent plays j′. This is already known, because lower levels were solved first.
    - $g_k(h)$: my belief that the opponent is step-h
    - The sum: go through every lower level and add up "chance they're this level × chance this level plays j′."

    **right-hand side -- outer sum**

    $$\sum_{j'=1}^{m_{-i}} \pi_i\big(s_i^j, s_{-i}^{j'}\big) \times  \text(innermost)$$

    - $\sum_{j'=1}^{m_{-i}}$: go through each of the opponent's strategies, from $j' = 1$ up to $m_{-i}$, the number of strategies the opponent has. In the stag hunt, $m_{-i} = 2$ (Stag, Hare).
    - $\pi_i(s_i^j, s_{-i}^{j'})$: not a policy! In econ notation, $\pi$ is payoff: what player $i$ earns when $i$ plays $j$ and the opponent plays $j′$. It's one cell of the payoff matrix
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RL rewrite

    - $Q$-action values

    $$Q_i^{(k_i)}(a_i) = \sum_{a_j \in \mathcal{A}_j} R_i(a_i, a_j)\; \bar\pi_j^{(k_i)}(a_j)$$


    - $Q_i^{(k_i)}(a_i)$: expected payoff of my action, as a level-$k_i$ agent
    - $a_j$: my action
    - $a_j$: my opponent's action
    - $\sum_{a_j \in \mathcal{A}_j}$: loop over opponent's actions
    - $R_{i} (a_{i},a_{j})$: reward when I take $a_{i}$ and opponent takes action $a_{j}$. From the payoff matrix


    $$\bar\pi_j^{(k_i)}(a_j) = \sum_{k_j=0}^{k_i-1} b_i^{(k_i)}(k_j)\,\pi_j^{(k_j)}(a_j)$$

    - $\bar\pi_j^{(k_i)}(a_j)$: my model of the opponent's policy
    - $b_i^{(k_i)}(k_j)$: my belief about the opponent's policy
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### visualization

    - to simulate what values $Q$ takes on as a function of what I believe about the opponent
    - x-axis: believed probability that the opponent chooses Stag ($\bar\pi_j$ in 2-action game)
    - y-axis: action's $Q$-value
    """)
    return


@app.cell
def _(np):
    # first, need to define the payoff matrix
    # C=cooperate, D=defect

    payoffs = {("C", "C"): 4, ("C", "D"): 0, ("D", "C"): 3, ("D", "D"): 2}

    A = ["C", "D"]  # index 0=stag, 1=hare
    R = np.array([[payoffs[(me, opp)] for opp in A] for me in A])
    print(R)
    return (R,)


@app.cell
def _(mo):
    q_slider = mo.ui.slider(
        0,
        1,
        step=0.01,
        value=0.5,
        label="q = believed P(opponent plays stag)",
        show_value=True,
    )
    q_slider
    return (q_slider,)


@app.cell(hide_code=True)
def _(R, np, plt, q_slider):
    def plot_q_lines(R: np.ndarray, q: float, level_qs: dict[int, float]):
        q_grid = np.linspace(0, 1, 101)
        opp = np.stack(
            [q_grid, 1 - q_grid]
        )  # (2, 101): each column is an opponent policy
        Q_grid = R @ opp  # (2, 101): Q(stag), Q(hare) at every q
        Q_now = R @ np.array([q, 1 - q])  # Eq 5 at the slider value

        (a, b), (c, d) = R
        q_star = (d - b) / ((a - c) + (d - b))  # where the lines cross

        STAG, HARE, INK, MUTED = "#2a78d6", "#eb6834", "#333333", "#8a8a85"
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(q_grid, Q_grid[0], color=STAG, lw=2, label="Q(stag)")
        ax.plot(q_grid, Q_grid[1], color=HARE, lw=2, label="Q(hare)")
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
            fontsize=10,
            loc="left",
        )
        plt.close(fig)
        return fig

    # q_k values for the risky matrix at tau = 1.5 (we'll compute these properly later)
    plot_q_lines(R, q_slider.value, level_qs={1: 0.50, 2: 0.20, 3: 0.14})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    % (blue) original CH, step 0
    $$P_0(s_i^j) = \frac{1}{m_i} \quad \forall j$$

    % RL rewrite, step 0
    $$\pi_i^{(0)}(a_i) = \frac{1}{|\mathcal{A}_i|} \quad \forall a_i \in \mathcal{A}_i$$

    % belief over the opponent's level (uniform version, as before)
    $$b_i^{(k_i)}(k_j) = \begin{cases} \dfrac{1}{k_i} & \text{if } 0 \le k_j < k_i \\ 0 & \text{otherwise} \end{cases}$$

    % level-1 policy: softmax over utilities
    $$\pi_i^{(1)}(a_i) \propto \exp\big(\beta\, U_i^{(1)}(a_i)\big)$$

    % level-1 utility: expected reward against a level-0 opponent
    $$U_i^{(1)}(a_i) = \sum_{a_j} \pi_j^{(0)}(a_j)\, R_i(a_i, a_j)$$

    % level-k utility: expected reward against the belief-weighted mix of lower levels
    $$U_i^{(k_i)}(a_i) = \sum_{a_j} \sum_{k_j} b_i^{(k_i)}(k_j)\, \pi_j^{(k_j)}(a_j)\, R_i(a_i, a_j)$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (6) softmax choice function, not argmax

    $$\pi_i^{(k)}(a_i) = \frac{\exp\big(\beta\, U_i^{(k)}(a_i)\big)}{\sum_{a'} \exp\big(\beta\, U_i^{(k)}(a')\big)}$$

    - $\beta$: inverse temperature. as it increases, agent chooses more deterministically compared to random noise
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    with 2 actions, softmax simplifies to the value gap between the two actions, multiplied by $\beta$, in the form of a sigmoid:

    $$P(\text{stag}) = \frac{1}{1 + e^{-\beta,(U_{\text{stag}} - U_{\text{hare}})}}$$

    - Since $\beta$ multiplies the value gap between the utilities of the two actions, scaling or changing the magnitudes of the utilities does not change the behavior. What matters is the relative difference between the higher-reward and lower-rewardoption
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
