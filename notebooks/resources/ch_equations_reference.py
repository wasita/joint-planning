import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", sql_output="polars")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def title_md(mo):
    mo.md(r"""
    # Poisson Cognitive Hierarchy: Equation Reference

    A term-by-term reference for every equation from Camerer, Ho & Chong (2004),
    *A Cognitive Hierarchy Model of Games*, including what each symbol means, how it fits the broader CH story, and how
    it plays out in the two running examples: the *p*-beauty contest and the
    stag hunt.
    """)
    return


@app.cell(hide_code=True)
def eq1_step0(mo):
    mo.md(r"""
    ## Equation 1 -- the step-0 decision rule

    $$P_0\left(s_i^j\right) = \frac{1}{m_i} \quad \forall j$$

    **Terms**

    - $P_0(\cdot)$ -- the step-0 choice-probability function.
    - $s_i^j$ -- player $i$'s $j$-th strategy.
    - $m_i$ -- the total number of strategies available to player $i$.
    - $\forall j$ -- holds for *every* strategy, which is what makes the
      distribution uniform.

    **What it means for cognitive hierarchy theory**

    Step 0 is the model's placeholder for zero strategic reasoning: a step-0
    player doesn't think about opponents, payoffs, or the structure of the game
    at all -- they just pick uniformly at random. Every level built on top of the
    hierarchy is defined *relative to* this non-strategic baseline, which is why
    Camerer et al. flag it as the single most consequential (and most easily
    mis-specified) modeling choice in the whole framework.

    **Beauty contest:** the step-0 guess is uniform over $[0, 100]$, with mean
    $50$ -- this seeds $s_0 = 50$ in the guess recursion (Equation 7).

    **Stag hunt:** a step-0 player picks Stag or Hare with probability $1/2$
    each. This $1/2$ is the only input a step-1 player has, so it single-handedly
    determines which side of the Stag/Hare threshold step 1 lands on -- and, as
    the stag hunt section at the end shows, every higher level follows step 1.
    """)
    return


@app.cell(hide_code=True)
def eq1_step0_rl(mo):
    mo.md(r"""
    ### RL rewrite of Equation 1

    $$\pi_0(a_i) = \frac{1}{|\mathcal{A}_i|} \quad \forall a_i \in \mathcal{A}_i$$

    Uniform random policy -- an untrained, no-lookahead policy with no value
    estimation behind it at all. Direct RL analogue of the step-0 decision
    rule: same math, agent-policy framing instead of player-strategy framing.
    """)
    return


@app.cell(hide_code=True)
def eq2_belief_bound(mo):
    mo.md(r"""
    ## Equation 2 -- the belief-bound assumption

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

    **Beauty contest:** a step-$k$ guesser never imagines an opponent doing $k$
    or more steps, so the guess recursion (Equation 7) only ever reaches down,
    never sideways or up.

    **Stag hunt:** the same bound is what keeps the matrix-game solver a clean
    recursion (step 0, then step 1, then step 2, ...) instead of requiring an
    equilibrium-style fixed-point search.
    """)
    return


@app.cell(hide_code=True)
def eq2_belief_bound_rl(mo):
    mo.md(r"""
    ### RL rewrite of Equation 2

    $$b_k(h) = 0,\ \forall h \ge k+1 \qquad \text{and} \qquad b_k(k) = 0$$

    $b_k$ -- an agent's belief distribution over which recursion-depth "tier"
    the opponent's policy belongs to. Truncating at $h < k$ is bounded
    opponent modeling: it prevents infinite recursive self-simulation ("I
    model them modeling me modeling them...").
    """)
    return


@app.cell(hide_code=True)
def eq3_poisson(mo):
    mo.md(r"""
    ## Equation 3 -- the Poisson population distribution

    $$f(k) = \frac{e^{-\tau}\tau^k}{k!}$$

    **Terms**

    - $f(k)$ -- the *true* share of the population doing exactly $k$ steps of
      thinking.
    - $\tau$ -- the model's single free parameter: simultaneously the mean and
      the variance of the distribution.
    - $\tau^k / k!$ -- together these set the *shape*. The key property is the
      ratio between neighbouring levels:
      $$\frac{f(k)}{f(k-1)} = \frac{\tau}{k}$$
      so each additional step of thinking is *relatively* rarer than the last
      (going from step 3 to step 4 loses more of the population than going from
      step 1 to step 2). Camerer et al. motivate this as a working-memory
      constraint on how many nested steps people can carry out. $k!$ itself has
      no separate psychological meaning -- it's part of the Poisson form.
    - $e^{-\tau}$ -- the normalizing constant (makes $\sum_k f(k) = 1$). Plugging
      in $k=0$ gives $f(0) = e^{-\tau}$ directly -- the model's predicted share
      of purely non-strategic (step-0) players.

    At $\tau = 1.5$: $f(0..5) \approx .223,\ .335,\ .251,\ .126,\ .047,\ .014$.

    **What it means for cognitive hierarchy theory**

    This is the generative model of the whole population: a single number,
    $\tau$, stands in for how strategically sophisticated a population is on
    average. Because $\tau$ is both the mean and the variance, a population with
    a higher average reasoning depth is also predicted to be more spread out in
    depth, not just uniformly shifted upward.

    **Beauty contest:** fitted values of $\tau$ are typically between 1 and 2,
    and Camerer et al. suggest $\tau \approx 1.5$ as a reasonable default. With
    most of the population at steps 0-2, the predicted average guess stays in
    the 20-35 range instead of going to the Nash prediction of 0 (worked out
    level by level under Equation 7).

    **Stag hunt:** $\tau$ sets the belief weights ($g_k(h)$, Equation 4), but
    with argmax responses it does *not* change which action the strategic
    players choose -- that is set by step 1 (see the stag hunt section). What
    $\tau$ does control is the share of step-0 noise, $f(0) = e^{-\tau}$.
    """)
    return


@app.cell(hide_code=True)
def eq3_poisson_rl(mo):
    mo.md(r"""
    ### RL rewrite of Equation 3

    $$p(k) = \frac{e^{-\tau}\tau^k}{k!}$$

    Distribution over how many rounds of *iterated best response* an agent in
    the population carries out -- the depth of "I think that you think that I
    think...". $\tau$ is the population's average depth of strategic recursion.

    Careful: this is **not** a planning horizon or rollout depth. The game is
    one-shot, so there is no lookahead through time at all. In a sequential
    or repeated game, strategic depth ($k$) and temporal horizon ($H$) are
    two separate parameters.
    """)
    return


@app.cell(hide_code=True)
def eq4_truncated_belief(mo):
    mo.md(r"""
    ## Equation 4 -- the truncated, renormalized belief

    $$g_k(h) = \frac{f(h)}{\sum_{l=0}^{k-1} f(l)}, \qquad h < k$$

    **Terms**

    - $f(h)$ -- the true population frequency at level $h$.
    - $\sum_{l=0}^{k-1} f(l)$ -- the total true mass sitting at every level below
      $k$; this renormalizes the truncated slice back into a valid probability
      distribution.

    **What it means for cognitive hierarchy theory**

    This is the piece that makes the model specifically *Poisson-CH* rather than
    generic "cognitive hierarchy": a step-$k$ player's belief isn't a made-up
    shape, it's the actual truth, just chopped off above $k-1$ and rescaled. As
    $k$ grows, that truncated slice converges toward the true $f$ -- the paper's
    "increasingly rational expectations" property. This is the direct opposite
    of the alternative ("level-$k$") models, which instead concentrate all
    belief on the single level directly below ($g_k(k-1) = 1$), producing
    *increasingly* irrational expectations as $k$ grows.

    **Beauty contest:** supplies the weights $g_k(h)$ used on each lower-level
    guess $s_h$ inside the recursion in Equation 7.

    **Stag hunt:** supplies the weights used to mix step-0 through step-$(k-1)$
    strategies into the belief a step-$k$ player best-responds to, when deciding
    between Stag and Hare.
    """)
    return


@app.cell(hide_code=True)
def eq4_truncated_belief_rl(mo):
    mo.md(r"""
    ### RL rewrite of Equation 4

    $$b_k(h) = \frac{p(h)}{\sum_{l=0}^{k-1} p(l)}, \qquad h < k$$

    A fixed prior over the opponent's policy tier: the population
    distribution with tiers $\ge k$ cut off and the remainder renormalized.
    Nothing is updated from data -- in a one-shot game there is no evidence to
    update on. (A genuine Bayesian update of $b_k$ from the opponent's observed
    actions is the natural extension for repeated play.) These weights build
    the opponent-model mixture in Equation 5's RL rewrite below.

    At $\tau = 1.5$: $b_1 = [1]$, $b_2 = [.40, .60]$, $b_3 = [.28, .41, .31]$.
    """)
    return


@app.cell(hide_code=True)
def eq5_expected_payoff(mo):
    mo.md(r"""
    ## Equation 5 -- expected payoff

    $$E_k\left(\pi_i\left(s_i^j\right)\right) = \sum_{j'=1}^{m_{-i}} \pi_i\left(s_i^j, s_{-i}^{j'}\right)\left\{\sum_{h=0}^{k-1} g_k(h)\, P_h\left(s_{-i}^{j'}\right)\right\}$$

    **Terms**

    - $E_k(\cdot)$ -- the expected-payoff operator for a step-$k$ thinker.
    - $\pi_i\left(s_i^j, s_{-i}^{j'}\right)$ -- player $i$'s payoff from playing
      strategy $j$ when the opponent plays strategy $j'$.
    - $m_{-i}$ -- the number of strategies available to the opponent.
    - $P_h\left(s_{-i}^{j'}\right)$ -- the probability that a step-$h$ opponent
      plays strategy $j'$.
    - $g_k(h)$ -- the belief weight (Equation 4) placed on opponents being at
      level $h$.

    **What it means for cognitive hierarchy theory**

    This is general machinery, shared by every model in the CH family regardless
    of what specific $g_k$ or $f(k)$ you plug in -- it's just "average payoff
    across every opponent strategy, weighted by how likely each opponent type is
    to play it, weighted by how likely each opponent type is to exist."

    **Beauty contest:** because the payoff depends only on how close your guess
    is to $p \times$ the average, the only thing about the opponents' strategies
    that matters is their believed *mean* guess -- see the point-value form in
    Equation 7.

    **Stag hunt (2 players):** there is an equally simple shortcut. The
    opponent has only two actions, so the belief-weighted mixture in braces
    reduces to one number,
    $$q_k = \sum_{h=0}^{k-1} g_k(h)\, P_h(\text{Stag}),$$
    the believed probability the opponent plays Stag. Writing the row player's
    payoffs as $a = \pi(S,S)$, $b = \pi(S,H)$, $c = \pi(H,S)$, $d = \pi(H,H)$
    (a stag hunt needs $a > c \ge d > b$):
    $$E_k(S) = q_k a + (1-q_k) b, \qquad E_k(H) = q_k c + (1-q_k) d.$$
    Both are linear in $q_k$, so Stag beats Hare exactly when $q_k$ exceeds a
    threshold:
    $$q_k > q^\ast = \frac{d-b}{(a-c)+(d-b)}.$$

    **With more than two players**, the sum over $j'$ runs over *profiles* of
    all opponents' strategies, and each opponent's level is treated as an
    independent draw from $g_k$.
    """)
    return


@app.cell(hide_code=True)
def eq5_expected_payoff_rl(mo):
    mo.md(r"""
    ### RL rewrite of Equation 5

    $$Q_k(a_i) = \sum_{a_{-i}} R_i(a_i, a_{-i})\, \bar\pi_{-i}^{(k)}(a_{-i}), \qquad \bar\pi_{-i}^{(k)}(a_{-i}) = \sum_{h=0}^{k-1} b_k(h)\, \pi_h(a_{-i})$$

    $\bar\pi_{-i}^{(k)}$ -- a belief-marginalized opponent policy: a mixture
    of lower-tier opponent policies weighted by belief. $Q_k(a_i)$ is the
    expected return of action $a_i$ against that mixture -- the standard
    opponent-modeling Q-value used in multi-agent RL. ($R_i$ replaces econ's
    $\pi_i$ here to avoid clashing with RL's own use of $\pi$ for policy.)

    Note that $Q_k$ here is just the expected *immediate* reward of a stateless,
    one-shot game: there are no states, transitions, discounting, or
    bootstrapping.
    """)
    return


@app.cell(hide_code=True)
def eq6_best_response(mo):
    mo.md(r"""
    ## Equation 6 -- the best-response rule

    $$P_k\left(s_i^{\ast}\right) = 1 \quad \text{iff} \quad s_i^{\ast} = \operatorname*{arg\,max}_{s_i^j} E_k\left(\pi_i\left(s_i^j\right)\right)$$

    **Terms**

    - $P_k(\cdot)$ -- the step-$k$ decision rule.
    - $s_i^{\ast}$ -- the strategy chosen with certainty.
    - $\operatorname*{arg\,max}_{s_i^j}$ -- whichever strategy maximizes expected
      payoff (Equation 5); ties are split evenly across the tied maximizers.

    **What it means for cognitive hierarchy theory**

    This closes the loop of the whole model: bounded belief (Equations 2 and 4)
    feeds into an expected-payoff calculation (Equation 5), and the step-$k$
    player simply best-responds to it. Nothing here is boundedly rational in
    itself: for every step $k \ge 1$, the bounded-rationality assumption lies in
    what a player believes about others, not in how they optimize given that
    belief. (The other departure from full rationality is step 0 itself, which
    doesn't optimize at all.)

    **Beauty contest:** picks the single number minimizing distance to $p \times$
    the believed average guess.

    **Stag hunt:** picks Stag iff $q_k > q^\ast$, Hare iff $q_k < q^\ast$, and
    splits evenly iff $q_k = q^\ast$ (Equation 5's threshold form).
    """)
    return


@app.cell(hide_code=True)
def eq6_best_response_rl(mo):
    mo.md(r"""
    ### RL rewrite of Equation 6

    $$\pi_k(a_i) = \mathbb{1}\left[a_i = \operatorname*{arg\,max}_{a_i'} Q_k(a_i')\right]$$

    A **best response**: act greedily with respect to $Q_k$ (ties split
    uniformly). This is not the "improve" step of policy iteration. There,
    $Q$ is the value of the agent's *own* current policy; here, $Q_k$ depends
    only on the fixed opponent mixture.

    The closest RL analogue for the *whole hierarchy* is iterated best response
    to a growing population of earlier policies: $\pi_1$ best-responds to
    $\pi_0$, $\pi_2$ best-responds to a mixture of $\{\pi_0, \pi_1\}$, and so on,
    with Poisson mixture weights. This is structurally like fictitious play or
    PSRO. Level-$k$ models instead best-respond only to the most recent policy.

    A common relaxation replaces the argmax with a softmax,
    $\pi_k(a_i) \propto \exp(\lambda Q_k(a_i))$, giving CH with logit
    ("quantal") responses.
    """)
    return


@app.cell(hide_code=True)
def eq7_beauty_contest_applied(mo):
    mo.md(r"""
    ## Equation 7 -- applied to the beauty contest (derived from Equations 4-6)

    $$s_0 = 50, \qquad s_k = p\sum_{h=0}^{k-1} g_k(h)\, s_h \quad (k \ge 1)$$

    This is the point-value instantiation of Equations 4-6 for a game whose
    payoff depends only on the mean of the guess distribution. It solves purely
    by substitution, bottom-up from $s_0$.

    **Worked levels** ($p = 2/3$, $\tau = 1.5$; weights $g_k$ from Equation 4):

    | $k$ | $g_k(0), g_k(1), \ldots$ | believed average | guess $s_k$ |
    |---|---|---|---|
    | 0 | -- | -- | 50 |
    | 1 | 1 | 50 | 33.3 |
    | 2 | .40, .60 | 40.0 | 26.7 |
    | 3 | .28, .41, .31 | 35.9 | 23.9 |
    | 4 | | | 22.8 |
    | 5 | | | 22.5 |

    The population average, $\sum_k f(k)\, s_k$, is about 33.6.

    **Dominance-solvability:** iterating this recursion converges to the Nash
    prediction of $0$ as $\tau \to \infty$. At $\tau \approx 1.5$ it stops well
    short: individual guesses level off around 22, because even very
    high-level players believe much of the population is at steps 0-2.

    A related result used to prove this: $f(k-1)/f(k-2) = \tau/(k-1)$. For
    $k \ll \tau$, this puts almost all belief-weight on the $k-1$ type directly
    below (i.e., $g_k(k-1) \approx 1-\epsilon$), which is what lets Camerer et
    al. show that step-by-step CH reasoning mimics, level by level, the
    classical *iterated deletion of dominated strategies* -- without ever
    building "delete the dominated strategy" into the rules by hand.

    **This machinery does not apply to the stag hunt.** The stag hunt is *not*
    dominance-solvable -- neither Stag nor Hare is ever weakly dominated, since
    which one pays better flips depending on what the rest of the group does.
    Dominance reasoning alone can't narrow the stag hunt down to a unique
    prediction; the full belief-weighted machinery of Equations 4-6 has to carry
    the entire predictive burden there instead.
    """)
    return


@app.cell(hide_code=True)
def eq7_beauty_contest_applied_rl(mo):
    mo.md(r"""
    ### RL rewrite of Equation 7

    $$a_0 = 50, \qquad a_k = p\sum_{h=0}^{k-1} b_k(h)\, a_h$$

    $s_k$ is an *action* (a guess), not a value, so the recursion is over
    policies: each tier's deterministic action is the best response to the
    belief-weighted mean action of the lower tiers. It is not a value backup --
    no tier's expected reward appears anywhere in the recursion.
    """)
    return


@app.cell(hide_code=True)
def stag_hunt_applied(mo):
    mo.md(r"""
    ## Stag hunt: working through the levels (derived from Equations 3-6)

    No new equation is needed. The stag hunt uses the same population
    distribution (Equation 3), belief rule (Equation 4), expected payoff
    (Equation 5, in its threshold form) and best response (Equation 6). Only
    the payoffs change. This is a worked example built from those equations,
    not a result quoted from the paper.

    **Payoffs (row player):** $a = \pi(S,S)$, $b = \pi(S,H)$, $c = \pi(H,S)$,
    $d = \pi(H,H)$, with $a > c \ge d > b$: mutual Stag is best, but Hare is
    safe. (If instead $c > a$ and $d > b$, Hare strictly dominates Stag and the
    game is a Prisoner's Dilemma, not a stag hunt.) Step $k$ plays Stag iff
    $q_k > q^\ast = (d-b)/\big((a-c)+(d-b)\big)$.

    **Worked levels** ($\tau = 1.5$), for two payoff matrices:

    *Risky Stag* -- $(a,b,c,d) = (4,0,3,2)$, $q^\ast = 2/3$:

    | $k$ | $q_k$ | $E_k(S)$ | $E_k(H)$ | choice |
    |---|---|---|---|---|
    | 0 | -- | -- | -- | 50/50 |
    | 1 | .50 | 2.00 | 2.50 | Hare |
    | 2 | $.4(.5) + .6(0) = .20$ | 0.80 | 2.20 | Hare |
    | 3 | .14 | 0.55 | 2.14 | Hare |

    *Easy Stag* -- $(a,b,c,d) = (4,0,2,1)$, $q^\ast = 1/3$:

    | $k$ | $q_k$ | $E_k(S)$ | $E_k(H)$ | choice |
    |---|---|---|---|---|
    | 0 | -- | -- | -- | 50/50 |
    | 1 | .50 | 2.00 | 1.50 | Stag |
    | 2 | $.4(.5) + .6(1) = .80$ | 3.20 | 1.80 | Stag |
    | 3 | .86 | 3.45 | 1.86 | Stag |

    **Every level $k \ge 1$ copies step 1.** If step 1 plays Stag, then every
    lower level plays Stag with probability $\ge 1/2$, so $q_k \ge 1/2 > q^\ast$
    and step $k$ plays Stag too. If step 1 plays Hare, then $q_k \le 1/2 < q^\ast$
    and step $k$ plays Hare. So the whole prediction is set by one comparison,
    $1/2$ versus $q^\ast$. The population then plays step 1's action with
    probability $1 - f(0)/2 \approx .89$ at $\tau = 1.5$; the remaining
    $f(0)/2$ is step-0 noise. $\tau$ changes how much noise there is, not which
    action the population favours. (With softmax responses instead of argmax,
    this collapse no longer happens exactly.)

    **Group size.** Use the normalized payoffs Hare $= x$ for sure, and Stag $= 1$
    only if *everyone* picks Stag (else 0). With 2 players, $q^\ast = x$, so step 1
    plays Stag iff $x < 1/2$. With 3 players, step 1 faces two independent step-0
    opponents who both pick Stag with probability $1/4$, so it plays Stag iff
    $x < 1/4$ (and splits evenly at $x = 1/4$). The same argument as above shows
    higher levels again copy step 1. Larger groups raise the bar for Stag,
    because coordination requires more people to pick Stag at the same time.
    The effect comes entirely through the **threshold on $x$**. The share
    $1 - f(0)/2$ playing step 1's action is the same at every group size.
    """)
    return


if __name__ == "__main__":
    app.run()
