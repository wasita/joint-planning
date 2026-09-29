import marimo

__generated_with = "0.24.2"
app = marimo.App()


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
    ## Equation 1 - the step-0 decision rule

    $$P_0\left(s_i^j\right) = \frac{1}{m_i} \quad \forall j$$

    **Terms**

    - $P_0(\cdot)$ -- the step-0 choice-probability function (general form)
    - $P_0(s_i^j)$  -- probability player $i$ chooses $j$-th strategy.
    - $s_i^j$ -- player $i$'s $j$-th strategy.
    - $m_i$ -- the total number of strategies available to player $i$.
    - $\forall j$ -- holds for *every* strategy, which is what makes the
      distribution uniform.

    **What it means for cognitive hierarchy theory**

    Step 0 is the model's placeholder for zero strategic reasoning: a step-0
    player doesn't think about opponents, payoffs, or the structure of the game
    at all -- they just pick uniformly at random.

    Every level built on top of the hierarchy is defined *relative to* this non-strategic baseline,
    which is why Camerer et al. flag it as the single most consequential (and most easily
    mis-specified) modeling choice in the whole framework.
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
    ## Equation 2 - the belief-bound assumption

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
 
    """)
    return


if __name__ == "__main__":
    app.run()
