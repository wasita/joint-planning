import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simulate a 2-player stag hunt game

    - Payoff matrix:
        - C1, C2 = (3, 3)
        - C1, D2 = (0, 5)
        - D1, C2 = (5, 0)
        - D1, D2 = (1, 1)
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
