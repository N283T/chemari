import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from molwidgets import ECFPStepper

    return ECFPStepper, mo


@app.cell
def _(ECFPStepper, mo):
    w = mo.ui.anywidget(ECFPStepper())
    w
    return


@app.cell
def _(ECFPStepper, mo):
    big = mo.ui.anywidget(ECFPStepper("CC(C)(C)NS(=O)(=O)C1(CNc2c(Br)cncc2C#N)CCC1", guided=False))
    big
    return


if __name__ == "__main__":
    app.run()
