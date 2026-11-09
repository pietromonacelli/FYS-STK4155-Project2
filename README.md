# FYS-STK4155-Project2

Project 2 of FYS-STK3155/4155 (Applied Data Analysis and Machine Learning), University of Oslo, fall 2026.

Our own feed forward neural network in NumPy, with backpropagation verified against JAX and Scikit-Learn, compared with OLS, Ridge and Lasso on the Runge function and with logistic regression on MNIST, with a study of activation functions, depth and width, $L_1$ and $L_2$ penalties and gradient methods.

The report is [`Project2_report.pdf`](Project2_report.pdf).

## Group members

* Pietro Monacelli
* Georgios Kotrotsios
* Giorgio Spreafico
* Elia Valenti

## Repository structure

```
FYS-STK4155-Project2/
    README.md
    requirements.txt
    Project2_report.pdf              the report
    Code/
        Project2.ipynb               the notebook, executed from top to bottom, with all outputs
        Project2.py                  the same code, exported as a Python script
        Figures/                     the 18 figures of the report, as PDF, written by the notebook
    Results/
        summary_regression.txt       all regression models side by side (part f)
        summary_classification.txt   all MNIST models side by side (part f)
        final_run_outputs.txt        every printed output of the final run, section by section
```

## Installation

The final run used Python 3.13.5. Install the packages with

```
git clone https://github.com/pietromonacelli/FYS-STK4155-Project2.git
cd FYS-STK4155-Project2
pip install -r requirements.txt
```

## How to run

```
cd Code
jupyter notebook Project2.ipynb
```

and run all cells from the top (Kernel, Restart and Run All).

* Run it from the `Code` folder: the notebook writes the figures to `Code/Figures` and downloads `mnist.npz` (11.5 MB) into `Code`, from the address given in the project text (section e.0, only the first time; it needs an internet connection).
* Seeds: 2026 everywhere, except for the initial weights of the networks of the gradient checks (seed 1, section b.4) and the starter cell of the project text with `LogisticRegression` (`random_state=42`).
* Run time: the notebook prints the time of every scan. On our machine the longest steps were the starter cell with Scikit-Learn's `LogisticRegression` (10 to 14 minutes in our runs) and the final MNIST models (366 s for our network, 299 s for `MLPClassifier`).
* `Project2.py` was exported with `jupyter nbconvert --to script Project2.ipynb`, to read the code as a Python file; to reproduce the results, run the notebook.

## The notebook and what it produces

`Code/Project2.ipynb` follows the parts of the project text. Every section is numbered (a.1, b.6, ...), and the report refers to it as (notebook, Sec. b.6).

| Part | Sections | What it does | Figures in `Code/Figures` |
|---|---|---|---|
| a) | a.0 to a.6 | cost functions, activation functions and their derivatives, checked against `jax.grad`; output error of matched and mismatched pairs of output activation and cost | `a5_activations_output_error.pdf` |
| b) | b.1 to b.8 | the data and functions of Project 1; our network class `FFNN` and the training function `train_network`; verification against `jax.grad`, finite differences, the closed form OLS and Ridge solutions and the Lasso; the OLS benchmark; gradient methods and learning rates; networks against OLS and Scikit-Learn's `MLPRegressor` | `b6_cv_learning_rate.pdf`, `b7_networks_vs_ols.pdf` |
| c) | c.1 to c.6 | learning rate for each activation; depth and width for the sigmoid, the ReLU and the leaky ReLU; signs of overfitting; vanishing gradients; the best network of each activation on the test set | `c2_depth_width.pdf`, `c3_train_cv_vs_parameters.pdf`, `c3_learning_curves.pdf`, `c4_vanishing_gradients.pdf`, `c5_best_fits.pdf` |
| d) | d.1 to d.5 | $L_2$ and $L_1$ penalties against depth and learning rate; the penalised networks against Ridge and Lasso of Project 1; what the penalties do to the weights | `d12_penalties.pdf`, `d3_penalised_vs_ridge_lasso.pdf`, `d4_weights.pdf` |
| e) | e.0 to e.9 | MNIST with a validation set; logistic regression as our network without hidden layers; gradient methods, learning rate and penalty, depth and width, activations; the final models with early stopping, evaluated once on the test set, against Scikit-Learn; confusion matrices | `e2_logistic_regression.pdf`, `e3_methods_learning_rates.pdf`, `e4_learning_rate_lambda.pdf`, `e5_depth_width.pdf`, `e6_activations.pdf`, `e7_final_models.pdf`, `e8_confusion_matrices.pdf` |
| f) | code for part f), f.1 | all results side by side (the two tables in `Results`) and the critical evaluation | |

After part f) the notebook lists its references and declares its use of large language models (Appendix: Use of AI/LLM tools). The four starter cells of the project text are kept unchanged.

## Results

* `Results/summary_regression.txt`: every regression model, with CV MSE, test MSE, MSE to the true function and number of parameters.
* `Results/summary_classification.txt`: every MNIST model, with test accuracy, number of parameters, epochs and training time.
* `Results/final_run_outputs.txt`: every printed output of the final run, under the heading of its section.

These files are copied from the outputs stored in the executed notebook; nothing in them was computed separately.

## Use of large language models

The use of large language models is declared in Appendix C of the report and, in the notebook, by an LLM note above every code cell of ours and by `LLM-assisted` tags in the docstrings.

## Data

MNIST: Y. LeCun, C. Cortes and C. J. C. Burges, *The MNIST database of handwritten digits*. The Runge data of Project 1 are generated in the notebook with seed 2026.
