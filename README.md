# Building, Breaking and Fixing a Neural Network — Fashion-MNIST

A from-scratch-to-tuned deep learning pipeline built for a Deep Learning course assignment: a two-layer MLP implemented in raw NumPy, verified against PyTorch autograd, then progressively rebuilt in PyTorch, deliberately overfit, repaired with a full regularisation study, and finally tuned with random search + 5-fold cross-validation — all on Fashion-MNIST.

Every design choice in this repo is *measured*, not assumed: each part reports the actual numbers produced by running the code, including cases where the result didn't match the "textbook" expectation.

---

## Results at a glance

| Metric | Value |
|---|---|
| **Final test accuracy** | **86.25%** |
| Macro precision | 86.39% |
| Macro recall | 86.25% |
| Macro F1 | 86.30% |

**Selected configuration** (found via 12-config random search, scored with 5-fold cross-validation):

| Hyperparameter | Value |
|---|---|
| Learning rate | 0.003 |
| Hidden width | 128 |
| Dropout | 0.2 |
| Regularisation | Dropout (p = 0.2) + L2 weight decay (λ = 1e-4) |

**Single change that helped most:** increasing the training set from 2,000 to **20,000 samples**. On the same deliberately over-parameterised network (four 512-unit hidden layers), this achieved **87.76% validation accuracy** at **99.30% training accuracy**, cutting the generalisation gap from the 15.89% baseline down to **11.54%** — a 4.35 percentage-point reduction in the gap for only a 0.70 percentage-point drop in training accuracy. No regularisation technique tested (L1, L2, dropout, batch norm, early stopping, data augmentation) came close to that trade-off on its own.

---

## What's in this repository

```
├── Fashion_MNIST_Full_Assignment.ipynb   # complete, executed notebook (all 7 parts)
├── Fashion_MNIST_Summary.docx            # one-page results summary
├── README.md                             # this file
└── assets/                               # screenshots referenced below (see "Screenshots")
```

---

## Project structure — what each part does

| Part | Task | Highlight result |
|---|---|---|
| **1. Backprop from scratch** | Two-layer MLP (784→64→10) implemented in pure NumPy — manual forward pass, manual backward pass, manual gradient descent — then checked against PyTorch autograd on an identical batch with identical weights. | Max gradient difference vs. PyTorch ≈ 1e-17 (floating-point precision — implementation verified correct). |
| **2. Activation study** | PyTorch MLP with two hidden layers, one run per activation (sigmoid, tanh, ReLU, leaky ReLU), everything else held fixed. | Sigmoid's first-layer gradients grow far more slowly across training than ReLU's (vanishing gradients); a measurable fraction of ReLU units go permanently dead. |
| **3. Loss functions** | Identical network trained with cross-entropy vs. MSE on one-hot targets, plus a small regression sanity-check on a tabular dataset. | Cross-entropy reaches usable accuracy in a fraction of the epochs MSE needs, because its gradient doesn't vanish for confidently-wrong predictions the way MSE-on-softmax does. |
| **4. Optimiser comparison** | SGD, SGD+momentum, RMSProp, Adam — first at a shared learning rate, then each tuned individually. | Adam reaches the 85% validation-accuracy target in the fewest epochs and is the least sensitive to learning-rate choice. |
| **5. Forced overfitting** | Training set cut to 2,000 samples, network widened to four 512-unit hidden layers, trained past 99% training accuracy on purpose. | 100.00% training accuracy vs. 84.11% validation accuracy — a textbook high-variance failure mode. |
| **6. Regularisation study** | Starting from the Part 5 model, applied one at a time: L2 (3 strengths), L1, dropout (3 rates), batch norm, early stopping, data augmentation, and more training data (10k / 20k samples). | More data was the single most effective lever (see above); over-regularising (very high dropout / very large L2) traded away training accuracy faster than it closed the gap. |
| **7. Hyperparameter tuning** | Random search over learning rate, hidden width, and dropout rate (12 configurations), each scored by 5-fold cross-validation; best configuration retrained on the full training set with the best Part 6 regularisation, evaluated **once** on the untouched test set. | 86.25% test accuracy, macro F1 = 86.30% — the model selection process never touched the test set until this final, single evaluation. |

---

## Reproducing the results

1. Go to **[kaggle.com/code](https://www.kaggle.com/code)** → **New Notebook**.
2. Under **Add Input**, search for and add the dataset `zalando-research/fashionmnist`. It mounts at `/kaggle/input/fashionmnist/` with `fashion-mnist_train.csv` and `fashion-mnist_test.csv` — the notebook reads these directly.
3. Set the accelerator to **GPU T4 x2** (Settings → Accelerator).
4. Import `Fashion_MNIST_Full_Assignment.ipynb` (File → Import Notebook).
5. **Run All.** Total runtime is roughly 15–20 minutes; the notebook auto-detects `cuda` and uses the GPU when available, falling back to CPU otherwise.

No extra installation is required — `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `seaborn` are all preinstalled on Kaggle.

> **Running outside Kaggle:** if `/kaggle/input/fashionmnist` isn't found, the data-loading cell automatically falls back to `tensorflow.keras.datasets.fashion_mnist.load_data()`, which downloads the dataset directly.

### Reproducibility

Every experiment fixes `SEED = 42` (with small, deterministic per-fold/per-run offsets wherever multiple independent draws are needed, e.g. cross-validation folds). The test set is loaded once at the top of the notebook and is not touched again until the single evaluation cell in Part 7 — there is no test-set leakage during model selection.



## Screenshots



### Confusion matrix

<img width="900" height="800" alt="image" src="https://github.com/user-attachments/assets/b15e542c-f1ca-4f25-ab13-41b3e2a776df" />



### Part 6 regularisation study —

 <img width="1105" height="674" alt="image" src="https://github.com/user-attachments/assets/5a1bf55d-cedc-484f-988f-042c59a12bd2" />


###  Final test Accuracy

<img width="910" height="444" alt="image" src="https://github.com/user-attachments/assets/35fadf8e-5ddf-4682-a565-9d79b5f9db27" />



---

## Notes on the results

The final tuned model (86.25% test accuracy) is a modest, genuine improvement over the earlier baseline models in this project  not a dramatic one, and that's an honest reflection of the setup rather than a shortcoming: Fashion-MNIST classified with a plain MLP (no convolutional structure) has a fairly low ceiling, and the Part 2 baseline was already reasonably well-tuned before Part 7's search began. The more interesting finding is methodological — that **training-set size dominated every regularisation technique tested** for closing the overfitting gap on this architecture, which is a useful, transferable lesson independent of the specific accuracy numbers.

---

## Author

Shahmeer Haider— BSCS, FAST-NUCES
