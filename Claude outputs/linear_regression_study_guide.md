# Linear Regression — Study Guide
### Based on Tufts CS135, Day 03 slides (Prof. Chris Magnano)

This lecture is what the professor calls "a mathy day": instead of just handing you the linear regression formula, it walks you through *deriving* it from scratch in three stages of increasing difficulty. The formulas aren't the point — the point is understanding *where the formulas come from* and picking up four recurring ideas along the way: **parameter space**, **convexity**, **the gradient**, and **when a solution is unique**. Below is a page-by-page walkthrough with the reasoning filled in and every formula written out in full math notation.

---

## Page 1 — Title slide

Sets the stage with two pictures: a simple 2D scatter plot with a best-fit line through it (one input variable, one output), and a 3D version where a flat *plane* is fit through points that depend on two input variables ($X_1, X_2$). This is the visual summary of the whole lecture: linear regression finds the best-fitting straight line (or plane, or hyperplane in higher dimensions) through a cloud of data points.

## Page 2 — Warmup

Before teaching anything new, the slide asks you to recall the objective from a previous lecture. The handwritten answer lays out the core vocabulary you'll use all class:

- **Training data**: pairs $\{x, y\}$ — inputs and their true outputs.
- **Model/prediction**: written $\hat{y}(x)$ or more explicitly $\hat{y}(x, w, b)$ — your function's guess for $y$ given $x$, where $w$ (weight) and $b$ (bias/intercept) are the *parameters* you get to choose.
- **MSE (mean squared error)** objective:

$$\operatorname*{arg\,min}_{w,b} \big(\hat{y}(x) - y\big)^2$$

expanded for a linear model as $(wx + b - y)^2$.

In words: search over all possible values of $w$ and $b$, and pick the ones that make your predictions as close as possible to the true $y$ values, where "close" is measured by squared error.

## Page 3 — Objectives for today

A roadmap. Three big takeaways are promised:

1. This is a "least squares" derivation day — you'll see the math worked out longhand in three scenarios of increasing complexity, mainly so the *properties* of linear regression (not the derivation itself) stick with you. You won't be asked to reproduce every derivation on an exam, but you should understand what's happening in each step.
2. Four concepts to watch for: **parameter space**, **convexity**, **gradient**, and **uniqueness of the optimal solution**.
3. A practical programming tip for later: when you actually solve for the best-fit parameters in code, use `np.linalg.solve` rather than computing a matrix inverse with `np.linalg.inv`. (Explained fully on page 25.)

## Page 4 — Task: Regression

This grounds linear regression in the general "machine learning recipe" — a system takes in a **task/goal**, **past data (experience)**, and a **performance measure**, and outputs a way to turn new input into a prediction.

For regression specifically: the **task** is predicting a continuous number $y$ (the example given is exam performance) from an input $x$ (hours of sleep). The **training data** is a collection of past $(x, y)$ pairs — actual students' sleep and exam scores. The scatter plot shows exactly this: as $x$ (hours of sleep) increases, $y$ (exam score) tends to increase too, and the red line is the linear regression fit through that trend.

The key word is *regression*: unlike classification (predicting a category), the output $y$ here is a real number that can take any value on a continuum.

## Page 5 — Linear Regression: Training

This formalizes the goal: find the weight $w$ and bias $b$ that produce the **lowest possible mean squared error** across all $N$ training examples. This is called a "least squares" problem because you're minimizing a sum of squares.

Two candidate loss functions are shown side by side:

**Mean squared error (MSE)** — boxed in red because it's the one used going forward:

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N} \big(y_i - \hat{y}_i(x_i)\big)^2$$

**Mean absolute error (MAE)**:

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N} \big|y_i - \hat{y}_i(x_i)\big|$$

Why does the course pick MSE over MAE? The slide doesn't say explicitly here, but it's worth knowing: MSE is smooth and differentiable everywhere (you can take its derivative cleanly, even at zero error), and it turns out to produce a loss function that is **convex** — bowl-shaped, with a single unique minimum you can solve for directly with algebra, as the next several slides show. MAE has a sharp corner at zero error where the derivative isn't defined, which makes it much harder to optimize with plain calculus. This is the practical reason "least squares" is the classic, foundational version of linear regression.

## Page 6 — Simple First Scenario

Before touching $x$ at all, the slides deliberately simplify to the *dumbest possible model*: **no predictors**. Imagine someone asks "how will I do on the final?" and you're not allowed to ask them anything about themselves — you just give the same single number to everyone.

The model is:

$$\hat{y}() = b$$

— a constant. There's no $x$; the only "parameter" is $b$ itself. You collect a bunch of $y_i$ values (no $x_i$ needed) and choose $b$ to minimize:

$$\frac{1}{N}\sum_{i=1}^{N} (y_i - b)^2$$

The question posed: intuitively, what should the optimal $b$ be, in relation to the data? (Answer worked out on the next page — spoiler: it's the average.)

## Page 7 — Simple First Scenario: Derivation

This is your first full worked derivation, and the pattern here repeats in every later derivation, so it's worth internalizing the steps:

**Step 1 — expand the square.**

$$(b - y)^2 = b^2 - 2by + y^2$$

**Step 2 — sum over all $N$ examples**, splitting the sum term-by-term:

$$\sum_{i=1}^{N}\big(b^2 - 2by_i + y_i^2\big) = \sum_{i=1}^{N} b^2 - \sum_{i=1}^{N} 2by_i + \sum_{i=1}^{N} y_i^2 = Nb^2 - 2b\sum_{i=1}^{N} y_i + \sum_{i=1}^{N} y_i^2$$

($\sum b^2$ becomes $Nb^2$ because $b$ doesn't depend on $i$, so you're just adding the same constant $N$ times.)

**Step 3 — recognize the shape.** Call this whole expression:

$$J(b) = Nb^2 - 2b\sum_{i=1}^N y_i + \sum_{i=1}^N y_i^2$$

As a function of $b$, this is a **quadratic** — a parabola opening upward (since the coefficient on $b^2$ is $N > 0$, which is positive). A parabola like this is **convex**: it looks like a bowl, with exactly one lowest point (its minimum), no other local dips or humps to get trapped in. This is the "convexity" concept the objectives page flagged — it's what guarantees that "set the derivative to zero" will find *the* global minimum, not just *a* local one.

**Step 4 — minimize by calculus.** Take the derivative of $J$ with respect to $b$, and set it to $0$ (this is exactly where the bottom of the bowl is — the flat point):

$$\frac{d}{db} J(b) = 2Nb - 2\sum_{i=1}^N y_i = 0$$

**Step 5 — solve for $b$.**

$$2Nb = 2\sum_{i=1}^N y_i \;\;\Longrightarrow\;\; Nb = \sum_{i=1}^N y_i \;\;\Longrightarrow\;\; b = \frac{\sum_{i=1}^N y_i}{N} = \bar{y}$$

The optimal constant prediction is simply the **mean (average) of all the $y$ values** in your training data. This matches intuition perfectly: if you have to give everyone the same answer with zero information about them, your best bet is the average outcome you've seen historically.

## Page 8

A blank continuation slide (kept blank because the derivation above fit on one slide and didn't need to spill over).

## Page 9 — Second Scenario: 1 predictor

Now you're allowed one input variable $x$. The model becomes:

$$\hat{y}(x_i) = wx_i + b$$

— the familiar "slope-intercept" line, where $w$ is the slope and $b$ is the $y$-intercept. The loss function is the same MSE idea, just now with predictions that depend on $x$.

This slide introduces an important distinction: **parameter space** vs. **data space**. This is one of the four flagged concepts, so it's worth pausing on:

- **Data space** is the ordinary $(x, y)$ plot you're used to seeing — each point is one training example, and a candidate line is a diagonal line drawn through that cloud of points.
- **Parameter space** is a *different* plot, where the axes are $w$ and $b$ themselves — not $x$ and $y$! Every single point in this space represents one *possible line* (one choice of slope and intercept). The point $(0, 1)$ marked on the sketch, for instance, represents the specific line "$w=0, b=1$" (a flat horizontal line at height 1).

Why does this matter? Because "finding the best line" is exactly the same problem as "finding the best point in parameter space" — and that reframing is what lets you use calculus/optimization tools on $w$ and $b$ as your variables, rather than trying to reason about lines directly.

## Page 10 — Visualizing the loss function (spoiler)

This shows what the loss function $J(w, b)$ (here labeled $\theta_0, \theta_1$ for the intercept and slope) looks like when you plot it *over parameter space*. Because $J$ is a sum of squared errors — and we just saw on page 7 that even the 1-parameter version is a convex parabola — the 2-parameter version is a **convex bowl** (a paraboloid) sitting over the $(\theta_0, \theta_1)$ plane. Every point on that surface tells you "if you picked this $w$ and $b$, this is how much total squared error you'd get."

The two 3D plots are the same bowl, viewed with and without contour lines drawn on its surface. The bottom two plots show those same contours from directly above, like a topographic map. The caption explains: **"level set" contours are all the points that share the same loss value** — like elevation lines on a map, where every point on one ring has exactly the same "height" (loss). The center of the innermost ring is the single lowest point of the bowl — the optimal $(w, b)$.

This is a genuinely useful mental picture for the rest of the course: training a model = finding your way to the center of this bowl.

## Page 11 — Second Scenario: Derivation

Same recipe as page 7, just with two parameters ($w, b$) instead of one. The objective:

$$\operatorname*{arg\,min}_{w,b} \sum_{i=1}^N \big(\hat{y}(x_i) - y_i\big)^2 = \operatorname*{arg\,min}_{w,b} \sum_{i=1}^N (wx_i + b - y_i)^2$$

**Step 1 — expand** $(wx_i + b - y_i)^2$ into all its cross-terms:

$$w^2x_i^2 + 2wx_ib - 2wx_iy_i - 2by_i + b^2 + y_i^2$$

**Step 2 — sum over $i$ and group by which parameter each term involves:**

$$J(w,b) = w^2\sum_{i=1}^N x_i^2 + 2wb\sum_{i=1}^N x_i - 2w\sum_{i=1}^N x_iy_i - 2b\sum_{i=1}^N y_i + Nb^2 + \sum_{i=1}^N y_i^2$$

The note "Convex — only has 1 max/min" appears here: even with two parameters, this is still a quadratic bowl shape (just now a 3D bowl instead of a 2D parabola), so it still has exactly one minimum.

**Step 3 — take partial derivatives and set both to zero.** Because there are now *two* parameters, you need *two* equations: $\partial J/\partial b = 0$ and $\partial J/\partial w = 0$ (2 equations, 2 unknowns — solvable simultaneously).

Working out $\partial J/\partial b$ first:

$$\frac{\partial J}{\partial b} = 2Nb + 2w\sum_{i=1}^N x_i - 2\sum_{i=1}^N y_i = 0$$

$$\Longrightarrow\; Nb = \sum_{i=1}^N y_i - w\sum_{i=1}^N x_i \;\;\Longrightarrow\;\; \boxed{b = \frac{\sum_{i=1}^N y_i - w\sum_{i=1}^N x_i}{N} = \bar{y} - w\bar{x}}$$

Notice the structure: this is the mean of $y$, adjusted by $w$ times the mean of $x$. The slide flags this as "plug this in for $b$, then solve $\partial J/\partial w = 0$" — i.e., you substitute this expression for $b$ into the *other* partial-derivative equation, which leaves you with one equation in the one remaining unknown, $w$, and you solve for it.

## Page 12

Another blank continuation slide.

## Page 13 — 1-D Solution and Summary

This is the punchline of the 1-predictor derivation — the final closed-form answer you get after finishing the algebra from page 11:

$$\min_{w \in \mathbb{R},\, b \in \mathbb{R}} \;\; \sum_{n=1}^{N} \big(y_n - \hat{y}(x_n, w, b)\big)^2$$

and the parameters that achieve that minimum are:

$$w = \frac{\sum_{n=1}^{N} (x_n - \bar{x})(y_n - \bar{y})}{\sum_{n=1}^{N} (x_n - \bar{x})^2} \qquad\qquad b = \bar{y} - w\bar{x}$$

where

$$\bar{x} = \operatorname{mean}(x_1, \dots, x_N) \qquad \bar{y} = \operatorname{mean}(y_1, \dots, y_N)$$

This is the classical "simple linear regression" formula you may have seen in a statistics class. Intuitively:

- The numerator of $w$, $\sum_n (x_n-\bar{x})(y_n-\bar{y})$, measures how $x$ and $y$ move together — it's large and positive when $x$ above-average tends to pair with $y$ above-average (and vice versa), which is essentially the *covariance* between $x$ and $y$. The denominator, $\sum_n (x_n-\bar{x})^2$, is the spread (*variance*) of $x$ alone. So $w = \text{covariance}(x,y) / \text{variance}(x)$ — the slope is "how much $y$ tends to change per unit change in $x$," normalized by how spread out $x$ is.
- $b$ is chosen so that the line passes exactly through the point $(\bar{x}, \bar{y})$ — the "center of mass" of the data.

The sidebar spells out the general recipe used to get here: (1) compute the gradient with respect to $w$ and with respect to $b$ — two expressions; (2) set both gradients equal to zero and solve the resulting system of 2 equations, 2 unknowns.

## Page 14 — Comparing Scenarios

A conceptual question, not more algebra. You and a friend each collect the same $(x_i,y_i)$ data. Your friend fits the full linear model ($w$ and $b$, scenario 2); you throw away the $x_i$ values entirely and just use the constant-prediction model from scenario 1 ($b = \bar{y}$). Question: whose model has lower *training* MSE?

The handwritten answer: **the friend will always have equal or smaller training MSE.**

Why? Because the constant model is a special case of the linear model — specifically, it's the linear model with $w$ locked at $0$. Your friend is optimizing over a strictly *larger* search space (all possible $w$ and $b$) that *includes* your restricted option ($w=0$) as one possibility among many. Since your friend is allowed to choose $w=0$ too if that were actually optimal, and is additionally allowed to choose something better if it exists, your friend can never end up doing worse than you — only equal or better.

This is a small preview of a much bigger idea you'll meet later in the course: **adding more flexibility/parameters to a model can never increase training error** (only match or reduce it). This is exactly why training error alone is a *dangerous* way to judge a model — an overly flexible model will always look best in-sample, even if it's just memorizing noise rather than learning something that generalizes (overfitting). That tension isn't discussed explicitly on this slide, but it's the natural next question this comparison sets up.

## Page 15 — Hard Scenario: many features

Now the real case: instead of one input $x$, each example has **$F$ different features** ($F$ predictors). Trying to write out $F$ separate weight terms every time gets unwieldy fast, so this slide introduces **matrix/vector notation** to handle all $F$ weights at once. This is the "hard scenario" — the notation is the hard part, not the underlying idea.

New notation introduced:

- Each example's feature vector: $x_i = \{x_{i1}, \dots, x_{iF}\}$ ($F$ raw feature values for example $i$).
- An **augmented** feature vector $\tilde{x}_i = \{1, x_{i1}, \dots, x_{iF}\}$ — a $1$ is tacked onto the front. This trick lets the bias term $b$ get folded into the same dot-product as the weights (more on why below).
- The parameter vector:

$$\theta = [b, w_1, w_2, \dots, w_F]^T$$

— all the weights *and* the bias stacked into one column vector.
- Prediction for one example:

$$\hat{y}(\tilde{x}_i) = \tilde{x}_i^T\theta = b + w_1 x_{i1} + w_2 x_{i2} + \dots + w_F x_{iF}$$

This is exactly the same linear model as before (an intercept plus a weighted sum of features) — just written as a single dot product instead of a long sum. The reason the trick with prepending "1" works: when you dot $[1, x_{i1},\dots,x_{iF}]$ against $[b, w_1,\dots,w_F]$, the "1" multiplies $b$, and each feature multiplies its matching weight — giving you $b + w_1x_{i1} + \dots$ in one clean operation.
- Stack *all* $N$ examples' augmented feature vectors as rows into one big matrix $\tilde{X}$ (shape $N \times (F+1)$). Multiplying $\tilde{X}\theta$ then gives you a **length-$N$ vector of every example's prediction at once** — this answers the question posed at the bottom of the slide, "what happens when we multiply $\tilde{X}\theta$?" You get $\hat{y}(\tilde{x}_1), \hat{y}(\tilde{x}_2), \dots, \hat{y}(\tilde{x}_N)$ stacked in a column, all computed in a single matrix-vector multiply.
- The loss function in this notation:

$$J(\theta, \tilde{X}) = (\tilde{X}\theta - y)^T(\tilde{X}\theta - y)$$

This is the matrix-algebra way of writing "sum of squared errors" — for any vector $v$, $v^Tv = \sum_i v_i^2$ (dot a vector with itself and you get the sum of its squared entries). Here $v = \tilde{X}\theta - y$, the vector of *residuals* (prediction minus true value) for every example, so $v^Tv$ is exactly $\sum(\text{residual})^2$ summed over all $N$ examples.
- The gradient of $J$ with respect to $\theta$, $\nabla_\theta J$, is itself a vector — one entry per parameter ($\partial J/\partial\theta_1, \partial J/\partial\theta_2, \dots$, stacked into a column). This generalizes "take the derivative" to many parameters at once.

The bottom-right handwritten work sketches the expansion of $J(\theta,\tilde{x}) = (\tilde{X}\theta-y)^T(\tilde{X}\theta-y)$, which multiplies out (using that a scalar equals its own transpose, so $\theta^T\tilde{X}^Ty = y^T\tilde{X}\theta$) to:

$$\theta^T\tilde{X}^T\tilde{X}\theta - 2y^T\tilde{X}\theta + y^Ty$$

Taking the gradient of this with respect to $\theta$ and setting it to zero gives:

$$2\tilde{X}^T\tilde{X}\theta - 2\tilde{X}^Ty = 0 \;\;\Longrightarrow\;\; \boxed{\tilde{X}^T\tilde{X}\theta = \tilde{X}^Ty}$$

This is exactly where the final least-squares formula (shown properly on pages 21–24) comes from — this slide is a first, informal pass through the same algebra that's cleaned up later.

## Page 16 — What is our loss function?

A pause-and-think slide: having introduced the matrix notation, it asks you to actually write out the loss function $J(\theta)$ yourself, and to ask: is it still "bowl-shaped" (convex), like the 1- and 2-parameter cases were? The intended answer is **yes** — the sum-of-squared-errors loss stays convex no matter how many parameters (features) you add, because it's still fundamentally a sum of squared linear terms in $\theta$ (a quadratic form), which is always convex (technically, because the "curvature matrix" $\tilde{X}^T\tilde{X}$ is positive semi-definite — but the intuitive takeaway is just: more features doesn't break the nice bowl shape).

## Page 17 — Set derivatives to 0, right?

Confirms the strategy: because the loss is convex, its single minimum is exactly the point where the derivative in *every* direction is zero — meaning every partial derivative (with respect to every individual parameter) must be zero simultaneously. With $F+1$ parameters, that's $F+1$ separate equations to solve at once, which is exactly the kind of bookkeeping the **gradient** notation ($\nabla_\theta J$) is designed to make manageable: instead of writing out $F+1$ separate equations by hand, you write one vector equation, $\nabla_\theta J(\theta) = 0$, that bundles them all together.

## Pages 18–19 — Gradient Practice

Two worked warm-up examples to build the vector-calculus toolkit you need for page 15's derivation:

- $\nabla_\theta(\theta^T\theta)$: the gradient of "$\theta$ dotted with itself" (i.e., $\sum_i \theta_i^2$) is $2\theta$:

$$\nabla_\theta(\theta^T\theta) = 2\theta$$

(Quick sanity check in 1 dimension: $\frac{d}{d\theta}(\theta^2) = 2\theta$ — matches.)

- $\nabla_\theta(\theta^Tv)$: the gradient of "$\theta$ dotted with a constant vector $v$" is simply $v$:

$$\nabla_\theta(\theta^Tv) = v$$

(1-D sanity check: $\frac{d}{d\theta}(\theta \cdot v) = v$ — matches.)

## Page 20 — Gradient Summary

A reference table collecting the vector-calculus identities you need, all of which behave exactly like their single-variable calculus counterparts:

| Expression | Shape | Gradient w.r.t. $\theta$ | Shape of gradient |
|---|---|---|---|
| any scalar function of $\theta$ | $1\times 1$ | a column vector | $F\times 1$ |
| $v^T\theta$ | $1\times 1$ | $v$ | $F\times 1$ |
| $\theta^Tv$ | $1\times 1$ | $v$ | $F\times 1$ |
| $\theta^T\theta$ | $1\times 1$ | $2\theta$ | $F\times 1$ |
| $\theta^TQ^TQ\theta$ | $1\times 1$ | $2Q^TQ\theta$ | $F\times 1$ |

The pattern to notice: no matter how complicated the expression, since the loss itself is always a single *number* (scalar) — it's one total error value — its gradient with respect to the parameter vector $\theta$ is always a vector the same shape as $\theta$ ($F+1$ entries, one gradient component per parameter). The last row, $\theta^TQ^TQ\theta$, is the general version that directly produces the result needed on page 15: if you let $Q = \tilde{X}$, then $\theta^T\tilde{X}^T\tilde{X}\theta$ has gradient $2\tilde{X}^T\tilde{X}\theta$ — exactly the term that shows up in the normal equations.

## Page 21 — Now: Deriving the Least Squares Solution

A section-break slide, signaling that everything from here on formalizes and cleans up the "hard scenario" derivation sketched informally on page 15.

## Page 22

Blank continuation slide.

## Page 23 — Matrix Notation Summary

A clean restatement of the notation (with a minor, harmless difference from page 15: here the augmented vector puts the "1" at the *end* instead of the front — order doesn't matter, as long as $\theta$'s ordering matches):

**Parameter vector**, shape $(F+1, 1)$:

$$\theta = [w_1, w_2, \dots, w_F, b]^T$$

**Feature vector for the $n$-th example**, shape $(F+1, 1)$:

$$\tilde{x}_n = [x_{n1}, x_{n2}, \dots, x_{nF}, 1]^T$$

**Prediction** (a scalar):

$$\hat{y}(x_n,\theta) = \theta^T\tilde{x}_n$$

**Training objective** (a scalar):

$$J(\theta) = \sum_{n=1}^{N} \big(y_n - \hat{y}(x_n,\theta)\big)^2$$

## Page 24 — Least Squares Summary

This is the full, final closed-form solution for linear regression with any number of features — the single most important slide in the deck to remember.

**Objective:**

$$\min_{\theta \in \mathbb{R}^{F+1}} \;\; \sum_{n=1}^{N} \big(y_n - \hat{y}(x_n,\theta)\big)^2$$

**Setup:**
- $\tilde{X}$ is the "design matrix," shape $(N, F+1)$ — every row is one training example's features plus a trailing $1$; every column (except the last) is one feature across all examples:

$$\tilde{X} = \begin{bmatrix} x_{11} & \dots & x_{1F} & 1 \\ x_{21} & \dots & x_{2F} & 1 \\ & \dots & & \\ x_{N1} & \dots & x_{NF} & 1 \end{bmatrix}$$

- $y$ is the $(N,1)$ column vector of all true target values:

$$y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{bmatrix}$$

**The normal equations** (this always holds, regardless of whether a unique solution exists):

$$(\tilde{X}^T\tilde{X})\, \theta = \tilde{X}^T y$$

**The closed-form solution, if the inverse exists:**

$$\boxed{\theta = (\tilde{X}^T\tilde{X})^{-1}\tilde{X}^Ty}$$

This single formula is what "training a linear regression model" boils down to mathematically: given your data matrix $\tilde{X}$ and target vector $y$, this formula hands you back the optimal weights and bias directly — no iterative search needed (unlike, say, gradient descent, which many other ML models require).

The "if inverse exists" caveat is where the fourth flagged concept — **uniqueness** — comes back in. $\tilde{X}^T\tilde{X}$ is a square $(F+1)\times(F+1)$ matrix, but it's only invertible if $\tilde{X}$ has "full column rank," which in practice means: you have at least as many training examples as parameters ($N \geq F+1$), and no feature is an exact linear combination of the others (no perfect multicollinearity — e.g., you didn't accidentally include both "height in inches" and "height in cm" as separate features, since one is just a scaled copy of the other). If $\tilde{X}^T\tilde{X}$ isn't invertible, the normal equations still hold, but they no longer pin down a single unique $\theta$ — there's an entire family of parameter settings that all achieve the same minimum loss, and you'd need extra criteria (like regularization, covered in a later lecture) to pick one.

## Page 25 — Use `np.linalg.solve`, not inverse

A crucial practical/implementation note. Even though the formula above is written with an explicit matrix inverse $(\tilde{X}^T\tilde{X})^{-1}$, you should almost never compute that inverse directly in code. Instead, recognize that $\theta$ is simply the solution to the linear system:

$$(\tilde{X}^T\tilde{X})\, \theta = \tilde{X}^Ty$$

and hand that system directly to a linear-system solver, which is both faster and numerically more stable than explicitly inverting a matrix and then multiplying (inverting is more prone to floating-point precision errors, especially when $\tilde{X}^T\tilde{X}$ is close to non-invertible). The example code:

```python
xTx_FF = x_NF.T.dot(x_NF)          # computes X̃ᵀX̃, shape (F+1, F+1)
xy_F1  = x_NF.T.dot(y_N1)          # computes X̃ᵀy, shape (F+1, 1)
theta_F1 = np.linalg.solve(xTx_FF, xy_F1)   # solves for θ directly
```

This is a general best practice worth remembering well beyond this course: **whenever you see "solve $Ax = b$ for $x$," use a linear-system solver (`np.linalg.solve`), not `inverse(A).dot(b)`.**

## Page 26 — day03 Lab

Points you to the take-home lab notebook (due Thursday, linked from the course Schedule page), which turns this whole lecture into hands-on practice:

- **Parts 1–2**: implement 1-dimensional (single-predictor) linear regression in NumPy, and plot the objective function together with its optimum — i.e., recreate the "bowl" picture from page 10 yourself.
- **Part 3**: implement the $F$-dimensional (many-feature) version in NumPy.
- **Part 4**: a refresher on the matrix math and linear-system-solving needed to do so.
- **Parts 5–6**: the practical tip from page 25 — use `np.linalg.solve`, avoid `np.linalg.inv`.

---

## Tying it all together

The whole lecture is really one idea taught three times at increasing difficulty, so that the pattern becomes automatic:

1. **Write your model** $\hat{y}$ as a function of your parameters ($b$ alone → then $w,b$ → then the full vector $\theta$).
2. **Write the loss** as the sum (or mean) of squared errors between predictions and true values.
3. **Expand the loss algebraically** and notice it's always a convex ("bowl-shaped") quadratic function of your parameters — which guarantees a single global minimum exists.
4. **Take the gradient** (the multi-parameter generalization of "take the derivative") and **set it to zero**.
5. **Solve the resulting equations** for your parameters — in the 0-predictor case this collapses to "take the mean"; in the general case it becomes the normal equations

$$\tilde{X}^T\tilde{X}\theta = \tilde{X}^Ty$$

solved as

$$\theta = (\tilde{X}^T\tilde{X})^{-1}\tilde{X}^Ty$$

(computed in practice with `np.linalg.solve`, not an explicit inverse).

And the four concepts to keep in your back pocket: **parameter space** (the space of all candidate $(w,b)$ — or $\theta$ — values you're searching over, distinct from the data space of $x,y$ points), **convexity** (why this bowl shape guarantees "set derivative to zero" actually finds the *global* best answer), **the gradient** (the vector of all partial derivatives at once, letting you handle many parameters as cleanly as one), and **uniqueness** (a closed-form unique answer exists only when $\tilde{X}^T\tilde{X}$ is invertible — enough independent data relative to the number of features).
