# Regression Basics — Study Guide
### Based on Tufts CS135, Day 02 slides (annotated), Prof. Chris Magnano

This is the lecture that comes *before* the "Linear Regression" math deep-dive you already have notes on — it sets up the vocabulary and the big picture before day03 gets into deriving formulas. Where day03 was "mathy day," day02 is "framework day": it defines the three-step recipe every regression method follows (predict, train, evaluate), introduces linear regression and K-nearest-neighbors as two contrasting ways to do that recipe, and works through a running real-world example (predicting an abalone's age) to make it concrete. Below is a page-by-page walkthrough, including the instructor's handwritten annotations.

---

## Page 1 — Title slide

"day02: Regression Basics." The three images set up what's ahead: a photo of an actual abalone shell (the running example for the whole lecture), a scatter plot with a fitted line (the classic linear-regression picture), and a scatter plot of real abalone data (ring count vs. shell length) that curves and flattens rather than following a straight line — a subtle hint, revisited later, that not every real relationship is best described by a line.

## Page 2 — Logistics

Pure course administration: start HW0, and email the professor if you're not yet on Piazza/Gradescope, need prerequisites waived, or are considering auditing. Not technical content — skip ahead for the ML material.

## Page 3 — Objectives for today (day 02)

The roadmap for the lecture, and worth keeping in mind as you read the rest:

1. Be able to identify and explain the **3 steps of a regression task**: training, prediction, and evaluation (with evaluation broken down into two specific metrics: mean squared error and mean absolute error).
2. Implement and compare **two different regression methods** that both fit into that same 3-step framework: linear regression and K-nearest neighbors — with the emphasis specifically on how they differ in the *predict* and *evaluate* steps.

## Page 4 — "abalone" (dictionary definition)

A short vocabulary/context slide: an abalone is an edible, rock-clinging sea snail (genus *Haliotis*) with a flattened, spiral shell lined with mother-of-pearl. This just grounds the running example in something concrete before diving into the actual prediction task.

## Page 5 — Motivating task: How old is this sea creature?

The real-world problem this whole lecture builds toward. Ecologists want to monitor abalone population health, which means knowing the age of abalones they find. The catch: age is normally measured by physically cutting open the shell, staining it, and counting growth rings under a microscope — slow and destructive. Age happens to equal exactly **1.5 + (number of rings)**, so predicting ring count is equivalent to predicting age.

Since ring count is a continuous number (not a category), this is naturally a **regression** problem: given only *easy-to-measure* physical traits (things you can get without cutting the shell open), predict the *hard-to-measure* one (rings/age).

## Page 6 — Abalone data: "features"

A table of the measurements you're actually allowed to use as inputs — these are called **features** (also "predictors"): whether the abalone is male (binary: 1 or 0), length, diameter, height, whole weight, shucked (meat) weight, viscera (guts) weight, and shell weight. Each column here is one feature — one measurable property of an abalone that might help predict its ring count.

## Page 7 — Abalone data: first 5 data rows

Shows what the actual dataset looks like in table form: five real abalone, each with its measured features in one table and its true `rings` count (highlighted in red, labeled **target**) in a separate column. The caption nails the core vocabulary: **each row is a data instance** (one abalone), and **each column is a feature** (one measurement). The `rings` column is treated separately because it's not an input — it's the thing you're trying to predict, i.e. the **target** (sometimes called the label, or y).

## Page 8 — Task/Goal (math notation)

This translates the abalone problem into the general math notation you'll reuse for the rest of the course. The goal: build a predictor that can look at a *new* abalone's features and predict its ring count. Training data is what lets you learn the relationship between features and target.

Notation introduced (handwritten in red):

- The training set is a collection of N examples: {x₁, x₂, ..., x_N}, with i ranging over 1, 2, ..., N.
- Each individual example's inputs are bundled into a **feature vector**: x_i = [x_{i1}, x_{i2}, ..., x_{iF}] — i.e., example i's F different feature values (length, weight, etc.), all packed into one vector.
- The model's prediction for example i is written ŷ(x_i) ∈ ℝ — a real number, meant to **approximate the true target** y_i.

This is exactly the same notation the day03 slides build on (x_i for a feature vector, ŷ for the prediction, y_i for ground truth) — this slide is where that notation is first introduced.

## Page 9 — Task/Goal (in code)

The same idea, but from a programming perspective, using NumPy shape conventions you'll use constantly in the labs:

```python
>>> # Given: pretrained regression object model
>>> # Given: 2D array of features x_NF
>>> x_NF.shape
(N, F)
>>> yhat_N1 = model.predict(x_NF)
>>> yhat_N1.shape
(N, 1)
```

`x_NF` is a 2D array with N rows (one per example/abalone) and F columns (one per feature) — the handwritten arrows label this directly: rows = N (examples), columns = F (features). Calling `model.predict(x_NF)` produces `yhat_N1`, a column of N predictions, one per input row.

The **note at the bottom is worth remembering**: in math, we usually write formulas for *one* prediction at a time (ŷ(x_i) for a single example i). In code, you almost always process *all N examples at once* in a single vectorized call — it's faster and is how libraries like scikit-learn are designed to be used. So when you see math with a single subscript i and code with no loop over i, that's not a contradiction — it's the same operation, just expressed two different ways.

## Page 10 — 3 Steps

The core framework of the whole lecture, laid out as a table (with the professor's handwritten answers filled in):

| Step | Input | Output |
|---|---|---|
| **Predict** | x_i (one, or many at once) | ŷ(x_i) — the prediction |
| **Training** | many pairs (x_i, y_i) | "the trained ŷ function" — i.e., a model/parameters that have been fit to the data |
| **Evaluation** | (y_i, ŷ(x_i)) — one or many | some metric of how good you did |

This is the skeleton every regression method in this course will follow, no matter how different they look internally: you **train** on labeled examples to produce a fitted function, you **predict** by feeding that function new inputs, and you **evaluate** by comparing predictions against true answers using some numeric score. The diagram in the corner (task/goal, past data, performance measure → prediction) is the same general ML framing seen in the day03 deck, now split explicitly into these three concrete steps.

## Page 11 — Linear Regression: Predict

Introduces the first concrete method. **Core idea**: decide that the prediction function ŷ will be a straight line (the slide notes "pros and cons of this choice later" — i.e., this is a deliberate simplifying assumption, not a law of nature). The sketch uses age vs. weight as an example: a rough, noisy, but roughly straight-line trend, contrasted with a second sketch (marked with an ✗) showing a curve that *isn't* well described by a line — a visual reminder that the linear assumption doesn't always fit.

**How predict works**: once you already have a line (i.e., already know its parameters), predicting for a new x is just plugging into the line equation. Handwritten:

ŷ(x_i) = wx_i + b

Here **w** and **b** are called the **parameters** of the model (also labeled "weight" and "bias" respectively in the sketch — not to be confused with the *features* "weight" and "age" in the abalone/running example, which is an unfortunate name collision the slide's own sketch runs into). For the general many-feature case, this generalizes to:

ŷ(x_i) = Σ_{f=1}^{F} w_f · x_{if} + b

— one weight per feature, each weight multiplying that feature's value, all summed together plus the bias. This is exactly the same formula introduced from a different angle on page 15 of the day03 deck.

## Page 12 — Linear Regression: Train

The sketch shows several different candidate lines drawn through the same scatter of points — some fit better than others. The question posed: **what is the goal of training?**

Handwritten answer: you're given many (x_i, y_i) pairs, and training turns them into a specific choice of **parameters, w and b**. In other words: out of the infinite number of possible lines you *could* draw, training is the process of searching for the one specific line (the one specific w, b pair) that best matches your data. This is the step where "training data" actually gets used — prediction and evaluation just use whatever line training already found.

## Page 13 — Linear Regression: Evaluation

Introduces the idea of measuring how wrong a single prediction is. For any one data point, there's the **observation y** (the true value) and the **prediction ŷ** (where the fitted line says y should be, given that point's x). The vertical gap between them — how far off the line is from the real point — is called the **error** or **residual**:

residual_i = y_i − ŷ(x_i)

The handwritten "age" / "weight" labels tie this back to the running sketch example. Intuitively: a good line has small residuals (points close to the line); a bad line has large residuals (points far from the line) — and the next several slides are all about how to turn *many individual residuals* into *one single number* summarizing overall model quality.

## Page 14 — Linear Regression: >1-d

Brings back the real abalone data (all 6+ features, not just one) to make the point: once you have more than one predictor, ŷ is still a **linear function** of the features — but geometrically it's no longer a 2D line, it's a flat plane (with 2 features) or a higher-dimensional flat "hyperplane" (with more features). You can't literally draw it on paper anymore once F > 2, but the *math* is the same idea, just extended.

**Predict**: ŷ(x_i) = Σ_{f=1}^{F} w_f x_f + b

**Train**: find w, b such that you minimize Σ_{i=1}^{N} (ŷ(x_i,w,b) − y_i)² — i.e., search over all possible weight/bias combinations for the one that makes the sum of squared errors across *all* training examples as small as possible. This is the first appearance of the "sum of squared errors" objective that the day03 deck derives a closed-form solution for.

## Page 15 — Linear Relationships

Uses a classic textbook example (predicting Sales from advertising spend on TV, Radio, and Newspaper) to illustrate what "linear" really means once you have multiple features. Each scatter plot shows Sales against *one* feature at a time, with its own fitted line. The caption is the key idea: **"The relationship between each feature and the target, fixing other features, is a line."**

In other words, the linear-regression assumption isn't just "the overall relationship looks linear" — it's the more specific claim that if you hold every other feature constant and vary just *one* feature, the target changes at a constant rate (a straight line) with respect to that one feature. That constant rate is exactly what that feature's weight w_f represents. This is a useful way to build intuition for what a fitted weight "means": w_f is "how much the target goes up (or down) per unit increase in feature f, holding everything else fixed."

## Page 16 — Linear Regression: Training (formalized)

Pulls the training goal from page 14 into precise, final form. **Goal**: find the weight coefficients w and intercept/bias b that produce the lowest possible **mean squared error** on your N training examples.

**Optimization problem ("Least Squares")**:

argmin_{w,b} (1/N) Σ_{i=1}^{N} (ŷ(x_i,w,b) − y_i)², where ŷ(x_i,w,b) = Σ_{f=1}^{F} w_f x_{if} + b

The note at the bottom is the important one: **"There is a well-known exact solution: optimal values of w, b. We'll find it next time."** That's a direct pointer to the day03 lecture you already have notes on — that entire lecture is the professor making good on this promise, deriving the closed-form least-squares solution θ = (X̃ᵀX̃)⁻¹X̃ᵀy step by step. So these two lectures are directly continuous: day02 states the optimization problem, day03 solves it.

## Page 17 — Linear Regression: Evaluation (the problem with many features)

A short but important conceptual pivot. With one feature, you could evaluate a model by *looking* at it — plotting the line and eyeballing the residuals, like page 13. But once you have many features (as with the real abalone data, shown again here), **you can no longer draw the picture**. The slide states this plainly: "I can't visualize residuals anymore! We'll have to make use of numbers instead of pictures." This motivates the next few slides, which introduce numeric summary metrics (MSE, RMSE, MAE) that work regardless of how many features you have.

## Page 18 — Squared Error

Formalizes **mean squared error (MSE)**:

MSE = (1/N) Σ_{n=1}^{N} (y_n − ŷ_n)²

Its units are the **square** of whatever units y was in — the handwritten annotation "age²" makes this concrete: if you're predicting age in years, MSE is measured in years², which isn't very intuitive to reason about directly (what does "4 years²" of error feel like?).

To fix that, the slide also introduces **root mean squared error (RMSE)** — literally the square root of MSE:

RMSE = √[(1/N) Σ_{n=1}^{N} (y_n − ŷ_n)²]

Taking the square root brings the units back to the **same units as y** (years, in this example), which makes RMSE much easier to interpret: "our predictions are typically off by about X years."

## Page 19 — Absolute error

Introduces the third metric, **mean absolute error (MAE)**:

MAE = (1/N) Σ_{n=1}^{N} |y_n − ŷ_n|

Like RMSE, its units match the original y units directly (no squaring/square-rooting needed) — it's simply "the average size of the error, ignoring whether it was too high or too low." Squared error and absolute error look similar on the surface, but as the next three slides show numerically, they behave quite differently when one prediction is way off.

## Pages 20–22 — Comparisons 1, 2, and 3

Three worked numeric examples, all using the same four predictions `yhat_N = [0.0, 0.0, 0.0, 0.0]`, but with progressively more extreme true values, to show how MSE, RMSE, and MAE respond differently to outliers.

**Comparison 1 — small errors**: y = [0.4, 0.5, 0.6, 0.7]. Every prediction is off by a modest, similar amount. Result: MSE = 0.315, RMSE ≈ 0.561, MAE = 0.55. All three metrics land in a similar ballpark.

**Comparison 2 — one big error**: y = [0.1, 0.2, 0.3, 2.0]. Three predictions are nearly right, but the fourth is very wrong. Result: MSE = 1.035, RMSE ≈ 1.017, MAE = 0.65. Notice MAE barely moved (0.55 → 0.65) since it treats the big miss the same, unit-for-unit, as a normal-sized miss — but MSE more than tripled, because squaring that one large error (2.0² = 4.0) makes it dominate the sum.

**Comparison 3 — one huge error**: y = [0.01, 0.02, 0.03, 33.0]. Three predictions are essentially perfect, but one is wildly wrong. Result: MSE = 272.25, RMSE ≈ 16.5, MAE = 8.265. Here the effect is dramatic: MAE (8.265) is roughly "one quarter of the huge error," since it's just averaging four numbers — but MSE explodes, because squaring 33 gives 1089, which completely swamps everything else in the sum.

**Takeaways** (stated explicitly on the slide): RMSE is generally recommended over raw MSE because its units are directly interpretable (same scale as y, unlike MSE's squared units). And — the more conceptually important point — **RMSE (and MSE) is far more sensitive to a few large errors than MAE is**. Squaring an error punishes big mistakes disproportionately harder than small ones, so a model evaluated with MSE/RMSE is effectively being told "a few huge mistakes are much worse than many small ones," while a model evaluated with MAE treats every unit of error equally regardless of size. Which metric is "right" depends on whether occasional big misses are actually more costly in your real application — there's no universally correct choice.

## Page 23 — A (very simple) alternative to linear regression when data is non-linear

This slide is a joke, and it's making a serious point through humor: it shows the (in)famous "spurious correlations" chart pairing the number of movies Nicolas Cage appeared in with the number of transportation security screeners in North Dakota — two utterly unrelated quantities that happen to move together (r² = 0.814, a strong statistical fit) purely by coincidence.

The point: a high correlation or a good-looking fit (even a fitted line with impressive r²) doesn't automatically mean two variables have any real, meaningful relationship. It's a cautionary aside before moving on to a genuinely different modeling approach — a reminder to stay skeptical of "the numbers look good" as the only bar for whether a model is trustworthy. (The slide's title, taken literally, promises "an alternative to linear regression for non-linear data" — that promise is actually delivered on the very next slide, with K-nearest-neighbors.)

## Page 24 — Nearest Neighbor Regression

Introduces a fundamentally different strategy from linear regression. **Idea**: instead of *summarizing* the whole relationship between features and target as one tidy line (which forces a specific shape onto the data), "let's be lazy" — don't summarize anything at all.

**Predict**: given a new x, find whichever *training* example x_i is closest to it, and just report that neighbor's y_i as your prediction.

**Training**: the handwritten answer is blunt — **"Nothing (collect the data)."** There's no line to fit, no parameters to search for. The entire "model" is just the stored training dataset itself. This style of method is sometimes called **lazy learning** or **instance-based learning**, in contrast to linear regression's **eager learning** (where all the work happens upfront, during training, to produce a compact set of parameters).

## Page 25 — Nearest Neighbor predictions as a function of one feature are piecewise constant

Shows what 1-nearest-neighbor (1-NN) predictions actually look like when plotted: a **staircase** shape ("piecewise constant"). This makes sense given how the method works — the prediction for any x is always *exactly* the y-value of whichever training point is closest, so the prediction stays flat as x moves around within "closest to the same training point" territory, then jumps abruptly the moment x crosses into "closest to a different training point" territory. This is a very different shape from the single smooth straight line linear regression produces — it's the visual signature of a much more flexible, less constrained kind of model.

## Page 26 — K-Nearest Neighbor Regression

Generalizes 1-NN to **K**-NN. Instead of using only the single closest training point, find the **K** closest training points to the new x, and report the **average** of their y-values as your prediction. (With K=1, this reduces exactly to page 24's method.) Training is still, again, essentially **"Nothing"** — you just store the data; there's no fitting step.

## Pages 27–28 — What value of K do we expect to do best when evaluated on our training data?

A conceptual question, mirroring the "friend with more predictors" comparison from the day03 deck. The handwritten/typed answer: **K = 1**.

Why? Because when you evaluate a K-NN model *on its own training data*, and K=1, the "nearest neighbor" to any training point x_i is *x_i itself* (distance zero) — so the model predicts each training point's own true y_i perfectly, back at itself. Training error is exactly zero. As K grows, predictions get averaged over more and more neighbors (including points that aren't identical to the one being evaluated), which pulls predictions away from any single point's exact true value, so training error rises.

This is the same underlying pattern as "more flexible models fit training data better" seen with linear regression's feature count — here, *smaller* K means a more flexible (less averaged, more jagged) model, which can always fit training data at least as well as a larger, smoother K can.

## Page 29 — kNN: Error vs K

A classic and important figure (from the *Elements of Statistical Learning* textbook) that completes the point pages 27–28 were building toward. It plots **test error** (orange) and **training error** (teal) as K decreases from left (K=151, very smooth/averaged) to right (K=1, most flexible), with K's flexibility also expressed as "degrees of freedom" (N/K) along the top axis.

The **training error** curve drops steadily and monotonically as K shrinks, reaching essentially zero at K=1 — exactly matching the conclusion from the previous two slides. But the **test error** curve (error measured on *new*, unseen data, not the training set) is **U-shaped**: it's high when K is very large (the model is too simple/too averaged — "underfitting"), decreases to a sweet spot at some intermediate K, and then rises again as K approaches 1 (the model has become too sensitive to individual training points — "overfitting"). The purple horizontal "Bayes" line marks the theoretical best-possible error even with a perfect model (irreducible noise in the data itself), and the orange/teal squares labeled "Linear" show where an ordinary linear regression model would land for comparison.

The big lesson of this figure: **the K that minimizes training error (K=1) is not the K that minimizes test error** — training error is a misleading guide to how well a model will actually perform on new data, which is precisely why the "3 steps" framework from page 10 keeps prediction, training, and evaluation carefully separated, and why (as later slides in the course will cover) you should really evaluate models on held-out data they didn't train on, not just the training set.

## Page 30 — Distance metrics

K-NN's whole method depends on being able to say which training points are "closest" to a new point — this slide defines what "distance" actually means numerically, for the general case of F features:

- **Euclidean distance**: dist(x, x′) = √[Σ_{f=1}^{F} (x_f − x′_f)²] — ordinary straight-line ("as the crow flies") distance, generalized to F dimensions.
- **Manhattan distance**: dist(x, x′) = Σ_{f=1}^{F} |x_f − x′_f| — the sum of absolute differences along each feature, like navigating a city grid where you can only move along axes (hence the name — think Manhattan's street grid).

"Many others are possible" — the choice of distance metric is itself a design decision (a "knob to tune," as the next slide calls it) that can meaningfully change which points count as "nearest," especially once you have many features on very different scales.

## Page 31 — Summary of Methods

A direct side-by-side comparison of the two methods covered today:

| | Function class flexibility | Knobs to tune | How to interpret |
|---|---|---|---|
| **Linear Regression** | Linear | Penalize weights *(regularization — more next week)* | Inspect the weights |
| **K-Nearest Neighbors Regression** | Piecewise constant | Number of neighbors (K), distance metric | Inspect the neighbors |

This table is a nice compact way to hold both methods in your head at once: linear regression commits to a rigid, smooth, globally-linear shape, and you understand *why* it made a prediction by looking at its learned weights (each w_f tells you a feature's effect). K-NN commits to no shape at all, instead just imitating whatever nearby examples looked like, and you understand *why* it made a prediction by literally looking at which training examples were closest to it. Linear regression's main tuning knob (weight penalization / regularization) is flagged as a topic for the following week; K-NN's tuning knobs are K itself and the choice of distance metric.

## Page 32 — Recap: day 02

Restates the two objectives from page 3 (3 steps of regression; two methods compared) and adds two extra, important callouts in blue:

- **"Chosen performance metric should be integrated at training."** In other words, the metric you use to judge a model (MSE, MAE, etc.) shouldn't be an afterthought tacked on only at evaluation time — ideally, that's the same thing your training procedure is actually optimizing. (This is exactly what page 16's "least squares" objective does: it trains by minimizing MSE, the same metric you'd use to evaluate.)
- **"Mean squared error is 'easy', but not always the right thing to do."** MSE is mathematically convenient (smooth, differentiable, leads to a closed-form solution, as day03 shows) — but pages 20–22 demonstrated concretely that it's also disproportionately sensitive to a few large errors, which isn't always the behavior you actually want from a model.

## Page 33 — Day 2 Lab

A social/logistics slide: encourages meeting a classmate, since the professor will "strongly" encourage projects to be completed in groups of two.

---

## Tying it together

This lecture's real throughline is the **3-step framework** (train → predict → evaluate) and the idea that different methods can plug into that same framework while making very different tradeoffs. Linear regression commits early to a specific, rigid shape (linear) in exchange for a compact, interpretable set of parameters and a training step that does real work up front. K-nearest-neighbors commits to nothing shape-wise, defers all the "work" to prediction time, and can represent much more complex/jagged relationships — but as the K=1-training-error result and the ESL error-vs-K figure both show, more flexibility isn't free: it comes with a real risk of fitting the training data *too* well at the expense of generalizing to new data. And underlying both methods, the choice of *evaluation metric itself* (MSE vs. RMSE vs. MAE) isn't a neutral bookkeeping detail — it encodes real assumptions about which kinds of mistakes matter more, as the three numeric comparisons made concrete.

This also sets up exactly where the day03 lecture picks up: page 16 here poses the least-squares optimization problem and promises an exact solution "next time" — that promise is what the entire day03 derivation (parameter space, convexity, gradients, and the final normal-equations formula) exists to deliver.
