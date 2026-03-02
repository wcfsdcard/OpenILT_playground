# Moreau-ILT (Logit-Space) via Co-evolving (u, z) with Quadratic Coupling  
*A practical, GPU-friendly, “analytical” plan with caveats and tuning suggestions.*

## 0) Goal and setting
- **Single process corner**
- **Loss**: L2 mismatch between printed/resist image and target
- **Parameterization**:  
  - Logits: `u ∈ R^n` (unconstrained, "param" variable in simpleilt.py)  
  - Mask: `m = σ(u)` (sigmoid mask)  
- **ILT objective (logit space)**:
  \[
  f(u) = \text{loss function in simpleilt.py}
  \]

---

## 1) Core formulation: coupled objective (inexact Moreau/prox)

Borrow the idea from Moreau envelope. Use a coupled objective in **logit space**:
\[
\Phi_\lambda(u, z) = f(u) + \frac{1}{2\lambda}\|u - z\|_2^2,
\]
where:
- `u` is the “fast” local variable that takes aggressive steps against `f`
- `z` is the “slow” center/anchor variable (low-pass filtered, stabilizes the trajectory)
- `λ` controls coupling strength (coupling stiffness is `1/λ`)

**Interpretation:**
- If `u` were solved (approximately) to minimize `f(u) + (1/(2λ))||u-z||^2` each cycle, this corresponds to a proximal-point / Moreau envelope method.  
- With **one u-step per cycle**, it is best viewed as **block/alternating gradient descent on Φλ** (an *inexact prox / relaxed proximal-point* scheme).

---

## 2) One optimization “cycle”: one u-step + one z-step
### 2.1 Recommended u-step: split update (preserves meaning of λ even if you use Adam on f)
Apply Adam/SGD to the *sum* gradient:
   \[
   x \leftarrow u - \eta_f \cdot (\nabla f(u)+\frac{1}{\lambda}(u-z))
   \]

> Minimal implementation: simply add $\frac{1}{2\lambda}\|u - z\|_2^2$ into the existing loss function, but keep $z$ OUT OF the autograd compute graph (z will be updated manually)

---

### 2.2 z-step: gradient step on the quadratic (EMA update)
Gradient w.r.t. z:
\[
\nabla_z\Big(\frac{1}{2\lambda}\|u-z\|^2\Big) = \frac{1}{\lambda}(z-u).
\]
One step:
\[
z \leftarrow z - \eta_z \frac{1}{\lambda}(z-u)
\;=\;(1-\beta)z + \beta u,\quad \beta \equiv \eta_z/\lambda.
\]

**Practical rule:** choose a constant **β ∈ (0,1)** and set:
- \[
  \eta_z = \beta\lambda.
  \]
Then β stays constant even if λ changes, and z remains a stable “center” EMA.

Suggested β range:
- Start with **β = 0.05–0.3**
  - smaller β ⇒ z slower ⇒ more “exploration” (u can move more independently)
  - larger β ⇒ z tracks u quickly ⇒ method collapses toward vanilla optimization if too large

---

## 3) λ scheduling (adiabatic continuation) (Optional)
### 3.1 Important convention reminder
With:
\[
f_\lambda(z)=\min_u{f(u)+\frac{1}{2\lambda}\|u-z\|^2},
\]
- **λ → 0**: coupling stiff, forces u ≈ z, and **fλ(z) → f(z)** (low bias)
- **λ large**: coupling weak, more “exploration / smoothing” effect but more bias and weaker enforcement u=z

### 3.2 Recommended direction: start larger λ, then decrease
**Start with larger λ** to allow u to explore (weak tether), then **decrease λ** to “lock in” so u≈z and the method becomes close to optimizing f.

Two safe “adiabatic” choices:
- **Geometric per-epoch:** λ ← 0.7 λ every M iterations (M large enough to see stable progress)
- **Gentle per-iter:** λ ← 0.98 λ every iteration

### 3.3 Keep invariants stable while changing λ
Because the coupling scales as 1/λ, tune step sizes by ratios:
- Keep **β = η_z/λ** constant (choose β once).
- Keep **η_f/λ** within a stable range (e.g., 0.05–0.5 as a starting point), or at least prevent it from exploding as λ shrinks.

> If λ shrinks but η_f is fixed, then η_f/λ grows and the “pull-to-z” gets too stiff; this can cause instability or make u barely move.

---

## 4) Practical hyperparameter suggestions (starting points)
These are **heuristics**; you should tune for your clip size, resist slope, and optics scaling.

### 4.1 Start with a stable β, then tune λ and η_f/λ
- Pick **β = 0.1** first.
- Choose λ so that the quadratic pull is noticeable but not dominating:
  - Target **η_f/λ ≈ 0.1** early (weak pull)
  - Increase **η_f/λ** later as you decrease λ (tighter lock-in)

### 4.2 A simple robust setting (often works)
- Use **SGD+momentum** for ∇f (momentum 0.9), or Adam on ∇f if needed.
- Split u-step with exact quadratic prox.
- z-step as EMA with β=0.1.
- λ schedule: start λ0, decrease by 0.7 every M iterations.

**How to pick λ0 without overthinking**
Compute (once at start) a rough gradient scale `g0 = ||∇f(u0)||∞` and choose λ0 such that a typical desired u−z mismatch Δ produces comparable quadratic force:
- Want `(Δ/λ0) ~ g0` ⇒ `λ0 ~ Δ / g0`
Take Δ as a logit-scale you’re willing to allow early (e.g., Δ≈0.5–2.0).

---

## 5) Caveats and failure modes (read this before running)
### 5.1 “Collapse to vanilla optimization”
If z tracks u too fast, you lose the intended effect.
- If β → 1, then z≈u and the method is close to plain optimization of f.
- Remedy: reduce β (e.g., 0.05–0.2).

### 5.2 Stiffness when λ becomes too small
As λ ↓, coupling ∝ 1/λ becomes stiff. Symptoms:
- oscillations in (u,z)
- u gets “stuck” near z and f decreases very slowly
- numerical issues in logits (large magnitudes)

Remedies:
- decrease λ more slowly (more adiabatic)
- cap η_f/λ (reduce η_f as λ shrinks)
- keep β fixed but ensure β is not large

### 5.3 Sigmoid saturation (both mask and resist)
This is a major ILT pathology:
- logits become too large in magnitude ⇒ σ(z) saturates ⇒ gradients vanish
- resist sigmoid with large slope β makes transitions very sharp ⇒ nonlinearity increases

Remedies (still “analytical”):
- mild logit regularizer early: add `(γ/2)||u||^2` with small γ, decay to 0 later
- do not start with overly aggressive η_f
- keep u tethered with reasonable η_f/λ early to prevent huge logit blow-up

### 5.4 Using Adam on the full u-gradient can blur λ
If you do Adam on `∇f(u) + (u-z)/λ`, Adam’s scale-invariance can reduce the effective control of λ.
- Prefer the split u-update (Adam on ∇f only + exact quadratic prox pull), or use SGD/momentum on the full gradient.

---

## 6) Stronger “local-minima-resistant” extension (best use of multi-GPU, leave for later)
The single-(u,z) scheme mainly stabilizes trajectory. A stronger basin-avoidance story comes from **multiple replicas** `{u_i}` coupled to a shared z:

\[
u_i \leftarrow u_i - \eta_f \nabla f(u_i)\quad(\text{then prox-pull to z})
\]
\[
z \leftarrow (1-\beta)z + \beta \cdot \frac{1}{K}\sum_{i=1}^K u_i
\]
Optionally inject controlled diversity early:
- different initial masks/logits
- small Gaussian noise on logits early
- slight perturbations of resist threshold τ (careful—this changes the objective)

This is closely related to elastic averaging / consensus methods and is far more likely to reduce “bad run” probability.

---

## 7) What to log (to validate the method scientifically)
Per iteration / per stage:
- f(u), f(z)
- ||u−z||2
- ||∇f(u)||2
- Effective ratios: η_f/λ and β=η_z/λ
- Print-image metrics you care about (EPE proxies if you later add them)

For the “less prone to local minima” claim:
- run **R random initializations** and report median/best and variance
- fixed wall-clock budget comparisons vs baseline Adam on logits

---

## 8) Suggested ablations (for ICCAD-style paper clarity)
1. Baseline: Adam on z optimizing f(z)
2. Co-evolution without λ schedule (fixed λ)
3. Co-evolution with λ schedule
4. Adam-on-full-gradient vs split-update (Adam on ∇f only + exact prox pull)
5. Multi-replica extension (if you have multi-GPU)

---

## 9) Summary in one line
**Treat ILT in logit space, introduce a coupled (u,z) objective with quadratic tether (1/λ), update u by a physics gradient step plus an exact quadratic prox-pull, update z by EMA, and decrease λ adiabatically while keeping β=η_z/λ and η_f/λ in stable ranges.**