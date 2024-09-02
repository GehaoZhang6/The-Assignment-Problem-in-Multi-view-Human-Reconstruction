# Ⅰ. Mathematical Derivation

## 1. Fundamental Matrix $F$:

Given:

$$s_1 p_1 = K_1 (R_1 P + t_1)$$

$$s_2 p_2 = K_2 (R_2 P + t_2)$$


where $s$ is a constant, $p$ is the pixel coordinate, $K$ is the intrinsic matrix, $R$ is the rotation matrix, $t$ is the translation vector, and $P$ is the point in world coordinates.

Let:

$$x_1 = K_1^{-1} s_1 p_1$$

$$x_2 = K_2^{-1} s_2 p_2$$

Since $P$ is the same, we get:

$$R_1^{-1} (x_1 - t_1) = R_2^{-1} (x_2 - t_2)$$

$$R_2 R_1^{-1} (x_1 - t_1) = x_2 - t_2$$

$$R_2 R_1^{-1} x_1 - R_2 R_1^{-1} t_1 = x_2 - t_2$$

$$R_2 R_1^{-1} x_1 = (R_2 R_1^{-1} t_1 - t_2) + x_2$$

$$ (R_2 R_1^{-1} t_1 - t_2)^\wedge R_2 R_1^{-1} x_1 = (R_2 R_1^{-1} t_1 - t_2)^\wedge x_2$$

$$x_2^T (R_2 R_1^{-1} t_1 - t_2)^\wedge R_2 R_1^{-1} x_1 = 0$$

$$s_2 p_2^T K_2^{-T} (R_2 R_1^{-1} t_1 - t_2)^\wedge R_2 R_1^{-1} K_1^{-1} p_1 s_1 = 0$$

Thus, the fundamental matrix $F$ is given by:

$$F = K_2^{-T} (R_2 R_1^{-1} t_1 - t_2)^\wedge R_2 R_1^{-1} K_1^{-1}$$

$$p_2^T F p_1 = 0$$

where $F$ describes the transformation from image 1 to image 2, and $F^T$ describes the transformation from image 2 to image 1.

## 2. Epipolar Constraint:

$$p_2^T F p_1 = 0$$

This equation measures the closeness to zero when substituting points $p_2$ and $p_1$.

### Extension:

Proof: The cross product of two 2D homogeneous coordinates gives the parameters $(a, b, c)$ of the line equation passing through these two points $X_1$ and $X_2$ on the plane.

Let $X_1$ and $X_2$ be two points on the plane:

$$\text{Let } (X_1^\wedge X_2) = (a, b, c)$$

$$ (X_1^\wedge X_2) * X_1 = 0$$

$$ (X_1^\wedge X_2) * X_2 = 0$$

$$a x_1 + b y_1 + c = 0$$

$$a x_2 + b y_2 + c = 0$$

This equation has a unique solution

Since $X_1$ and $X_2$ multiplied by this line are zero, it shows that the line passes through $X_1$ and $X_2$

## 3. Lagrange Function
### (1) Definition
### Lagrange Function

Given a primal optimization problem:

$$
\text{minimize} \quad f(x)\\
\text{subject to} \quad h_i(x) = 0, \quad i = 1, \dots, m\\
g_j(x) \leq 0, \quad j = 1, \dots, p
$$

where $f(x)$ is the objective function, $h_i(x) = 0$ represents the equality constraints, and $g_j(x) \leq 0$ represents the inequality constraints.

The Lagrange function $L(x, \lambda, \nu)$ is defined as:

$$
L(x, \lambda, \nu) = f(x) + \sum_{i=1}^{m} \lambda_i h_i(x) + \sum_{j=1}^{p} \nu_j g_j(x)
$$

where $\lambda_i$ and $\nu_j$ are the Lagrange multipliers corresponding to the equality and inequality constraints, respectively.

### (2) Example
### Example of an Optimization Problem

Suppose we want to minimize an objective function $f(x_1, x_2) = x_1^2 + x_2^2$, subject to one equality constraint and one inequality constraint:

**Minimize:**

$$f(x_1, x_2) = x_1^2 + x_2^2$$

**Equality Constraint:**

$$h(x_1, x_2) = x_1 + x_2 - 1 = 0$$

**Inequality Constraint:**

$$g(x_1, x_2) = x_1 - x_2 \leq 1$$

**Constructing the Lagrange Function**

First, we introduce the Lagrange multiplier $\lambda$ corresponding to the equality constraint $h(x_1, x_2)$, and the Lagrange multiplier $\nu \geq 0$ corresponding to the inequality constraint $g(x_1, x_2)$. The Lagrange function $L(x_1, x_2, \lambda, \nu)$ can be written as:

$$L(x_1, x_2, \lambda, \nu) = x_1^2 + x_2^2 + \lambda(x_1 + x_2 - 1) + \nu(x_1 - x_2 - 1)$$

**Taking Partial Derivatives and Setting Them to Zero**

We need to take partial derivatives of the Lagrange function $L(x_1, x_2, \lambda, \nu)$ with respect to $x_1$, $x_2$, and $\lambda$, and set them to zero:

**For $x_1$:**

$$\frac{\partial L}{\partial x_1} = 2x_1 + \lambda + \nu = 0$$

**For $x_2$:**

$$\frac{\partial L}{\partial x_2} = 2x_2 + \lambda - \nu = 0$$

**For $\lambda$:**

$$\frac{\partial L}{\partial \lambda} = x_1 + x_2 - 1 = 0$$

**Solving the System of Equations**

We now have a system of equations:

$$2x_1 + \lambda + \nu = 0$$

$$2x_2 + \lambda - \nu = 0$$

$$x_1 + x_2 = 1$$

We can solve these equations by following these steps:

From the first and second equations, we can eliminate $\lambda$ and $\nu$, yielding:

$$2x_1 + \nu = -\lambda = 2x_2 - \nu$$

This implies $2x_1 + 2\nu = 2x_2$, which leads to $x_1 = x_2 - \nu$.

Substituting $x_1 = x_2 - \nu$ into the third equation $x_1 + x_2 = 1$, we get:

$$(x_2 - \nu) + x_2 = 1 \Rightarrow 2x_2 - \nu = 1$$

Since $\nu \geq 0$, we can solve for $x_2$ and $\nu$:

$$x_2 = \frac{1 + \nu}{2}$$

Substituting into $x_1 = x_2 - \nu$, we get:

$$x_1 = \frac{1 - \nu}{2}$$

**Checking the Inequality Constraint**

The inequality constraint $g(x_1, x_2) = x_1 - x_2 \leq 1$ must be satisfied. By calculating:

$$g(x_1, x_2) = \frac{1 - \nu}{2} - \frac{1 + \nu}{2} = -\nu$$

Since $\nu \geq 0$, the inequality is automatically satisfied.

**Final Solution**

Combining $x_1 = \frac{1 - \nu}{2}$, $x_2 = \frac{1 + \nu}{2}$, and the condition $\nu \geq 0$, the optimal solution occurs when $\nu = 0$, at which point:

$$x_1 = \frac{1}{2}, \quad x_2 = \frac{1}{2}, \quad \nu = 0$$

**Verification**

The value of the objective function is:

$$f(x_1, x_2) = \left(\frac{1}{2}\right)^2 + \left(\frac{1}{2}\right)^2 = \frac{1}{4} + \frac{1}{4} = \frac{1}{2}$$

The equality and inequality constraints are both satisfied, and $\nu = 0$.

This indicates that $(x_1, x_2) = \left(\frac{1}{2}, \frac{1}{2}\right)$ is the optimal solution to this optimization problem, satisfying all the constraints.

### (3) Why can we establish the equation like this?
When we find the extreme value, the gradient of $f(x)$ is linearly related to the gradient of the constraint condition, i.e.:

$$
\nabla f(x)=\lambda \nabla g(x) + \mu  \nabla h(x) \\
h(x)<0 \quad \mu>0
$$

Thus, the given conditions become:

$$
\text{min } f(x)\\
g(x)=0\\
h(x)<0\\
\mu >0
$$

Construct the Lagrange function:

$$
L(x,\lambda )=f(x)+\lambda g(x) +\mu h(x)
$$

The solving conditions are:

$$
\frac {\nabla L(x,\lambda )}{\nabla x}=0 \quad \text{(The linearly related gradients cancel each other out)}\\
\frac {\nabla L(x,\lambda )}{\nabla \lambda}=0 \quad \text{(Ensuring the constraint condition)}
$$

The reason $\mu > 0$ is as follows: On the side of $g(x) \leq 0$, the gradient of $g(x)$ points towards the side greater than $0$, which is the infeasible side. The problem we are solving is a minimization problem, so the gradient of $f(x)$ at the intersection points towards the feasible side. In other words, the two gradients must be opposite, hence $\mu > 0$.

### (4) Lagrangian Dual Solution
The Lagrangian dual function is the function obtained by taking the Lagrangian function $\lambda,\mu$ as constants and minimizing with respect to $x$:

$$
g(\lambda, \mu )= \min_{x}(L(x,λ,\mu))\\
L(x,λ,\mu)=f(x)+λ^Th(x)+\mu^Tg(x)\\
h(x)=0, \quad g(x) \mu <= 0\\
\therefore \text{min }f(x)\geq \text{min }L(x,λ,\mu)\geq g(\lambda, \mu )
$$

The problem of solving $\text{min }f(x)$ turns into solving the maximum of $g(\lambda, \mu)$.

The above example can also be solved using the dual approach:

**Construct the Lagrange function:**

$$
L(x_1,x_2,\lambda,\mu)=x_1^2+x_2^2+\lambda(x_1+x_2-1)+\mu(x_1-x_2-1)
$$

**Take the derivative with respect to $x$:**

$$
\begin{cases}
\frac{\nabla L(x_1,x_2,\lambda,\mu)}{\nabla x_1}=2x_1+\lambda+\mu=0\\
\frac{\nabla L(x_1,x_2,\lambda,\mu)}{\nabla x_2}=2x_2+\lambda-\mu=0\\
\end{cases}
$$

$$
\begin{cases}
x_1=-\frac{\lambda+\mu}{2}\\
x_2=\frac{\mu-\lambda}{2}\\
\end{cases}
$$

We get:

$$
g(\lambda,\mu)=-\frac{\lambda^2+\mu^2}{2}-\lambda-\mu
$$

**Take the derivative with respect to $\lambda$ and $\mu$ and set it to zero to find the maximum value:**

We differentiate the Lagrangian function with respect to $\lambda$ and $\mu$ and set it to zero:

$$
\begin{cases}
\frac{\partial}{\partial \lambda}\left(-\lambda^2 + \frac{\mu^2}{2} - \lambda - \mu\right) = -\lambda - 1 = 0 \quad \Rightarrow \quad \lambda = -1\\
\frac{\partial}{\partial \mu}\left(-\lambda^2 + \frac{\mu^2}{2} - \lambda - \mu\right) = -\mu - 1 = 0 \quad \Rightarrow \quad \mu = -1
\end{cases}
$$

Since $\lambda \geq 0$ and $\mu \geq 0$, we get $\lambda = 0$ and $\mu = 0$.

**Substitute $\lambda = 0$ and $\mu = 0$:**

$$
g(0,0) = -0^2 + \frac{0^2}{2} - 0 - 0 = 0
$$

**Result of the Dual Problem:**

The maximum value of the dual problem is $0$, which is a lower bound on the optimal solution of the original problem. By solving the original problem, we can get the optimal solution $x_1 = \frac{1}{\sqrt{2}}$ and $x_2 = \frac{1}{\sqrt{2}}$, with the objective function value:

$$
f^* = x_1^2 + x_2^2 = \frac{1}{4} + \frac{1}{4} = \frac{1}{2}
$$

This value is greater than the solution of the dual problem $0$, indicating that the dual problem does not provide an exact lower bound for the optimal solution of the original problem, but it still provides a lower bound.

## 4. Constrained Optimization Problem
### (1) Convex Function
A real-valued function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is called a convex function if for all $x_1, x_2 \in \mathbb{R}^n$ and all $\theta \in [0,1]$, the following holds:

$$
f(\theta x_1 + (1 - \theta) x_2) \leq \theta f(x_1) + (1 - \theta) f(x_2)
$$

How to determine if a function is convex?
- For a univariate function $f(x)$, we can determine this by checking the sign of its second derivative $f''(x)$. If the second derivative is always non-negative, i.e., $f''(x) \geq 0$, then $f(x)$ is a convex function.
- For a multivariate function $f(X)$, we can determine this by checking the positive definiteness of its Hessian matrix. The Hessian matrix is a square matrix of second-order partial derivatives of the multivariate function. If the Hessian matrix is positive semi-definite, then $f(X)$ is a convex function.

Why do we require the function to be convex?
- A convex function has a global optimal solution.

### (2) Dual Ascent Method
### Dual Ascent Method:

Consider the following optimization problem:

$$
\text{minimize} \ f(x) \quad \text{s.t.} \quad Ax = b \quad \text{(1)}
$$

Its Lagrangian form is:

$$
L(x, \lambda) = f(x) + \lambda^{T}(Ax - b) \quad \text{(2)}
$$

The dual form is:

$$
g(\lambda) = \inf_{x} L(x, \lambda) = -f^{*}(-A^{T}\lambda) - b^{T}\lambda \quad \text{(3)}
$$

where $f^{*}$ is the conjugate function of $f$.

The dual problem is:

$$
\text{maximize} \ g(\lambda) \quad \text{(4)}
$$

The iterative update of the dual ascent method is:

**x-minimization, i.e., constructing $\min_{x}g(\lambda)$:**

$$
x^{k+1} = \text{argmin}_{x} L(x, \lambda^{k}) \quad \text{(5)}
$$

**Dual variable update, then maximize $g(\lambda)$:**

$$
\lambda^{k+1} = \lambda^{k} + \alpha^{k}(Ax^{k+1} - b) \quad \text{(6)}
$$

where $\alpha^{k} > 0$ is the step size.
Sure! Here's the translation of your text into English:

### Example:

**Optimization Problem:**

$$
\text{minimize}_{x_1, x_2} \quad f(x_1, x_2) = 2(x_1 - 1)^2 + (x_2 + 2)^2
$$
subject to:
$$
\begin{cases}
x_1 \geq 2 \\
x_2 \geq 0
\end{cases}
$$

**Lagrangian Function:**

$$
L(x_1, x_2, \lambda_1, \lambda_2) = 2(x_1 - 1)^2 + (x_2 + 2)^2 + \lambda_1(2 - x_1) + \lambda_2(-x_2)
$$

- **Analytical Solution for Decision Variables $x_1, x_2$:**

$$
\frac{\partial L(x_1, x_2, \lambda_1, \lambda_2)}{\partial x_1} = 0
$$
$$
\frac{\partial L(x_1, x_2, \lambda_1, \lambda_2)}{\partial x_2} = 0
$$

Solving these equations, we get:

$$
x_1^* = \frac{\lambda_1}{4} + 1
$$
$$
x_2^* = \frac{\lambda_2}{2} - 2
$$

- **Iterative Solution for Decision Variables $x_1, x_2$:**
$$
\frac{\partial L(x_1, x_2, \lambda_1, \lambda_2)}{\partial x_1} = 4(x_1-1)-\lambda_1
$$
$$
\frac{\partial L(x_1, x_2, \lambda_1, \lambda_2)}{\partial x_2} = 2(x_2+2)-\lambda_2
$$

Solving these equations, we get:

$$
x_1^{k+1} = x_1^k - t_x[4(x_1-1) - \lambda_1]
$$
$$
x_2^{k+1} = x_2^k - t_x[2(x_2+2) - \lambda_2]
$$

The negative sign is because we are moving in the direction of the negative gradient, which will gradually reduce the value of the Lagrangian function $L(x, \lambda)$.

**Lagrangian Function Update Formulas:**

For the $(k+1)$-th iteration:

$$
\lambda_1^{k+1} = \lambda_1^k + t_\lambda (2 - x_1)
$$
$$
\lambda_2^{k+1} = \lambda_2^k + t_\lambda (-x_2)
$$

The positive sign is because we are maximizing the dual function.

Termination Conditions:
- Dual gap: Iteration stops when $f(x) - L(x_1, x_2, \lambda_1, \lambda_2) < \epsilon$
- Dual function change: Iteration stops when the change in the dual function $g(\lambda)$ between consecutive iterations is less than a preset threshold $\epsilon$, indicating convergence.
- Change in Lagrange multipliers: Iteration stops when the change in Lagrange multipliers is less than a certain threshold $\epsilon$.

### (3) Augmented Lagrangian
To increase the robustness of the dual ascent method and relax the strong convex constraints of the function $f(x)$, we introduce the Augmented Lagrangian (AL) form:
$$
L_{\rho}(x, \lambda) = f(x) + \lambda^{T}(Ax - b) + \frac{\rho}{2} \|Ax - b\|_{2}^{2}
$$
where the penalty factor $\rho > 0$.
Penalty function: If constraints are satisfied, there is no effect, but if constraints are not satisfied, a penalty will be applied.

### (4) ADMM
**Problem Model**

The Alternating Direction Method of Multipliers (ADMM) is often used to solve equality-constrained optimization problems involving two variables. Its general form is:

$$
\min_{x,z} f(x) + g(z) \quad \text{s.t.} \quad Ax + Bz = c
$$

where $x \in \mathbb{R}^n$, $z \in \mathbb{R}^m$ are optimization variables, and in the equality constraint $A \in \mathbb{R}^{p \times n}$, $B \in \mathbb{R}^{p \times m}$, $c \in \mathbb{R}^p$, and both $f$ and $g$ are convex functions.

The standard ADMM algorithm solves an equality-constrained problem where the two functions $f$ and $g$ are additive. This means that the two functions are parts of the overall optimization, contributing differently but simply added together.

The core of the ADMM algorithm is the Augmented Lagrangian Method (ALM) for primal-dual problems. The Lagrangian function solves an optimization problem with $n$ variables and $k$ constraints. The Augmented Lagrangian Method (ALM) includes a penalty term to accelerate convergence.

$$
L_{\rho}(x, z, u) = f(x) + g(z) + u^T(Ax + Bz - c) + \frac{\rho}{2}\|Ax + Bz - c\|^2_2
$$

The Augmented Lagrangian function adds a penalty term related to the constraint conditions to the original problem's Lagrangian function, with $\rho > 0$ as a parameter influencing iteration efficiency. The function for minimizing is + dual term and penalty term, for maximizing is - dual term and penalty term.

**Algorithm Process**

Each step updates one variable while fixing the other two, and repeats this alternation.

- Step 1: Update $x$:
  $$
  x^{(i)} = \mathrm{argmin}_x \, L_{\rho}(x, z^{(i-1)}, u^{(i-1)})
  $$

- Step 2: Update $z$:
  $$
  z^{(i)} = \mathrm{argmin}_z \, L_{\rho}(x^{(i)}, z, u^{(i-1)})
  $$

- Step 3: Update $u$:
  $$
  u^{(i)} = u^{(i-1)} + \rho (Ax^{(i)} + Bz^{(i)} - c)
  $$

## 5. Linear Algebra
### (1) Rank
Why is $rank(A)$ a non-convex problem?
Consider the matrix set $\{P : \text{rank}(P) \leq k\}$; this is a non-convex set. Suppose two matrices $P_1$ and $P_2$ have ranks $k_1$ and $k_2$ respectively (assuming $k_1 = k_2 = k$). Then, $\lambda P_1 + (1 - \lambda) P_2$ (where $0 \leq \lambda \leq 1$) may have a rank greater than $k$. This means the rank minimization problem does not have the properties of a convex set.

### (2) Nuclear Norm
The nuclear norm, also known as the trace norm, is a convex function. It is defined as the sum of all singular values of a matrix.

### (3) Frobenius Norm
The Frobenius norm is a measure of matrix size. It is defined as the square root of the sum of the squares of all elements in the matrix. For a matrix $A \in \mathbb{R}^{m \times n}$, its Frobenius norm is defined as:

$$
\|A\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij}|^2}
$$

where $a_{ij}$ represents the element in the $i$-th row and $j$-th column of matrix $A$.

### (4) Singular Value Thresholding (SVT)
Singular Value Thresholding (SVT) is used to handle nuclear norm minimization problems. Specifically, SVT applies "soft thresholding" to the singular values of a matrix, i.e., subtracting a fixed threshold from each singular value (setting to zero if below the threshold), to obtain a low-rank approximation matrix. This processing can effectively approximate the solution to nuclear norm minimization.

**Example:**
Suppose we have a partially observed matrix $M$, with observed data at positions in matrix $\Omega$. We want to complete the matrix by minimizing its nuclear norm, i.e., finding a matrix $X$ such that the nuclear norm of $X$ is minimized and $X$ is consistent with $M$ at $\Omega$.

This problem can be formalized as:

$$
\min_X \|X\|_* \\
\text{subject to} \quad X_\Omega = M_\Omega
$$

where $\|X\|_*$ denotes the nuclear norm of matrix $X$, and $X_\Omega = M_\Omega$ ensures consistency of $X$ with $M$ at $\Omega$.

### Solving Using ADMM

#### Introducing Auxiliary Variables

To use ADMM, we introduce

 an auxiliary matrix variable $Z$ to represent the matrix $X$.

$$
\min_{X, Z} \|Z\|_* \\
\text{subject to} \quad X = Z \\
\text{and} \quad X_\Omega = M_\Omega
$$

#### Augmented Lagrangian Form

The Augmented Lagrangian for this problem is:

$$
L(X, Z, U) = \|Z\|_* + \langle U, X - Z \rangle + \frac{\rho}{2} \|X - Z\|_F^2
$$

where $U$ is the dual variable associated with the equality constraint $X = Z$.

#### Iterative Update

1. **Update $X$:** Solve for $X$ by minimizing the Augmented Lagrangian with respect to $X$ while holding $Z$ and $U$ fixed. This involves solving:

$$
X^{(k+1)} = \text{argmin}_X \, \|Z^{(k)}\|_* + \langle U^{(k)}, X - Z^{(k)} \rangle + \frac{\rho}{2} \|X - Z^{(k)}\|_F^2
$$

2. **Update $Z$:** Solve for $Z$ by minimizing the Augmented Lagrangian with respect to $Z$ while holding $X$ and $U$ fixed. This involves solving:

$$
Z^{(k+1)} = \text{argmin}_Z \, \|Z\|_* + \langle U^{(k)}, X^{(k+1)} - Z \rangle + \frac{\rho}{2} \|X^{(k+1)} - Z\|_F^2
$$

3. **Update $U$:** Update the dual variable $U$ by:

$$
U^{(k+1)} = U^{(k)} + \rho (X^{(k+1)} - Z^{(k+1)})
$$

#### Termination Conditions

The iterations stop when the primal and dual residuals are sufficiently small, indicating convergence. The residuals measure the satisfaction of the constraints and the optimality of the solution, respectively.

**Termination Conditions:**
1. Primal residual: $\|X^{(k+1)} - Z^{(k+1)}\|_F$
2. Dual residual: $\rho \|Z^{(k+1)} - Z^{(k)}\|_F$

When these residuals are smaller than predefined thresholds, the algorithm is considered to have converged.

I hope this translation captures the essence of your content! If you have more details or further questions, feel free to ask.
