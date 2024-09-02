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
