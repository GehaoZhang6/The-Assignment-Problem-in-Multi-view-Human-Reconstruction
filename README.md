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


