# Ⅰ.数学推导
## 1.基础矩阵F:  
已知：
$$
s_1p_1=K_1(R_1P+t_1)\\
s_2p_2=K_2(R_2P+t_2)
$$
$s$为常数，$p$为像素坐标，$K$为内参，$R$为旋转矩阵，$t$为平移矩阵，$P$为世界坐标的点。

设：
$$
x_1=K_1^{-1}s_1p_1\\
x_2=K_2^{-1}s_2p_2
$$
由于$P$相同，得：
$$
R_1^{-1}(x_1-t_1)=R_2^{-1}(x_2-t_2)\\
R_2R_1^{-1}(x_1-t_1)=x_2-t_2\\
R_2R_1^{-1}x_1-R_2R_1^{-1}t_1=x_2-t_2\\
R_2R_1^{-1}x_1=(R_2R_1^{-1}t_1-t_2)+x_2\\
(R_2R_1^{-1}t_1-t_2)^\wedge R_2R_1^{-1}x_1=(R_2R_1^{-1}t_1-t_2)^\wedge x_2\\
x_2^T(R_2R_1^{-1}t_1-t_2)^\wedge R_2R_1^{-1}x_1=0\\
s_2p_2^TK_2^{-T}(R_2R_1^{-1}t_1-t_2)^\wedge R_2R_1^{-1}K_1^{-1}p_1s_1=0
$$
得到基础矩阵为:
$$
F=K_2^{-T}(R_2R_1^{-1}t_1-t_2)^\wedge R_2R_1^{-1}K_1^{-1}
$$

$$p_2^TFp_1=0$$
$$F$$描述了图像1到图像2的过程，$F^T$描述了图像2到图像1的过程
## 2.极线约束:
$$
p_2^TFp_1=0
$$
上式带入点$p_2,p_1$后与0的接近程度
### 扩展：
证明：两个二维的齐次坐标叉乘为经过这两个点的直线方程的参数$(a,b,c)$
$X_1,X_2$为平面上两点
$$
\text{设}(X_1^\wedge X_2)=(a,b,c)\\
$$
$$
\begin{cases}
(X_1^\wedge X_2)*X_1=0\\
(X_1^\wedge X_2)*X_2=0\\
\end{cases}
$$

$$
\begin{cases}
ax_1+bx_1+c=0\\
ax_2+bx_2+c=0\\
\end{cases}
$$

$$
此方程有唯一解\\
X_1与X_2与此直线相乘都为0说明该直线通过X_1与X_2
$$
## 3.拉格朗日函数
### (1) 定义
### 拉格朗日函数

给定一个原始优化问题：

$$
\text{minimize} \quad f(x)\\
\text{subject to} \quad h_i(x) = 0, \quad i = 1, \dots, m\\

g_j(x) \leq 0, \quad j = 1, \dots, p
$$

其中，$ f(x) $ 是目标函数，$ h_i(x) = 0 $ 是等式约束，$ g_j(x) \leq 0 $ 是不等式约束。

拉格朗日函数 $ L(x, \lambda, \nu) $ 被定义为：

$$
L(x, \lambda, \nu) = f(x) + \sum_{i=1}^{m} \lambda_i h_i(x) + \sum_{j=1}^{p} \nu_j g_j(x)
$$

其中，$ \lambda_i $ 和 $ \nu_j $ 是拉格朗日乘子，分别对应于等式约束和不等式约束。

### (2)例子
### 优化问题的例子

假设我们要最小化一个目标函数 $f(x_1, x_2) = x_1^2 + x_2^2$，并且它受到一个等式约束和一个不等式约束：

**minimize:**
$$
f(x_1, x_2) = x_1^2 + x_2^2
$$

**等式约束:**
$$
h(x_1, x_2) = x_1 + x_2 - 1 = 0
$$

**不等式约束:**
$$
g(x_1, x_2) = x_1 - x_2 \leq 1
$$

**构造拉格朗日函数**

首先，我们引入拉格朗日乘子 $\lambda$ 对应于等式约束 $h(x_1, x_2)$，以及拉格朗日乘子 $\nu \geq 0$ 对应于不等式约束 $g(x_1, x_2)$。拉格朗日函数 $L(x_1, x_2, \lambda, \nu)$ 可以写成：

$$
L(x_1, x_2, \lambda, \nu) = x_1^2 + x_2^2 + \lambda(x_1 + x_2 - 1) + \nu(x_1 - x_2 - 1)
$$

**求偏导数并设置为零**

我们需要对拉格朗日函数 $L(x_1, x_2, \lambda, \nu)$ 分别对 $x_1$、$x_2$ 以及 $\lambda$ 求偏导数，并设置为零：

**对于 $x_1$:**
$$
\frac{\partial L}{\partial x_1} = 2x_1 + \lambda + \nu = 0
$$

**对于 $x_2$:**
$$
\frac{\partial L}{\partial x_2} = 2x_2 + \lambda - \nu = 0
$$

**对于 $\lambda$:**
$$
\frac{\partial L}{\partial \lambda} = x_1 + x_2 - 1 = 0
$$

**解方程组**

我们现在得到了一个方程组：

$$
2x_1 + \lambda + \nu = 0
$$

$$
2x_2 + \lambda - \nu = 0
$$

$$
x_1 + x_2 = 1
$$

我们可以通过以下步骤求解这些方程：

从第一个和第二个方程中，我们可以消去 $\lambda$ 和 $\nu$，得到：

$$
2x_1 + \nu = -\lambda = 2x_2 - \nu
$$

这意味着 $2x_1 + 2\nu = 2x_2$，即 $x_1 = x_2 - \nu$。

将 $x_1 = x_2 - \nu$ 代入第三个等式约束 $x_1 + x_2 = 1$，得到：

$$
(x_2 - \nu) + x_2 = 1 \Rightarrow 2x_2 - \nu = 1
$$

由于 $\nu \geq 0$，我们可以解出 $x_2$ 和 $\nu$：

$$
x_2 = \frac{1 + \nu}{2}
$$

代入 $x_1 = x_2 - \nu$ 中，得到：

$$
x_1 = \frac{1 - \nu}{2}
$$

**不等式约束的检查**

不等式约束 $g(x_1, x_2) = x_1 - x_2 \leq 1$ 必须得到满足。通过计算：

$$
g(x_1, x_2) = \frac{1 - \nu}{2} - \frac{1 + \nu}{2} = -\nu
$$

由于 $\nu \geq 0$，该不等式自动满足。

**最终解**

结合 $x_1 = \frac{1 - \nu}{2}$、$x_2 = \frac{1 + \nu}{2}$ 和 $\nu \geq 0$ 的条件，问题的最优解发生在 $\nu = 0$ 时，此时：

$$
x_1 = \frac{1}{2}, \quad x_2 = \frac{1}{2}, \quad \nu = 0
$$

**验证**

目标函数的值为：

$$
f(x_1, x_2) = \left(\frac{1}{2}\right)^2 + \left(\frac{1}{2}\right)^2 = \frac{1}{4} + \frac{1}{4} = \frac{1}{2}
$$

等式约束和不等式约束均得到了满足，且 $\nu = 0$。

这表明 $(x_1, x_2) = \left(\frac{1}{2}, \frac{1}{2}\right)$ 是这个优化问题的最优解，且满足所有约束条件。

### (3)为什么可以这样建立方程？
当求取到极值的时候，$f(x)$的梯度与限制条件的梯度是线性相关的，即：
$$
\nabla f(x)=\lambda \nabla g(x) + \mu  \nabla h(x) \\
h(x)<0 \quad \mu>0
$$
所以，给定条件变为：
$$
\text{min } f(x)\\
g(x)=0\\
h(x)<0\\
\mu >0
$$
建立拉格朗日函数：
$$
L(x,\lambda )=f(x)+\lambda g(x) +\mu h(x)
$$
求解条件为：
$$
        \frac {\nabla L(x,\lambda )}{\nabla x}=0 \text{ 线性相关的梯度相抵消}\\
\frac {\nabla L(x,\lambda )}{\nabla \lambda}=0\text{保证约束条件}\\
$$
$\mu >0$原因如下：在 $g(x)≤0$ 一侧， $g(x)$的梯度是指向大于 $0 $的一侧,也就是不是可行域的一侧。而求的问题是极小值，所以 $f(x)$在交点处的梯度是指向可行域的一侧，也就是说两个梯度一定是相反的，故有 ，$\mu >0$.
### (4)拉格朗日对偶求解
拉格朗日对偶函数为拉格朗日函数$\lambda,\mu$当作常数，关于$x$取最小值得到的函数：                
$$
g(\lambda, \mu )= \min_{x}(L(x,λ,\mu))\\
L(x,λ,\mu)=f(x)+λ^Th(x)+\mu^Tg(x)\\
h(x)=0,g(x)\mu<=0\\
\therefore \text{min }f(x)\geq \text{min }L(x,λ,\mu)\geq g(\lambda, \mu )
$$
求解$\text{min }f(x)$变为求解$g(\lambda, \mu)$的最大值问题
上面的例子也可以用对偶求解：
**构建拉格朗日函数**
$$
L(x_1,x_2,\lambda,\mu)=x_1^2+x_2^2+\lambda(x_1+x_2-1)+\mu(x_1-x_2-1)
$$
**对$x$求导**
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
得
$$
g(\lambda,\mu)=-\frac{\lambda^2+\mu^2}{2}-\lambda-\mu
$$
**对 $\lambda$ 和 $\mu$ 进行求导并设为零，找到最大值**

我们对拉格朗日函数进行对 $\lambda$ 和 $\mu$ 的求导，并设为零：

$$
\begin{cases}
\frac{\partial}{\partial \lambda}\left(-\lambda^2 + \frac{\mu^2}{2} - \lambda - \mu\right) = -\lambda - 1 = 0 \quad \Rightarrow \quad \lambda = -1\\
\frac{\partial}{\partial \mu}\left(-\lambda^2 + \frac{\mu^2}{2} - \lambda - \mu\right) = -\mu - 1 = 0 \quad \Rightarrow \quad \mu = -1
\end{cases}
$$

由于 $\lambda \geq 0$ 和 $\mu \geq 0$，所以 $\lambda = 0$ 和 $\mu = 0$。

**代入 $\lambda = 0$ 和 $\mu = 0$ 中：**

$$
g(0,0) = -0^2 + \frac{0^2}{2} - 0 - 0 = 0
$$

**对偶问题的结果**

对偶问题的最大值是 $0$，这是原始问题最优解的下界。通过求解原始问题，我们可以得到最优解 $x_1 = \frac{1}{\sqrt{2}}$ 和 $x_2 = \frac{1}{\sqrt{2}}$，目标函数值为：

$$
f^* = x_1^2 + x_2^2 = \frac{1}{4} + \frac{1}{4} = \frac{1}{2}
$$
这个值大于对偶问题的解 $0$，说明对偶问题没有提供原问题最优解的精确下界，但仍然提供了一个下界。

## 4.约束优化问题
### (1)凸函数
一个实值函数 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ 被称为凸函数，如果对于所有的 $x_1, x_2 \in \mathbb{R}^n$ 和所有的 $\theta \in [0,1]$，都有：

$$
f(\theta x_1 + (1 - \theta) x_2) \leq \theta f(x_1) + (1 - \theta) f(x_2)
$$

如何判断函数是否为凸函数？
- 对于一元函数 $f(x)$，我们可以通过其二阶导数 $f''(x)$ 的符号来判断。如果函数的二阶导数总是非负，即 $f''(x) \geq 0$，则 $f(x)$ 是凸函数。
- 对于多元函数 $f(X)$，我们可以通过其 Hessian 矩阵的正定性来判断。Hessian 矩阵是由多元函数的二阶导数组成的方阵。如果 Hessian 矩阵是半正定矩阵，则 $f(X)$ 是凸函数。

为什么要求是凸函数呢？
- 凸函数有全局最优解

### (2)对偶上升法
### 对偶上升法:

设有如下优化问题：

$$
\text{minimize} \ f(x) \quad \text{s.t.} \quad Ax = b \quad \text{(1)}
$$

它的拉格朗日形式为：

$$
L(x, \lambda) = f(x) + \lambda^{T}(Ax - b) \quad \text{(2)}
$$

对偶形式为：

$$
g(\lambda) = \inf_{x} L(x, \lambda) = -f^{*}(-A^{T}\lambda) - b^{T}\lambda \quad \text{(3)}
$$

其中 $ f^{*} $ 是 $ f $ 的共轭函数。

对偶问题为：

$$
\text{maximize} \ g(\lambda) \quad \text{(4)}
$$

对偶上升法的迭代更新为：

**x-最小化,即构建出来$\min_{x}g(\lambda)$**：

$$
x^{k+1} = \text{argmin}_{x} L(x, \lambda^{k}) \quad \text{(5)}
$$

**对偶变量更新,再去最大化$g(\lambda)$**：

$$
\lambda^{k+1} = \lambda^{k} + \alpha^{k}(Ax^{k+1} - b) \quad \text{(6)}
$$

其中 $ \alpha^{k} > 0 $ 是步长。  

### 举例：
**优化问题为：**
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

**拉格朗日函数为：**

$$
L(x_1, x_2, \lambda_1, \lambda_2) = 2(x_1 - 1)^2 + (x_2 + 2)^2 + \lambda_1(2 - x_1) + \lambda_2(-x_2)
$$

- **决策变量 $x_1, x_2$ 的解析解为：**

$$
\frac{\partial L(x_1, x_2, \lambda_1, \lambda_2)}{\partial x_1} = 0
$$
$$
\frac{\partial L(x_1, x_2, \lambda_1, \lambda_2)}{\partial x_2} = 0
$$

求解上述方程，解得：

$$
x_1^* = \frac{\lambda_1}{4} + 1
$$
$$
x_2^* = \frac{\lambda_2}{2} - 2
$$
- **决策变量 $x_1, x_2$ 的迭代解为：**
$$
\frac{\partial L(x_1, x_2, \lambda_1, \lambda_2)}{\partial x_1} = 4(x_1-1)-\lambda_1
$$
$$
\frac{\partial L(x_1, x_2, \lambda_1, \lambda_2)}{\partial x_2} = 2(x_2+2)-\lambda_2
$$

求解上述方程，解得：

$$
x_1^{k+1} = x_1^k-t_x[4(x_1-1)-\lambda_1]
$$
$$
x_2^{k+1} = x_2^k-t_x[2(x_2+2)-\lambda_2]
$$
负号是因为负梯度的方向前进，这会逐步减少拉格朗日函数$L(x,λ)$的值。
**拉格朗日函数更新公式：**

第 $k+1$ 轮迭代更新公式：

$$
\lambda_1^{k+1} = \lambda_1^k + t_\lambda(2 - x_1)
$$
$$
\lambda_2^{k+1} = \lambda_2^k + t_\lambda(-x_2)
$$
正号是因为求最大值
迭代终止条件：
- 对偶间隙：当$f(x)-L(x_1, x_2, \lambda_1, \lambda_2)<e$时迭代结束
- 对偶函数变化量：当对偶函数 g(λ) 在相邻迭代间的变化量小于预设的阈值 ϵ，可以认为算法已经收敛。
- 拉格朗日乘子变化量：当拉格朗日乘子的更新量小于某个阈值 ϵ，也可以作为终止算法的条件。
### (3)增广拉格朗日
为了增加对偶上升法的鲁棒性和放松函数 $ f(x) $ 的强凸约束，我们引入增广拉格朗日 (Augmented Lagrangians) 形式：
$$
L_{\rho}(x, \lambda) = f(x) + \lambda^{T}(Ax - b) + \frac{\rho}{2} \|Ax - b\|_{2}^{2}
$$
其中惩罚因子 $ \rho > 0 $。
罚函数:如果满足约束条件则无影响，但是如果没有满足约束条件，则会施加惩罚。
### (4)ADMM
**问题模型**

交替方向乘子法（ADMM）通常用于解决存在两个优化变量的等式约束优化类问题，其一般形式为：

$$
\min_{x,z} f(x) + g(z) \quad \text{s.t.} \quad Ax + Bz = c
$$

其中 $x \in \mathbb{R}^n, z \in \mathbb{R}^m$ 为优化变量，等式约束中 $A \in \mathbb{R}^{p \times n}, B \in \mathbb{R}^{p \times m}, c \in \mathbb{R}^p$，目标函数中 $f, g$ 都是凸函数。

标准的 ADMM 算法解决的是一个等式约束的问题，且该问题两个函数 $f$ 和 $g$ 是线性加法的关系。这意味着两者实际上是整体优化的两个部分，两者的资源占用符合一定等式，对整体优化贡献不同，但是是简单加在一起的。



ADMM 算法的核心是原始对偶算法的增广拉格朗日法（ALM）。拉格朗日函数是解决了多个约束条件下的优化问题，这种方法可以求解一个有 $n$ 个变量与 $k$ 个约束条件的优化问题。原始对偶方法中的增广拉格朗日法（Augmented Lagrangian）是加了惩罚项的拉格朗日法，目的是使得算法收敛的速度更快。

$$
L_{\rho}(x,z,u) = f(x) + g(z) + u^T(Ax + Bz - c) + \frac{\rho}{2}\|Ax + Bz - c\|^2_2
$$

增广拉格朗日函数就是在关于原问题的拉格朗日函数之后增加了一个和约束条件有关的惩罚项，惩罚项参数 $\rho > 0$。惩罚项参数影响迭代效率。增广拉格朗日函数对 min 是 + 对偶项和惩罚项，对 max 是 - 对偶项和惩罚项。

原问题 $\min_{x,z} f(x) + g(z)$，对偶问题 $\max_{u} \min_{x,z} L_{\rho}(x,z,u)$，两个问题的最优解等价，并且没有了约束条件。

**算法流程**

每一步只更新一个变量而固定另外两个变量，如此交替重复更新。

- Step 1: 更新 $x$：
  $$
  x^{(i)} = \mathrm{argmin}_x \, L_{\rho}(x,z^{(i-1)},u^{(i-1)})
  $$

- Step 2: 更新 $z$：
  $$
  z^{(i)} = \mathrm{argmin}_z \, L_{\rho}(x^{(i)},z,u^{(i-1)})
  $$

- Step 3: 更新 $u$：
  $$
  u^{(i)} = u^{(i-1)} + \rho (Ax^{(i)} + Bz^{(i)} - c)
  $$
## 5.线性代数
### (1)秩
为什么$rank(A)$是一个非凸问题？
考虑矩阵集合 $\{P : \text{rank}(P) \leq k\}$，这是一个非凸集。设想两个矩阵 $P_1$ 和 $P_2$，它们的秩分别是 $k_1$ 和 $k_2$（假设 $k_1 = k_2 = k$）。那么，$\lambda P_1 + (1 - \lambda) P_2$（其中 $0 \leq \lambda \leq 1$）的秩可能大于 $k$。这意味着秩最小化问题不具有凸集合的性质。
### (2)核范数
核范数，又称为迹范数，是一个凸函数。它被定义为一个矩阵的所有奇异值之和。
### (3)Frobenius范数
Frobenius范数（Frobenius norm）是一种用于衡量矩阵大小的范数。它是矩阵中所有元素的平方和的平方根。对于一个矩阵 $A \in \mathbb{R}^{m \times n}$，其Frobenius范数定义为：

$$
\|A\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij}|^2}
$$

其中 $a_{ij}$ 表示矩阵 $A$ 的第 $i$ 行第 $j$ 列的元素。
### (4)奇异值阈值算法
奇异值阈值处理（SVT）的作用：
在解决核范数最小化问题时，我们需要对矩阵的奇异值进行阈值化处理，这就是奇异值阈值处理（SVT）。具体来说，SVT操作会对矩阵的奇异值进行“软阈值化”（soft thresholding），即对每个奇异值减去一个固定的阈值（如果奇异值小于阈值则置零），从而得到一个低秩近似矩阵。这种处理方式可以有效地逼近核范数最小化的解。
**例子：**
假设我们有一个部分观察到的矩阵 $M$，其观测数据位于矩阵 $\Omega$ 的位置。我们希望通过最小化矩阵的核范数来完成这个矩阵，即找到一个矩阵 $X$，使得 $X$ 的核范数最小，并且 $X$ 在 $\Omega$ 处与 $M$ 保持一致。

这个问题可以形式化为以下优化问题：

$$
\min_X \|X\|_* \\
\text{subject to} \quad X_\Omega = M_\Omega
$$

其中，$\|X\|_*$ 表示矩阵 $X$ 的核范数，即其所有奇异值的和，$X_\Omega = M_\Omega$ 表示 $X$ 在 $\Omega$ 上与 $M$ 一致。

### 使用ADMM求解

#### 引入辅助变量

为了使用ADMM，我们引入一个辅助变量 $Z$，问题变为：

$$
\min_{X, Z} \|Z\|_* \\
\text{subject to} \quad X_\Omega = M_\Omega, \quad X = Z
$$

#### 构建拉格朗日函数

采用拉格朗日乘数法，我们引入乘子 $Y$ 并构造拉格朗日函数：

$$
L(X, Z, Y) = \|Z\|_* + \frac{\rho}{2} \|X - Z + Y\|_F^2 + \frac{\lambda}{2} \|X_\Omega - M_\Omega\|_F^2
$$

其中 $\rho$ 是ADMM中的一个调节参数，$\lambda$ 是约束条件的惩罚系数，$\|\cdot\|_F$ 表示矩阵的Frobenius范数。

#### ADMM迭代步骤

ADMM通过交替最小化拉格朗日函数来更新变量 $X$、$Z$ 和 $Y$。每一步的更新公式如下：

**更新 $X$：**

$$
X^{k+1} = \arg \min_X \left( \frac{\rho}{2} \|X - Z^k + Y^k\|_F^2 + \frac{\lambda}{2} \|X_\Omega - M_\Omega\|_F^2 \right)
$$

这个更新公式是一个带约束的最小二乘问题，可以通过简单的代数运算得到。

<h3 style="color: red;">更新Z（使用奇异值阈值处理）：</h3>

$$
Z^{k+1} = \arg \min_Z \left( \|Z\|_* + \frac{\rho}{2} \|X^{k+1} - Z + Y^k\|_F^2 \right)
$$

这个更新步骤中，使用了奇异值阈值处理（SVT）来更新矩阵 $Z$。具体地，设矩阵 $X^{k+1} + Y^k$ 的奇异值分解为 $U \Sigma V^T$，则：

$$
Z^{k+1} = U \mathcal{S}_{\frac{1}{\rho}}(\Sigma) V^T
$$

其中 $\mathcal{S}_{\frac{1}{\rho}}(\Sigma)$ 是对奇异值进行软阈值处理，即：

$$
\mathcal{S}_{\frac{1}{\rho}}(\sigma_i) = \max(\sigma_i - \frac{1}{\rho}, 0)
$$

对矩阵的每个奇异值 $\sigma_i$ 进行阈值处理，得出更新后的 $Z^{k+1}$。

**更新拉格朗日乘子 $Y$：**

$$
Y^{k+1} = Y^k + X^{k+1} - Z^{k+1}
$$


重复上述更新步骤，直到变量 $X$、$Z$ 和 $Y$ 收敛为止。
## 6.图
### （1）二分图
- 二分图是图论中的一种特殊模型。若能将无向图G=(V,E)的顶点V划分为两个交集为空的顶点集，并且任意边的两个端点都分属于两个集合，则称图G为一个为二分图。
- 二分图（Bipartite graph）是一类特殊的图，它可以被划分为两个部分，每个部分内的点互不相连。

### （2）匹配
- 图G的一个匹配是由一组没有公共端点的不是圈的边构成的集合
- 匹配的两个重点：1. 匹配是边的集合；2. 在该集合中，任意两条边不能有共同的顶点。
### （3）最大匹配
- 最大匹配是指在给定的图中选择一个边的子集，使得每条边的两个顶点都不相交，并且选出的边的数量最多。换句话说，最大匹配是要找到一个图中的最大边数的匹配。这个问题的关键在于找出不相交边的最大集合，使得尽可能多的顶点都能匹配。

### （4）完美匹配
- 完美匹配是最大匹配的一个特例。在一个图中，如果一个匹配的边能够使得图中的每个顶点都恰好与一条边相关联，那么这个匹配就称为完美匹配。

### （5）最优匹配
- 最优匹配又称为带权最大匹配，是指在带有权值边的二分图中，求一个匹配使得匹配边上的权值和最大。

例子：

                                    X: {员工A, 员工B, 员工C}
                                    Y: {任务1, 任务2, 任务3}

                                    边的权值如下：
                                    A-1: 5, A-2: 8, A-3: 6
                                    B-1: 4, B-2: 7, B-3: 3
                                    C-1: 9, C-2: 6, C-3: 4

                                    最优匹配可能是：
                                    A -> 任务2 (权值8)
                                    B -> 任务3 (权值3)
                                    C -> 任务1 (权值9)
                                    总权值：8 + 3 + 9 = 20
### （6）最小顶点覆盖
- 最小顶点覆盖（Minimum Vertex Cover）是指在一个图中，找到最少数量的顶点，使得图中的每条边都至少有一个顶点与之相连。换句话说，如果你选出的顶点集合是一个顶点覆盖，那么每条边都至少与这个集合中的一个顶点关联；
### （7）交替路
- 从未匹配点出发，依次经过未匹配的边和已匹配的边，即为交替路
### （8）增广路）
- 如果交替路经过除出发点外的另一个未匹配点，则这条交替路称为增广路
### （9）匈牙利算法
匈牙利算法主要用来解决两个问题：求二分图的最大匹配数和最小点覆盖数。
本质就是迭代寻找最优解。
### （9）KM算法
整体思路就是：每次都帮一个顶点匹配最大权重边，利用匈牙利算法完成最大匹配，最终我们完成的就是最优匹配！
# Ⅱ挑战
## 1.亲和度矩阵:
输入：基础矩阵，所有关键点的集合
输出：匹配矩阵
匹配矩阵内的数值是通过极线限制计算得到的，如果两个关键点是匹配的，那么与基础矩阵相乘结果应该为0。
## 2.循环一致性：
### (1)如何保证循环一致性？
假设矩阵$P$为$m*m$的匹配矩阵，$P_{ij}=1$表示第$i$个框和第$j$个框匹配
$$
\left[
\begin{matrix}
 P_{11}     & P_{12}      & \cdots & P_{1m}     \\
 P_{21}      & P_{22}      & \cdots & P_{2m}      \\
 \vdots & \vdots & \ddots & \vdots \\
 P_{m1}      & P_{m2}      & \cdots & P_{mm}      \\
\end{matrix}
\right]
$$
其中$P_{ii}$应该为基础矩阵。想要保证循环一致性：$rank(P)\leq s$,$s$是潜在的物体个数

所以当前问题变为用亲和度矩阵$A$来构建一个匹配矩阵$P$，并且使$P$满足：
$$
\begin{cases}
rank(P)\leq s\\
max\langle A,P\rangle
\end {cases}
$$
这可以用拉格朗日方程解决
### (2)构建拉格朗日方程
原来是想$max\langle A,P\rangle$,我们可以给它加一个负号，转化为求$-min\langle A,P\rangle$
$$
f(P)=-\sum_{i=0}^n\sum_{j=0}^n\langle A_{ij},P_{ij}\rangle+\lambda rank(P)\\
=-\langle A,P\rangle+\lambda rank(P)
$$
由于最小化秩是非凸问题，所以我们将其转化为最小化核范数近似求解
所以问题变为
$$
\min_P-\langle A,P\rangle+\lambda\mid\mid P\mid\mid_*\\
s.t.\ P\in C\\
C:\begin{cases}
P_{ij}=P_{ji}^T\\
P_{ii}=I\\
0\leq P_{ij}1\leq1,0\leq P_{ij}^T1\leq1
\end{cases}
$$
### (3)ADMM+SVT求解拉格朗日方程
引入辅助变量Q(辅助变量主要是为了求解出最小秩矩阵)
$$
\min_P-\langle A,P\rangle+\lambda\mid\mid Q\mid\mid_*\\
s.t.\ P\in C,P=Q\\
C:\begin{cases}
P_{ij}=P_{ji}^T\\
P_{ii}=I\\
0\leq P_{ij}1\leq1,0\leq P_{ij}^T1\leq1
\end{cases}
$$
增广拉格朗日方程为
$$
L_\rho(P,Q,Y)=-\langle A,P\rangle+\lambda\mid\mid Q\mid\mid_*+\langle Y,P-Q\rangle+\frac{\rho}{2}\mid\mid P-Q\mid\mid_F^2
$$
优化$Q$：
$$
将与Q无关的参数去掉\\
\min_Q\ \lambda\mid\mid Q\mid\mid_*+\langle Y,P-Q\rangle+\frac{\rho}{2}\mid\mid P-Q\mid\mid_F^2\\
展开\\
\min_Q\ \lambda\mid\mid Q\mid\mid_*+\langle Y,P\rangle-\langle Y,Q\rangle+\frac{\rho}{2}(\mid\mid P\mid\mid_F^2-2\langle Q,P\rangle+\mid\mid Q\mid\mid_F^2)\\
\min_Q\ \lambda\mid\mid Q\mid\mid_*+\frac{\rho}{2}(\mid\mid Q\mid\mid_F^2-2\langle Q,P+\frac{1}{\rho}Y\rangle)+const\\
由于const不影响优化，所以上式可以化为：\\
\min_Q\ \lambda\mid\mid Q\mid\mid_*+\frac{\rho}{2}\mid\mid Q-(P+\frac{1}{\rho}Y)\mid\mid_F^2\\
这个式子可以用SVT方法求解\\
Q\leftarrow SVT(P+\frac{1}{\rho}Y)
$$
优化$P$:

$$
将与P无关的参数去掉\\
\min_P \langle A, P \rangle + \langle Y, P - Q \rangle + \frac{\rho}{2} \|P - Q\|_F^2\\
\min_P \langle A + Y, P \rangle - \langle Y, Q \rangle + \frac{\rho}{2} \|P - Q\|_F^2\\
由于  \langle Y, Q \rangle  与  P  无关，因此可以忽略该常数项\\
\min_P \langle A + Y, P \rangle + \frac{\rho}{2} \|P - Q\|_F^2\\
\min_P \frac{\rho}{2} \|P - (Q - \frac{1}{\rho} (A + Y))\|_F^2\\
最后，由于有附加约束，通过投影操作，可以得出  P  的更新步骤为：\\
P \leftarrow \text{Proj}_{\mathcal{C}}\left(Q - \frac{1}{\rho}(A + Y)\right)\\
其中，\text{Proj}_{\mathcal{C}}(\cdot) 表示投影操作。
$$
优化$Y$:
$$
梯度下降法：
Y^{k+1}\leftarrow Y^{k}+\rho (P-Q)
$$
## 3.连续建立：
### (1)匈牙利算法
我们需要构建一个权重矩阵，其中矩阵的每个元素代表两帧中一个人之间的匹配代价（根据两个点云数据中人体姿态的距离或相似性度量）。然后，通过匈牙利算法，我们可以在这个矩阵中找到一个最优的匹配，即选择一组配对，使得总匹配代价最小。
