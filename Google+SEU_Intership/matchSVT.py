from match_solver import myproj2dpam
import time
import torch

def matchSVT(S, dimGroup, **kwargs):
    # 从kwargs中获取参数，如果没有提供则使用默认值
    alpha = kwargs.get('alpha', 0.1)  # 参数 alpha，用于调整矩阵的权重
    pSelect = kwargs.get('pselect', 1)  # 选择概率 pSelect，用于设置对角线元素
    tol = kwargs.get('tol', 1e-2)  # 容差 tol，用于判断收敛条件
    maxIter = kwargs.get('maxIter', 500)  # 最大迭代次数 maxIter
    verbose = kwargs.get('verbose', False)  # 是否打印详细信息
    eigenvalues = kwargs.get('eigenvalues', False)  # 是否计算并返回矩阵的特征值
    _lambda = kwargs.get('_lambda', 50)  # 正则化参数 lambda，用于更新 Q
    mu = kwargs.get('mu', 64)  # 惩罚参数 mu，用于平衡约束条件
    dual_stochastic = kwargs.get('dual_stochastic_SVT', True)  # 是否应用双重随机投影

    if verbose:
        print(f'Running SVT-Matching: alpha = {alpha:.2f}, pSelect = {pSelect:.2f}, _lambda = {_lambda:.2f} \n')

    info = dict()  # 存储算法运行信息的字典
    N = S.shape[0]  # 获取矩阵 S 的维度大小
    S[torch.arange(N), torch.arange(N)] = 0  # 将矩阵 S 的对角线元素置为 0
    S = (S + S.t()) / 2  # 保证矩阵 S 是对称的
    X = S.clone()  # 初始化矩阵 X 为矩阵 S 的副本
    Y = torch.zeros_like(S)  # 初始化矩阵 Y 为与 S 维度相同的零矩阵
    W = alpha - S  # 初始化矩阵 W，用于调整矩阵的权重
    t0 = time.time()  # 记录算法开始时间

    for iter_ in range(maxIter):  # 迭代主循环
        # =========================================更新Q===========================================
        X0 = X  # 保存上一轮迭代的 X 值
        # 使用奇异值软阈值化(SVT)更新 Q
        U, s, V = torch.svd(1.0 / mu * Y + X)  # 对 (1/mu) * Y + X 进行奇异值分解
        diagS = s - _lambda / mu  # 对奇异值 s 进行软阈值化处理
        diagS[diagS < 0] = 0  # 将所有小于 0 的奇异值置为 0
        Q = U @ diagS.diag() @ V.t()  # 重构矩阵 Q

        # =========================================更新P===========================================
        X = Q - (W + Y) / mu
        # 投影 X 使其满足约束条件
        for i in range(len(dimGroup) - 1):
            ind1, ind2 = dimGroup[i], dimGroup[i + 1]
            X[ind1:ind2, ind1:ind2] = 0  # 对自己图像内的子矩阵的元素强制置为 0

        if pSelect == 1:
            X[torch.arange(N), torch.arange(N)] = 1  # 将 X 的对角线元素设置为 1

        X[X < 0] = 0  # 将所有小于 0 的元素置为 0
        X[X > 1] = 1  # 将所有大于 1 的元素置为 1

        if dual_stochastic:
            # 双重随机投影以满足行列和约束
            for i in range(len(dimGroup) - 1):
                row_begin, row_end = int(dimGroup[i]), int(dimGroup[i + 1])
                for j in range(len(dimGroup) - 1):
                    col_begin, col_end = int(dimGroup[j]), int(dimGroup[j + 1])
                    if row_end > row_begin and col_end > col_begin:
                        X[row_begin:row_end, col_begin:col_end] = myproj2dpam(
                            X[row_begin:row_end, col_begin:col_end], 1e-2)

        X = (X + X.t()) / 2  # 保持矩阵 X 对称
        #=========================================更新Y==========================================
        Y = Y + mu * (X - Q)
        # 判断是否收敛
        pRes = torch.norm(X - Q) / N  # 原始残差
        dRes = mu * torch.norm(X - X0) / N  # 对偶残差
        if verbose:
            print(f'Iter = {iter_}, Res = ({pRes}, {dRes}), mu = {mu}')

        if pRes < tol and dRes < tol:
            break  # 如果残差足够小，停止迭代

        # 调整 mu 的值
        if pRes > 10 * dRes:
            mu = 2 * mu
        elif dRes > 10 * pRes:
            mu = mu / 2

    X = (X + X.t()) / 2  # 最终保持矩阵 X 对称
    info['time'] = time.time() - t0  # 记录算法总运行时间
    info['iter'] = iter_  # 记录迭代次数

    if eigenvalues:
        info['eigenvalues'] = torch.eig(X)  # 如果需要，计算 X 的特征值

    X_bin = X > 0.5  # 将矩阵 X 二值化
    if verbose:
        print(f"Alg terminated. Time = {info['time']}, #Iter = {info['iter']}, Res = ({pRes}, {dRes}), mu = {mu} \n")

    return X_bin  # 返回二值化后的矩阵 X
