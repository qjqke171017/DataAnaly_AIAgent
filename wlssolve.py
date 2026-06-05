def _wls_solve(X, y, w=None):
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # 如果没有传入权重，就退化为普通最小二乘
    if w is None:
        w = np.ones(len(y), dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64).reshape(-1)

    # 过滤非法值
    mask = (
        np.isfinite(y)
        & np.isfinite(w)
        & (w > 0)
        & np.all(np.isfinite(X), axis=1)
    )

    X = X[mask]
    y = y[mask]
    w = w[mask]

    if len(y) == 0:
        raise ValueError("有效样本数为 0，无法拟合 WLS/OLS 模型")

    # 不要构造 W = np.diag(w)
    # X.T @ W @ X 等价于 X.T @ (w[:, None] * X)
    XtWX = X.T @ (w[:, None] * X)
    XtWy = X.T @ (w * y)

    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtWX) @ XtWy

    return beta
