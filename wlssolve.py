def _wls_solve(X, y, w):
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
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

    # 不要构造 W = np.diag(w)
    # X.T @ W @ X 等价于 X.T @ (w[:, None] * X)
    XtWX = X.T @ (w[:, None] * X)
    XtWy = X.T @ (w * y)

    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtWX) @ XtWy

    return beta
