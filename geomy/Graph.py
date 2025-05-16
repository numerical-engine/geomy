"""グラフ構造に関するアルゴリズム
Note:
    * オーダーがnの無向グラフについて、隣接行列は(n, n)の要素で定義されていると仮定。つまり上三角行列のみ保存されたデータ構造になっていない。
"""


import numpy as np

def get_degree_coo(adjmat:np.ndarray, order:int)->np.ndarray:
    """無向グラフの各ノードの次数を計算

    Args:
        adjmat (np.ndarray): COO形式隣接行列
        order (int): グラフのノード数
    Returns:
        np.ndarray: 各ノードの次数。shapeは(order, )
    """
    degree = np.array([len(np.where(adjmat[0] == i)) for i in range(order)])
    return degree

def get_inoutdegree_coo(adjmat:np.ndarray, order:int)->tuple[np.ndarray, np.ndarray]:
    """有向グラフの各ノードの入次数、出次数を計算

    Args:
        adjmat (np.ndarray): COO形式隣接行列
        order (int): グラフのノード数
    Returns:
        tuple[np.ndarray, np.ndarray]: 順に入次数、出次数。shapeは(order, )
    """
    indegree = np.array([len(np.where(adjmat[1] == i)) for i in range(order)])
    outdegree = np.array([len(np.where(adjmat[0] == i)) for i in range(order)])

    return indegree, outdegree