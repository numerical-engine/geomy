import numpy as np
import sys

def get_centroid(points:np.ndarray)->np.ndarray:
    """点群の重心を計算

    Args:
        points (np.ndarray): 点群の座標値。shapeは(N, dim)
    Returns:
        np.ndarray: 重心座標。shapeは(dim, )
    """
    return np.mean(points, axis = 0)

def get_area(points:np.ndarray)->float:
    """凸多角形の面積を計算

    Args:
        points (np.ndarray): 頂点集合。shapeは(N, 2)。反時計周りに定義。
    Returns:
        float: 面積。
    """
    assert points.shape[1] == 2
    x, y = points[:,0], points[:,1]
    shoelace = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return 0.5*abs(shoelace)

def get_facetnormal_line(points:np.ndarray)->np.ndarray:
    """2点間の線分の外向き単位法線ベクトルを出力する

    Args:
        points (np.ndarray): 2点の座標。shapeは(2, 2)
    Returns:
        np.ndarray: 外向き単位法線ベクトル。shapeは(2, )
    Note:
        * 対象の線分は何か多角形の辺であり、頂点集合は反時計周りに定義されていると仮定。
        * 2次元であることを仮定。3次元だと一意に定まらない。
    """
    assert points.shape == (2, 2)
    l = np.sqrt((points[0,0] - points[1,0])**2 + (points[0,1] - points[1,1])**2)
    n = np.array([points[1,1] - points[0,1], points[0,0] - points[1,0]])/l
    return n

def get_facetnormal_surface(points:np.ndarray)->np.ndarray:
    """3次元空間で定義された面の外向き単位法線ベクトルを出力。

    Args:
        points (np.ndarray): 頂点集合。shapeは(N, 3)でNは頂点の数。
    Returns:
        np.ndarray: 外向き単位法線ベクトル。shapeは(3, )
    Note:
        * 頂点集合であるpointsは反時計周りの順に定義されている。
        * 外向き = 頂点の順に対し右ねじの向き。
    """
    assert points.shape[1] == 3
    N = len(points)
    assert N > 2

    p0 = points[0]
    normal = np.zeros(3)
    v1 = np.stack([p - p0 for p in points[1:N-1]])
    v2 = np.stack([p - p0 for p in points[2:]])
    normal = np.sum(np.cross(v1, v2), axis = 0)
    normal /= np.linalg.norm(normal)

    return normal

def triangulate(points:np.ndarray)->list[np.ndarray]:
    """凸多角形を三角形で分割する

    Args:
        points (np.ndarray): 頂点の集合。
            shapeは(N, dim)でNは頂点数。dimは次元であり、2と3があり得る。
            頂点は反時計周りに定義されている。
    Returns:
        list[np.ndarray]: 三角形要素の集合。
            listの長さは3角形の数。要素のnumpy配列のshapeは(3, dim)。
    """
    triangles = [points[(0, i, i+1),:] for i in range(1, len(points)-1)]
    return triangles

def get_volume_tetra(points:np.ndarray)->float:
    """四面体の体積を計算する

    Args:
        points (np.ndarray): 頂点の集合。shapeは(4, 3)。
    Returns:
        float: 体積
    """
    assert points.shape == (4, 3)
    p0, p1, p2, p3 = points
    mat = np.vstack((p1-p0, p2-p0, p3-p0))
    det = np.linalg.det(mat)
    return abs(det)/6.0

def get_volume_voxel(points:np.ndarray)->float:
    """ボクセルの体積を計算

    Args:
        points (np.ndarray): 頂点集合。shapeは(8, 3)
            底面が(0, 1, 2, 3)要素で定義されており、上面が(4, 5, 6, 7)で定義されていると仮定
    Returns:
        float: ボクセルの体積
    """
    assert points.shape == (8, 3)
    width = np.abs(points[1,0] - points[0,0])
    height = np.abs(points[3,1] - points[0,1])
    depth = np.abs(points[4,2] - points[0,2])

    return width*height*depth

def get_volume_hexa(points:np.ndarray)->float:
    """六面体の体積を計算

    Args:
        points (np.ndarray): 頂点集合。shapeは(8, 3)
            底面が(0, 1, 2, 3)要素で定義されており、上面が(4, 5, 6, 7)で定義されていると仮定
    Returns:
        float: 六面体の体積
    """
    assert points.shape == (8, 3)
    tetra_indice = [(0,1,3,4), (1,2,3,6), (1,4,5,6), (3,4,6,7), (1,3,4,6)]

    volume = 0.0
    for tetra_idx in tetra_indice:
        tetra_points = points[tetra_idx,:]
        volume += get_volume_tetra(tetra_points)
    return volume

def get_volume(facets:list[np.ndarray])->float:
    """凸多面体の体積を計算する

    Args:
        facets (list[np.ndarray]): 凸多面体を構成するファセットの集合
            リストの各要素はファセットの頂点座標の集合。shapeは(N, 3)であり、Nはファセットの頂点の数
            len(facets)は凸多面体を構成するファセットの数
    Returns:
        float: 凸多面体の体積
    """
    centroid = get_centroid(np.unique(np.concatenate(facets), axis = 0)).reshape((1,-1))
    volume = 0.0
    for facet in facets:
        triangles = triangulate(facet)
        for triangle in triangles:
            volume += get_volume_tetra(np.concatenate((centroid, triangle)))
    return volume