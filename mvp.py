import numpy as np, json
from scipy import ndimage as ndi
from sklearn.neural_network import MLPClassifier
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

N, P = 256, 7

def make_object(seed, n_pores=None, n_cracks=None, n_incl=None):
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:N, 0:N]
    thick = 1.0 - 0.35*((xx-N/2)/(N/2))**2 - 0.15*((yy-N/2)/(N/2))**2
    thick = thick + ndi.gaussian_filter(r.normal(0, 1, (N, N)), 12)*0.04
    mask_obj = ((np.abs(xx-N/2) < 112) & (np.abs(yy-N/2) < 100)).astype(float)
    mu = 2.2
    defect = np.zeros((N, N), bool); dl = np.zeros((N, N))

    for _ in range(n_pores if n_pores is not None else r.integers(3, 8)):
        cx, cy = r.integers(35, N-35, 2); rad = r.uniform(2.0, 5.5)
        m = (xx-cx)**2 + (yy-cy)**2 < rad**2
        dl += m*r.uniform(0.25, 0.5); defect |= m

    for _ in range(n_cracks if n_cracks is not None else r.integers(1, 3)):
        x0, y0 = r.integers(40, N-40, 2); ang = r.uniform(0, np.pi); L = r.integers(25, 55)
        t = np.linspace(0, L, 4*L)
        xs = (x0 + t*np.cos(ang) + np.cumsum(r.normal(0, .35, t.size))).astype(int).clip(0, N-1)
        ys = (y0 + t*np.sin(ang) + np.cumsum(r.normal(0, .35, t.size))).astype(int).clip(0, N-1)
        m = np.zeros((N, N), bool); m[ys, xs] = True; m = ndi.binary_dilation(m, iterations=1)
        dl += m*0.22; defect |= m

    for _ in range(n_incl if n_incl is not None else r.integers(1, 4)):
        cx, cy = r.integers(35, N-35, 2); rad = r.uniform(2.5, 5)
        m = (xx-cx)**2 + (yy-cy)**2 < rad**2
        dl -= m*0.35; defect |= m
    path = np.clip(thick - dl, 0.05, None) * mask_obj
    I = np.exp(-mu*path*0.9)
    I = np.where(mask_obj > 0, I, 1.0)
    I = ndi.gaussian_filter(I, 0.8)
    I = I + r.normal(0, 0.012, I.shape)
    defect &= mask_obj > 0
    return I.astype(np.float32), defect

def features(I):
    res = I - ndi.gaussian_filter(I, 10)
    gx, gy = ndi.sobel(ndi.gaussian_filter(I, 1), 1), ndi.sobel(ndi.gaussian_filter(I, 1), 0)
    grad = np.hypot(gx, gy)
    return res, grad

def patches(res, grad, pts):
    rp, gp = np.pad(res, P, mode="reflect"), np.pad(grad, P, mode="reflect")
    out = [np.concatenate([rp[y:y+2*P+1, x:x+2*P+1].ravel()/0.05,
                           gp[y:y+2*P+1, x:x+2*P+1].ravel()/0.3]) for y, x in pts]
    return np.array(out, dtype=np.float32)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X, y = [], []

    for s in range(1, 41):
        I, d = make_object(s); res, grad = features(I)
        pos = np.argwhere(d); neg = np.argwhere(~d)
        pos = pos[rng.choice(len(pos), min(len(pos), 400), replace=False)]
        neg = neg[rng.choice(len(neg), 1200, replace=False)]
        pts = np.vstack([pos, neg]); X.append(patches(res, grad, pts))
        y += [1]*len(pos) + [0]*len(neg)
    X = np.vstack(X); y = np.array(y)
    clf = MLPClassifier((64, 32), max_iter=60, random_state=0).fit(X, y)
    print("обучающих патчей:", len(y), "train acc:", round(clf.score(X, y), 3))

    I, gt = make_object(1000, n_pores=5, n_cracks=1, n_incl=2)
    res, grad = features(I)
    pts = [(yy, xx) for yy in range(N) for xx in range(N)]
    prob = clf.predict_proba(patches(res, grad, pts))[:, 1].reshape(N, N)
    pred = prob > 0.5
    pred = ndi.binary_opening(pred, iterations=1)
    lab, n = ndi.label(pred)

    for i in range(1, n+1):
        if (lab == i).sum() < 6: pred[lab == i] = False
    tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
    prec, rec, dice = tp/(tp+fp+1e-9), tp/(tp+fn+1e-9), 2*tp/(2*tp+fp+fn+1e-9)
    gl, gn = ndi.label(ndi.binary_dilation(gt, iterations=2))
    found = sum(1 for i in range(1, gn+1) if pred[gl == i].any())
    pl, pn = ndi.label(pred)
    false_obj = sum(1 for i in range(1, pn+1) if not (gt & ndi.binary_dilation(pl == i, iterations=3)).any())
    m = dict(pixel_precision=round(float(prec), 3), pixel_recall=round(float(rec), 3), dice=round(float(dice), 3),
             defects_in_object=int(gn), defects_found=int(found), false_alarms=int(false_obj))
    print(m); json.dump(m, open("metrics.json", "w"), ensure_ascii=False, indent=1)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.3))
    ax[0].imshow(I, cmap="gray"); ax[0].set_title("1. Синтетическая рентгенограмма")
    ax[1].imshow(grad, cmap="magma"); ax[1].set_title("2. Градиент яркости")
    ax[2].imshow(prob, cmap="viridis", vmin=0, vmax=1); ax[2].set_title("3. Вероятность дефекта (MLP)")
    ov = np.dstack([I]*3).clip(0, 1)
    ov[ndi.binary_dilation(pred) & ~pred] = [1, 0, 0]
    ov[ndi.binary_dilation(gt) & ~gt] = [0, 1, 0]
    ax[3].imshow(ov); ax[3].set_title("4. Красный: найдено, зеленый: эталон")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig("mvp_result.png", dpi=110)
    plt.imsave("test_object.png", I, cmap="gray")
