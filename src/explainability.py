import numpy as np
import json, sys, base64, io
import matplotlib.pyplot as plt

FEATURES = [
    'LB','AC','FM','UC','DL','DS','DP','ASTV','MSTV','ALTV',
    'MLTV','Width','Min','Max','Nmax','Nzeros','Mode','Mean',
    'Median','Variance','Tendency'
]


def explain_ctg(ctg):

    values = np.abs(ctg)
    values = values / (np.sum(values) + 1e-8)

    sorted_idx = np.argsort(values)[::-1]

    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(
        [FEATURES[i] for i in sorted_idx],
        values[sorted_idx]
    )

    ax.set_title("CTG Feature Importance")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()

    return base64.b64encode(buf.getvalue()).decode()


def run(data):

    ctg = data.get("ctg")

    if not ctg or len(ctg) != 21:
        return {"shap": "", "status": "error"}

    return {
        "shap": explain_ctg(ctg),
        "status": "success"
    }


if __name__ == "__main__":
    print(json.dumps(run(json.loads(sys.stdin.read()))))