import pandas as pd
import numpy as np

df = pd.read_csv("data/density_mu_delta.csv")
mus    = np.sort(df["mu"].unique())
deltas = np.sort(df["delta"].unique())

col = "n4"
Z = df.pivot(index="delta", columns="mu", values=col).reindex(index=deltas, columns=mus).values
X, Y = np.meshgrid(mus, deltas)

import plotly.graph_objects as go
fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
fig.update_layout(
    title="Density: ⟨n⟩(μ, Δ)",
    scene=dict(xaxis_title="μ", yaxis_title="Δ", zaxis_title="⟨n⟩"),
    width=900, height=650
)
fig.show()
# ================= end of file =================

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111, projection="3d")
# surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
# ax.set_xlabel("μ"); ax.set_ylabel("Δ"); ax.set_zlabel("⟨n⟩ per LLL orbital")
# fig.colorbar(surf, shrink=0.65, aspect=12, label="⟨n⟩")
# plt.tight_layout(); plt.show()

