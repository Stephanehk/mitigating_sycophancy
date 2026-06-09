"""Plot per-iteration DPO training summaries for the 1000-prompt OEQ / PAS runs.

For each env (OEQ, PAS) produces one figure with two subplots:
  - left:  train_loss (mean DPO-epoch loss) per outer iteration
  - right: # prompts for which ALL sampled responses were vetoed, per iteration
with one line per preference-source method (base / mixed / selfpref).

Data source: read verbatim from the SLURM .out logs in logs/. Each plotted
marker is one logged value -- no interpolation or synthesis. Methods/iterations
that have not yet produced a logged value (run still sampling/training, or job
PENDING) simply contribute no marker. Source lines:

  OEQ base     dpo_oeq_n1000_15777085.out
    '[iter k/3] ... all_vetoed_prompts='  k=1:408  k=2:177  k=3:77
    "'train_loss'"                        k=1:0.145  k=2:0.1502   (k=3 still training)
  OEQ mixed    dpo_oeq_n1000_mixed_15785864.out
    all_vetoed_prompts=                   k=1:355                 (k=2 sampling)
    'train_loss'                          k=1:0.07965
  OEQ selfpref dpo_oeq_n1000_selfpref_15788361.out   (iter 1 sampling, no value yet)
  PAS base     dpo_pas_n1000_15777086.out   COMPLETED
    all_vetoed_prompts=                   k=1:678  k=2:370  k=3:144
    'train_loss'                          k=1:0.1641  k=2:0.1558  k=3:0.173
  PAS mixed    dpo_pas_n1000_mixed_15788362.out      (iter 1 sampling, no value yet)
  PAS selfpref 15788363                              PENDING (no log)
"""
import os
import matplotlib.pyplot as plt

# Each method holds logged (iteration, value) tuples only -- one tuple per
# value actually present in the log. Empty list == nothing logged yet.
RUNS = {
    "OEQ": {
        "base": {
            "loss": [(1, 0.145), (2, 0.1502)],
            "vetoed": [(1, 408), (2, 177), (3, 77)],
        },
        "mixed": {
            "loss": [(1, 0.07965)],
            "vetoed": [(1, 355)],
        },
        "selfpref": {
            "loss": [],
            "vetoed": [],
        },
    },
    "PAS": {
        "base": {
            "loss": [(1, 0.1641), (2, 0.1558), (3, 0.173)],
            "vetoed": [(1, 678), (2, 370), (3, 144)],
        },
        "mixed": {
            "loss": [],
            "vetoed": [],
        },
        "selfpref": {
            "loss": [],
            "vetoed": [],
        },
    },
}

METHOD_COLORS = {
    "base": "tab:blue",
    "mixed": "tab:orange",
    "selfpref": "tab:green",
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots_dpo_summary")


def plot_series(ax, methods, key):
    """Plot one (iter, value) line per method on ax. Returns True if any marker
    was drawn. Asserts iterations are the contiguous 1..k expected by the loop."""
    drew = False
    for method, series in methods.items():
        pts = series[key]
        if not pts:
            continue
        iters = [p[0] for p in pts]
        assert iters == list(range(1, len(iters) + 1)), (
            f"{method}/{key}: expected contiguous iters 1..n, got {iters}"
        )
        vals = [p[1] for p in pts]
        ax.plot(iters, vals, marker="o", color=METHOD_COLORS[method], label=method)
        drew = True
    return drew


def plot_env(env, methods):
    fig, (ax_loss, ax_veto) = plt.subplots(1, 2, figsize=(11, 4.5))

    drew_loss = plot_series(ax_loss, methods, "loss")
    ax_loss.set_title("train_loss")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("mean DPO-epoch loss")
    ax_loss.set_xticks([1, 2, 3])
    ax_loss.grid(True, alpha=0.3)
    if drew_loss:
        ax_loss.legend(title="method")

    drew_veto = plot_series(ax_veto, methods, "vetoed")
    ax_veto.set_title("all-vetoed prompts")
    ax_veto.set_xlabel("iteration")
    ax_veto.set_ylabel("# prompts (of 1000) with all responses vetoed")
    ax_veto.set_xticks([1, 2, 3])
    ax_veto.grid(True, alpha=0.3)
    if drew_veto:
        ax_veto.legend(title="method")

    fig.suptitle(f"{env} DPO+LoRA (Qwen3-8B, n=1000) -- results so far")
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, f"dpo_summary_{env.lower()}_n1000.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for env, methods in RUNS.items():
        plot_env(env, methods)


if __name__ == "__main__":
    main()
