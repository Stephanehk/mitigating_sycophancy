"""Plot per-iteration DPO training summaries for the 300-prompt OEQ and PAS runs.

For each run, produces a figure with two subplots:
  - left:  train_loss (mean DPO-epoch loss) across the 3 outer iterations
  - right: number of prompts for which ALL sampled responses were vetoed

Data source: the values are read verbatim from the SLURM .out logs
  logs/dpo_oeq_15651220.out   (OEQ, May 30)
  logs/dpo_pas_15623593.out   (PAS, May 29)
specifically the `[iter k/3] ... all_vetoed_prompts=...` summary lines and the
per-iteration `'train_loss'` epoch summaries. No interpolation or synthesis:
each marker is one logged iteration value.
"""
import os
import matplotlib.pyplot as plt

# (iteration, train_loss, all_vetoed_prompts) straight from the logs.
RUNS = {
    "OEQ": {
        "log": "dpo_oeq_15651220.out",
        "iters": [1, 2, 3],
        "train_loss": [0.1703, 0.1401, 0.1521],
        "all_vetoed": [128, 70, 39],
    },
    "PAS": {
        "log": "dpo_pas_15623593.out",
        "iters": [1, 2, 3],
        "train_loss": [0.2172, 0.1426, 0.1689],
        "all_vetoed": [212, 168, 85],
    },
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots_dpo_summary")


def plot_run(env, run):
    iters = run["iters"]
    n = len(iters)
    assert len(run["train_loss"]) == n and len(run["all_vetoed"]) == n, (
        f"{env}: expected {n} values per series, "
        f"got loss={len(run['train_loss'])} vetoed={len(run['all_vetoed'])}"
    )

    fig, (ax_loss, ax_veto) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(iters, run["train_loss"], marker="o", color="tab:blue")
    ax_loss.set_title("train_loss")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("mean DPO-epoch loss")
    ax_loss.set_xticks(iters)
    ax_loss.grid(True, alpha=0.3)

    ax_veto.plot(iters, run["all_vetoed"], marker="o", color="tab:red")
    ax_veto.set_title("all-vetoed prompts")
    ax_veto.set_xlabel("iteration")
    ax_veto.set_ylabel("# prompts (of 300) with all responses vetoed")
    ax_veto.set_xticks(iters)
    ax_veto.grid(True, alpha=0.3)

    fig.suptitle(f"{env} DPO+LoRA — 300 prompts ({run['log']})")
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, f"dpo_summary_{env.lower()}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for env, run in RUNS.items():
        plot_run(env, run)


if __name__ == "__main__":
    main()
