"""Train an algorithm."""
import argparse
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass


def bootstrap_mean_ci(values, n_boot=10000, ci=95, seed=42):
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot_means = x[idx].mean(axis=1)

    alpha = (100 - ci) / 2
    low = np.percentile(boot_means, alpha)
    high = np.percentile(boot_means, 100 - alpha)
    return float(x.mean()), float(low), float(high)



def build_complete_directed_edge_index(N: int, order: str = "row_major") -> np.ndarray:
    """
    Build edge_index_low for complete directed graph without self-loops.

    order="row_major" produces:
        for u in 0..N-1:
            for v in 0..N-1, v!=u:
                append (u->v)
    E = N*(N-1)

    Returns: edge_index_low shape [2, E]
    """
    edges_u = []
    edges_v = []
    if order == "row_major":
        for u in range(N):
            for v in range(N):
                if v == u:
                    continue
                edges_u.append(u)
                edges_v.append(v)
    elif order == "col_major":
        # alternative order:
        for v in range(N):
            for u in range(N):
                if u == v:
                    continue
                edges_u.append(u)
                edges_v.append(v)
    else:
        raise ValueError("order must be 'row_major' or 'col_major'")
    return np.array([edges_u, edges_v], dtype=int)


# =========================
# 2) Group id remapping + stats
# =========================

def remap_groups(plan_1d: np.ndarray) -> np.ndarray:
    """
    Remap arbitrary group ids to consecutive 0..Nh-1 for stable counting.
    plan_1d: shape [N]
    """
    uniq = np.unique(plan_1d)
    mp = {g:i for i, g in enumerate(uniq)}
    return np.array([mp[g] for g in plan_1d], dtype=int)

def cluster_stats(plan: np.ndarray) -> dict:
    """
    plan: [N,1] or [N]
    """
    g = np.asarray(plan).reshape(-1).astype(int)
    c = remap_groups(g)
    uniq, counts = np.unique(c, return_counts=True)
    Nh = len(uniq)
    return {
        "Nh": int(Nh),
        "min_size": int(counts.min()),
        "max_size": int(counts.max()),
        "mean_size": float(counts.mean()),
        "std_size": float(counts.std(ddof=0)),
        "cv_size": float(counts.std(ddof=0) / (counts.mean() + 1e-12)),
        "imbalance_ratio": float(counts.max() / (counts.min() + 1e-12)),
    }

def switch_rate(plan_prev: np.ndarray, plan_curr: np.ndarray) -> float:
    """
    Switch rate computed on raw group ids to avoid remap mismatch.
    """
    g0 = np.asarray(plan_prev).reshape(-1).astype(int)
    g1 = np.asarray(plan_curr).reshape(-1).astype(int)
    return float(np.mean(g0 != g1))


# =========================
# 3) High-level graph stats (group-level edge_index)
# =========================

def high_graph_stats(edge_index_high: np.ndarray, Nh: int,
                     undirected: bool = True, remove_self_loops: bool = True) -> dict:
    """
    edge_index_high: shape [2, Eh_raw]
    Nh: number of groups (should match group ids range)
    """
    ei = np.asarray(edge_index_high)
    src = ei[0].astype(int)
    dst = ei[1].astype(int)

    if remove_self_loops:
        m = src != dst
        src, dst = src[m], dst[m]

    if src.size == 0:
        return {"Eh": 0, "density_high": 0.0, "avg_degree_high": 0.0, "max_degree_high": 0}

    if undirected:
        a = np.minimum(src, dst)
        b = np.maximum(src, dst)
        edges = np.stack([a, b], axis=1)
        edges = np.unique(edges, axis=0)  # undirected unique edges
        Eh = edges.shape[0]

        deg = np.zeros((Nh,), dtype=int)
        for u, v in edges:
            if u < Nh and v < Nh:
                deg[u] += 1
                deg[v] += 1

        max_edges = Nh * (Nh - 1) / 2
        return {
            "Eh": int(Eh),
            "density_high": float(Eh / (max_edges + 1e-12)),
            "avg_degree_high": float(deg.mean()),
            "max_degree_high": int(deg.max()) if Nh > 0 else 0,
        }
    else:
        edges = np.stack([src, dst], axis=1)
        edges = np.unique(edges, axis=0)
        return {"Eh_directed": int(edges.shape[0])}


# =========================
# 4) Low-level graph stats from weights (complete directed graph)
# =========================

def low_graph_stats(w_low: np.ndarray) -> dict:
    """
    w_low: shape [E], where E=N*(N-1), zero means no edge.
    """
    w = np.asarray(w_low, dtype=float)
    E = w.size
    nz = w > 0
    E_nz = int(nz.sum())
    return {
        "E_low": int(E),
        "E_low_nonzero": int(E_nz),
        "nonzero_ratio": float(E_nz / (E + 1e-12)),
        "sparsity": float(1.0 - E_nz / (E + 1e-12)),
        "mean_w_all": float(w.mean()),
        "mean_w_nonzero": float(w[nz].mean()) if E_nz > 0 else 0.0,
    }


# =========================
# 5) Intra vs inter weights (requires edge_index_low aligned with w_low)
# =========================

def intra_inter_weight(plan: np.ndarray, edge_index_low: np.ndarray, w_low: np.ndarray,
                       only_nonzero: bool = True) -> dict:
    """
    plan: [N,1] or [N]
    edge_index_low: [2,E]
    w_low: [E]
    """
    g = np.asarray(plan).reshape(-1).astype(int)
    c = remap_groups(g)

    ei = np.asarray(edge_index_low)
    u = ei[0].astype(int)
    v = ei[1].astype(int)
    w = np.asarray(w_low, dtype=float)

    if only_nonzero:
        mask = w > 0
        u, v, w = u[mask], v[mask], w[mask]

    if w.size == 0:
        return {
            "intra_mean": np.nan,
            "inter_mean": np.nan,
            "intra_inter_gap": np.nan,
            "intra_ratio_nonzero": np.nan,
            "num_nonzero_edges_used": 0,
        }

    same = (c[u] == c[v])
    intra = w[same]
    inter = w[~same]
    intra_ratio = float(intra.size / (w.size + 1e-12))

    return {
        "intra_mean": float(intra.mean()) if intra.size else np.nan,
        "inter_mean": float(inter.mean()) if inter.size else np.nan,
        "intra_inter_gap": float(intra.mean() - inter.mean()) if (intra.size and inter.size) else np.nan,
        "intra_ratio_nonzero": intra_ratio,
        "num_nonzero_edges_used": int(w.size),
    }


# =========================
# 6) Compression stats
# =========================

def compression_stats(N: int, Nh: int, Eh: int, E_low_nonzero: int) -> dict:
    return {
        "compression_nodes": float(Nh / (N + 1e-12)),
        "compression_edges": float(Eh / (E_low_nonzero + 1e-12)),
    }


# =========================
# 7) Full per-step diagnostics
# =========================

def diagnostics_one_step(plan_t, h_edge_index_t, w_low_t, edge_index_low,
                         high_undirected=True) -> dict:
    cst = cluster_stats(plan_t)
    Nh = cst["Nh"]
    N = int(np.asarray(plan_t).reshape(-1).shape[0])

    hg = high_graph_stats(h_edge_index_t, Nh, undirected=high_undirected, remove_self_loops=True)
    lg = low_graph_stats(w_low_t)

    # For compression, use undirected Eh if available
    Eh = hg.get("Eh", hg.get("Eh_directed", 0))

    comp = compression_stats(N, Nh, Eh, lg["E_low_nonzero"])
    ii = intra_inter_weight(plan_t, edge_index_low, w_low_t, only_nonzero=True)

    return {**cst, **hg, **lg, **comp, **ii}


# =========================
# 8) Episode-level runner
# =========================

@dataclass
class DiagConfig:
    low_edge_order: str = "row_major"   # must match your edge_weights ordering
    high_undirected: bool = True
    save_csv: str = "diag_per_step.csv"
    plot: bool = True

def run_episode_diagnostics(pooling_plan, h_edge_index, edge_weights, eval_times, cfg: DiagConfig):
    """
    pooling_plan: list/array length T, each is plan [N,1] or [N]
    h_edge_index: list/array length T, each is edge_index_high [2, Eh_t]
    edge_weights: list/array length T, each is w_low [E]
    eval_times: optional, length T (timestamps or step indices)
    """
    T = len(pooling_plan)
    N = int(np.asarray(pooling_plan[0]).reshape(-1).shape[0])

    # rebuild low-level edge_index from N (complete directed, no self loops)
    edge_index_low = build_complete_directed_edge_index(N, order=cfg.low_edge_order)

    rows = []
    for t in range(T):
        row = diagnostics_one_step(
            plan_t=pooling_plan[t],
            h_edge_index_t=h_edge_index[t],
            w_low_t=edge_weights[t],
            edge_index_low=edge_index_low,
            high_undirected=cfg.high_undirected
        )
        row["t"] = int(t)
        if eval_times is not None:
            row["eval_time"] = float(eval_times[t])
        rows.append(row)

    df = pd.DataFrame(rows)

    # switch rate (needs previous)
    switch = [np.nan]
    for t in range(1, T):
        switch.append(switch_rate(pooling_plan[t-1], pooling_plan[t]))
    df["switch_rate"] = switch

    # summary stats over episode
    summary = df.drop(columns=["t", "eval_time"], errors="ignore").agg(["mean", "std"])
    # Make it easier to print / copy
    summary_T = summary.T  # index = metric, columns = mean/std

    if cfg.save_csv:
        df.to_csv(cfg.save_csv, index=False)
        summary_T.to_csv(cfg.save_csv.replace(".csv", "_summary.csv"))

    # Optional plots (very lightweight)
    if cfg.plot:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        axes[0].plot(df["t"], df["Nh"])
        axes[0].set_ylabel("Nh (#groups)")
        axes[0].grid(True)

        axes[1].plot(df["t"], df["compression_edges"])
        axes[1].set_ylabel("Eh / E_low_nonzero")
        axes[1].grid(True)

        axes[2].plot(df["t"], df["switch_rate"])
        axes[2].set_ylabel("switch rate")
        axes[2].set_xlabel("t")
        axes[2].grid(True)

        plt.tight_layout()
        # plt.show()
        plt.savefig("fig.pdf", dpi=300, bbox_inches='tight')

    return df, summary_T


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="gat_mappo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
            "eghn_mappo",     # 原始的EGHN
            "eghnv2_mappo",   # 可以处理多个等变特征
            "eghn_critic_mappo",  # critic使用EGHNv2
            "eghn_critic_happo",  # critic使用EGHNv2, actor使用HAPPO
            "eghn_actor_mappo",   # actor使用EGHNv2
            "egnn_actor_mappo",   # actor使用EGNNv3
            "egnn_critic_mappo",  # critic使用EGNNv3
            "egnn_critic_happo",
            "egnn_mappo",     # 原始论文的EGNN
            "egnnv2_mappo",   # EGHN论文里的EGNN
            "egnn_mix_mappo",   # 混合EGNN
            "egnnv3_mappo",   # 可以处理多个等变特征
            # "eghn_maddpg",  # 有问题
            # "egnn_maddpg",  # 有问题
            "gat_mappo",
            "gcn_mappo",
            "graphsage_mappo",
            "mappo_data_aug",
            "hie_mappo",
            'hie_critic_mappo',
            "hepn_mappo",      # 结构熵分层
            "hmf_mappo", # "hmf",
            "hama_mappo"
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="navigation",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            # "dexhands",
            "smacv2",
            "lag",
            "rendezvous",
            "pursuit",
            "navigation",
            "navigation_v2",
            "cover",
            "mujoco3d"
            # "obercooked"
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag, rendezvous.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )

    parser.add_argument(
        "--env_num1",
        type=float,
        default=0.0,
        help="Environment parameter 1, default 0.0."
    )

    parser.add_argument(
        "--env_num2",
        type=float,
        default=0.0,
        help="Environment parameter 2, default 0.0."
    )

    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        args["exp_name"] = all_config["main_args"]["exp_name"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line
    env_args["env_num1"] = args["env_num1"]
    env_args["env_num2"] = args["env_num2"]

    # if args["env"] == "dexhands":
    #     import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # start training
    from harl.runners import RUNNER_REGISTRY
    # algo_args["seed"]["seed"] = args["my_seed"]

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    if algo_args["train"]["train_flag"]:
        runner.run()
    # 这个episodes的意义是：
    # info, rewards, dis_matrix, _ = runner.transfor_exp(episodes=30)
    # top5 = np.partition(rewards, -10)[-10:]
    # print(top5.max())
    # top5_mean = float(np.mean(top5))
    # top5_std = float(np.std(top5))
    # print(f"top5 mean: {top5_mean}, top5 std: {top5_std}")
    
    # ESWA evaluation experiments
    # _, _, _, pooling_plan, h_edge_index, edge_weights, eval_times = runner.mod_render(episodes=1)
    # eval_times_arr = np.asarray(eval_times, dtype=float).reshape(-1)
    # k = min(512, eval_times_arr.size)
    # if k > 0:
    #     mink = np.partition(eval_times_arr, k - 1)[:k]
    #     mink_mean = float(np.mean(mink))
    #     mink_std = float(np.std(mink))
    #     # print(f"min{k} eval_times mean: {mink_mean:.1f}, std: {mink_std:.1f}")
    #     print(f"${mink_mean:.1f} \pm {mink_std:.1f}$")
    # else:
    #     print("eval_times is empty.")

    # cfg = DiagConfig(low_edge_order="row_major", plot=True, save_csv="diag_episode1.csv")
    # df_step, df_summary = run_episode_diagnostics(pooling_plan, h_edge_index, edge_weights, eval_times, cfg)
    # print(df_summary.sort_values("mean", ascending=False).head(30))

    # errs, _ = runner.sym_test(episodes=10)
    # print(errs)
    # for i, group_err in enumerate(errs, start=1):
    #     mean, ci_low, ci_high = bootstrap_mean_ci(group_err, n_boot=10000, ci=95, seed=42)
    #     print(f"${mean:.2f}~[{ci_low:.2f}, {ci_high:.2f}]$")


    
    # print(eval_times)
    # print(dis_matrix)
    # if args["env"] == 'rendezvous':
    #     info, _, dis_matrix = runner.mod_render(episodes=512)
    #     tra = []
    #     for i in info:
    #         tra.append(i['pos'])
    #     np.save('tra/rendezvous/esp_global.npy', tra)
    #     for step, m in enumerate(dis_matrix):
    #         m = m - np.diag(-1 * np.ones(m.shape[0]))
    #         dis = np.sum(m) / 2
    #         if dis <= 200:
    #             print(step)
    #             break
    # elif args["env"] == 'pursuit':
    #     info, _, dis_matrix = runner.mod_render(episodes=512)
    #     tra = []
    #     for i in info:
    #         tra.append(i["state"])
    #     np.save('tra/pursuit/egnn_global.npy', tra)
    #     for step, m in enumerate(dis_matrix):
    #         if m[-1][:-1].min() <= 5:
    #             print(step)
    #             break
    # elif args["env"] == 'cover':
    #     info, ratio, dis_matrix = runner.mod_render(episodes=512)
    #     tra = []
    #     for i in info:
    #         tra.append(i["state"])
    #     np.save('tra/cover/sage_global.npy', tra)
    #     for step, r in enumerate(ratio):
    #         if r >= 0.4:
    #             print(step)
    #             break
    # trajectories = runner.mod_render(episodes=1)
    # runner.show_envs.envs[0].make_ani(trajectories)
    # top_10 = np.sort(trajectories[1])[-10:]
    # print('${} \pm {}$'.format(round(np.mean(top_10), 2), round(np.std(top_10), 2)))
    
    runner.close()


if __name__ == "__main__":
    main()
