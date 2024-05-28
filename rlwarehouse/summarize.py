
import os 
from typing import List, Tuple, Union, Dict, Any
import jsonlines

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import colormaps
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

@staticmethod
def summarize(path: os.PathLike, result_path: os.PathLike, ncolsrows: Tuple[int], colormap: str = "Set1", smooth_window: int = 3, figsize: Tuple[int] = (16, 9)):
    """Summarize everything about the results
    """
    # ex: Agent.summarize("logs", "res", (6, 1), colormap="Set1", smooth_window=3)
    COLORMAP = colormaps.get(colormap)
    ncol, nrow = ncolsrows
    fig_score = plt.figure(figsize=figsize)
    fig_error = plt.figure(figsize=figsize)
    num_envs = len(os.listdir(path))
    assert ncol*nrow == len(os.listdir(path)), "Number of environments do not match layout"
    env_dict = {}
    for i, env in enumerate(os.listdir(path)):
        ax_score = fig_score.add_subplot(ncol, nrow, i+1)
        ax_score.set_title(env)
        ax_error = fig_error.add_subplot(ncol, nrow, i+1)
        ax_error.set_title(env)
        auc_scores, max_scores = np.zeros(num_envs), np.zeros(num_envs)
        algo_dict = {}
        env_path = os.path.join(path, env)
        for j, algo in enumerate(sorted(os.listdir(env_path))):
            algo_for_legend = "$"+algo.replace("_", "\\:").replace("@", "\\")+"$"
            algo_path = os.path.join(env_path, algo)
            results = {}
            for k, trial in enumerate(sorted(os.listdir(algo_path))):
                trial_dir = os.path.join(algo_path, trial)
                with jsonlines.open(os.path.join(trial_dir, "data.jsonl"), "r") as f: # reads as str
                    #print(os.path.join(trial_dir, "data.jsonl"))
                    for line in f:
                        step_ = line["step"]
                        if step_ not in results.keys(): 
                            results[step_] = {}
                        for param, valparam in line.items():
                            #print(f"{param} = {valparam}")
                            if param not in results[step_]:
                                results[step_][param] = []
                            results[step_][param].append(valparam)
                            #print(results[step_])
                            #print("---")
            step = list(results.keys())
            step.sort() # sort stuff
            #print(results)
            eval_score = np.array([results[s]["eval_score"] for s in step])
            if "eval_value_error" in results[step[0]]:
                eval_error = np.array([results[s]["eval_value_error"] for s in step])
            else:
                eval_error = None
            step = np.array(step) # make it also numpy array
            # -----eval score-----
            eval_score_mean = eval_score.mean(axis=1) # mean across trials
            eval_score_var = eval_score.var(axis=1) # var across trials
            eval_score_std = np.sqrt(eval_score_var)
            # moving average filtering for better visual
            #eval_score_mean_ma = np.convolve(eval_score_mean, np.ones(smooth_window)/smooth_window, mode="same")
            #eval_score_var_ma = np.convolve(eval_score_var, np.ones(smooth_window)/smooth_window, mode="same")
            eval_score_mean_ma = uniform_filter1d(eval_score_mean, smooth_window)
            eval_score_var_ma = uniform_filter1d(eval_score_var, smooth_window)
            eval_score_std_ma = np.sqrt(eval_score_var_ma)
            ax_score.plot(step, eval_score_mean_ma, color=COLORMAP(j), alpha=1.0, label=algo_for_legend)
            ax_score.fill_between(step, 
                eval_score_mean_ma - eval_score_std_ma,
                eval_score_mean_ma + eval_score_std_ma,
                facecolor=COLORMAP(j), alpha=0.3)
            # -----eval value error-----
            if eval_error is not None: 
                eval_error_mean = eval_error.mean(axis=1) # mean across trials
                eval_error_var = eval_error.var(axis=1) # var across trials
                eval_error_std = np.sqrt(eval_error_var)
                # moving average filtering for better visual
                #eval_error_mean_ma = np.convolve(eval_error_mean, np.ones(smooth_window)/smooth_window, mode="same")
                #eval_error_var_ma = np.convolve(eval_error_var, np.ones(smooth_window)/smooth_window, mode="same")
                eval_error_mean_ma = uniform_filter1d(eval_error_mean, smooth_window)
                eval_error_var_ma = uniform_filter1d(eval_error_var, smooth_window)
                eval_error_std_ma = np.sqrt(eval_error_var_ma)
                ax_error.plot(step, eval_error_mean_ma, color=COLORMAP(j), alpha=1.0, label=algo_for_legend)
                ax_error.fill_between(step, 
                    eval_error_mean_ma - eval_error_std_ma,
                    eval_error_mean_ma + eval_error_std_ma,
                    facecolor=COLORMAP(j), alpha=0.3)
            # 
            algo_dict[algo] = {
                "auc_scores": eval_score.mean(), 
                "max_scores": eval_score.max(), 
                "last_scores_mean": eval_score_mean[-1].mean(), 
                "last_scores_std": eval_score_std[-1]
            }
        env_dict[env] = algo_dict
        ax_score.set_ylabel("total reward", fontsize=10)
        ax_score.set_xlabel("# env interactions", fontsize=10)
        ax_error.set_ylabel("value error", fontsize=10)
        ax_error.set_xlabel("# env interactions", fontsize=10)
        #if i == 0: # only for first plot
        ax_score.legend()
        ax_error.legend()
        ax_score.grid()
        ax_error.grid()
        ax_score.xaxis.set_major_formatter(ticker.EngFormatter()) 
        ax_error.xaxis.set_major_formatter(ticker.EngFormatter()) 
    fig_score.tight_layout()
    fig_error.tight_layout()
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    fig_score.savefig(os.path.join(result_path, "score.png"))
    fig_error.savefig(os.path.join(result_path, "error.png"))
    fig_score.show()
    fig_error.show()
    env_df = pd.concat({env: pd.DataFrame.from_dict(algo_dict) for env, algo_dict in env_dict.items()})
    env_df.to_csv(os.path.join(result_path, "summary.csv"))
    return env_df