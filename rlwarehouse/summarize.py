
import os 
from typing import List, Tuple, Union, Dict, Any
import jsonlines

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import colormaps
import matplotlib.ticker as ticker


@staticmethod
def summarize(path: os.PathLike, result_path: os.PathLike, ncolsrows: Tuple[int], colormap: str = "tab10", smooth_window: int = 3, figsize: Tuple[int] = (16, 9)):
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
    for i, env in enumerate(sorted(os.listdir(path))):
        ax_score = fig_score.add_subplot(ncol, nrow, i+1)
        ax_score.set_title(env)
        ax_error = fig_error.add_subplot(ncol, nrow, i+1)
        ax_error.set_title(env)
        auc_scores, max_scores = np.zeros(num_envs), np.zeros(num_envs)
        algo_dict = {}
        env_path = os.path.join(path, env)
        for j, algo in enumerate(sorted(os.listdir(env_path))):
            algo_for_legend = "$"+algo.replace("__", "\\quad").replace("@", "\\").replace("~", "/")+"$" 
            algo_path = os.path.join(env_path, algo)
            results = {}
            for k, trial in enumerate(sorted(os.listdir(algo_path))):
                trial_dir = os.path.join(algo_path, trial)
                with jsonlines.open(os.path.join(trial_dir, "data.jsonl"), "r") as f: # reads as str
                    #print(os.path.join(trial_dir, "data.jsonl"))
                    for line in f:
                        step_ = line["step"]
                        for param, valparam in line.items():
                            if param not in ["eval_score", "eval_value_error"]:
                                continue
                            else:
                                if step_ not in results.keys(): 
                                    results[step_] = {}                                
                            if param not in results[step_]:
                                results[step_][param] = []
                            results[step_][param].append(valparam)
            step = list(results.keys())
            step.sort() # sort stuff
            try:
                eval_score = np.array([results[s]["eval_score"] for s in step]) # shape: step, trial
                eval_error = np.array([results[s]["eval_value_error"] for s in step]) # shape: step, trial
            except:
                print(env)
                print(algo)
                print("4234234324")
                exit()

            step = np.array(step) # make it also numpy array
            num_steps = len(step)
            # calculate mean and std. 
            mean_eval_score = np.zeros(num_steps)         
            std_eval_score = np.zeros(num_steps)
            mean_eval_error = np.zeros(num_steps)                
            std_eval_error = np.zeros(num_steps)            
            for i in range(num_steps):
                lft_lim = smooth_window if i>=smooth_window else i
                rht_lim = smooth_window if i<=num_steps-smooth_window else num_steps - i
                eval_score_window = eval_score[i-lft_lim:i+rht_lim, :].flatten()
                eval_error_window = eval_error[i-lft_lim:i+rht_lim, :].flatten()
                # 
                mean_eval_score[i] = eval_score_window.mean()             
                std_eval_score[i] = eval_score_window.std()
                mean_eval_error[i] = eval_error_window.mean()                
                std_eval_error[i] = eval_error_window.std()
            # ----- eval score -----
            ax_score.plot(step, mean_eval_score, color=COLORMAP(j), alpha=1.0, label=algo_for_legend)
            ax_score.fill_between(step, 
                mean_eval_score - std_eval_score,
                mean_eval_score + std_eval_score,
                facecolor=COLORMAP(j), alpha=0.3)
            # ----- eval value error -----
            ax_error.plot(step, mean_eval_error, color=COLORMAP(j), alpha=1.0, label=algo_for_legend)
            ax_error.fill_between(step, 
                mean_eval_error - std_eval_error,
                mean_eval_error + std_eval_error,
                facecolor=COLORMAP(j), alpha=0.3)
            # 
            algo_dict[algo] = {
                "auc_scores": eval_score.mean(), 
                "max_scores": eval_score.max(), 
                "last_score_mean": mean_eval_score[-1], 
                "last_score_std": std_eval_score[-1]
            }
        env_dict[env] = algo_dict
        ax_score.set_ylabel("total reward", fontsize=10)
        ax_score.set_xlabel("# env interactions", fontsize=10)
        ax_error.set_ylabel("value error", fontsize=10)
        ax_error.set_xlabel("# env interactions", fontsize=10)
        #if i == 0: # only for first plot
        ax_score.legend(loc="upper left", framealpha=0.25, prop={'size': 8})
        ax_error.legend(loc="upper left", framealpha=0.25, prop={'size': 8})
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
