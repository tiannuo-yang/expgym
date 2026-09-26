"""ExpGym: evaluating LLM agents under costly feedback, and PoolAct for multi-agent coordination."""
from expgym.agent import REGIMES, Regime, get_regime, run_agent
from expgym.envs import available_tasks, get_task, register_task
from expgym.poolact import run_pool

__version__ = "0.1.0"
__all__ = ["REGIMES", "Regime", "get_regime", "run_agent", "run_pool",
           "available_tasks", "get_task", "register_task"]
