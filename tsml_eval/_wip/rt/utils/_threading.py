import functools
import inspect
import threading
from typing import Any, Callable

from aeon.utils.validation import check_n_jobs
from numba.np.ufunc import parallel as nbpar  # 更可靠的 numba 并行 API

# 线程本地栈，支持装饰器嵌套时“进出栈”恢复正确的旧值
_tls = threading.local()
if not hasattr(_tls, "numba_threads_stack"):
    _tls.numba_threads_stack = []  # type: ignore[attr-defined]


def _get_max_threads_safe() -> int:
    """尽可能获取 numba 后端允许的最大线程数。"""
    try:
        # 某些 numba 版本提供私有函数；若无则进入 except
        from numba.np.ufunc.parallel import _get_max_threads  # type: ignore
        return int(_get_max_threads())
    except Exception:
        # 回退：用当前线程数作为“上限”下界，至少为 1
        try:
            return max(1, int(nbpar.get_num_threads()))
        except Exception:
            return 1


def _safe_set_num_threads(n: int) -> None:
    """将线程数裁剪到允许范围并安全设置。"""
    n = int(n)
    max_thr = _get_max_threads_safe()
    n_eff = max(1, min(n, max_thr))
    try:
        nbpar.set_num_threads(n_eff)
    except Exception:
        # 后端被硬锁成 1 的场景
        try:
            nbpar.set_num_threads(1)
        except Exception:
            pass  # 实在不行就保持现状


def threaded(func: Callable) -> Callable:
    """根据 n_jobs 临时设置 numba 线程数，调用结束后安全恢复。

    期望被装饰函数有参数 'n_jobs'（位置或关键字均可）。
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # 1) 获取调用前的“当前 numba 线程数”（而非环境变量/active_count）
        try:
            current_threads = int(nbpar.get_num_threads())
        except Exception:
            current_threads = 1

        # 入栈，支持嵌套
        _tls.numba_threads_stack.append(current_threads)  # type: ignore[attr-defined]

        # 2) 解析 n_jobs（位置/关键字/默认值）
        if "n_jobs" in kwargs:
            n_jobs = kwargs["n_jobs"]
        else:
            sig = inspect.signature(func)
            params = sig.parameters
            if "n_jobs" in params:
                # 找到 n_jobs 的位置参数索引
                names = list(params.keys())
                idx = names.index("n_jobs")
                if idx < len(args):
                    n_jobs = args[idx]
                else:
                    default = params["n_jobs"].default
                    n_jobs = default if default is not inspect.Parameter.empty else None
            else:
                n_jobs = None  # 容错：没有 n_jobs 时不改线程数

        # 3) 设置新的线程数（裁剪到允许上限）
        try:
            if n_jobs is not None:
                adj = int(check_n_jobs(n_jobs))
                # 有些 check_n_jobs 可能返回 -1 表示“所有核”，这里仍要裁剪到上限
                max_thr = _get_max_threads_safe()
                target = max(1, min(adj if adj != -1 else max_thr, max_thr))
                _safe_set_num_threads(target)
        except Exception:
            # 出错就不改线程数
            pass

        try:
            return func(*args, **kwargs)
        finally:
            # 4) 安全恢复到“调用前”的线程数（再次裁剪 + 容错）
            try:
                prev = _tls.numba_threads_stack.pop()  # type: ignore[attr-defined]
            except Exception:
                prev = current_threads
            _safe_set_num_threads(prev)

    return wrapper
