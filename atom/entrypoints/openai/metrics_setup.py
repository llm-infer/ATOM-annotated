"""Compose component-owned metrics for one API process."""

# 【译】为一个API进程组装各组件自有的指标。
#
# 【导读】API服务进程的Prometheus指标装配：创建AtomMetricsExporter（渲染/metrics）、RequestMetrics（TTFT等请求延迟直方图）、
#   StreamMetrics（流式交付指标），并把streaming_dispatch.longest_silence_seconds交给exporter，作为"最饥饿的流已等待多久"的数据源
#   （Prometheus：拉取式监控系统，定期HTTP GET /metrics，拿到"指标名{标签} 数值"这样的文本行并存成时间序列；
#    TTFT这类直方图指标按耗时分桶计数，监控端据此算P50 / P99）
# 调用方：api_server.py在模块加载时调用一次create_metrics_exporter()
# 与主干的关系：只做装配；指标本身定义在atom/metrics/下，【分支】指标与可观测性见第10批

import os

from prometheus_client import multiprocess, values

from atom.metrics.exporter import AtomMetricsExporter
from atom.metrics.request import RequestMetrics, StreamMetrics

from .streaming_dispatch import longest_silence_seconds


# 返回(exporter, request_metrics, stream_metrics)；设置了PROMETHEUS_MULTIPROC_DIR时走prometheus_client的多进程模式
def create_metrics_exporter() -> (
    tuple[AtomMetricsExporter, RequestMetrics, StreamMetrics]
):
    exporter = AtomMetricsExporter(stream_silence=longest_silence_seconds)
    registry = exporter.registry
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        if not values.ValueClass._multiprocess:
            raise RuntimeError(
                "Set PROMETHEUS_MULTIPROC_DIR before importing prometheus_client"
            )
        multiprocess.MultiProcessCollector(registry)
        # Register only the multiprocess collector, not its instruments too.
        # 【译】只注册多进程收集器，不再注册它的各个指标本身。
        # 解释：多进程模式下各进程的指标值写进PROMETHEUS_MULTIPROC_DIR下的文件，由MultiProcessCollector汇总；
        #   registry置为None后，RequestMetrics / StreamMetrics创建指标时就不再注册到exporter的registry上，避免重复
        registry = None
    request_metrics = RequestMetrics(registry)
    stream_metrics = StreamMetrics(registry)
    return exporter, request_metrics, stream_metrics
