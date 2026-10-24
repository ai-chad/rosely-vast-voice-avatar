"""Vast.ai Serverless SDK worker; one request at a time per route on the RTX 3090."""
import os
from vastai import Worker, WorkerConfig, HandlerConfig, BenchmarkConfig, LogActionConfig

ADAPTER_PORT = int(os.getenv('SERVERLESS_ADAPTER_PORT', '8004'))
WORKSPACE = os.getenv('WORKSPACE', '/workspace')

def benchmark():
    return {'input': 'Hello from Rosely.'}

worker_config = WorkerConfig(
    model_server_url='http://127.0.0.1',
    model_server_port=ADAPTER_PORT,
    model_log_file=f'{WORKSPACE}/rosely-logs/serverless-adapter.log',
    model_healthcheck_url='/health',
    handlers=[
        HandlerConfig(route='/v1/tts', allow_parallel_requests=False,
            max_queue_time=180.0, workload_calculator=lambda payload: 1.0,
            benchmark_config=BenchmarkConfig(generator=benchmark, runs=2,
                                             concurrency=1, do_warmup=True)),
        HandlerConfig(route='/v1/stt', allow_parallel_requests=False,
            max_queue_time=180.0, workload_calculator=lambda payload: 1.0),
        HandlerConfig(route='/v1/avatar/generate', allow_parallel_requests=False,
            max_queue_time=900.0, workload_calculator=lambda payload: 20.0),
        # Pre-registers a character's motion clips when a video chat opens.
        HandlerConfig(route='/v1/ditto/warm', allow_parallel_requests=False,
            max_queue_time=300.0, workload_calculator=lambda payload: 5.0),
    ],
    log_action_config=LogActionConfig(
        on_load=['Application startup complete.'],
        on_error=['Traceback (most recent call last):', 'Application startup failed.'],
        on_info=[],
    ),
)
Worker(worker_config).run()
