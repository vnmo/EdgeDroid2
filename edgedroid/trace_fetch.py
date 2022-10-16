from loguru import logger


def fetch_traces(trace_names):
    from edgedroid import data as e_data

    for trace_name in trace_names:
        logger.info(f"Fetching traces: {trace_name}")
        e_data.load_default_trace(trace_name=trace_name)


def fetch_all_traces():
    logger.warning("Prefetching all available traces")
    from edgedroid import data as e_data

    fetch_traces(e_data.load._default_traces.keys())
