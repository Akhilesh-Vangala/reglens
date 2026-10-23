"""OpenTelemetry tracing using the GenAI semantic conventions (gen_ai.*). Spans go to a local JSONL
file; set OTEL_EXPORTER_OTLP_ENDPOINT to also ship them to any OTLP backend (e.g. Phoenix, Azure Monitor
via the OpenTelemetry Collector)."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult


class JsonlExporter(SpanExporter):
    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans):
        with self.lock, self.path.open("a") as fh:
            for s in spans:
                fh.write(json.dumps(json.loads(s.to_json())) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


_provider: TracerProvider | None = None


def tracer(path: Path | None = None):
    global _provider
    if _provider is None:
        _provider = TracerProvider(resource=Resource.create({"service.name": "reglens"}))
        if path:
            _provider.add_span_processor(SimpleSpanProcessor(JsonlExporter(path)))
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    return _provider.get_tracer("reglens", "0.1.0")


def shutdown() -> None:
    if _provider:
        _provider.shutdown()
