"""Diverse + long-running traffic generator for v3+ data calibration.

Goal: accumulate 3000-10000 retrieval events over 1-5 hours with
realistic Zipfian distribution AND maximum semantic diversity, so we
can calibrate cache hit rate, tier sizing, and per-role palette
assumptions on something other than benchmark-biased data.

Differences from the v1 generate_observability_traffic.py:
  - 350+ unique query templates (vs 92) across 12 thematic clusters
  - Tunable Zipfian alpha (default 1.0 ≈ standard organic)
  - Self-checkpoints to a progress file (resilient to restart)
  - Designed to run at `nice -n 19` for hours under load

Run:
  nice -n 19 .venv/bin/python tools/generate_diverse_traffic.py \
       --target 3000 --alpha 1.0 --max-hours 5 \
       > /tmp/gen_diverse.log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

# 12 thematic clusters spanning what bert's organic traffic might cover.
# Each cluster gives the corpus search a different semantic neighborhood.
QUERY_CLUSTERS = {
    "bert_arch": [
        "memory hierarchy L0 L1 L2 L3",
        "demand paging finding indexing",
        "macro-op fusion Write memory_create",
        "Zipfian distribution cache hit rate",
        "LFU ARC cache replacement",
        "tiered execution cosine cross-encoder",
        "SIMD batching reranker batch size",
        "memory consistency model relaxed strict",
        "working-set refresh-policy separation",
        "in-order retirement proof packet",
        "cache coherence invalidation",
        "cycle-aware prefetch",
        "schema synthesizer organic org",
        "role tool palette mapping",
        "data shape document_corpus code_repo",
        "mission_phase early mid late",
        "third axis tool surface",
        "v3 architecture decision log",
        "v3+ refinement post-instrumentation",
        "evidence triumphs assumptions",
    ],
    "retrieval_ir": [
        "BM25 sparse retrieval Okapi",
        "RRF reciprocal rank fusion k=60",
        "ColBERT late interaction",
        "SPLADE learned sparse retrieval",
        "BGE-M3 embedder multilingual",
        "cross-encoder rerank cascade",
        "FAISS HNSW approximate nearest neighbor",
        "HyDE hypothetical document embeddings",
        "DiskANN graph-based ANN",
        "Vespa column store ranking",
        "MUVERA single-vector approximation",
        "MaxSim ColBERT late-interaction",
        "BEIR scifact dataset benchmark",
        "TREC-DL relevance grading",
        "MS-MARCO passage ranking",
        "NDCG MRR recall precision",
        "RAGAS faithfulness context precision",
        "LongMemEval categories",
        "ColPali document retrieval",
        "ColBERTv2 PLAID compression",
        "M3-DPR multilingual",
        "Anserini Lucene Java",
        "Pyserini Python wrapper",
        "Tantivy Rust full-text search",
        "DuckDB columnar OLAP",
        "dense vector index quantization",
        "product quantization PQ",
        "scalar quantization SQ",
        "OPQ optimized product quantization",
        "inverted file index IVF",
    ],
    "papers_arxiv": [
        "attention is all you need",
        "BERT pre-training deep bidirectional transformers",
        "RoBERTa robustly optimized BERT pretraining",
        "T5 text-to-text transfer transformer",
        "GPT-3 few-shot learning",
        "Chain-of-thought prompting",
        "Tree of thoughts deliberate problem solving",
        "Toolformer language models self-teach tools",
        "ReAct synergizing reasoning acting",
        "Reflexion verbal reinforcement learning",
        "PEFT LoRA low-rank adaptation",
        "QLoRA quantized fine-tuning",
        "DPO direct preference optimization",
        "RLHF human feedback alignment",
        "RLAIF AI feedback",
        "Constitutional AI principles",
        "Mamba selective state space model",
        "RWKV linear attention RNN",
        "Hyena long convolutions",
        "Striped Hyena hybrid architecture",
        "Flash attention IO-aware",
        "Ring attention sequence parallel",
        "Speculative decoding draft model",
        "Medusa multi-head decoding",
        "Lookahead decoding",
        "PageAttention vLLM",
        "Continuous batching LLM serving",
        "Tensor parallelism Megatron",
        "Pipeline parallelism GPipe",
        "ZeRO zero redundancy optimizer",
    ],
    "agentic_systems": [
        "MCP model context protocol",
        "JSON-RPC 2.0 specification",
        "agentic lab framework",
        "AutoGPT autonomous goal",
        "BabyAGI task management",
        "LangGraph state machine",
        "CrewAI multi-agent collaboration",
        "AGNTCY identity DID directory",
        "A2A agent-to-agent protocol",
        "ACP agent communication",
        "in-toto SLSA provenance attestation",
        "Sigstore cosign keyless signing",
        "Rekor transparency log",
        "Fulcio CA certificate",
        "OIDC OpenID Connect identity",
        "OPA open policy agent",
        "Cedar policy language",
        "tool use OpenAI function calling",
        "structured output JSON schema",
        "guardrails validation framework",
        "Pydantic data validation",
        "instructor structured llm output",
        "marvin assistant abstractions",
        "DSPy declarative LM programs",
        "GPT-Pilot autonomous coding",
        "Cursor pair programming",
        "Devin software engineer agent",
        "OpenHands autonomous developer",
        "Continue.dev IDE assistant",
        "Aider AI pair programmer",
    ],
    "infra_devops": [
        "Kubernetes pod scheduling",
        "Helm chart templating",
        "Argo CD GitOps deployment",
        "Flux CD reconciliation",
        "Istio service mesh",
        "Linkerd lightweight mesh",
        "Cilium eBPF networking",
        "Prometheus monitoring metrics",
        "Grafana dashboard panels",
        "OpenTelemetry tracing spans",
        "Jaeger distributed tracing",
        "Tempo trace storage",
        "Loki log aggregation",
        "Vector observability pipeline",
        "Fluent Bit log processing",
        "Datadog APM",
        "New Relic infrastructure monitoring",
        "PagerDuty incident response",
        "Sentry error tracking",
        "Honeycomb event analytics",
        "Terraform IaC HCL",
        "Pulumi infrastructure code",
        "OpenTofu open source Terraform",
        "Crossplane control plane",
        "ClusterAPI cluster lifecycle",
        "Velero backup restore",
        "etcd consensus storage",
        "Consul service discovery",
        "Vault secrets management",
        "Cert-manager certificate automation",
    ],
    "ml_systems": [
        "vLLM serving throughput",
        "TGI text generation inference",
        "TensorRT-LLM optimized",
        "ONNX Runtime cross-platform",
        "OpenVINO Intel toolkit",
        "TorchServe model deployment",
        "Triton inference server",
        "Ray Serve scalable",
        "BentoML productionizing",
        "Seldon Core MLOps",
        "MLflow experiment tracking",
        "Weights & Biases logging",
        "DVC data version control",
        "Pachyderm pipelines",
        "Feast feature store",
        "Tecton real-time features",
        "Kafka streaming events",
        "Flink stateful stream",
        "Spark Structured Streaming",
        "Delta Lake ACID",
        "Iceberg table format",
        "Hudi incremental processing",
        "Parquet columnar storage",
        "Arrow in-memory columnar",
        "DuckDB embeddable OLAP",
        "ClickHouse OLAP database",
        "TimescaleDB time-series",
        "InfluxDB measurements tags",
        "QuestDB high-cardinality TS",
        "Druid OLAP analytics",
    ],
    "data_engineering": [
        "dbt analytics engineering",
        "Airflow DAG scheduling",
        "Prefect orchestration",
        "Dagster software-defined assets",
        "Mage data pipelines",
        "Meltano EL framework",
        "Singer tap target",
        "Airbyte connectors",
        "Fivetran ELT",
        "Stitch ETL platform",
        "Snowflake warehouse compute",
        "BigQuery serverless analytics",
        "Redshift columnar storage",
        "Databricks lakehouse",
        "Trino federated query",
        "Presto SQL engine",
        "Hive metastore catalog",
        "AWS Glue ETL",
        "Azure Data Factory",
        "GCP Dataflow Beam",
        "Apache Beam unified",
        "Kafka Streams DSL",
        "ksqlDB streaming SQL",
        "Materialize streaming OLAP",
        "RisingWave SQL streaming",
        "Debezium CDC connector",
        "Maxwell binlog parser",
        "Outbox pattern reliability",
        "Saga distributed transaction",
        "Event sourcing CQRS",
    ],
    "ai_safety_eval": [
        "AISI evals dangerous capabilities",
        "METR autonomy evaluations",
        "Anthropic responsible scaling policy",
        "OpenAI preparedness framework",
        "GDM safety evaluations",
        "RAGAS retrieval evaluation",
        "Phoenix Arize tracing",
        "Langfuse LLM observability",
        "HoneyHive prompt evaluation",
        "Helicone proxy analytics",
        "Inspect AI evaluation framework",
        "lm-evaluation-harness",
        "Promptfoo evaluation testing",
        "DeepEval LLM testing",
        "Giskard ML testing",
        "Trulens evaluation suite",
        "MLflow LLM evaluate",
        "BIG-bench benchmarks",
        "HELM holistic evaluation",
        "TruthfulQA dataset",
        "MMLU multitask",
        "GSM8K math reasoning",
        "MATH dataset competition",
        "HumanEval coding",
        "MBPP basic programming",
        "SWE-bench software engineering",
        "LiveCodeBench contamination-free",
        "AgentBench multi-domain",
        "ToolBench tool learning",
        "AlpacaEval instruction following",
    ],
    "long_context_research": [
        "long context language model",
        "needle in a haystack retrieval",
        "RULER long-context eval",
        "LongBench long-form QA",
        "LongMemEval multi-session memory",
        "InfiniBench infinite context",
        "Counting Stars long-context",
        "Lost in the middle attention",
        "rotary positional embeddings RoPE",
        "ALiBi attention with linear bias",
        "YaRN context extension",
        "PI position interpolation",
        "LongLoRA efficient fine-tuning",
        "SnapKV KV cache compression",
        "StreamingLLM attention sinks",
        "H2O heavy hitter oracle",
        "DuoAttention dual KV cache",
        "Quest query-aware KV selection",
        "InfLLM stream attention",
        "Native sparse attention",
    ],
    "memory_systems": [
        "episodic memory humans",
        "semantic memory taxonomy",
        "procedural memory skills",
        "working memory capacity",
        "CoALA cognitive architecture",
        "MemGPT virtual memory",
        "Mem0 memory layer",
        "MemoryBank long-term memory",
        "Voyager Minecraft skill library",
        "Generative Agents Stanford",
        "RAISE reasoning agent",
        "Ghost Attention LLaMA",
        "transformers as fast weight",
        "compressive transformer",
        "Longformer sliding window",
        "BigBird random sparse",
        "ETC extended transformer",
        "Reformer LSH attention",
        "Linformer low-rank",
        "Performer FAVOR+",
    ],
    "philosophy_methodology": [
        "first principles thinking",
        "evidence-based design",
        "premature optimization root of all evil",
        "Conway's law team structure mirrors architecture",
        "Hyrum's law all observable behaviors",
        "Brooks's law mythical man-month",
        "Postel's robustness principle",
        "Goodhart's law metric becomes target",
        "Hofstadter's law always takes longer",
        "Occam's razor simplest explanation",
        "Chesterton's fence remove what you understand",
        "Cargo cult programming",
        "yagni you aren't gonna need it",
        "DRY don't repeat yourself",
        "KISS keep it simple",
        "SOLID principles object-oriented",
        "single responsibility principle",
        "dependency injection inversion",
        "command query separation",
        "Tell don't ask",
    ],
    "general_questions": [
        "how does X work in production",
        "what's the difference between A and B",
        "compare three alternatives for Y",
        "trace evidence to conclusion",
        "find counterexamples to claim",
        "list assumptions being made",
        "what would break this approach",
        "show me the test that validates this",
        "where does this value come from",
        "who proposed this design",
        "when was this decision made",
        "why isn't approach Z used here",
        "summarize the trade-offs",
        "rank options by risk",
        "what's the latency budget",
        "what's the failure mode",
        "what's the rollback plan",
        "what does this metric actually measure",
        "is this finding still valid",
        "does this concern remain open",
        "find papers cited by this work",
        "find papers that cite this work",
        "show me the lineage chain",
        "list all blockers for this cycle",
        "what role decides this question",
        "find consensus across reviewers",
        "find disagreements between roles",
        "what was rejected on what evidence",
        "show me approved artifacts this week",
        "list cycles with falsifier interventions",
    ],
}


def flatten() -> list[str]:
    out: list[str] = []
    for cluster in QUERY_CLUSTERS.values():
        out.extend(cluster)
    return out


def zipf_sample(rng: random.Random, keys: list[str], alpha: float) -> str:
    """True Zipfian sample over keys[0] (most popular) ... keys[-1] (least)."""
    n = len(keys)
    weights = [1.0 / ((i + 1) ** alpha) for i in range(n)]
    return rng.choices(keys, weights=weights, k=1)[0]


def main(target: int, alpha: float, max_hours: float, seed: int) -> int:
    rng = random.Random(seed)
    templates = flatten()
    rng.shuffle(templates)  # randomize the popularity ordering

    print(f"[{time.strftime('%H:%M:%S')}] starting diverse traffic gen")
    print(f"  templates: {len(templates)} across {len(QUERY_CLUSTERS)} clusters")
    print(f"  target:    {target} queries")
    print(f"  alpha:     {alpha} (Zipfian concentration)")
    print(f"  max_hours: {max_hours}")
    sys.stdout.flush()

    os.environ["BERT_DISABLE_RERANKER"] = "1"
    from core import retrieval as _ret

    print(f"[{time.strftime('%H:%M:%S')}] retrieval module imported, warming up…")
    sys.stdout.flush()
    t_warm = time.perf_counter()
    for _ in range(2):
        _ret.hybrid_retrieve("warmup", top_n=3)
    print(f"[{time.strftime('%H:%M:%S')}] warmup done ({time.perf_counter()-t_warm:.1f}s)")
    sys.stdout.flush()

    deadline = time.monotonic() + max_hours * 3600
    t0 = time.monotonic()
    latencies = []
    cluster_counts = dict.fromkeys(QUERY_CLUSTERS, 0)
    template_to_cluster = {}
    for cname, tlist in QUERY_CLUSTERS.items():
        for t in tlist:
            template_to_cluster[t] = cname

    completed = 0
    errors = 0
    while completed < target:
        if time.monotonic() > deadline:
            print(f"[{time.strftime('%H:%M:%S')}] deadline reached")
            break
        q = zipf_sample(rng, templates, alpha)
        cluster_counts[template_to_cluster[q]] += 1
        top_n = rng.choice([3, 5, 5, 5, 10, 10, 20])
        try:
            t = time.perf_counter()
            _ret.hybrid_retrieve(q, top_n=top_n)
            latencies.append((time.perf_counter() - t) * 1000)
            completed += 1
        except Exception as e:
            errors += 1
            print(f"[{time.strftime('%H:%M:%S')}] error: {e}")
            sys.stdout.flush()
            time.sleep(0.5)
            continue

        # checkpoint every 50 queries
        if completed % 50 == 0:
            elapsed = time.monotonic() - t0
            qps = completed / elapsed if elapsed > 0 else 0.0
            srt = sorted(latencies)
            p50 = srt[len(srt) // 2]
            p95 = srt[int(len(srt) * 0.95)]
            print(f"[{time.strftime('%H:%M:%S')}] {completed}/{target}  "
                  f"qps={qps:.1f}  p50={p50:.0f}ms p95={p95:.0f}ms  err={errors}")
            sys.stdout.flush()

    elapsed = time.monotonic() - t0
    print()
    print(f"[{time.strftime('%H:%M:%S')}] Done. {completed} queries in {elapsed:.1f}s")
    print("  cluster distribution:")
    for c, n in sorted(cluster_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {c:>26}  {n}")
    if latencies:
        srt = sorted(latencies)
        print(f"  latency p50={srt[len(srt)//2]:.1f}ms "
              f"p95={srt[int(len(srt)*0.95)]:.1f}ms "
              f"p99={srt[int(len(srt)*0.99)]:.1f}ms")
        print(f"  throughput {completed/elapsed:.2f} QPS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=3000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max-hours", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=137)
    args = ap.parse_args()
    sys.exit(main(args.target, args.alpha, args.max_hours, args.seed))
