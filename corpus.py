"""Deterministic synthetic corpus with ground truth, for benchmarks and the soak test.

Each document describes one fictional internal service with a handful of unique
facts (owner team, datastore, port, region, ...). Because we generated the facts,
every question has a known gold document and gold answer — so retrieval quality
(recall@k, MRR, answer accuracy) can be *measured* under load, not guessed.

Deliberately hard for a lexical retriever: all 2,000 docs share the same template
vocabulary, names overlap ("Crimson Falcon" vs "Crimson Heron"), and a third of the
questions carry a typo or a paraphrase that never appears in the text.
"""

import random

ADJ = """amber arctic azure brisk cobalt coral crimson dusky ember feral frosted gilded hollow indigo ivory jade
lunar mellow nimble obsidian opal pale quiet rustic scarlet silent solar tidal umber velvet verdant wild""".split()
NOUN = """badger bison condor coyote crane dolphin falcon ferret gecko heron ibex jackal kestrel lemur lynx
marten moose narwhal ocelot osprey otter panther puffin quail raven salmon stoat tapir viper walrus wombat yak""".split()
TEAMS = ["Payments", "Search", "Identity", "Growth", "Logistics", "Billing", "Risk", "Messaging", "Catalog",
         "Analytics", "Platform", "Fulfilment", "Pricing", "Onboarding", "Support Tools", "Ledger"]
STORES = ["PostgreSQL", "Cassandra", "DynamoDB", "Redis", "ClickHouse", "MongoDB", "CockroachDB", "SQLite",
          "Elasticsearch", "FoundationDB", "ScyllaDB", "MySQL"]
LANGS = ["Rust", "Go", "Kotlin", "Python", "Elixir", "TypeScript", "Scala", "Zig", "Java", "Swift"]
REGIONS = ["eu-west-1", "us-east-2", "ap-south-1", "sa-east-1", "eu-north-1", "us-west-2", "ap-northeast-1",
           "af-south-1", "ca-central-1", "me-central-1"]
QUEUES = ["Kafka", "NATS", "RabbitMQ", "Pulsar", "SQS", "Redpanda"]

# field -> (heading, sentence template, question phrasings; the last one is a paraphrase
# that avoids the template's own wording)
FIELDS = {
    "team": ("Ownership", "{name} is owned and operated by the {v} team, who carry the pager for it.",
             ["Which team owns {name}?", "Who is responsible for {name}?", "who gets paged when {name} breaks"]),
    "store": ("Storage", "{name} persists its state in a {v} database, replicated across three zones.",
              ["Which database does {name} use?", "Where does {name} persist its state?",
               "what is the datastore behind {name}"]),
    "lang": ("Implementation", "The {name} codebase is written in {v} and built with a hermetic toolchain.",
             ["What language is {name} written in?", "What is the {name} codebase written in?",
              "which programming language powers {name}"]),
    "port": ("Networking", "{name} listens for traffic on port {v} behind the internal mesh.",
             ["What port does {name} listen on?", "Which port does {name} use for traffic?",
              "what port number is {name} reachable at"]),
    "region": ("Deployment", "{name} is deployed primarily to the {v} region with a warm standby elsewhere.",
               ["Which region is {name} deployed to?", "Where is {name} deployed?",
                "what is the primary deployment region of {name}"]),
    "queue": ("Messaging", "{name} publishes its domain events to {v} for downstream consumers.",
              ["Where does {name} publish its events?", "Which message bus does {name} publish to?",
               "what does {name} use for domain events"]),
    "slo": ("Reliability", "The latency objective for {name} is {v} milliseconds at the 99th percentile.",
            ["What is the latency objective for {name}?", "What is the p99 latency target of {name}?",
             "how fast must {name} respond at the 99th percentile"]),
}


def _typo(word: str, rng: random.Random) -> str:
    """Drop or swap one interior character."""
    if len(word) < 5:
        return word
    i = rng.randrange(1, len(word) - 2)
    return word[:i] + word[i + 1:] if rng.random() < 0.5 else word[:i] + word[i + 1] + word[i] + word[i + 2:]


def build(n_docs: int = 2000, seed: int = 7) -> tuple[list[dict], list[dict]]:
    """Returns (documents, questions).

    documents: [{"doc_name", "content"}]
    questions: [{"query", "doc_name", "answer", "kind"}]  kind in {plain, paraphrase, typo}
    """
    rng = random.Random(seed)
    names = [f"{a.capitalize()} {b.capitalize()}" for a in ADJ for b in NOUN]
    rng.shuffle(names)
    if n_docs > len(names):  # extend with numbered variants: "Crimson Falcon 2"
        names += [f"{n} {i}" for i in range(2, 2 + n_docs // len(names) + 1) for n in names]
    docs, questions = [], []
    for name in names[:n_docs]:
        facts = {"team": rng.choice(TEAMS), "store": rng.choice(STORES), "lang": rng.choice(LANGS),
                 "port": str(rng.randrange(2000, 9999)), "region": rng.choice(REGIONS),
                 "queue": rng.choice(QUEUES), "slo": str(rng.choice([15, 25, 40, 60, 80, 120, 200, 350]))}
        doc_name = "svc-" + name.lower().replace(" ", "-") + ".md"
        parts = [f"# {name}\n\n{name} is an internal service. This page is its runbook and reference."]
        order = list(FIELDS)
        rng.shuffle(order)
        for field in order:
            heading, template, phrasings = FIELDS[field]
            parts.append(f"## {name} {heading}\n\n" + template.format(name=name, v=facts[field]) +
                         " See the service catalog for the change history.")
            kind = rng.choice(["plain", "plain", "paraphrase", "typo"])
            if kind == "paraphrase":
                q = phrasings[2].format(name=name)
            else:
                q = rng.choice(phrasings[:2]).format(name=name)
                if kind == "typo":
                    first, rest = name.split(" ", 1)
                    q = q.replace(name, _typo(first.lower(), rng).capitalize() + " " + rest)
            questions.append({"query": q, "doc_name": doc_name, "answer": facts[field], "kind": kind})
        docs.append({"doc_name": doc_name, "content": "\n\n".join(parts) + "\n"})
    return docs, questions
