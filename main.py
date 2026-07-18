from fastapi import FastAPI
from pydantic import BaseModel
import re
from collections import defaultdict

app = FastAPI()

# ---------------- Models ----------------

class ExtractRequest(BaseModel):
    chunk_id: str
    text: str

class GraphQueryRequest(BaseModel):
    question: str
    graph: dict

class CommunityRequest(BaseModel):
    community_id: str
    entities: list[str]
    relationships: list[dict]

# ---------------- Root ----------------

@app.get("/")
def root():
    return {"status": "ok"}

# ---------------- Extract Graph ----------------

from fastapi import APIRouter
 

@app.post("/extract-graph")
def extract_graph(req: ExtractRequest):

    text = req.text

    entities = []
    relationships = []

    def add_entity(name, typ):
        name = name.strip(" .,")
        if name and not any(e["name"] == name for e in entities):
            entities.append({"name": name, "type": typ})

    # -------- Entity Extraction --------

    # Person
    for m in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
        add_entity(m, "Person")

    # Organizations
    org_patterns = [
        r"\b([A-Z][A-Za-z0-9& ]+(?:Inc|Labs|AI|Research|Company|Corporation|Google|Microsoft|Meta|OpenAI))\b"
    ]

    for p in org_patterns:
        for m in re.findall(p, text):
            add_entity(m, "Organization")

    # Framework / Product
    for m in re.findall(r"\b([A-Z][A-Za-z0-9_-]+)\b", text):
        if any(e["name"] == m for e in entities):
            continue

        if m in [
            "LangChain",
            "LlamaIndex",
            "FAISS",
            "Qdrant",
            "ChromaDB",
            "Milvus",
            "Pinecone",
            "Weaviate",
            "OpenAI",
            "Anthropic",
            "Gemini",
            "Claude"
        ]:
            typ = "Framework"
            if m in ["OpenAI", "Anthropic"]:
                typ = "Organization"
            add_entity(m, typ)

    # -------- Relationship Extraction --------

    rules = [

        (r"(.+?) was created by (.+)", "CREATED"),
        (r"(.+?) was developed by (.+)", "DEVELOPED"),
        (r"(.+?) was founded by (.+)", "FOUNDED"),
        (r"(.+?) founded (.+)", "FOUNDED"),
        (r"(.+?) created (.+)", "CREATED"),
        (r"(.+?) developed (.+)", "DEVELOPED"),
        (r"(.+?) integrates with (.+)", "INTEGRATED_INTO"),
        (r"(.+?) integrated into (.+)", "INTEGRATED_INTO"),
        (r"(.+?) uses (.+)", "INTEGRATED_INTO"),
        (r"(.+?) built on (.+)", "INTEGRATED_INTO"),
        (r"(.+?) hired (.+)", "HIRED"),
        (r"(.+?) authored (.+)", "AUTHORED"),
        (r"(.+?) wrote (.+)", "AUTHORED"),
    ]

    for pat, rel in rules:

        m = re.search(pat, text, re.I)

        if m:

            a = m.group(1).strip(" .")
            b = m.group(2).strip(" .")

            relationships.append({
                "source": b,
                "target": a,
                "relation": rel
            })

    return {
        "entities": entities,
        "relationships": relationships
    }


# ---------------- Graph Query ----------------

@app.post("/graph-query")
def graph_query(req: GraphQueryRequest):

    rels = req.graph.get("relationships", [])

    graph = defaultdict(list)

    for r in rels:
        graph[r["source"]].append((r["target"], r["relation"]))
        graph[r["target"]].append((r["source"], r["relation"]))

    q = req.question.lower()

    target = None

    for e in req.graph.get("entities", []):
        if e["name"].lower() in q:
            target = e["name"]

    if target is None:
        return {
            "answer": "Unknown",
            "reasoning_path": [],
            "hops": 0
        }

    visited = set([target])
    queue = [(target, [target])]

    while queue:

        node, path = queue.pop(0)

        for nxt, rel in graph[node]:

            if nxt in visited:
                continue

            visited.add(nxt)

            new_path = path + [nxt]

            if rel in ["CREATED", "DEVELOPED", "FOUNDED", "AUTHORED"]:
                return {
                    "answer": nxt,
                    "reasoning_path": new_path,
                    "hops": len(new_path) - 1
                }

            queue.append((nxt, new_path))

    return {
        "answer": target,
        "reasoning_path": [target],
        "hops": 0
    }

# ---------------- Community Summary ----------------

@app.post("/community-summary")
def community_summary(req: CommunityRequest):

    names = ", ".join(req.entities)

    rels = ", ".join(
        r["relation"] for r in req.relationships
    )

    summary = (
        f"This community contains {names}. "
        f"Relationships include {rels}."
    )

    return {
        "community_id": req.community_id,
        "summary": summary
    }