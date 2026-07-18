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

@app.post("/extract-graph")
def extract_graph(req: ExtractRequest):

    text = req.text

    entities = []
    relationships = []

    def add_entity(name, typ):
        if not any(e["name"] == name for e in entities):
            entities.append({"name": name, "type": typ})

    # Frameworks / Products
    known_frameworks = [
        "LangChain",
        "LlamaIndex",
        "OpenAI",
        "FAISS",
        "Qdrant",
        "ChromaDB",
        "Pinecone"
    ]

    for item in known_frameworks:
        if item.lower() in text.lower():
            typ = "Framework"
            if item == "OpenAI":
                typ = "Organization"
            add_entity(item, typ)

    # Persons
    for m in re.findall(r"([A-Z][a-z]+(?: [A-Z][a-z]+)+)", text):
        add_entity(m, "Person")

    patterns = [
        (r"(.+?) was created by (.+)", "CREATED"),
        (r"(.+?) was developed by (.+)", "DEVELOPED"),
        (r"(.+?) integrates with (.+)", "INTEGRATED_INTO"),
        (r"(.+?) authored (.+)", "AUTHORED"),
        (r"(.+?) hired (.+)", "HIRED")
    ]

    for pat, rel in patterns:
        m = re.search(pat, text, re.I)
        if m:
            left = m.group(1).strip().split()[-2:]
            right = m.group(2).strip().split(".")[0]
            relationships.append({
                "source": right,
                "target": " ".join(left),
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