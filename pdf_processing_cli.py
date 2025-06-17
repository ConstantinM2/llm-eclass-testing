import argparse
import json
import re
from pathlib import Path
import PyPDF2
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT


def load_nodes(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    bag = []
    skip_re = re.compile(r"\b(?:unspecified|parts|accessories)\b", re.I)

    def walk(node):
        if not skip_re.search(node["label"]):
            bag.append(node)
        for ch in node.get("children", []):
            walk(ch)
    for root in data:
        walk(root)
    return bag


def node_tuple(node):
    return node["label"], node["classification_id"], node.get("description", "")


def extract_full_text(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def clean_text(raw: str) -> str:
    t = re.sub(r"\d{3,}(?:\.\d+)?", "", raw)
    t = t.replace("-\n", "")
    return re.sub(r"\n{2,}", "\n\n", t)


class Classifier:
    def __init__(self, nodes):
        self.nodes = nodes
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.emb_full = self.model.encode(
            [n["label"] for n in nodes],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    @staticmethod
    def _matches_domain(node, word: str) -> bool:
        patt = re.compile(word, re.I)
        return bool(patt.search(node["label"]) or patt.search(node.get("description", "")))

    def top_k(self, snippet: str, domain_word: str | None, k: int = 10):
        if domain_word:
            mask = [self._matches_domain(n, domain_word) for n in self.nodes]
            if any(mask):
                nodes = [n for n, m in zip(self.nodes, mask) if m]
                emb = self.emb_full[[i for i, m in enumerate(mask) if m]]
            else:
                nodes = self.nodes
                emb = self.emb_full
        else:
            nodes = self.nodes
            emb = self.emb_full

        query_emb = self.model.encode(snippet, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(query_emb, emb).flatten()
        vals, idxs = sims.topk(min(k, len(nodes)))
        return [(nodes[i], vals[j].item()) for j, i in enumerate(idxs)]


def main():
    parser = argparse.ArgumentParser(description="Extract terms from PDF and map to ECLASS")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--json", default="eclass_15_fluid_power_51_data.json", help="ECLASS JSON path")
    args = parser.parse_args()

    nodes = load_nodes(Path(args.json))
    clf = Classifier(nodes)
    kb = KeyBERT("all-MiniLM-L6-v2")

    pdf = PyPDF2.PdfReader(args.pdf)
    full = "\n".join(p.extract_text() or "" for p in pdf.pages)
    page1 = pdf.pages[0].extract_text() or ""

    has_h = re.search(r"\bhydraulic", full, re.I)
    has_p = re.search(r"\bpneumatic", full, re.I)
    domain = "hydraulic" if has_h and not has_p else "pneumatic" if has_p and not has_h else None

    candidates = clf.top_k(page1[:300], domain, 10)
    for i, (node, score) in enumerate(candidates, 1):
        lbl, cid, _ = node_tuple(node)
        print(f"{i}. {lbl} (ID {cid}) – {score:.3f}")

    idx = int(input("Choose category [1..{}]: ".format(len(candidates)))) - 1
    node, score = candidates[idx]
    lbl, cid, _ = node_tuple(node)
    features = [f["preferred_name"] for f in node.get("features", [])]

    print(f"Selected: {lbl} (ID {cid}) with {len(features)} features")

    clean = clean_text(full)
    kws = kb.extract_keywords(clean, keyphrase_ngram_range=(1,3), stop_words="english", top_n=100)
    raw_terms = [kw for kw, _ in kws]

    if features:
        feat_emb = clf.model.encode(features, convert_to_tensor=True, normalize_embeddings=True)
        term_emb = clf.model.encode(raw_terms, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(term_emb, feat_emb)
        max_sims, _ = sims.max(dim=1)
        scored = list(zip(raw_terms, max_sims.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        found = [t for t, _ in scored]
    else:
        found = raw_terms

    print("Top extracted terms:")
    for t in found[:30]:
        print("-", t)

if __name__ == "__main__":
    main()
