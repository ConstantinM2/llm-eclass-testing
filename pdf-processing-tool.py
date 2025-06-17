"""
IFAS – Feature-Driven Term Finder (Busy Cursor + Save)

• Reads eclass_15_fluid_power_51_data.json and drops nodes whose
  label contains “unspecified”, “parts”, or “accessories”.
• Scans the PDF for “hydraulic” vs “pneumatic” to set a domain filter.
• Embeds the first 300 chars of page 1 → top-10 matches.
• Shows first 200 chars of page 1 in a read-only text box.
• Lets you pick exactly one category via radio buttons.
• After “Next”:
    – extracts the top 30 KeyBERT phrases from the PDF
    – filters them by semantic match to the category’s JSON “features”
    – displays the remaining PDF-origin terms as checkboxes
    – lets you delete unchecked ones
    – lets you save the kept terms to a .txt file
• Shows a busy cursor during all heavy work.

Requires:
    pip install sentence-transformers torch PyPDF2 pdfplumber keybert
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import json, re, torch, PyPDF2, pdfplumber
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


def split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    return [p.strip() for p in parts if p.strip()]


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
            mask = torch.tensor(
                [self._matches_domain(n, domain_word) for n in self.nodes],
                dtype=torch.bool
            )
            if mask.any():
                nodes = [n for n, m in zip(self.nodes, mask) if m]
                emb = self.emb_full[mask]
            else:
                nodes, emb = self.nodes, self.emb_full
        else:
            nodes, emb = self.nodes, self.emb_full

        query_emb = self.model.encode(
            snippet, convert_to_tensor=True, normalize_embeddings=True
        )
        sims = util.cos_sim(query_emb, emb).flatten()
        vals, idxs = torch.topk(sims, min(k, len(nodes)))
        return [(nodes[i], vals[j].item()) for j, i in enumerate(idxs)]


def extract_full_text(path: str) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def clean_text(raw: str) -> str:
    t = re.sub(r"\d{3,}(?:\.\d+)?", "", raw)
    t = t.replace("-\n", "")
    return re.sub(r"\n{2,}", "\n\n", t)


class FinderGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IFAS – Feature-Driven Term Finder")
        self.geometry("900x900")

        tk.Button(self, text="Open PDF", font=("Segoe UI", 11),
                  command=self.open_pdf).pack(pady=8)

        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.preview = tk.Text(self, wrap="word", font=("Consolas", 10), height=6)
        self.preview.pack(fill="x", padx=10)
        self.preview.config(state="disabled")

        # Category selection
        self.results_frame = tk.Frame(self)
        self.results_frame.pack(fill="x", padx=10, pady=10)
        self.canvas = tk.Canvas(self.results_frame, height=200)
        self.scrollbar = tk.Scrollbar(self.results_frame,
                                      orient="vertical",
                                      command=self.canvas.yview)
        self.inner_frame = tk.Frame(self.canvas)
        self.inner_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0,0), window=self.inner_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.next_button = tk.Button(self, text="Next", font=("Segoe UI", 11),
                                     state="disabled", command=self.on_next)
        self.next_button.pack(pady=8)

        # Detected feature terms list
        tk.Label(self, text="Detected PDF Terms:", font=("Segoe UI", 11)).pack(pady=(10,0))
        self.terms_frame = tk.Frame(self)
        self.terms_frame.pack(fill="both", expand=True, padx=10, pady=(0,10))
        self.terms_canvas = tk.Canvas(self.terms_frame, height=250)
        self.terms_scrollbar = tk.Scrollbar(self.terms_frame,
                                            orient="vertical",
                                            command=self.terms_canvas.yview)
        self.terms_inner = tk.Frame(self.terms_canvas)
        self.terms_inner.bind(
            "<Configure>",
            lambda e: self.terms_canvas.configure(scrollregion=self.terms_canvas.bbox("all"))
        )
        self.terms_canvas.create_window((0,0), window=self.terms_inner, anchor="nw")
        self.terms_canvas.configure(yscrollcommand=self.terms_scrollbar.set)
        self.terms_canvas.pack(side="left", fill="both", expand=True)
        self.terms_scrollbar.pack(side="right", fill="y")

        self.terms_canvas.bind("<Enter>", lambda e: self.terms_canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.terms_canvas.bind("<Leave>", lambda e: self.terms_canvas.unbind_all("<MouseWheel>"))

        # Control buttons
        self.delete_button = tk.Button(self, text="Delete Unchecked Terms",
                                       font=("Segoe UI",11), state="disabled",
                                       command=self.delete_unchecked)
        self.delete_button.pack(pady=(0,5))

        self.save_button = tk.Button(self, text="Save Terms to TXT",
                                     font=("Segoe UI",11), state="disabled",
                                     command=self.save_terms)
        self.save_button.pack(pady=(0,10))

        self.results = []
        self.selected_index = tk.IntVar(value=-1)
        self.term_vars: list[tuple[tk.BooleanVar, tk.Checkbutton]] = []
        self.pdf_path = None

        json_path = Path(__file__).with_name("eclass_15_fluid_power_51_data.json")
        if not json_path.exists():
            messagebox.showerror("Startup error", f"JSON not found:\n{json_path}")
            self.destroy()
            return
        nodes = load_nodes(json_path)
        self.classifier = Classifier(nodes)
        self.kb = KeyBERT("all-MiniLM-L6-v2")

    def _on_mousewheel(self, event):
        self.terms_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def show_loading(self):
        self.config(cursor="watch")
        self.progress.pack(fill="x", padx=10, pady=5)
        self.progress.start()
        self.update_idletasks()

    def hide_loading(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.config(cursor="")
        self.update_idletasks()

    def open_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files","*.pdf")])
        if not path:
            return
        self.pdf_path = path
        self.show_loading()
        try:
            pdf = PyPDF2.PdfReader(path)
            full = "\n".join(p.extract_text() or "" for p in pdf.pages)
            page1 = pdf.pages[0].extract_text() or ""
        except Exception as e:
            self.hide_loading()
            messagebox.showerror("Read error", str(e))
            return

        snippet = page1[:200].replace("\n"," ")
        self.preview.config(state="normal")
        self.preview.delete("1.0","end")
        self.preview.insert("1.0", snippet)
        self.preview.config(state="disabled")

        has_h = re.search(r"\bhydraulic", full, re.I)
        has_p = re.search(r"\bpneumatic", full, re.I)
        domain = "hydraulic" if has_h and not has_p else \
                 "pneumatic" if has_p and not has_h else None

        self.results = self.classifier.top_k(page1[:300], domain, 10)

        for w in self.inner_frame.winfo_children():
            w.destroy()
        self.selected_index.set(-1)
        for i,(node,score) in enumerate(self.results):
            lbl,cid,_ = node_tuple(node)
            txt = f"{i+1}. {lbl} (ID {cid}) – {score:.3f}"
            rb = tk.Radiobutton(self.inner_frame, text=txt,
                                variable=self.selected_index, value=i,
                                anchor="w", justify="left", wraplength=800)
            rb.pack(fill="x", pady=2)

        self.next_button.config(state="normal")
        self.hide_loading()
        messagebox.showinfo("Loaded", f"{len(self.results)} candidates available")

    def on_next(self):
        idx = self.selected_index.get()
        if idx < 0 or idx >= len(self.results):
            messagebox.showwarning("No selection","Select exactly one category.")
            return

        node, score = self.results[idx]
        lbl, cid, _ = node_tuple(node)
        features = [f["preferred_name"] for f in node.get("features", [])]
        messagebox.showinfo(
            "Selected Category",
            f"{lbl} (ID {cid}) – {score:.3f}\n"
            f"{len(features)} built-in terms loaded for filtering"
        )

        self.show_loading()
        raw = extract_full_text(self.pdf_path)
        clean = clean_text(raw)

        # extract raw KeyBERT terms
        kws = self.kb.extract_keywords(
            clean,
            keyphrase_ngram_range=(1,3),
            stop_words="english",
            top_n=100
        )
        raw_terms = [kw for kw,_ in kws]

        # filter by semantic match to JSON features
        if features:
            feat_emb = self.classifier.model.encode(
                features, convert_to_tensor=True, normalize_embeddings=True
            )
            term_emb = self.classifier.model.encode(
                raw_terms, convert_to_tensor=True, normalize_embeddings=True
            )
            sims = util.cos_sim(term_emb, feat_emb)      # shape (R, F)
            max_sims, _ = sims.max(dim=1)               # best match per term

            # pair each term with its score and sort descending
            scored = list(zip(raw_terms, max_sims.tolist()))
            scored.sort(key=lambda x: x[1], reverse=True)
            found = [term for term, _ in scored]
        else:
            # no features → just use raw order
            found = raw_terms

        self.hide_loading()

        for w in self.terms_inner.winfo_children():
            w.destroy()
        self.term_vars.clear()
        for term in found:
            var = tk.BooleanVar(value=True)
            cb = tk.Checkbutton(self.terms_inner, text=term, variable=var,
                                anchor="w", justify="left", wraplength=800)
            cb.pack(fill="x", pady=2)
            self.term_vars.append((var, cb))

        self.delete_button.config(state="normal")
        self.save_button.config(state="normal")

    def delete_unchecked(self):
        kept = []
        for var, cb in self.term_vars:
            if var.get():
                kept.append((var, cb))
            else:
                cb.destroy()
        self.term_vars = kept

    def save_terms(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files","*.txt")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            for var, cb in self.term_vars:
                f.write(cb.cget("text") + "\n")
        messagebox.showinfo("Saved", f"Terms saved to:\n{path}")


if __name__ == "__main__":
    FinderGUI().mainloop()
