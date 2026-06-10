"""
MCP Server — exposes RAG capabilities as Model Context Protocol tools.

What is MCP?
    The Model Context Protocol is an open standard that lets LLM applications
    connect to external tools and data sources in a uniform way. This server
    acts as a bridge between the LangGraph agent and the ChromaDB vector store.

Tools exposed:
    search_knowledge_base(query, n_results) → semantic search over stored docs
    add_document(content, source)           → add a new document at runtime
    knowledge_base_stats()                  → count of indexed documents

Transport:
    SSE (Server-Sent Events) over HTTP — suitable for networked / Docker setups.
    The agent connects to http://mcp-server:8080/sse inside Docker Compose.

RAG pipeline (this file handles the Retrieval part):
    Index time  : documents → sentence-transformer embeddings → ChromaDB
    Query time  : question  → embedding → cosine similarity search → top-k docs
"""

import os
import time
import uuid

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from fastmcp import FastMCP

# ── Configuration ─────────────────────────────────────────────────────────────
# Injected by Docker Compose; fall back to localhost for local development.
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# FastMCP creates the MCP server. The name is shown in client logs.
mcp = FastMCP("RAG Knowledge Base")

# Module-level cache so we create the ChromaDB client only once per process.
_chroma_client = None
_collection = None


def get_collection():
    """
    Return (or lazily create) the ChromaDB collection.

    ChromaDB stores documents as vectors. A 'collection' is equivalent to a
    table in a relational database. We use the DefaultEmbeddingFunction which
    wraps the 'all-MiniLM-L6-v2' sentence-transformer model (~22M parameters).
    This model maps any text to a 384-dimensional float vector.
    """
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        # DefaultEmbeddingFunction uses all-MiniLM-L6-v2 via onnxruntime
        ef = DefaultEmbeddingFunction()
        _collection = _chroma_client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=ef,
        )
    return _collection


# ── MCP Tools ─────────────────────────────────────────────────────────────────
# Each @mcp.tool() decorated function becomes a tool the LLM can call.
# The docstring is sent to the LLM as the tool description — write it clearly.

@mcp.tool()
def search_knowledge_base(query: str, n_results: int = 4) -> str:
    """
    Search the knowledge base for documents relevant to the query.
    Use this tool whenever you need factual information to answer a question.

    How it works:
        1. The query string is embedded using the same model as at index time.
        2. ChromaDB computes cosine similarity between the query vector and all
           stored document vectors.
        3. The top n_results documents are returned, sorted by relevance.

    Args:
        query:     Natural language question or keywords to search for.
        n_results: Number of documents to retrieve (default 4).

    Returns:
        Formatted string with document content, source, and relevance score.
    """
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return "The knowledge base is empty."

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, count),  # can't request more than what exists
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return "No relevant documents found."

    lines = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        source    = meta.get("source", "unknown")
        # ChromaDB returns L2 distance; convert to a 0-1 relevance score
        relevance = round(1 - dist, 3)
        lines.append(f"[{i}] source={source} relevance={relevance}\n{doc}")

    return "\n\n---\n\n".join(lines)


@mcp.tool()
def add_document(content: str, source: str = "manual") -> str:
    """
    Add a new document to the knowledge base.

    The document is embedded and stored in ChromaDB immediately.
    It becomes searchable on the next call to search_knowledge_base.

    Args:
        content: The text content of the document.
        source:  A label identifying the origin (e.g. 'wikipedia', 'manual').
    """
    collection = get_collection()
    doc_id = str(uuid.uuid4())  # unique ID required by ChromaDB
    collection.add(
        documents=[content],
        metadatas=[{"source": source}],
        ids=[doc_id],
    )
    return f"Document added (id={doc_id})."


@mcp.tool()
def knowledge_base_stats() -> str:
    """Return the number of documents currently indexed in the knowledge base."""
    collection = get_collection()
    return f"The knowledge base contains {collection.count()} documents."


# ── Startup helpers ───────────────────────────────────────────────────────────

def _wait_for_chroma(retries: int = 30, delay: float = 2.0) -> None:
    """
    Block until ChromaDB is accepting connections.
    Necessary because Docker Compose starts all containers roughly in parallel
    and ChromaDB may not be ready when this server starts.
    """
    for attempt in range(retries):
        try:
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            client.heartbeat()
            print("ChromaDB is ready.")
            return
        except Exception:
            print(f"Waiting for ChromaDB ({attempt + 1}/{retries})...")
            time.sleep(delay)
    raise RuntimeError("ChromaDB did not become ready in time.")


def _seed_data() -> None:
    """
    Pre-populate the knowledge base with sample documents on first start.

    The check 'if collection.count() > 0' makes this idempotent:
    re-running the server will not duplicate documents.
    """
    collection = get_collection()
    if collection.count() > 0:
        print(f"Knowledge base already has {collection.count()} documents — skipping seed.")
        return
    
    # ── Domain documents (anonymised) ─────────────────────────────────────────
    documents = [
        {
            "content": (
                "[ENTREPRISE] est une entreprise de coaching sportif et de team building fondée par [FONDATEUR], "
                "Coach Fondateur. Contact : [EMAIL], téléphone : [TELEPHONE], site web : [SITE_WEB]. "
                "[ENTREPRISE] propose notamment l'animation d'Olympiades Koh Lanta pour des groupes. "
                "Tous droits réservés © 2020. La reproduction ou distribution de tout contenu est interdite."
            ),
            "source": "kohlanta_presentation",
        },
        {
            "content": (
                "Olympiades Koh Lanta — Conseils d'animation : "
                "Pratiquer l'activité dans un grand parc où les participants peuvent se déplacer d'une épreuve à l'autre avec les cordes ondulatoires. "
                "Donner un rythme de course léger pour mettre les participants en condition. "
                "Mettre une musique correspondant à l'activité. "
                "Marquer un temps d'arrêt entre la première et la deuxième partie. "
                "Adapter les activités en fonction du groupe. "
                "S'amuser et prendre du plaisir à animer."
            ),
            "source": "kohlanta_conseils",
        },
        {
            "content": (
                "Olympiades Koh Lanta — Règles du jeu : "
                "Les participants sont séparés en deux équipes (les rouges et les jaunes). "
                "Chaque épreuve rapporte des points. L'équipe avec le plus de points gagne du temps à l'épreuve finale. "
                "Chaque point représente 5 secondes sur l'épreuve finale. "
                "L'équipe qui remporte l'épreuve finale remporte le Totem."
            ),
            "source": "kohlanta_regles",
        },
        {
            "content": (
                "Olympiades Koh Lanta — Partie 1, Matériel nécessaire : "
                "Sac en toile de jute, pierres de couleur (rouge et jaune), corde ondulatoire, allume-feu, plots, anneaux. "
                "Épreuve 1 — Sélection des équipes : Mettre des pierres rouges et jaunes dans un sac en toile de jute pour former deux équipes. "
                "Formule : 'on tend la main, on retourne et on ouvre'. Si nombre impair, ajouter une pierre de chaque couleur. "
                "La première épreuve consiste à trouver un nom d'équipe et un cri de guerre. "
                "Épreuve 2 — Équilibre : Utiliser les cordes ondulatoires reliées en deux posées au sol. "
                "Les deux équipes font un aller-retour, la première à faire passer tous ses membres remporte le point."
            ),
            "source": "kohlanta_partie1_ep1_2",
        },
        {
            "content": (
                "Olympiades Koh Lanta — Partie 1 (suite) : "
                "Épreuve 3 — Course poursuite : Un carré matérialisé au sol avec des plots. "
                "Chaque équipe porte la corde ondulatoire sur les épaules. "
                "Les deux équipes se placent à l'opposé l'une de l'autre et doivent rattraper l'adversaire. "
                "Si un membre ne peut plus continuer, toute l'équipe s'arrête avant qu'il lâche la corde, puis repart avec une personne en moins. "
                "Épreuve 4 — Allume-feu : Chaque équipe dispose d'un allume-feu. "
                "Objectif : allumer un feu et le maintenir allumé pendant 5 secondes. "
                "Matérialiser des zones sur béton, sable ou terre (pas sur herbe). Remplacer si temps de pluie."
            ),
            "source": "kohlanta_partie1_ep3_4",
        },
        {
            "content": (
                "Olympiades Koh Lanta — Partie 1 (suite) : "
                "Épreuve 5 — Tir à la corde : Une équipe de chaque côté, l'équipe qui ramène l'autre de son côté gagne. "
                "Possible en deux manches gagnantes. "
                "Épreuve 6 — Poteaux : Toute l'équipe alignée sur un banc ou surface surélevée (ou au sol). "
                "Le coach donne des consignes : lever le bras, lever une jambe, fermer les yeux, etc. "
                "Le point revient à l'équipe dont la dernière personne maintient la position. "
                "Épreuve 7 — Anneaux : Choisir un ou plusieurs représentants par équipe ou faire du 1v1. "
                "Objectif : se maintenir accroché aux anneaux le plus longtemps possible."
            ),
            "source": "kohlanta_partie1_ep5_6_7",
        },
        {
            "content": (
                "Olympiades Koh Lanta — Partie 2, Matériel nécessaire : "
                "21 baguettes, insectes comestibles, plots, scotch noir, pommes de terre, 2 bouteilles 1,5L vides, "
                "2 balles de tennis, 2 cerceaux ou plots, 2 bassines, gobelets rouges et jaunes. "
                "Épreuve 8 — Baguettes : 21 baguettes posées sur une table. Jeu en 1v1 ou équipe contre équipe. "
                "Les participants retirent 1, 2 ou 3 baguettes à tour de rôle. Celui qui retire la dernière baguette a perdu. "
                "Épreuve 9 — Dégustation d'insecte : Des insectes sont placés dans un récipient. "
                "Chaque participant mange un insecte et doit deviner son goût. "
                "Après leur réponse, les participants sont isolés pour ne pas influencer les suivants. "
                "L'équipe avec le plus de bonnes réponses remporte le point."
            ),
            "source": "kohlanta_partie2_ep8_9",
        },
        {
            "content": (
                "Olympiades Koh Lanta — Partie 2 (suite) : "
                "Épreuve 10 — Retrouve ton équipe à l'aveugle : Tout le monde met un bandeau sur les yeux. "
                "Le coach disperse les participants (20 pas tout droit chacun). "
                "Un leader par équipe (sans bandeau) guide son équipe vocalement. Les leaders ne se déplacent pas. "
                "Objectif : toute l'équipe rejoint son leader. "
                "Épreuve 11 — Trouve le plot à l'aveugle : 15 plots disposés à ~40m des participants, dont 1 avec du scotch noir en dessous. "
                "Chaque équipe choisit 2 personnes à l'aveugle, les autres sont guides vocaux depuis une zone fixe. "
                "La première équipe qui trouve le plot avec le scotch noir remporte le point."
            ),
            "source": "kohlanta_partie2_ep10_11",
        },
        {
            "content": (
                "Olympiades Koh Lanta — Épreuve finale (épreuve 12) en trois parties : "
                "Partie A — Eau : Chaque équipe envoie une personne chercher de l'eau dans une bassine pour remplir une bouteille de 1,5L. "
                "La première équipe à remplir et boucher la bouteille passe à la suite. "
                "Partie B — Tir à la bouteille : Chaque équipe utilise une balle de tennis pour faire tomber la bouteille d'eau adverse depuis une distance définie. "
                "Le coach définit le nombre de fois requis. "
                "Partie C — Pommes de terre : L'équipe doit transporter des pommes de terre coincées aux hanches, serrés épaule contre épaule, d'un point A à un point B. "
                "Nombre de pommes de terre = nombre de membres - 1. Si une tombe, on recommence au début."
            ),
            "source": "kohlanta_epreuve_finale",
        },
        {
            "content": (
                "Olympiades Koh Lanta — Épreuve 13 — La marche de la mariée : "
                "Un petit couloir est constitué avec des plots. "
                "Les participants reçoivent de la farine, de l'eau restante et d'autres éléments à envoyer sur 'la mariée'. "
                "L'équipe perdante ou uniquement la mariée reçoit les projectiles selon le choix du coach. "
                "Tout le monde fredonne la petite chanson de la mariée. "
                "Règles générales pour les intervenants : "
                "Au minimum 1 activité avec des insectes obligatoire. "
                "Au minimum 1 activité avec du matériel sportif hors plots (corde, TRX, élastique, gilet lesté, parachute ou chariot de résistance). "
                "Les intervenants peuvent adapter les activités en fonction du groupe et de leurs préférences."
            ),
            "source": "kohlanta_ep13_regles_generales",
        },
    ]

    # Bulk insert — ChromaDB embeds all documents in one batch for efficiency
    ids = [f"seed_{i}" for i in range(len(documents))]
    collection.add(
        documents=[d["content"] for d in documents],
        metadatas=[{"source": d["source"]} for d in documents],
        ids=ids,
    )
    print(f"Seeded {len(documents)} documents into the knowledge base.")


if __name__ == "__main__":
    _wait_for_chroma()
    _seed_data()
    # Pre-warm the collection so the first search request is fast
    get_collection()
    # Start the MCP server with SSE transport on port 8080.
    # The SSE endpoint will be available at http://0.0.0.0:8080/sse
    mcp.run(transport="sse", host="0.0.0.0", port=8080)
