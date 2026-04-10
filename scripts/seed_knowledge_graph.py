"""Seed the clinical knowledge graph.

Usage:
    python scripts/seed_knowledge_graph.py [--stats]

Builds the default in-memory KG and prints statistics.
In production with Neo4j, this would also persist to the database.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nstea.safety.knowledge_graph import build_default_knowledge_graph


def main():
    parser = argparse.ArgumentParser(description="Seed the clinical knowledge graph")
    parser.add_argument("--stats", action="store_true", help="Print detailed stats")
    args = parser.parse_args()

    print("Building clinical knowledge graph...")
    kg = build_default_knowledge_graph()

    print(f"\nKnowledge Graph Summary:")
    print(f"  Nodes: {kg.node_count}")
    print(f"  Edges: {kg.edge_count}")

    if args.stats:
        # Count by type
        node_types: dict[str, int] = {}
        for _, data in kg.graph.nodes(data=True):
            t = data.get("type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1

        edge_types: dict[str, int] = {}
        for _, _, data in kg.graph.edges(data=True):
            t = data.get("type", "unknown")
            edge_types[t] = edge_types.get(t, 0) + 1

        print(f"\n  Node types:")
        for t, count in sorted(node_types.items()):
            print(f"    {t}: {count}")

        print(f"\n  Edge types:")
        for t, count in sorted(edge_types.items()):
            print(f"    {t}: {count}")

        # Sample queries
        print("\n  Sample Queries:")
        interactions = kg.get_drug_interactions("Warfarin")
        print(f"    Warfarin interactions: {len(interactions)}")
        for i in interactions:
            print(f"      - {i['drug']} ({i['severity']}): {i['effect']}")

        contras = kg.get_contraindications("Metformin")
        print(f"    Metformin contraindications: {len(contras)}")
        for c in contras:
            print(f"      - {c['condition']}: {c['reason']}")

        cross = kg.get_allergy_cross_reactions("Aspirin Allergy")
        print(f"    Aspirin allergy cross-reactions: {len(cross)}")
        for cr in cross:
            print(f"      - {cr['drug']} (risk: {cr['risk_level']}, alt: {cr['alternative']})")

    print("\nKnowledge graph ready.")


if __name__ == "__main__":
    main()
