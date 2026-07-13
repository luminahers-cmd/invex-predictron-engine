"""Technology evidence provider — retrieves domain knowledge for tech stacks.

Gathers contextual facts about the technologies in the startup's stack.
Each fact is an objective observation about the technology, not a
conclusion about the startup.
"""

from __future__ import annotations

from predictron_engine.evidence.evidence_models import EvidenceItem

TECH_EVIDENCE: dict[str, list[dict[str, str]]] = {
    "python": [
        {"category": "ecosystem", "statement": "Python has a large, mature ecosystem with extensive libraries."},
        {"category": "talent", "statement": "Python developers are widely available in the talent market."},
        {"category": "performance", "statement": "Python may require optimization for compute-intensive workloads."},
    ],
    "javascript": [
        {"category": "ecosystem", "statement": "JavaScript has the largest language ecosystem globally."},
        {"category": "talent", "statement": "JavaScript developers are the most available talent pool."},
        {"category": "versatility", "statement": "JavaScript enables full-stack development with one language."},
    ],
    "typescript": [
        {"category": "ecosystem", "statement": "TypeScript adds type safety to the JavaScript ecosystem."},
        {"category": "talent", "statement": "TypeScript adoption is growing rapidly across teams."},
    ],
    "react": [
        {"category": "ecosystem", "statement": "React has the largest frontend framework community."},
        {"category": "competition", "statement": "React expertise is a baseline for many frontend roles."},
    ],
    "aws": [
        {"category": "market_position", "statement": "AWS is the leading cloud provider by market share."},
        {"category": "switching_costs", "statement": "Deep AWS integration creates significant switching costs."},
    ],
    "gcp": [
        {"category": "market_position", "statement": "GCP is strong in data analytics and ML workloads."},
        {"category": "competition", "statement": "GCP competes aggressively on price and AI/ML capabilities."},
    ],
    "azure": [
        {"category": "market_position", "statement": "Azure has strong enterprise penetration via Microsoft."},
        {"category": "switching_costs", "statement": "Enterprise Microsoft agreements drive natural Azure adoption."},
    ],
    "docker": [
        {"category": "ecosystem", "statement": "Docker is the standard for containerized deployment."},
        {"category": "operations", "statement": "Containers add complexity but improve scalability."},
    ],
    "kubernetes": [
        {"category": "operations", "statement": "Kubernetes provides production-grade container orchestration."},
        {"category": "complexity", "statement": "Kubernetes overhead requires dedicated DevOps expertise."},
    ],
    "postgresql": [
        {"category": "ecosystem", "statement": "PostgreSQL is the leading open-source relational database."},
        {"category": "operations", "statement": "PostgreSQL scales well for most application workloads."},
    ],
    "mongodb": [
        {"category": "ecosystem", "statement": "MongoDB is the leading document-oriented NoSQL database."},
        {"category": "operations", "statement": "MongoDB simplifies schema design for evolving applications."},
    ],
    "tensorflow": [
        {"category": "ecosystem", "statement": "TensorFlow is a mature ML framework with deployment tools."},
        {"category": "competition", "statement": "TensorFlow competes with PyTorch for deep learning adoption."},
    ],
    "pytorch": [
        {"category": "ecosystem", "statement": "PyTorch is the dominant framework for ML research."},
        {"category": "talent", "statement": "PyTorch expertise is common among ML engineers."},
    ],
    "blockchain": [
        {"category": "regulatory", "statement": "Blockchain applications face evolving regulatory frameworks."},
        {"category": "competition", "statement": "Blockchain ecosystems are highly competitive and fast-changing."},
    ],
}


class TechnologyEvidenceProvider:
    """Gathers domain evidence based on the startup's technology stack."""

    def gather(self, features: object) -> list[EvidenceItem]:
        from predictron_engine.models.extracted_features import ExtractedFeatures

        if not isinstance(features, ExtractedFeatures):
            return []
        if not features.technology_stack:
            return []

        items: list[EvidenceItem] = []
        for tech in features.technology_stack:
            tech_lower = tech.lower()
            if tech_lower not in TECH_EVIDENCE:
                continue
            for fact in TECH_EVIDENCE[tech_lower]:
                items.append(
                    EvidenceItem(
                        domain="technology",
                        category=fact["category"],
                        statement=fact["statement"],
                        source=f"knowledge/tech:{tech_lower}",
                        relevance_score=1.0,
                    )
                )
        return items
