"""Synthetic enterprise document corpus.

All documents describe a fictional company ("Northstar Retail Group") and exist
purely so the knowledge graph, retrieval, and eval have deterministic,
inspectable ground truth. No real company, person, or system is referenced.

Each passage intentionally spreads multi-hop facts across documents: no single
passage contains the full chain needed to answer the multi-hop eval questions,
which is exactly the gap GraphRAG is meant to close versus chunk-only RAG.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


DOCUMENTS: list[Document] = [
    Document(
        doc_id="org-chart",
        title="Org Chart — Engineering & Operations",
        text=(
            "Maria Chen leads the Data Platform Team. "
            "David Okafor leads the Checkout Team. "
            "Priya Nair leads the Support Operations Team. "
            "Tom Alvarez leads the Inventory Team. "
            "Sara Lindqvist reports to Maria Chen. "
            "James Park reports to David Okafor."
        ),
    ),
    Document(
        doc_id="platform-ownership",
        title="Platform Ownership Register",
        text=(
            "The Data Platform Team owns the Atlas Warehouse. "
            "The Data Platform Team owns the Beacon API Gateway. "
            "The Checkout Team owns the Checkout Service. "
            "The Inventory Team owns the StockSync Service."
        ),
    ),
    Document(
        doc_id="service-dependencies",
        title="Service Dependency Notes",
        text=(
            "The Checkout Service depends on the Beacon API Gateway. "
            "The Checkout Service depends on the Atlas Warehouse. "
            "The StockSync Service depends on the Atlas Warehouse. "
            "The Returns Portal depends on the Beacon API Gateway."
        ),
    ),
    Document(
        doc_id="support-operations",
        title="Support Operations Handbook",
        text=(
            "The Support Operations Team operates the Returns Portal. "
            "The Returns Portal handles returns for the Aero Blender Pro. "
            "The Returns Portal handles returns for the Lumen Desk Lamp. "
            "Escalations from the Returns Portal are routed to the Support Operations Team."
        ),
    ),
    Document(
        doc_id="supplier-catalog",
        title="Supplier Catalog — Q3",
        text=(
            "Helix Components supplies motors for the Aero Blender Pro. "
            "Helix Components supplies heating elements for the Aero Blender Pro. "
            "BrightPack supplies packaging for the Trail Backpack 40L. "
            "NordLogistics supplies LED panels for the Lumen Desk Lamp. "
            "NordLogistics supplies shipping cartons for the Trail Backpack 40L."
        ),
    ),
    Document(
        doc_id="product-catalog",
        title="Product Catalog — Home & Outdoor",
        text=(
            "The Aero Blender Pro is part of the Kitchen Appliances category. "
            "The Trail Backpack 40L is part of the Outdoor Gear category. "
            "The Lumen Desk Lamp is part of the Home Lighting category. "
            "The Aero Blender Pro uses a 900 watt motor. "
            "The Trail Backpack 40L uses recycled nylon fabric."
        ),
    ),
    Document(
        doc_id="warehouse-notes",
        title="Atlas Warehouse — Operations Notes",
        text=(
            "The Atlas Warehouse stores order history for the Checkout Service. "
            "The Atlas Warehouse stores inventory levels for the StockSync Service. "
            "Nightly reconciliation jobs run inside the Atlas Warehouse. "
            "The Atlas Warehouse is hosted in the Central Data Region."
        ),
    ),
    Document(
        doc_id="meeting-notes-may",
        title="Architecture Review — May Notes",
        text=(
            "Attendees noted that checkout latency improved after response caching was "
            "introduced at the gateway layer. The platform group will publish freshness "
            "SLAs for all warehouse stores in June. The checkout group confirmed its "
            "migration to the new gateway routes will finish before the July freeze."
        ),
    ),
]
