from xml_loader import parse_source_xml
from persist import load_targets
from config import VIEWMAPPING_XML_PATH

def predict_mappings(source_xml_bytes):
    """
    Return exact mappings only.
    Any StudyEventOID with no exact mapping goes to HITL.
    """
    src_events = parse_source_xml(source_xml_bytes)
    targets, _ = load_targets()
    if not targets:
        raise RuntimeError("No trained targets found. Train the model first.")

    results = []

    for ev in src_events:
        se_oid = ev.get("StudyEventOID")
        items = ev.get("Items", [])

        confident = []
        unmapped = []

        for itm in items:
            item_oid = itm.get("ItemOID")
            if not item_oid:
                continue

            # Exact match only
            if item_oid in targets:
                confident.append({
                    "ItemOID": item_oid,
                    "Target": item_oid,
                    "Score": 1.0,
                    "Cosine": 1.0
                })
            else:
                unmapped.append(item_oid)

        # If no exact mapping exists for this StudyEventOID, HITL
        hitl = []
        if not confident:
            hitl = [se_oid]  # Only alert once per StudyEventOID

        results.append({
            "StudyEventOID": se_oid,
            "confident": confident,
            "hitl": hitl
        })

    return results
