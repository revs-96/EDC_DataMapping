import xml.etree.ElementTree as ET
import json
import numpy as np
from embedder import Embedder
from config import VIEWMAPPING_XML_PATH, TARGETS_JSON, TARGET_EMB_PATH

def load_targets():
    """Load targets and embeddings for exact matching."""
    try:
        with open(TARGETS_JSON, "r", encoding="utf-8") as f:
            targets = json.load(f)
        target_embs = np.load(TARGET_EMB_PATH, allow_pickle=True)
        return targets, target_embs
    except Exception:
        return [], None

def save_new_mapping(study_event_oid, mapping_text):
    """
    Add a new mapping from human feedback into ViewMapping.xml.
    Also update targets.json and embeddings so the agent "learns".
    """
    # --- Update ViewMapping.xml ---
    try:
        tree = ET.parse(VIEWMAPPING_XML_PATH)
        root = tree.getroot()
    except FileNotFoundError:
        root = ET.Element("VisitDesign")
        tree = ET.ElementTree(root)

    # Check if visit exists
    visit = None
    for v in root.findall("visit"):
        if v.get("EDCVisitID") == study_event_oid:
            visit = v
            break

    if visit is None:
        # Create new visit
        visit = ET.SubElement(root, "visit", {
            "IMPACTVisitID": mapping_text,  # Here user gives the IMPACTVisitID
            "EDCVisitID": study_event_oid
        })
    else:
        # Add Attribute if not exists
        existing_attrs = [a.get("IMPACTAttributeID") for a in visit.findall("Attribute")]
        if mapping_text not in existing_attrs:
            ET.SubElement(visit, "Attribute", {
                "IMPACTAttributeID": mapping_text,
                "EDCAttributeID": study_event_oid
            })

    # Save ViewMapping.xml
    tree.write(VIEWMAPPING_XML_PATH, encoding="utf-8", xml_declaration=True)

    # --- Update targets.json and embeddings ---
    try:
        with open(TARGETS_JSON, "r", encoding="utf-8") as f:
            targets = json.load(f)
    except FileNotFoundError:
        targets = []

    if mapping_text not in targets:
        targets.append(mapping_text)
        with open(TARGETS_JSON, "w", encoding="utf-8") as f:
            json.dump(targets, f, ensure_ascii=False, indent=2)

        # Recompute embeddings
        embedder = Embedder()
        target_embs = embedder.encode(targets)
        np.save(TARGET_EMB_PATH, target_embs)

    return True
