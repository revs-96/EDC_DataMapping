import xml.etree.ElementTree as ET

def parse_source_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)
    events = []
    for sed in root.findall('.//StudyEventData'):
        evt = {
            'StudyEventOID': sed.get('StudyEventOID'),
            'PatientID': sed.findtext('PatientID') or '',
            'SiteID': sed.findtext('SiteID') or '',
            'Date': sed.findtext('Date') or '',
            'Items': [{'ItemOID': it.get('ItemOID'), 'Value': it.get('Value')} for it in sed.findall('ItemData')]
        }
        events.append(evt)
    return events

def parse_viewmapping_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)
    mappings = []
    for visit in root.findall('.//visit'):
        edc_visit = visit.get('EDCVisitID')
        impact = visit.get('IMPACTVisitID')
        for attr in visit.findall('Attribute'):
            edc_attr = attr.get('EDCAttributeID')
            mappings.append({'EDCVisitID': edc_visit, 'IMPACTVisitID': impact, 'EDCAttributeID': edc_attr})
    return mappings

def update_viewmapping(xml_path, edc_visit_id, impact_visit_id, edc_attr_id):
    """Append a new visit->attribute mapping into ViewMapping.xml"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    visit_elem = ET.Element("visit", {
        "IMPACTVisitID": impact_visit_id,
        "EDCVisitID": edc_visit_id
    })
    attr_elem = ET.Element("Attribute", {
        "IMPACTAttributeID": edc_attr_id,
        "EDCAttributeID": edc_attr_id
    })
    visit_elem.append(attr_elem)
    root.append(visit_elem)

    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
