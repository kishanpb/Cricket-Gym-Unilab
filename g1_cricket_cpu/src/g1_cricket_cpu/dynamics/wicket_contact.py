"""Opt-in numerical wicket contact; not a calibrated cricket material model."""
import xml.etree.ElementTree as ET
VERSION = 'cricket_wicket_contact_v1'
WICKETS = tuple((f'wicket_{i}' for i in range(3)))
SOLREF = (0.001, 1.0)

def add_wicket_pairs(root):
    names = {geom.get('name') for geom in root.iter('geom')}
    if not {'ball_geom', *WICKETS} <= names:
        raise ValueError('Wicket contact requires the ball and all three target stumps')
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    for pair in contact.findall('pair'):
        geoms = {pair.get('geom1'), pair.get('geom2')}
        if any((geoms == {'ball_geom', wicket} for wicket in WICKETS)):
            raise ValueError('Ball/wicket contact is already explicit')
    for wicket in WICKETS:
        ET.SubElement(contact, 'pair', name=f'{VERSION}_{wicket}', geom1='ball_geom', geom2=wicket, condim='3', friction='1 1 .01 .001 .001', solref='.001 1', solimp='.9 .95 .001 .5 2', margin='0', gap='0')
