#!/usr/bin/env python3
"""Shrink the images of a docx without touching its text or revision marks (manager 11-27: tracked Word <= 18 MB).
Each image in word/media is resampled (Lanczos) to DPI at its largest displayed width in word/document.xml
(wp:extent), never enlarged. It is saved as JPEG at QUALITY when it is photo-like (filled colour maps and sections), otherwise as optimised
PNG so lines and text stay sharp. Line plot, bar chart or schematic = mostly white background (>= 80% of pixels),
or few colours (<= 40 bins of a 16-level RGB grid each covering >= 0.05% of pixels), or PNG < 2.5 x JPEG size.
A JPEG gets a new file name, so only word/_rels/document.xml.rels and [Content_Types].xml change besides the
media; word/document.xml is copied byte for byte (checked at the end).
Usage: shrink_docx_images.py in.docx out.docx [dpi=200] [quality=85]"""
import io
import sys
import zipfile

import numpy as np
from lxml import etree
from PIL import Image

IN, OUT = sys.argv[1:3]
DPI = float(sys.argv[3]) if len(sys.argv) > 3 else 200.0
QUALITY = int(sys.argv[4]) if len(sys.argv) > 4 else 85
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
WP = '{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
CT = '{http://schemas.openxmlformats.org/package/2006/content-types}'

z = zipfile.ZipFile(IN)
rels = etree.fromstring(z.read('word/_rels/document.xml.rels'))
rid2t = {e.get('Id'): e.get('Target') for e in rels}
doc = z.read('word/document.xml')
width_in = {}
for d in etree.fromstring(doc).iter(W + 'drawing'):
    e, b = d.find('.//' + WP + 'extent'), d.find('.//' + A + 'blip')
    if e is None or b is None:
        continue
    t = 'word/' + rid2t[b.get(R + 'embed')]
    width_in[t] = max(width_in.get(t, 0), int(e.get('cx')) / 914400)

new_data, renamed = {}, {}
before = after = 0
for n in z.namelist():
    if not n.startswith('word/media/') or n not in width_in:
        continue
    data = z.read(n)
    before += len(data)
    im = Image.open(io.BytesIO(data))
    im.load()
    if im.mode in ('RGBA', 'LA', 'P'):
        bg = Image.new('RGB', im.size, 'white')
        bg.paste(im.convert('RGBA'), mask=im.convert('RGBA').split()[-1])
        im = bg
    else:
        im = im.convert('RGB')
    tw = int(round(width_in[n] * DPI))
    if im.width > tw:
        im = im.resize((tw, max(1, int(round(im.height * tw / im.width)))), Image.LANCZOS)
    bp, bj = io.BytesIO(), io.BytesIO()
    im.save(bp, 'PNG', optimize=True, dpi=(DPI, DPI))
    im.save(bj, 'JPEG', quality=QUALITY, optimize=True, dpi=(int(DPI), int(DPI)))
    a = np.asarray(im).reshape(-1, 3)
    white = (a.min(1) > 245).mean()
    q = (a // 16).astype(np.int32)
    bins = int((np.bincount(q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2], minlength=4096) >= 0.0005 * len(a)).sum())
    lineart = white >= 0.80 or bins <= 40 or bp.tell() < 2.5 * bj.tell()
    if not lineart:
        nn = n.rsplit('.', 1)[0] + '.jpeg'
        new_data[nn] = bj.getvalue()
        renamed[n] = nn
        kind = 'JPEG'
    elif n.lower().endswith('.png') and len(data) <= bp.tell():
        new_data[n] = data                              # the original is already smaller: keep it as it is
        kind = 'kept'
        im = Image.open(io.BytesIO(data))               # for the log: the kept image's own size
    else:
        new_data[n] = bp.getvalue()
        kind = 'PNG'
    after += len(new_data[renamed.get(n, n)])
    print(f'{n[11:]:24s} {len(data) / 1e6:5.2f} -> {len(new_data[renamed.get(n, n)]) / 1e6:5.2f} MB  '
          f'{im.width}x{im.height} {kind} (white {white:.2f}, bins {bins}, shown {width_in[n]:.2f} in)')

for e in rels:
    t = 'word/' + e.get('Target')
    if t in renamed:
        e.set('Target', renamed[t][5:])
ct = etree.fromstring(z.read('[Content_Types].xml'))
if not any(e.get('Extension', '').lower() == 'jpeg' for e in ct):
    d = etree.SubElement(ct, CT + 'Default')
    d.set('Extension', 'jpeg')
    d.set('ContentType', 'image/jpeg')

with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as o:
    for info in z.infolist():
        n = info.filename
        if n.endswith('/'):
            continue
        if n in renamed:
            o.writestr(renamed[n], new_data[renamed[n]])
        elif n in new_data:
            o.writestr(info, new_data[n])
        elif n == 'word/_rels/document.xml.rels':
            o.writestr(info, etree.tostring(rels, xml_declaration=True, encoding='UTF-8', standalone=True))
        elif n == '[Content_Types].xml':
            o.writestr(info, etree.tostring(ct, xml_declaration=True, encoding='UTF-8', standalone=True))
        else:
            o.writestr(info, z.read(n))
assert zipfile.ZipFile(OUT).read('word/document.xml') == doc, 'document.xml changed'
print(f'media {before / 1e6:.1f} -> {after / 1e6:.1f} MB; {len(renamed)} images to JPEG; document.xml unchanged -> {OUT}')
