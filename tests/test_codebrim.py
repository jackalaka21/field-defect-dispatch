import pytest

from dispatch.codebrim import CLASS_INDEX, parse_annotation, to_yolo_lines
from dispatch.taxonomy import DefectClass

XML = """<?xml version='1.0' encoding='utf-8'?>
<annotation><folder>SiteA</folder><filename>image_1.jpg</filename>
<size><width>1000</width><height>500</height><depth>3</depth></size>
<object><name>defect</name><bndbox><xmin>100</xmin><ymin>50</ymin><xmax>300</xmax><ymax>250</ymax></bndbox>
<Defect><Background>0</Background><Crack>0</Crack><Spallation>1</Spallation><Efflorescence>0</Efflorescence>
<ExposedBars>1</ExposedBars><CorrosionStain>0</CorrosionStain></Defect></object>
<object><name>defect</name><bndbox><xmin>-5</xmin><ymin>0</ymin><xmax>1200</xmax><ymax>100</ymax></bndbox>
<Defect><Background>0</Background><Crack>1</Crack><Spallation>0</Spallation><Efflorescence>0</Efflorescence>
<ExposedBars>0</ExposedBars><CorrosionStain>0</CorrosionStain></Defect></object>
<object><name>defect</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>20</xmax><ymax>20</ymax></bndbox>
<Defect><Background>1</Background><Crack>0</Crack><Spallation>0</Spallation><Efflorescence>0</Efflorescence>
<ExposedBars>0</ExposedBars><CorrosionStain>0</CorrosionStain></Defect></object>
<object><name>defect</name><bndbox><xmin>50</xmin><ymin>50</ymin><xmax>50</xmax><ymax>80</ymax></bndbox>
<Defect><Background>0</Background><Crack>0</Crack><Spallation>1</Spallation><Efflorescence>0</Efflorescence>
<ExposedBars>0</ExposedBars><CorrosionStain>0</CorrosionStain></Defect></object>
</annotation>"""


def test_multilabel_box_is_expanded_and_clipped():
    site, W, H, boxes = parse_annotation(XML)
    assert (site, W, H) == ("SiteA", 1000, 500)
    got = sorted(boxes)
    spall, rebar = CLASS_INDEX[DefectClass.SPALLING], CLASS_INDEX[DefectClass.EXPOSED_REBAR]
    crack = CLASS_INDEX[DefectClass.STRUCTURAL_CRACK]
    # one multi-label box -> two YOLO boxes with identical geometry
    pair = [b for b in got if b[0] in (spall, rebar)]
    assert len(pair) == 2 and pair[0][1] == pair[1][1] == pytest.approx((0.2, 0.3, 0.2, 0.4))
    # an out-of-image box is clipped to the frame: x in [0, 1000]
    c = [b for b in got if b[0] == crack][0][1]
    assert c == pytest.approx((0.5, 0.1, 1.0, 0.2))


def test_background_only_and_degenerate_boxes_are_dropped():
    _, _, _, boxes = parse_annotation(XML)
    assert len(boxes) == 3  # 2 from the multi-label box + 1 crack; background and zero-width dropped


def test_yolo_line_format():
    line = to_yolo_lines([(2, (0.5, 0.5, 0.25, 0.125))])[0]
    assert line == "2 0.500000 0.500000 0.250000 0.125000"
