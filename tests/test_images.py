from io import BytesIO
from PIL import Image, ImageDraw
import pytest
from backend.app.services.images import prepare_image

def handwritten_page():
    image=Image.new('RGB',(800,360),'white'); draw=ImageDraw.Draw(image)
    draw.line((50,70,640,70),fill='black',width=5)
    draw.line((50,170,700,170),fill='black',width=5)
    draw.line((50,280,600,280),fill='black',width=5)
    out=BytesIO(); image.save(out,format='PNG'); return out.getvalue()

def test_line_segmentation_detects_separate_rows():
    result=prepare_image(handwritten_page())
    assert len(result.lines)==3
    assert result.image.size==(800,360)

def test_single_region_mode_returns_one_region():
    assert len(prepare_image(handwritten_page(),mode='single').lines)==1

def test_wider_vertical_gap_marks_paragraph_boundary():
    image=Image.new('RGB',(800,400),'white'); draw=ImageDraw.Draw(image)
    for y in (45,110,245,310): draw.line((50,y,650,y),fill='black',width=5)
    out=BytesIO(); image.save(out,format='PNG')
    result=prepare_image(out.getvalue())
    assert len(result.lines)==4
    assert 2 in result.paragraph_breaks

def test_invalid_image_rejected():
    with pytest.raises(ValueError,match='readable image'): prepare_image(b'not an image')

def test_oversized_image_rejected():
    image=Image.new('RGB',(6000,6000),'white'); out=BytesIO(); image.save(out,format='PNG')
    with pytest.raises(ValueError,match='30 megapixels'): prepare_image(out.getvalue())
