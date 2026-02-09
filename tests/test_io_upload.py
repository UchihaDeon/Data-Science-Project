import os
from io import BytesIO
from modules.io import save_uploaded_file

def test_save_uploaded_file(tmp_path):
    data = b"col1,col2\n1,2\n3,4\n"
    bio = BytesIO(data)
    save_dir = tmp_path / "uploads"
    path = save_uploaded_file(bio, save_dir=str(save_dir), filename="test_upload.csv")
    assert os.path.exists(path)
    # check content
    with open(path, "rb") as f:
        content = f.read()
    assert b"col1,col2" in content