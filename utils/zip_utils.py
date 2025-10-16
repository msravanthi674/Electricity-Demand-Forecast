from pathlib import Path
import io, zipfile

def zip_folder_to_bytes(folder: Path):
    """
    Create an in-memory zip of `folder` and return bytes (for Streamlit download_button).
    """
    folder = Path(folder)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in folder.rglob("*"):
            if p.is_file():
                arcname = p.relative_to(folder.parent)
                zf.write(p, arcname.as_posix())
    mem.seek(0)
    return mem.getvalue()
