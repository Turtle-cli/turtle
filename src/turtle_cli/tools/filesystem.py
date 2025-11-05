import os
from pathlib import Path
from typing import List, Dict


class FileSystem:
    """Simple file system operations with basic safety checks."""
    
    def __init__(self, working_dir: str = "."):
        self.working_dir = Path(working_dir).resolve()
    
    def _get_full_path(self, path: str) -> Path:
        full_path = (self.working_dir / path).resolve()
        
        if not str(full_path).startswith(str(self.working_dir)):
            raise ValueError(f"Path outside working directory: {path}")
        
        return full_path
    
    def read_file(self, path: str) -> str:
        full_path = self._get_full_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        return full_path.read_text()
    
    def write_file(self, path: str, content: str) -> None:
        full_path = self._get_full_path(path)
        
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        full_path.write_text(content)
    
    def append_file(self, path: str, content: str) -> None:
        full_path = self._get_full_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        with open(full_path, 'a') as f:
            f.write(content)
    
    def replace_in_file(self, path: str, old: str, new: str) -> None:
        full_path = self._get_full_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        content = full_path.read_text()
        
        if old not in content:
            raise ValueError(f"Text not found in file: {old}")
        
        new_content = content.replace(old, new)
        full_path.write_text(new_content)
    
    def list_directory(self, path: str = ".") -> List[Dict[str, any]]:
        full_path = self._get_full_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if not full_path.is_dir():
            raise ValueError(f"Not a directory: {path}")
        
        items = []
        for item in full_path.iterdir():
            items.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None
            })
        
        return sorted(items, key=lambda x: (x["type"] != "dir", x["name"]))
    
    def exists(self, path: str) -> bool:
        try:
            full_path = self._get_full_path(path)
            return full_path.exists()
        except:
            return False
    
    def is_file(self, path: str) -> bool:
        try:
            full_path = self._get_full_path(path)
            return full_path.is_file()
        except:
            return False
    
    def is_dir(self, path: str) -> bool:
        try:
            full_path = self._get_full_path(path)
            return full_path.is_dir()
        except:
            return False
    
    def delete_file(self, path: str) -> None:
        full_path = self._get_full_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if not full_path.is_file():
            raise ValueError(f"Not a file: {path}")
        
        full_path.unlink()
    
    def create_directory(self, path: str) -> None:
        full_path = self._get_full_path(path)
        full_path.mkdir(parents=True, exist_ok=True)
        