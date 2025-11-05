import pytest
from pathlib import Path
import tempfile
import shutil
from turtle_cli.tools.filesystem import FileSystem


@pytest.fixture
def temp_workspace():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def fs(temp_workspace):
    return FileSystem(temp_workspace)


def test_read_file(fs, temp_workspace):
    test_file = Path(temp_workspace) / "test.txt"
    test_file.write_text("Hello, World!")
    
    content = fs.read_file("test.txt")
    assert content == "Hello, World!"


def test_read_file_not_found(fs):
    with pytest.raises(FileNotFoundError):
        fs.read_file("nonexistent.txt")


def test_write_file(fs, temp_workspace):
    fs.write_file("new.txt", "Test content")
    
    test_file = Path(temp_workspace) / "new.txt"
    assert test_file.exists()
    assert test_file.read_text() == "Test content"


def test_write_file_with_directories(fs, temp_workspace):
    fs.write_file("sub/dir/file.txt", "Nested content")
    
    test_file = Path(temp_workspace) / "sub" / "dir" / "file.txt"
    assert test_file.exists()
    assert test_file.read_text() == "Nested content"


def test_append_file(fs, temp_workspace):
    test_file = Path(temp_workspace) / "append.txt"
    test_file.write_text("First line\n")
    
    fs.append_file("append.txt", "Second line\n")
    
    assert test_file.read_text() == "First line\nSecond line\n"


def test_append_file_not_found(fs):
    with pytest.raises(FileNotFoundError):
        fs.append_file("nonexistent.txt", "content")


def test_replace_in_file(fs, temp_workspace):
    test_file = Path(temp_workspace) / "replace.txt"
    test_file.write_text("Hello World")
    
    fs.replace_in_file("replace.txt", "World", "Python")
    
    assert test_file.read_text() == "Hello Python"


def test_replace_in_file_not_found_text(fs, temp_workspace):
    test_file = Path(temp_workspace) / "replace.txt"
    test_file.write_text("Hello World")
    
    with pytest.raises(ValueError):
        fs.replace_in_file("replace.txt", "NotThere", "Python")


def test_list_directory(fs, temp_workspace):
    (Path(temp_workspace) / "file1.txt").write_text("content")
    (Path(temp_workspace) / "file2.txt").write_text("content")
    (Path(temp_workspace) / "subdir").mkdir()
    
    items = fs.list_directory(".")
    
    assert len(items) == 3
    assert items[0]["type"] == "dir"
    assert items[0]["name"] == "subdir"
    assert items[1]["type"] == "file"
    assert items[1]["name"] == "file1.txt"
    assert items[2]["type"] == "file"
    assert items[2]["name"] == "file2.txt"


def test_list_directory_not_found(fs):
    with pytest.raises(FileNotFoundError):
        fs.list_directory("nonexistent")


def test_exists(fs, temp_workspace):
    test_file = Path(temp_workspace) / "exists.txt"
    test_file.write_text("content")
    
    assert fs.exists("exists.txt") is True
    assert fs.exists("notexists.txt") is False


def test_is_file(fs, temp_workspace):
    test_file = Path(temp_workspace) / "file.txt"
    test_file.write_text("content")
    test_dir = Path(temp_workspace) / "dir"
    test_dir.mkdir()
    
    assert fs.is_file("file.txt") is True
    assert fs.is_file("dir") is False
    assert fs.is_file("notexists.txt") is False


def test_is_dir(fs, temp_workspace):
    test_file = Path(temp_workspace) / "file.txt"
    test_file.write_text("content")
    test_dir = Path(temp_workspace) / "dir"
    test_dir.mkdir()
    
    assert fs.is_dir("dir") is True
    assert fs.is_dir("file.txt") is False
    assert fs.is_dir("notexists") is False


def test_delete_file(fs, temp_workspace):
    test_file = Path(temp_workspace) / "delete.txt"
    test_file.write_text("content")
    
    fs.delete_file("delete.txt")
    
    assert not test_file.exists()


def test_delete_file_not_found(fs):
    with pytest.raises(FileNotFoundError):
        fs.delete_file("notexists.txt")


def test_create_directory(fs, temp_workspace):
    fs.create_directory("newdir")
    
    test_dir = Path(temp_workspace) / "newdir"
    assert test_dir.exists()
    assert test_dir.is_dir()


def test_create_nested_directory(fs, temp_workspace):
    fs.create_directory("parent/child/grandchild")
    
    test_dir = Path(temp_workspace) / "parent" / "child" / "grandchild"
    assert test_dir.exists()
    assert test_dir.is_dir()


def test_path_escape_prevention(fs):
    with pytest.raises(ValueError):
        fs.read_file("../../../etc/passwd")
    
    with pytest.raises(ValueError):
        fs.write_file("../outside.txt", "content")


def test_replace_in_file_not_found(fs):
    with pytest.raises(FileNotFoundError):
        fs.replace_in_file("nonexistent.txt", "old", "new")


def test_list_directory_not_a_directory(fs, temp_workspace):
    test_file = Path(temp_workspace) / "file.txt"
    test_file.write_text("content")
    
    with pytest.raises(ValueError):
        fs.list_directory("file.txt")


def test_delete_directory_not_a_file(fs, temp_workspace):
    test_dir = Path(temp_workspace) / "dir"
    test_dir.mkdir()
    
    with pytest.raises(ValueError):
        fs.delete_file("dir")


def test_path_escape_in_append(fs):
    with pytest.raises(ValueError):
        fs.append_file("../outside.txt", "content")


def test_path_escape_in_replace(fs):
    with pytest.raises(ValueError):
        fs.replace_in_file("../outside.txt", "old", "new")


def test_path_escape_in_list_directory(fs):
    with pytest.raises(ValueError):
        fs.list_directory("../../../etc")


def test_path_escape_in_delete(fs):
    with pytest.raises(ValueError):
        fs.delete_file("../outside.txt")


def test_path_escape_in_create_directory(fs):
    with pytest.raises(ValueError):
        fs.create_directory("../outside")


def test_exists_with_path_escape():
    fs = FileSystem("/tmp")
    assert fs.exists("../../../etc/passwd") is False


def test_is_file_with_path_escape():
    fs = FileSystem("/tmp")
    assert fs.is_file("../../../etc/passwd") is False


def test_is_dir_with_path_escape():
    fs = FileSystem("/tmp")
    assert fs.is_dir("../../../etc") is False
