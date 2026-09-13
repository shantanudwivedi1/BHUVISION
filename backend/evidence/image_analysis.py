import struct
from typing import Any


class UnsupportedImageError(ValueError):
    pass


def inspect_image(data: bytes, filename: str = "image") -> dict[str, Any]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return _inspect_png(data, filename)
    if data.startswith(b"\xff\xd8\xff"):
        return _inspect_jpeg(data, filename)
    if data[:4] in (b"II*\x00", b"MM\x00*") or data[:4] in (b"II+\x00", b"MM\x00+"):
        return _inspect_tiff(data, filename)
    raise UnsupportedImageError(f"Unsupported image format for {filename}")


def _base_result(filename: str, image_format: str, width: int, height: int, channels: int) -> dict[str, Any]:
    megapixels = round((width * height) / 1_000_000, 3) if width and height else 0
    quality = {
        "file_size_bytes": 0,
        "megapixels": megapixels,
        "aspect_ratio": round(width / height, 3) if height else None,
        "warnings": [],
    }
    if width < 128 or height < 128:
        quality["warnings"].append("low spatial resolution")
    if channels == 1:
        quality["warnings"].append("single-channel image")
    return {
        "filename": filename,
        "format": image_format,
        "width": width,
        "height": height,
        "channels": channels,
        "quality": quality,
    }


def _finish(result: dict[str, Any], size: int) -> dict[str, Any]:
    result["quality"]["file_size_bytes"] = size
    return result


def _inspect_png(data: bytes, filename: str) -> dict[str, Any]:
    if len(data) < 26 or data[12:16] != b"IHDR":
        raise UnsupportedImageError(f"Invalid PNG header for {filename}")
    width, height, _depth, color_type = struct.unpack(">IIBB", data[16:26])
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type, 0)
    if not channels:
        raise UnsupportedImageError(f"Unsupported PNG color type for {filename}")
    return _finish(_base_result(filename, "PNG", width, height, channels), len(data))


def _inspect_jpeg(data: bytes, filename: str) -> dict[str, Any]:
    position = 2
    sof_markers = set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0))
    while position + 9 < len(data):
        if data[position] != 0xFF:
            position += 1
            continue
        marker = data[position + 1]
        position += 2
        if marker in (0xD8, 0xD9):
            continue
        if position + 2 > len(data):
            break
        segment_length = struct.unpack(">H", data[position:position + 2])[0]
        if marker in sof_markers and position + 8 < len(data):
            height, width, channels = struct.unpack(">HHB", data[position + 3:position + 8])
            return _finish(_base_result(filename, "JPEG", width, height, channels), len(data))
        if segment_length < 2:
            break
        position += segment_length
    raise UnsupportedImageError(f"Invalid JPEG header for {filename}")


def _inspect_tiff(data: bytes, filename: str) -> dict[str, Any]:
    endian = ">" if data[:2] == b"MM" else "<"
    if len(data) < 8:
        raise UnsupportedImageError(f"Invalid TIFF header for {filename}")
    offset = struct.unpack_from(endian + "I", data, 4)[0]
    if offset + 2 > len(data):
        raise UnsupportedImageError(f"Invalid TIFF directory for {filename}")
    entries = struct.unpack_from(endian + "H", data, offset)[0]
    width = height = channels = 0
    type_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8}
    for index in range(entries):
        entry_offset = offset + 2 + (index * 12)
        if entry_offset + 12 > len(data):
            break
        tag, value_type, count = struct.unpack_from(endian + "HHI", data, entry_offset)
        size = type_sizes.get(value_type, 0) * count
        raw_offset = entry_offset + 8 if size <= 4 else struct.unpack_from(endian + "I", data, entry_offset + 8)[0]
        if raw_offset + min(size, 4) > len(data):
            continue
        if value_type == 3 and count == 1:
            value = struct.unpack_from(endian + "H", data, raw_offset)[0]
        elif value_type == 4 and count == 1:
            value = struct.unpack_from(endian + "I", data, raw_offset)[0]
        else:
            continue
        if tag == 256:
            width = value
        elif tag == 257:
            height = value
        elif tag == 277:
            channels = value
    if not width or not height:
        raise UnsupportedImageError(f"Invalid TIFF dimensions for {filename}")
    return _finish(_base_result(filename, "TIFF", width, height, channels or 1), len(data))
