"""

"""
import argparse
import multiprocessing
import os
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
import asyncio

import numpy as np
from scipy.signal import medfilt2d
from PIL import Image
from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import (
    PhotometricInterpretation,
    CFAPattern,
    CalibrationIlluminant,
    DNGVersion,
    PreviewColorSpace,
)
from skimage.util import img_as_float, img_as_uint

__VERSION__ = "0.0.1"


META_KEY = b"META\x00\xa5\xa5\xa5"
RAW_KEY = b"RAW0\x000\x00\xa5"
TEST_FILES = Path("test_files_wb")
MODEL = {"e22": "Emotion 22"}
pyEMDNG_HOME = Path(Path.home() / ".local/pyEMDNG")
FLAT_DIR = pyEMDNG_HOME / "flats"
pyEMDNG_HOME.mkdir(exist_ok=True)
FLAT_DIR.mkdir(exist_ok=True)
FILENAME = "pyEMDNG_flat_iso{iso}_{lens}mm.npy"


class WhiteBalance(Enum):
    Manual = 7
    Flash = 0
    Neon = 1
    Tungsten = 2
    Shadow = 3
    Sun = 4
    Cloudy = 5

    def __str__(self):
        return str(self.name)


def get_wad(raw, start, offset):
    return raw[start : start + offset]


def read_raw(path):
    with open(path, "rb") as raw_fp:
        raw = raw_fp.read()
    return raw


def read_pwad_lumps(raw):
    lumps = {}
    file_type, num_file, offset = get_pwad_info(raw)
    for file_number in range(num_file):
        file_entry_start = offset + file_number * 16
        fe_offset = gle(raw[file_entry_start : file_entry_start + 4])
        fe_size = gle(raw[file_entry_start + 4 : file_entry_start + 8])
        name = raw[file_entry_start + 8 : file_entry_start + 16]
        lumps[name] = (fe_offset, fe_size)
    return lumps


def get_pwad_info(raw):
    file_type = raw[0:4]
    num_file = gle(raw[4:8])
    offset = gle(raw[8:12])
    return file_type, num_file, offset


@dataclass
class SinarIA:
    shutter_count: int
    camera: str
    measured_shutter_us: int
    req_shutter_us: int
    f_stop: float
    black_ref: str
    iso: int
    serial: str
    white_balance_name: WhiteBalance
    focal_length: float
    filename: Path = field(default=Path(""))
    raw_data: bytes = field(default_factory=bytes, repr=False)
    meta: bytes = field(default_factory=bytes, repr=False)
    black_path: Path = field(default=Path(""))


def process_meta(meta: bytes):
    shutter_count = gli(meta, 4)
    camera = meta[20:64].decode("ascii").rstrip("\x00")
    white_balance_name = WhiteBalance(gls(meta, 100))
    shutter_time_us = gli(meta, 104)
    black_ref = meta[108 : 108 + 128].decode("ascii").rstrip("\x00")
    iso = gli(meta, 252)
    serial = meta[272 : 272 + 16].decode("ascii").rstrip("\x00")
    shutter_time_us_2 = gli(meta, 344)
    f_stop = round(gls(meta, 352) / 256, 1)
    focal_length = round(gli(meta, 356) / 1000, 0)
    return SinarIA(
        shutter_count=shutter_count,
        camera=camera,
        measured_shutter_us=shutter_time_us,
        req_shutter_us=shutter_time_us_2,
        f_stop=f_stop,
        black_ref=black_ref,
        iso=iso,
        serial=serial,
        meta=meta,
        white_balance_name=white_balance_name,
        focal_length=focal_length,
    )


def read_sinar(path: Path):
    raw = read_raw(path)
    lumps = read_pwad_lumps(raw)
    meta = get_wad(raw, *lumps[META_KEY])
    sinar_ia = process_meta(meta)
    raw_data = get_wad(raw, *lumps[RAW_KEY])
    sinar_ia.raw_data = raw_data
    sinar_ia.filename = path
    sinar_ia.black_path = (
        sinar_ia.filename.parent.absolute() / Path(sinar_ia.black_ref).name
    )
    return sinar_ia


def read_black_ref(path: Path, nd_img: np.ndarray):
    black = read_raw(path)
    lumps = read_pwad_lumps(black)
    black0 = get_wad(black, *lumps[b"BLACK0\x00\xa5"])
    black1 = get_wad(black, *lumps[b"BLACK1\x00\xa5"])
    h, w = nd_img.shape  # int. flipped
    black0_img = img_as_float(
        np.asarray(Image.frombytes("I;16L", (w, h), black0, "raw"))
    )
    black1_img = img_as_float(
        np.asarray(Image.frombytes("I;16L", (w, h), black1, "raw"))
    )
    return black0_img, black1_img


def get_raw_pillow(raw: SinarIA, h, w):
    raw0 = raw.raw_data
    assert (h * w * 16) / 8 == len(raw0)
    img = Image.frombytes("I;16L", (w, h), raw0, "raw")
    return img


def apply_local_black_ref(nd_img: np.array, black_path: Path):
    b0, b1 = read_black_ref(black_path, nd_img)
    nd_fp = nd_img - b1
    hot_pixels = b0 - b1
    nd_fp_stack = get_colors(nd_fp)
    hot_pixel_stack = get_colors(hot_pixels)
    pixel_mask = np.greater(hot_pixel_stack, 0)
    # Because scipy does not have way to run median filter on specific axis (or, I don't know how to at least)
    for i in range(4):
        color = nd_fp_stack[::, ::, i]
        mask = pixel_mask[::, ::, i]
        hot = hot_pixel_stack[::, ::, i]
        color[mask] = medfilt2d((color - hot))[mask]
        nd_fp_stack[::, ::, i] = color
    return unstack_colors(nd_fp_stack)


def process_raw(raw: SinarIA, h=5344, w=4008, flat_disable=False):
    img = get_raw_pillow(raw, h, w)
    nd_img = img_as_float(img)
    nd_img_b = apply_local_black_ref(nd_img, raw.black_path)
    if flat_disable:
        nd_img_flat = nd_img_b
    else:
        # TODO figure out better way to ident files, or build lensless flat?
        iso, focal = raw.iso, raw.focal_length
        nd_img_flat = apply_flat(nd_img_b, 50, 80.0)
    return nd_img_flat


def norm2(x):
    return np.clip((x - x.max()) * (1 / (x.min() - x.max())), 0, 1)


def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def c_norm(x, c_min, c_max):
    return (x - c_min) / (c_max - c_min)


def gle(b):
    return int.from_bytes(b, byteorder="little")


def gli(b, s):
    return int.from_bytes(b[s : s + 4], byteorder="little")


def gls(b, s):
    return int.from_bytes(b[s : s + 2], byteorder="little")


def sub(a: np.ndarray, b: np.ndarray):
    res = a - b
    res[b > a] = 0
    return res


def create_master_flat(flats, h=5344, w=4008):
    corrected = []
    for flat in flats:
        if (
            abs(flat.measured_shutter_us - flat.req_shutter_us) / flat.req_shutter_us
            > 0.5
        ):
            # Our shutter was more than 50% slower than requested, skip this file
            print(
                f"Skipping {flat.filename}, shutter speed was {flat.measured_shutter_us}uS, requested {flat.req_shutter_us}uS!"
            )
            continue
        nd_img = img_as_float(get_raw_pillow(flat, h, w))
        black_path = flat.filename.parent.absolute() / Path(flat.black_ref).name
        b0, b1 = read_black_ref(black_path, nd_img)
        biased_nd_img = nd_img - b1
        norms = color_norm(biased_nd_img)
        corrected.append(unstack_colors(norms))
    flat_file = np.stack(corrected, axis=0)
    return np.median(flat_file, axis=0)


def color_norm(flat):
    colors = get_colors(flat)
    means = colors.mean(axis=(0, 1))
    return colors / means


def get_colors(arr):
    green1 = arr[0::2, 0::2]
    blue = arr[0::2, 1::2]
    red = arr[1::2, 0::2]
    green2 = arr[1::2, 1::2]
    return np.stack([green1, blue, red, green2], axis=-1)


def unstack_colors(colors):
    shape = [i * 2 for i in colors.shape[:-1]]
    arr = np.empty(shape=shape)
    arr[0::2, 0::2] = colors[::, ::, 0]
    arr[0::2, 1::2] = colors[::, ::, 1]
    arr[1::2, 0::2] = colors[::, ::, 2]
    arr[1::2, 1::2] = colors[::, ::, 3]
    return arr


def process_list_of_flats_to_flat(flat_files, h=5344, w=4008):
    ia_flats = [read_sinar(flat) for flat in flat_files]
    f0: SinarIA = ia_flats[0]
    lens = (
        f0.focal_length if f0.focal_length else input("Please enter the focal length: ")
    )
    corrected = create_master_flat(ia_flats, h, w)
    np.save(str(FLAT_DIR / FILENAME.format(iso=f0.iso, lens=lens)), corrected)


def apply_flat(nd_img, iso, lens):
    try:
        flat = np.load(str(FLAT_DIR / FILENAME.format(iso=iso, lens=lens)))
        return nd_img / flat
    except FileNotFoundError:
        print(f"No flat file exists iso{iso}, {lens}mm !")
        return nd_img


MULT = 1000000000
CCM1 = [
    [690277635, MULT],
    [81520691, MULT],
    [-52482637, MULT],
    [-674076304, MULT],
    [1469691720, MULT],
    [769255116, MULT],
    [-180923460, MULT],
    [390943643, MULT],
    [1601514930, MULT],
]


def create_ia_dng(
    img: SinarIA, output_dir: Path, bpp=16, h=5344, w=4008, flat_disable=False
):
    corrected_flat = process_raw(img, h, w, flat_disable)
    nd_int = img_as_uint(corrected_flat)

    filename, r = write_dng(img, nd_int, output_dir, h, w, bpp)
    return filename, r


def write_dng(img, nd_int, output_dir, h, w, bpp):
    db_model = img.serial.split("-")[0]
    t = DNGTags()
    t.set(Tag.ImageWidth, w)
    t.set(Tag.ImageLength, h)
    t.set(Tag.Orientation, 8)  # Rotate 270 CW
    t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
    t.set(Tag.SamplesPerPixel, 1)
    t.set(Tag.BitsPerSample, [bpp])
    t.set(Tag.CFARepeatPatternDim, [2, 2])
    t.set(Tag.CFAPattern, CFAPattern.RGGB)
    t.set(Tag.BlackLevel, [nd_int.min()])
    t.set(Tag.WhiteLevel, [nd_int.max()])
    t.set(Tag.ColorMatrix1, CCM1)
    t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.Daylight)
    t.set(Tag.AsShotNeutral, [[1, 1], [1, 1], [1, 1]])
    t.set(Tag.Make, img.camera)
    t.set(Tag.Model, MODEL[db_model])
    t.set(Tag.DNGVersion, DNGVersion.V1_4)
    t.set(Tag.DNGBackwardVersion, DNGVersion.V1_4)
    t.set(Tag.EXIFPhotoBodySerialNumber, img.serial)
    t.set(Tag.CameraSerialNumber, img.serial)
    t.set(Tag.ExposureTime, [(img.measured_shutter_us, 1000000)])
    t.set(Tag.PhotographicSensitivity, [img.iso])
    t.set(Tag.SensitivityType, 3)
    t.set(Tag.FocalLengthIn35mmFilm, [int(img.focal_length * 100), 62])
    t.set(Tag.FocalLength, [(int(img.focal_length), 1)])
    t.set(Tag.UniqueCameraModel, f"{MODEL[db_model]} ({img.serial}) on {img.camera}")
    t.set(Tag.FNumber, [(int(img.f_stop * 100), 100)])
    t.set(Tag.BayerGreenSplit, 0)
    t.set(Tag.Software, f"PYEmotionDNG v{__VERSION__}")
    t.set(Tag.PreviewColorSpace, PreviewColorSpace.sRGB)
    r = RAW2DNG()
    r.options(t, path="", compress=False)
    filename = output_dir / f"{img.filename.stem}.dng"
    r.convert(nd_int, filename=str(filename.absolute()))
    return filename, r


async def main(parsed):
    if parsed.build_flat:
        flats = list(Path(parsed.build_flat).glob("*.IA"))
        process_list_of_flats_to_flat(flats)
        exit(0)

    output_path = Path(parsed.output)
    os.makedirs(str(output_path.absolute()), exist_ok=True)
    files = []
    for i in parsed.images:
        img_path = Path(i)
        if img_path.is_dir():
            files.extend(img_path.glob("*.IA"))
        elif img_path.suffix == ".IA":
            files.append(img_path)
    print(f"Processing {len(files)} images!")
    flat_disable = parsed.flat_disable
    threads = []
    for f in files:
        t = partial(convert_ia_file_to_dng, f, flat_disable, output_path)
        threads.append(asyncio.to_thread(t))
    thread_count = multiprocessing.cpu_count()
    for t_group in range(0, len(threads), thread_count):
        await asyncio.gather(*threads[t_group : t_group + thread_count])


def convert_ia_file_to_dng(f, flat_disable, output_path):
    ia_img = read_sinar(f)
    f_name, _ = create_ia_dng(ia_img, output_path, flat_disable=flat_disable)
    print(f"Processed: {f.absolute()} to {f_name.absolute()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", default=".", help="Output directory.")
    p.add_argument(
        "-m", "--multi", default=False, action="store_true", help="Use multiprocessing"
    )
    p.add_argument(
        "-f",
        "--flat-disable",
        default=False,
        action="store_true",
        help="Disable flat files",
    )
    p.add_argument(
        "--build-flat",
        default=False,
        help="Use this directory of files to build a flat file.",
    )
    p.add_argument(
        "images", nargs="*", help="List of images or directory of images to process."
    )
    args = p.parse_args()
    asyncio.run(main(args))
