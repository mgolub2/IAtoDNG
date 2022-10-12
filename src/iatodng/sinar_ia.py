"""
Author: Maximilian Golub
Copyright: 2022

Convert Sinar IA raw files to the Adobe DNG format.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# import numpy as np
from PIL.Image import Transpose
from numpy import greater, array, clip, stack, median, empty, save, load
from numpy import min as npmin
from numpy import max as npmax
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

__VERSION__ = "0.2.0"

META_KEY = b"META"
RAW_KEY = b"RAW0"
THUMB_KEY = b'THUMB'

MODEL_NAME = {"e22": "Emotion 22", "e75": "Emotion 75"}
MODEL_TO_SIZE = {
    "e22": (5344, 4008),
    "e75": (6668, 4992),
}
pyEMDNG_HOME = Path(Path.home() / ".local/pyEMDNG")
FLAT_DIR = pyEMDNG_HOME / "flats"
pyEMDNG_HOME.mkdir(exist_ok=True)
FLAT_DIR.mkdir(exist_ok=True)
FILENAME = "pyEMDNG_flat_iso{iso}_{lens}mm.npy"
MULT = 1000000000

# TODO this is taken from the CCM created by the original emotionDNG for
#  my back specifically - probably not optimal for others
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


class WhiteBalance(Enum):
    Manual = 7
    Flash = 0
    Neon = 1
    Tungsten = 2
    Shadow = 3
    Sun = 4
    Cloudy = 5
    Unknown = 6

    def __str__(self):
        return str(self.name)


async def get_wad(byte_data: bytes, start, offset):
    return byte_data[start: start + offset]


async def read_raw(path):
    with open(path, "rb") as raw_fp:
        raw = raw_fp.read()
    return raw


async def read_pwad_lumps(raw):
    lumps = {}
    file_type, num_file, offset = get_pwad_info(raw)
    for file_number in range(num_file):
        file_entry_start = offset + file_number * 16
        fe_offset = gle(raw[file_entry_start: file_entry_start + 4])
        fe_size = gle(raw[file_entry_start + 4: file_entry_start + 8])
        name = raw[file_entry_start + 8: file_entry_start + 16]
        # TODO clean this up
        lumps[name.split(b"\xa5")[0].split(b"\x00")[0]] = (fe_offset, fe_size)
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
    model: str
    height: int
    width: int
    white_balance_name: WhiteBalance
    focal_length: int
    white_ref: str = field(default="")  # Only some backs generate these
    filename: Path = field(default=Path(""))
    raw_data: bytes = field(default_factory=bytes, repr=False)
    meta: bytes = field(default_factory=bytes, repr=False)
    black_path: Path = field(default=None)
    white_path: Path = field(default=None)
    thumb: Image = field(default=None)


async def process_meta(meta: bytes):
    shutter_count = gli(meta, 4)
    camera = meta[20:64].decode("ascii").rstrip("\x00")
    white_balance_name = WhiteBalance(gls(meta, 100))
    shutter_time_us = gli(meta, 104)
    black_ref = meta[108: 108 + 64].decode("ascii").rstrip("\x00")
    white_ref = meta[172: 172 + 64].decode("ascii").rstrip("\x00")
    iso = gli(meta, 252)
    serial = meta[272: 272 + 16].decode("ascii").rstrip("\x00")
    shutter_time_us_2 = gli(meta, 344)
    f_stop = round(gls(meta, 352) / 256, 1)
    focal_length = int(round(gli(meta, 356) / 1000, 0))
    short_model = serial.split("-")[0]
    model = MODEL_NAME[short_model]
    height, width = MODEL_TO_SIZE[short_model]
    return SinarIA(
        shutter_count=shutter_count,
        camera=camera,
        measured_shutter_us=shutter_time_us,
        req_shutter_us=shutter_time_us_2,
        f_stop=f_stop,
        black_ref=black_ref,
        white_ref=white_ref,
        iso=iso,
        serial=serial,
        meta=meta,
        white_balance_name=white_balance_name,
        focal_length=focal_length,
        model=model,
        height=height,
        width=width,
    )


async def read_sinar(path: Path):
    raw = await read_raw(path)
    lumps = await read_pwad_lumps(raw)
    meta = await get_wad(raw, *lumps[META_KEY])
    sinar_ia = await process_meta(meta)
    raw_data = await get_wad(raw, *lumps[RAW_KEY])
    thumb = await get_wad(raw, *lumps[THUMB_KEY])
    sinar_ia.thumb = Image.frombytes("RGB", (356, 476), thumb[:-15920], "raw")
    sinar_ia.raw_data = raw_data
    sinar_ia.filename = path
    sinar_ia.black_path = (
            sinar_ia.filename.parent.absolute() / Path(sinar_ia.black_ref).name
    )
    if sinar_ia.white_ref:
        sinar_ia.white_path = (
                sinar_ia.filename.parent.absolute() / Path(sinar_ia.white_ref).name
        )
    return sinar_ia


# noinspection PyTypeChecker
async def read_black_ref(path: Path, nd_img: array):
    black = await read_raw(path)
    lumps = await read_pwad_lumps(black)
    black0 = await get_wad(black, *lumps[b"BLACK0"])
    black1 = await get_wad(black, *lumps[b"BLACK1"])
    h, w = nd_img.shape  # int. flipped
    black0_img = img_as_float(Image.frombytes("I;16L", (w, h), black0, "raw"))
    black1_img = img_as_float(Image.frombytes("I;16L", (w, h), black1, "raw"))
    return black0_img, black1_img


# noinspection PyTypeChecker
async def get_raw_pillow(raw: SinarIA):
    raw0 = raw.raw_data
    assert (raw.height * raw.width * 16) / 8 == len(raw0)
    img = Image.frombytes("I;16L", (raw.width, raw.height), raw0, "raw")
    return img


async def apply_local_black_simple(nd_img: array, black_path: Path):
    b0, b1 = await read_black_ref(black_path, nd_img)
    nd_fp = (nd_img - b0) - (b1 - b0)
    return nd_fp


async def apply_local_black_ref_v8(nd_img: array, black_path: Path):
    b0, b1 = await read_black_ref(black_path, nd_img)
    nd_fp = nd_img - b1
    hot_pixels = b0 - b1
    nd_fp_stack = get_colors(nd_fp)
    hot_pixel_stack = get_colors(hot_pixels)
    pixel_mask = greater(hot_pixel_stack, 0)
    # Because scipy does not have way to run median filter on specific axis (or, I don't know how to at least)
    for i in range(4):
        color = nd_fp_stack[::, ::, i]
        mask = pixel_mask[::, ::, i]
        hot = hot_pixel_stack[::, ::, i]
        color[mask] = medfilt2d((color - hot))[mask]
        nd_fp_stack[::, ::, i] = color
    return unstack_colors(nd_fp_stack)


async def process_raw(raw: SinarIA, dark_disable=False, flat_disable=False, simple_dark=True):
    img = await get_raw_pillow(raw)
    nd_img = img_as_float(img)

    if dark_disable:
        nd_img_b = nd_img
    else:
        if simple_dark:
            nd_img_b = await apply_local_black_simple(nd_img, raw.black_path)
        else:
            nd_img_b = await apply_local_black_ref_v8(nd_img, raw.black_path)

    if flat_disable:
        nd_img_flat = nd_img_b
    else:
        # TODO figure out better way to ident files, or build lensless flat?
        nd_img_flat = await apply_flat(raw, nd_img_b)

    nd_img_flat = nd_img_flat.clip(0, 1)
    return nd_img_flat


def norm2(x):
    return clip((x - x.max()) * (1 / (x.min() - x.max())), 0, 1)


def norm(x):
    return (x - npmin(x)) / (npmax(x) - npmin(x))


def c_norm(x, c_min, c_max):
    return (x - c_min) / (c_max - c_min)


def gle(b):
    return int.from_bytes(b, byteorder="little")


def gli(b, s):
    return int.from_bytes(b[s: s + 4], byteorder="little")


def gls(b, s):
    return int.from_bytes(b[s: s + 2], byteorder="little")


def sub(a: array, b: array):
    res = a - b
    res[b > a] = 0
    return res


async def create_master_flat(flats):
    corrected = []
    for flat in flats:
        if (
                abs(flat.measured_shutter_us - flat.req_shutter_us) / flat.req_shutter_us
                > 0.5
        ):
            # Our shutter was more than 50% slower than requested, skip this file
            print(
                f"Skipping {flat.filename}, shutter speed was {flat.measured_shutter_us}uS, "
                f"requested {flat.req_shutter_us}uS!"
            )
            continue
        nd_img = img_as_float(await get_raw_pillow(flat))
        black_path = flat.filename.parent.absolute() / Path(flat.black_ref).name
        biased_nd_img = apply_local_black_ref_v8(nd_img, black_path)
        norms = color_norm(biased_nd_img)
        corrected.append(unstack_colors(norms))
    flat_file = stack(corrected, axis=0)
    return median(flat_file, axis=0)


def color_norm(flat):
    colors = get_colors(flat)
    means = colors.mean(axis=(0, 1))
    return colors / means


def get_colors(arr):
    green1 = arr[0::2, 0::2]
    blue = arr[0::2, 1::2]
    red = arr[1::2, 0::2]
    green2 = arr[1::2, 1::2]
    return stack([green1, blue, red, green2], axis=-1)


def unstack_colors(colors):
    shape = [i * 2 for i in colors.shape[:-1]]
    arr = empty(shape=shape)
    arr[0::2, 0::2] = colors[::, ::, 0]
    arr[0::2, 1::2] = colors[::, ::, 1]
    arr[1::2, 0::2] = colors[::, ::, 2]
    arr[1::2, 1::2] = colors[::, ::, 3]
    return arr


async def process_list_of_flats_to_flat(flat_files):
    ia_flats = [await read_sinar(flat) for flat in flat_files]
    f0: SinarIA = ia_flats[0]
    lens = (
        f0.focal_length if f0.focal_length else input("Please enter the focal length: ")
    )
    corrected = await create_master_flat(ia_flats)
    save(str(FLAT_DIR / FILENAME.format(iso=f0.iso, lens=lens)), corrected)


# noinspection PyTypeChecker
async def apply_flat(raw: SinarIA, nd_img, use_lens=True):
    if raw.white_ref:
        try:
            flat_file = await read_raw(raw.white_path)
            lumps = await read_pwad_lumps(flat_file)
            flat_bytes = await get_wad(flat_file, *lumps[b"WHITE"])
            flat = img_as_float(
                Image.frombytes("I;16L", (raw.width, raw.height), flat_bytes, "raw")
            )
        except FileNotFoundError:
            print(f"Missing flat file: {raw.white_path}")
            return nd_img
    else:
        # Use custom-built flat vs. integrated one.
        iso, lens = raw.iso, raw.focal_length
        try:
            if use_lens and lens:
                flat = load(str(FLAT_DIR / FILENAME.format(iso=iso, lens=lens)))
            else:
                flat = load(str(FLAT_DIR / FILENAME.format(iso=iso, lens=lens)))
        except FileNotFoundError:
            print(f"No flat file exists iso{iso}, {lens}mm !")
            return nd_img
    return nd_img / flat


async def create_ia_dng(img: SinarIA, output_dir: Path, flat_disable=False, dark_disable=False):
    corrected_flat = await process_raw(img, flat_disable=flat_disable, dark_disable=dark_disable)
    nd_int = img_as_uint(corrected_flat)
    filename, r = await write_dng(img, nd_int, output_dir)
    return filename, r


def thumb_correct(img: SinarIA) -> Image:
    return img.thumb.transpose(Transpose.ROTATE_90)


async def write_dng(img, nd_int, output_dir):
    t = DNGTags()
    t.set(Tag.ImageWidth, img.width)
    t.set(Tag.ImageLength, img.height)
    t.set(Tag.Orientation, 8)  # Rotate 270 CW
    t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
    t.set(Tag.SamplesPerPixel, 1)
    t.set(Tag.BitsPerSample, [16])
    t.set(Tag.CFARepeatPatternDim, [2, 2])
    t.set(Tag.CFAPattern, CFAPattern.RGGB)
    t.set(Tag.BlackLevel, [nd_int.min()])
    t.set(Tag.WhiteLevel, [nd_int.max()])
    t.set(Tag.ColorMatrix1, CCM1)
    t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.Daylight)
    # TODO There is probably a way to get this from meta or calculate it automatically
    t.set(Tag.AsShotNeutral, [[1, 1], [1, 1], [1, 1]])
    t.set(Tag.Make, img.camera)
    t.set(Tag.Model, img.model)
    t.set(Tag.DNGVersion, DNGVersion.V1_4)
    t.set(Tag.DNGBackwardVersion, DNGVersion.V1_4)
    t.set(Tag.EXIFPhotoBodySerialNumber, img.serial)
    t.set(Tag.CameraSerialNumber, img.serial)
    t.set(Tag.ExposureTime, [(img.measured_shutter_us, 1000000)])
    t.set(Tag.PhotographicSensitivity, [img.iso])
    t.set(Tag.SensitivityType, 3)
    t.set(Tag.FocalLengthIn35mmFilm, [int(img.focal_length * 100), 62])
    t.set(Tag.FocalLength, [(int(img.focal_length), 1)])
    t.set(Tag.UniqueCameraModel, f"{img.model} ({img.serial}) on {img.camera}")
    t.set(Tag.FNumber, [(int(img.f_stop * 100), 100)])
    t.set(Tag.BayerGreenSplit, 0)
    t.set(Tag.Software, f"IAtoDNG v{__VERSION__}")
    t.set(Tag.PreviewColorSpace, PreviewColorSpace.sRGB)
    r = RAW2DNG()
    r.options(t, path="", compress=False)
    filename = output_dir / f"{img.filename.stem}.dng"
    r.convert(nd_int, filename=str(filename.absolute()))
    return filename, r


# async def main(parsed):
#     if parsed.build_flat:
#         flats = list(Path(parsed.build_flat).glob("*.IA"))
#         await process_list_of_flats_to_flat(flats)
#         return
#     if parsed.dump_meta:
#         for file in Path(parsed.dump_meta).glob("*.IA"):
#             print(f"Saving meta for {file.name}")
#             await dump_meta(file, Path(parsed.output))
#         return
#
#     output_path = Path(parsed.output)
#     os.makedirs(str(output_path.absolute()), exist_ok=True)
#     files = []
#     for i in parsed.images:
#         img_path = Path(i)
#         if img_path.is_dir():
#             files.extend(img_path.glob("*.IA"))
#         elif img_path.suffix == ".IA":
#             files.append(img_path)
#     print(f"Processing {len(files)} images!")
#     flat_disable = parsed.flat_disable
#     threads = []
#     for f in files:
#         # TODO Convert for better async?
#         t = partial(convert_ia_file_to_dng, f, flat_disable, output_path)
#         threads.append(asyncio.to_thread(t))
#     thread_count = multiprocessing.cpu_count()
#     for t_group in range(0, len(threads), thread_count):
#         await asyncio.gather(*threads[t_group: t_group + thread_count])


async def convert_ia_file_to_dng(f, flat_disable, output_path):
    ia_img = await read_sinar(f)
    f_name, _ = await create_ia_dng(ia_img, output_path, flat_disable=flat_disable)
    print(f"Processed: {f.absolute()} to {f_name.absolute()}")


async def dump_meta(f, output_dir):
    ia_img = await read_sinar(f)
    with open(output_dir / ia_img.filename.with_suffix(".meta").name, "wb") as fp:
        print(output_dir / ia_img.filename.with_suffix(".meta").name)
        fp.write(ia_img.meta)


async def read_meta(f):
    with open(f, "rb") as fp:
        meta = fp.read()
        ia = await process_meta(meta)
    return ia


# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("-o", "--output", default=".", help="Output directory.")
#     p.add_argument(
#         "-m", "--multi", default=False, action="store_true", help="Use multiprocessing"
#     )
#     p.add_argument(
#         "-f",
#         "--flat-disable",
#         default=False,
#         action="store_true",
#         help="Disable flat files",
#     )
#     p.add_argument(
#         "--build-flat",
#         default=False,
#         help="Use this directory of files to build a flat file.",
#     )
#     p.add_argument(
#         "images", nargs="*", help="List of images or directory of images to process."
#     )
#     p.add_argument(
#         "--dump-meta", default=False, help="Dump metadata of this directory"
#     )
#     args = p.parse_args()
#     asyncio.run(main(args))
