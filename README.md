# IAtoDNG
Converts Sinar Emotion 22, 54, 54LV, 75, and 75LV .IA files to DNG files.

## Features
 * Parses the META section of Sinar IA files and converts them to the proper exif tags. 
   * Note - Capture One can't see these for some reason, but Lightroom can.
 * Is able to find and apply the dark frame/bias frame that emotion 22 and 75 backs capture automatically.
(These are the BR files included with the IA files.)
 * Uses the .WR flat file produced by emotion 75 backs to correct differences between CCD tiles.
 * Can Apply a manually created white flat to each image. This corrects the
difference between the CCD tiles and any dust in the optical path (If using a lens).


## Install
For now, this is not packaged as a Python module.
You have to clone the repo and run things manually. 

```bash
git clone https://github.com/pyemotiondng/pyemotiondng.git
cd pyemotiondng
python3 -m venv venv
source venv/bin/activate
pip intall -r requirements.txt
```

## Usage

```bash
> python3 sinar_ia.py --help
usage: sinar_ia.py [-h] [-o OUTPUT] [-m] [-f] [--build-flat BUILD_FLAT]
                   [--dump-meta DUMP_META]
                   [images ...]

positional arguments:
  images                List of images or directory of images to process.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output directory.
  -m, --multi           Use multiprocessing
  -f, --flat-disable    Disable flat files
  --build-flat BUILD_FLAT
                        Use this directory of files to build a flat file.
  --dump-meta DUMP_META
                        Dump meta data of this directory

```
### Examples

#### Process a directory of Sinar IA images:
```bash
python3 sinar_ia.py /VOLUMES/PHASEONE/002000E8.EMO/ -o ~/Pictures/pyemotiondng_output
```

#### With multithreading:
```bash
python3 sinar_ia.py -m /VOLUMES/PHASEONE/002000E8.EMO/ -o ~/Pictures/pyemotiondng_output
```

#### Building flat files from directory of images:
```bash
python3  sinar_ia.py --build-flat ~/Pictures/emotion22_80mm_f4_iso50_flats/
```

* This will store a flat file for the ISO/lens combination in `~/.local/pyEMDNG/flats/`.
* There is no limit to the number of files you can use to generate your master flat,
but the recommendation seems to be 20+ shots. 
* If the Meta section of the raw file does not have any lens info, then the file is saved 
with 0mm as the focal length - It seems only Rollei AF(D) lens record their focal length to the raw metadata.
* I recommend creating a "lensless" flat file(s) without any lens mounted, for each ISO 
to balance the CCD halves and remove sensor dust. Any lens that does not report a focal length
will use the correct lensless flat file for that iso. 

#### Dump META section of raw files:
```bash
python3  sinar_ia.py --dump-meta ~/Pictures/emotion22_80mm_f4_iso50_flats/ -o ~/Pictures/pyemotiondng_meta
```

## TODO
* Read and apply the CCD calibration files included with Sinar backs.
  * This is possibly already captured by emotion 75 .WR files.
* Understand and apply the LINEAR and BLEMISH components of .WR files.
* Tweak meta processing for other Camera bodies, such as the Mamiya AFD.
* Package this repo into a Python module.
* Some kind of GUI.
* This could all be wrong :) 
