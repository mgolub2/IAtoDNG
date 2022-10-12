#!/bin/sh
mkdir nuitka_build 
cd nuitka_build
python -m nuitka  --remove-output --macos-create-app-bundle --macos-signed-app-name=io.maxg.iatodngv1 --disable-console --macos-app-icon=../src/iatodng/resources/logo.png --experimental=macos-sign-runtime --enable-plugin=numpy --standalone --include-data-dir=../src/iatodng/resources/=resources --include-data-dir=../venv/lib/python3.9/site-packages/toga/resources/=resources ../src/iatodng/app.py
