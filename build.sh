#!/bin/sh
rm -r IAtoDNG.dmg IAtoDNG/IAtoDNG.app/
nuitka3  --remove-output --output-dir=IAtoDNG --macos-create-app-bundle --macos-signed-app-name=io.maxg.iatodngv1  --macos-sign-identity='Developer ID Application: Max Golub (2V8B69SADT)' --disable-console --macos-app-icon=src/iatodng/resources/logo.png --enable-plugin=numpy --standalone --include-data-dir=src/iatodng/resources/=resources --include-data-dir=venv/lib/python3.9/site-packages/toga/resources/=resources src/iatodng/IAtoDNG.py
create-dmg \
  --volname "IAtoDNG" \
  --volicon IAtoDNG/IAtoDNG.app/Contents/Resources/Icons.icns \
  --background src/iatodng/resources/iatodng.jpg \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "IAtoDNG.app" 200 190 \
  --hide-extension "IAtoDNG.app" \
  --app-drop-link 600 185 \
  "IAtoDNG.dmg" \
  IAtoDNG/
