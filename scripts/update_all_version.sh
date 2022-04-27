#!/bin/bash

# Find this folder
d=`dirname $0`

version=`head -1 ${d}/VERSION`
forte_version=`head -1 ${d}/FORTE_VERSION`
echo "Updating versions to "${version}", depends on Forte "${forte_version}

for f in src/*/setup.py; do
    echo "Processing $f"
    sed -r -i "" "s/version=\"[0-9A-Za-z\.]+\",/version=\"${version}\",/1" $f
    sed -r -i "" "s/forte==[0-9A-Za-z\.]+\",/forte==${forte_version}\",/1" $f
done
