for f in src/*/setup.py; do
    echo "Processing $f"
    sed -r -i "" "s/version=\"[0-9\.]+\",/version=\"$1\",/1" $f
done