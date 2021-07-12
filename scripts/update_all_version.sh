for f in src/*/setup.py; do
    echo "Processing $f"
    sed -r -i"" "s/version=\"[0-9A-Za-z\.]+\",/version=\"$1\",/1" $f
done
