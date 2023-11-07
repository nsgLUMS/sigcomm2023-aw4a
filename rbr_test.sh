#!/usr/bin/bash
filename="$1"
while read -r line; do
    IFS=',' read -r -a urlarr <<< $line
    url=${urlarr[0]}
    echo $url

    python3 main.py -w https://www.$url -r ${urlarr[1]} -pm -t 0.9
    
done < "$filename"
