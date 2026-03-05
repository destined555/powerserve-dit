adb -s 3B657Q003DJ00000 shell "rm -rf /data/local/tmp/hjy/models_cpu/bin"
adb -s 3B657Q003DJ00000 push ./models_cpu/bin /data/local/tmp/hjy/models_cpu/
adb -s 3B657Q003DJ00000 shell
cd /data/local/tmp/hjy/models_cpu
./bin/powerserve-run-sd --workfolder ./SD3.5 --prompts "cat" --nprompts "ugly" 