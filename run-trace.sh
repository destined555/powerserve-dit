echo "正在传输文件到手机"
adb -s 3B15CR0014H00000  shell  "rm -rf /data/local/tmp/hjy/rag"
adb -s 3B15CR0014H00000  shell  "mkdir -p /data/local/tmp/hjy/rag"
adb -s 3B15CR0014H00000 push  ./models/. /data/local/tmp/hjy/rag
adb -s 3B15CR0014H00000  push  ./prompts-1k  /data/local/tmp/hjy/rag

echo "文件传输完成，正在进入手机执行"
adb -s 3B15CR0014H00000 shell "cd data/local/tmp/hjy/rag && export LD_LIBRARY_PATH=/data/local/tmp/hjy/rag/qnn_libs && ./bin/powerserve-run --work-folder .   --n-predicts 128 -f prompts-1k/2wikimqa_idx18_1301tk.txt"

echo "执行完成,正在将结果传回本地"
rm -rf ./trace.data
adb -s 3B15CR0014H00000  pull /data/local/tmp/hjy/rag/trace.data /home/frp/jinye/rag.xpu