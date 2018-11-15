if [ $# -eq 0 ]; then
      echo "Must pass model path (without .meta or other extensions) as an argument"
      exit
fi

mkdir -p sample_results
for t in blend++ blend chars phones; do
    mkdir -p sample_results/
    if [[ -z "$2" ]]; then
        python -u sample_rnn_unaligned_speech_ljspeech.py "$1" custom_test.txt taco_prosody_test.txt taco_small_test.txt quote_test.txt basic_test.txt valid --inp=$t --sonify=1000 2>&1 | tee /Tmp/kastner/sample_log.txt
    fi
    if [[ ! -z "$2" ]]; then 
        python -u sample_rnn_unaligned_speech_ljspeech.py "$1" custom_test.txt taco_prosody_test.txt taco_small_test.txt quote_test.txt basic_test.txt valid "$2" --inp=$t --sonify=1000 2>&1 | tee /Tmp/kastner/sample_log.txt
        #python sample_rnn_unaligned_speech_ljspeech.py "$1" "$2" --inp=$t --test=$s --sonify=1000 2>&1 | tee sample_results/"$t"_"$s"/sample_log.txt
    fi
    #python sample_rnn_unaligned_speech_ljspeech.py "$1" --inp=$t --test=$s 2>&1 | tee sample_results/"$t"_"$s"/sample_log.txt
    mv *sampled_text_summary.txt sample_results/
    mv /Tmp/kastner/sample_log.txt sample_results/
done

mv *.wav sample_results/
mv *.png sample_results/
