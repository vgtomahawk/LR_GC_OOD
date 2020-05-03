
dname="fbreleasecoarse"

for seed in 2000 845738 34 454 5459
do
  echo "Seed ${seed}"
  #python -u train.py --epochs=30 --batch_size=8  --gpu=2 --dataset=${dname}  --unsup  --short_circuit_main --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=discriminativeClassifiers/oodExp3007_${dname}_unsup_discriminativeClassifierWithGlove_shortcircuitmain_${seed} --hendrycks --oodT=1000 --seed=${seed} --super_root="/home/vgangal/OOD_Detect_NLP/" 
  python train.py --epochs=30 --batch_size=8  --gpu=2 --dataset=${dname}  --unsup  --short_circuit_main --word_vectors=vector_cache/glove.6B.100d.txt.p  --resume_snapshot=discriminativeClassifiers/oodExp3007_${dname}_unsup_discriminativeClassifierWithGlove_shortcircuitmain_${seed}/best_snapshot.pt --ratioKldPrune --oodT=1 --infer_only --seed=${seed} --super_root="/home/vgangal/OOD_Detect_NLP/" | tee inference_logs/oodExp3007_${dname}_unsup_discriminativeClassifierWithGlove_shortcircuitmain_ratioKldPrune_${seed}
done

#echo "Entering L_{simple} or in other words, Plain LM based approaches on the fbreleasecoarse data"
#for seed in 2000 845738 34 454 5459
#do
#  echo "Seed ${seed}"
#  echo "Training  Plain LM, Various Sizes" 
#  for backInputSize in 64 128 256
#  do
#    python train.py  --epochs=7  --batch_size=8 --gpu=0 --dataset=${dname}  --unsup --generative --fore_lm --back_input_size ${backInputSize} --no-bidirectional  --marginal  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_marginal_foreLm_backInputSize=${backInputSize}_${seed} --seed=${seed} &> generative_classifier_logs/oodExp3007_${dname}_unsup_generative_marginal_foreLm_backInputSize=${backInputSize}_${seed}
#  done
#
#  echo "Training  Plain LM With Background, Various Sizes, Uniform , Unigram and Uniroot" 
#  for noiseType in uniform unigram uniroot
#  do
#    for noiseLevel in 0.5
#    do
#      for backInputSize in 64 128 256
#      do
#        echo "nothing"
#        python train.py  --epochs=7  --batch_size=8 --gpu=0 --dataset=${dname}  --unsup --generative --fore_lm --back_input_size ${backInputSize} --no-bidirectional  --marginal --back_lm --corrupt_back --noise_level ${noiseLevel} --noise_type=${noiseType}  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_marginal_foreLm_backLm_backInputSize=${backInputSize}_corruptBack_noiseLevel=${noiseLevel}_noiseType=${noiseType}_${seed} --seed=${seed} &> generative_classifier_logs/oodExp3007_${dname}_unsup_generative_marginal_foreLm_backLm_backInputSize=${backInputSize}_corruptBack_noiseLevel=${noiseLevel}_noiseType=${noiseType}_${seed}
#      done
#    done
#  done
#  
#done
#echo "Exiting L_{simple} or in other words, Plain LM based approaches on the fbreleasecoarse data"


#echo "Entering Discriminative Approaches - MSP [Hendrycks], MSP with high \tau [Hendrycks, oodT=1000], KLD(P|U) [kldPrune], LOF [lof]"
#for seed in 2000 845738 34 454 5459
#do
#  echo "Seed ${seed}"
#  python train.py --epochs=30 --batch_size=8  --gpu=0 --dataset=${dname}  --unsup  --short_circuit_main --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=discriminativeClassifiers/oodExp3007_${dname}_unsup_discriminativeClassifierWithGlove_shortcircuitmain_${seed} --hendrycks --oodT=1000 --seed=${seed}  &> discriminative_classifier_logs/oodExp3007_${dname}_unsup_discriminativeClassifierWithGlove_shortcircuitmain_${seed}
#
#  python train.py --epochs=30 --batch_size=8  --gpu=0 --dataset=${dname}  --unsup  --short_circuit_main --word_vectors=vector_cache/glove.6B.100d.txt.p  --resume_snapshot=discriminativeClassifiers/oodExp3007_${dname}_unsup_discriminativeClassifierWithGlove_shortcircuitmain_${seed}/best_snapshot.pt --hendrycks --oodT=1 --infer_only --seed=${seed}  &> inference_logs/oodExp3007_${dname}_unsup_discriminativeClassifierWithGlove_shortcircuitmain_hendrycks_oodT=1_${seed}
#
#  python train.py --epochs=30 --batch_size=8  --gpu=0 --dataset=${dname}  --unsup  --short_circuit_main --word_vectors=vector_cache/glove.6B.100d.txt.p  --resume_snapshot=discriminativeClassifiers/oodExp3007_${dname}_unsup_discriminativeClassifierWithGlove_shortcircuitmain_${seed}/best_snapshot.pt --hendrycks --oodT=1000 --infer_only --seed=${seed}  &> inference_logs/oodExp3007_${dname}_unsup_discriminativeClassifierWithGlove_shortcircuitmain_hendrycks_oodT=1000_${seed}
#
#  python train.py --epochs=30 --batch_size=8  --gpu=0 --dataset=${dname}  --unsup  --short_circuit_main --word_vectors=vector_cache/glove.6B.100d.txt.p  --resume_snapshot=discriminativeClassifiers/oodExp3007_${dname}_unsup_discriminativeClassifierWithGlove_shortcircuitmain_${seed}/best_snapshot.pt --kldPrune --oodT=1 --infer_only --seed=${seed}  &> inference_logs/oodExp3007_${dname}_unsup_discriminativeClassifierWithGlove_shortcircuitmain_kldPrune_${seed}
#
#  for contamination in 0.02 0.04 0.06 0.08 0.10 0.20 0.30
#  do
#    echo "Inference for LOF+${contamination}"
#    python train.py --epochs=30 --batch_size=8  --gpu=0 --dataset=${dname}  --unsup  --short_circuit_main --word_vectors=vector_cache/glove.6B.100d.txt.p  --resume_snapshot=discriminativeClassifiers/oodExp3007_${dname}_unsup_discriminativeClassifierWithGlove_shortcircuitmain_${seed}/best_snapshot.pt --lof --contamination ${contamination} --returnIntermediate --oodT=1 --infer_only --seed=${seed}  &> inference_logs/oodExp3007_${dname}_unsup_discriminativeClassifierWithGlove_shortcircuitmain_lof_returnIntermediate_contamination=${contamination}_${seed}
#  done
#done
#echo "Exiting Discriminative Approaches - MSP [Hendrycks], MSP with high \tau [Hendrycks, oodT=1000], KLD(P|U) [kldPrune], LOF [lof]"

#echo "Entering Generative Classifier Based Approaches - L_{gen}, L_{gen} with background LMs" 
#for seed in 2000 845738 34 454 5459
#do
#  echo "Seed ${seed}"
#
#  echo "Training Generative Classifier Without any Background"
#  python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset=${dname} --unsup --generative  --no-bidirectional --at_hidden  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_atHidden_${seed} --hendrycks --oodT=1000 --seed=${seed} &> generative_classifier_logs/oodExp3007_${dname}_unsup_generative_atHidden_${seed}
#
#  echo "Generative Classifier Inference Without any Background Hendrycks"
#  python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset=${dname} --unsup --generative  --no-bidirectional --at_hidden  --word_vectors=vector_cache/glove.6B.100d.txt.p  --resume_snapshot=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_atHidden_${seed}/best_snapshot.pt --hendrycks --oodT=1000 --infer_only --seed=${seed} &> inference_logs/oodExp3007_${dname}_unsup_generative_atHidden_hendrycks_oodT=1000_${seed}
#
#  echo "Generative Classifier Inference Without any Background; Marginal"
#  python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset=${dname} --unsup --generative  --no-bidirectional --at_hidden  --word_vectors=vector_cache/glove.6B.100d.txt.p  --resume_snapshot=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_atHidden_${seed}/best_snapshot.pt --marginal --infer_only --seed=${seed} &> inference_logs/oodExp3007_${dname}_unsup_generative_atHidden_marginal_${seed}
#
#
#  echo "Training Generative Classifier With Background LM, noiseType uniroot, uniform and unigram, Background LM various sizes"
#  for noiseType in uniroot uniform unigram
#  do  
#    for noiseLevel in 0.5
#    do
#      for backInputSize in 64
#      do
#        python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset=${dname} --unsup --generative  --no-bidirectional --at_hidden  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_atHidden_marginal_backLm_backInputSize=${backInputSize}_corruptBack_noiseLevel=${noiseLevel}_noiseType=${noiseType}_${seed} --marginal --back_lm --back_input_size=${backInputSize} --corrupt_back --noise_level=${noiseLevel} --noise_type=${noiseType} --oodT=1 --seed=${seed} &> generative_classifier_logs/oodExp3007_${dname}_unsup_generative_atHidden_marginal_backLm_backInputSize=${backInputSize}_corruptBack_noiseLevel=${noiseLevel}_noiseType=${noiseType}_${seed} 
#      done
#    done
#  done
#
#done
#echo "Exiting Generative Classifier Based Approaches - L_{gen}, L_{gen} with background LMs" 

#echo "Entering Generative Classifier With Background Mirror"
#for seed in 2000 845738 34 454 5459
#do
#  echo "Seed ${seed}"
#  for noiseLevel in 0.5
#  do
#    python train.py  --epochs=15  --batch_size=8 --gpu=1 --dataset=${dname} --unsup --generative  --no-bidirectional --at_hidden  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_atHidden_marginal_backMirror_corruptBack_noiseLevel=${noiseLevel}_noiseType=uniroot_${seed} --marginal --back_mirror  --corrupt_back --noise_level=${noiseLevel} --noise_type=uniroot --oodT=1 --seed=${seed} &> generative_classifier_logs/oodExp3007_${dname}_unsup_generative_atHidden_marginal_backMirror_corruptBack_noiseLevel=${noiseLevel}_noiseType=uniroot_${seed} 
#  done
#done
#echo "Exiting Generative Classifier With Background Mirror"


#python train.py --epochs=30 --batch_size=8  --gpu=0 --dataset="fbrelease"  --unsup  --short_circuit_main --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=discriminativeClassifiers/oodExp3007_fbrelease_unsup_discriminativeClassifierWithGlove_shortcircuitmain --hendrycks --oodT=1000  &> discriminative_classifier_logs/oodExp3007_fbrelease_unsup_discriminativeClassifierWithGlove_shortcircuitmain


#python train.py --epochs=30 --batch_size=8  --gpu=0 --dataset="fbrelease"  --unsup  --short_circuit_main --word_vectors=vector_cache/glove.6B.100d.txt.p  --resume_snapshot=discriminativeClassifiers/oodExp3007_fbrelease_unsup_discriminativeClassifierWithGlove_shortcircuitmain/best_snapshot.pt --hendrycks --oodT=1 --infer_only  &> inference_logs/oodExp3007_fbrelease_unsup_discriminativeClassifierWithGlove_shortcircuitmain_hendrycks_oodT=1

#python train.py --epochs=30 --batch_size=8  --gpu=0 --dataset="fbrelease"  --unsup  --short_circuit_main --word_vectors=vector_cache/glove.6B.100d.txt.p  --resume_snapshot=discriminativeClassifiers/oodExp3007_fbrelease_unsup_discriminativeClassifierWithGlove_shortcircuitmain/best_snapshot.pt --hendrycks --oodT=1000 --infer_only  &> inference_logs/oodExp3007_fbrelease_unsup_discriminativeClassifierWithGlove_shortcircuitmain_hendrycks_oodT=1000

#python train.py --epochs=30 --batch_size=8  --gpu=0 --dataset="fbrelease"  --unsup  --short_circuit_main --word_vectors=vector_cache/glove.6B.100d.txt.p  --resume_snapshot=discriminativeClassifiers/oodExp3007_fbrelease_unsup_discriminativeClassifierWithGlove_shortcircuitmain/best_snapshot.pt --kldPrune --oodT=1 --infer_only  &> inference_logs/oodExp3007_fbrelease_unsup_discriminativeClassifierWithGlove_shortcircuitmain_kldPrune

#for contamination in 0.02 0.04 0.06 0.08 0.10 0.20 0.30
#do
#  echo "Inference for LOF+${contamination}"
#  python train.py --epochs=30 --batch_size=8  --gpu=0 --dataset="fbrelease"  --unsup  --short_circuit_main --word_vectors=vector_cache/glove.6B.100d.txt.p  --resume_snapshot=discriminativeClassifiers/oodExp3007_fbrelease_unsup_discriminativeClassifierWithGlove_shortcircuitmain/best_snapshot.pt --lof --contamination ${contamination} --returnIntermediate --oodT=1 --infer_only  &> inference_logs/oodExp3007_fbrelease_unsup_discriminativeClassifierWithGlove_shortcircuitmain_lof_returnIntermediate_contamination=${contamination}
#done

#for calibFrac in 0.05 0.1 0.2 0.3
#do
#  echo "Inference for PPNF+${calibFrac}"
#  python train.py --epochs=30 --batch_size=8  --gpu=0 --dataset="fbrelease"  --unsup  --short_circuit_main --word_vectors=vector_cache/glove.6B.100d.txt.p  --resume_snapshot=discriminativeClassifiers/oodExp3007_fbrelease_unsup_discriminativeClassifierWithGlove_shortcircuitmain/best_snapshot.pt --ppnf --calib_frac ${calibFrac} --returnIntermediate --oodT=1 --infer_only  &> inference_logs/oodExp3007_fbrelease_unsup_discriminativeClassifierWithGlove_shortcircuitmain_ppnf_returnIntermediate_calibFrac=${calibFrac}
#done


dname="fbrelease"

#echo "Training Generative Classifier Without any Background"
#python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset=${dname} --unsup --generative  --no-bidirectional --at_hidden  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_atHidden --hendrycks --oodT=1000 &> generative_classifier_logs/oodExp3007_${dname}_unsup_generative_atHidden

#echo "Generative Classifier Inference Without any Background Hendrycks"
#python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset=${dname} --unsup --generative  --no-bidirectional --at_hidden  --word_vectors=vector_cache/glove.6B.100d.txt.p  --resume_snapshot=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_atHidden/best_snapshot.pt --hendrycks --oodT=1000 --infer_only &> inference_logs/oodExp3007_${dname}_unsup_generative_atHidden_hendrycks_oodT=1000

#echo "Generative Classifier Inference Without any Background; Marginal"
#python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset=${dname} --unsup --generative  --no-bidirectional --at_hidden  --word_vectors=vector_cache/glove.6B.100d.txt.p  --resume_snapshot=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_atHidden/best_snapshot.pt --marginal --infer_only &> inference_logs/oodExp3007_${dname}_unsup_generative_atHidden_marginal


#echo "Training Generative Classifier With Background LM, Background LM various sizes"
#for noiseLevel in 0.5 0.3 0.7
#do
#  for backInputSize in 64 128 256
#  do
    #python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset=${dname} --unsup --generative  --no-bidirectional --at_hidden  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_atHidden_marginal_backLm_backInputSize=${backInputSize}_corruptBack_noiseLevel=${noiseLevel}_noiseType=uniroot --marginal --back_lm --back_input_size=${backInputSize} --corrupt_back --noise_level=${noiseLevel} --noise_type=uniroot --oodT=1 &> generative_classifier_logs/oodExp3007_${dname}_unsup_generative_atHidden_marginal_backLm_backInputSize=${backInputSize}_corruptBack_noiseLevel=${noiseLevel}_noiseType=uniroot 
#  done
#done

#for noiseType in uniform
#do
#  for noiseLevel in 0.5
#  do
#    for backInputSize in 64
#    do
#      echo "Inference"
#      python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset=${dname} --unsup --generative  --no-bidirectional --at_hidden  --word_vectors=vector_cache/glove.6B.100d.txt.p  --resume_snapshot=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_atHidden_marginal_backLm_backInputSize=${backInputSize}_corruptBack_noiseLevel=${noiseLevel}_noiseType=${noiseType}/best_snapshot.pt --marginal --back_lm --back_input_size=${backInputSize} --corrupt_back --noise_level=${noiseLevel} --noise_type=${noiseType} --oodT=1 --infer_only &> inference_logs/oodExp3007_${dname}_unsup_generative_atHidden_marginal_backLm_backInputSize=${backInputSize}_corruptBack_noiseLevel=${noiseLevel}_noiseType=${noiseType}_marginal 
#    done
#  done
#done

#echo "Training Generative Classifier With Background LM, noiseType uniform and unigram, Background LM various sizes"
#for noiseType in uniform unigram
#do  
#  for noiseLevel in 0.5
#  do
#    for backInputSize in 64
#    do
#      python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset=${dname} --unsup --generative  --no-bidirectional --at_hidden  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_atHidden_marginal_backLm_backInputSize=${backInputSize}_corruptBack_noiseLevel=${noiseLevel}_noiseType=${noiseType} --marginal --back_lm --back_input_size=${backInputSize} --corrupt_back --noise_level=${noiseLevel} --noise_type=${noiseType} --oodT=1 &> generative_classifier_logs/oodExp3007_${dname}_unsup_generative_atHidden_marginal_backLm_backInputSize=${backInputSize}_corruptBack_noiseLevel=${noiseLevel}_noiseType=${noiseType} 
#    done
#  done
#done


#echo "Training Generative Classifier With Background Mirror, Background Mirror various sizes"
#for noiseLevel in 0.5 0.3 0.7
#do
#  python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset=${dname} --unsup --generative  --no-bidirectional --at_hidden  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_atHidden_marginal_backMirror_corruptBack_noiseLevel=${noiseLevel}_noiseType=uniroot --marginal --back_mirror  --corrupt_back --noise_level=${noiseLevel} --noise_type=uniroot --oodT=1 &> generative_classifier_logs/oodExp3007_${dname}_unsup_generative_atHidden_marginal_backMirror_corruptBack_noiseLevel=${noiseLevel}_noiseType=uniroot 
#done


#echo "Running Generative Classifier With Scrambled Background"
#python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset=${dname} --unsup --generative  --no-bidirectional --at_hidden  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_atHidden_marginal_scramble --marginal --scramble --oodT=1 &> generative_classifier_logs/oodExp3007_${dname}_unsup_generative_atHidden_marginal_scramble

#echo "Training  Plain LM, Various Sizes" 
#for backInputSize in 64 128 256
#do
#  python train.py  --epochs=7  --batch_size=8 --gpu=0 --dataset=${dname}  --unsup --generative --fore_lm --back_input_size ${backInputSize} --no-bidirectional  --marginal  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_marginal_foreLm_backInputSize=${backInputSize} &> generative_classifier_logs/oodExp3007_${dname}_unsup_generative_marginal_foreLm_backInputSize=${backInputSize}
#done

#echo "Training  Plain LM With Background, Various Sizes," 
#for noiseLevel in 0.5 0.3 0.7
#do
#  for backInputSize in 64 128 256
#  do
#    python train.py  --epochs=7  --batch_size=8 --gpu=1 --dataset=${dname}  --unsup --generative --fore_lm --back_input_size ${backInputSize} --no-bidirectional  --marginal --back_lm --corrupt_back --noise_level ${noiseLevel} --noise_type=uniroot  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_marginal_foreLm_backLm_backInputSize=${backInputSize}_corruptBack_noiseLevel=${noiseLevel}_noiseType=uniroot &> generative_classifier_logs/oodExp3007_${dname}_unsup_generative_marginal_foreLm_backLm_backInputSize=${backInputSize}_corruptBack_noiseLevel=${noiseLevel}_noiseType=uniroot
#  done
#done

#echo "Training  Plain LM With Background, Various Sizes, Uniform and Unigram" 
#for noiseType in uniform unigram
#do
#  for noiseLevel in 0.5
#  do
#    for backInputSize in 64 128 256
#    do
#      python train.py  --epochs=7  --batch_size=8 --gpu=1 --dataset=${dname}  --unsup --generative --fore_lm --back_input_size ${backInputSize} --no-bidirectional  --marginal --back_lm --corrupt_back --noise_level ${noiseLevel} --noise_type=${noiseType}  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_generative_marginal_foreLm_backLm_backInputSize=${backInputSize}_corruptBack_noiseLevel=${noiseLevel}_noiseType=${noiseType} &> generative_classifier_logs/oodExp3007_${dname}_unsup_generative_marginal_foreLm_backLm_backInputSize=${backInputSize}_corruptBack_noiseLevel=${noiseLevel}_noiseType=${noiseType}
#    done
#  done
#done


#for id_ratio in 0.25 0.75
#do
#  for split_id in 0 1 2 3 4 5 6 7 8 9
#  do
#    dname="fbmlto"
#    echo "Running Discriminative Classifier"
    #python train.py --epochs=15 --batch_size=8  --gpu=0 --dataset="fbmlto" --id_ratio=${id_ratio} --split_id=${split_id} --unsup  --short_circuit_main --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=discriminativeClassifiers/oodExp3007_fbmlto_unsup_idRatio${id_ratio}_splitId${split_id}_discriminativeClassifierWithGlove_shortcircuitmain --hendrycks --oodT=1000  &> discriminative_classifier_logs/oodExp3007_fbmlto_unsup_idRatio${id_ratio}_splitId${split_id}_discriminativeClassifierWithGlove_shortcircuitmain
    #python train.py --epochs=15 --batch_size=8  --gpu=0 --dataset="fbmlto" --id_ratio=${id_ratio} --split_id=${split_id} --unsup  --short_circuit_main --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=discriminativeClassifiers/oodExp3007_fbmlto_unsup_idRatio${id_ratio}_splitId${split_id}_discriminativeClassifierWithGlove_shortcircuitmain --hendrycks --oodT=1000  &> discriminative_classifier_logs/oodExp3007_fbmlto_unsup_idRatio${id_ratio}_splitId${split_id}_discriminativeClassifierWithGlove_shortcircuitmain
    #echo "Running Generative Classifier Without any Background  With in-domain ratio ${id_ratio} and split-id ${split_id}"
    #python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset="fbmlto" --id_ratio=${id_ratio} --split_id=${split_id} --unsup --generative  --no-bidirectional --at_hidden  --marginal   --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_fbmlto_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_marginal &> generative_classifier_logs/oodExp3007_fbmlto_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_marginal
    #echo "Running Plain LM and Marginal for ${id_ratio} and split-id ${split_id}"
    #python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset=${dname} --id_ratio=${id_ratio} --split_id=${split_id} --unsup --generative --fore_lm --back_input_size 64 --no-bidirectional  --marginal  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_idRatio${id_ratio}_splitId${split_id}_generative_marginal_foreLm_backInputSize=64 &> generative_classifier_logs/oodExp3007_${dname}_unsup_idRatio${id_ratio}_splitId${split_id}_generative_marginal_foreLm_backInputSize=64
    #echo "Running Plain LM and Background-LM for ${id_ratio} and split-id ${split_id}"
    #python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset=${dname} --id_ratio=${id_ratio} --split_id=${split_id} --unsup --generative --fore_lm --back_input_size 64 --no-bidirectional  --marginal --back_lm --corrupt_back --noise_level=0.5 --noise_type=uniroot  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_idRatio${id_ratio}_splitId${split_id}_generative_marginal_foreLm_backLm_backInputSize=64_corruptBack_noiseLevel=0.5_noiseType=uniroot &> generative_classifier_logs/oodExp3007_${dname}_unsup_idRatio${id_ratio}_splitId${split_id}_generative_marginal_foreLm_backLm_backInputSize=64_corruptBack_noiseLevel=0.5_noiseType=uniroot
    #echo "Running Generative Classifier Without any Background  With in-domain ratio ${id_ratio} and split-id ${split_id} And Using Hendrycks"
    #python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset=${dname} --id_ratio=${id_ratio} --split_id=${split_id} --unsup --generative  --no-bidirectional --at_hidden  --hendrycks  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_hendrycks --oodT=1000 &> generative_classifier_logs/oodExp3007_${dname}_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_hendrycks
    #echo "Running Generative Classifier Without any Background  With in-domain ratio ${id_ratio} and split-id ${split_id} and Scrambled Background"
    #python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset="fbmlto" --id_ratio=${id_ratio} --split_id=${split_id} --unsup --generative  --no-bidirectional --at_hidden  --marginal --scramble  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_fbmlto_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_marginal &> generative_classifier_logs/oodExp3007_fbmlto_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_marginal_scramble
    #echo "Running Generative Classifier Without any Background  With in-domain ratio ${id_ratio} and split-id ${split_id} and Scrambled Background+BackgroundLM"
    #python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset="fbmlto" --id_ratio=${id_ratio} --split_id=${split_id} --unsup --generative  --no-bidirectional --at_hidden  --marginal --back_lm --corrupt_back --noise_level=0.5 --noise_type=uniroot --back_input_size=64 --scramble  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_fbmlto_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_marginal_backLm_backInputSize=64_corruptBack_noiseLevel=0.5_noiseType=uniroot_withMasking_scramble &> generative_classifier_logs/oodExp3007_fbmlto_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_marginal_backLm_backInputSize=64_corruptBack_noiseLevel=0.5_noiseType=uniroot_withMasking_scramble
    #echo "Running Generative Classifier + Background LM  With in-domain ratio ${id_ratio} and split-id ${split_id}"
    #python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset="fbmlto" --id_ratio=${id_ratio} --split_id=${split_id} --unsup --generative  --no-bidirectional --at_hidden  --marginal --back_lm --corrupt_back --noise_level=0.5 --noise_type=uniroot --back_input_size=64 --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_fbmlto_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_marginal_backLm_backInputSize=64_corruptBack_noiseLevel=0.5_noiseType=uniroot &> generative_classifier_logs/oodExp3007_fbmlto_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_marginal_backLm_backInputSize=64_corruptBack_noiseLevel=0.5_noiseType=uniroot_withMasking
    #echo "Running Generative Classifier + Background LM With Diff Training  With in-domain ratio ${id_ratio} and split-id ${split_id}"
    #python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset="snips" --id_ratio=${id_ratio} --split_id=${split_id} --unsup --generative  --no-bidirectional --at_hidden  --marginal --back_lm --back_lm_diff --corrupt_back --noise_level=0.5 --noise_type=uniroot --back_input_size=64 --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp0507_snips_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_marginal_backLm_backLmDiff_backInputSize=64_corruptBack_noiseLevel=0.5_noiseType=uniroot &> logs/oodExp0507_snips_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_marginal_backLm_backLmDiff_backInputSize=64_corruptBack_noiseLevel=0.5_noiseType=uniroot
    #echo "Running Generative Classifier Without any Background  With in-domain ratio ${id_ratio} and split-id ${split_id}"
    #python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset="snips" --id_ratio=${id_ratio} --split_id=${split_id} --unsup --generative  --no-bidirectional --at_hidden  --marginal  --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp0507_snips_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_marginal &> logs/oodExp0507_snips_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_marginal
    #echo "Running Generative Classifier + Background Mirror  With in-domain ratio ${id_ratio} and split-id ${split_id}"
    #python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset=${dname} --id_ratio=${id_ratio} --split_id=${split_id} --unsup --generative  --no-bidirectional --at_hidden  --marginal --back_mirror --corrupt_back --noise_level=0.5 --noise_type=uniroot --word_vectors=vector_cache/glove.6B.100d.txt.p  --save_path=generativeClassifiersWithBackgrounds/ckpt_oodExp3007_${dname}_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_marginal_backMirror_corruptBack_noiseLevel=0.5_noiseType=uniroot &> generative_classifier_logs/oodExp3007_${dname}_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_marginal_backMirror_corruptBack_noiseLevel=0.5_noiseType=uniroot 
    #python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset="snips" --id_ratio=${id_ratio} --split_id=${split_id} --unsup --generative  --no-bidirectional --at_hidden    --word_vectors=vector_cache/glove.6B.100d.txt.p  --resume_snapshot=generativeClassifiersWithBackgrounds/ckpt_oodExp0507_snips_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_marginal/best_snapshot.pt --infer_only --hendrycks --oodT=1000
    #python train.py  --epochs=15  --batch_size=8 --gpu=0 --dataset="snips" --id_ratio=${id_ratio} --split_id=${split_id} --unsup --generative  --no-bidirectional --at_hidden    --word_vectors=vector_cache/glove.6B.100d.txt.p  --resume_snapshot=generativeClassifiersWithBackgrounds/ckpt_oodExp0507_snips_unsup_idRatio${id_ratio}_splitId${split_id}_generative_atHidden_marginal/best_snapshot.pt --infer_only --tentropy
  #done
#done
