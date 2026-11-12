#!/bin/bash
# Round 6 (2026-11-03 21:45 EST): resubmit h_fm1600; ensemble-teacher students; FM one-factor tests; 2 more SAM teachers per split.
cd /scratch/as21153/dl1
S=scripts/hpc/train.sbatch
C="--ema 0.999 --student-noise 0.04 --mix-alpha 1.0 --mix-prob 0.5"
T="--model wrn28_10 --epochs 400 --ema 0.999 --label-smoothing 0 --sam-rho 0.05"
sub() { sbatch --parsable "$@" | cut -d';' -f1; }
echo h_fm1600 $(sub --job-name=h_fm1600 --time=12:00:00 $S --model resnet_c --epochs 1600 $C --teacher runs/teacher_wrn300/final.pt)
echo h_fm800_ens2 $(sub --job-name=h_fm800_ens2 --time=10:00:00 $S --model resnet_c --epochs 800 $C --teacher runs/teacher_wrn300/final.pt runs/h_teacher_sam45/final.pt)
echo h_final800_ens2 $(sub --job-name=h_final800_ens2 --time=10:00:00 $S --model resnet_c --epochs 800 $C --val-size 0 --teacher runs/teacher50/final.pt runs/h_teacher_sam50/final.pt)
echo h_fm800_a1 $(sub --job-name=h_fm800_a1 $S --model resnet_c --epochs 800 $C --kd-alpha 1.0 --teacher runs/teacher_wrn300/final.pt)
echo h_fm800_nn $(sub --job-name=h_fm800_nn $S --model resnet_c --epochs 800 --ema 0.999 --student-noise 0 --mix-alpha 1.0 --mix-prob 0.5 --teacher runs/teacher_wrn300/final.pt)
a=$(sub --job-name=h_teacher_sam45_s1 $S $T --seed 1)
b=$(sub --job-name=h_teacher_sam45_s2 $S $T --seed 2)
c=$(sub --job-name=h_teacher_sam50_s1 $S $T --seed 1 --val-size 0)
d=$(sub --job-name=h_teacher_sam50_s2 $S $T --seed 2 --val-size 0)
echo teachers $a $b $c $d
echo h_fm800_ens4 $(sub --job-name=h_fm800_ens4 --time=16:00:00 --dependency=afterok:$a:$b $S --model resnet_c --epochs 800 $C --teacher runs/teacher_wrn300/final.pt runs/h_teacher_sam45/final.pt runs/h_teacher_sam45_s1/final.pt runs/h_teacher_sam45_s2/final.pt)
echo h_final800_ens4 $(sub --job-name=h_final800_ens4 --time=16:00:00 --dependency=afterok:$c:$d $S --model resnet_c --epochs 800 $C --val-size 0 --teacher runs/teacher50/final.pt runs/h_teacher_sam50/final.pt runs/h_teacher_sam50_s1/final.pt runs/h_teacher_sam50_s2/final.pt)
scancel 21068
