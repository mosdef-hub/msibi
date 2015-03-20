GPU_ID=$(get_gpu <$PBS_GPUFILE)

cd q0
rm query.dcd
hoomd query.hoomd.txt --gpu=${GPU_ID} > log1.txt &
wait

cd ../q2
rm query.dcd
hoomd query.hoomd.txt --gpu=${GPU_ID} > log1.txt &
wait

cd ../q3
rm query.dcd
hoomd query.hoomd.txt --gpu=${GPU_ID} > log1.txt &
wait
