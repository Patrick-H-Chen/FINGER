## Caveat

A caveat is that you have to make it on a CPU machine with AVX512f support. And the results reported are based on the benchmark on AWS instances mentioned in the paper, which indeed supported it.

## Dataset

Extract the prepared fashion mnist dataset

cat fashion.tar.gz.* | zcat > fashion.tar.gz

tar -xf fashion.tar.gz

## How to run?

We have tested that it could be built on AWS instances used in the paper. You only need to perform make under ann-only/v4/ directory (or v5). There are two versions (v4 and v5) which correspond to 64 and 128 rank projections. Following ann-benchmark, we basically treat this r as a hyper-parameter so we report the best of these two ranks in the paper. We also have r = 256 implementations but empricially it would be slower so I didn't put it here. I do find there will be some problems building the package on some machines due to CPU support. Basically the safest way is to use AWS. You would need openblas or other version to run the code. There is an instruction on the PECOS library general page https://github.com/amzn/pecos. I prepared standalone npy files for fashion-mnist so that you could use to test. In pecos ANN, we require database points to be X.trn.npy, testing points to be X.tst.npy and labels to be Yi.tst.npy. You could run the built binary by 

./go ../../fashion ../../fashion l2 500 96 24 10 0 0 1 0

500 is the M hyper-parameter, and 10 is efS in HSNW paper. 24 is the #threads to run, so you can keep it as 24 for index building (first time running) but to profile it needs to be changed to 1. The trailing 0 0 1 0 is legacy code which I haven't cleaned yet. Overall, we use the following script to run ann-benchmark

```
DATA=(fashion sift gist)
for data in ${DATA[@]}; do
    MLIST=(4 8 12 24 36 48 64 96)
    for M in ${MLIST[@]}; do
        echo ${data} " M = " ${M} | tee -a log.${data}
        EFSLIST=(10 20 30 40 50 60 70 80 90 100 200 400)
        for EFS in ${EFSLIST[@]}; do
            ./go ${data} ${data} l2 ${M} 500 96 ${EFS} 0 0 1 0 | tee -a log.${data}
        done
    done
done
```

```
DATA=(glove nytimes deep)
for data in ${DATA[@]}; do
    MLIST=(4 8 12 24 36 48 64 96)
    for M in ${MLIST[@]}; do
        echo ${data} " M = " ${M} | tee -a log.${data}
        EFSLIST=(10 20 30 40 50 60 70 80 90 100 200 400)
        for EFS in ${EFSLIST[@]}; do
            ./go ${data} ${data} ip ${M} 500 96 ${EFS} 0 0 1 0 | tee -a log.${data}
        done
    done
done
```
