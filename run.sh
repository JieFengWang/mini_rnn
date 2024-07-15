echo `date`

rm -rf build 
mkdir build 
cd build 
cmake ..
make -j 

cd ..


./build/rnn \
/home/jeff/data/sift1m/sift1m_base.fvecs \
out_index.ivecs