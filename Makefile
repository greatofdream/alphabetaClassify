files:=0 7 8 9
idir:=./data/ghostPlay/
odir:=./output/ghostPlay/
.PHONY: all merge
all: $(files:%=ab-%.h5)
	echo "finish make"
ab-%.h5:
	python3 dataprepare.py -f $@ -i $(idir) -o $(odir)
merge: $(odir)train.h5
	echo 'finish merge'
$(odir)train.h5: $(files:%=$(odir)ab-%.h5)
	python3 merge.py $^ -o $@

train: $(odir)model.pth
	echo "finish train"
$(odir)model.pth: $(files:%=$(odir)ab-%.h5)
	python3 train.py $^ -o $@ > $@.log 2>&1

predict: $(odir)answer.h5
	echo "finish predict"
$(odir)answer.h5: $(odir)model.pth $(idir)problem.h5
	python3 predict.py -i $(word 2, $^) -o $@ -m $<
